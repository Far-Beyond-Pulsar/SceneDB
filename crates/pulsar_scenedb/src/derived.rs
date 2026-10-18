//! A derived-data cache on disk: the Disk tier of an asset.
//!
//! An asset's loader turns a source file (a JPEG, a model) into the form the
//! engine wants (mips, compressed blocks, parsed geometry). That conversion
//! can be expensive and its result depends only on the source and some
//! settings, so it is done once and kept: [`derive_cached`] returns the cached result
//! when the source and settings are unchanged, and runs the kind's `build`
//! function and stores what it returns when they are not.
//!
//! Together with [`crate::Residency`] this gives the asset layer its tiers.
//! The source file is the cold copy, the derived file is the warm one, the
//! residency cache is RAM, and a texture store is VRAM. A value dropped from
//! RAM is rebuilt by reading the derived file, which costs a read, where
//! rebuilding from the source costs a decode and an encode.
//!
//! The cache knows nothing about what it stores. The payload is opaque bytes;
//! the kind owns the format and bumps `version` when it changes.
//!
//! # Keys
//!
//! A key is a 128-bit FNV-1a hash of the kind, the source path, the source's
//! length and modification time, the settings string and the format version.
//! It does not read the source's contents: a file whose length and mtime are
//! unchanged is assumed unchanged. That is the usual build-system trade, and
//! it makes a cache hit cost one `stat` and one read.
//!
//! # Files
//!
//! `<dir>/<kind>/<key as 32 hex digits>.pdd`, a 32-byte header (magic, format
//! version, payload length, key) followed by the payload. Files are written to
//! a temporary name and renamed into place, so a reader sees a whole file or
//! none. A file whose header does not match is treated as a miss and replaced.
//! The payload is not checksummed; truncation is caught by the length check.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

const MAGIC: [u8; 4] = *b"PDD1";
const HEADER_LEN: usize = 32;
const EXTENSION: &str = "pdd";

/// Process-wide cache settings and counters, reached through the asset
/// registries so every statically linked copy of this crate shares them.
#[derive(Default)]
pub(crate) struct State {
    dir: Option<PathBuf>,
    hits: u64,
    misses: u64,
    bytes_read: u64,
    bytes_written: u64,
    errors: u64,
    /// Makes temporary file names unique within the process.
    temp: u64,
}

fn state() -> std::sync::MutexGuard<'static, State> {
    crate::assets::registries()
        .derived
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Where derived files go; `None` (the default) turns the cache off and makes
/// [`derive_cached`] always run `build`.
pub fn set_derived_cache_dir(dir: Option<PathBuf>) {
    state().dir = dir;
}

pub fn derived_cache_dir() -> Option<PathBuf> {
    state().dir.clone()
}

/// Counters since the process started, for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DerivedStats {
    /// Requests answered from a file.
    pub hits: u64,
    /// Requests that ran `build`.
    pub misses: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// Cache files that could not be read or written (the request still
    /// succeeded by building).
    pub errors: u64,
}

pub fn derived_stats() -> DerivedStats {
    let state = state();
    DerivedStats {
        hits: state.hits,
        misses: state.misses,
        bytes_read: state.bytes_read,
        bytes_written: state.bytes_written,
        errors: state.errors,
    }
}

/// What to derive: the kind, the source, and everything else the result
/// depends on.
#[derive(Clone, Copy, Debug)]
pub struct DerivedRequest<'a> {
    /// The asset kind; also the folder the file goes in.
    pub kind: &'a str,
    pub source: &'a Path,
    /// Every setting the result depends on, as a string (a size cap, a target
    /// format). A change makes a new key.
    pub settings: &'a str,
    /// The kind's format version: bump it when `build` changes what it returns.
    pub version: u32,
}

fn source_stamp(source: &Path) -> Result<(u64, u128), String> {
    let meta = std::fs::metadata(source)
        .map_err(|error| format!("could not stat '{}': {error}", source.display()))?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    Ok((meta.len(), modified))
}

/// The key of `request` for a source with the given length and mtime.
pub(crate) fn key_of(request: &DerivedRequest, length: u64, modified: u128) -> u128 {
    let mut hash: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    let prime: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut feed = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u128::from(byte);
            hash = hash.wrapping_mul(prime);
        }
    };
    feed(request.kind.as_bytes());
    feed(&[0]);
    feed(request.source.to_string_lossy().as_bytes());
    feed(&[0]);
    feed(&length.to_le_bytes());
    feed(&modified.to_le_bytes());
    feed(request.settings.as_bytes());
    feed(&[0]);
    feed(&request.version.to_le_bytes());
    hash
}

fn file_of(dir: &Path, kind: &str, key: u128) -> PathBuf {
    dir.join(kind).join(format!("{key:032x}.{EXTENSION}"))
}

fn read_cached(path: &Path, key: u128, version: u32) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < HEADER_LEN || bytes[..4] != MAGIC {
        return None;
    }
    let stored_version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let length = u64::from_le_bytes(bytes[8..16].try_into().ok()?) as usize;
    let stored_key = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
    if stored_version != version
        || stored_key != key as u64
        || bytes.len() != HEADER_LEN + length
    {
        return None;
    }
    Some(bytes[HEADER_LEN..].to_vec())
}

fn write_cached(
    path: &Path,
    key: u128,
    version: u32,
    payload: &[u8],
    unique: u64,
) -> std::io::Result<()> {
    let dir = path.parent().expect("cache files have a folder");
    std::fs::create_dir_all(dir)?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(key as u64).to_le_bytes());
    bytes.extend_from_slice(&[0; 8]);
    bytes.extend_from_slice(payload);
    let temp = dir.join(format!(
        ".{:032x}.{}.{}.tmp",
        key,
        std::process::id(),
        unique
    ));
    std::fs::write(&temp, &bytes)?;
    std::fs::rename(&temp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

/// The derived form of `request.source`: read from the cache when the source
/// and settings are unchanged, otherwise built by `build` and stored. With no
/// cache directory set this is just `build()`. A cache file that cannot be
/// read or written never fails the request; it is counted in
/// [`DerivedStats::errors`] and `build` runs.
pub fn derive_cached(
    request: &DerivedRequest,
    build: impl FnOnce() -> Result<Vec<u8>, String>,
) -> Result<Vec<u8>, String> {
    let Some(dir) = derived_cache_dir() else {
        return build();
    };
    let (length, modified) = source_stamp(request.source)?;
    let key = key_of(request, length, modified);
    let path = file_of(&dir, request.kind, key);

    if let Some(payload) = read_cached(&path, key, request.version) {
        // Recently used files are the last to be pruned.
        if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&path) {
            let _ = file.set_modified(SystemTime::now());
        }
        let mut state = state();
        state.hits += 1;
        state.bytes_read += payload.len() as u64;
        return Ok(payload);
    }

    let payload = build()?;
    let unique = {
        let mut state = state();
        state.temp += 1;
        state.temp
    };
    let written = write_cached(&path, key, request.version, &payload, unique);
    let mut state = state();
    state.misses += 1;
    match written {
        Ok(()) => state.bytes_written += payload.len() as u64,
        Err(_) => state.errors += 1,
    }
    Ok(payload)
}

/// What [`prune_derived_cache`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PruneReport {
    pub removed_files: usize,
    pub removed_bytes: u64,
    pub kept_bytes: u64,
}

/// Delete the least recently used derived files until the cache is at most
/// `max_bytes`. Recency is the file's modification time, which [`derive_cached`]
/// refreshes on every hit.
pub fn prune_derived_cache(max_bytes: u64) -> std::io::Result<PruneReport> {
    let Some(dir) = derived_cache_dir() else {
        return Ok(PruneReport::default());
    };
    let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    for kind in std::fs::read_dir(&dir)?.flatten() {
        if !kind.path().is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(kind.path())?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == EXTENSION) {
                if let Ok(meta) = entry.metadata() {
                    files.push((meta.modified().unwrap_or(SystemTime::UNIX_EPOCH), meta.len(), path));
                }
            }
        }
    }
    files.sort_by_key(|(modified, _, _)| *modified);
    let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
    let mut report = PruneReport::default();
    for (_, len, path) in &files {
        if total <= max_bytes {
            break;
        }
        if std::fs::remove_file(path).is_ok() {
            total -= len;
            report.removed_files += 1;
            report.removed_bytes += len;
        }
    }
    report.kept_bytes = total;
    Ok(report)
}

/// Delete the cached result for `request`, if there is one. For a caller that
/// finds a payload it cannot use (a format the cache's own checks cannot
/// judge): the next [`derive_cached`] builds a fresh one.
pub fn forget_derived(request: &DerivedRequest) {
    let Some(dir) = derived_cache_dir() else {
        return;
    };
    if let Ok((length, modified)) = source_stamp(request.source) {
        let key = key_of(request, length, modified);
        let _ = std::fs::remove_file(file_of(&dir, request.kind, key));
    }
}
