//! The derived-data cache: a result is built once, read back while its source
//! and settings are unchanged, rebuilt when they change, and pruned oldest
//! first.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pulsar_scenedb::{
    derive_cached, derived_stats, prune_derived_cache, set_derived_cache_dir, DerivedRequest,
};

/// The cache directory and counters are process-wide; tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

/// A scratch directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "scenedb-derived-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        set_derived_cache_dir(None);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn source(dir: &Path, bytes: &[u8]) -> PathBuf {
    let path = dir.join("source.bin");
    std::fs::write(&path, bytes).unwrap();
    path
}

fn request<'a>(source: &'a Path, settings: &'a str, version: u32) -> DerivedRequest<'a> {
    DerivedRequest {
        kind: "test",
        source,
        settings,
        version,
    }
}

/// Derive `source` with a build that counts its calls.
fn derive_counting(
    request: &DerivedRequest,
    builds: &Arc<AtomicUsize>,
    payload: &[u8],
) -> Result<Vec<u8>, String> {
    let builds = builds.clone();
    let payload = payload.to_vec();
    derive_cached(request, move || {
        builds.fetch_add(1, Ordering::SeqCst);
        Ok(payload)
    })
}

#[test]
fn without_a_cache_directory_every_request_builds() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("off");
    set_derived_cache_dir(None);
    let src = source(&scratch.0, b"abc");
    let builds = Arc::new(AtomicUsize::new(0));
    derive_counting(&request(&src, "", 1), &builds, b"x").unwrap();
    derive_counting(&request(&src, "", 1), &builds, b"x").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 2);
}

#[test]
fn a_result_is_built_once_and_then_read_back() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("hit");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"the source image");
    let builds = Arc::new(AtomicUsize::new(0));
    let before = derived_stats();

    let first = derive_counting(&request(&src, "cap=2048", 1), &builds, b"derived payload").unwrap();
    let second = derive_counting(&request(&src, "cap=2048", 1), &builds, b"WRONG").unwrap();

    assert_eq!(first, b"derived payload");
    assert_eq!(second, b"derived payload", "the second call reads the first call's result");
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    let after = derived_stats();
    assert_eq!((after.misses - before.misses, after.hits - before.hits), (1, 1));
    assert_eq!(after.bytes_written - before.bytes_written, 15);
    assert_eq!(after.bytes_read - before.bytes_read, 15);
}

#[test]
fn a_changed_source_setting_or_version_makes_a_new_result() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("invalidate");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"version one of the file");
    let builds = Arc::new(AtomicUsize::new(0));

    derive_counting(&request(&src, "a", 1), &builds, b"1").unwrap();
    derive_counting(&request(&src, "a", 1), &builds, b"1").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 1);

    derive_counting(&request(&src, "b", 1), &builds, b"2").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 2, "other settings, other result");
    derive_counting(&request(&src, "a", 2), &builds, b"3").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 3, "a new format version rebuilds");

    // Edit the source: its length changes, so its key does.
    std::fs::write(&src, b"version two of the file, now longer").unwrap();
    let after_edit = derive_counting(&request(&src, "a", 1), &builds, b"4").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 4);
    assert_eq!(after_edit, b"4");
}

#[test]
fn a_damaged_cache_file_is_replaced_not_trusted() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("damaged");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"source");
    let builds = Arc::new(AtomicUsize::new(0));
    derive_counting(&request(&src, "", 1), &builds, b"good payload").unwrap();

    let files: Vec<PathBuf> = std::fs::read_dir(scratch.0.join("cache/test"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(files.len(), 1);
    // Truncate the file, as a crash mid-copy would.
    let bytes = std::fs::read(&files[0]).unwrap();
    std::fs::write(&files[0], &bytes[..bytes.len() - 3]).unwrap();

    let rebuilt = derive_counting(&request(&src, "", 1), &builds, b"good payload").unwrap();
    assert_eq!(rebuilt, b"good payload");
    assert_eq!(builds.load(Ordering::SeqCst), 2, "the damaged file was a miss");
    derive_counting(&request(&src, "", 1), &builds, b"unused").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 2, "and it was replaced by a whole one");
}

#[test]
fn a_failed_build_is_reported_and_stores_nothing() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("failed");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"source");
    let error = derive_cached(&request(&src, "", 1), || Err("cook failed".to_string()));
    assert_eq!(error.err().as_deref(), Some("cook failed"));
    assert!(!scratch.0.join("cache/test").exists() || std::fs::read_dir(scratch.0.join("cache/test")).unwrap().next().is_none());

    let missing = scratch.0.join("nope.bin");
    let error = derive_cached(&request(&missing, "", 1), || Ok(vec![]));
    assert!(error.unwrap_err().contains("could not stat"));
}

#[test]
fn concurrent_requests_for_one_key_all_succeed_with_one_valid_file() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("concurrent");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"shared source");
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let (src, barrier) = (src.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                derive_cached(&request(&src, "", 1), || Ok(b"same bytes".to_vec())).unwrap()
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), b"same bytes");
    }
    let files: Vec<_> = std::fs::read_dir(scratch.0.join("cache/test"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files.len(), 1, "no temporary files left behind: {files:?}");
    assert!(files[0].ends_with(".pdd"));
}

#[test]
fn pruning_removes_the_least_recently_used_files_first() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("prune");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let builds = Arc::new(AtomicUsize::new(0));
    let sources: Vec<PathBuf> = (0..4)
        .map(|i| {
            let path = scratch.0.join(format!("s{i}.bin"));
            std::fs::write(&path, format!("source {i}")).unwrap();
            path
        })
        .collect();
    for (i, src) in sources.iter().enumerate() {
        derive_counting(&request(src, "", 1), &builds, &vec![i as u8; 1000]).unwrap();
        // Distinct modification times, oldest first.
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    // A hit refreshes a file: source 0 becomes the most recently used.
    derive_counting(&request(&sources[0], "", 1), &builds, b"unused").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 4);

    let each = 1000 + 32;
    let report = prune_derived_cache(2 * each).unwrap();
    assert_eq!(report.removed_files, 2);
    assert_eq!(report.kept_bytes, 2 * each);

    // Sources 1 and 2 were the oldest; 0 (refreshed) and 3 survive.
    derive_counting(&request(&sources[0], "", 1), &builds, b"unused").unwrap();
    derive_counting(&request(&sources[3], "", 1), &builds, b"unused").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 4, "the survivors are still hits");
    derive_counting(&request(&sources[1], "", 1), &builds, &vec![1; 1000]).unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 5, "a pruned file is rebuilt on demand");
}

#[test]
fn a_forgotten_result_is_built_again() {
    let _turn = SERIAL.lock().unwrap();
    let scratch = Scratch::new("forget");
    set_derived_cache_dir(Some(scratch.0.join("cache")));
    let src = source(&scratch.0, b"source");
    let builds = Arc::new(AtomicUsize::new(0));
    derive_counting(&request(&src, "", 1), &builds, b"payload").unwrap();
    derive_counting(&request(&src, "", 1), &builds, b"payload").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 1);

    pulsar_scenedb::forget_derived(&request(&src, "", 1));
    derive_counting(&request(&src, "", 1), &builds, b"payload").unwrap();
    assert_eq!(builds.load(Ordering::SeqCst), 2, "forgetting made the next request a miss");
    // Forgetting something that is not there is fine.
    pulsar_scenedb::forget_derived(&request(&scratch.0.join("absent.bin"), "", 1));
}
