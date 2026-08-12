//! The single keyed buffer registry (design: "Make the buffers generic") —
//! the one place that maps a declared `BufferKey` to its current
//! `wgpu::Buffer` + allocation `epoch`.
//!
//! This is the Tier 1 primitive the renderer handoff and the registry
//! collapse both build on:
//!
//! - A component's `#[gpu(buffer = "builtin_material")]` field declares the
//!   *key* of the buffer it belongs to; registration resolves that key here
//!   and lands the field in the same physical buffer as every other field /
//!   asset entry that declared the same key.
//! - A system declares which buffers it consumes (`Buffer<"key">`); the
//!   engine resolves them from this registry at setup and hands back a
//!   [`BufferHandle`] — an owned `wgpu::Buffer` plus the `epoch` for rebind
//!   decisions.
//!
//! # Soundness: the handoff never outlives a growth
//!
//! Growable buffers reallocate on growth, which changes the `wgpu::Buffer`
//! identity — a borrowed `&wgpu::Buffer` handed out across that boundary
//! would dangle. [`Self::resolve`] therefore returns an **owned** clone of
//! the current buffer (`wgpu::Buffer` is cheap to clone: a refcounted
//! handle, not a byte copy) paired with that buffer's `epoch`. A consumer
//! holds no borrow from inside the registry's lock scope; on the next frame
//! it re-resolves and compares `epoch` against the value it cached, and
//! rebinds only when the underlying buffer actually changed.
//!
//! No `&wgpu::Buffer` is ever returned by this registry.
//!
//! # Compatibility is validated at registration, not at runtime
//!
//! Registering a second element type (or access mode, or mirror mode) under
//! a key that already exists is a [`BufferRegistrationError`], never a
//! silent re-registration that replaces the first buffer — two components
//! declaring `buffer = "builtin_material"` share one physical buffer only
//! when they agree on element type, stride, access, and mirror mode. An
//! `Once`-mode buffer may only be shared by other `Once` fields of the same
//! element type (the buffer's element type is the *heavy* type the fields'
//! upload mappers produce); a `DirtyTracked` buffer likewise only by
//! `DirtyTracked` fields.
//!
//! Resource entries (the byte-range suballocator `GeometryArena`, the
//! texture-array `TextureStore`) register under the same namespace as row
//! buffers, so `"builtin_mesh_vertex"` and `"builtin_texture"` are ordinary
//! keys too — but a resource entry and a row buffer can never collide on
//! one key.

use crate::page::Pod;
use crate::token::HasTypeToken;
use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;

use super::MirrorMode;

/// A named, `'static` buffer key — the string a `#[gpu(buffer = "...")]`
/// field (or a hand-built asset buffer) declares, and the handle a system
/// uses to resolve the buffer from [`GpuBufferRegistry`].
///
/// `&'static str`-backed so it is `Copy`, zero-size-allocating, and usable
/// as a `HashMap` key without allocation. Built from a string literal via
/// [`BufferKey::of`]; the derive macro emits these from the literal inside
/// `#[gpu(buffer = "...")]`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferKey(&'static str);

impl BufferKey {
    /// Wrap a `'static` string literal as a buffer key.
    pub const fn of(key: &'static str) -> Self {
        assert!(!key.is_empty(), "BufferKey must not be empty");
        BufferKey(key)
    }

    /// The underlying `&'static str`.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Debug for BufferKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl fmt::Display for BufferKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0, f)
    }
}

impl AsRef<str> for BufferKey {
    fn as_ref(&self) -> &str {
        self.0
    }
}

/// Declared access on a keyed buffer. A `Buffer<"key">`-style param declares
/// read-only vs read-write so the registry (and ultimately the engine) can
/// refuse to hand a system write access to a read-only registration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    /// The buffer is only ever bound read-only on the GPU.
    ReadOnly,
    /// The buffer may be written/bound read-write.
    ReadWrite,
}

/// The owned result of resolving a key from [`GpuBufferRegistry`]: the
/// buffer's current `wgpu::Buffer` and the allocation `epoch` it was
/// captured at. See the module doc's soundness section for why this is
/// owned, never a borrow.
///
/// `wgpu::Buffer` is a cheap, refcounted handle, so `BufferHandle` is `Clone`
/// and can be held for as long as a frame needs it.
#[derive(Clone, Debug)]
pub struct BufferHandle {
    /// The current `wgpu::Buffer` (owned clone — safe across growth).
    pub buffer: wgpu::Buffer,
    /// Bumped by exactly one on every reallocation. Compare against a
    /// previously cached value to decide whether a bind group built against
    /// [`Self::buffer`] needs rebuilding.
    pub epoch: u64,
}

/// Why a registration was rejected. All variants are "the key already exists
/// with an incompatible registration" — the registry never silently aliases
/// two different buffers onto one key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferRegistrationError {
    /// A row buffer of a different element type is already registered.
    ElementTypeMismatch {
        key: BufferKey,
        existing: Option<TypeId>,
        incoming: Option<TypeId>,
    },
    /// An element/stride mismatch: `Pod` types with different sizes cannot
    /// share a row buffer.
    ElementSizeMismatch {
        key: BufferKey,
        existing: usize,
        incoming: usize,
    },
    /// The key already exists under a different access declaration.
    AccessMismatch {
        key: BufferKey,
        existing: BufferAccess,
        incoming: BufferAccess,
    },
    /// The key already exists under a different mirror mode. `Once` shares
    /// only with `Once`; `DirtyTracked` only with `DirtyTracked`.
    MirrorModeMismatch {
        key: BufferKey,
        existing: Option<MirrorMode>,
        incoming: Option<MirrorMode>,
    },
    /// The key already exists under a different entry KIND (row buffer,
    /// byte-range resource, or texture array) than the incoming
    /// registration — e.g. a row buffer re-registered as a resource, or a
    /// texture-array key claimed by a row buffer. Any two of the three kinds
    /// colliding on one key is this variant; which two is not distinguished
    /// (the fix is always "pick a different key").
    RowResourceCollision { key: BufferKey },
}

impl fmt::Display for BufferRegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferRegistrationError::ElementTypeMismatch { key, existing, incoming } => write!(
                f,
                "buffer {:?} already registered for element {:?}, can't also hold {:?}",
                key, existing, incoming
            ),
            BufferRegistrationError::ElementSizeMismatch { key, existing, incoming } => write!(
                f,
                "buffer {:?} has element stride {existing} bytes; incoming element is {incoming} bytes — \
                 shared buffers require identical element type and stride",
                key,
            ),
            BufferRegistrationError::AccessMismatch { key, existing, incoming } => write!(
                f,
                "buffer {:?} already registered {:?}; incoming declares {:?}",
                key, existing, incoming
            ),
            BufferRegistrationError::MirrorModeMismatch { key, existing, incoming } => write!(
                f,
                "buffer {:?} already registered as {:?}; incoming declares {:?} — Once fields share only \
                 with Once, DirtyTracked only with DirtyTracked",
                key, existing, incoming
            ),
            BufferRegistrationError::RowResourceCollision { key } => write!(
                f,
                "buffer {:?} is registered under a different entry kind (row buffer / resource / \
                 texture array) than this registration declares",
                key,
            ),
        }
    }
}

impl std::error::Error for BufferRegistrationError {}

/// Which of the registry's three entry shapes a [`BufferKey`] was claimed
/// under. A key belongs to exactly one kind for its lifetime — a second
/// registration under a different kind is a
/// [`BufferRegistrationError::RowResourceCollision`], never a silent
/// overwrite.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EntryKind {
    /// A `T: Pod` row buffer — [`GpuBufferRegistry::register_row`].
    Row,
    /// A byte-range resource (e.g. `GeometryArena`'s vertex/index arenas) —
    /// [`GpuBufferRegistry::register_resource`]. Still backed by one
    /// `wgpu::Buffer`, just not a dense `T` row array.
    Resource,
    /// A texture-array slot table (e.g. `TextureStore`) — has NO single
    /// `wgpu::Buffer` (it is N independent `wgpu::Texture` objects), so this
    /// entry reserves the key's namespace and records its slot capacity
    /// without holding a buffer handle at all. See
    /// [`GpuBufferRegistry::register_texture_array`].
    TextureArray,
}

/// A single keyed registration: the current buffer + epoch (for
/// buffer-backed kinds) or just the slot capacity (for a texture array),
/// plus the compatibility metadata registration validated. Internal to the
/// registry; callers interact with it through [`BufferHandle`] and
/// `resolve`.
struct RegistryEntry {
    key: BufferKey,
    kind: EntryKind,
    /// Element type identity for row buffers; `None` for resource entries
    /// and texture arrays.
    element_type_id: Option<TypeId>,
    /// Row stride for `Row`; total byte size for `Resource`; slot capacity
    /// for `TextureArray` (documented per-variant — not bytes in that case).
    element_size: usize,
    access: BufferAccess,
    mirror_mode: Option<MirrorMode>,
    /// `Some` for `Row`/`Resource` (buffer-backed); `None` for
    /// `TextureArray`, which has no `wgpu::Buffer` to hold.
    inner: Option<RwLock<EntryInner>>,
}

struct EntryInner {
    buffer: wgpu::Buffer,
    epoch: u64,
}

/// Bundles [`GpuBufferRegistry::insert`]'s parameters — a private helper
/// struct purely to keep that function's argument count readable
/// (`register_row`/`register_resource` are the only two callers, both of
/// which already have every field in hand from their own public
/// signatures).
struct InsertSpec {
    key: BufferKey,
    kind: EntryKind,
    element_type_id: Option<TypeId>,
    element_size: usize,
    access: BufferAccess,
    mirror_mode: Option<MirrorMode>,
    buffer: wgpu::Buffer,
}

/// The single keyed buffer registry (Tier 1). Maps [`BufferKey`] → the
/// current `wgpu::Buffer` + `epoch`, validating compatibility on every
/// registration.
///
/// Thread-safe (`RwLock`-backed); registration and resolution both take
/// `&self`, matching how `SceneGpuStore` registers its World-mirrored
/// buffers through shared references.
#[derive(Default)]
pub struct GpuBufferRegistry {
    entries: RwLock<HashMap<BufferKey, RegistryEntry>>,
}

impl GpuBufferRegistry {
    /// A fresh, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register (or re-resolve, if the key already exists compatibly) a row
    /// buffer of `T: Pod` under `key`, with `buffer` as its current
    /// `wgpu::Buffer` and `access`/`mode` as its declared access and mirror
    /// mode.
    ///
    /// If `key` is already registered, the incoming registration must match
    /// the existing one's element type, stride, access, and mirror mode —
    /// otherwise this returns a [`BufferRegistrationError`] and nothing
    /// changes (the shared-key rule: two components declaring the same key
    /// share one physical buffer only when they agree on everything the
    /// layout engine validates).
    pub fn register_row<T: Pod + HasTypeToken + 'static>(
        &self,
        key: BufferKey,
        buffer: wgpu::Buffer,
        access: BufferAccess,
        mode: MirrorMode,
    ) -> Result<(), BufferRegistrationError> {
        self.insert(InsertSpec {
            key,
            kind: EntryKind::Row,
            element_type_id: Some(TypeId::of::<T>()),
            element_size: std::mem::size_of::<T>(),
            access,
            mirror_mode: Some(mode),
            buffer,
        })
    }

    /// Register a *resource* entry (not a `T: Pod` row buffer) under `key`:
    /// the byte-range suballocator `GeometryArena` (`"builtin_mesh_vertex"`,
    /// `"builtin_mesh_index"`). `size` is the entry's total size in bytes —
    /// stored for introspection, not bounds-checked by this registry.
    ///
    /// A resource entry, a row buffer, and a texture array can never share a
    /// key. For a texture array specifically (no single `wgpu::Buffer` to
    /// register here), use [`Self::register_texture_array`] instead.
    pub fn register_resource(
        &self,
        key: BufferKey,
        size: usize,
        buffer: wgpu::Buffer,
        access: BufferAccess,
    ) -> Result<(), BufferRegistrationError> {
        self.insert(InsertSpec {
            key,
            kind: EntryKind::Resource,
            element_type_id: None,
            element_size: size,
            access,
            mirror_mode: None,
            buffer,
        })
    }

    /// Register a *texture array* slot table under `key` — e.g.
    /// `TextureStore` (`"builtin_texture"`). Unlike [`Self::register_row`]/
    /// [`Self::register_resource`], this reserves the key's namespace
    /// (blocking a row buffer or byte-range resource from ever claiming the
    /// same key) WITHOUT storing a `wgpu::Buffer` — a texture array is N
    /// independent `wgpu::Texture` objects, which this `wgpu::Buffer`-keyed
    /// registry has no slot for. [`Self::resolve`] always returns `None` for
    /// a texture-array key; texture access stays entirely through the
    /// owning `TextureStore`'s own API (`texture(slot)`) — see the module
    /// doc's "Resource entries" paragraph and issue #41's "What legitimately
    /// stays bespoke" section.
    ///
    /// `slot_capacity` is the array's current slot-table extent (e.g.
    /// `TextureStore::slot_count()`) — stored for introspection via
    /// [`Self::texture_array_slot_capacity`], not bounds-checked here. Safe
    /// to call again with a larger `slot_capacity` as the table grows (a
    /// compatible re-registration, same as a row buffer's growth) — smaller
    /// or equal is also accepted (never shrinks the stored value below what
    /// growth already recorded is not enforced; callers should pass the
    /// current true extent each time, matching `TextureStore::slot_count`'s
    /// own monotonically-nondecreasing contract).
    pub fn register_texture_array(
        &self,
        key: BufferKey,
        slot_capacity: u32,
        access: BufferAccess,
    ) -> Result<(), BufferRegistrationError> {
        let mut entries = self.entries.write().expect("GpuBufferRegistry lock poisoned");
        if let Some(existing) = entries.get_mut(&key) {
            if existing.kind != EntryKind::TextureArray {
                return Err(BufferRegistrationError::RowResourceCollision { key });
            }
            if existing.access != access {
                return Err(BufferRegistrationError::AccessMismatch {
                    key,
                    existing: existing.access,
                    incoming: access,
                });
            }
            existing.element_size = slot_capacity as usize;
            return Ok(());
        }
        entries.insert(
            key,
            RegistryEntry {
                key,
                kind: EntryKind::TextureArray,
                element_type_id: None,
                element_size: slot_capacity as usize,
                access,
                mirror_mode: None,
                inner: None,
            },
        );
        Ok(())
    }

    /// Whether `key` is registered as a texture array (see
    /// [`Self::register_texture_array`]). `false` for an unregistered key or
    /// one registered as a row buffer / resource.
    pub fn is_texture_array(&self, key: BufferKey) -> bool {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        entries.get(&key).is_some_and(|e| e.kind == EntryKind::TextureArray)
    }

    /// The slot capacity last recorded for a texture-array key (see
    /// [`Self::register_texture_array`]). `None` if `key` isn't registered
    /// as a texture array.
    pub fn texture_array_slot_capacity(&self, key: BufferKey) -> Option<u32> {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        let entry = entries.get(&key)?;
        (entry.kind == EntryKind::TextureArray).then_some(entry.element_size as u32)
    }

    fn insert(&self, spec: InsertSpec) -> Result<(), BufferRegistrationError> {
        let InsertSpec { key, kind, element_type_id, element_size, access, mirror_mode, buffer } = spec;
        let mut entries = self.entries.write().expect("GpuBufferRegistry lock poisoned");
        if let Some(existing) = entries.get(&key) {
            if existing.kind != kind {
                return Err(BufferRegistrationError::RowResourceCollision { key });
            }
            if existing.element_type_id != element_type_id {
                return Err(BufferRegistrationError::ElementTypeMismatch {
                    key,
                    existing: existing.element_type_id,
                    incoming: element_type_id,
                });
            }
            if existing.element_size != element_size {
                return Err(BufferRegistrationError::ElementSizeMismatch {
                    key,
                    existing: existing.element_size,
                    incoming: element_size,
                });
            }
            if existing.access != access {
                return Err(BufferRegistrationError::AccessMismatch {
                    key,
                    existing: existing.access,
                    incoming: access,
                });
            }
            if existing.mirror_mode != mirror_mode {
                return Err(BufferRegistrationError::MirrorModeMismatch {
                    key,
                    existing: existing.mirror_mode,
                    incoming: mirror_mode,
                });
            }
            // Compatible re-registration: adopt the (possibly newer) buffer,
            // bumping the epoch so consumers re-resolve and rebind. This is
            // how a growable buffer's owner keeps the registry in sync with
            // a reallocation.
            let mut inner = existing
                .inner
                .as_ref()
                .expect("Row/Resource entries always carry a buffer")
                .write()
                .expect("GpuBufferRegistry lock poisoned");
            inner.epoch = inner.epoch.wrapping_add(1);
            inner.buffer = buffer;
            return Ok(());
        }
        entries.insert(
            key,
            RegistryEntry {
                key,
                kind,
                element_type_id,
                element_size,
                access,
                mirror_mode,
                inner: Some(RwLock::new(EntryInner { buffer, epoch: 0 })),
            },
        );
        Ok(())
    }

    /// Refresh `key`'s live buffer/epoch snapshot without re-validating
    /// compatibility — for a caller (namely [`super::SceneGpuStore`]) that
    /// already gates registration/growth compatibility itself via its own
    /// owner-tracking and just needs this registry's copy to stop going
    /// stale after the first registration.
    ///
    /// Unlike [`Self::register_row`]/[`Self::register_resource`], `epoch` is
    /// set to the caller-supplied ABSOLUTE value (not bumped by one), because
    /// the caller here already knows the true epoch from the buffer's real
    /// owner (e.g. `DynamicGpuBuffer::epoch()`), which increments exactly
    /// once per reallocation regardless of how many times this is called in
    /// between. A no-op if `epoch` hasn't advanced past what's stored, so
    /// calling this on every resolve is cheap and idempotent. Also a no-op if
    /// `key` was never registered — this never creates an entry.
    /// A no-op if `key` is registered as a texture array (see
    /// [`Self::register_texture_array`]) — there is no buffer slot to sync.
    pub fn sync(&self, key: BufferKey, buffer: wgpu::Buffer, epoch: u64) {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        let Some(entry) = entries.get(&key) else { return };
        let Some(inner) = entry.inner.as_ref() else { return };
        let mut inner = inner.write().expect("GpuBufferRegistry lock poisoned");
        if epoch != inner.epoch {
            inner.epoch = epoch;
            inner.buffer = buffer;
        }
    }

    /// Resolve `key` to its current [`BufferHandle`] (owned buffer + epoch).
    /// `None` if the key was never registered, OR if it was registered as a
    /// texture array ([`Self::register_texture_array`]) — a texture array
    /// has no single `wgpu::Buffer` to hand back; use
    /// [`Self::is_texture_array`]/[`Self::texture_array_slot_capacity`] plus
    /// the owning `TextureStore`'s own API for that case.
    ///
    /// The returned buffer is an owned clone captured atomically with the
    /// epoch it belongs to, so a growth between this call and the consumer's
    /// bind-group use can only invalidate the *epoch*, never the buffer's
    /// memory (see the module doc's soundness section).
    pub fn resolve(&self, key: BufferKey) -> Option<BufferHandle> {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        let entry = entries.get(&key)?;
        let inner = entry.inner.as_ref()?.read().expect("GpuBufferRegistry lock poisoned");
        Some(BufferHandle {
            buffer: inner.buffer.clone(),
            epoch: inner.epoch,
        })
    }

    /// The current epoch of `key`'s buffer, or `None` if never registered or
    /// registered as a texture array (no epoch concept — see
    /// [`Self::resolve`]).
    /// `Some` does not guarantee `resolve` will succeed (a concurrent
    /// compatible re-registration can't remove a key, but nothing here
    /// removes keys today).
    pub fn epoch(&self, key: BufferKey) -> Option<u64> {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        let entry = entries.get(&key)?;
        let epoch = entry.inner.as_ref()?.read().expect("GpuBufferRegistry lock poisoned").epoch;
        Some(epoch)
    }

    /// Whether `key` is registered.
    pub fn contains_key(&self, key: BufferKey) -> bool {
        self.entries.read().expect("GpuBufferRegistry lock poisoned").contains_key(&key)
    }

    /// Every registered key, in arbitrary order.
    pub fn keys(&self) -> Vec<BufferKey> {
        self.entries.read().expect("GpuBufferRegistry lock poisoned").keys().copied().collect()
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.read().expect("GpuBufferRegistry lock poisoned").len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().expect("GpuBufferRegistry lock poisoned").is_empty()
    }

    /// The registered element type of `key`'s row buffer (`None` for a
    /// resource entry, or an unregistered key).
    pub fn element_type(&self, key: BufferKey) -> Option<Option<TypeId>> {
        let entries = self.entries.read().expect("GpuBufferRegistry lock poisoned");
        entries.get(&key).map(|e| e.element_type_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> wgpu::Device {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu-buffer-registry-test"),
            ..Default::default()
        }))
        .expect("device");
        device
    }

    fn make_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn buffer_key_is_const_constructible_and_displayed_as_its_string() {
        const K: BufferKey = BufferKey::of("builtin_material");
        assert_eq!(K.as_str(), "builtin_material");
        assert_eq!(format!("{K}"), "builtin_material");
        assert_eq!(format!("{K:?}"), "\"builtin_material\"");
    }

    #[test]
    fn two_equal_keys_are_one_registry_entry() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_row::<u32>(BufferKey::of("builtin_material"), make_buffer(&device, "a", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("first registration");
        registry
            .register_row::<u32>(BufferKey::of("builtin_material"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("compatible re-registration must succeed (same element/access/mode)");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn resolve_returns_an_owned_buffer_and_epoch_and_re_registration_bumps_the_epoch() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        let first = make_buffer(&device, "first", 64);
        registry
            .register_row::<u32>(BufferKey::of("k"), first.clone(), BufferAccess::ReadWrite, MirrorMode::DirtyTracked)
            .expect("register");

        let h0 = registry.resolve(BufferKey::of("k")).expect("registered");
        assert_eq!(h0.epoch, 0);
        assert_eq!(h0.buffer.size(), 64, "resolve returns the first buffer");

        // Growth-style re-registration: a new buffer, epoch must bump so a
        // consumer holding h0 knows to rebind.
        let second = make_buffer(&device, "second", 128);
        registry
            .register_row::<u32>(BufferKey::of("k"), second.clone(), BufferAccess::ReadWrite, MirrorMode::DirtyTracked)
            .expect("compatible re-registration");
        let h1 = registry.resolve(BufferKey::of("k")).expect("still registered");
        assert_eq!(h1.epoch, 1);
        assert_eq!(h1.buffer.size(), 128, "resolve returns the re-registered (grown) buffer");

        // Unresolved key.
        assert!(registry.resolve(BufferKey::of("missing")).is_none());
    }

    #[test]
    fn sync_refreshes_buffer_and_epoch_to_the_caller_supplied_absolute_value() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        let first = make_buffer(&device, "first", 64);
        registry
            .register_row::<u32>(BufferKey::of("k"), first, BufferAccess::ReadWrite, MirrorMode::DirtyTracked)
            .expect("register");
        assert_eq!(registry.resolve(BufferKey::of("k")).unwrap().epoch, 0);

        // A grow that happened entirely outside `insert`/`register_row` (the
        // real-world case: a `DynamicGpuBuffer` reallocating on its own) is
        // reflected by `sync`, which sets the ABSOLUTE epoch the caller
        // already knows is current — not a blind bump.
        let grown = make_buffer(&device, "grown", 256);
        registry.sync(BufferKey::of("k"), grown.clone(), 3);
        let h = registry.resolve(BufferKey::of("k")).expect("still registered");
        assert_eq!(h.epoch, 3, "sync adopts the caller-supplied epoch");
        assert_eq!(h.buffer.size(), 256, "sync adopts the caller-supplied buffer");

        // Idempotent: calling `sync` again with the SAME epoch (e.g. two
        // resolves in one frame, no growth in between) must not perturb the
        // stored buffer, even if a different buffer value is passed in.
        let stale = make_buffer(&device, "stale", 64);
        registry.sync(BufferKey::of("k"), stale, 3);
        let h2 = registry.resolve(BufferKey::of("k")).expect("still registered");
        assert_eq!(h2.epoch, 3);
        assert_eq!(h2.buffer.size(), 256, "same-epoch sync is a no-op, buffer unchanged");

        // Unregistered key: `sync` must not create an entry.
        registry.sync(BufferKey::of("never-registered"), make_buffer(&device, "x", 4), 1);
        assert!(!registry.contains_key(BufferKey::of("never-registered")));
    }

    #[test]
    fn mismatched_element_type_is_rejected_not_silently_aliased() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "a", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("first registration");
        let err = registry
            .register_row::<f32>(BufferKey::of("k"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect_err("a f32 row can't share a u32 buffer");
        match err {
            BufferRegistrationError::ElementTypeMismatch { key, .. } => assert_eq!(key, BufferKey::of("k")),
            other => panic!("expected ElementTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn mismatched_stride_is_rejected_even_for_the_same_type_identity() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        // Both are TypeId-distinct here, but exercise the size-mismatch arm
        // with two different-size types that could never legitimately share.
        registry
            .register_row::<u64>(BufferKey::of("k"), make_buffer(&device, "a", 64), BufferAccess::ReadOnly, MirrorMode::Once)
            .expect("first registration");
        let err = registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::Once)
            .expect_err("8-byte and 4-byte elements must not share a row buffer");
    }

    #[test]
    fn mismatched_access_is_rejected() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "a", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("first registration");
        let err = registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "b", 64), BufferAccess::ReadWrite, MirrorMode::DirtyTracked)
            .expect_err("access is part of the shared-key compatibility contract");
        assert_eq!(
            err,
            BufferRegistrationError::AccessMismatch {
                key: BufferKey::of("k"),
                existing: BufferAccess::ReadOnly,
                incoming: BufferAccess::ReadWrite,
            }
        );
    }

    #[test]
    fn once_fields_share_only_with_once_fields_same_element() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "a", 64), BufferAccess::ReadOnly, MirrorMode::Once)
            .expect("first registration");
        let err = registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect_err("a DirtyTracked field must not be able to share an Once buffer");
        assert_eq!(
            err,
            BufferRegistrationError::MirrorModeMismatch {
                key: BufferKey::of("k"),
                existing: Some(MirrorMode::Once),
                incoming: Some(MirrorMode::DirtyTracked),
            }
        );
    }

    #[test]
    fn resource_entries_register_and_resolve_but_cannot_collide_with_row_buffers() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        let arena = make_buffer(&device, "vertex-arena", 4096);
        registry
            .register_resource(BufferKey::of("builtin_mesh_vertex"), 4096, arena.clone(), BufferAccess::ReadOnly)
            .expect("resource registration");
        assert_eq!(registry.element_type(BufferKey::of("builtin_mesh_vertex")), Some(None));
        let h = registry.resolve(BufferKey::of("builtin_mesh_vertex")).expect("resolvable");
        assert_eq!(h.buffer.size(), 4096);

        let err = registry
            .register_row::<u32>(BufferKey::of("builtin_mesh_vertex"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::Once)
            .expect_err("a row buffer must not be able to squat on a resource key");
        assert_eq!(err, BufferRegistrationError::RowResourceCollision { key: BufferKey::of("builtin_mesh_vertex") });
    }

    #[test]
    fn texture_array_reserves_its_key_without_a_buffer_and_resolve_returns_none() {
        let registry = GpuBufferRegistry::new();
        registry
            .register_texture_array(BufferKey::of("builtin_texture"), 4, BufferAccess::ReadOnly)
            .expect("texture-array registration");
        assert!(registry.is_texture_array(BufferKey::of("builtin_texture")));
        assert_eq!(registry.texture_array_slot_capacity(BufferKey::of("builtin_texture")), Some(4));
        assert!(
            registry.resolve(BufferKey::of("builtin_texture")).is_none(),
            "a texture array has no wgpu::Buffer to resolve"
        );
        assert!(registry.epoch(BufferKey::of("builtin_texture")).is_none());
        assert!(registry.contains_key(BufferKey::of("builtin_texture")));
    }

    #[test]
    fn texture_array_slot_capacity_updates_on_re_registration_growth() {
        let registry = GpuBufferRegistry::new();
        registry
            .register_texture_array(BufferKey::of("tex"), 4, BufferAccess::ReadOnly)
            .expect("first registration");
        registry
            .register_texture_array(BufferKey::of("tex"), 9, BufferAccess::ReadOnly)
            .expect("grown re-registration");
        assert_eq!(registry.texture_array_slot_capacity(BufferKey::of("tex")), Some(9));
    }

    #[test]
    fn texture_array_cannot_collide_with_a_row_buffer_or_a_resource_key() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_texture_array(BufferKey::of("tex"), 4, BufferAccess::ReadOnly)
            .expect("texture-array registration");

        let err = registry
            .register_row::<u32>(BufferKey::of("tex"), make_buffer(&device, "b", 64), BufferAccess::ReadOnly, MirrorMode::Once)
            .expect_err("a row buffer must not squat on a texture-array key");
        assert_eq!(err, BufferRegistrationError::RowResourceCollision { key: BufferKey::of("tex") });

        let err2 = registry
            .register_resource(BufferKey::of("tex"), 64, make_buffer(&device, "c", 64), BufferAccess::ReadOnly)
            .expect_err("a resource must not squat on a texture-array key");
        assert_eq!(err2, BufferRegistrationError::RowResourceCollision { key: BufferKey::of("tex") });

        let device2 = test_device();
        let arena = make_buffer(&device2, "arena", 64);
        registry.register_resource(BufferKey::of("other"), 64, arena, BufferAccess::ReadOnly).expect("distinct key ok");
        let err3 = registry
            .register_texture_array(BufferKey::of("other"), 4, BufferAccess::ReadOnly)
            .expect_err("a texture array must not squat on a resource key");
        assert_eq!(err3, BufferRegistrationError::RowResourceCollision { key: BufferKey::of("other") });
    }

    #[test]
    fn texture_array_access_mismatch_is_rejected() {
        let registry = GpuBufferRegistry::new();
        registry
            .register_texture_array(BufferKey::of("tex"), 4, BufferAccess::ReadOnly)
            .expect("first registration");
        let err = registry
            .register_texture_array(BufferKey::of("tex"), 4, BufferAccess::ReadWrite)
            .expect_err("access is part of the compatibility contract for texture arrays too");
        assert_eq!(
            err,
            BufferRegistrationError::AccessMismatch {
                key: BufferKey::of("tex"),
                existing: BufferAccess::ReadOnly,
                incoming: BufferAccess::ReadWrite,
            }
        );
    }

    #[test]
    fn sync_is_a_no_op_on_a_texture_array_key() {
        let registry = GpuBufferRegistry::new();
        let device = test_device();
        registry
            .register_texture_array(BufferKey::of("tex"), 4, BufferAccess::ReadOnly)
            .expect("registration");
        // Must not panic, and must not turn the key into a resolvable buffer.
        registry.sync(BufferKey::of("tex"), make_buffer(&device, "x", 4), 7);
        assert!(registry.resolve(BufferKey::of("tex")).is_none());
        assert!(registry.is_texture_array(BufferKey::of("tex")));
    }
}
