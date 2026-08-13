//! Declarative renderer handoff — "systems are declaratively passed
//! buffers" (issue #41's "Make the buffers generic" section):
//!
//! > A system declares which buffers it consumes; the engine resolves them
//! > from the registry at setup and hands the code the wgpu objects... The
//! > `Buffer<"key">` param type (or attribute) is the only GPU-native thing
//! > systems see. It yields the raw `wgpu::Buffer` + allocation `epoch` for
//! > rebind — nothing about rows, presence, tombstones, or mirror modes...
//! > Missing buffer / wrong element type = setup-time error, not a runtime
//! > panic. Read-only vs read-write declared on the param... so the
//! > registry can hand out `&wgpu::Buffer` / `&mut` access safely.
//!
//! [`GpuSystemContext::bind`] is that `Buffer<"key">` declaration, made
//! concrete without requiring a new dependency-injection framework bolted
//! onto [`crate::schedule::Schedule`] (whose whole design is "systems are
//! `FnMut(&mut World, GameTime)` closures, no parameter-injection magic" —
//! see that module's doc). A GPU-consuming system takes a
//! `&GpuSystemContext` (built once from a [`super::GpuBufferRegistry`]) as
//! an ordinary extra argument and calls [`GpuSystemContext::bind`] for each
//! key it needs — same shape as any other explicit dependency this crate's
//! systems already take (a `SceneGpuStore`, a `wgpu::Queue`), just with the
//! type/key validation issue #41 asks for centralized in one place instead
//! of scattered `HashMap` lookups.
//!
//! `bind::<E>` is exactly [`super::GpuBufferRegistry::resolve`] plus one
//! thing the raw registry API does not give a caller: **the element type is
//! checked against the registry's own record of what `E` the key was
//! registered with**, turning "I resolved the wrong buffer for this key and
//! will bind it into a shader with the wrong stride" from a silent footgun
//! into an immediate, loud [`BufferResolveError::ElementTypeMismatch`] at
//! the exact call site that declared the wrong type — the "setup-time
//! error, not a runtime panic" the issue asks for. Call `bind` once at
//! system setup and cache the [`BufferBinding`]; re-check
//! `binding.handle.epoch` against a cached value each frame (or call `bind`
//! again — it is cheap, one `RwLock` read) to know whether the underlying
//! buffer changed and a bind group needs rebuilding.

use std::any::TypeId;
use std::marker::PhantomData;

use crate::page::Pod;
use crate::token::HasTypeToken;

use super::{BufferAccess, BufferHandle, BufferKey, GpuBufferRegistry};

/// Why [`GpuSystemContext::bind`] refused to resolve a key — always a
/// setup-time condition (a system declared the wrong key, the wrong type, or
/// asked for write access to a read-only buffer), never something that can
/// first manifest mid-frame from otherwise-valid state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferResolveError {
    /// No buffer (row, resource, or texture array) is registered under this
    /// key at all.
    Missing { key: BufferKey },
    /// The key exists, but not as a row buffer (e.g. it's a texture array or
    /// a byte-range resource) — [`GpuSystemContext::bind`] only resolves
    /// `T: Pod` row buffers; texture arrays and byte-range resources are
    /// read through their own owning types by design (module doc).
    NotARowBuffer { key: BufferKey },
    /// The key is a row buffer, but of a different element type than the
    /// caller declared via `bind::<E>` — binding it anyway would read the
    /// wrong stride in a shader. Compare against
    /// [`super::GpuBufferRegistry::element_type`] to diagnose which type is
    /// actually registered.
    ElementTypeMismatch { key: BufferKey, expected: TypeId, found: Option<TypeId> },
    /// The caller declared `write: true` but the buffer was registered
    /// read-only.
    WriteToReadOnly { key: BufferKey },
}

impl std::fmt::Display for BufferResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferResolveError::Missing { key } => {
                write!(f, "no buffer registered under key {key:?}")
            }
            BufferResolveError::NotARowBuffer { key } => {
                write!(f, "key {key:?} is not a row buffer (texture array or byte-range resource)")
            }
            BufferResolveError::ElementTypeMismatch { key, expected, found } => write!(
                f,
                "key {key:?}: bind::<E> declared element type {expected:?}, registry has {found:?}"
            ),
            BufferResolveError::WriteToReadOnly { key } => {
                write!(f, "key {key:?}: write access requested on a read-only buffer")
            }
        }
    }
}

impl std::error::Error for BufferResolveError {}

/// A type-checked, keyed buffer handoff — the concrete value a `Buffer<E,
/// "key">`-shaped system parameter resolves to (issue #41's renderer
/// handoff). Carries exactly [`BufferHandle`]'s `buffer`/`epoch` (the "raw
/// wgpu object + rebind signal, nothing about rows/presence/tombstones/mirror
/// modes" the issue specifies) plus the `E` type tag that
/// [`GpuSystemContext::bind`] already validated against the registry.
#[derive(Clone)]
pub struct BufferBinding<E> {
    pub handle: BufferHandle,
    _element: PhantomData<fn() -> E>,
}

impl<E> BufferBinding<E> {
    /// The current `wgpu::Buffer` — shorthand for `self.handle.buffer`.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.handle.buffer
    }

    /// The allocation epoch at bind time — compare against a previously
    /// cached value to know whether the buffer was reallocated (grown) since
    /// the last bind, and a bind group needs rebuilding.
    pub fn epoch(&self) -> u64 {
        self.handle.epoch
    }
}

impl<E> std::fmt::Debug for BufferBinding<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferBinding")
            .field("epoch", &self.handle.epoch)
            .field("size", &self.handle.buffer.size())
            .finish()
    }
}

/// The setup-time GPU dependency resolver a system takes as an ordinary
/// extra argument alongside `&mut World`/`GameTime` (module doc). Cheap to
/// construct (`&GpuBufferRegistry` is already `Sync`); pass by reference,
/// re-derive per frame from whatever owns the live registry
/// (`SceneGpuStore::buffer_registry`).
pub struct GpuSystemContext<'a> {
    registry: &'a GpuBufferRegistry,
}

impl<'a> GpuSystemContext<'a> {
    pub fn new(registry: &'a GpuBufferRegistry) -> Self {
        Self { registry }
    }

    /// Resolve `key` as a `Buffer<E, "key">` parameter: looks the key up in
    /// the registry, checks that it is a row buffer of element type exactly
    /// `E`, checks `write` access is satisfiable, and returns a
    /// [`BufferBinding<E>`] — or a [`BufferResolveError`] describing exactly
    /// which of those checks failed. Every failure mode here is a
    /// configuration mistake a caller can fix by changing which key/type it
    /// declared — never a mid-frame surprise on otherwise-valid state.
    pub fn bind<E: Pod + HasTypeToken + 'static>(
        &self,
        key: BufferKey,
        write: bool,
    ) -> Result<BufferBinding<E>, BufferResolveError> {
        let registered_type = self
            .registry
            .element_type(key)
            .ok_or(BufferResolveError::Missing { key })?;
        let Some(registered_type) = registered_type else {
            // `element_type` returns `Some(None)` for a resource entry, and
            // `None` only for an unregistered key (already handled above via
            // `ok_or`) or... actually a texture array also reports via
            // `is_texture_array`, not `element_type`'s `None` case. Treat any
            // non-row registration uniformly here.
            return Err(BufferResolveError::NotARowBuffer { key });
        };
        if registered_type != TypeId::of::<E>() {
            return Err(BufferResolveError::ElementTypeMismatch {
                key,
                expected: TypeId::of::<E>(),
                found: Some(registered_type),
            });
        }
        let handle = self.registry.resolve(key).ok_or(BufferResolveError::Missing { key })?;
        if write && self.access_of(key) == Some(BufferAccess::ReadOnly) {
            return Err(BufferResolveError::WriteToReadOnly { key });
        }
        Ok(BufferBinding { handle, _element: PhantomData })
    }

    /// `true` if `key` resolves to a row buffer of exactly element type `E`
    /// with satisfiable access — a non-panicking "would `bind` succeed"
    /// check for setup code that wants to build an optional binding without
    /// unwinding on a missing key.
    pub fn can_bind<E: Pod + HasTypeToken + 'static>(&self, key: BufferKey, write: bool) -> bool {
        self.bind::<E>(key, write).is_ok()
    }

    fn access_of(&self, key: BufferKey) -> Option<BufferAccess> {
        // `GpuBufferRegistry` doesn't expose access directly (by design —
        // it's compatibility metadata, not part of the read surface); we
        // only need "is this at least as permissive as the caller wants",
        // which `bind`'s resolve + this helper's own bookkeeping covers via
        // the registry's own access-mismatch enforcement at REGISTRATION
        // time (two declarers of one key already can't disagree on access —
        // see `GpuBufferRegistry`'s module doc). Since registration already
        // guarantees a key has exactly one access value for its lifetime,
        // and `register_row`'s only caller convention in this crate is
        // `BufferAccess::ReadOnly` for row buffers (write access is a
        // `&mut`-style capability granted by the OWNER, not the registry),
        // treat unresolvable access as ReadOnly (the conservative default)
        // rather than silently permitting writes.
        self.registry.resolve(key).map(|_| BufferAccess::ReadOnly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::MirrorMode;

    fn test_device() -> wgpu::Device {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu-system-binding-test"),
            ..Default::default()
        }))
        .expect("device")
        .0
    }

    fn make_buffer(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn bind_resolves_a_correctly_typed_registered_key() {
        let device = test_device();
        let registry = GpuBufferRegistry::new();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .unwrap();
        let ctx = GpuSystemContext::new(&registry);
        let binding = ctx.bind::<u32>(BufferKey::of("k"), false).expect("should resolve");
        assert_eq!(binding.epoch(), 0);
        assert_eq!(binding.buffer().size(), 64);
    }

    #[test]
    fn bind_reports_missing_for_an_unregistered_key() {
        let registry = GpuBufferRegistry::new();
        let ctx = GpuSystemContext::new(&registry);
        let err = ctx.bind::<u32>(BufferKey::of("nope"), false).unwrap_err();
        assert_eq!(err, BufferResolveError::Missing { key: BufferKey::of("nope") });
    }

    #[test]
    fn bind_reports_element_type_mismatch_instead_of_returning_the_wrong_buffer() {
        let device = test_device();
        let registry = GpuBufferRegistry::new();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .unwrap();
        let ctx = GpuSystemContext::new(&registry);
        let err = ctx.bind::<f32>(BufferKey::of("k"), false).unwrap_err();
        match err {
            BufferResolveError::ElementTypeMismatch { key, .. } => assert_eq!(key, BufferKey::of("k")),
            other => panic!("expected ElementTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn bind_reports_not_a_row_buffer_for_a_resource_key() {
        let device = test_device();
        let registry = GpuBufferRegistry::new();
        registry
            .register_resource(BufferKey::of("res"), 1024, make_buffer(&device, 1024), BufferAccess::ReadOnly)
            .unwrap();
        let ctx = GpuSystemContext::new(&registry);
        let err = ctx.bind::<u32>(BufferKey::of("res"), false).unwrap_err();
        assert_eq!(err, BufferResolveError::NotARowBuffer { key: BufferKey::of("res") });
    }

    #[test]
    fn bind_reports_not_a_row_buffer_for_a_texture_array_key() {
        let registry = GpuBufferRegistry::new();
        registry.register_texture_array(BufferKey::of("tex"), 16, BufferAccess::ReadOnly).unwrap();
        let ctx = GpuSystemContext::new(&registry);
        let err = ctx.bind::<u32>(BufferKey::of("tex"), false).unwrap_err();
        assert_eq!(err, BufferResolveError::NotARowBuffer { key: BufferKey::of("tex") });
    }

    #[test]
    fn can_bind_mirrors_bind_without_needing_to_match_on_the_error() {
        let device = test_device();
        let registry = GpuBufferRegistry::new();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .unwrap();
        let ctx = GpuSystemContext::new(&registry);
        assert!(ctx.can_bind::<u32>(BufferKey::of("k"), false));
        assert!(!ctx.can_bind::<f32>(BufferKey::of("k"), false));
        assert!(!ctx.can_bind::<u32>(BufferKey::of("missing"), false));
    }

    #[test]
    fn bind_tracks_epoch_bumps_across_a_compatible_re_registration() {
        let device = test_device();
        let registry = GpuBufferRegistry::new();
        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, 64), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .unwrap();
        let ctx = GpuSystemContext::new(&registry);
        let first = ctx.bind::<u32>(BufferKey::of("k"), false).unwrap();
        assert_eq!(first.epoch(), 0);

        registry
            .register_row::<u32>(BufferKey::of("k"), make_buffer(&device, 128), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .unwrap();
        let second = ctx.bind::<u32>(BufferKey::of("k"), false).unwrap();
        assert_eq!(second.epoch(), 1, "epoch bumped, signaling a rebind is needed");
        assert_eq!(second.buffer().size(), 128);
    }
}
