//! Bridges [`crate::world::World`]'s entity-indexed archetype storage to
//! [`SceneGpuStore`]'s GPU-mirrored columns.
//!
//! `World` is `CellStorage`/`Handle`-free by design (it's the crate's
//! archetype ECS, not the paged storage layer `#[gpu]` mirroring was
//! originally built against — see `GpuColumnSet::write_gpu`'s
//! `Handle`-taking signature). This module lets a component's `#[gpu]`
//! fields be mirrored to their registered GPU buffer anyway, keyed by
//! `Entity::index()` instead of `Handle::index()` — the two are the same
//! *kind* of stable per-slot key, just from two different storage layers.
//!
//! The mechanism is entirely additive and opt-in:
//!
//! - Nothing here runs unless a [`GpuMirrorHandle`] has been attached to a
//!   `World` via [`crate::world::World::attach_gpu_mirror`]. Until then,
//!   `World::insert` behaves exactly as it always has.
//! - Once attached, every `World::insert`/`insert_tracked` call looks up
//!   `T`'s `ComponentId` in a link-time-populated dispatch registry (see
//!   "Dispatch mechanism" below) and, if `T` has `#[gpu]` fields, writes
//!   them to their buffer at row = `entity.index()`.
//!
//! This keeps [`crate::world`] itself free of any *public* GPU-specific
//! API surface — the new state lives behind a `#[cfg(feature = "gpu")]`
//! field and one `#[cfg(feature = "gpu")]` block in `insert_inner`, so a
//! `--no-default-features` build of this crate is byte-for-byte the
//! `World` that existed before this module — C0's actual guarantee (zero
//! GPU deps without the feature) holds exactly as before.
//!
//! # Dispatch mechanism (and why it isn't compile-time specialization)
//!
//! An earlier version of this module tried to resolve "does `T` have
//! `#[gpu]` fields" at compile time via the "autoref specialization" trick
//! (an inherent method competing with a blanket trait method, exploiting
//! Rust's inherent-beats-trait method-resolution priority). **That does not
//! work here, and the reason is worth recording precisely** so it doesn't
//! get re-attempted: `World::insert_inner<T: Component>` is itself an
//! unconstrained generic function. Rust resolves method calls inside a
//! generic function body once, using only `T`'s *declared* bounds
//! (`Component`) — never per-monomorphization — so a specialization
//! decision written inside that body can't observe whether the *substituted*
//! `T` additionally implements `GpuColumnSet`; only code written where `T`
//! is *already concrete* (e.g. inside a macro-generated, non-generic
//! function) can. Confirmed empirically two ways before landing this
//! version: (1) a minimal standalone repro of the autoref trick called from
//! inside a generic wrapper function always picked the fallback arm,
//! regardless of the concrete type substituted, while the exact same trait
//! setup called directly on concrete types (no generic wrapper) picked the
//! right arm every time; (2) `tests/world_gpu_mirror.rs`'s real-device
//! readback caught the resulting silent no-op directly (an all-zero buffer)
//! before this fix landed.
//!
//! The working mechanism is a link-time registry instead: `#[derive(SceneStore)]`
//! emits, for any type with at least one `#[gpu]` field, a **non-generic**
//! dispatch function (concrete `T` baked in at macro-expansion time) plus a
//! [`GpuMirrorRegistration`] submitted via `inventory` (the same link-time
//! registration mechanism `SubsystemRegistry`/`DynMethodRegistry` already
//! use elsewhere in this crate/`pulsar_reflection`). [`World::insert_inner`]
//! looks the registration up by the `ComponentId` it already computes for
//! archetype indexing (no extra `TypeId` resolution) — a single `HashMap`
//! lookup, paid only when a mirror is attached, same as `component_id::<T>()`
//! itself already costs a thread-local scan on every insert regardless.
//! Not literally free, but the closest achievable without nightly
//! `#[feature(specialization)]`, and correct — which the compile-time
//! attempt, despite compiling cleanly with no errors or warnings pointing
//! at the problem, was not.
use crate::component::ComponentId;
use crate::gpu::{GpuColumnSet, SceneGpuStore};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// The two resources `World`'s automatic GPU mirroring needs: the store
/// every `#[gpu]` field's buffer lives in, and the queue to write through.
///
/// Attach via [`crate::world::World::attach_gpu_mirror`]. Cheap to clone
/// (`Arc<SceneGpuStore>` + `Arc<wgpu::Queue>`).
#[derive(Clone)]
pub struct GpuMirrorHandle {
    store: Arc<SceneGpuStore>,
    queue: Arc<wgpu::Queue>,
}

impl GpuMirrorHandle {
    pub fn new(store: Arc<SceneGpuStore>, queue: Arc<wgpu::Queue>) -> Self {
        Self { store, queue }
    }

    #[inline]
    pub fn store(&self) -> &SceneGpuStore {
        &self.store
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Writes every `#[gpu]` field of `data` into its registered GPU buffer at
/// `row`, using each field's byte offset (`GpuColumnDesc::field_offset`,
/// computed by the derive at macro-expansion time) and
/// [`SceneGpuStore::write_row_bytes`].
///
/// Works for *any* `T: GpuColumnSet` with no per-type code beyond what
/// `#[derive(SceneStore)]` already generates — this walks `T::gpu_columns()`,
/// it doesn't need to know the struct's shape ahead of time. Called both
/// directly (by anything that already has a concrete `T: GpuColumnSet` in
/// hand) and indirectly, through each type's generated
/// [`GpuMirrorRegistration::dispatch`] function, by [`crate::world::World::insert`].
///
/// A field whose buffer hasn't been registered yet (`T::register_gpu_columns`
/// not called, or called with a device/capacity that predates this field)
/// is silently skipped, not an error — legitimate during bring-up, and
/// `write_row_bytes` itself already treats "unregistered" as a no-op for
/// the same reason.
pub fn write_gpu_columns_at_row<T: GpuColumnSet>(
    store: &SceneGpuStore,
    queue: &wgpu::Queue,
    row: u32,
    data: &T,
) {
    for col in T::gpu_columns() {
        let size = col.field_token.desc().size as usize;
        // SAFETY: `field_offset`/`size` describe a field within `T`, computed
        // by the derive from `T`'s own layout (`offset_of!` + `size_of`) at
        // macro-expansion time, so the byte range is in-bounds of `data` and
        // fully initialized. `T: GpuColumnSet: Pod` guarantees every bit
        // pattern in that range is a valid read (no padding-UB, no enum
        // niches to violate).
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (data as *const T as *const u8).add(col.field_offset),
                size,
            )
        };
        store.write_row_bytes(col.field_token.id(), queue, bytes, row);
    }
}

// ── Link-time dispatch registry ─────────────────────────────────────────

/// One `#[derive(SceneStore)]` type's entry in the world-mirror dispatch
/// table: `component_id` identifies the type (a plain `fn` pointer, not the
/// already-resolved `ComponentId`, since resolving it requires the global
/// `component_id::<T>()` registry lock and must happen lazily, not at
/// `inventory::submit!`'s const-eval time); `dispatch` is a **non-generic**
/// function — `T` is already concrete at the point the derive macro
/// generates it — that downcasts the type-erased `data` pointer back to
/// `&T` and calls [`write_gpu_columns_at_row`].
///
/// Not constructed by hand — `#[derive(SceneStore)]` emits one of these
/// (via `inventory::submit!`) for every type with at least one `#[gpu]`
/// field. Types with none don't submit a registration at all, so they
/// never appear in [`registry_map`] and cost nothing beyond the one
/// `HashMap` miss `World::insert` already pays when a mirror is attached.
pub struct GpuMirrorRegistration {
    pub component_id: fn() -> ComponentId,
    /// `data` must point to a live, correctly-aligned `T` for the
    /// registration's own (macro-generated, concrete) `T` — upheld by
    /// `World::insert_inner`, the only caller, which passes `&value as *const
    /// T as *const ()` for the exact `T` this registration's `component_id`
    /// resolved from.
    pub dispatch: fn(&GpuMirrorHandle, row: u32, data: *const ()),
}

pulsar_reflection::inventory::collect!(GpuMirrorRegistration);

fn registry_map() -> &'static HashMap<ComponentId, fn(&GpuMirrorHandle, u32, *const ())> {
    static MAP: OnceLock<HashMap<ComponentId, fn(&GpuMirrorHandle, u32, *const ())>> = OnceLock::new();
    MAP.get_or_init(|| {
        pulsar_reflection::inventory::iter::<GpuMirrorRegistration>()
            .map(|r| ((r.component_id)(), r.dispatch))
            .collect()
    })
}

/// Looks up `id`'s dispatch function, if `#[derive(SceneStore)]` generated
/// one for it (i.e. the type has at least one `#[gpu]` field). `id` is
/// expected to already be in hand — [`crate::world::World::insert_inner`]
/// computes it via `component_id::<T>()` for archetype indexing regardless
/// of GPU mirroring, so this adds exactly one `HashMap` lookup on top, not
/// a second `TypeId` resolution.
#[inline]
pub(crate) fn dispatch_for(id: ComponentId) -> Option<fn(&GpuMirrorHandle, u32, *const ())> {
    registry_map().get(&id).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuColumnDesc, MirrorMode};
    use crate::token::TypeToken;

    /// A minimal hand-rolled `GpuColumnSet` type (mirrors the shape
    /// `#[derive(SceneStore)]` would generate for one `#[gpu]` field),
    /// registered exactly the way the derive's generated code would.
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    struct TestField(u32);
    unsafe impl crate::page::Pod for TestField {}

    #[derive(Clone, Copy)]
    struct TestComponent {
        value: TestField,
    }
    unsafe impl crate::page::Pod for TestComponent {}
    impl GpuColumnSet for TestComponent {
        fn gpu_columns() -> Vec<GpuColumnDesc> {
            vec![GpuColumnDesc {
                field_token: TypeToken::of::<TestField>(),
                field_offset: std::mem::offset_of!(TestComponent, value),
                mode: MirrorMode::DirtyTracked,
                buffer_name: "value",
            }]
        }
        fn write_gpu(
            _store: &SceneGpuStore,
            _id: crate::gpu::CellId,
            _cell: &mut crate::cell::CellStorage,
            _handle: crate::handle::Handle,
            _data: &Self,
            _phase: &impl crate::gpu::SimulateWitness,
        ) {
        }
    }

    fn test_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const ()) {
        let data = unsafe { &*(data as *const TestComponent) };
        write_gpu_columns_at_row(mirror.store(), mirror.queue(), row, data);
    }

    pulsar_reflection::inventory::submit! {
        GpuMirrorRegistration {
            component_id: crate::component::component_id::<TestComponent>,
            dispatch: test_dispatch,
        }
    }

    #[test]
    fn a_type_with_a_submitted_registration_is_found_by_its_component_id() {
        let id = crate::component::component_id::<TestComponent>();
        assert!(
            dispatch_for(id).is_some(),
            "TestComponent submitted a GpuMirrorRegistration in this same module — must be found"
        );
    }

    #[test]
    fn a_type_with_no_registration_is_not_found() {
        struct NeverRegistered;
        let id = crate::component::component_id::<NeverRegistered>();
        assert!(dispatch_for(id).is_none());
    }
}
