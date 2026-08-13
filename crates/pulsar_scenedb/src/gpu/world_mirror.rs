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
use crate::gpu::{DirtyTrackedSceneBuffer, GpuColumnSet, SceneGpuStore};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// The initial capacity `#[derive(SceneStore)]`'s generated per-type
/// dispatch function passes to `T::register_gpu_columns_growable` when
/// auto-registering a never-before-seen `#[gpu]`-bearing type on its first
/// insert (issue #41: "auto-registration on first use"). Deliberately small
/// — matches [`GenerationMirror`]'s own initial capacity and every other
/// World-mirrored buffer's recommended sizing: growth is transparent and
/// cheap-to-start-small is the right default when the eventual entity count
/// isn't known ahead of time (see `SceneGpuStore::register_growable_gpu_buffer`'s
/// doc). A caller who wants a different starting size, or who wants to move
/// the first-growth cost off the per-insert critical path entirely, still
/// registers manually (or calls `World::reserve_gpu_mirror_capacity`) BEFORE
/// the type's first insert — manual registration always wins over
/// auto-registration, since this constant is only ever consulted when
/// nothing registered the type yet.
pub const DEFAULT_AUTO_REGISTER_CAPACITY: u32 = 64;

/// Row-indexed liveness/generation buffer, mirroring `World::entity_slots`'s
/// `generation` field on the GPU, keyed by `Entity::index()` exactly like
/// every other World-mirrored buffer. See the "Liveness" section of the
/// README for the read-side contract this exists to support: a GPU consumer
/// holding a captured `(row, generation)` pair compares `generation` against
/// this buffer's value at `row` before trusting any other World-mirrored
/// buffer's contents at that row -- the same staleness check
/// `World::is_alive` already performs on the CPU side, made available to
/// shaders.
///
/// Built on [`DirtyTrackedSceneBuffer<u32>`] — the same shadow+dirty-mask
/// mechanism `#[gpu]` (`DirtyTracked`) fields use — rather than the
/// CellStorage-oriented `GenerationBuffer` type ([`super::GenerationBuffer`])
/// or a hand-rolled pending-write queue. See "Why `DirtyTrackedSceneBuffer`,
/// not a lighter structure" below for why this is the right call even though
/// it costs one CPU-side `u32` shadow per row.
///
/// # Deferred, gated writes (SceneDB#39)
///
/// Writes here used to be immediate (`queue.write_buffer`, once per spawn
/// AND once per despawn — i.e. at least two synchronous GPU calls per churn
/// cycle) and unconditional: `World::spawn`/`despawn` wrote a generation
/// entry for every entity regardless of whether it ever carried a `#[gpu]`
/// field, since a freshly spawned entity's eventual components aren't known
/// yet at spawn time. Confirmed by reading the call sites while
/// investigating SceneDB#39 — not merely suspected. Fixed two ways,
/// together:
///
/// - **Deferred**: [`Self::note_gpu_bearing_insert`]/[`Self::note_despawn`]
///   only mark a row dirty ([`DirtyTrackedSceneBuffer::mark_dirty`], no GPU
///   work, read-lock-first fast path); the actual upload happens in
///   [`Self::flush`], coalesced the same way `#[gpu]` field flushes already
///   are. A row despawned and respawned within the same unflushed frame
///   collapses to one write of the final generation, not two.
/// - **Gated**: an entity that never receives a `#[gpu]`-bearing component
///   insert now costs *zero* GPU-mirror work, at spawn or at despawn —
///   [`GpuMirroredRows`] tracks, per row, whether it has ever actually
///   carried GPU-mirrored data, and both `note_*` methods no-op unless that
///   flag says there is something on the GPU that actually needs
///   invalidating. This is what makes "non-GPU entities are never affected"
///   true by construction rather than by convention: an entity with no
///   `#[gpu]` fields anywhere in its component set is now indistinguishable,
///   cost-wise, from spawning/despawning it with no [`GpuMirrorHandle`]
///   attached to the `World` at all.
///
/// # Why `DirtyTrackedSceneBuffer`, not a lighter structure
///
/// An earlier version of this fix used a hand-rolled `Mutex<HashMap<row,
/// bytes>>` pending-write queue instead, specifically to avoid paying a
/// persistent per-row CPU shadow for data that (for `Once`-mode fields) is
/// only ever written once. Real-device re-benchmarking after that version
/// showed it was the wrong trade: the `Mutex` (always exclusive, even
/// single-threaded) plus a `HashMap` insert plus a `Box<[u8]>` heap
/// allocation *per queued write* cost more, on the hot churn path, than the
/// coalescing it bought back -- low-density churn (a small fraction of a
/// large population respawning per frame, where few queued rows end up
/// adjacent) measured *slower* than the original immediate-write code, not
/// faster. `DirtyTrackedSceneBuffer::mark_dirty`'s read-lock-first fast path
/// (Helio#213) has none of that: no heap allocation (the shadow's already
/// there), no `HashMap` hashing, and multiple threads marking disjoint rows
/// don't serialize against each other at all. The one thing it costs that
/// the hand-rolled version didn't is a permanent `Vec<u32>` shadow sized to
/// capacity -- 4 bytes/row for this specific column, and paid only by
/// columns that actually opt into deferred writes (`Once`-mode fields and
/// this liveness mirror), not by anything else. That's the right place to
/// spend memory: a little, scoped to the fields that need it, in exchange
/// for using the one write path in this crate already proven fast under
/// real concurrent load.
pub struct GenerationMirror {
    buf: DirtyTrackedSceneBuffer<u32>,
    gpu_mirrored_rows: GpuMirroredRows,
}

impl GenerationMirror {
    fn new(device: Arc<wgpu::Device>) -> Self {
        // Small initial capacity, unbounded growth -- matches every other
        // World-mirrored buffer's recommended (register_gpu_columns_growable)
        // configuration; see that method's doc for why World-mirrored
        // buffers specifically should never set a max_capacity ceiling.
        Self {
            buf: DirtyTrackedSceneBuffer::new(device, "scenedb-world-mirror-generations", 64),
            gpu_mirrored_rows: GpuMirroredRows::new(),
        }
    }

    /// Called from `World::insert_inner` the first time `row` receives a
    /// component with `#[gpu]` fields (i.e. exactly once per entity, not
    /// once per `#[gpu]`-bearing component it happens to carry, and never
    /// for an entity that never gets one at all). Marks `generation` (the
    /// value already assigned at spawn, just not yet communicated to the
    /// GPU) dirty for the next [`Self::flush`]. A no-op if this row was
    /// already marked — a later insert of a second `#[gpu]`-bearing
    /// component type onto the same entity doesn't change its generation,
    /// so there is nothing new to mark.
    pub(crate) fn note_gpu_bearing_insert(&self, row: u32, generation: u32) {
        if self.gpu_mirrored_rows.mark_first_time(row) {
            self.buf.mark_dirty(row, generation);
        }
    }

    /// Called from `World::despawn_inner`. Marks the freshly-bumped
    /// `new_generation` dirty for the next [`Self::flush`] — but only if
    /// this row ever actually received a `#[gpu]`-bearing insert
    /// ([`Self::note_gpu_bearing_insert`]); otherwise there is nothing on
    /// the GPU at this row that needs invalidating, and this is a complete
    /// no-op (one `GpuMirroredRows` read-lock check, nothing else). Clears
    /// the row's flag either way, so a future entity that reuses this slot
    /// starts fresh and must earn its own GPU-mirror cost by actually
    /// carrying a `#[gpu]` field, exactly like a brand-new entity would.
    pub(crate) fn note_despawn(&self, row: u32, new_generation: u32) {
        if self.gpu_mirrored_rows.clear(row) {
            self.buf.mark_dirty(row, new_generation);
        }
    }

    /// Uploads every row marked dirty since the last flush, coalesced into
    /// contiguous runs. Called from `World::flush_gpu_mirror`, alongside
    /// (not instead of) [`super::SceneGpuStore::flush_gpu_mirror`] — call
    /// both once per frame, same as before this type deferred its writes.
    pub(crate) fn flush(&self, queue: &wgpu::Queue) {
        self.buf.flush(queue);
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        self.buf.with_buffer(f);
    }

    pub fn epoch(&self) -> u64 {
        self.buf.epoch()
    }
}

/// Per-row "has this entity ever received a `#[gpu]`-bearing component
/// insert" flags, backing [`GenerationMirror`]'s gating (see its doc).
/// `RwLock<Vec<AtomicBool>>`, not a plain `Mutex<Vec<bool>>`: the common
/// case (row already within the backing `Vec`, which is true for almost
/// every call once a `World` has warmed up) only needs the **read** side of
/// the lock plus one atomic swap -- multiple threads marking/clearing
/// *disjoint* rows proceed concurrently, the same read-lock-first shape
/// [`DirtyTrackedSceneBuffer::mark_dirty`] uses for its own shadow. Only
/// growing the backing `Vec` (a row past the current end) needs the write
/// side.
struct GpuMirroredRows {
    rows: RwLock<Vec<AtomicBool>>,
}

impl GpuMirroredRows {
    fn new() -> Self {
        Self { rows: RwLock::new(Vec::new()) }
    }

    /// Returns `true` the first time `row` is marked; returns `false` on
    /// every call after that for the same row, until [`Self::clear`] resets
    /// it.
    fn mark_first_time(&self, row: u32) -> bool {
        let idx = row as usize;
        {
            let rows = self.rows.read().expect("GpuMirroredRows lock poisoned");
            if idx < rows.len() {
                return !rows[idx].swap(true, Ordering::Relaxed);
            }
        }
        let mut rows = self.rows.write().expect("GpuMirroredRows lock poisoned");
        if idx >= rows.len() {
            rows.resize_with(idx + 1, || AtomicBool::new(false));
        }
        !rows[idx].swap(true, Ordering::Relaxed)
    }

    /// Clears `row`'s flag and returns what it was before clearing (i.e.
    /// whether the caller should treat this as "there was GPU-mirrored data
    /// here that needs invalidating"). A row past the end of the backing
    /// `Vec` (never marked) returns `false` without growing anything --
    /// read-lock only, never escalates.
    fn clear(&self, row: u32) -> bool {
        let idx = row as usize;
        let rows = self.rows.read().expect("GpuMirroredRows lock poisoned");
        if idx >= rows.len() {
            return false;
        }
        rows[idx].swap(false, Ordering::Relaxed)
    }
}

/// The resources `World`'s automatic GPU mirroring needs: the store every
/// `#[gpu]` field's buffer lives in, the queue to write through, and the
/// liveness/generation mirror (see [`GenerationMirror`]).
///
/// Attach via [`crate::world::World::attach_gpu_mirror`]. Cheap to clone
/// (`Arc<SceneGpuStore>` + `Arc<wgpu::Queue>` + `Arc<GenerationMirror>`).
#[derive(Clone)]
pub struct GpuMirrorHandle {
    store: Arc<SceneGpuStore>,
    queue: Arc<wgpu::Queue>,
    generations: Arc<GenerationMirror>,
}

impl GpuMirrorHandle {
    pub fn new(store: Arc<SceneGpuStore>, queue: Arc<wgpu::Queue>) -> Self {
        let generations = Arc::new(GenerationMirror::new(store.device_arc()));
        Self { store, queue, generations }
    }

    #[inline]
    pub fn store(&self) -> &SceneGpuStore {
        &self.store
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    #[inline]
    pub fn generations(&self) -> &GenerationMirror {
        &self.generations
    }
}

/// Writes every `#[gpu]` field of `data` into its registered GPU buffer at
/// `row`, honoring each field's declared [`crate::gpu::MirrorMode`]
/// (`GpuColumnDesc::mode`):
///
/// - **`Once`**: written only when `is_new_insert` is `true` — i.e. the
///   first time this entity gets this component, never again on a later
///   in-place update. Matches the mode's own documented meaning ("uploaded
///   once at registration and never touched again").
/// - **`DirtyTracked`** (the default): marked dirty
///   ([`SceneGpuStore::mark_gpu_row_dirty`]) instead of written immediately.
///   [`crate::world::World::flush_gpu_mirror`] performs the actual,
///   coalesced upload — call it once per frame, analogous to the
///   cell-mirrored path's own boundary-phase sync.
///
/// Each field's `ComponentId` is looked up in whichever of the three
/// registration maps it was actually registered through (fixed
/// [`SceneGpuStore::write_row_bytes`], growable
/// [`SceneGpuStore::write_row_bytes_growing`], or dirty-tracked
/// [`SceneGpuStore::mark_gpu_row_dirty`]) — a given field lives in exactly
/// one, so this is at most a few cheap map lookups, not redundant work. A
/// field whose buffer was never registered through any of them is silently
/// skipped, not an error — legitimate during bring-up.
///
/// Works for *any* `T: GpuColumnSet` with no per-type code beyond what
/// `#[derive(SceneStore)]` already generates — this walks `T::gpu_columns()`,
/// it doesn't need to know the struct's shape ahead of time. Called both
/// directly (by anything that already has a concrete `T: GpuColumnSet` in
/// hand) and indirectly, through each type's generated
/// [`GpuMirrorRegistration::dispatch`] function, by [`crate::world::World::insert`].
///
/// # Panics
///
/// Only if a field was registered growable with an explicit `max_capacity`
/// ceiling (via `register_growable_gpu_buffer`, not through the derive's
/// generated `register_gpu_columns_growable`, which never sets one) and
/// `row` exceeds it. This is deliberate, not an oversight: `World::insert`
/// has no `Result` to propagate a capacity failure through, so a caller who
/// opts into a hard ceiling on a World-mirrored column is opting into this
/// panic as the price of that ceiling — documented at
/// [`SceneGpuStore::register_growable_gpu_buffer`].
pub fn write_gpu_columns_at_row<T: GpuColumnSet>(
    store: &SceneGpuStore,
    queue: &wgpu::Queue,
    row: u32,
    data: &T,
    is_new_insert: bool,
) {
    for col in T::gpu_columns() {
        if col.mode == crate::gpu::MirrorMode::Once && !is_new_insert {
            continue; // Once fields never re-write after the first insert
        }
        // SAFETY: `field_offset` describes a field within `T`, computed by
        // the derive from `T`'s own layout (`offset_of!`) at macro-expansion
        // time, so this pointer is in-bounds of `data` and correctly aligned
        // for the field's own type.
        let field_ptr = unsafe { (data as *const T as *const u8).add(col.field_offset) };
        // A `heavy` field's `upload` mapper produces freshly-computed,
        // Element-sized bytes from the handle at `field_ptr` (see
        // `GpuUploadSource`) — an owned allocation, since the mapped bytes
        // don't live anywhere else. Every other field (the overwhelming
        // majority) reads its own bytes directly, zero-copy, exactly as
        // before `upload` existed.
        let mapped;
        let bytes: &[u8] = if let Some(upload) = col.upload {
            mapped = upload(field_ptr as *const ());
            &mapped
        } else {
            let size = col.field_token.desc().size as usize;
            // SAFETY: `size` is this field's own type's size (`T:
            // GpuColumnSet: Pod` guarantees every bit pattern in that range
            // is a valid read — no padding-UB, no enum niches to violate).
            unsafe { std::slice::from_raw_parts(field_ptr, size) }
        };
        let id = col.field_token.id();

        // SceneDB#39: `Once`-mode fields are registered through the SAME
        // `register_dirty_tracked_gpu_buffer` path as `DirtyTracked` ones
        // (`#[derive(SceneStore)]` does this automatically), and marked
        // dirty here the same way -- the two modes only differ in WHEN this
        // point is reached: `DirtyTracked` reaches it on every insert,
        // `Once` only on the first (the early `continue` above skips every
        // later one). Once marked, both flush identically. This reuses
        // `DirtyTrackedSceneBuffer::mark_dirty`'s proven, mostly-lock-free
        // fast path instead of a separate mechanism -- see
        // `GenerationMirror`'s doc for why a from-scratch alternative tried
        // first was measurably worse, not just different.
        if (col.mode == crate::gpu::MirrorMode::DirtyTracked || col.mode == crate::gpu::MirrorMode::Once)
            && store.mark_gpu_row_dirty(id, row, bytes)
        {
            continue;
        }
        if store.write_row_bytes(id, queue, bytes, row) {
            continue;
        }
        match store.write_row_bytes_growing(id, queue, bytes, row) {
            None => {} // not registered through any path -- bring-up, not an error
            Some(Ok(())) => {}
            Some(Err(cap_err)) => panic!(
                "World-mirrored GPU column (ComponentId {id:?}) hit its configured max_capacity \
                 ({}) at row {row} (requested {}) -- see SceneGpuStore::register_growable_gpu_buffer's \
                 doc: World-mirrored columns should be registered with max_capacity: None precisely \
                 to make this unreachable",
                cap_err.max, cap_err.requested,
            ),
        }
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
/// `data`: as documented on [`GpuMirrorRegistration::dispatch`]. `bool`:
/// `is_new_insert`, forwarded to [`write_gpu_columns_at_row`] — `true` on
/// the first insert of this component onto this entity, `false` on a later
/// in-place update, so `Once`-mode fields know whether to (re-)write at all.
pub type DispatchFn = fn(&GpuMirrorHandle, u32, *const (), bool);

pub struct GpuMirrorRegistration {
    pub component_id: fn() -> ComponentId,
    /// `data` must point to a live, correctly-aligned `T` for the
    /// registration's own (macro-generated, concrete) `T` — upheld by
    /// `World::insert_inner`, the only caller, which passes `&value as *const
    /// T as *const ()` for the exact `T` this registration's `component_id`
    /// resolved from.
    pub dispatch: DispatchFn,
}

pulsar_reflection::inventory::collect!(GpuMirrorRegistration);

fn registry_map() -> &'static HashMap<ComponentId, DispatchFn> {
    static MAP: OnceLock<HashMap<ComponentId, DispatchFn>> = OnceLock::new();
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
pub(crate) fn dispatch_for(id: ComponentId) -> Option<DispatchFn> {
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
                upload: None,
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

    fn test_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
        let data = unsafe { &*(data as *const TestComponent) };
        write_gpu_columns_at_row(mirror.store(), mirror.queue(), row, data, is_new_insert);
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
