//! The SceneDB-owned multi-cell device-side store (M2b-α §2, design Rev 2):
//! region-partitioned scene SSBOs shared across every registered CELL. Each
//! cell owns a disjoint `[row_base, row_base+capacity)` slice of the
//! transform and slot-mirror buffers and a disjoint
//! `[slot_base, slot_base+capacity+headroom)` slice of the generation buffer
//! (`RegionPool`, §7). Constructed on the engine-level device context (C0).
//!
//! Mirrored columns must be written via `SceneGpuStore::write_transform` and
//! compacted via `SceneGpuStore::compact_all`; raw column access bypasses
//! dirty tracking.

use super::{
    BufferAccess, BufferHandle, BufferKey, CapacityError, DirtyMask, DirtyTrackedGpuBufferDispatch,
    DirtyTrackedSceneBuffer, EngineGpuContext, GenerationBuffer, GpuBufferDispatch,
    GpuBufferRegistry, GrowableGpuBufferDispatch, GrowableSceneBuffer, RegionError, RegionPool,
    SceneBuffer, SimulateWitness, SubmissionTracker, SyncStats,
};
use crate::page::Pod;
use crate::cell::{CellStorage, PendingRetire};
use crate::component::{component_id, ComponentId};
use crate::handle::Handle;
use crate::spatial::InstanceInfo;
use crate::token::HasTypeToken;
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// One size class of region (design Rev 2 §2/§7): every cell registered
/// under this class gets a fixed-size `capacity`-row region, and at most
/// `max_resident_cells` such regions ever coexist.
#[derive(Debug, Clone, Copy)]
pub struct RegionClassConfig {
    pub capacity: u32,
    pub max_resident_cells: u32,
}

/// Fixed store capacities (SSBOs never reallocate — exceeding them is a hard
/// error at the call site, §8), expressed as size classes rather than one
/// flat cap (M2a's `GpuStoreConfig`).
#[derive(Debug, Clone)]
pub struct SceneGpuConfig {
    pub classes: Vec<RegionClassConfig>,
    /// Extra slots reserved per region beyond `capacity`, absorbing
    /// tombstoned (retired-but-not-yet-recycled) slots without stealing a
    /// neighbor's region (§4.1).
    pub tombstone_headroom: u32,
    /// Per-cell metadata SSBO entries (α: allocated, no writer).
    pub max_cells_metadata: u32,
}

impl SceneGpuConfig {
    /// The default tombstone headroom used across M2b-α fixtures.
    pub fn default_headroom() -> u32 {
        64
    }
}

/// Keyed-registry identity for the store's always-present built-in buffers
/// (issue #41's "Collapse map": "Store builtins (transform, generation) →
/// `GpuBuffer<...>` @ `"builtin_transform"` / `"builtin_generation"`"). The
/// transform/instance-info buffers already register under
/// `"scenedb-instances"`/`"scenedb-instance-info"` at construction (see
/// [`SceneGpuStore::new`]); these three cover the remaining built-ins that
/// were previously reachable only through their dedicated accessors
/// ([`SceneGpuStore::generation_buffer`], [`SceneGpuStore::slot_mirror_buffer`],
/// [`SceneGpuStore::cell_metadata_buffer`]) and NOT through the keyed
/// registry — a renderer/tooling consumer had no `Buffer<"key">`-style path
/// to them. Registered once at [`SceneGpuStore::new`] (all three are fixed
/// capacity, allocated once, never reallocated — no epoch/growth sync
/// needed after that single registration).
pub const GENERATION_BUFFER_KEY: BufferKey = BufferKey::of("builtin_generation");
pub const SLOT_MIRROR_BUFFER_KEY: BufferKey = BufferKey::of("builtin_slot_mirror");
pub const CELL_METADATA_BUFFER_KEY: BufferKey = BufferKey::of("builtin_cell_metadata");

/// The 8-byte per-cell metadata record `StreamingGrid::write_cell_metadata`
/// packs directly by hand (`f32` alpha + `u32` domain code) — defined here
/// purely so [`CELL_METADATA_BUFFER_KEY`] can be registered as a proper `T:
/// Pod` row buffer instead of an opaque byte-range resource. Not a new wire
/// format: `write_cell_metadata` still writes the same two `to_le_bytes()`
/// calls it always did: this type documents that layout for the registry,
/// it does not change it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellMetadataRow {
    pub alpha: f32,
    pub domain: u32,
}
const _: () = assert!(std::mem::size_of::<CellMetadataRow>() == 8);
// SAFETY: `#[repr(C)]`, `Copy`, both fields are themselves POD (f32/u32),
// and the const assert pins the layout to exactly 8 bytes with no hidden
// padding — matching `write_cell_metadata`'s byte-for-byte packing.
unsafe impl crate::page::Pod for CellMetadataRow {}
// SAFETY: a bare marker impl (`HasTypeToken` derives its `TypeToken` purely
// from `Self`'s `TypeId`/layout via the blanket impl this crate provides for
// every `Pod` type elsewhere) — see `token.rs`. `CellMetadataRow` needs no
// bespoke reflection metadata beyond what registering it as a row-buffer
// element requires (`Pod + HasTypeToken + 'static`).

/// How a GPU-mirrored field is synced to the GPU buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MirrorMode {
    /// Delta-synced every frame via dirty tracking.  Default for `#[gpu]`.
    DirtyTracked,
    /// Uploaded once at registration and never touched again (e.g. asset data).
    Once,
}

/// Type-erased `handle bytes -> heavy element bytes` mapper for a
/// `#[gpu(mirror = Once, heavy)]` field whose declared type implements
/// [`GpuUploadSource`]. `handle_ptr` points at the field's own bytes (the
/// lightweight CPU handle, e.g. 4-16 bytes); the return value is the
/// freshly-computed heavy element's bytes, sized to `GpuUploadSource::Element`
/// — NOT to the handle. `None` on [`GpuColumnDesc::upload`] means "no split":
/// the field's own type IS the GPU buffer's element type, exactly as every
/// `#[gpu]` field (including plain `Once`) has always worked.
///
/// A plain `fn` pointer, not a boxed closure: the derive generates a
/// non-capturing closure (concrete types baked in at macro-expansion time),
/// which coerces to `fn` for free — no allocation, no dynamic dispatch
/// beyond the one indirect call already implied by a `fn` pointer.
pub type UploadMapperFn = fn(handle_ptr: *const ()) -> Vec<u8>;

/// Declares that a `#[gpu(mirror = Once, heavy)]` field's CPU-side type is a
/// lightweight HANDLE for a separate, heavier GPU-resident type — the CPU
/// never holds a full copy of `Element`. Tier 1 calls [`Self::upload_element`]
/// to produce it at registration, and again whenever the handle changes
/// (`World::get_mut`'s [`crate::world::Mut`] guard marks the field's GPU slot
/// dirty on drop, the same as any other mutated `#[gpu]` field).
///
/// This is the "CPU Asset Handle / Descriptor Pattern": the field's CPU type
/// (`Self`) is the handle — a `u32` slot index, an asset ID, a small
/// `Copy`/`Pod` descriptor — and `Element` is the heavy GPU-native record
/// (a full mesh-metadata row, a material, anything far larger than the
/// handle) that actually lives in VRAM. Reads (`world.get::<T>()`) return the
/// handle instantly from system RAM; they never trigger a VRAM readback.
///
/// Only meaningful for `#[gpu(mirror = Once)]` fields — `DirtyTracked`
/// fields keep a full CPU shadow of whatever type they declare by design (so
/// delta-sync has something to diff against), which the handle/heavy split
/// has no useful meaning against; the derive rejects `heavy` combined with
/// `mirror = DirtyTracked` at macro-expansion time.
pub trait GpuUploadSource {
    /// The heavy GPU-resident type this handle maps to. Must be `Pod` (it's
    /// written directly into a GPU buffer) and `Send + Sync` (the buffer
    /// storing it is shared, like every other GPU-mirrored column).
    type Element: crate::page::Pod + crate::token::HasTypeToken + Send + Sync + 'static;

    /// Produce the current heavy element for this handle. Called by Tier 1,
    /// never by ordinary CPU-side code — `world.get::<T>()` returns the
    /// handle itself, not this.
    fn upload_element(&self) -> Self::Element;
}

/// Describes one GPU-mirrored field in a component type.
/// Collected by `GpuColumnSet::gpu_columns()` and used by the derive macro's
/// generated `write_gpu()` and by `SceneGpuStore::sync_all()`.
#[derive(Clone, Debug)]
pub struct GpuColumnDesc {
    /// TypeToken of the field's type (used for column lookup + GPU buffer lookup).
    pub field_token: crate::token::TypeToken,
    /// Byte offset of the field within the struct (via `offset_of!`).
    pub field_offset: usize,
    /// Mirror mode: DirtyTracked (per-frame delta) or Once (bulk upload).
    pub mode: MirrorMode,
    /// Logical buffer name for GPU binding.
    pub buffer_name: &'static str,
    /// `Some` for a `#[gpu(mirror = Once, heavy)]` field whose type
    /// implements [`GpuUploadSource`] — see [`UploadMapperFn`]'s doc. `None`
    /// for every other field (the overwhelming majority): the field's own
    /// bytes ARE the GPU buffer's element, written directly with no mapping
    /// step, exactly as before this existed.
    pub upload: Option<UploadMapperFn>,
}

/// Trait for GPU-mirrored component types.  Generated by `#[derive(SceneStore)]`.
pub trait GpuColumnSet: crate::page::Pod + crate::token::HasTypeToken {
    /// Describe each `#[gpu]` field in declaration order.
    fn gpu_columns() -> Vec<GpuColumnDesc>;

    /// Write a component instance's GPU fields to the store + cell.
    /// Called by `SceneGpuStore::write_gpu()`.
    fn write_gpu(
        store: &SceneGpuStore,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &Self,
        phase: &impl SimulateWitness,
    );
}

/// Opaque handle to a registered cell's region assignment. Indexes
/// `SceneGpuStore`'s internal per-cell state; never crosses the FFI/shader
/// boundary (that's `row_region_base`'s job).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Write,
    Retired,
    Compacted,
}

pub(crate) struct QueuedRetire {
    pub(crate) pending: PendingRetire,
    pub(crate) serial: u64,
}

/// Per-cell GPU-side bookkeeping: the region assignment, the dirty state
/// that used to live on `GpuStore` directly (M2a), and the deferred-retire
/// queue (now one per cell rather than store-wide).
pub(crate) struct CellGpuState {
    /// Size class this cell was registered under (M2b-β `unregister_cell`:
    /// selects which `row_pools`/`slot_pools` entry to return the region
    /// pair to at eviction).
    pub(crate) class: usize,
    pub(crate) row_base: u32,
    pub(crate) slot_base: u32,
    /// Class capacity + headroom; bounds every gen/slot write into this
    /// cell's region.
    pub(crate) slot_capacity: u32,
    /// Per-column dirty masks in a DENSE table indexed by `ComponentId.0`,
    /// replacing the old named `dirty_transforms` / `dirty_infos` fields
    /// (pre-work item 2). The slot-mirror dirty mask (`dirty_slots`) is kept
    /// separate because it is driven by the self-healing boundary scan, not
    /// by column writes.
    ///
    /// **Why dense, not a `HashMap`:** every mirrored write (`write_transform`,
    /// `write_instance_info`, and every derive-generated `write_gpu`) marks a
    /// row through this table, so it sits in the hottest path the crate has.
    /// `ComponentId`s are dense and allocated from 1 (`component::component_id`),
    /// so a `Vec` index replaces a hash of the key on every single write.
    /// Measured on the delta-vs-legacy head-to-head (`legacy_model_bench`,
    /// 2 runs): the hashed form cost ~+15% CPU at 1% mutation and ~+50% at
    /// 100% mutation versus the named-field form it generalized. Index 0 is
    /// always `None` (`ComponentId` 0 is reserved); the table is sized to the
    /// highest registered GPU-column id at `register_cell`.
    pub(crate) dirty_columns: Vec<Option<DirtyMask>>,
    dirty_slots: DirtyMask,
    /// Per-row global-slot staging. `sync_all`'s self-healing boundary scan
    /// is scan-first, not dirty-mask-first: it compares `slot_shadow` against
    /// the authoritative slot column for every occupied row, and for each
    /// mismatch it fills this scratch entry AND marks `dirty_slots` together
    /// (neither drives the other) before uploading into the shared
    /// slot-mirror SSBO (T4; C6 GPU handle validation).
    slot_scratch: Vec<u32>,
    /// Per-ROW shadow of the last LOCAL slot uploaded into the mirror for
    /// that row; `u32::MAX` = never uploaded. Read and written ONLY by
    /// `sync_all`'s self-healing boundary scan (`&mut self`), which compares
    /// it against the authoritative slot column and re-uploads every
    /// mismatch. Row-scoped on purpose: per-event triggers (gen-gate,
    /// write-path shadow check, compaction marks) each missed a staleness
    /// path — e.g. an alloc re-occupying a compaction-vacated row that is
    /// never written (Task 4 review + re-review); the boundary scan closes
    /// them all with one invariant.
    slot_shadow: Vec<u32>,
    /// Per-cell deferred-retire queue; nondecreasing serials (debug-asserted,
    /// T11).
    pub(crate) pending: VecDeque<QueuedRetire>,
    /// CPU-side shadow of the last generation uploaded per LOCAL slot (§4
    /// delta-minimality on the write path), seeded from the registry at
    /// `register_cell`. Atomic because `write_transform` takes `&self`.
    gen_shadow: Vec<AtomicU32>,
}

impl CellGpuState {
    /// The dirty mask for one mirrored column, or `None` if this cell has no
    /// such column registered. One bounds-checked index — no hashing (see
    /// [`CellGpuState::dirty_columns`]).
    #[inline]
    fn dirty_column(&self, id: ComponentId) -> Option<&DirtyMask> {
        self.dirty_columns.get(id.0 as usize).and_then(Option::as_ref)
    }

    /// Every registered column's `(id, mask)` pair, skipping the dense
    /// table's holes. Frame-boundary use (compact/sync), not the write path.
    #[inline]
    fn dirty_columns_iter(&self) -> impl Iterator<Item = (ComponentId, &DirtyMask)> {
        self.dirty_columns
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.as_ref().map(|mask| (ComponentId(i as u32), mask)))
    }
}

/// One cell's region assignment paired with its mutable storage, for the
/// bulk `*_all` frame-boundary stages.
///
/// The (id, cell) pairing is TRUSTED: the store cannot verify that `cell` is
/// the storage `id` was registered with, and a mismatched pair commits
/// retires and dirty marks into the wrong cell's regions.
pub struct CellSlot<'a> {
    pub id: CellId,
    pub cell: &'a mut CellStorage,
}

/// Which registration path a shared buffer was created through — the same
/// `BufferKey` may never span two paths (a key registered fixed/
/// cell-mirrored and also growable/World-mirrored is a config error, not a
/// shared buffer).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SharedKindTag {
    Fixed,
    Growable,
    DirtyTracked,
}

/// One shared buffer's live dispatch object, aliased under every
/// `ComponentId` that declared its `BufferKey`.
enum SharedKind {
    Fixed(Arc<dyn GpuBufferDispatch>),
    Growable(Arc<dyn GrowableGpuBufferDispatch>),
    DirtyTracked(Arc<dyn DirtyTrackedGpuBufferDispatch>),
}

/// Store-side record of one shared `BufferKey`: the compatibility metadata
/// every later declaration of the same key is validated against, plus the
/// live dispatch object. This is the registry-collapse authority — the
/// `GpuBufferRegistry` is populated alongside it (so the renderer handoff
/// and future resource-entry namespace collisions work), but the live Arc
/// here is what aliasing actually shares, so growth/epoch/readback stay
/// consistent across every `ComponentId` that declares the key.
struct SharedOwner {
    element_type_id: TypeId,
    element_size: usize,
    /// Row capacity at registration. Fixed/cell-mirrored shared buffers must
    /// agree on this exactly (a shared key spanning two different capacities
    /// is a config error); growable/dirty-tracked buffers carry it as
    /// metadata only.
    capacity: u32,
    access: BufferAccess,
    mode: MirrorMode,
    kind: SharedKind,
}

/// The SceneDB-owned multi-cell device-side store (M2b-α §2): persistent
/// region-partitioned scene SSBOs, the mirrored-column writer, delta-sync,
/// and the retirement drain — generalizing M2a's single-cell `GpuStore` to
/// N cells sharing one set of buffers via `RegionPool` (§7).
pub struct SceneGpuStore {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// The keyed buffer registry (Tier 1, `gpu::buffer_registry`): the
    /// shared namespace every registration flows through. Row buffers are
    /// registered here at first-creation (validating against resource
    /// entries); [`Self::resolve_buffer_handle`] keeps this entry's
    /// buffer/epoch synced to the LIVE dispatch object on every call (via
    /// [`GpuBufferRegistry::sync`]) and returns [`GpuBufferRegistry::resolve`]
    /// as its terminal step — that's what makes this registry the actual
    /// renderer handoff rather than a snapshot frozen at first registration.
    /// `owners` below still owns the live dispatch objects (`Arc<dyn
    /// GpuBufferDispatch>` etc.) that `write_row_bytes`/`mark_column_dirty`
    /// write through; this registry is the read-side keyed view over them.
    registry: GpuBufferRegistry,
    /// One entry per shared `BufferKey` (and per unique auto-generated key),
    /// holding the live dispatch object + the compatibility metadata every
    /// later declaration of the same key is checked against. This is what
    /// makes `#[gpu(buffer = "key")]` collapse work: a second component
    /// declaring the same key adopts this Arc instead of allocating another
    /// buffer, so both components' `ComponentId`s alias ONE physical buffer.
    ///
    /// `RwLock`-backed (not a plain `HashMap`, unlike before this doc note):
    /// registration ([`Self::register_gpu_buffer`] and friends) needs to be
    /// reachable through the `Arc<SceneGpuStore>` a `World`'s attached
    /// [`crate::gpu::GpuMirrorHandle`] holds — and an `Arc` never yields
    /// `&mut` access once more than one owner exists (`World`'s mirror
    /// clone plus whatever the caller kept). Interior mutability here is
    /// what makes "auto-registration on first use"
    /// (`world_mirror::write_gpu_columns_at_row`'s dispatch caller) possible
    /// at all — see that module's doc for the full rationale. This is
    /// registration-time-only churn (a handful of calls at startup, or once
    /// per never-before-seen `#[gpu]` type on first insert); the actual
    /// per-frame hot path (`write_row_bytes`/`mark_gpu_row_dirty`/`sync_all`)
    /// only ever takes the read side.
    owners: RwLock<HashMap<BufferKey, SharedOwner>>,
    /// Reverse map `ComponentId -> BufferKey`, populated at registration.
    /// Lets `register_cell`'s dirty-mask warm-up skip aliased columns (a
    /// shared buffer is warmed through exactly one ComponentId, so the first
    /// sync doesn't let one alias's cold bytes overwrite another's rows).
    column_keys: RwLock<HashMap<ComponentId, BufferKey>>,
    /// Type-erased GPU buffers keyed by `ComponentId`.  Replaces the old
    /// concrete `transforms` / `instance_infos` fields (pre-work item 3).
    /// Registered via [`Self::register_gpu_buffer`]. `Arc` (not `Box`)
    /// because two `ComponentId`s declaring the same `BufferKey` alias the
    /// SAME dispatch object — that is the registry-collapse sharing.
    gpu_buffers: RwLock<HashMap<ComponentId, Arc<dyn GpuBufferDispatch>>>,
    /// Growable counterpart to `gpu_buffers`, disjoint from it — a given
    /// `ComponentId` is registered in exactly one of the two maps, never
    /// both. Registered via [`Self::register_growable_gpu_buffer`]; see
    /// `gpu::growable_scene_buffer`'s module docs for why this needs its own
    /// map/trait rather than living in `gpu_buffers` (`GpuBufferDispatch::
    /// buffer(&self) -> &wgpu::Buffer` can't be implemented soundly once the
    /// buffer lives behind the `RwLock` growth requires).
    growable_gpu_buffers: RwLock<HashMap<ComponentId, Arc<dyn GrowableGpuBufferDispatch>>>,
    /// Dirty-tracked counterpart to `growable_gpu_buffers` -- disjoint from
    /// both it and `gpu_buffers`; a given `ComponentId` lives in exactly one
    /// of the three. Registered via
    /// [`Self::register_dirty_tracked_gpu_buffer`], for `#[gpu(mirror =
    /// DirtyTracked)]` World-mirrored fields (the default `#[gpu]` mode) --
    /// writes are marked dirty ([`Self::mark_gpu_row_dirty`]) rather than
    /// uploaded immediately; [`Self::flush_gpu_mirror`] performs the actual
    /// coalesced upload.
    /// `#[gpu(mirror = Once)]` World-mirrored fields (SceneDB#39) are ALSO
    /// registered here, not in `growable_gpu_buffers` — Once-mode's "never
    /// re-write after the first insert" behavior is enforced one layer up,
    /// at the call site that decides whether to mark a row dirty at all
    /// (`world_mirror::write_gpu_columns_at_row`'s early `continue` on a
    /// non-first insert), not by which map the column lives in. Reusing
    /// this proven, read-lock-first, mostly-lock-free path instead of a
    /// separate immediate-write or hand-rolled pending-queue mechanism is
    /// deliberate — see `GenerationMirror`'s doc (`gpu::world_mirror`) for
    /// why an earlier, more "obviously cheap" alternative measured worse in
    /// practice.
    dirty_tracked_gpu_buffers: RwLock<HashMap<ComponentId, Arc<dyn DirtyTrackedGpuBufferDispatch>>>,
    /// Shared pools backing `Vec<T>`-typed `#[gpu]` fields
    /// ([`crate::gpu::VarLenGpuPool`]), keyed by [`BufferKey`] directly
    /// (not `ComponentId`) — the pool itself is the thing multiple fields
    /// share via `#[gpu(buffer = "key")]`, same sharing intent as
    /// `owners`/`gpu_buffers` above, kept in its own map rather than folded
    /// into them: a var-len pool has no fixed per-row stride the way every
    /// other registration here does (`assert_share_compatible`'s size check
    /// has no meaning for it), and its "row" concept — an offset/count pair
    /// into a suballocated pool — is fundamentally different from a plain
    /// `T`-per-row column. `Arc<dyn Any + Send + Sync>`, downcast by
    /// [`Self::var_len_pool`] to the caller's own concrete element type —
    /// the caller always knows `T` statically (it comes from the `Vec<T>`
    /// field's own declared type), so this is a plain, infallible-in-
    /// practice downcast, not a real type-erasure boundary.
    var_len_pools: RwLock<HashMap<BufferKey, Arc<dyn std::any::Any + Send + Sync>>>,
    slot_mirror: SceneBuffer<u32>,
    generations: GenerationBuffer,
    // `material` (32-byte placeholder buffer + `material_buffer()` accessor)
    // retired M3-α T11 (Rev 2.4 R8, approved 2026-07-16): the real 64-byte
    // material row (`gpu::MaterialRow`) is owned by the standalone
    // `gpu::MaterialRegistry`, mirroring `MeshRegistry`'s shape — not by
    // this store. The placeholder was never written to by anything and
    // predated R8's row layout by two size classes (32 B here vs. 64 B
    // there); keeping both would leave two unrelated "material buffer"
    // concepts in the crate (Ownership Law, CONTRACTS C0: one clear owner
    // per buffer).
    cell_metadata: wgpu::Buffer,
    tracker: SubmissionTracker,
    phase: Phase,
    /// One row pool and one slot pool per size class, base offsets laid end
    /// to end in class order (§7).
    row_pools: Vec<RegionPool>,
    slot_pools: Vec<RegionPool>,
    /// One slot per ever-registered `CellId`. `None` means the cell was
    /// evicted (`unregister_cell`, M2b-β §4.1) — `CellId`s are NOT recycled,
    /// so a hole here is permanent for the life of the store; a fresh
    /// registration always pushes a new `Some(..)` at the end regardless of
    /// how many holes precede it.
    cells: Vec<Option<CellGpuState>>,
    /// Instrumentation: total generation-buffer writes issued across every
    /// cell, so tests can assert generation-upload minimality.
    gen_writes: AtomicU64,
}

impl SceneGpuStore {
    /// Cheap `Arc` clone of the device this store was constructed with.
    /// Needed by anything that must allocate a NEW buffer alongside this
    /// store's own (e.g. [`crate::gpu::world_mirror::GenerationMirror`],
    /// which owns a device-independent growable buffer rather than one
    /// registered in this store's own `gpu_buffers`/`growable_gpu_buffers`
    /// maps) without threading a separate `Arc<wgpu::Device>` through every
    /// call site that already has a `&SceneGpuStore` in hand.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// This store's own queue — the same one every internal write already
    /// goes through. Lets a caller that only has a `&SceneGpuStore` in hand
    /// (e.g. `SceneDb::step_gpu`, driving the automatic World-mirror flush)
    /// reach a `&wgpu::Queue` without separately threading one through.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn new(ctx: &EngineGpuContext, cfg: SceneGpuConfig) -> Self {
        let mut row_pools = Vec::with_capacity(cfg.classes.len());
        let mut slot_pools = Vec::with_capacity(cfg.classes.len());
        let mut row_offset = 0u32;
        let mut slot_offset = 0u32;
        for class in &cfg.classes {
            let slot_region_size = class.capacity + cfg.tombstone_headroom;
            row_pools.push(RegionPool::new(row_offset, class.capacity, class.max_resident_cells));
            slot_pools.push(RegionPool::new(slot_offset, slot_region_size, class.max_resident_cells));
            // Checked accumulation: a pathological config must fail loudly at
            // construction, not wrap into silently-undersized SSBOs.
            row_offset = row_offset
                .checked_add(
                    class
                        .capacity
                        .checked_mul(class.max_resident_cells)
                        .expect("row capacity overflow"),
                )
                .expect("row capacity overflow");
            slot_offset = slot_offset
                .checked_add(
                    slot_region_size
                        .checked_mul(class.max_resident_cells)
                        .expect("slot capacity overflow"),
                )
                .expect("slot capacity overflow");
        }
        let store = Self {
            device: Arc::clone(ctx.device()),
            queue: Arc::clone(ctx.queue()),
            registry: GpuBufferRegistry::new(),
            owners: RwLock::new(HashMap::new()),
            column_keys: RwLock::new(HashMap::new()),
            gpu_buffers: RwLock::new(HashMap::new()),
            growable_gpu_buffers: RwLock::new(HashMap::new()),
            dirty_tracked_gpu_buffers: RwLock::new(HashMap::new()),
            var_len_pools: RwLock::new(HashMap::new()),
            slot_mirror: SceneBuffer::new(ctx.device(), "scenedb-slot-mirror", row_offset),
            generations: GenerationBuffer::new(ctx.device(), slot_offset),
            // Per-cell metadata stride is 8 bytes (design §4.1: f32 alpha +
            // u32 domain). Allocated at final stride now (§10); α has no
            // writer.
            cell_metadata: ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("scenedb-cell-metadata"),
                size: cfg.max_cells_metadata as u64 * 8,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            tracker: SubmissionTracker::new(),
            phase: Phase::Write,
            row_pools,
            slot_pools,
            cells: Vec::new(),
            gen_writes: AtomicU64::new(0),
        };
        // Register built-in GPU buffers under stable, documented keys — these
        // are the store's two always-present shared buffers (every cell's
        // transform / instance-info region lives in one physical buffer).
        store.register_gpu_buffer::<[f32; 16], [f32; 16]>(
            row_offset,
            ctx.device(),
            BufferKey::of("scenedb-instances"),
            MirrorMode::DirtyTracked,
        );
        store.register_gpu_buffer::<InstanceInfo, InstanceInfo>(
            row_offset,
            ctx.device(),
            BufferKey::of("scenedb-instance-info"),
            MirrorMode::DirtyTracked,
        );
        // Remaining store builtins (issue #41 collapse map): reachable by
        // key through the SAME registry the shared/`#[gpu(buffer=...)]`
        // fields resolve through, alongside their existing dedicated
        // accessors (`generation_buffer()`/`slot_mirror_buffer()`/
        // `cell_metadata_buffer()`, unchanged). All three are fixed-capacity
        // and allocated exactly once, right here — never reallocated, so
        // one `register_row` call each is the complete registration; no
        // `sync` call site is needed anywhere else in this store for them.
        store
            .registry
            .register_row::<u32>(GENERATION_BUFFER_KEY, store.generations.buffer().clone(), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("GENERATION_BUFFER_KEY is a fresh key on every new store");
        store
            .registry
            .register_row::<u32>(SLOT_MIRROR_BUFFER_KEY, store.slot_mirror.buffer().clone(), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("SLOT_MIRROR_BUFFER_KEY is a fresh key on every new store");
        store
            .registry
            .register_row::<CellMetadataRow>(CELL_METADATA_BUFFER_KEY, store.cell_metadata.clone(), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)
            .expect("CELL_METADATA_BUFFER_KEY is a fresh key on every new store");
        store
    }

    /// The Tier 1 keyed buffer registry this store registers every built-in
    /// and `#[gpu(buffer = "...")]`-shared buffer into (issue #41's single
    /// namespace: "the registry — the single namespace resource entries...
    /// will eventually share"). Exposed so external, non-`SceneGpuStore`-
    /// owned GPU asset stores (`MeshRegistry`, `ClusterBuffer`,
    /// `MaterialRegistry`, `GeometryArena`, `TextureStore`, and friends) can
    /// register their OWN buffers into the SAME registry this store already
    /// builds — see each type's `register_key`/`register_into` method. Read-
    /// only: registration into it still goes through `GpuBufferRegistry`'s
    /// own `register_row`/`register_resource`/`register_texture_array`,
    /// which validate compatibility exactly the same way whether the caller
    /// is this store or an external asset registry.
    pub fn buffer_registry(&self) -> &GpuBufferRegistry {
        &self.registry
    }

    /// Diagnostic-only convenience: [`super::readback_row`] for a key
    /// registered as a row buffer in this store's [`Self::buffer_registry`]
    /// (a store builtin, or any `#[gpu(buffer = "...")]`-shared key). `None`
    /// if `key` isn't registered as a row buffer at all (unregistered,
    /// registered as a resource, or registered as a texture array — see
    /// [`GpuBufferRegistry::resolve`]).
    ///
    /// Like every function in [`super::readback`], this is explicit,
    /// decoupled, diagnostic-only VRAM readback: it blocks the calling
    /// thread on `device.poll`, and nothing in the `get`/`get_mut`/query
    /// path ever calls it. See that module's doc for the full rationale.
    pub fn readback_row<T: Pod>(&self, device: &wgpu::Device, key: BufferKey, row: u32) -> Option<T> {
        let handle = self.registry.resolve(key)?;
        Some(super::readback_row::<T>(device, &self.queue, &handle.buffer, row))
    }

    pub fn tracker(&self) -> &SubmissionTracker {
        &self.tracker
    }

    /// Register a GPU buffer for a Pod column type.  Called automatically
    /// for `[f32; 16]` (transforms) and `InstanceInfo` during [`Self::new`];
    /// derive macros for custom SceneComponent types will call this during
    /// store construction.
    ///
    /// `key` is the buffer's identity in the keyed registry:
    /// `#[gpu(buffer = "key")]` fields resolve here. The FIRST registration
    /// of a key allocates its physical buffer; every later registration of
    /// the SAME key adopts that buffer (one shared allocation) provided the
    /// element type, stride, access, and mirror mode all match — otherwise
    /// this panics at construction time, loudly (a shared key whose
    /// declarers disagree on element identity is a config error, never a
    /// silent alias). `T` is the CPU-side column type this store dispatches
    /// writes through (a derive-generated per-field wrapper); `E` is the
    /// ELEMENT type the shared buffer is keyed on — the field's RAW type,
    /// so two different components' same-typed shared fields (whose wrapper
    /// types necessarily differ) still count as one compatible element.
    ///
    /// The declared row `capacity` must agree across declarers of a key
    /// (each fixed/cell-mirrored column is sized to the store's row-region
    /// count); a mismatch is a config error.
    pub fn register_gpu_buffer<T, E>(
        &self,
        capacity: u32,
        device: &wgpu::Device,
        key: BufferKey,
        mode: MirrorMode,
    ) where
        T: Pod + Send + Sync + HasTypeToken + 'static,
        E: Pod + HasTypeToken + 'static,
    {
        assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
            "gpu buffer key {key:?}: CPU dispatch type and element type disagree in size ({} vs {} bytes) — \
             the per-field wrapper must be byte-identical to the element it wraps",
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
        );
        let id = <T as HasTypeToken>::type_token().id();
        // Whole check-then-insert sequence under ONE write lock: two threads
        // racing to register the same never-before-seen key must not both
        // observe "not present" and both allocate a physical buffer.
        let mut owners = self.owners.write().expect("SceneGpuStore owners lock poisoned");
        if let Some(existing) = owners.get(&key) {
            Self::assert_share_compatible(
                existing,
                key,
                TypeId::of::<E>(),
                std::mem::size_of::<E>(),
                BufferAccess::ReadOnly,
                mode,
                SharedKindTag::Fixed,
            );
            assert!(
                capacity == existing.capacity,
                "buffer key {key:?} already registered at capacity {} rows; incoming declares {capacity} — \
                 shared fixed buffers must agree on capacity (the store's row-region count)",
                existing.capacity,
            );
            let SharedKind::Fixed(buf) = &existing.kind else { unreachable!("tag checked above") };
            let buf = Arc::clone(buf);
            drop(owners);
            self.gpu_buffers.write().expect("SceneGpuStore gpu_buffers lock poisoned").insert(id, buf);
            self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
            return;
        }
        let buf: Arc<dyn GpuBufferDispatch> = Arc::new(SceneBuffer::<T>::new(device, key.as_str(), capacity));
        self.registry
            .register_row::<E>(key, buf.buffer().clone(), BufferAccess::ReadOnly, mode)
            .unwrap_or_else(|e| panic!("failed to register buffer key {key:?}: {e}"));
        owners.insert(
            key,
            SharedOwner {
                element_type_id: TypeId::of::<E>(),
                element_size: std::mem::size_of::<E>(),
                capacity,
                access: BufferAccess::ReadOnly,
                mode,
                kind: SharedKind::Fixed(Arc::clone(&buf)),
            },
        );
        drop(owners);
        self.gpu_buffers.write().expect("SceneGpuStore gpu_buffers lock poisoned").insert(id, buf);
        self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
    }

    /// Clone the current `wgpu::Buffer` handle out of a dispatch object whose
    /// buffer lives behind a lock (growable/dirty-tracked) — cheap, wgpu
    /// handles are refcounted clones, not data copies. Used to snapshot the
    /// handle into the registry at first registration.
    fn current_handle<F>(f: &mut F) -> wgpu::Buffer
    where
        F: FnMut(&mut dyn FnMut(&wgpu::Buffer)),
    {
        let mut out: Option<wgpu::Buffer> = None;
        f(&mut |b| out = Some(b.clone()));
        out.expect("dispatch with_buffer must invoke its callback")
    }

    /// Panics unless `existing` (the already-registered owner of `key`) is
    /// compatible with a new declaration of element type `elem_tid`, size
    /// `elem_size`, access, mode, and registration path `incoming_tag`. A
    /// shared `BufferKey` may never span the fixed/growable/dirty-tracked
    /// paths, and any metadata disagreement is a config error, loudly
    /// panicked at store-construction time (never a silent alias).
    fn assert_share_compatible(
        existing: &SharedOwner,
        key: BufferKey,
        elem_tid: TypeId,
        elem_size: usize,
        access: BufferAccess,
        mode: MirrorMode,
        incoming_tag: SharedKindTag,
    ) {
        let existing_tag = match &existing.kind {
            SharedKind::Fixed(_) => SharedKindTag::Fixed,
            SharedKind::Growable(_) => SharedKindTag::Growable,
            SharedKind::DirtyTracked(_) => SharedKindTag::DirtyTracked,
        };
        assert_eq!(
            existing_tag, incoming_tag,
            "buffer key {key:?}: already registered through the {existing_tag:?} path, \
             incoming declaration is {incoming_tag:?} — the same key may never span the \
             fixed / growable / dirty-tracked registration paths",
        );
        assert_eq!(
            existing.element_type_id, elem_tid,
            "buffer key {key:?}: shared-key element type mismatch — first declarer used type id {:?}, \
             this declarer is {:?}. Two components may share a buffer key only when their fields agree \
             on the raw element type",
            existing.element_type_id, elem_tid,
        );
        assert_eq!(
            existing.element_size, elem_size,
            "buffer key {key:?}: shared-key element size mismatch ({} vs {elem_size} bytes)",
            existing.element_size,
        );
        assert_eq!(
            existing.access, access,
            "buffer key {key:?}: shared-key access mismatch ({:?} vs {access:?})",
            existing.access,
        );
        assert_eq!(
            existing.mode, mode,
            "buffer key {key:?}: shared-key mirror-mode mismatch ({:?} vs {mode:?}) — Once fields share \
             only with Once, DirtyTracked only with DirtyTracked",
            existing.mode,
        );
    }

    /// Resolve a shared `BufferKey` to the physical `wgpu::Buffer` backing it
    /// (the renderer handoff: build bind groups against the buffer a set of
    /// shared-key columns actually live in) plus its current element count.
    /// `None` if `key` was never registered. The handle reflects the LIVE
    /// buffer even after growth — it is cloned fresh from the dispatch object
    /// on every call, not cached at registration.
    ///
    /// The second value is an ELEMENT COUNT, not an allocation epoch — for a
    /// rebind-detecting handoff (owned buffer + epoch), use
    /// [`Self::resolve_buffer_handle`] instead.
    pub fn resolve_buffer(&self, key: BufferKey) -> Option<(wgpu::Buffer, u64)> {
        let owners = self.owners.read().expect("SceneGpuStore owners lock poisoned");
        let owner = owners.get(&key)?;
        let handle = match &owner.kind {
            SharedKind::Fixed(b) => b.buffer().clone(),
            SharedKind::Growable(b) => Self::current_handle(&mut |f| b.with_buffer(f)),
            SharedKind::DirtyTracked(b) => Self::current_handle(&mut |f| b.with_buffer(f)),
        };
        let element_count = handle.size() / owner.element_size as u64;
        Some((handle, element_count))
    }

    /// The Tier 1 renderer handoff: resolve `key` to its current
    /// [`BufferHandle`] (an owned `wgpu::Buffer` clone plus an allocation
    /// `epoch` that bumps exactly once per reallocation). `None` if `key` was
    /// never registered.
    ///
    /// A `Fixed` (cell-mirrored) buffer never reallocates after
    /// registration, so its epoch is always `0`. A `Growable`/`DirtyTracked`
    /// (World-mirrored) buffer's epoch is read live from its owning dispatch
    /// object — [`GrowableGpuBufferDispatch::epoch`] /
    /// [`DirtyTrackedGpuBufferDispatch::epoch`], which already increment
    /// correctly on every grow-triggered reallocation — so this is always
    /// current, never a stale snapshot frozen at first registration.
    ///
    /// This also syncs the live buffer/epoch into the keyed
    /// [`GpuBufferRegistry`] (`self.registry`) and returns its
    /// [`GpuBufferRegistry::resolve`] result, so the registry — the single
    /// namespace resource entries (`GeometryArena`, `TextureStore`) will
    /// eventually share — is the actual terminal source of truth for this
    /// handoff, not a write-only snapshot nobody reads.
    pub fn resolve_buffer_handle(&self, key: BufferKey) -> Option<BufferHandle> {
        let (buffer, epoch) = {
            let owners = self.owners.read().expect("SceneGpuStore owners lock poisoned");
            let owner = owners.get(&key)?;
            match &owner.kind {
                SharedKind::Fixed(b) => (b.buffer().clone(), 0u64),
                SharedKind::Growable(b) => (Self::current_handle(&mut |f| b.with_buffer(f)), b.epoch()),
                SharedKind::DirtyTracked(b) => {
                    (Self::current_handle(&mut |f| b.with_buffer(f)), b.epoch())
                }
            }
        };
        self.registry.sync(key, buffer, epoch);
        self.registry.resolve(key)
    }

    /// The `BufferKey` a `ComponentId` was registered under, if any — lets
    /// callers learn which shared (or auto-generated) buffer a column aliases,
    /// and lets [`Self::register_cell`] deduplicate warm-up across aliases.
    pub fn buffer_key_for(&self, id: ComponentId) -> Option<BufferKey> {
        self.column_keys.read().expect("SceneGpuStore column_keys lock poisoned").get(&id).copied()
    }

    /// Whether `T`'s `#[gpu]` fields are already registered on this store —
    /// checks only `T::gpu_columns()`'s FIRST field as a proxy for "is `T`
    /// registered at all", which is correct because registration
    /// ([`Self::register_gpu_buffer`]/[`Self::register_growable_gpu_buffer`]/
    /// [`Self::register_dirty_tracked_gpu_buffer`]) always registers every
    /// field of a type together, in one call — never partially. `true`
    /// (nothing to register) for a `T` with zero `#[gpu]` fields.
    ///
    /// This is what makes "auto-registration on first use"
    /// (`gpu::world_mirror`'s per-type dispatch function) a single cheap
    /// lookup rather than a scan: one `RwLock::read` + `HashMap::get`, the
    /// same cost `buffer_key_for` alone already had.
    pub fn is_registered<T: GpuColumnSet>(&self) -> bool {
        // `map_or(true, ..)`, not `is_none_or` (stabilized 1.82) -- this
        // crate's declared MSRV is 1.81 (workspace Cargo.toml). Silences
        // clippy::unnecessary_map_or deliberately, not by oversight.
        #[allow(clippy::unnecessary_map_or)]
        T::gpu_columns()
            .first()
            .map_or(true, |col| self.buffer_key_for(col.field_token.id()).is_some())
    }

    /// Growable counterpart to [`Self::register_gpu_buffer`] — for
    /// World-mirrored `#[gpu]` columns, whose required capacity isn't known
    /// ahead of time (`World`'s entity count has no ceiling). `initial_capacity`
    /// only needs to be cheap to allocate, not sized for the eventual world —
    /// the buffer grows (doubling, GPU-to-GPU copy) transparently on writes
    /// past its current capacity, via [`Self::write_row_bytes_growing`].
    ///
    /// `max_capacity: None` (the recommendation for World-mirrored columns,
    /// and what `#[derive(SceneStore)]`'s generated `register_gpu_columns_growable`
    /// always passes) means growth can never fail — exactly the property
    /// that makes this safe to call from `World::insert`'s automatic
    /// dispatch path with no `Result` to propagate through `insert` itself.
    /// Pass `Some(max)` only for a deliberate, caller-managed VRAM ceiling
    /// (not recommended for World-mirrored columns specifically, since
    /// hitting it turns an ordinary `world.insert()` call into one that can
    /// fail with no way to report that failure back to the caller — see
    /// [`Self::write_row_bytes_growing`]'s doc).
    ///
    /// Sharing (`#[gpu(buffer = "key")]`, `World`-mirror side): the FIRST
    /// registration of `key` in this mode allocates; later same-key
    /// declarations (including from `#[derive(SceneStore)]`'s generated
    /// `register_gpu_columns_growable`) adopt the existing dispatch Arc —
    /// same element/access/mode must hold, else panic. The same key may
    /// never span fixed AND growable AND dirty-tracked paths.
    ///
    /// A given `ComponentId` must be registered through exactly one of
    /// [`Self::register_gpu_buffer`]/`Self::register_growable_gpu_buffer` —
    /// registering the same id both ways leaves the growable one shadowed by
    /// [`Self::write_row_bytes`]'s lookup order and is almost certainly not
    /// what was intended; this is not currently asserted against (no test
    /// covers the double-registration case), so avoid it by construction.
    pub fn register_growable_gpu_buffer<T, E>(
        &self,
        initial_capacity: u32,
        max_capacity: Option<u32>,
        device: &Arc<wgpu::Device>,
        key: BufferKey,
        mode: MirrorMode,
    ) where
        T: Pod + Send + Sync + HasTypeToken + 'static,
        E: Pod + HasTypeToken + 'static,
    {
        assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
            "gpu buffer key {key:?}: CPU dispatch type and element type disagree in size ({} vs {} bytes) — \
             the per-field wrapper must be byte-identical to the element it wraps",
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
        );
        let id = <T as HasTypeToken>::type_token().id();
        let mut owners = self.owners.write().expect("SceneGpuStore owners lock poisoned");
        if let Some(existing) = owners.get(&key) {
            Self::assert_share_compatible(
                existing,
                key,
                TypeId::of::<E>(),
                std::mem::size_of::<E>(),
                BufferAccess::ReadOnly,
                mode,
                SharedKindTag::Growable,
            );
            let SharedKind::Growable(buf) = &existing.kind else { unreachable!("tag checked above") };
            let buf = Arc::clone(buf);
            drop(owners);
            self.growable_gpu_buffers.write().expect("SceneGpuStore growable_gpu_buffers lock poisoned").insert(id, buf);
            self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
            return;
        }
        let buf: Arc<dyn GrowableGpuBufferDispatch> = Arc::new(GrowableSceneBuffer::<T>::new(
            Arc::clone(device),
            key.as_str(),
            initial_capacity,
            max_capacity,
        ));
        let handle = Self::current_handle(&mut |f| buf.with_buffer(f));
        self.registry
            .register_row::<E>(key, handle, BufferAccess::ReadOnly, mode)
            .unwrap_or_else(|e| panic!("failed to register buffer key {key:?}: {e}"));
        owners.insert(
            key,
            SharedOwner {
                element_type_id: TypeId::of::<E>(),
                element_size: std::mem::size_of::<E>(),
                capacity: initial_capacity,
                access: BufferAccess::ReadOnly,
                mode,
                kind: SharedKind::Growable(Arc::clone(&buf)),
            },
        );
        drop(owners);
        self.growable_gpu_buffers.write().expect("SceneGpuStore growable_gpu_buffers lock poisoned").insert(id, buf);
        self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
    }

    /// Mark a column's row as dirty for the next GPU sync.
    /// Called by the derive macro's generated `write_gpu()`.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been registered.  Silently ignores unknown
    /// component ids (the column was registered via `register_gpu_buffer`,
    /// and every `#[gpu]` field's type gets one).
    pub fn mark_column_dirty(&self, id: CellId, component_id: ComponentId, row: u32) {
        let state = self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered");
        if let Some(mask) = state.dirty_column(component_id) {
            mask.mark(row);
        }
    }

    /// Generic write path for any GPU-mirrored component type.
    /// Writes data to the CPU columns via the trait's `write_gpu`, marks dirty
    /// for GPU sync, and stamps the generation.
    ///
    /// Returns `false` for stale/invalid handles or pinned rows.
    pub fn write_gpu<T: GpuColumnSet>(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &T,
        phase: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(self.phase, Phase::Write, "mutation outside the write window");
        let state = self.cells[id.0 as usize].as_ref().expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false;
        }
        // Delegate to the trait's write_gpu — writes CPU columns + marks dirty
        T::write_gpu(self, id, cell, handle, data, phase);
        // Stamp generation (shadow-gated, idempotent)
        self.write_generation(state, handle.index(), handle.generation());
        true
    }

    /// Test instrument: how many generation-buffer writes this store has
    /// issued across every cell (asserting upload minimality, §4).
    ///
    /// Counts only `write_generation` calls (per-slot, shadow-gated). The
    /// recycled-region tail scrub in `register_cell` (§11 D2-tail
    /// carry-forward) bypasses this counter deliberately: it is a
    /// region-lifecycle bulk write (once per promotion, not per slot), not
    /// the per-slot stamp this instrument exists to bound.
    #[doc(hidden)]
    pub fn generation_write_count(&self) -> u64 {
        self.gen_writes.load(Ordering::Relaxed)
    }

    /// Shadow-gated generation upload: writes VRAM (and the shadow) only when
    /// `generation` differs from the last value uploaded for `local_slot`,
    /// translated to the cell's global slot via `state.slot_base`.
    ///
    /// Deliberately NOT the slot-mirror dirty trigger: this gate is
    /// SLOT-scoped, but mirror staleness is ROW-scoped — a retired slot
    /// recycled into a different row arrives with its generation already
    /// shadowed (the retire stamped it), so the gate stays silent while the
    /// new row's mirror entry is stale (fail-open C6, Task 4 review). The
    /// mirror trigger is `sync_all`'s self-healing boundary scan.
    fn write_generation(&self, state: &CellGpuState, local_slot: u32, generation: u32) {
        assert!(
            local_slot < state.slot_capacity,
            "slot {local_slot} beyond region capacity {} — write must never land in a neighbor's region",
            state.slot_capacity
        );
        if state.gen_shadow[local_slot as usize].load(Ordering::Relaxed) == generation {
            return;
        }
        self.generations.write(&self.queue, state.slot_base + local_slot, generation);
        state.gen_shadow[local_slot as usize].store(generation, Ordering::Relaxed);
        self.gen_writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Ensure a dirty mask exists for the given component id in this cell.
    /// Used by write paths to lazily create the mask when a column is first
    /// written (instead of requiring it to exist at registration time).
    fn ensure_dirty_column(state: &mut CellGpuState, id: ComponentId, row_capacity: u32) -> &mut DirtyMask {
        let idx = id.0 as usize;
        if state.dirty_columns.len() <= idx {
            state.dirty_columns.resize_with(idx + 1, || None);
        }
        state.dirty_columns[idx].get_or_insert_with(|| DirtyMask::new(row_capacity))
    }

    /// §4.1 promotion primitive (α: registration; β reuses it for promotion):
    /// allocates row+slot regions, bulk-rebuilds the generation region from
    /// the registry, seeds the gen-shadow, marks all occupied rows dirty in
    /// the transform mask. The slot mirror needs no warm-up — the first
    /// `sync_all` boundary scan uploads every occupied row's slot entry.
    ///
    /// **Panic vs. decline:** the two failure modes below are deliberately
    /// different in kind. Pool exhaustion (`RowsExhausted`/`SlotsExhausted`)
    /// is an ordinary runtime condition — a graceful `Err` the executor
    /// declines on (the cell stays in its current domain, §8) — because a
    /// full grid is expected, not a bug. The `assert!`s below it (cell rows/
    /// slots exceeding the class's region capacity) are invariant violations:
    /// a cell that does not fit the size class it was registered under is a
    /// caller/config bug, not a transient condition, so those panic by
    /// design rather than returning `Err`.
    pub fn register_cell(&mut self, cell: &CellStorage, class: usize) -> Result<CellId, RegionError> {
        if self.row_pools[class].free_count() == 0 {
            return Err(RegionError::RowsExhausted);
        }
        if self.slot_pools[class].free_count() == 0 {
            return Err(RegionError::SlotsExhausted);
        }
        let row_base = self.row_pools[class].alloc().expect("checked free_count above");
        let slot_base = self.slot_pools[class].alloc().expect("checked free_count above");
        let row_capacity = self.row_pools[class].region_size();
        let slot_capacity = self.slot_pools[class].region_size();

        assert!(
            cell.rows_in_use() <= row_capacity,
            "cell occupies {} rows but class capacity is {row_capacity}",
            cell.rows_in_use()
        );
        let gens = cell.registry().generations();
        assert!(gens.len() as u32 <= slot_capacity, "cell has more slots ({}) than its region capacity {slot_capacity}", gens.len());
        self.generations.rebuild_region(&self.queue, slot_base, gens);

        // Tail scrub (§11 D2-tail carry-forward, binding for β): `gens`
        // covers only the registry's allocated-slot prefix, never the full
        // region. On a FRESH region the tail is already zero (never
        // written), but on a RECYCLED region (post-eviction, `unregister_cell`)
        // the tail still holds the prior tenant's VRAM generations — an
        // allocated-but-never-written slot in the new tenant would fail
        // fail-open, validating against a stranger's stale generation. Zero
        // the tail unconditionally; the write is a no-op in cost terms only
        // when `occupied == slot_capacity` (headroom-free region). Cold
        // path: once per promotion, not per slot — deliberately NOT counted
        // by `gen_writes` (see that counter's doc).
        let occupied = gens.len() as u32;
        if occupied < slot_capacity {
            let tail_zeros = vec![0u32; (slot_capacity - occupied) as usize];
            self.queue.write_buffer(
                self.generations.buffer(),
                (slot_base + occupied) as u64 * 4,
                super::as_bytes(&tail_zeros),
            );
        }

        // gen_shadow: 0 by construction for every slot (including the tail
        // computed above — a recycled region's shadow must forget the prior
        // tenant's shadow state too, not just its VRAM bytes), then seeded
        // for the occupied prefix from the registry.
        let gen_shadow: Vec<AtomicU32> = (0..slot_capacity).map(|_| AtomicU32::new(0)).collect();
        for (slot, &generation) in gens.iter().enumerate() {
            gen_shadow[slot].store(generation, Ordering::Relaxed);
        }

        // Warm up dirty masks for every GPU buffer column THIS CELL HAS.
        // For built-in buffers ([f32; 16] and InstanceInfo), this ensures
        // the first sync_all uploads every occupied row.  Any additional
        // GPU-mirrored columns registered via register_gpu_buffer are also
        // warmed up here when the cell actually carries them.
        // Dense table sized to the highest registered GPU-column id; holes
        // (unregistered ids, and index 0 which `ComponentId` reserves) stay
        // `None`. Built once here so every subsequent write is a Vec index.
        //
        // Shared-key aliasing (`#[gpu(buffer = "key")]`, cell-mirrored
        // path): two `ComponentId`s that declared the same `BufferKey` alias
        // ONE physical buffer whose rows are a shared pool. Per CELL, each
        // key's pool slice is consumed by AT MOST ONE column — so a cell
        // that carries two same-key columns would have both claiming the
        // identical row region of the same buffer (a contradiction: either
        // column's first sync would overwrite the other's rows). That is a
        // config error, and it panics LOUDLY here at registration rather
        // than corrupting the pool on the first sync. Otherwise, exactly the
        // one column this cell carries for a key is warmed (an alias the
        // cell doesn't carry gets no mask — its sync_all contribution is
        // nothing).
        let gpu_buffers = self.gpu_buffers.read().expect("SceneGpuStore gpu_buffers lock poisoned");
        let widest = gpu_buffers.keys().map(|id| id.0 as usize).max().unwrap_or(0);
        let mut dirty_columns: Vec<Option<DirtyMask>> = Vec::new();
        dirty_columns.resize_with(widest + 1, || None);
        let mut warmed: HashSet<BufferKey> = HashSet::new();
        for id in gpu_buffers.keys() {
            if cell.column_raw_bytes(*id).is_none() {
                continue;
            }
            let key = self
                .buffer_key_for(*id)
                .expect("registered GPU buffer always has a column key");
            assert!(
                !warmed.contains(&key),
                "cell carries two columns sharing buffer key {key:?} ({:?} and a prior column) — \
                 a shared pool's rows are one column's per cell; two same-key columns in one cell \
                 would overwrite each other's rows on the first sync. Split them across cells, or \
                 use distinct keys",
                id,
            );
            warmed.insert(key);
            let mask = DirtyMask::new(row_capacity);
            mask.mark_range(cell.rows_in_use());
            dirty_columns[id.0 as usize] = Some(mask);
        }
        // No slot-mirror warm-up: `sync_all`'s self-healing boundary scan is
        // the SOLE dirty_slots marker and scratch/shadow writer. The shadow
        // starts all-MAX (= never uploaded; real local slots are always
        // < slot_capacity < u32::MAX), so the first boundary marks and
        // uploads every occupied row on its own. Keeping mark/fill paired in
        // exactly one place removes the "mark without a scratch fill uploads
        // stale bytes" footgun.
        let dirty_slots = DirtyMask::new(row_capacity);

        self.cells.push(Some(CellGpuState {
            class,
            row_base,
            slot_base,
            slot_capacity,
            dirty_columns,
            dirty_slots,
            slot_scratch: vec![0u32; row_capacity as usize],
            slot_shadow: vec![u32::MAX; row_capacity as usize],
            pending: VecDeque::new(),
            gen_shadow,
        }));
        Ok(CellId(self.cells.len() as u32 - 1))
    }

    /// Test 14 (C0 companion gate): build a fresh multi-cell store on a fresh
    /// device purely from every cell's CPU-authoritative columns (no GPU-only
    /// state exists to lose, design §3 "derived data is not stored"). Returns
    /// the rebuilt store paired with each input cell's freshly assigned
    /// `CellId`, in the same order as `cells`.
    ///
    /// Precondition per cell: no rows may be pinned (all pending retires
    /// drained via `retire_all`) — recovery of in-flight retirement across a
    /// device loss is M4 scope; rebuilding while a pin is outstanding would
    /// strand it permanently (the pin bit lives only in `CellStorage`, and
    /// this fresh store has no queued `PendingRetire` to eventually unpin
    /// it). Verbatim message carried over from M2a's `GpuStore::rebuild_from`.
    ///
    /// For each cell: `register_cell` already rebuilds that cell's generation
    /// region, seeds the gen shadow, and marks every occupied row dirty in
    /// the transform mask (§4.1 warm-up) — but `write_rows` below is an
    /// UNCONDITIONAL bulk write, so those warm-up marks are cleared right
    /// after to avoid double-uploading the same bytes at the first boundary.
    /// The slot mirror has no warm-up marker of its own (its sole dirty
    /// trigger is `sync_all`'s self-healing boundary scan, which hasn't run
    /// yet for a freshly rebuilt store), so it is bulk-filled here too —
    /// scratch and shadow alike — matching exactly what that boundary scan
    /// would otherwise produce on its first pass.
    pub fn rebuild(
        ctx: &EngineGpuContext,
        cfg: SceneGpuConfig,
        cells: &[(usize, &CellStorage)],
    ) -> (Self, Vec<CellId>) {
        let mut store = Self::new(ctx, cfg);
        let mut ids = Vec::with_capacity(cells.len());
        for &(class, cell) in cells {
            debug_assert!(
                (0..cell.rows_in_use()).all(|r| !cell.is_row_pinned(r)),
                "rebuild_from with in-flight retirement: drain retire() before device-loss rebuild — pins would be permanently stranded"
            );
            let id = store.register_cell(cell, class).expect("rebuild: cell must fit its class region");
            let rows = cell.rows_in_use();
            let row_base = store.cells[id.0 as usize].as_ref().expect("cell unregistered").row_base;
            let slot_base = store.cells[id.0 as usize].as_ref().expect("cell unregistered").slot_base;

            // Bulk-write every registered GPU buffer via the generic
            // type-erased path.  Replaces the old hardcoded downcasts to
            // `SceneBuffer<[f32; 16]>` / `SceneBuffer<InstanceInfo>`.
            for (id, buffer) in store.gpu_buffers.read().expect("SceneGpuStore gpu_buffers lock poisoned").iter() {
                if let Some(col_bytes) = cell.column_raw_bytes(*id) {
                    let row_bytes = buffer.element_size() * rows as usize;
                    if row_bytes > 0 && col_bytes.len() >= row_bytes {
                        buffer.write_rows_raw(&store.queue, &col_bytes[..row_bytes], row_base);
                    }
                }
            }

            // Clear warm-up marks that were just satisfied by the bulk writes.
            let state = store.cells[id.0 as usize].as_mut().expect("cell unregistered");
            for mask in state.dirty_columns.iter_mut().flatten() {
                mask.clear_all();
            }

            let col0 = cell.slot_column();
            {
                let state = store.cells[id.0 as usize].as_mut().expect("cell unregistered");
                for row in 0..rows {
                    let local_slot = col0[row as usize];
                    state.slot_scratch[row as usize] = slot_base + local_slot;
                    state.slot_shadow[row as usize] = local_slot;
                }
            }
            store.slot_mirror.write_rows(
                &store.queue,
                &store.cells[id.0 as usize].as_ref().expect("cell unregistered").slot_scratch[..rows as usize],
                row_base,
            );

            ids.push(id);
        }
        (store, ids)
    }

    /// §4.1 eviction (M2b-β, closes the D2 pending-retire disposition and the
    /// region-recycle side of the two §11 carry-forwards): demote a resident
    /// cell out of its region.
    ///
    /// 1. Every retire still queued in `cell`'s pending-retire queue is
    ///    committed **CPU-side immediately** (`cell.commit_retire`: unpin the
    ///    row, bump the registry generation, pool the slot) — **zero VRAM
    ///    writes**. The design originally deferred this commit until the
    ///    region's pin serial completed, to keep a late write from landing in
    ///    a freed (possibly reallocated) region; since this commit never
    ///    touches VRAM there is nothing for a late write to corrupt, so the
    ///    audit remediation (see the M2b-β holistic-audit doc) moves the
    ///    commit earlier, to immediately at eviction. `last_serial` still
    ///    gates the region's byte-range reuse (step 2) — only the retire
    ///    *commit* moved earlier.
    /// 2. Both the row and slot regions are returned to their class's pools
    ///    via `free_pinned(last_serial)` — reusable only once `last_serial`
    ///    completes (`retire_all` drains both pool families every boundary).
    /// 3. The `CellGpuState` (gen-shadow slice, dirty words, slot
    ///    scratch/shadow) is dropped outright — re-created at the region's
    ///    next promotion via `register_cell`'s rebuild + tail scrub.
    ///    CPU-side cell data persists (host memory is authoritative); the
    ///    `cells` slot for `id` becomes `None` and stays `None` for the life
    ///    of the store — `CellId`s are NOT recycled, unlike the row/slot
    ///    regions they used to index.
    ///
    /// Panics if `id` was already unregistered (double-eviction is a caller
    /// bug, not a runtime condition to tolerate).
    pub fn unregister_cell(&mut self, id: CellId, cell: &mut CellStorage, last_serial: u64) {
        let idx = id.0 as usize;
        let state = self.cells[idx].take().expect("cell already unregistered");
        debug_assert!(
            state.pending.back().map_or(true, |q| q.serial <= last_serial),
            "unregister_cell: last_serial must dominate every queued pending serial — the region pin IS the C6 protection for in-flight reads"
        );
        for QueuedRetire { pending, .. } in state.pending {
            cell.commit_retire(pending);
        }
        self.row_pools[state.class].free_pinned(state.row_base, last_serial);
        self.slot_pools[state.class].free_pinned(state.slot_base, last_serial);
        // `state` (gen-shadow, dirty masks, slot scratch/shadow) drops here.
    }

    /// The single mutation path for the GPU-mirrored transform column (§4):
    /// writes the core column AND sets the row's dirty bit in one operation.
    /// False for stale/invalid handles.
    ///
    /// This is the hardcoded equivalent of [`Self::write_gpu`] — `[f32; 16]`
    /// is a bare type, not a derive'd struct, but uses the same generic
    /// `dirty_columns` machinery internally.
    ///
    /// Also stamps the handle's generation into the slot-indexed generation
    /// buffer (adaptation to §5/§7): the design's "written by retirement"
    /// trigger only ever bumps a slot's entry on retire, so a slot's *first*
    /// generation (assigned at `alloc`, which does not pass through
    /// `SceneGpuStore` at all) would otherwise never reach VRAM until that
    /// slot is later retired. The stamp is shadow-gated (`gen_shadow`): a
    /// generation reaches VRAM on the first write after alloc and on
    /// retirement — NOT per `write_transform` call — so repeat writes to a
    /// live handle (the §4 hot path) issue zero generation-buffer traffic
    /// while the buffer still mirrors `HandleRegistry::generations()` for
    /// every allocated slot (C6).
    pub fn write_transform(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        m: &[f32; 16],
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(self.phase, Phase::Write, "mutation outside the write window");
        let state = self.cells[id.0 as usize].as_ref().expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false; // in-flight retirement: logically deleted (§8) — no further mutation
        }
        let col = cell
            .column_for_mut::<[f32; 16]>()
            .expect("cell has no [f32; 16] transform column");
        col[row as usize] = *m;
        let comp_id = component_id::<[f32; 16]>();
        state
            .dirty_column(comp_id)
            .expect("transform dirty mask missing — register_gpu_buffer not called for [f32; 16]")
            .mark(row);
        self.write_generation(state, handle.index(), handle.generation());
        true
    }

    /// The GPU-mirrored [`InstanceInfo`] column companion to
    /// [`Self::write_transform`] (M3-α T4, cull's token→mesh link): writes
    /// the core column AND sets the row's dirty bit in one operation. False
    /// for stale/invalid handles. Requires the cell to carry an
    /// `InstanceInfo` column (e.g. built via `SpatialCell::with_transform`);
    /// panics otherwise, mirroring `write_transform`'s own unconditional
    /// `.expect()` on its column.
    ///
    /// This is the hardcoded equivalent of [`Self::write_gpu`] — `InstanceInfo`
    /// is a bare type, not a derive'd struct, but uses the same generic
    /// `dirty_columns` machinery internally.
    ///
    /// Body is IDENTICAL to `write_transform`'s minus the generation stamp:
    /// **the generation stamp stays transform-only, by design — one
    /// stamping path.** `write_transform`'s doc explains why the stamp lives
    /// there (a slot's first generation, assigned at `alloc`, must reach VRAM
    /// on *some* write path, and retirement is the other trigger); splitting
    /// it across two mirrored-column writers would mean either double-
    /// stamping (harmless but redundant — the stamp is shadow-gated and
    /// idempotent) or deciding which column "owns" a given slot's first
    /// stamp, a distinction with no purpose. Keeping the stamp solely on
    /// `write_transform` means every handle's first VRAM generation write is
    /// guaranteed by the ONE column every registered cell is required to
    /// carry (transforms, C5-mandatory) rather than the optional one.
    pub fn write_instance_info(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        info: InstanceInfo,
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(self.phase, Phase::Write, "mutation outside the write window");
        let state = self.cells[id.0 as usize].as_ref().expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false; // in-flight retirement: logically deleted (§8) — no further mutation
        }
        let col = cell
            .column_for_mut::<InstanceInfo>()
            .expect("cell has no InstanceInfo column — register via SpatialCell::with_transform");
        col[row as usize] = info;
        let comp_id = component_id::<InstanceInfo>();
        state
            .dirty_column(comp_id)
            .expect("InstanceInfo dirty mask missing — register_gpu_buffer not called for InstanceInfo")
            .mark(row);
        true
    }

    /// §5 flow step 1: liveness-dead + pinned + enqueued against `serial`.
    /// Registry and GPU buffers unchanged until the serial completes.
    pub fn free_deferred(
        &mut self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        serial: u64,
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(self.phase, Phase::Write, "free_deferred outside the write window");
        let Some(pending) = cell.mark_pending_retire(handle) else {
            return false;
        };
        let state = self.cells[id.0 as usize].as_mut().expect("cell unregistered");
        debug_assert!(
            state.pending.back().map_or(true, |q| q.serial <= serial),
            "free_deferred serials must be nondecreasing per cell — the retire \
             drain's FIFO early-break would silently stall retirement behind an \
             out-of-order serial"
        );
        state.pending.push_back(QueuedRetire { pending, serial });
        true
    }

    /// §5 flow step 3, frame boundary, runs FIRST: for every cell, drain that
    /// cell's queue (FIFO, early-break on the first incomplete serial)
    /// against `tracker.completed()`; for every drained entry write the new
    /// generation to VRAM, then commit in the registry — the gen bump
    /// reaches the GPU before the slot can be re-allocated (C6). Returns the
    /// total number of slots retired across every cell.
    ///
    /// Also drains both pool families (M2b-β §4.1 eviction): every class's
    /// `row_pools`/`slot_pools` recycles any region pinned by
    /// `unregister_cell` whose serial has now completed, at exactly this
    /// frame boundary — the same watermark this cell-level drain uses.
    pub(crate) fn retire_all(&mut self, cells: &mut [CellSlot<'_>]) -> u32 {
        debug_assert_eq!(self.phase, Phase::Write, "retire_all must open the frame boundary");
        self.phase = Phase::Retired;
        let done = self.tracker.completed();
        let mut drained = 0u32;
        for slot in cells.iter_mut() {
            let idx = slot.id.0 as usize;
            loop {
                let state = self.cells[idx].as_ref().expect("cell unregistered");
                let ready = matches!(state.pending.front(), Some(front) if front.serial <= done);
                if !ready {
                    break; // FIFO serials: everything behind is also incomplete
                }
                let QueuedRetire { pending, .. } =
                    self.cells[idx].as_mut().expect("cell unregistered").pending.pop_front().unwrap();
                // Retirement always bumps the generation, so the shadow-gated
                // write always lands in VRAM (and updates the shadow) before
                // the registry commit can recycle the slot (C6).
                self.write_generation(self.cells[idx].as_ref().expect("cell unregistered"), pending.slot, pending.next_gen);
                slot.cell.commit_retire(pending);
                drained += 1;
            }
        }
        for pool in self.row_pools.iter_mut().chain(self.slot_pools.iter_mut()) {
            pool.drain_completed(done);
        }
        drained
    }

    /// Frame-boundary compaction (§4): every moved row's destination is
    /// marked dirty in the transform mask (AND the instance-info mask,
    /// M3-α T4 — a moved row carries both mirrored columns to their new
    /// position) so the next sync re-uploads it. The slot mirror is NOT
    /// marked here — `sync_all`'s boundary scan detects moved slots on its
    /// own.
    pub(crate) fn compact_all(&mut self, cells: &mut [CellSlot<'_>]) {
        self.compact_all_gated(cells, |_| true);
    }

    /// Lease-gated variant of [`Self::compact_all`] (§9.2.1, contract #32).
    /// Identical sweep, except `ready` is consulted once per cell (called
    /// with that cell's `CellId`) and a cell it reports NOT ready for is
    /// excluded from THIS boundary's compaction entirely — its holes
    /// persist to the next boundary, the same "deferred" shape
    /// `CellStorage::compact_report` already uses for a pinned tail
    /// (`cell.rs`'s doc). Callers build `ready` from
    /// `gpu::HarvestPipeline::compaction_ready` for each cell's
    /// `(LeaseMask, outstanding leases)` pair — this store has no notion of
    /// leases itself (see that method's ownership-gap doc). `compact_all`
    /// is exactly this with an always-ready gate, so the two can never
    /// silently diverge in behavior.
    pub(crate) fn compact_all_gated(&mut self, cells: &mut [CellSlot<'_>], mut ready: impl FnMut(CellId) -> bool) {
        debug_assert_eq!(self.phase, Phase::Retired, "compact_all must follow retire_all");
        self.phase = Phase::Compacted;
        for slot in cells.iter_mut() {
            if !ready(slot.id) {
                continue;
            }
            let state = self.cells[slot.id.0 as usize].as_ref().expect("cell unregistered");
            slot.cell.compact_report(|_from, to| {
                // Mark every registered GPU buffer column's dirty mask for
                // the moved row, so the next sync re-uploads everything.
                // The slot mirror is NOT marked here — `sync_all`'s
                // self-healing boundary scan detects moved slots on its own.
                for mask in state.dirty_columns.iter().flatten() {
                    mask.mark(to);
                }
            });
        }
    }

    /// Frame-boundary upload (§4): coalesced dirty-row write of each cell's
    /// GPU-mirrored columns into their disjoint slices of the shared SSBOs,
    /// then the slot mirror via the self-healing boundary scan — every
    /// occupied row whose shadow disagrees with the slot column is marked,
    /// staged, and uploaded, regardless of HOW the slot got there. Closes
    /// the boundary (next phase is the write window). Post-condition: mirror
    /// entries `[row_base, row_base + rows_in_use)` equal
    /// `slot_base + slot_column()[row]` exactly.
    pub(crate) fn sync_all(&mut self, cells: &mut [CellSlot<'_>]) -> SyncStats {
        debug_assert_eq!(self.phase, Phase::Compacted, "sync_all must follow compact_all");
        self.phase = Phase::Write;
        let mut total = SyncStats { ranges: 0, bytes: 0 };
        for slot in cells.iter_mut() {
            let rows = slot.cell.rows_in_use() as usize;
            let cell_ref = &*slot.cell; // reborrow as shared for column reads

            // Sync every dirty GPU-mirrored column. One read-lock acquire
            // per cell (not per column) -- the map itself only ever changes
            // at registration time, well outside this per-frame hot path.
            let state = self.cells[slot.id.0 as usize].as_ref().expect("cell unregistered");
            let gpu_buffers = self.gpu_buffers.read().expect("SceneGpuStore gpu_buffers lock poisoned");
            for (id, dirty_mask) in state.dirty_columns_iter() {
                if let Some(buffer) = gpu_buffers.get(&id) {
                    if let Some(col_bytes) = cell_ref.column_raw_bytes(id) {
                        let row_count = rows.min(col_bytes.len() / buffer.element_size());
                        if row_count > 0 {
                            let stats = buffer.sync_region(
                                &self.queue,
                                &col_bytes[..row_count * buffer.element_size()],
                                state.row_base,
                                dirty_mask,
                            );
                            total.ranges += stats.ranges;
                            total.bytes += stats.bytes;
                        }
                    }
                }
            }

            // Self-healing boundary scan — the ONLY slot-mirror dirty
            // trigger (Task 4 re-review): compare every occupied row's
            // shadow against the authoritative slot column and mark exactly
            // the mismatches, whatever moved the slot there (write after
            // alloc, compaction swap, or an alloc re-occupying a vacated row
            // that is never written — the ghost-duplicate case no per-event
            // trigger caught). O(rows) u32 compares per cell per boundary;
            // uploads only actual mismatches.
            let col0 = slot.cell.slot_column();
            let state = self.cells[slot.id.0 as usize].as_mut().expect("cell unregistered");
            for row in 0..rows as u32 {
                let expect = col0[row as usize];
                if state.slot_shadow[row as usize] != expect {
                    // Audit A, G3: mirror `write_generation`'s release-assert
                    // (scene_store.rs, `write_generation`) so a slot value
                    // that is allocated but never written (tombstone-headroom
                    // exhaustion after extreme generation churn — `expect`
                    // beyond `slot_capacity`) fails loud here instead of
                    // silently mirroring this row onto a NEIGHBOR cell's
                    // generation region (fail-open C6). Release, not
                    // debug-only: this is a corruption guard, not a debug
                    // convenience.
                    assert!(
                        expect < state.slot_capacity,
                        "slot {expect} beyond region capacity {} — mirror must never point into a neighbor's region",
                        state.slot_capacity
                    );
                    state.dirty_slots.mark(row);
                    state.slot_scratch[row as usize] = state.slot_base + expect;
                    state.slot_shadow[row as usize] = expect;
                }
            }
            let state = self.cells[slot.id.0 as usize].as_ref().expect("cell unregistered");
            let stats =
                self.slot_mirror.sync_region(&self.queue, &state.slot_scratch[..rows], state.row_base, &state.dirty_slots);
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        total
    }

    pub fn row_region_base(&self, id: CellId) -> u32 {
        self.cells[id.0 as usize].as_ref().expect("cell unregistered").row_base
    }

    /// Returns an OWNED clone (`wgpu::Buffer` is a cheap, refcounted handle
    /// — see `BufferHandle`'s doc for the same convention) rather than a
    /// borrow: `gpu_buffers` lives behind a `RwLock` now (see the field's
    /// doc for why), so a `&wgpu::Buffer` tied to a temporary read-guard
    /// can't outlive this call.
    pub fn transform_buffer(&self) -> wgpu::Buffer {
        let id = component_id::<[f32; 16]>();
        self.gpu_buffers
            .read()
            .expect("SceneGpuStore gpu_buffers lock poisoned")
            .get(&id)
            .expect("transform buffer not registered")
            .buffer()
            .clone()
    }

    /// Generic counterpart to [`Self::transform_buffer`]/
    /// [`Self::instance_info_buffer`] for a caller-registered type: the same
    /// `ComponentId` lookup those two do by hand for their hardcoded types,
    /// available for any `T` previously passed to
    /// [`Self::register_gpu_buffer`]. `None` if `T` was never registered —
    /// callers that know their type is a required built-in should keep using
    /// the specific accessor (whose `.expect()` gives a clearer panic
    /// message); this is for generic/optional columns.
    pub fn buffer_for<T: HasTypeToken + 'static>(&self) -> Option<wgpu::Buffer> {
        let id = <T as HasTypeToken>::type_token().id();
        self.gpu_buffers
            .read()
            .expect("SceneGpuStore gpu_buffers lock poisoned")
            .get(&id)
            .map(|b| b.buffer().clone())
    }

    /// `ComponentId`-keyed counterpart to [`Self::buffer_for`], for callers
    /// that only have the id (e.g. from `GpuColumnDesc::field_token.id()`)
    /// and not the concrete registered type — notably, `#[derive(SceneStore)]`
    /// `#[gpu]` fields, whose per-field wrapper type is deliberately
    /// unnameable from outside the macro's own generated code (see
    /// `pulsar_scenedb_derive`'s `FieldInfo::gpu_wrapper` doc). Reading a
    /// derive-mirrored field's buffer back — in a test, in editor tooling,
    /// or in `Self::write_row_bytes`'s own caller, [`crate::gpu::world_mirror`]
    /// — goes through this, not `buffer_for::<Wrapper>()`.
    pub fn buffer_for_id(&self, id: ComponentId) -> Option<wgpu::Buffer> {
        self.gpu_buffers
            .read()
            .expect("SceneGpuStore gpu_buffers lock poisoned")
            .get(&id)
            .map(|b| b.buffer().clone())
    }

    /// Raw, `ComponentId`-keyed counterpart to [`Self::register_gpu_buffer`]
    /// + [`GpuBufferDispatch::write_rows_raw`], for callers that only have a
    /// `ComponentId` at hand (not the concrete `T`) — e.g. a generic helper
    /// walking `GpuColumnSet::gpu_columns()`'s per-field metadata, which
    /// carries each field's `TypeToken`/`ComponentId` but not its Rust type.
    ///
    /// Bypasses dirty tracking, `CellStorage`, and `Handle` entirely, same as
    /// `write_rows_raw` itself — an unconditional `queue.write_buffer` at
    /// `row`. This is the primitive [`crate::gpu::world_mirror`] builds on to
    /// mirror `World`-owned (`Entity`-indexed, cell-free) component fields:
    /// `row` there is `Entity::index()`, which needs no `CellId`/region
    /// concept at all.
    ///
    /// Returns `false` (a no-op, not a panic) if `id` was never registered
    /// via `register_gpu_buffer` — a caller inserting a component before its
    /// GPU buffer is wired up is expected during bring-up, not a bug.
    pub fn write_row_bytes(&self, id: ComponentId, queue: &wgpu::Queue, data: &[u8], row: u32) -> bool {
        match self.gpu_buffers.read().expect("SceneGpuStore gpu_buffers lock poisoned").get(&id) {
            Some(buf) => {
                buf.write_rows_raw(queue, data, row);
                true
            }
            None => false,
        }
    }

    /// Growable counterpart to [`Self::write_row_bytes`], for `id`s
    /// registered via [`Self::register_growable_gpu_buffer`] — grows the
    /// buffer first if `row` doesn't fit the current capacity.
    ///
    /// Returns `None` if `id` was never registered as growable (mirrors
    /// `write_row_bytes`'s "unregistered = no-op" contract exactly — the
    /// caller can't tell "not registered" from "registered fixed instead"
    /// from this alone, which is fine: [`crate::gpu::world_mirror`]'s
    /// dispatch tries `write_row_bytes` first and only falls back to this
    /// method when that returns `false`, so by the time this is called,
    /// "not found here" really does mean "not registered at all").
    ///
    /// Returns `Some(Err(CapacityError))` only if the buffer was registered
    /// with a `max_capacity` ceiling (via `register_growable_gpu_buffer`)
    /// and `row` exceeds it — registrations made with `max_capacity: None`
    /// (the recommendation for World-mirrored columns) can never reach this
    /// case, since unbounded growth cannot fail.
    pub fn write_row_bytes_growing(
        &self,
        id: ComponentId,
        queue: &wgpu::Queue,
        data: &[u8],
        row: u32,
    ) -> Option<Result<(), CapacityError>> {
        self.growable_gpu_buffers
            .read()
            .expect("SceneGpuStore growable_gpu_buffers lock poisoned")
            .get(&id)
            .map(|buf| buf.write_row_growing(queue, row, data))
    }

    /// Lock-safe access to a growable column's current buffer, by
    /// `ComponentId` — the growable-buffer counterpart to
    /// [`Self::buffer_for_id`]. See [`GrowableGpuBufferDispatch::with_buffer`]
    /// for why this is callback-shaped rather than returning `&wgpu::Buffer`
    /// directly. Silently does nothing if `id` isn't registered as growable.
    pub fn with_growable_buffer_for_id(&self, id: ComponentId, f: &mut dyn FnMut(&wgpu::Buffer)) {
        if let Some(buf) = self.growable_gpu_buffers.read().expect("SceneGpuStore growable_gpu_buffers lock poisoned").get(&id) {
            buf.with_buffer(f);
        }
    }

    /// `epoch()` of a growable column's buffer, by `ComponentId` — bump
    /// count since registration; compare against a previously-observed value
    /// to know whether a bind group built against this column needs
    /// rebuilding. `None` if `id` isn't registered as growable.
    pub fn growable_epoch_for_id(&self, id: ComponentId) -> Option<u64> {
        self.growable_gpu_buffers
            .read()
            .expect("SceneGpuStore growable_gpu_buffers lock poisoned")
            .get(&id)
            .map(|buf| buf.epoch())
    }

    /// Current capacity of a growable column's buffer, by `ComponentId`.
    /// `None` if `id` isn't registered as growable.
    pub fn growable_capacity_for_id(&self, id: ComponentId) -> Option<u32> {
        self.growable_gpu_buffers
            .read()
            .expect("SceneGpuStore growable_gpu_buffers lock poisoned")
            .get(&id)
            .map(|buf| buf.capacity())
    }

    /// Registers a dirty-tracked, growable World-mirrored column — for
    /// `#[gpu(mirror = DirtyTracked)]` fields (the default `#[gpu]` mode).
    /// Writes go through [`Self::mark_gpu_row_dirty`] (CPU-side bookkeeping
    /// only) instead of landing on the GPU immediately; [`Self::flush_gpu_mirror`]
    /// performs the actual, coalesced upload. Like
    /// [`Self::register_growable_gpu_buffer`], never bounded by a
    /// `max_capacity` — a World-mirrored column with a capacity ceiling has
    /// no `Result` to fail through on an ordinary `world.insert()` call.
    ///
    /// Sharing (`#[gpu(buffer = "key")]`, `World`-mirror side): the FIRST
    /// registration of `key` in this mode allocates; later same-key
    /// declarations adopt the existing dispatch Arc. The same key may never
    /// span two of the fixed/growable/dirty-tracked registration paths.
    pub fn register_dirty_tracked_gpu_buffer<T, E>(
        &self,
        initial_capacity: u32,
        device: &Arc<wgpu::Device>,
        key: BufferKey,
        mode: MirrorMode,
    ) where
        T: Pod + Send + Sync + HasTypeToken + 'static,
        E: Pod + HasTypeToken + 'static,
    {
        assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
            "gpu buffer key {key:?}: CPU dispatch type and element type disagree in size ({} vs {} bytes) — \
             the per-field wrapper must be byte-identical to the element it wraps",
            std::mem::size_of::<T>(),
            std::mem::size_of::<E>(),
        );
        let id = <T as HasTypeToken>::type_token().id();
        let mut owners = self.owners.write().expect("SceneGpuStore owners lock poisoned");
        if let Some(existing) = owners.get(&key) {
            Self::assert_share_compatible(
                existing,
                key,
                TypeId::of::<E>(),
                std::mem::size_of::<E>(),
                BufferAccess::ReadOnly,
                mode,
                SharedKindTag::DirtyTracked,
            );
            let SharedKind::DirtyTracked(buf) = &existing.kind else { unreachable!("tag checked above") };
            let buf = Arc::clone(buf);
            drop(owners);
            self.dirty_tracked_gpu_buffers.write().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").insert(id, buf);
            self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
            return;
        }
        let buf: Arc<dyn DirtyTrackedGpuBufferDispatch> = Arc::new(DirtyTrackedSceneBuffer::<T>::new(
            Arc::clone(device),
            key.as_str(),
            initial_capacity,
        ));
        let handle = Self::current_handle(&mut |f| buf.with_buffer(f));
        self.registry
            .register_row::<E>(key, handle, BufferAccess::ReadOnly, mode)
            .unwrap_or_else(|e| panic!("failed to register buffer key {key:?}: {e}"));
        owners.insert(
            key,
            SharedOwner {
                element_type_id: TypeId::of::<E>(),
                element_size: std::mem::size_of::<E>(),
                capacity: initial_capacity,
                access: BufferAccess::ReadOnly,
                mode,
                kind: SharedKind::DirtyTracked(Arc::clone(&buf)),
            },
        );
        drop(owners);
        self.dirty_tracked_gpu_buffers.write().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").insert(id, buf);
        self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
    }

    /// Registers a `#[gpu(mirror = Once, heavy)]` field's GPU buffer — the
    /// handle/heavy-element split ([`GpuUploadSource`]). Unlike
    /// [`Self::register_dirty_tracked_gpu_buffer`], the physical buffer is
    /// built from `Element` (the heavy GPU-resident type), NOT from
    /// `Wrapper` (the CPU dispatch type, sized to the lightweight handle) —
    /// there is deliberately no `size_of::<Wrapper>() == size_of::<Element>()`
    /// assertion here, because breaking that equality is the entire point of
    /// `heavy`: the CPU column stays handle-sized while the GPU buffer holds
    /// the (typically much larger) mapped element. `write_gpu_columns_at_row`
    /// is what actually produces `Element`-sized bytes to write here, via
    /// the field's [`GpuColumnDesc::upload`] mapper — never raw handle
    /// bytes.
    ///
    /// Always registers as [`MirrorMode::Once`] — a heavy field's whole
    /// premise (a lightweight handle standing in for asset data uploaded via
    /// its mapper) has no sensible `DirtyTracked` reading of "delta-sync
    /// this every frame"; the derive rejects `heavy` on a `DirtyTracked`
    /// field at macro-expansion time, so this never needs a `mode` param.
    ///
    /// Sharing (`#[gpu(buffer = "key")]`): identical rules to
    /// [`Self::register_dirty_tracked_gpu_buffer`], keyed on `Element`, not
    /// `Wrapper` — two heavy fields (even from different handle types) may
    /// share a key as long as their `Element`s agree.
    pub fn register_dirty_tracked_gpu_buffer_heavy<Wrapper, Element>(
        &self,
        initial_capacity: u32,
        device: &Arc<wgpu::Device>,
        key: BufferKey,
    ) where
        Wrapper: Pod + Send + Sync + HasTypeToken + 'static,
        Element: Pod + Send + Sync + HasTypeToken + 'static,
    {
        let mode = MirrorMode::Once;
        let id = <Wrapper as HasTypeToken>::type_token().id();
        let mut owners = self.owners.write().expect("SceneGpuStore owners lock poisoned");
        if let Some(existing) = owners.get(&key) {
            Self::assert_share_compatible(
                existing,
                key,
                TypeId::of::<Element>(),
                std::mem::size_of::<Element>(),
                BufferAccess::ReadOnly,
                mode,
                SharedKindTag::DirtyTracked,
            );
            let SharedKind::DirtyTracked(buf) = &existing.kind else { unreachable!("tag checked above") };
            let buf = Arc::clone(buf);
            drop(owners);
            self.dirty_tracked_gpu_buffers.write().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").insert(id, buf);
            self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
            return;
        }
        // Built from `Element`, not `Wrapper` — the load-bearing difference
        // from `register_dirty_tracked_gpu_buffer`.
        let buf: Arc<dyn DirtyTrackedGpuBufferDispatch> = Arc::new(DirtyTrackedSceneBuffer::<Element>::new(
            Arc::clone(device),
            key.as_str(),
            initial_capacity,
        ));
        let handle = Self::current_handle(&mut |f| buf.with_buffer(f));
        self.registry
            .register_row::<Element>(key, handle, BufferAccess::ReadOnly, mode)
            .unwrap_or_else(|e| panic!("failed to register buffer key {key:?}: {e}"));
        owners.insert(
            key,
            SharedOwner {
                element_type_id: TypeId::of::<Element>(),
                element_size: std::mem::size_of::<Element>(),
                capacity: initial_capacity,
                access: BufferAccess::ReadOnly,
                mode,
                kind: SharedKind::DirtyTracked(Arc::clone(&buf)),
            },
        );
        drop(owners);
        self.dirty_tracked_gpu_buffers.write().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").insert(id, buf);
        self.column_keys.write().expect("SceneGpuStore column_keys lock poisoned").insert(id, key);
    }

    /// Marks `row` dirty with `data`'s bytes, for a column registered via
    /// [`Self::register_dirty_tracked_gpu_buffer`] — no GPU work happens
    /// here at all; [`Self::flush_gpu_mirror`] does the actual upload.
    /// Returns `false` if `id` isn't registered dirty-tracked (mirrors
    /// [`Self::write_row_bytes`]'s "unregistered = no-op" contract).
    pub fn mark_gpu_row_dirty(&self, id: ComponentId, row: u32, data: &[u8]) -> bool {
        match self.dirty_tracked_gpu_buffers.read().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").get(&id) {
            Some(buf) => {
                buf.mark_dirty_bytes(row, data);
                true
            }
            None => false,
        }
    }

    /// Reads `row`'s current CPU-side shadow value for a column registered
    /// via [`Self::register_dirty_tracked_gpu_buffer`] — `None` if `id`
    /// isn't registered dirty-tracked, or if `row` has never been written.
    /// No GPU access (see [`DirtyTrackedSceneBuffer::get`]'s doc).
    ///
    /// Exists for [`crate::gpu::var_len_pool`]'s benefit: the derive-
    /// generated dispatch for a `Vec<T>`-typed `#[gpu]` field reads the
    /// entity's PREVIOUS [`crate::gpu::VarLenHandle`] this way before
    /// writing a new one, so the old pool allocation can be freed rather
    /// than orphaned.
    pub fn read_dirty_tracked_row_bytes(&self, id: ComponentId, row: u32) -> Option<Vec<u8>> {
        self.dirty_tracked_gpu_buffers
            .read()
            .expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned")
            .get(&id)?
            .read_row_bytes(row)
    }

    /// Registers (or, for a `key` already registered, adopts) a shared
    /// [`crate::gpu::VarLenGpuPool<T>`] backing every `Vec<T>`-typed
    /// `#[gpu]` field declaring this `key` (same `#[gpu(buffer = "key")]`
    /// sharing intent every other registration method here has — see
    /// `var_len_pools`'s own doc for why this lives in a separate map
    /// rather than folding into `owners`).
    ///
    /// # Panics
    ///
    /// If `key` is already registered with a DIFFERENT element type `T` —
    /// two fields sharing one pool key must agree on what they're pooling,
    /// the same "share-compatible" principle
    /// [`Self::register_dirty_tracked_gpu_buffer`] enforces for plain
    /// columns (that method's own richer compatibility metadata doesn't
    /// apply here — a pool has no fixed per-row stride — so this checks the
    /// one thing that actually matters: is it the same `T`).
    pub fn register_var_len_gpu_pool<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &self,
        key: BufferKey,
        initial_capacity: u32,
        device: &Arc<wgpu::Device>,
    ) -> Arc<crate::gpu::VarLenGpuPool<T>> {
        let mut pools = self.var_len_pools.write().expect("SceneGpuStore var_len_pools lock poisoned");
        if let Some(existing) = pools.get(&key) {
            return Arc::clone(existing)
                .downcast::<crate::gpu::VarLenGpuPool<T>>()
                .unwrap_or_else(|_| {
                    panic!(
                        "var-len GPU pool key {key:?} already registered with a different element type -- \
                         every #[gpu(buffer = \"{key:?}\")] Vec<T> field sharing this key must declare the same T"
                    )
                });
        }
        // `STORAGE` alone is enough for every var-len pool's ordinary
        // consumer (a shader reading it as a storage buffer), but adding
        // `VERTEX | INDEX` unconditionally costs nothing for those and is
        // what lets a `Vec<T>`-typed `#[gpu]` field ALSO be bound directly
        // through the fixed-function vertex/index pipeline stages -- e.g. a
        // mesh component's own `vertices`/`indices` fields, read straight
        // by a renderer's draw calls with zero separate "is this a mesh
        // buffer" attribute or opt-in (see `helio::mesh::MeshPool`'s own
        // `VarLenGpuPool`-backed storage, Pulsar-Native#561 Phase D: the
        // SAME pool this call creates for `StaticMeshComponent::vertices`
        // is what Helio's draw calls end up bound to).
        let pool = Arc::new(crate::gpu::VarLenGpuPool::<T>::new_with_usage(
            Arc::clone(device),
            &format!("scenedb-var-len-pool-{key:?}"),
            initial_capacity,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
        ));
        pools.insert(key, Arc::clone(&pool) as Arc<dyn std::any::Any + Send + Sync>);
        pool
    }

    /// Looks up an already-registered [`crate::gpu::VarLenGpuPool<T>`] by
    /// key. `None` if nothing registered this key yet (auto-registration —
    /// mirroring every other World-mirrored column — calls
    /// [`Self::register_var_len_gpu_pool`] on first use instead of calling
    /// this and panicking).
    pub fn var_len_pool<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &self,
        key: BufferKey,
    ) -> Option<Arc<crate::gpu::VarLenGpuPool<T>>> {
        let pools = self.var_len_pools.read().expect("SceneGpuStore var_len_pools lock poisoned");
        pools.get(&key)?.clone().downcast::<crate::gpu::VarLenGpuPool<T>>().ok()
    }

    /// Uploads every row marked dirty (via [`Self::mark_gpu_row_dirty`])
    /// across every dirty-tracked World-mirrored column, coalesced. Call
    /// once per frame — the World-mirror analogue of the cell-mirrored
    /// path's own boundary-phase `sync_all`. A no-op, zero-cost beyond one
    /// empty-map iteration, if nothing was registered dirty-tracked (e.g. no
    /// `#[gpu]` field anywhere used the default `DirtyTracked` mode through
    /// this registration path). As of SceneDB#39, `#[gpu(mirror = Once)]`
    /// fields are ALSO registered here (see the `dirty_tracked_gpu_buffers`
    /// field doc for why), so this same loop covers both modes — nothing
    /// extra needed here to make `Once` deferred/batched too.
    pub fn flush_gpu_mirror(&self, queue: &wgpu::Queue) -> SyncStats {
        let mut total = SyncStats { ranges: 0, bytes: 0 };
        for buf in self.dirty_tracked_gpu_buffers.read().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").values() {
            let stats = buf.flush(queue);
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        total
    }

    /// Lock-safe access to a dirty-tracked column's current buffer, by
    /// `ComponentId` — the dirty-tracked counterpart to
    /// [`Self::with_growable_buffer_for_id`]/[`Self::buffer_for_id`].
    pub fn with_dirty_tracked_buffer_for_id(&self, id: ComponentId, f: &mut dyn FnMut(&wgpu::Buffer)) {
        if let Some(buf) = self.dirty_tracked_gpu_buffers.read().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").get(&id) {
            buf.with_buffer(f);
        }
    }

    /// `epoch()` of a dirty-tracked column's buffer, by `ComponentId` — the
    /// dirty-tracked counterpart to [`Self::growable_epoch_for_id`]. `None`
    /// if `id` isn't registered dirty-tracked.
    pub fn dirty_tracked_epoch_for_id(&self, id: ComponentId) -> Option<u64> {
        self.dirty_tracked_gpu_buffers
            .read()
            .expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned")
            .get(&id)
            .map(|buf| buf.epoch())
    }

    /// Reserves capacity `n` on every World-mirrored buffer registered so
    /// far (both the growable and dirty-tracked maps) — call before a
    /// known-size batch spawn (streaming in a sublevel with M known
    /// objects, spawning a wave of enemies from a design-time-known count)
    /// to move every affected buffer's reallocation cost off the per-insert
    /// critical path, in one call. See
    /// [`crate::world::World::reserve_gpu_mirror_capacity`] for the
    /// `World`-level convenience this backs.
    ///
    /// Stops at the first error and returns it — a caller reserving ahead
    /// of a batch wants to know immediately if some column can't grow that
    /// far (e.g. it hit `wgpu::Limits::max_buffer_size`, see
    /// `DynamicGpuBuffer::ensure_capacity`'s doc), not partway through
    /// spawning the batch itself. Buffers already visited before the
    /// failing one keep whatever capacity they were successfully grown to
    /// — this is a best-effort batch operation, not transactional.
    pub fn reserve_world_mirror_capacity(&self, queue: &wgpu::Queue, n: u32) -> Result<(), CapacityError> {
        for buf in self.growable_gpu_buffers.read().expect("SceneGpuStore growable_gpu_buffers lock poisoned").values() {
            buf.reserve(queue, n)?;
        }
        for buf in self.dirty_tracked_gpu_buffers.read().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").values() {
            buf.reserve(queue, n)?;
        }
        Ok(())
    }

    /// Shrinks every World-mirrored buffer registered so far (both maps) to
    /// the smallest capacity that still covers `highest_live_row`, with
    /// `slack_factor` headroom. See
    /// [`crate::gpu::DynamicGpuBuffer::shrink_to_fit`]'s doc — same
    /// semantics, applied across every registered column at once. Call at a
    /// natural boundary (a level unload, a large despawn batch settling),
    /// not every frame; this is a real GPU-to-GPU copy per buffer that
    /// actually shrinks, same cost profile as growth.
    pub fn shrink_world_mirror_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) {
        for buf in self.growable_gpu_buffers.read().expect("SceneGpuStore growable_gpu_buffers lock poisoned").values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
        for buf in self.dirty_tracked_gpu_buffers.read().expect("SceneGpuStore dirty_tracked_gpu_buffers lock poisoned").values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
    }

    /// Cull's token→mesh link (M3-α T4, C5 amendment): row-indexed beside
    /// `transform_buffer()`, mirrored via [`Self::write_instance_info`].
    ///
    /// Content contract (same as transforms): a row's bytes are defined only
    /// once it has been written via [`Self::write_instance_info`]. Rows in a
    /// recycled region that were never written may hold a prior tenant's
    /// bytes — readers must not treat `mesh_index` as trusted without the
    /// row having passed a harvest/liveness gate; the M3-β cull shader
    /// additionally bounds-checks `mesh_index` against the mesh table.
    /// Owned clone (see [`Self::transform_buffer`]'s doc for why) rather
    /// than a borrow.
    pub fn instance_info_buffer(&self) -> wgpu::Buffer {
        let id = component_id::<InstanceInfo>();
        self.gpu_buffers
            .read()
            .expect("SceneGpuStore gpu_buffers lock poisoned")
            .get(&id)
            .expect("InstanceInfo buffer not registered")
            .buffer()
            .clone()
    }

    /// Row-indexed global-slot mirror (T4; C6 GPU handle validation).
    ///
    /// Guarantee: after every `sync_all`, entries
    /// `[row_base, row_base + rows_in_use)` of each registered cell mirror
    /// `slot_base + slot_column()[row]` EXACTLY — the boundary scan
    /// self-heals any row whose slot changed by any mechanism (write after
    /// alloc, compaction swap, or an alloc into a previously vacated row
    /// that is never written).
    ///
    /// Mirror entries beyond a cell's `rows_in_use` are stale-but-inert:
    /// compaction shrinks the row count without erasing the mirror tail, so
    /// those entries may hold old slot IDs. Nothing may index the mirror
    /// past the harvested row count (M3 contract) — consumers dispatch over
    /// `rows_in_use`, never region capacity.
    pub fn slot_mirror_buffer(&self) -> &wgpu::Buffer {
        self.slot_mirror.buffer()
    }

    /// Return the registered GpuColumnDesc entries for the given cell.
    /// Stub for future use — the derive macro will wire this up for editor
    /// property inspection and debug tooling. Currently returns empty.
    pub fn gpu_column_descs_for(&self, _cell_id: CellId) -> Vec<GpuColumnDesc> {
        Vec::new()
    }

    pub fn generation_buffer(&self) -> &wgpu::Buffer {
        self.generations.buffer()
    }

    /// Per-cell metadata SSBO (α: allocated, no writer).
    pub fn cell_metadata_buffer(&self) -> &wgpu::Buffer {
        &self.cell_metadata
    }

    // ── Telemetry / snapshot accessors (pub(crate)) ──────────────────────

    /// GPU buffers map for telemetry snapshot. Returns the read-lock guard
    /// itself (not a `&HashMap` — `gpu_buffers` lives behind a `RwLock`, see
    /// its field doc) so callers `Deref` through it exactly like a plain
    /// reference.
    pub(crate) fn telemetry_gpu_buffers(&self) -> std::sync::RwLockReadGuard<'_, HashMap<ComponentId, Arc<dyn GpuBufferDispatch>>> {
        self.gpu_buffers.read().expect("SceneGpuStore gpu_buffers lock poisoned")
    }

    /// Row region pools for telemetry snapshot.
    pub(crate) fn telemetry_row_pools(&self) -> &[RegionPool] {
        &self.row_pools
    }

    /// Slot region pools for telemetry snapshot.
    pub(crate) fn telemetry_slot_pools(&self) -> &[RegionPool] {
        &self.slot_pools
    }

    /// Per-cell GPU state for telemetry snapshot.
    pub(crate) fn telemetry_cells(&self) -> &[Option<CellGpuState>] {
        &self.cells
    }
}
