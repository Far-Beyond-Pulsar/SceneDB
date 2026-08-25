//! SceneDB GPU layer (M2b-α, design Rev 2): persistent region-partitioned
//! scene SSBOs, CPU→GPU delta-sync, and pin-by-serial retirement across N
//! registered cells. Feature-gated (`gpu`); the core crate stays
//! graphics-free (CONTRACTS C0).
//!
//! Mirrored columns must be written via `SceneGpuStore::write_transform` and
//! compacted via the frame-boundary drivers in [`phase`]; raw column access
//! bypasses dirty tracking. The frame phase itself is enforced at compile
//! time (design Rev 2 §6, C3): mutation requires a [`SimulateWitness`], and
//! the boundary stages (retire → compact → sync) are reachable only through
//! [`FrameDriver`] and [`BoundaryPhase`]'s consuming transitions — see
//! `phase.rs` for the witness chain and its compile_fail doc-tests.

mod assets;
mod buffer;
mod buffer_registry;
mod context;
mod dirty;
mod dirty_tracked_scene_buffer;
mod dynamic_buffer;
mod freelist;
mod growable_scene_buffer;
mod generation;
mod grid;
mod harvest;
mod interned_pool;
mod phase;
mod placement;
mod readback;
mod region;
mod scatter_write;
mod scene_store;
mod slot_allocator;
mod system_binding;
mod tier;
mod tier_layout;
mod tracker;
mod var_len_pool;
mod view_upload;
pub mod world_mirror;

pub use assets::{
    ArenaError, ClusterBuffer, ClusterError, ClusterNode, GeometryArena, MaterialError,
    MaterialRegistry, MaterialRow, MeshError, MeshMetadata, MeshRegistry, MeshletBuffer,
    MeshletEntry, MeshletError, TextureError, TextureStore, CLUSTER_BUFFER_KEY,
    GEOMETRY_INDEX_BUFFER_KEY, GEOMETRY_VERTEX_BUFFER_KEY, MATERIAL_BUFFER_KEY,
    MAX_TEXTURE_SLOTS, MESHLET_BUFFER_KEY, MESH_METADATA_BUFFER_KEY, TEXTURE_STORE_KEY,
};
pub use buffer::{GpuBufferDispatch, SceneBuffer, SyncStats, GAP_MERGE_THRESHOLD};
pub use buffer_registry::{
    BufferAccess, BufferHandle, BufferKey, BufferRegistrationError, GpuBufferRegistry,
};
pub use context::EngineGpuContext;
pub use dirty::DirtyMask;
pub use dirty_tracked_scene_buffer::{DirtyTrackedGpuBufferDispatch, DirtyTrackedSceneBuffer};
pub use dynamic_buffer::{CapacityError, DynamicGpuBuffer};
pub use growable_scene_buffer::{GrowableGpuBufferDispatch, GrowableSceneBuffer};
pub use generation::GenerationBuffer;
pub use grid::{
    execute_transitions, execute_transitions_with_ram, BudgetError, CellCoord, Domain, GridConfig,
    RamHooks, StreamingBudget, StreamingGrid, Transition, TransitionStats, WarmTierConfig,
};
pub use harvest::{
    revalidate_run, HarvestLease, HarvestPipeline, HarvestStaging, HarvestStats, MeshClass, View,
};
pub use interned_pool::InternedVarLenPool;
pub use placement::{GpuHeavy, GpuRef, structural_content_id};
pub use phase::{BoundaryPhase, CompactedPhase, FrameDriver, HarvestPhase, RetiredPhase, SimulateA, SimulateB, SimulateWitness};
pub use readback::{readback_bytes, readback_row};
pub use region::{RegionPool, RegionError};
pub use scene_store::{
    CellId, CellMetadataRow, CellSlot, GpuColumnDesc, GpuColumnSet, GpuUploadSource, MirrorMode,
    RegionClassConfig, SceneGpuConfig, SceneGpuStore, UploadMapperFn, CELL_METADATA_BUFFER_KEY,
    GENERATION_BUFFER_KEY, SLOT_MIRROR_BUFFER_KEY,
};
pub use slot_allocator::{SlotAllocator, SlotHandle};
pub use system_binding::{BufferBinding, BufferResolveError, GpuSystemContext};
pub use tier::{
    MaterializationSpec, Tier, TierAuditKey, TierAuditRecord, TierConfig, TierError, TierFetch,
    TierPeek, TierSelector, TierSpan, TierStats,
};
pub use tier_layout::{register_segment_layout, register_segment_layout_for_type, LayoutError, Segment};
pub use tracker::SubmissionTracker;
pub use var_len_pool::{VarLenBufferRef, VarLenGpuPool, VarLenHandle};
pub use view_upload::ViewTokenBuffers;
pub use world_mirror::{
    free_interned_var_len_field_at_row, free_var_len_field_at_row, write_gpu_columns_at_row,
    write_interned_var_len_field_at_row, write_interned_var_len_field_with_id_at_row, write_var_len_field_at_row, GenerationMirror,
    GpuMirrorHandle, GpuMirrorRegistration, VarLenReleaseRegistration,
};
// `InstanceInfo` is defined graphics-free in `crate::spatial` (CONTRACTS C0)
// and already re-exported at the crate root; re-exported here too so GPU-
// adjacent consumers (e.g. Helio's `helio-scenedb` seam reflection harness,
// M3-a T10) can reach every C5 struct type through one `gpu::` path.
pub use crate::spatial::InstanceInfo;

/// Reinterpret a Pod slice as bytes for `queue.write_buffer`.
pub(crate) fn as_bytes<T: crate::page::Pod>(s: &[T]) -> &[u8] {
    // SAFETY: T: Pod guarantees no padding-UB and no invalid bit patterns.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
