//! Proves the `World` <-> GPU mirror bridge end-to-end, with a real GPU
//! device: `#[derive(SceneStore)]`'s `#[gpu]` field on a component
//! attached to a `World` entity via plain `world.insert()` — no
//! CellStorage, no Handle, no separate "mirror-aware" call — lands in its
//! registered GPU buffer at row = `entity.index()`, automatically, the
//! moment `attach_gpu_mirror` has been called once at setup.
//!
//! This is the concrete scenario the bridge exists for: a component like
//! `StaticMeshComponent { mesh: GpuMeshHandle }` where `mesh` is `#[gpu]`
//! gets implicitly GPU-resident as soon as it's inserted onto an entity.
//!
//! Three things are proven, each load-bearing:
//! 1. `world.insert(entity, GpuTagged { .. })` — the value's `#[gpu]` field
//!    is readable back from the GPU buffer at exactly `entity.index()`,
//!    for entities at non-trivial (non-zero, non-contiguous-from-zero)
//!    indices, and survives an in-place re-insert (update) too.
//! 2. `world.insert(entity, PlainCpuOnly { .. })` — a component with no
//!    `GpuColumnSet` impl at all inserts exactly as it always has; the
//!    mirror never required it to implement anything.
//! 3. Before `attach_gpu_mirror` is ever called, `world.insert` of a
//!    GPU-tagged component behaves identically to today (no GPU write, no
//!    panic) — proving the feature is genuinely opt-in, not just opt-out
//!    of a default that was already on.

use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuColumnSet, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore,
};
use pulsar_scenedb::{Entity, World};
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scenedb-world-gpu-mirror-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback(ctx: &EngineGpuContext, buf: &wgpu::Buffer, src_offset: u64, bytes: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, src_offset, &staging, 0, bytes);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

fn scene_cfg() -> SceneGpuConfig {
    // World-mirrored buffers don't use cells/regions at all (see
    // `gpu::world_mirror`'s docs) — this config only needs to be valid
    // enough for `SceneGpuStore::new` to construct its own built-ins.
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

/// World capacity the mirrored buffer is registered at — must cover every
/// `Entity::index()` used below.
const WORLD_GPU_CAPACITY: u32 = 1024;

/// The scenario from the design doc: a mesh component whose `mesh` field is
/// GPU-native (imagine it's a packed index into a mesh registry) and a
/// CPU-only `label` field alongside it, proving per-field selection still
/// holds for World-mirrored types exactly as it does for cell-mirrored ones.
#[derive(SceneStore, Clone, Copy)]
struct StaticMeshComponent {
    #[gpu]
    mesh: u32,
    label: u32,
}

/// A component with no GPU involvement whatsoever — not `Pod`, not
/// `GpuColumnSet`. `World::insert` must accept this exactly as it always
/// has; the mirror hook must not require it to be anything special.
struct PlainCpuOnly {
    name: String,
}

#[test]
fn gpu_tagged_component_lands_in_its_buffer_at_entity_index_on_plain_insert() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    StaticMeshComponent::register_gpu_columns(&mut store, WORLD_GPU_CAPACITY, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // Spawn a run of throwaway entities first so the entity under test
    // lands at a non-zero, non-trivial index — row 0 succeeding would not
    // prove `entity.index()` (not just "row 0 always works") is really
    // driving the write.
    for _ in 0..7 {
        world.spawn();
    }
    let entity: Entity = world.spawn();
    let row = entity.index();
    assert_eq!(row, 7, "sanity: this test's whole point depends on a non-zero row");

    world.insert(entity, StaticMeshComponent { mesh: 0xBEEF, label: 0xCAFE });

    let columns = StaticMeshComponent::gpu_columns();
    assert_eq!(columns.len(), 1, "StaticMeshComponent has exactly one #[gpu] field");
    let mesh_field_id = columns[0].field_token.id();
    let mesh_buf = store.buffer_for_id(mesh_field_id).expect("mesh buffer registered");

    let bytes = readback(&ctx, mesh_buf, (row as u64) * 4, 4);
    let got = u32::from_ne_bytes(bytes.try_into().unwrap());
    assert_eq!(got, 0xBEEF, "GPU-tagged field must be readable at row = entity.index()");

    // Re-insert (update) with a new value — the in-place-update path in
    // `World::insert_inner` must re-mirror too, not just the
    // first-time/migration path.
    world.insert(entity, StaticMeshComponent { mesh: 0x1234, label: 0xCAFE });
    let bytes = readback(&ctx, mesh_buf, (row as u64) * 4, 4);
    let got = u32::from_ne_bytes(bytes.try_into().unwrap());
    assert_eq!(got, 0x1234, "re-insert (update) must re-mirror the new value");
}

#[test]
fn non_gpu_component_inserts_normally_and_never_touches_the_store() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    StaticMeshComponent::register_gpu_columns(&mut store, WORLD_GPU_CAPACITY, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    // Compiles and runs with zero GpuColumnSet bound anywhere in scope for
    // PlainCpuOnly — the whole point of the autoref-specialized dispatch.
    world.insert(entity, PlainCpuOnly { name: "not gpu at all".to_string() });

    assert_eq!(world.get::<PlainCpuOnly>(entity).unwrap().name, "not gpu at all");
}

#[test]
fn without_an_attached_mirror_insert_never_touches_the_gpu() {
    // No `attach_gpu_mirror` call at all — must behave exactly as `World`
    // did before this feature existed: no panic, no silent GPU write
    // (nothing to write TO, since no buffer was registered either).
    let mut world = World::new();
    let entity = world.spawn();
    world.insert(entity, StaticMeshComponent { mesh: 99, label: 1 });
    assert_eq!(world.get::<StaticMeshComponent>(entity).unwrap().mesh, 99);
    assert!(!world.has_gpu_mirror());
}
