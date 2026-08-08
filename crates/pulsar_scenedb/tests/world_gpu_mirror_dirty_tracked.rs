//! Proves the two behaviors #29 adds to `gpu::world_mirror`:
//!
//! 1. `#[gpu(mirror = Once)]` fields write on the first insert and never
//!    again — a later `world.insert()` re-inserting the same component
//!    (updating some other field) must NOT re-touch a `Once`-mode field's
//!    GPU buffer.
//! 2. `#[gpu]` (`DirtyTracked`, the default) fields defer their write --
//!    `World::insert` alone does not reach the GPU; `World::flush_gpu_mirror`
//!    does, coalesced.

use pulsar_scenedb::gpu::{EngineGpuContext, GpuColumnSet, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
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
        label: Some("scenedb-world-gpu-mirror-dirty-tracked-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback_u32(ctx: &EngineGpuContext, buf: &wgpu::Buffer, row: u64) -> u32 {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, row * 4, &staging, 0, 4);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    u32::from_ne_bytes(data.try_into().unwrap())
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[derive(SceneStore, Clone, Copy)]
struct MixedModeComponent {
    #[gpu(mirror = Once)]
    mesh_id: u32, // set once at spawn, never changes
    #[gpu]
    hp: u32, // DirtyTracked (default) -- changes every frame in a real game
}

#[test]
fn once_mode_field_never_rewrites_after_the_first_insert() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;

    let columns = MixedModeComponent::gpu_columns();
    let mesh_id_col = columns.iter().find(|c| c.buffer_name == "mesh_id").unwrap();
    let mesh_id_field_id = mesh_id_col.field_token.id();

    world.insert(entity, MixedModeComponent { mesh_id: 42, hp: 100 });
    let mut got = 0u32;
    store.with_growable_buffer_for_id(mesh_id_field_id, &mut |buf| got = readback_u32(&ctx, buf, row));
    assert_eq!(got, 42, "Once field must write on the first insert");

    // Re-insert (an in-place update -- entity already has this component):
    // mesh_id changes in the CPU-side value, but the GPU buffer must NOT
    // reflect it -- Once means "uploaded once at registration and never
    // touched again."
    world.insert(entity, MixedModeComponent { mesh_id: 999, hp: 50 });
    let mut got_after = 0u32;
    store.with_growable_buffer_for_id(mesh_id_field_id, &mut |buf| got_after = readback_u32(&ctx, buf, row));
    assert_eq!(got_after, 42, "Once field must NOT re-write on a later update, even though the CPU value changed");
}

#[test]
fn dirty_tracked_field_defers_until_flush_gpu_mirror() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;
    let hp_field_id = MixedModeComponent::gpu_columns().iter().find(|c| c.buffer_name == "hp").unwrap().field_token.id();

    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 100 });

    let mut before_flush = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| before_flush = readback_u32(&ctx, buf, row));
    assert_eq!(before_flush, 0, "DirtyTracked field must NOT be on the GPU yet -- insert alone must not reach it");

    // Several updates before any flush -- only the LAST value should matter
    // once flushed, and only ONE upload should be needed regardless of how
    // many times it changed in between.
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 80 });
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 60 });
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 42 });

    let stats = world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert!(stats.ranges >= 1, "flush must have uploaded something");

    let mut after_flush = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| after_flush = readback_u32(&ctx, buf, row));
    assert_eq!(after_flush, 42, "flush must upload the LATEST value, not an intermediate one");

    // A second flush with nothing newly dirty must be a true no-op.
    let stats2 = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(stats2.ranges, 0);
    assert_eq!(stats2.bytes, 0);
}

#[test]
fn dirty_tracked_writes_across_multiple_entities_coalesce_on_flush() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 16, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let hp_field_id = MixedModeComponent::gpu_columns().iter().find(|c| c.buffer_name == "hp").unwrap().field_token.id();

    // 5 adjacent entities (rows 0..5) -- must coalesce into ONE range.
    let mut entities = Vec::new();
    for i in 0..5u32 {
        let e = world.spawn();
        world.insert(e, MixedModeComponent { mesh_id: 0, hp: i * 10 });
        entities.push(e);
    }
    let stats = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(stats.ranges, 1, "5 adjacent dirty rows must coalesce into exactly one upload range");

    for (i, e) in entities.iter().enumerate() {
        let mut got = 0u32;
        store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| got = readback_u32(&ctx, buf, e.index() as u64));
        assert_eq!(got, (i as u32) * 10);
    }
}
