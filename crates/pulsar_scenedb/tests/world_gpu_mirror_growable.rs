//! Proves the growable World-mirror path end-to-end: a `#[gpu]`-tagged
//! component registered via the derive's generated
//! `register_gpu_columns_growable` (a small initial capacity, no
//! `max_capacity` ceiling) survives `World::insert`s at entity indices far
//! past that initial capacity -- the exact scenario `SceneBuffer`'s
//! fixed-capacity contract guarantees eventually panics on, without this.

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
        label: Some("scenedb-world-gpu-mirror-growable-test"),
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
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[derive(SceneStore, Clone, Copy)]
struct GrowableTagComponent {
    #[gpu]
    tag: u32,
}

#[test]
fn world_insert_past_initial_growable_capacity_does_not_panic_and_reads_back_correctly() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // Deliberately tiny initial capacity -- this test's whole point is that
    // entities spawned well past it still work.
    GrowableTagComponent::register_gpu_columns_growable(&mut store, 2, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // Spawn far more entities than the initial capacity of 2.
    let mut entities = Vec::new();
    for i in 0..50u32 {
        let e = world.spawn();
        world.insert(e, GrowableTagComponent { tag: i * 7 });
        entities.push(e);
    }

    let columns = GrowableTagComponent::gpu_columns();
    assert_eq!(columns.len(), 1);
    let id = columns[0].field_token.id();

    let capacity = store.growable_capacity_for_id(id).expect("registered growable");
    assert!(capacity > 2, "must have grown past the initial capacity of 2, got {capacity}");
    assert!(store.growable_epoch_for_id(id).unwrap() >= 1, "at least one reallocation must have happened");

    // Every entity's value must still be correct after however many
    // reallocations happened along the way.
    let mut buf_bytes = Vec::new();
    store.with_growable_buffer_for_id(id, &mut |buf| {
        buf_bytes = readback(&ctx, buf, 0, (capacity as u64) * 4);
    });
    for (i, entity) in entities.iter().enumerate() {
        let row = entity.index() as usize;
        let got = u32::from_ne_bytes(buf_bytes[row * 4..row * 4 + 4].try_into().unwrap());
        assert_eq!(got, (i as u32) * 7, "row {row} (entity #{i}) must survive every intervening growth");
    }
}

#[test]
fn non_gpu_component_still_registers_a_growable_stub_with_no_effect() {
    // GrowableTagComponent's own register_gpu_columns_growable is exercised
    // above; this proves a type with ZERO #[gpu] fields also gets a valid
    // (no-op) register_gpu_columns_growable, so generic code that calls it
    // uniformly across every #[derive(SceneStore)] type doesn't need to
    // special-case "has no #[gpu] fields."
    #[derive(SceneStore, Clone, Copy)]
    struct NoGpuFields {
        value: u32,
    }

    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    NoGpuFields::register_gpu_columns_growable(&mut store, 4, ctx.device());
    assert!(NoGpuFields::gpu_columns().is_empty());
}
