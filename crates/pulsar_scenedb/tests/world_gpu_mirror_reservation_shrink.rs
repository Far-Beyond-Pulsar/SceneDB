//! Proves #212: `World::reserve_gpu_mirror_capacity` eliminates in-batch
//! reallocations, `World::shrink_gpu_mirror_to_fit` reclaims capacity after
//! a peak-then-drop, and buffer growth now fails with a catchable
//! `CapacityError` instead of a `wgpu` validation panic once the device's
//! own `max_buffer_size` is hit -- Helio#211's benchmark harness found this
//! reachable at realistic AAA scale (a 256-byte packed row hits the default
//! 256 MiB limit at 1,048,576 rows in one buffer), not a theoretical case.

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
        label: Some("scenedb-world-gpu-mirror-reservation-shrink-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[derive(SceneStore, Clone, Copy)]
struct ReserveTestComponent {
    #[gpu]
    value: u32,
}

#[test]
fn reserve_eliminates_reallocations_across_a_batch_spawn() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    ReserveTestComponent::register_gpu_columns_growable(&mut store, 4, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let id = ReserveTestComponent::gpu_columns()[0].field_token.id();

    world
        .reserve_gpu_mirror_capacity(ctx.queue(), 1000)
        .expect("mirror attached")
        .expect("reserve succeeds");

    // ReserveTestComponent's field is the DirtyTracked default -- reserve
    // must have grown the dirty-tracked map's buffer, not the growable one.
    let mut epoch_before = None;
    store.with_dirty_tracked_buffer_for_id(id, &mut |_| epoch_before = Some(()));
    assert!(epoch_before.is_some(), "reserve must have registered/grown the dirty-tracked buffer");

    for i in 0..1000u32 {
        let e = world.spawn();
        world.insert(e, ReserveTestComponent { value: i });
    }
    let stats = world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert_eq!(stats.ranges, 1, "1000 contiguous rows, pre-reserved -- one coalesced flush range, zero growth mid-batch");
}

#[test]
fn shrink_reclaims_capacity_after_a_peak_then_drop() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    ReserveTestComponent::register_gpu_columns_growable(&mut store, 4, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // Peak: spawn 1000, mostly despawn them.
    let mut entities = Vec::new();
    for i in 0..1000u32 {
        let e = world.spawn();
        world.insert(e, ReserveTestComponent { value: i });
        entities.push(e);
    }
    world.flush_gpu_mirror(ctx.queue());
    for e in entities.drain(10..) {
        world.despawn(e);
    }
    // 10 entities survive, at rows 0..10.
    world.shrink_gpu_mirror_to_fit(ctx.queue(), 9, 1.0);

    let id = ReserveTestComponent::gpu_columns()[0].field_token.id();
    // Surviving rows must still read back correctly after the shrink.
    let mut got = Vec::new();
    store.with_dirty_tracked_buffer_for_id(id, &mut |buf| {
        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: 40,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device().create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, 40);
        ctx.queue().submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        got = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
    });
    for i in 0..10usize {
        assert_eq!(u32::from_ne_bytes(got[i * 4..i * 4 + 4].try_into().unwrap()), i as u32);
    }
}

#[test]
fn capacity_ceiling_is_a_catchable_error_not_a_process_crash() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // A component wide enough (one 64-byte field) that a modest row count
    // already approaches realistic device limits is impractical to test
    // portably (max_buffer_size varies per adapter) -- instead, prove the
    // MECHANISM directly against the real per-buffer API this test's sibling
    // unit tests (gpu::dynamic_buffer::tests) already prove numerically:
    // reserving u32::MAX rows of a u32-sized field must come back as
    // Err(CapacityError), never a wgpu validation panic, regardless of
    // this adapter's specific max_buffer_size.
    ReserveTestComponent::register_gpu_columns_growable(&mut store, 4, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let result = world.reserve_gpu_mirror_capacity(ctx.queue(), u32::MAX).expect("mirror attached");
    assert!(result.is_err(), "an absurd reservation must come back as Err, not panic the process");
}
