//! Proves `SceneGpuStore::resolve_buffer_handle`'s epoch tracks REAL growth
//! of a World-mirrored `#[gpu(buffer = "key")]` column, end to end, through
//! the keyed `GpuBufferRegistry` (Tier 1) — not just at first registration.
//!
//! Before this test's fix, `GpuBufferRegistry`'s copy of a key's
//! buffer/epoch was captured once at first registration and never refreshed:
//! `register_gpu_columns_growable`'s "adopt the existing owner" fast path
//! (taken by every insert after the very first) never re-touched
//! `self.registry`, so a caller resolving straight from the registry would
//! see a permanently stale buffer after any growth. `resolve_buffer_handle`
//! now re-syncs the registry from the LIVE dispatch object's own `epoch()`
//! (which already tracked growth correctly) on every call.

use pulsar_scenedb::gpu::{
    BufferKey, EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
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
        label: Some("scenedb-gpu-buffer-registry-epoch-test"),
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
struct GrowTagged {
    #[gpu(buffer = "grow_key")]
    tag: u32,
}

#[test]
fn resolve_buffer_handle_epoch_tracks_growth_and_is_idempotent_between_growths() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // Deliberately tiny initial capacity so a handful of inserts forces
    // multiple reallocations.
    GrowTagged::register_gpu_columns_growable(&mut store, 2, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // A few inserts, still within (or barely past) the initial capacity.
    let mut entities = Vec::new();
    for i in 0..2u32 {
        let e = world.spawn();
        world.insert(e, GrowTagged { tag: i });
        entities.push(e);
    }
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let h0 = store
        .resolve_buffer_handle(BufferKey::of("grow_key"))
        .expect("registered");

    // Resolving again with no intervening writes must be a no-op: same
    // epoch, same buffer size — proves `sync` doesn't spuriously bump.
    let h0_again = store
        .resolve_buffer_handle(BufferKey::of("grow_key"))
        .expect("registered");
    assert_eq!(h0_again.epoch, h0.epoch, "no growth happened, epoch must not change between resolves");
    assert_eq!(h0_again.buffer.size(), h0.buffer.size());

    // Spawn far more entities than the initial capacity of 2 -- forces
    // several doubling reallocations of the underlying dispatch buffer.
    for i in 2..60u32 {
        let e = world.spawn();
        world.insert(e, GrowTagged { tag: i });
        entities.push(e);
    }
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let h1 = store
        .resolve_buffer_handle(BufferKey::of("grow_key"))
        .expect("still registered");

    assert!(
        h1.epoch > h0.epoch,
        "epoch must advance past growth: h0.epoch={}, h1.epoch={}",
        h0.epoch,
        h1.epoch
    );
    assert!(
        h1.buffer.size() > h0.buffer.size(),
        "the resolved buffer must actually be the grown (bigger) allocation: {} vs {} bytes",
        h0.buffer.size(),
        h1.buffer.size()
    );

    // Resolving again with no further writes must again be stable.
    let h1_again = store
        .resolve_buffer_handle(BufferKey::of("grow_key"))
        .expect("still registered");
    assert_eq!(h1_again.epoch, h1.epoch, "epoch must be stable across repeated resolves with no growth");
}

#[test]
fn resolve_buffer_handle_is_none_for_an_unregistered_key() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    assert!(store.resolve_buffer_handle(BufferKey::of("never-registered")).is_none());
}
