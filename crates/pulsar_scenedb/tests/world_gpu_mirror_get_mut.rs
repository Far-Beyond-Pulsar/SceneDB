//! Proves `World::get_mut`'s `Mut<T>` guard actually reaches the GPU mirror
//! on drop — for BOTH `MirrorMode`s, not just `DirtyTracked`.
//!
//! Before this, `World::insert` had a GPU dispatch hook but `get_mut` had
//! none at all: mutating a `#[gpu]` field through `get_mut` silently never
//! reached the GPU. That's a real divergence from `insert`-driven mutation,
//! not a documented limitation of `Once` specifically — it affected
//! `DirtyTracked` fields too.
//!
//! `Once` gets an extra, sharper assertion here: `tests/world_gpu_mirror_
//! dirty_tracked.rs::once_mode_field_never_rewrites_after_the_first_insert`
//! already proves a routine RE-INSERT must NOT re-touch an `Once` field's
//! GPU buffer (the whole point of the mode). This file proves the opposite
//! is true for an explicit `get_mut` MUTATION: because the caller is
//! deliberately changing the value (not just re-supplying the same
//! component), `Once` fields DO re-upload through `get_mut`, exactly like
//! `DirtyTracked` fields do — the pinning is insert-path-specific, not a
//! blanket "never touch this field again."

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
        label: Some("scenedb-world-gpu-mirror-get-mut-test"),
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
    mesh_id: u32,
    #[gpu]
    hp: u32,
}

#[test]
fn get_mut_mutation_reaches_gpu_for_both_dirty_tracked_and_once_fields() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;

    let columns = MixedModeComponent::gpu_columns();
    let mesh_id_field_id = columns.iter().find(|c| c.buffer_name == "mesh_id").unwrap().field_token.id();
    let hp_field_id = columns.iter().find(|c| c.buffer_name == "hp").unwrap().field_token.id();

    world.insert(entity, MixedModeComponent { mesh_id: 42, hp: 100 });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let mut mesh_id_before = 0u32;
    store.with_dirty_tracked_buffer_for_id(mesh_id_field_id, &mut |buf| mesh_id_before = readback_u32(&ctx, buf, row));
    assert_eq!(mesh_id_before, 42);
    let mut hp_before = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| hp_before = readback_u32(&ctx, buf, row));
    assert_eq!(hp_before, 100);

    // Explicit get_mut mutation, not a re-insert -- both fields change, the
    // guard drops at the end of this scope.
    {
        let mut c = world.get_mut::<MixedModeComponent>(entity).expect("component present");
        c.hp = 55;
        c.mesh_id = 777;
    }
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let mut hp_after = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| hp_after = readback_u32(&ctx, buf, row));
    assert_eq!(hp_after, 55, "DirtyTracked field must reach the GPU after a get_mut mutation + flush");

    let mut mesh_id_after = 0u32;
    store.with_dirty_tracked_buffer_for_id(mesh_id_field_id, &mut |buf| mesh_id_after = readback_u32(&ctx, buf, row));
    assert_eq!(
        mesh_id_after, 777,
        "Once field must ALSO reach the GPU after an explicit get_mut mutation -- unlike a routine re-insert, \
         this is the caller deliberately changing the value"
    );

    // Also confirm the CPU-side value itself was actually updated (the
    // guard's Deref/DerefMut didn't just satisfy the borrow checker).
    let c = world.get::<MixedModeComponent>(entity).unwrap();
    assert_eq!(c.hp, 55);
    assert_eq!(c.mesh_id, 777);
}

#[test]
fn get_mut_on_a_plain_non_gpu_component_still_works_and_is_a_gpu_no_op() {
    #[derive(SceneStore, Clone, Copy)]
    struct NoGpuFields {
        value: u32,
    }

    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    world.insert(e, NoGpuFields { value: 1 });
    {
        let mut v = world.get_mut::<NoGpuFields>(e).unwrap();
        v.value = 2;
    }
    assert_eq!(world.get::<NoGpuFields>(e).unwrap().value, 2);
}

#[test]
fn get_mut_without_a_gpu_mirror_attached_still_works() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, MixedModeComponent { mesh_id: 1, hp: 2 });
    {
        let mut c = world.get_mut::<MixedModeComponent>(e).unwrap();
        c.hp = 9;
    }
    assert_eq!(world.get::<MixedModeComponent>(e).unwrap().hp, 9);
}
