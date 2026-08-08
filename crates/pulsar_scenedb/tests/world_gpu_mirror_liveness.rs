//! Proves the World-mirror liveness/generation mirror: `World::spawn`/
//! `despawn`, when a GPU mirror is attached, keep a GPU-resident generation
//! buffer in lockstep with `World::entity_slots`'s own generation -- giving
//! a GPU-side consumer the same staleness check `World::is_alive` already
//! performs on the CPU side.

use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle};
use pulsar_scenedb::World;
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
        label: Some("scenedb-world-gpu-mirror-liveness-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback_u32(ctx: &EngineGpuContext, buf: &wgpu::Buffer, row: u32) -> u32 {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, (row as u64) * 4, &staging, 0, 4);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    u32::from_ne_bytes(data.try_into().unwrap())
}

use pulsar_scenedb::gpu::{RegionClassConfig, SceneGpuConfig, SceneGpuStore};

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

#[test]
fn spawn_and_despawn_keep_the_gpu_generation_mirror_in_lockstep_with_entity_slots() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));

    let mut world = World::new();
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));
    world.attach_gpu_mirror(mirror.clone());

    // Spawn: row's generation on the GPU must equal entity.generation().
    let e1 = world.spawn();
    let row = e1.index();
    let mut gen_buf = None;
    mirror.generations().with_buffer(&mut |buf| gen_buf = Some(readback_u32(&ctx, buf, row)));
    assert_eq!(gen_buf.unwrap(), e1.generation(), "spawn must mirror the entity's own generation");

    // Despawn: the GPU generation must advance PAST e1's own generation --
    // a reader still holding `e1` must be able to detect staleness by
    // comparing e1.generation() against this row going forward.
    world.despawn(e1);
    let mut gen_after_despawn = None;
    mirror.generations().with_buffer(&mut |buf| gen_after_despawn = Some(readback_u32(&ctx, buf, row)));
    let gen_after_despawn = gen_after_despawn.unwrap();
    assert_ne!(
        gen_after_despawn,
        e1.generation(),
        "a stale reader holding e1 must see a generation MISMATCH after despawn, not a false-positive match"
    );

    // Respawn recycling the same slot: the GPU generation must advance
    // again, to the NEW entity's generation -- and must still differ from
    // the original e1's.
    let e2 = world.spawn();
    assert_eq!(e2.index(), row, "sanity: this test's whole point depends on the slot being recycled");
    assert_ne!(e2.generation(), e1.generation(), "sanity: a recycled slot must get a new generation");

    let mut gen_after_respawn = None;
    mirror.generations().with_buffer(&mut |buf| gen_after_respawn = Some(readback_u32(&ctx, buf, row)));
    assert_eq!(gen_after_respawn.unwrap(), e2.generation(), "respawn must mirror the NEW entity's generation");
}

#[test]
fn no_attached_mirror_means_spawn_despawn_behave_exactly_as_before() {
    // No attach_gpu_mirror call at all -- must not panic, must not require
    // any GPU resources to even exist.
    let mut world = World::new();
    let e = world.spawn();
    assert!(world.is_alive(e));
    assert!(world.despawn(e));
    assert!(!world.is_alive(e));
    assert!(world.gpu_mirror().is_none());
}
