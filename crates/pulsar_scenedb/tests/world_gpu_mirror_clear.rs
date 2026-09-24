//! Removing a `#[gpu]` component from an entity (remove or despawn) clears
//! its GPU row, so consumers reading rows by entity index stop seeing it.

use pulsar_scenedb::gpu::{BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "clear_test_packed")]
struct Packed {
    #[gpu]
    a: u32,
    #[gpu]
    b: u32,
}

#[derive(SceneStore, Clone, Copy)]
struct PerField {
    #[gpu(buffer = "clear_test_field")]
    value: u32,
}

fn setup() -> Option<(EngineGpuContext, Arc<SceneGpuStore>, World)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    let ctx = EngineGpuContext::new(Arc::new(device), Arc::new(queue));
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig { classes: vec![], tombstone_headroom: 0, max_cells_metadata: 0 },
    );
    Packed::register_gpu_columns_growable(&mut store, 8, ctx.device());
    PerField::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(store.clone(), ctx.queue().clone()));
    Some((ctx, store, world))
}

fn row_words(ctx: &EngineGpuContext, store: &SceneGpuStore, key: &'static str, row: u32, words: u64) -> Vec<u32> {
    let handle = store.resolve_buffer_handle(BufferKey::of(key)).expect("registered");
    let bytes = words * 4;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&handle.buffer, row as u64 * bytes, &staging, 0, bytes);
    ctx.queue().submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out = staging
        .slice(..)
        .get_mapped_range()
        .unwrap()
        .chunks_exact(4)
        .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
        .collect();
    staging.unmap();
    out
}

#[test]
fn remove_and_despawn_clear_gpu_rows() {
    let Some((ctx, store, mut world)) = setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let removed = world.spawn();
    let despawned = world.spawn();
    for &e in &[removed, despawned] {
        world.insert(e, Packed { a: 7, b: 9 });
        world.insert(e, PerField { value: 5 });
    }
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row_words(&ctx, &store, "clear_test_packed", removed.index(), 2), [7, 9]);
    assert_eq!(row_words(&ctx, &store, "clear_test_field", despawned.index(), 1), [5]);

    world.remove::<Packed>(removed);
    world.remove::<PerField>(removed);
    world.despawn(despawned);
    world.flush_gpu_mirror(ctx.queue());
    for &e in &[removed, despawned] {
        assert_eq!(row_words(&ctx, &store, "clear_test_packed", e.index(), 2), [0, 0]);
        assert_eq!(row_words(&ctx, &store, "clear_test_field", e.index(), 1), [0]);
    }
}

#[test]
fn reinserting_in_the_same_frame_keeps_the_new_value() {
    let Some((ctx, store, mut world)) = setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let e = world.spawn();
    world.insert(e, Packed { a: 1, b: 2 });
    world.flush_gpu_mirror(ctx.queue());
    world.remove::<Packed>(e);
    world.insert(e, Packed { a: 3, b: 4 });
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row_words(&ctx, &store, "clear_test_packed", e.index(), 2), [3, 4]);
}

#[test]
fn erased_writes_reach_the_gpu_mirror() {
    let Some((ctx, store, mut world)) = setup() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let e = world.spawn();
    world.insert(e, Packed { a: 1, b: 2 });
    world.flush_gpu_mirror(ctx.queue());
    let id = pulsar_scenedb::component_id::<Packed>();
    world.get_dyn_mut(e, id).unwrap().downcast_mut::<Packed>().unwrap().b = 42;
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row_words(&ctx, &store, "clear_test_packed", e.index(), 2), [1, 42]);
}
