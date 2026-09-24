//! `BufferHandle::row_capacity` reports a packed row buffer's live capacity,
//! so consumers size their loops from the buffer instead of a constant that
//! goes stale when the buffer grows.

use pulsar_scenedb::gpu::{BufferKey, GpuColumnSet, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "row_capacity_rows")]
struct Row {
    #[gpu]
    model: [f32; 16],
    #[gpu]
    tag: u32,
}

/// Packed stride: the `#[gpu]` fields in declaration order.
const ROW_BYTES: u64 = 64 + 4;

#[test]
fn row_capacity_follows_growth() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let ctx = EngineGpuContext::new(Arc::new(device), Arc::new(queue));
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig { classes: vec![], tombstone_headroom: 0, max_cells_metadata: 0 },
    );
    Row::register_gpu_columns_growable(&mut store, 4, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(store.clone(), ctx.queue().clone()));

    let key = BufferKey::of("row_capacity_rows");
    let handle = store.resolve_buffer_handle(key).expect("registered");
    assert_eq!(handle.row_bytes, ROW_BYTES);
    assert!(handle.row_capacity() >= 4);

    for i in 0..100 {
        let e = world.spawn();
        world.insert(e, Row { model: [i as f32; 16], tag: i });
    }
    world.flush_gpu_mirror(ctx.queue());
    let grown = store.resolve_buffer_handle(key).expect("registered");
    assert!(grown.row_capacity() >= 100, "capacity {} did not follow growth", grown.row_capacity());
    assert_eq!(grown.buffer.size() / grown.row_bytes, grown.row_capacity() as u64);
}

#[test]
fn content_generation_tracks_uploaded_writes_only() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let ctx = EngineGpuContext::new(Arc::new(device), Arc::new(queue));
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig { classes: vec![], tombstone_headroom: 0, max_cells_metadata: 0 },
    );
    Row::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(store.clone(), ctx.queue().clone()));
    let key = BufferKey::of("row_capacity_rows");
    let generation = || store.resolve_buffer_handle(key).unwrap().content_generation;

    let e = world.spawn();
    world.insert(e, Row { model: [0.0; 16], tag: 1 });
    world.flush_gpu_mirror(ctx.queue());
    let after_insert = generation();

    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(generation(), after_insert, "a flush with nothing dirty is not a write");
    let _ = world.get::<Row>(e);
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(generation(), after_insert, "reading is not a write");

    world.get_mut::<Row>(e).unwrap().tag = 2;
    world.flush_gpu_mirror(ctx.queue());
    assert!(generation() > after_insert, "an in-place edit must bump the generation");
}
