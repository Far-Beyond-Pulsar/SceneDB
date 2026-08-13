//! End-to-end proof that `SceneGpuStore`'s built-in buffers (generation,
//! slot-mirror, cell-metadata — beyond the transform/instance-info pair that
//! already registered) are reachable by key through
//! `SceneGpuStore::buffer_registry()`, and that the new diagnostic
//! `readback_row` convenience reads real, freshly-written bytes back from
//! them. Closes issue #41's "Store builtins... → `GpuBuffer<...>` @
//! `"builtin_transform"` / `"builtin_generation"`" collapse-map row for the
//! three builtins that previously had no keyed-registry path at all.

use pulsar_scenedb::gpu::{
    BufferKey, CellSlot, EngineGpuContext, FrameDriver, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore, CELL_METADATA_BUFFER_KEY, GENERATION_BUFFER_KEY, SLOT_MIRROR_BUFFER_KEY,
};
use pulsar_scenedb::{CellStorage, CellType, TypeToken};

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
        label: Some("scenedb-gpu-store-builtin-registry-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 32, max_resident_cells: 2 }],
        tombstone_headroom: 4,
        max_cells_metadata: 8,
    }
}

fn transform_cell(capacity: u32) -> CellStorage {
    let ct = CellType::new("builtin-registry-test")
        .with(TypeToken::of::<[f32; 16]>())
        .build()
        .unwrap();
    CellStorage::from_cell_type(&ct, capacity).unwrap()
}

#[test]
fn transform_and_instance_info_are_reachable_by_key_from_construction() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    let registry = store.buffer_registry();
    assert!(registry.contains_key(BufferKey::of("scenedb-instances")));
    assert!(registry.contains_key(BufferKey::of("scenedb-instance-info")));
}

#[test]
fn generation_slot_mirror_and_cell_metadata_builtins_are_reachable_by_key() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    let registry = store.buffer_registry();

    for key in [GENERATION_BUFFER_KEY, SLOT_MIRROR_BUFFER_KEY, CELL_METADATA_BUFFER_KEY] {
        let handle = registry.resolve(key).unwrap_or_else(|| panic!("{key:?} must resolve"));
        assert!(handle.buffer.size() > 0);
    }

    assert_eq!(
        registry.resolve(GENERATION_BUFFER_KEY).unwrap().buffer.size(),
        store.generation_buffer().size(),
        "registry entry is the SAME buffer generation_buffer() exposes"
    );
    assert_eq!(
        registry.resolve(SLOT_MIRROR_BUFFER_KEY).unwrap().buffer.size(),
        store.slot_mirror_buffer().size(),
    );
    assert_eq!(
        registry.resolve(CELL_METADATA_BUFFER_KEY).unwrap().buffer.size(),
        store.cell_metadata_buffer().size(),
    );
}

#[test]
fn readback_row_reads_back_a_freshly_written_generation() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    let mut cell = transform_cell(32);
    let cell_id = store.register_cell(&cell, 0).expect("register cell");

    let mut driver = FrameDriver::new();
    let sim = driver.begin();
    let handle = cell.alloc().expect("slot");
    store.write_transform(cell_id, &mut cell, handle, &[1.0; 16], &sim);
    let mut cell_slots = [CellSlot { id: cell_id, cell: &mut cell }];
    sim.end().end().end().run(&mut store, &mut cell_slots);

    let got: u32 = store
        .readback_row(ctx.device(), GENERATION_BUFFER_KEY, handle.index())
        .expect("generation key resolves as a row buffer");
    assert_eq!(got, handle.generation(), "readback_row reads the real, freshly-written generation");
}

#[test]
fn readback_row_returns_none_for_a_key_that_is_not_a_row_buffer() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    assert!(store.readback_row::<u32>(ctx.device(), BufferKey::of("never-registered"), 0).is_none());
}
