//! End-to-end proof of shared `#[gpu(buffer = "key")]` columns: two
//! different `#[derive(SceneStore)]` types whose `#[gpu]` fields declare the
//! SAME buffer key collapse onto ONE physical buffer (the store's
//! registry-collapse), instead of each allocating its own column.
//!
//! Before this feature, a second type declaring a shared key would have
//! silently replaced the first type's buffer (`HashMap::insert`), or —
//! depending on the path — registered a parallel buffer no one else knew
//! about. Now the store keeps a keyed owner table: the FIRST registration of
//! a key allocates the physical buffer, and every later same-key declaration
//! (same raw element type, size, access, mirror mode, and registration path)
//! ADOPTS the existing dispatch object. Incompatible declarations panic
//! loudly at registration instead of corrupting the pool at sync time.
//!
//! The same-shape-with-different-key control (`Solo`) proves sharing is key-
//! driven, not shape-driven: the disambiguation from `gpu_derive_field_collision.rs`
//! (a per-(struct, field) wrapper type) still gives every field a unique
//! ComponentId; the SHARING is the buffer-key layer on top of that.

use pulsar_scenedb::gpu::{
    BufferKey, CellSlot, EngineGpuContext, FrameDriver, GpuColumnSet, MirrorMode,
    RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::CellStorage;
use pulsar_scenedb_derive::SceneStore;

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
        label: Some("scenedb-gpu-shared-buffer-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

const ROW_CAPACITY: u32 = 64 * 2;

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig {
            capacity: 64,
            max_resident_cells: 2,
        }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
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

/// Both declare the SAME key with the SAME raw element type (f32) — the
/// shape that should collapse to one physical buffer.
#[derive(SceneStore, Clone, Copy)]
struct SharedA {
    #[gpu(buffer = "shared_pool")]
    value: f32,
}

#[derive(SceneStore, Clone, Copy)]
struct SharedB {
    #[gpu(buffer = "shared_pool")]
    value: f32,
}

/// Same shape as A/B but WITHOUT the shared key — must stay on its own
/// buffer (the disambiguation control).
#[derive(SceneStore, Clone, Copy)]
struct Solo {
    #[gpu]
    value: f32,
}

/// Same key as A/B but a DIFFERENT raw element type — an incompatible
/// declaration that must panic at registration.
#[derive(SceneStore, Clone, Copy)]
struct SharedBad {
    #[gpu(buffer = "shared_pool")]
    value: u32,
}

fn shared_field_id<T: GpuColumnSet>() -> pulsar_scenedb::component::ComponentId {
    let cols = T::gpu_columns();
    assert_eq!(cols.len(), 1, "test types carry exactly one #[gpu] field");
    cols[0].field_token.id()
}

#[test]
fn shared_key_fields_resolve_to_one_physical_buffer_and_writes_land_in_it() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());

    let a_id = shared_field_id::<SharedA>();
    let b_id = shared_field_id::<SharedB>();
    let solo_id = shared_field_id::<Solo>();

    // Registration order: A first (allocates "shared_pool"), then B (must
    // ADOPT A's buffer, not panic, not allocate a second one), then the
    // shape-identical-but-keyless control.
    SharedA::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
    SharedB::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
    Solo::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());

    // Distinct ComponentIds (per-(struct, field) disambiguation intact)...
    assert_ne!(a_id, b_id, "A and B are different types: different wrapper ComponentIds");
    assert_ne!(a_id, solo_id);
    // ...that the buffer-key layer resolves to ONE key for the sharers.
    assert_eq!(
        store.buffer_key_for(a_id),
        Some(BufferKey::of("shared_pool")),
        "A's field registered under the shared key"
    );
    assert_eq!(
        store.buffer_key_for(b_id),
        Some(BufferKey::of("shared_pool")),
        "B's field adopted the shared key, not a fresh one"
    );
    assert_ne!(
        store.buffer_key_for(solo_id),
        Some(BufferKey::of("shared_pool")),
        "keyless field got its own `{{Type}}::{{field}}` key, not the shared one"
    );

    // Exactly one owner exists for the shared key, sized to the row pool.
    let (shared_buf, element_count) = store
        .resolve_buffer(BufferKey::of("shared_pool"))
        .expect("shared key registered by A");
    assert_eq!(element_count, ROW_CAPACITY as u64, "one buffer holds the whole row pool");

    // Two cells, one per type, each consuming its own row slice of the ONE
    // shared buffer.
    let a_type = <SharedA as pulsar_scenedb::cell_type::SceneColumnSet>::cell_type();
    let b_type = <SharedB as pulsar_scenedb::cell_type::SceneColumnSet>::cell_type();
    let mut a_cell = CellStorage::from_cell_type(&a_type, 64).expect("a cell");
    let mut b_cell = CellStorage::from_cell_type(&b_type, 64).expect("b cell");
    let a_cell_id = store.register_cell(&a_cell, 0).expect("register a cell");
    let b_cell_id = store.register_cell(&b_cell, 0).expect("register b cell");
    assert_ne!(
        store.row_region_base(a_cell_id),
        store.row_region_base(b_cell_id),
        "cells own disjoint row slices of the shared pool"
    );

    let mut driver = FrameDriver::new();
    let sim = driver.begin();

    let a_handle = a_cell.alloc().expect("a slot");
    let b_handle = b_cell.alloc().expect("b slot");
    store.write_gpu(
        a_cell_id,
        &mut a_cell,
        a_handle,
        &SharedA { value: 1.5 },
        &sim,
    );
    store.write_gpu(
        b_cell_id,
        &mut b_cell,
        b_handle,
        &SharedB { value: 2.5 },
        &sim,
    );

    let mut cell_slots = [
        CellSlot { id: a_cell_id, cell: &mut a_cell },
        CellSlot { id: b_cell_id, cell: &mut b_cell },
    ];
    sim.end().end().end().run(&mut store, &mut cell_slots);

    // Both cells' rows are readable through the ONE shared buffer.
    let a_row = store.row_region_base(a_cell_id) as u64 * 4;
    let b_row = store.row_region_base(b_cell_id) as u64 * 4;
    let bytes = readback(&ctx, &shared_buf, a_row, 4);
    assert_eq!(
        f32::from_ne_bytes(bytes.try_into().unwrap()),
        1.5,
        "A's write landed in the shared buffer at A's row slice"
    );
    let bytes = readback(&ctx, &shared_buf, b_row, 4);
    assert_eq!(
        f32::from_ne_bytes(bytes.try_into().unwrap()),
        2.5,
        "B's write landed in the SAME shared buffer at B's row slice — collapse, not a parallel allocation"
    );
}

#[test]
fn shared_key_growable_registrations_collapse_to_one_world_buffer() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());

    let a_id = shared_field_id::<SharedA>();
    let b_id = shared_field_id::<SharedB>();

    // World-mirror (growable/dirty-tracked) path: same key, both types.
    SharedA::register_gpu_columns_growable(&mut store, 8, ctx.device());
    SharedB::register_gpu_columns_growable(&mut store, 8, ctx.device());

    assert_eq!(
        store.buffer_key_for(a_id),
        Some(BufferKey::of("shared_pool")),
        "growable A under the shared key"
    );
    assert_eq!(
        store.buffer_key_for(b_id),
        Some(BufferKey::of("shared_pool")),
        "growable B adopted the shared key"
    );

    // Both ComponentIds mark rows through their dirty-tracked dispatch
    // objects; if B had gotten its own buffer, its mark would flush into a
    // buffer nothing else can see.
    let a_bytes = 1.0f32.to_ne_bytes();
    let b_bytes = 2.0f32.to_ne_bytes();
    assert!(store.mark_gpu_row_dirty(a_id, 0, &a_bytes), "A registered dirty-tracked");
    assert!(store.mark_gpu_row_dirty(b_id, 1, &b_bytes), "B registered dirty-tracked");
    store.flush_gpu_mirror(ctx.queue());

    // resolve_buffer looks the key up in the owners table — the ONE buffer.
    let (shared_buf, _) = store
        .resolve_buffer(BufferKey::of("shared_pool"))
        .expect("shared key registered by A's growable path");
    let row0 = readback(&ctx, &shared_buf, 0, 4);
    let row1 = readback(&ctx, &shared_buf, 4, 4);
    assert_eq!(f32::from_ne_bytes(row0.try_into().unwrap()), 1.0, "A's row in the shared buffer");
    assert_eq!(f32::from_ne_bytes(row1.try_into().unwrap()), 2.0, "B's row in the SAME buffer");
}

#[test]
#[should_panic(expected = "element type mismatch")]
fn shared_key_with_different_element_type_panics_at_registration() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    SharedA::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
    // Same key "shared_pool", but the field's RAW type is u32, not f32 —
    // two components sharing a key must agree on element identity.
    SharedBad::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
}

#[test]
#[should_panic(expected = "may never span")]
fn same_key_across_fixed_and_growable_paths_panics() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // First registration is cell-mirrored (fixed)...
    SharedA::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
    // ...then the World-mirror (dirty-tracked/growable) path claims the same
    // key. One key, two registration paths = a config error, never a silent
    // alias.
    SharedB::register_gpu_columns_growable(&mut store, 8, ctx.device());
}
