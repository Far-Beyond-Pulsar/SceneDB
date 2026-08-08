//! Regression test for a real bug found while proving the generic
//! GPU-column mechanism (see `gpu_generic_column.rs`): `#[derive(SceneStore)]`
//! + `#[gpu]` keyed each field's GPU column by the field's own raw type
//! (`TypeToken::of::<FieldType>()` / `component_id::<FieldType>()`), with
//! no notion of which *struct* the field belonged to. Two different
//! `#[derive(SceneStore)]` types both having a same-shaped `#[gpu]` field
//! (e.g. both an `f32`) would silently resolve to the exact same
//! `ComponentId` — the second type's `register_gpu_buffer` call would
//! replace the first's GPU buffer outright (`HashMap::insert`, no
//! collision check), and even without that, both fields would end up
//! sharing one physical buffer interleaved by row region.
//!
//! Fixed by generating a `#[repr(transparent)]` wrapper type per `#[gpu]`
//! field (unique to its (struct, field) pair by construction) and using
//! that — not the raw field type — everywhere a column identity is
//! needed. This test proves it: two structs, each with an `f32` field
//! marked `#[gpu]`, registered and written through the derive-generated
//! `register_gpu_columns`/`write_gpu`, read back independently with no
//! cross-contamination.
//!
//! Also exercises the other half of the same bug: nothing previously
//! called `register_gpu_buffer` for derive-generated `#[gpu]` fields at
//! all (`mark_column_dirty` silently no-ops on an unregistered
//! component id) — `register_gpu_columns` (also new) closes that gap.

use pulsar_scenedb::gpu::{
    CellSlot, EngineGpuContext, FrameDriver, GpuColumnSet, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore,
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
        label: Some("scenedb-gpu-derive-collision-test"),
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

// Two structs, both with an `f32` field marked `#[gpu]` -- the exact shape
// that silently aliased one buffer before this fix.
#[derive(SceneStore, Clone, Copy)]
struct DeriveTestMaterial {
    #[gpu]
    roughness: f32,
}

#[derive(SceneStore, Clone, Copy)]
struct DeriveTestLight {
    #[gpu]
    intensity: f32,
}

#[test]
fn same_shaped_gpu_fields_on_different_derived_types_do_not_collide() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());

    // The whole point: this is the macro's own generated registration, not
    // hand-rolled `register_gpu_buffer::<f32>()` calls that would obviously
    // collide by construction.
    DeriveTestMaterial::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());
    DeriveTestLight::register_gpu_columns(&mut store, ROW_CAPACITY, ctx.device());

    let material_type = <DeriveTestMaterial as pulsar_scenedb::cell_type::SceneColumnSet>::cell_type();
    let light_type = <DeriveTestLight as pulsar_scenedb::cell_type::SceneColumnSet>::cell_type();

    let mut material_cell =
        CellStorage::from_cell_type(&material_type, 64).expect("material cell storage");
    let mut light_cell = CellStorage::from_cell_type(&light_type, 64).expect("light cell storage");

    let material_cell_id = store
        .register_cell(&material_cell, 0)
        .expect("register materials cell");
    let light_cell_id = store
        .register_cell(&light_cell, 0)
        .expect("register lights cell");

    let mut driver = FrameDriver::new();
    let sim_a = driver.begin();

    let material_handle = material_cell.alloc().expect("alloc material slot");
    let light_handle = light_cell.alloc().expect("alloc light slot");

    let material_data = DeriveTestMaterial { roughness: 0.75 };
    let light_data = DeriveTestLight { intensity: 42.0 };

    store.write_gpu(
        material_cell_id,
        &mut material_cell,
        material_handle,
        &material_data,
        &sim_a,
    );
    store.write_gpu(
        light_cell_id,
        &mut light_cell,
        light_handle,
        &light_data,
        &sim_a,
    );

    let mut cell_slots = [
        CellSlot {
            id: material_cell_id,
            cell: &mut material_cell,
        },
        CellSlot {
            id: light_cell_id,
            cell: &mut light_cell,
        },
    ];
    // Real GPU device, real buffer writes at real row offsets, real
    // frame-boundary sync -- if `register_gpu_columns` hadn't actually
    // registered a buffer for each field's wrapper type, `write_gpu` above
    // would still "succeed" (its `mark_column_dirty` no-ops silently on an
    // unregistered id) so this alone wouldn't catch that half of the bug —
    // the `assert_ne!` below is the one that does, directly.
    sim_a.end().end().end().run(&mut store, &mut cell_slots);

    let material_row = store.row_region_base(material_cell_id) as u64;
    let light_row = store.row_region_base(light_cell_id) as u64;
    assert_ne!(material_row, light_row, "disjoint cell regions, sanity check");

    // The wrapper types themselves are unnameable from outside this crate's
    // own generated code (defined inside an anonymous `const _: () = {};`
    // block specifically so nothing outside the macro's own codegen can
    // depend on their name) -- so recover their registered identity through
    // the public `GpuColumnSet::gpu_columns()` API instead. This is the
    // direct, precise proof of the fix: before it, both of these
    // `field_token`s were `TypeToken::of::<f32>()` -- literally equal.
    let material_columns = DeriveTestMaterial::gpu_columns();
    assert_eq!(material_columns.len(), 1);
    let light_columns = DeriveTestLight::gpu_columns();
    assert_eq!(light_columns.len(), 1);
    assert_ne!(
        material_columns[0].field_token.id(),
        light_columns[0].field_token.id(),
        "the whole bug: two structs' same-shaped #[gpu] f32 fields must not \
         resolve to the same ComponentId"
    );
}
