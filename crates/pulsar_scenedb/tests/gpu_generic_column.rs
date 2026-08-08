//! Proves the generic GPU-column mechanism — `SceneGpuStore::
//! register_gpu_buffer<T>` + a hand-written `GpuColumnSet` treating a whole
//! caller-defined Pod struct as one column, the same shape
//! `write_transform`/`write_instance_info` use internally for their two
//! hardcoded built-in types (`[f32;16]`, `InstanceInfo`) — actually works
//! end-to-end for arbitrary caller types, with a real GPU device, through a
//! full frame boundary.
//!
//! Before this test, this path had **zero** coverage anywhere in the crate:
//! `register_gpu_buffer` and `GpuColumnSet` were reachable, compiled, and
//! never exercised by anything outside the two built-ins.
//!
//! Two independently-registered types are used specifically to prove there
//! is no accidental collision between them: `ComponentId` (this crate's GPU
//! buffer key) is derived from `TypeId`, globally, per Rust type — so two
//! *different* struct types can never collide with each other's whole-struct
//! column (each type's own `TypeId` is unique by construction), which is
//! the reason the manual "whole struct = one column" `GpuColumnSet` impl
//! below is the correct generic mechanism for something like a material or
//! light row — as opposed to `#[derive(SceneStore)]`'s per-*field*
//! splitting, where two different structs sharing a same-shaped primitive
//! field (e.g. both having an `[f32;4]` field) legitimately *do* share one
//! physical column, by design (that derive targets CPU records that mirror
//! only some of their fields to the GPU, not "this whole type is a GPU row").

use pulsar_scenedb::gpu::{
    CellId, CellSlot, EngineGpuContext, FrameDriver, GpuColumnDesc, GpuColumnSet, MirrorMode,
    RegionClassConfig, SceneGpuConfig, SceneGpuStore, SimulateWitness,
};
use pulsar_scenedb::{component_id, CellStorage, CellType, Handle, Pod, TypeToken};

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
        label: Some("scenedb-generic-column-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
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

// `row_offset` (the total per-column GPU buffer size every `register_gpu_buffer`
// call must match -- see `SceneGpuStore::new`'s own use of it for the two
// built-in columns) is `capacity * max_resident_cells`, summed across
// classes: one class, capacity 64, exactly 2 resident cells (this test
// registers exactly 2: materials, lights) = 128.
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

/// Stand-in for a material row — deliberately shaped nothing like
/// [`TestLight`] below, so a field-level collision would be implausible
/// even by accident (the point being proven is type-level isolation, not
/// merely "these two happen not to overlap").
#[repr(C)]
#[derive(Clone, Copy)]
struct TestMaterial {
    base_color: [f32; 4],
    roughness: f32,
    metallic: f32,
    _pad: [f32; 2],
}

// SAFETY: #[repr(C)], every field is itself Pod (f32/[f32;N]), no padding
// bytes are ever read as anything but the struct's own declared fields.
unsafe impl Pod for TestMaterial {}

impl GpuColumnSet for TestMaterial {
    fn gpu_columns() -> Vec<GpuColumnDesc> {
        vec![GpuColumnDesc {
            field_token: TypeToken::of::<TestMaterial>(),
            field_offset: 0,
            mode: MirrorMode::DirtyTracked,
            buffer_name: "test-materials",
        }]
    }

    fn write_gpu(
        store: &SceneGpuStore,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &Self,
        _phase: &impl SimulateWitness,
    ) {
        let row = cell.row_of(handle).expect("valid handle");
        let col = cell
            .column_for_mut::<TestMaterial>()
            .expect("cell has no TestMaterial column");
        col[row as usize] = *data;
        store.mark_column_dirty(id, component_id::<TestMaterial>(), row);
    }
}

/// Stand-in for a light row.
#[repr(C)]
#[derive(Clone, Copy)]
struct TestLight {
    position: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    _pad: f32,
}

// SAFETY: see TestMaterial's impl above -- same reasoning.
unsafe impl Pod for TestLight {}

impl GpuColumnSet for TestLight {
    fn gpu_columns() -> Vec<GpuColumnDesc> {
        vec![GpuColumnDesc {
            field_token: TypeToken::of::<TestLight>(),
            field_offset: 0,
            mode: MirrorMode::DirtyTracked,
            buffer_name: "test-lights",
        }]
    }

    fn write_gpu(
        store: &SceneGpuStore,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &Self,
        _phase: &impl SimulateWitness,
    ) {
        let row = cell.row_of(handle).expect("valid handle");
        let col = cell
            .column_for_mut::<TestLight>()
            .expect("cell has no TestLight column");
        col[row as usize] = *data;
        store.mark_column_dirty(id, component_id::<TestLight>(), row);
    }
}

#[test]
fn two_independently_registered_custom_types_round_trip_without_cross_contamination() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());

    // The exact call shape SceneGpuStore::new uses internally for the two
    // built-ins ([f32;16] transforms, InstanceInfo) -- proving it works
    // identically for caller-defined types is the whole point.
    store.register_gpu_buffer::<TestMaterial>(ROW_CAPACITY, ctx.device(), "test-materials");
    store.register_gpu_buffer::<TestLight>(ROW_CAPACITY, ctx.device(), "test-lights");

    let material_type = CellType::new("materials")
        .with(TypeToken::of::<TestMaterial>())
        .build()
        .expect("valid cell type");
    let light_type = CellType::new("lights")
        .with(TypeToken::of::<TestLight>())
        .build()
        .expect("valid cell type");

    let mut material_cell =
        CellStorage::from_cell_type(&material_type, 64).expect("material cell storage");
    let mut light_cell = CellStorage::from_cell_type(&light_type, 64).expect("light cell storage");

    let material_cell_id = store
        .register_cell(&material_cell, 0)
        .expect("register materials cell");
    let light_cell_id = store
        .register_cell(&light_cell, 0)
        .expect("register lights cell");
    // Disjoint GPU regions -- same guarantee every multi-cell SceneGpuStore
    // test in this crate relies on (see gpu_harvest.rs).
    assert_ne!(
        store.row_region_base(material_cell_id),
        store.row_region_base(light_cell_id),
    );

    let mut driver = FrameDriver::new();
    let sim_a = driver.begin();

    let material_handle = material_cell.alloc().expect("alloc material slot");
    let light_handle = light_cell.alloc().expect("alloc light slot");

    let material_data = TestMaterial {
        base_color: [1.0, 0.0, 0.0, 1.0],
        roughness: 0.5,
        metallic: 0.1,
        _pad: [0.0; 2],
    };
    let light_data = TestLight {
        position: [1.0, 2.0, 3.0],
        intensity: 10.0,
        color: [0.0, 1.0, 0.0],
        _pad: 0.0,
    };

    assert!(store.write_gpu(
        material_cell_id,
        &mut material_cell,
        material_handle,
        &material_data,
        &sim_a,
    ));
    assert!(store.write_gpu(
        light_cell_id,
        &mut light_cell,
        light_handle,
        &light_data,
        &sim_a,
    ));

    // Drive the full frame boundary so the dirty rows actually reach VRAM.
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
    sim_a.end().end().end().run(&mut store, &mut cell_slots);

    // Read back each buffer independently and check it holds exactly its
    // own type's data -- proving neither registration stomped the other's
    // buffer nor left it holding the wrong bytes.
    let material_buf = store
        .buffer_for::<TestMaterial>()
        .expect("material buffer registered");
    // Row-region-based offset: material_cell's rows land at
    // row_region_base(material_cell_id).. within the shared per-class row
    // space, same as every other registered column (transform, InstanceInfo,
    // or this test's types) -- there's no special-casing for custom types.
    let material_row = store.row_region_base(material_cell_id) as u64;
    let material_bytes = readback(
        &ctx,
        material_buf,
        material_row * std::mem::size_of::<TestMaterial>() as u64,
        std::mem::size_of::<TestMaterial>() as u64,
    );
    // SAFETY: `material_bytes` holds exactly size_of::<TestMaterial>() bytes
    // read back from a buffer this test itself wrote a real TestMaterial
    // into; Vec<u8>'s allocation isn't guaranteed aligned for TestMaterial,
    // hence read_unaligned rather than a direct cast/read.
    let material_readback: TestMaterial =
        unsafe { std::ptr::read_unaligned(material_bytes.as_ptr() as *const TestMaterial) };
    assert_eq!(material_readback.base_color, material_data.base_color);
    assert_eq!(material_readback.roughness, material_data.roughness);
    assert_eq!(material_readback.metallic, material_data.metallic);

    let light_buf = store
        .buffer_for::<TestLight>()
        .expect("light buffer registered");
    let light_row = store.row_region_base(light_cell_id) as u64;
    let light_bytes = readback(
        &ctx,
        light_buf,
        light_row * std::mem::size_of::<TestLight>() as u64,
        std::mem::size_of::<TestLight>() as u64,
    );
    // SAFETY: same reasoning as the material readback above.
    let light_readback: TestLight =
        unsafe { std::ptr::read_unaligned(light_bytes.as_ptr() as *const TestLight) };
    assert_eq!(light_readback.position, light_data.position);
    assert_eq!(light_readback.intensity, light_data.intensity);
    assert_eq!(light_readback.color, light_data.color);

    // A type nobody registered has no buffer at all -- confirms buffer_for
    // doesn't silently alias onto an unrelated column.
    assert!(store.buffer_for::<u64>().is_none());
}
