//! Proves `#[gpu(layout = packed)]` end-to-end: every `#[gpu]` field on a
//! World-mirrored component lands in ONE interleaved GPU buffer (matching a
//! shader-facing packed struct, the motivating case being something shaped
//! like Helio's `GpuInstanceData`) instead of the default one-buffer-per-field
//! split -- and that the *cell-mirrored* path (`gpu_columns()`/`write_gpu`/
//! the fixed `register_gpu_columns`) is completely untouched by the
//! attribute, exactly as documented.

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
        label: Some("scenedb-gpu-packed-layout-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
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
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

/// Deliberately mixed field sizes/types and a non-#[gpu] field interleaved
/// in declaration order -- the exact shape that would break a naive
/// "read N contiguous bytes starting at the first #[gpu] field's offset"
/// implementation, and the reason the real implementation assembles the
/// packed value by field access instead of a raw byte-range read.
#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
struct PackedInstance {
    #[gpu]
    model: [f32; 16], // a full 4x4 transform -- the crate's one built-in Pod array type
    not_gpu_at_all: u32,
    #[gpu]
    mesh_id: u32,
    #[gpu]
    flags: u32,
}

/// Packed layout is 64 + 4 + 4 = 72 bytes: model([f32;16], 64B) + mesh_id
/// (u32, 4B) + flags (u32, 4B), in DECLARATION order among #[gpu] fields
/// only -- `not_gpu_at_all` contributes nothing, despite sitting between
/// `model` and `mesh_id` in the struct itself.
const PACKED_ROW_BYTES: u64 = 64 + 4 + 4;

#[test]
fn packed_fields_land_in_one_buffer_in_gpu_field_declaration_order() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    PackedInstance::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;
    let mut model = [0.0f32; 16];
    for (i, v) in model.iter_mut().enumerate() {
        *v = i as f32;
    }
    world.insert(
        entity,
        PackedInstance {
            model,
            not_gpu_at_all: 0xDEAD_BEEF, // must NOT appear anywhere in the packed buffer
            mesh_id: 42,
            flags: 7,
        },
    );

    let id = PackedInstance::packed_gpu_component_id();
    let mut bytes = Vec::new();
    store.with_growable_buffer_for_id(id, &mut |buf| {
        bytes = readback(&ctx, buf, row * PACKED_ROW_BYTES, PACKED_ROW_BYTES);
    });
    assert_eq!(bytes.len(), PACKED_ROW_BYTES as usize, "packed_gpu_component_id must resolve to a registered buffer");

    let mut got_model = [0.0f32; 16];
    for (i, v) in got_model.iter_mut().enumerate() {
        *v = f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let mesh_id = u32::from_ne_bytes(bytes[64..68].try_into().unwrap());
    let flags = u32::from_ne_bytes(bytes[68..72].try_into().unwrap());

    assert_eq!(got_model, model);
    assert_eq!(mesh_id, 42, "mesh_id must sit right after model, NOT after where not_gpu_at_all would be");
    assert_eq!(flags, 7);
    // The whole point: not_gpu_at_all's value (0xDEADBEEF) appears nowhere
    // in these 24 bytes -- already implied by the exact byte-for-byte
    // asserts above (there's no slot it could occupy without one of them
    // failing), but worth stating as the explicit claim being tested.
}

#[test]
fn packed_attribute_does_not_change_the_cell_mirrored_metadata_or_write_path() {
    // The whole scope guarantee this issue promised: gpu_columns() (used by
    // the cell-mirrored write_gpu and by register_gpu_columns, the FIXED,
    // non-growable registration) stays per-field, completely unaffected by
    // `#[gpu(layout = packed)]`.
    let columns = PackedInstance::gpu_columns();
    assert_eq!(columns.len(), 3, "gpu_columns() must still list every #[gpu] field individually");
    let names: Vec<&str> = columns.iter().map(|c| c.buffer_name).collect();
    assert!(names.contains(&"model"));
    assert!(names.contains(&"mesh_id"));
    assert!(names.contains(&"flags"));
}
