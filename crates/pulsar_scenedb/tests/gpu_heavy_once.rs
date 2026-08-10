//! Proves the `#[gpu(mirror = Once, heavy)]` handle/heavy-element split
//! (`GpuUploadSource`) end to end: the CPU column stays handle-sized (a
//! `u32`-wrapping `MeshHandle`), but the GPU buffer holds the much larger
//! `MeshMetadataRow` produced by the handle's `upload_element()` — never the
//! handle's own raw bytes.
//!
//! Two things are proven, matching the design spec's two invariants for
//! `Once`:
//! 1. `World::insert` uploads the MAPPED element, not the handle, and a
//!    routine re-insert still leaves it pinned (unchanged from plain
//!    `Once`'s existing behavior — the handle/heavy split doesn't change
//!    WHEN a write happens, only WHAT bytes get written).
//! 2. An explicit `World::get_mut` mutation of the handle DOES re-upload —
//!    re-running the mapper against the NEW handle value — because that's
//!    the caller deliberately changing the value (see `tests/world_gpu_
//!    mirror_get_mut.rs` for the same contract on non-heavy fields).

use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuColumnSet, GpuMirrorHandle, GpuUploadSource, RegionClassConfig,
    SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::page::Pod;
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
        label: Some("scenedb-gpu-heavy-once-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback(ctx: &EngineGpuContext, buf: &wgpu::Buffer, offset: u64, bytes: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, offset, &staging, 0, bytes);
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

/// The lightweight CPU handle -- 4 bytes, nowhere near the size of the
/// element it maps to.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MeshHandle(u32);
unsafe impl Pod for MeshHandle {}

/// The heavy GPU-resident record -- 32 bytes, only ever assembled by
/// `upload_element`, never held by the CPU column.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct MeshMetadataRow {
    vertex_count: u32,
    index_count: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}
unsafe impl Pod for MeshMetadataRow {}

impl GpuUploadSource for MeshHandle {
    type Element = MeshMetadataRow;
    fn upload_element(&self) -> MeshMetadataRow {
        // Deterministic stand-in for "look this mesh id up in the asset
        // system and build its GPU metadata row" -- what matters for this
        // test is that it's derived from the handle, not from any state
        // held on the CPU column itself.
        MeshMetadataRow {
            vertex_count: self.0 * 100,
            index_count: self.0 * 150,
            bounds_min: [0.0, 0.0, 0.0],
            bounds_max: [self.0 as f32, self.0 as f32, self.0 as f32],
        }
    }
}

fn row_bytes(row: &MeshMetadataRow) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(row as *const MeshMetadataRow as *const u8, std::mem::size_of::<MeshMetadataRow>())
            .to_vec()
    }
}

#[derive(SceneStore, Clone, Copy)]
struct StaticMeshMeta {
    #[gpu(mirror = Once, heavy)]
    mesh: MeshHandle,
}

#[test]
fn heavy_once_field_uploads_the_mapped_element_not_the_handle_bytes() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    StaticMeshMeta::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;
    let field_id = StaticMeshMeta::gpu_columns()[0].field_token.id();
    let stride = std::mem::size_of::<MeshMetadataRow>() as u64;

    world.insert(entity, StaticMeshMeta { mesh: MeshHandle(3) });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let mut got = Vec::new();
    store.with_dirty_tracked_buffer_for_id(field_id, &mut |buf| {
        got = readback(&ctx, buf, row * stride, stride);
    });
    assert_eq!(
        got,
        row_bytes(&MeshHandle(3).upload_element()),
        "GPU buffer must hold the MAPPED heavy element, not MeshHandle(3)'s own 4 raw bytes"
    );

    // The CPU side stays the handle -- instant read, no mapper involved.
    assert_eq!(world.get::<StaticMeshMeta>(entity).unwrap().mesh, MeshHandle(3));

    // A routine re-insert (not get_mut) must NOT re-upload -- Once's
    // pinning is unchanged by the handle/heavy split.
    world.insert(entity, StaticMeshMeta { mesh: MeshHandle(999) });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    let mut got_after = Vec::new();
    store.with_dirty_tracked_buffer_for_id(field_id, &mut |buf| {
        got_after = readback(&ctx, buf, row * stride, stride);
    });
    assert_eq!(
        got_after,
        row_bytes(&MeshHandle(3).upload_element()),
        "a routine re-insert must NOT re-run the mapper, even though the CPU handle changed"
    );
}

#[test]
fn get_mut_on_a_heavy_once_field_reruns_the_mapper_against_the_new_handle() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    StaticMeshMeta::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;
    let field_id = StaticMeshMeta::gpu_columns()[0].field_token.id();
    let stride = std::mem::size_of::<MeshMetadataRow>() as u64;

    world.insert(entity, StaticMeshMeta { mesh: MeshHandle(3) });
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    {
        let mut c = world.get_mut::<StaticMeshMeta>(entity).expect("present");
        c.mesh = MeshHandle(11);
    }
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let mut got = Vec::new();
    store.with_dirty_tracked_buffer_for_id(field_id, &mut |buf| {
        got = readback(&ctx, buf, row * stride, stride);
    });
    assert_eq!(
        got,
        row_bytes(&MeshHandle(11).upload_element()),
        "get_mut must re-run the mapper against the NEW handle and upload the result"
    );
    assert_eq!(world.get::<StaticMeshMeta>(entity).unwrap().mesh, MeshHandle(11));
}
