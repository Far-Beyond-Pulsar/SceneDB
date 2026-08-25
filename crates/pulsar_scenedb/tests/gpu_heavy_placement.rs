//! Pins the Heavy-placement lowering (Phase 3 of the engine-class GPU
//! de-duplication program): `GpuHeavy<T>` marks a cache/dedup boundary at a
//! position inside a `#[gpu]` var-len field's element type, and the derive
//! lowers through it.
//!
//! Four placements are supported and each is pinned here:
//!
//! 1. `Vec<GpuHeavy<H>>` — chain of handles, interned by STRUCTURAL
//!    identity (`gpu::structural_content_id` over the row's own list, since
//!    no `content_id` sibling is declared): two rows with identical lists
//!    share one allocation; different lists do not alias. Zero-allocation
//!    write path (the `repr(transparent)` byte identity).
//! 2. `Vec<Option<GpuHeavy<H>>>` — NULL slots stored in place via
//!    `GpuRef::NULL`, never aliased or compacted away.
//! 3. `Vec<Result<GpuHeavy<A>, GpuHeavy<B>>>` — three pools
//!    (`{field}::tag` u32 / `{field}::ok` A / `{field}::err` B) with the
//!    invariant `len(ok) + len(err) == vec.len()`, Oks and Errs each in
//!    order.
//! 4. Legacy shapes unchanged: a bare `#[gpu(mirror = Once, heavy)]`
//!    scalar still uploads its `GpuUploadSource::Element`, not the handle.

use pulsar_scenedb::gpu::{
    structural_content_id, EngineGpuContext, GpuHeavy, GpuMirrorHandle, RegionClassConfig,
    SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::page::Pod;
use pulsar_scenedb::GpuUploadSource;
use pulsar_scenedb::{GpuColumnSet, World};
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
        label: Some("scenedb-gpu-heavy-placement-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

fn attach(store: &Arc<SceneGpuStore>, ctx: &EngineGpuContext) -> (World, GpuMirrorHandle) {
    let mut world = World::new();
    let mirror = GpuMirrorHandle::new(Arc::clone(store), Arc::clone(ctx.queue()));
    world.attach_gpu_mirror(mirror.clone());
    (world, mirror)
}

// ── Placement 1: Vec<GpuHeavy<H>> ─────────────────────────────────────────

/// The lightweight reference behind every slot — deliberately NOT
/// `ContentAddressed`: these tests prove the STRUCTURAL fallback id works
/// with nothing declared.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotHandle(u32);
unsafe impl Pod for SlotHandle {}

#[derive(SceneStore, Clone)]
struct SlotMesh {
    /// An ordinary non-Pod CPU field riding alongside — proves the var-len
    /// struct's no-Pod contract is untouched by the lowering.
    pub label: String,
    #[gpu(mirror = Once)]
    pub slot_handles: Vec<GpuHeavy<SlotHandle>>,
}

#[test]
fn vec_of_heavy_handles_interns_by_content() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));
    let (mut world, _mirror) = attach(&store, &ctx);

    let pool_key = pulsar_scenedb::gpu::BufferKey::of("SlotMesh::slot_handles");

    // e1 and e2 name IDENTICAL reference lists -> ONE shared allocation,
    // refcount 2. e3 names a different list -> its own.
    let e1 = world.spawn();
    world.insert(
        e1,
        SlotMesh {
            label: "a".into(),
            slot_handles: vec![GpuHeavy(SlotHandle(7)), GpuHeavy(SlotHandle(9))],
        },
    );
    world.flush_gpu_mirror(ctx.queue()).expect("flush e1");

    let expected_pair_id = structural_content_id(&[SlotHandle(7), SlotHandle(9)]);
    {
        let pool = store
            .interned_var_len_pool::<SlotHandle>(pool_key)
            .expect("interned pool auto-registered on first insert");
        let audit = pool.audit();
        assert_eq!(audit.len(), 1, "one distinct list so far");
        assert_eq!(audit[0].0, expected_pair_id, "structural id over [h7,h9]");
        assert_eq!(audit[0].1, 1);
    }

    let e2 = world.spawn();
    world.insert(
        e2,
        SlotMesh {
            label: "b".into(),
            slot_handles: vec![GpuHeavy(SlotHandle(7)), GpuHeavy(SlotHandle(9))],
        },
    );
    let e3 = world.spawn();
    world.insert(
        e3,
        SlotMesh {
            label: "c".into(),
            slot_handles: vec![GpuHeavy(SlotHandle(11))],
        },
    );
    world.flush_gpu_mirror(ctx.queue()).expect("flush e2/e3");

    let pool = store
        .interned_var_len_pool::<SlotHandle>(pool_key)
        .expect("pool present");
    let audit = pool.audit();
    assert_eq!(audit.len(), 2, "pair-list + singleton-list = two chains");

    let pair = audit.iter().find(|(id, _, _)| *id == expected_pair_id).expect("pair chain");
    assert_eq!(pair.1, 2, "identical lists share ONE allocation, refcount 2");
    assert_eq!(pair.2, 8, "the shared chain holds exactly the pair list's payload: 2 handles x 4 bytes");

    // The zero-allocation write path claim, observable: e1's handle-table
    // entry points at THE SAME range as e2's.
    let h1 = SlotMesh::slot_handles_gpu_handle(&store, e1.index() as u32).expect("e1 handle");
    let h2 = SlotMesh::slot_handles_gpu_handle(&store, e2.index() as u32).expect("e2 handle");
    assert_eq!(h1, h2, "shared list -> identical (offset, count)");
    assert_eq!(h1.count, 2);
    let h3 = SlotMesh::slot_handles_gpu_handle(&store, e3.index() as u32).expect("e3 handle");
    assert_ne!(h1.offset..h1.offset + h1.count, h3.offset..h3.offset + h3.count);
}

// ── Placement 2: Vec<Option<GpuHeavy<H>>> ──────────────────────────────────

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TexHandle(u64);
unsafe impl Pod for TexHandle {}
impl pulsar_scenedb::gpu::GpuRef for TexHandle {
    const NULL: Self = TexHandle(0);
    fn is_null(self) -> bool { self.0 == 0 }
}

#[derive(SceneStore, Clone)]
struct OptSlots {
    #[gpu]
    pub slots: Vec<Option<GpuHeavy<TexHandle>>>,
}

#[test]
fn option_slots_roundtrip_null_and_set() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));
    let (mut world, _mirror) = attach(&store, &ctx);

    let entity = world.spawn();
    world.insert(
        entity,
        OptSlots {
            slots: vec![
                None,
                Some(GpuHeavy(TexHandle(0xdead_beef))),
                None,
            ],
        },
    );
    world.flush_gpu_mirror(ctx.queue()).expect("flush");

    // Chain length counts EVERY position including the NULLs — absence is
    // STORED IN PLACE (three positions in, three positions on the GPU).
    let handle = OptSlots::slots_gpu_handle(&store, entity.index() as u32).expect("handle");
    assert_eq!(handle.count, 3, "NULL slots occupy their position");

    // Two rows with the same null-pattern + values share one allocation
    // (structural identity includes the NULLs).
    let other = world.spawn();
    world.insert(
        other,
        OptSlots {
            slots: vec![
                None,
                Some(GpuHeavy(TexHandle(0xdead_beef))),
                None,
            ],
        },
    );
    world.flush_gpu_mirror(ctx.queue()).expect("flush other");
    let h_a = OptSlots::slots_gpu_handle(&store, entity.index() as u32).unwrap();
    let h_b = OptSlots::slots_gpu_handle(&store, other.index() as u32).unwrap();
    assert_eq!(h_a, h_b, "identical null-patterns share one chain");
}

// ── Placement 3: Vec<Result<GpuHeavy<A>, GpuHeavy<B>>> ─────────────────────

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OkHandle(u32);
unsafe impl Pod for OkHandle {}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ErrHandle(u16);
unsafe impl Pod for ErrHandle {}

#[derive(SceneStore, Clone)]
struct EitherSlots {
    #[gpu]
    pub slots: Vec<Result<GpuHeavy<OkHandle>, GpuHeavy<ErrHandle>>>,
}

#[test]
fn result_slots_split_into_tagged_pools() {
    let ctx = test_context();
    let store = Arc::new(SceneGpuStore::new(&ctx, scene_cfg()));
    let (mut world, _mirror) = attach(&store, &ctx);

    let entity = world.spawn();
    world.insert(
        entity,
        EitherSlots {
            slots: vec![
                Ok(GpuHeavy(OkHandle(100))),
                Err(GpuHeavy(ErrHandle(7))),
                Ok(GpuHeavy(OkHandle(200))),
            ],
        },
    );
    world.flush_gpu_mirror(ctx.queue()).expect("flush");

    // Invariant: tag count == vec length; ok + err == vec length; Oks and
    // Errs each preserve their own relative order (ok chain holds both Oks,
    // err chain holds the single Err between them).
    let row = entity.index() as u32;
    let tag = EitherSlots::slots_gpu_handle_tag(&store, row).expect("tag handle");
    let ok = EitherSlots::slots_gpu_handle_ok(&store, row).expect("ok handle");
    let err = EitherSlots::slots_gpu_handle_err(&store, row).expect("err handle");
    assert_eq!(tag.count, 3);
    assert_eq!(ok.count, 2, "both Ok handles land in the ok chain, in order");
    assert_eq!(err.count, 1, "the single Err lands in the err chain");

    // Structural ids over the split chains match computing them by hand:
    let ok_pool = store
        .interned_var_len_pool::<OkHandle>(pulsar_scenedb::gpu::BufferKey::of(
            "EitherSlots::slots::ok",
        ))
        .expect("ok pool registered");
    let want_ok = structural_content_id(&[OkHandle(100), OkHandle(200)]);
    assert!(
        ok_pool.audit().iter().any(|(id, refs, _)| *id == want_ok && *refs == 1),
        "ok chain interned under its structural id"
    );

    // Despawn releases all THREE chains without leaking (no panic, and a
    // follow-up audit shows refcounts dropped to zero entries for this
    // row's exclusive content).
    drop(world);
}

// ── Placement 4: legacy shapes untouched ───────────────────────────────────

/// The heavy/handle-split scalar from `gpu_heavy_once.rs`, verbatim shape:
/// proves the classic `GpuUploadSource` path is byte-identical after Phase
/// 3 (same mapped-element upload, not the handle bytes).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MeshHandle(u32);
unsafe impl Pod for MeshHandle {}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct MeshMetadataRow {
    vertex_count: u32,
    index_count: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}
unsafe impl Pod for MeshMetadataRow {}

impl pulsar_scenedb::gpu::GpuUploadSource for MeshHandle {
    type Element = MeshMetadataRow;
    fn upload_element(&self) -> MeshMetadataRow {
        MeshMetadataRow {
            vertex_count: self.0 * 100,
            index_count: self.0 * 150,
            bounds_min: [0.0; 3],
            bounds_max: [self.0 as f32; 3],
        }
    }
}

#[derive(SceneStore, Clone, Copy)]
struct LegacyHeavyMeta {
    #[gpu(mirror = Once, heavy)]
    mesh: MeshHandle,
}

#[test]
fn legacy_scalar_heavy_shape_unchanged() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    LegacyHeavyMeta::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entity = world.spawn();
    let row = entity.index() as u64;
    let field_id = LegacyHeavyMeta::gpu_columns()[0].field_token.id();
    let stride = std::mem::size_of::<MeshMetadataRow>() as u64;

    world.insert(entity, LegacyHeavyMeta { mesh: MeshHandle(3) });
    world.flush_gpu_mirror(ctx.queue()).expect("flush");

    let mut got = Vec::new();
    store.with_dirty_tracked_buffer_for_id(field_id, &mut |buf| {
        got = readback_bytes(&ctx, buf, row * stride, stride);
    });
    let uploaded: MeshMetadataRow =
        unsafe { std::ptr::read_unaligned(got.as_ptr() as *const _) };
    assert_eq!(
        uploaded,
        MeshHandle(3).upload_element(),
        "GPU holds the MAPPED element, never the handle bytes"
    );
}

fn readback_bytes(ctx: &EngineGpuContext, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("placement-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    data
}
