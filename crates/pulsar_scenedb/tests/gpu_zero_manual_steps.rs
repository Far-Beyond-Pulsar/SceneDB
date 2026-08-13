//! The end-to-end proof for issue #41's "no manual steps at all": a
//! `#[gpu]`-bearing component, inserted into a `World` with a mirror
//! attached, driven PURELY through `SceneDb::step()`/`step_gpu()` —
//! this test never calls `T::register_gpu_columns_growable` or
//! `world.flush_gpu_mirror` itself. Neither needs a manual call anymore:
//! registration happens on the type's first insert (`gpu::world_mirror`'s
//! per-type dispatch function), and flush happens automatically inside
//! BOTH `SceneDb::step` and `SceneDb::step_gpu` (`scene_db.rs`) — so a
//! host whose GPU-mirrored state is entirely `World`-side needs only
//! `step()`, never `step_gpu`, for full automatic sync (see
//! `step_alone_is_enough_for_a_host_with_no_cell_mirrored_store` below).

use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{GpuColumnSet, SceneDb};
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
        label: Some("scenedb-zero-manual-steps-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

fn empty_scene_cfg() -> SceneGpuConfig {
    // step_gpu's cell-mirrored path needs SOME store even though this test
    // has zero cells — smallest legal config.
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 1, max_resident_cells: 1 }],
        tombstone_headroom: 1,
        max_cells_metadata: 1,
    }
}

/// No `buffer = "..."`, no `mirror = ...` override — the plain, default
/// `#[gpu]` shape every ordinary World-mirrored component uses. Critically:
/// NOTHING in this test file ever calls
/// `AutoRegistered::register_gpu_columns_growable`.
#[derive(SceneStore, Clone, Copy)]
struct AutoRegistered {
    #[gpu]
    value: u32,
}

#[test]
fn a_never_manually_registered_component_lands_on_the_gpu_through_scenedb_alone() {
    let ctx = test_context();

    // Two SEPARATE stores by construction (see SceneDb::step_gpu's doc):
    // one Arc'd for the World mirror, one owned for step_gpu's cell path.
    let world_store = Arc::new(SceneGpuStore::new(&ctx, empty_scene_cfg()));
    let mut cell_store = SceneGpuStore::new(&ctx, empty_scene_cfg());

    let mut db = SceneDb::new();
    db.world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&world_store), Arc::clone(ctx.queue())));

    // Sanity: confirm nothing is registered yet — proves what follows is
    // genuinely automatic, not a no-op against pre-existing registration.
    let field_id = AutoRegistered::gpu_columns()[0].field_token.id();
    assert!(world_store.buffer_key_for(field_id).is_none(), "must NOT be registered before the first insert");

    let entity = db.world.spawn();
    // The ONLY GPU-relevant call in this entire test: an ordinary insert.
    // No register_gpu_columns_growable call anywhere in this file.
    db.world.insert(entity, AutoRegistered { value: 0xC0FFEE });

    // Auto-registration already happened (inside the insert's dispatch),
    // before any step_gpu call — prove that separately from the flush.
    assert!(world_store.buffer_key_for(field_id).is_some(), "insert alone must auto-register the buffer");

    // Drive the CPU phase (no subsystems registered — a no-op, included for
    // completeness of "the normal loop") and the GPU phase. `step_gpu` is
    // the ONLY place this test asks for anything GPU-flush-related — no
    // `world.flush_gpu_mirror` call anywhere in this file.
    db.step();
    db.step_gpu(&mut cell_store, &mut []);

    let handle = world_store.resolve_buffer_handle(world_store.buffer_key_for(field_id).unwrap()).expect("resolvable");
    let got: u32 = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());
    assert_eq!(got, 0xC0FFEE, "value must have reached the GPU with zero manual register/flush calls");
}

#[test]
fn new_with_gpu_mirror_wires_the_mirror_at_construction_with_no_separate_attach_call() {
    let ctx = test_context();
    let world_store = Arc::new(SceneGpuStore::new(&ctx, empty_scene_cfg()));

    // The mirror is passed straight to the constructor -- no
    // `db.world.attach_gpu_mirror(..)` call anywhere in this test.
    let mut db = SceneDb::new_with_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&world_store), Arc::clone(ctx.queue())));
    assert!(db.world.has_gpu_mirror(), "mirror must already be attached right out of the constructor");

    let entity = db.world.spawn();
    db.world.insert(entity, AutoRegistered { value: 0xABCD });
    db.step();

    let field_id = AutoRegistered::gpu_columns()[0].field_token.id();
    let handle = world_store.resolve_buffer_handle(world_store.buffer_key_for(field_id).unwrap()).expect("resolvable");
    let got: u32 = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());
    assert_eq!(got, 0xABCD, "construct-and-go: new_with_gpu_mirror + step() is the entire setup");
}

#[test]
fn step_alone_is_enough_for_a_host_with_no_cell_mirrored_store() {
    // The key new capability: a host that never touches the cell-mirrored
    // path at all -- no SceneGpuStore for step_gpu, no CellSlots, no call
    // to step_gpu whatsoever -- still gets fully automatic GPU sync from
    // `step()` alone. This is what "not GPU specific" means in practice:
    // the host's frame loop calls one generic method and never needs to
    // know a GPU concept exists unless it opts into cell-mirroring too.
    let ctx = test_context();
    let world_store = Arc::new(SceneGpuStore::new(&ctx, empty_scene_cfg()));

    let mut db = SceneDb::new();
    db.world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&world_store), Arc::clone(ctx.queue())));

    let entity = db.world.spawn();
    db.world.insert(entity, AutoRegistered { value: 0xFACE });

    // ONLY step() -- step_gpu is never called in this test at all.
    db.step();

    let field_id = AutoRegistered::gpu_columns()[0].field_token.id();
    let handle = world_store.resolve_buffer_handle(world_store.buffer_key_for(field_id).unwrap()).expect("resolvable");
    let got: u32 = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, entity.index());
    assert_eq!(got, 0xFACE, "step() alone must register AND flush -- no step_gpu call anywhere in this test");
}

#[test]
fn a_second_insert_after_the_first_step_gpu_is_also_flushed_automatically() {
    let ctx = test_context();
    let world_store = Arc::new(SceneGpuStore::new(&ctx, empty_scene_cfg()));
    let mut cell_store = SceneGpuStore::new(&ctx, empty_scene_cfg());

    let mut db = SceneDb::new();
    db.world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&world_store), Arc::clone(ctx.queue())));

    let field_id = AutoRegistered::gpu_columns()[0].field_token.id();
    let e1 = db.world.spawn();
    db.world.insert(e1, AutoRegistered { value: 111 });
    db.step_gpu(&mut cell_store, &mut []);

    let e2 = db.world.spawn();
    db.world.insert(e2, AutoRegistered { value: 222 });
    db.step_gpu(&mut cell_store, &mut []);

    let handle = world_store.resolve_buffer_handle(world_store.buffer_key_for(field_id).unwrap()).expect("resolvable");
    let got1: u32 = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, e1.index());
    let got2: u32 = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, e2.index());
    assert_eq!(got1, 111);
    assert_eq!(got2, 222);
}

#[test]
fn manual_registration_still_wins_when_a_caller_wants_a_custom_capacity() {
    // Auto-registration must never clobber a caller who registered ahead of
    // time on purpose (e.g. to move the first-growth cost off the insert
    // path). Confirms the "manual registration wins" contract documented on
    // DEFAULT_AUTO_REGISTER_CAPACITY.
    let ctx = test_context();
    let world_store = Arc::new(SceneGpuStore::new(&ctx, empty_scene_cfg()));
    let mut cell_store = SceneGpuStore::new(&ctx, empty_scene_cfg());

    // Manual registration with a deliberately large capacity, BEFORE any insert.
    AutoRegistered::register_gpu_columns_growable(&world_store, 4096, ctx.device());

    let mut db = SceneDb::new();
    db.world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&world_store), Arc::clone(ctx.queue())));
    let entity = db.world.spawn();
    db.world.insert(entity, AutoRegistered { value: 7 });
    db.step_gpu(&mut cell_store, &mut []);

    let field_id = AutoRegistered::gpu_columns()[0].field_token.id();
    let key = world_store.buffer_key_for(field_id).expect("registered");
    let (_buf, element_count) = world_store.resolve_buffer(key).expect("resolvable");
    assert_eq!(
        element_count, 4096,
        "manual registration's custom capacity must survive — auto-registration must not have re-registered over it"
    );
}
