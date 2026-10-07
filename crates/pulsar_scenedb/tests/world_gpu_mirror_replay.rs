//! Attaching a GPU mirror to an already-populated `World` initializes it
//! from the world's contents, generically: every `#[gpu]`-bearing
//! component present at attach time is written exactly as its first insert
//! would have been (per-field, packed, `Once`-mode and var-len fields, plus
//! the liveness mirror), with buffers auto-registered. Nothing is re-inserted
//! by the caller and no list of types is involved. Erased inserts reach the
//! mirror the same way typed ones do.
//!
//! Needs a GPU adapter (any Vulkan device, including lavapipe); skips
//! without one.

use pulsar_scenedb::gpu::{BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::{component_id, register_component, World};
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

#[derive(SceneStore, Clone, Copy)]
struct PerField {
    #[gpu(buffer = "replay_per_field")]
    value: u32,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "replay_packed")]
struct Packed {
    #[gpu]
    a: u32,
    #[gpu]
    b: u32,
}

#[derive(SceneStore, Clone, Copy)]
struct WrittenOnce {
    #[gpu(mirror = Once, buffer = "replay_once")]
    value: u32,
}

#[derive(SceneStore, Clone)]
struct VarLen {
    #[gpu(buffer = "replay_var_len_pool")]
    values: Vec<u32>,
    #[gpu(buffer = "replay_var_len_tag")]
    tag: u32,
}

/// A component with no `#[gpu]` fields; replay must leave it alone.
#[derive(Clone, Copy)]
struct CpuOnly(#[allow(dead_code)] u32);

fn context() -> Option<EngineGpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    Some(EngineGpuContext::new(Arc::new(device), Arc::new(queue)))
}

fn empty_store(ctx: &EngineGpuContext) -> Arc<SceneGpuStore> {
    Arc::new(SceneGpuStore::new(
        ctx,
        SceneGpuConfig { classes: vec![], tombstone_headroom: 0, max_cells_metadata: 0 },
    ))
}

fn read_words(ctx: &EngineGpuContext, buffer: &wgpu::Buffer, offset_bytes: u64, words: u64) -> Vec<u32> {
    let bytes = words * 4;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, offset_bytes, &staging, 0, bytes);
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

fn row(ctx: &EngineGpuContext, store: &SceneGpuStore, key: &'static str, row: u32, words: u64) -> Vec<u32> {
    let handle = store
        .resolve_buffer_handle(BufferKey::of(key))
        .unwrap_or_else(|| panic!("buffer `{key}` was never registered"));
    read_words(ctx, &handle.buffer, row as u64 * words * 4, words)
}

fn generation_row(ctx: &EngineGpuContext, mirror: &GpuMirrorHandle, row: u32) -> u32 {
    let mut out = None;
    mirror
        .generations()
        .with_buffer(&mut |buffer| out = Some(read_words(ctx, buffer, row as u64 * 4, 1)[0]));
    out.unwrap()
}

#[test]
fn attaching_after_population_writes_every_existing_gpu_row() {
    let Some(ctx) = context() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut world = World::new();
    // Burn a few slots so the rows of interest are not 0..n, and give one
    // reused slot a non-zero generation.
    let gone = world.spawn();
    world.spawn();
    world.despawn(gone);
    let reused = world.spawn();
    let other = world.spawn();
    assert_ne!(reused.generation(), 0);

    world.insert(reused, PerField { value: 11 });
    world.insert(reused, Packed { a: 3, b: 4 });
    world.insert(reused, WrittenOnce { value: 77 });
    world.insert(reused, CpuOnly(1));
    world.insert(other, VarLen { values: vec![5, 6, 7], tag: 9 });
    world.insert(other, PerField { value: 22 });

    // Nothing registered any buffer; the attach must do it.
    let store = empty_store(&ctx);
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));
    world.attach_gpu_mirror(mirror.clone());
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    assert_eq!(row(&ctx, &store, "replay_per_field", reused.index(), 1), [11]);
    assert_eq!(row(&ctx, &store, "replay_per_field", other.index(), 1), [22]);
    assert_eq!(row(&ctx, &store, "replay_packed", reused.index(), 2), [3, 4]);
    assert_eq!(row(&ctx, &store, "replay_once", reused.index(), 1), [77], "Once fields replay as first inserts");
    assert_eq!(row(&ctx, &store, "replay_var_len_tag", other.index(), 1), [9]);
    let pool = store.var_len_pool::<u32>(BufferKey::of("replay_var_len_pool")).expect("pool registered");
    let mut values = Vec::new();
    pool.with_buffer(&mut |buffer| values = read_words(&ctx, buffer, 0, 3));
    assert_eq!(values, [5, 6, 7]);

    assert_eq!(generation_row(&ctx, &mirror, reused.index()), reused.generation());
    assert_eq!(generation_row(&ctx, &mirror, other.index()), other.generation());

    // After the attach, ordinary writes keep the rows current.
    world.get_mut::<PerField>(reused).unwrap().value = 12;
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_per_field", reused.index(), 1), [12]);
}

#[test]
fn attaching_a_new_mirror_replays_into_it_and_reattaching_a_clone_does_not() {
    let Some(ctx) = context() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut world = World::new();
    let first_store = empty_store(&ctx);
    let first = GpuMirrorHandle::new(Arc::clone(&first_store), Arc::clone(ctx.queue()));
    world.attach_gpu_mirror(first.clone());
    let e = world.spawn();
    world.insert(e, PerField { value: 5 });
    world.flush_gpu_mirror(ctx.queue());

    // Re-attaching a clone of the current handle replays nothing: no row is
    // queued for upload.
    world.attach_gpu_mirror(first.clone());
    let stats = world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    assert_eq!(stats.bytes, 0, "re-attaching the same mirror must not rewrite rows");

    // A recreated device/store gets the whole world without any re-insert.
    let second_store = empty_store(&ctx);
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&second_store), Arc::clone(ctx.queue())));
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &second_store, "replay_per_field", e.index(), 1), [5]);
}

#[test]
fn erased_insert_and_remove_reach_the_mirror() {
    let Some(ctx) = context() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    register_component::<Packed>();
    let store = empty_store(&ctx);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
    let e = world.spawn();

    world.insert_dyn(e, Box::new(Packed { a: 1, b: 2 })).unwrap();
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_packed", e.index(), 2), [1, 2]);

    world.insert_dyn(e, Box::new(Packed { a: 8, b: 9 })).unwrap();
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_packed", e.index(), 2), [8, 9]);

    world.remove_dyn(e, component_id::<Packed>()).unwrap();
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_packed", e.index(), 2), [0, 0]);
}

/// An authored component whose GPU representation is a derived type, the
/// shape a generated `*GpuMirror` companion has. Its own registration
/// forwards to the derived type's dispatch, so the derived row follows the
/// authored value's writes with no second component involved.
struct Authored {
    value: u32,
    #[allow(dead_code)]
    cpu_only: String,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "replay_derived")]
struct AuthoredGpu {
    #[gpu]
    doubled: u32,
}

fn authored_dispatch(mirror: &GpuMirrorHandle, row: u32, data: *const (), is_new_insert: bool) {
    let authored = unsafe { &*(data as *const Authored) };
    let derived = AuthoredGpu { doubled: authored.value * 2 };
    assert!(pulsar_scenedb::gpu::world_mirror::write_derived_row(mirror, row, &derived, is_new_insert));
}

fn authored_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::world_mirror::clear_derived_row::<AuthoredGpu>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<Authored>,
        dispatch: authored_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::world_mirror::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<Authored>,
        clear: authored_clear,
    }
}

#[test]
fn a_derived_gpu_row_follows_the_authored_component() {
    let Some(ctx) = context() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Authored { value: 4, cpu_only: "x".into() });

    let store = empty_store(&ctx);
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_derived", e.index(), 1), [8], "replayed on attach");

    world.get_mut::<Authored>(e).unwrap().value = 5;
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_derived", e.index(), 1), [10], "mutation");

    world.remove::<Authored>(e);
    world.flush_gpu_mirror(ctx.queue());
    assert_eq!(row(&ctx, &store, "replay_derived", e.index(), 1), [0], "removal");
}
