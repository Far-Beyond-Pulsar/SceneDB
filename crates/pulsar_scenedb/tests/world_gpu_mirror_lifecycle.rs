//! World GPU mirror lifecycle cases from the scene-data acceptance matrix
//! (Pulsar-Native#1035, #1081) not covered elsewhere: a type that first
//! appears after other rows were uploaded, a bulk load read back row by row,
//! growth that keeps every earlier row, a mirror rebuilt on a new device,
//! two worlds with their own mirrors, and uploads bounded to the rows that
//! changed.
//!
//! Needs a GPU adapter (any Vulkan device, including lavapipe); skips
//! without one.

use pulsar_scenedb::gpu::{
    BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "lifecycle_first")]
struct First {
    #[gpu]
    a: u32,
    #[gpu]
    b: u32,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "lifecycle_late")]
struct Late {
    #[gpu]
    value: u32,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "lifecycle_bulk")]
struct Bulk {
    #[gpu]
    value: u32,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "lifecycle_device")]
struct OnDevice {
    #[gpu]
    value: u32,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed, buffer = "lifecycle_bounded")]
struct Bounded {
    #[gpu]
    a: u32,
    #[gpu]
    b: u32,
}

fn context() -> Option<EngineGpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    Some(EngineGpuContext::new(Arc::new(device), Arc::new(queue)))
}

fn empty_store(ctx: &EngineGpuContext) -> Arc<SceneGpuStore> {
    Arc::new(SceneGpuStore::new(
        ctx,
        SceneGpuConfig {
            classes: vec![],
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    ))
}

fn mirrored_world(ctx: &EngineGpuContext, store: &Arc<SceneGpuStore>) -> World {
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(store),
        Arc::clone(ctx.queue()),
    ));
    world
}

fn read_words(
    ctx: &EngineGpuContext,
    buffer: &wgpu::Buffer,
    offset_bytes: u64,
    words: u64,
) -> Vec<u32> {
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
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
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

/// Rows `0..rows` of a packed buffer of `words`-word rows, in one readback.
fn rows(
    ctx: &EngineGpuContext,
    store: &SceneGpuStore,
    key: &'static str,
    rows: u32,
    words: u64,
) -> Vec<Vec<u32>> {
    let handle = store
        .resolve_buffer_handle(BufferKey::of(key))
        .unwrap_or_else(|| panic!("buffer `{key}` was never registered"));
    read_words(ctx, &handle.buffer, 0, rows as u64 * words)
        .chunks_exact(words as usize)
        .map(<[u32]>::to_vec)
        .collect()
}

#[test]
fn a_type_first_inserted_after_other_rows_uploaded_lands_without_touching_them() {
    let Some(ctx) = context() else { return };
    let store = empty_store(&ctx);
    let mut world = mirrored_world(&ctx, &store);

    let first: Vec<_> = (0..8u32)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, First { a: i, b: 100 + i });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert!(
        store
            .resolve_buffer_handle(BufferKey::of("lifecycle_late"))
            .is_none(),
        "the late type has no buffer before its first insert"
    );

    // The new type appears on some of the same entities and on new ones.
    let late: Vec<_> = first
        .iter()
        .copied()
        .step_by(2)
        .chain((0..3).map(|_| world.spawn()))
        .collect();
    for (n, &e) in late.iter().enumerate() {
        world.insert(
            e,
            Late {
                value: 7000 + n as u32,
            },
        );
    }
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let max = late.iter().chain(&first).map(|e| e.index()).max().unwrap() + 1;
    let late_rows = rows(&ctx, &store, "lifecycle_late", max, 1);
    for (n, e) in late.iter().enumerate() {
        assert_eq!(late_rows[e.index() as usize], vec![7000 + n as u32]);
    }
    let first_rows = rows(&ctx, &store, "lifecycle_first", max, 2);
    for (i, e) in first.iter().enumerate() {
        assert_eq!(
            first_rows[e.index() as usize],
            vec![i as u32, 100 + i as u32]
        );
    }
}

#[test]
fn a_bulk_load_past_capacity_reads_back_row_by_row_and_keeps_earlier_rows() {
    let Some(ctx) = context() else { return };
    let store = empty_store(&ctx);
    let mut world = mirrored_world(&ctx, &store);

    // A few rows uploaded first, at the auto-registered capacity.
    let early: Vec<_> = (0..4u32)
        .map(|i| {
            let e = world.spawn();
            world.insert(
                e,
                Bulk {
                    value: 1_000_000 + i,
                },
            );
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let before = store
        .resolve_buffer_handle(BufferKey::of("lifecycle_bulk"))
        .unwrap();

    // Then a bulk load far past it, in one frame.
    const COUNT: u32 = 5_000;
    let bulk: Vec<_> = (0..COUNT)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Bulk { value: i * 3 + 1 });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let after = store
        .resolve_buffer_handle(BufferKey::of("lifecycle_bulk"))
        .unwrap();
    assert!(
        after.epoch > before.epoch,
        "growth publishes a new buffer epoch"
    );
    assert!(after.row_capacity() > before.row_capacity());

    let max = bulk.iter().map(|e| e.index()).max().unwrap() + 1;
    let all = rows(&ctx, &store, "lifecycle_bulk", max, 1);
    for (i, e) in early.iter().enumerate() {
        assert_eq!(
            all[e.index() as usize],
            vec![1_000_000 + i as u32],
            "a row from before the growth"
        );
    }
    let wrong = bulk
        .iter()
        .enumerate()
        .filter(|(i, e)| all[e.index() as usize] != vec![*i as u32 * 3 + 1])
        .count();
    assert_eq!(wrong, 0, "every bulk-loaded row reads back");
}

#[test]
fn a_mirror_on_a_new_device_is_rebuilt_from_the_world() {
    let Some(old_ctx) = context() else { return };
    let mut world = {
        let store = empty_store(&old_ctx);
        mirrored_world(&old_ctx, &store)
    };
    let entities: Vec<_> = (0..20u32)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, OnDevice { value: 50 + i });
            e
        })
        .collect();
    world.flush_gpu_mirror(old_ctx.queue()).unwrap();

    // The device is lost: the old mirror goes, and its context with it.
    assert!(world.detach_gpu_mirror().is_some());
    drop(old_ctx);
    // Writes while there is no device stay in the World.
    world.get_mut::<OnDevice>(entities[3]).unwrap().value = 999;
    world.despawn(entities[4]);

    let Some(ctx) = context() else { return };
    let store = empty_store(&ctx);
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let max = entities.iter().map(|e| e.index()).max().unwrap() + 1;
    let got = rows(&ctx, &store, "lifecycle_device", max, 1);
    for (i, e) in entities.iter().enumerate() {
        let expected = match i {
            3 => 999,
            4 => 0,
            _ => 50 + i as u32,
        };
        assert_eq!(got[e.index() as usize], vec![expected], "entity {i}");
    }
}

#[test]
fn two_worlds_with_their_own_mirrors_do_not_share_rows() {
    let Some(ctx) = context() else { return };
    let (store_a, store_b) = (empty_store(&ctx), empty_store(&ctx));
    let mut a = mirrored_world(&ctx, &store_a);
    let mut b = mirrored_world(&ctx, &store_b);

    // The same entity indices in both worlds, with different values.
    let ea: Vec<_> = (0..6u32).map(|_| a.spawn()).collect();
    let eb: Vec<_> = (0..6u32).map(|_| b.spawn()).collect();
    assert_eq!(ea[2].index(), eb[2].index());
    for (i, (&x, &y)) in ea.iter().zip(&eb).enumerate() {
        a.insert(x, First { a: i as u32, b: 1 });
        b.insert(
            y,
            First {
                a: 500 + i as u32,
                b: 2,
            },
        );
    }
    a.flush_gpu_mirror(ctx.queue()).unwrap();
    b.flush_gpu_mirror(ctx.queue()).unwrap();

    // Edit and despawn in one world only.
    a.get_mut::<First>(ea[1]).unwrap().a = 77;
    b.despawn(eb[3]);
    a.flush_gpu_mirror(ctx.queue()).unwrap();
    b.flush_gpu_mirror(ctx.queue()).unwrap();

    let rows_a = rows(&ctx, &store_a, "lifecycle_first", 6, 2);
    let rows_b = rows(&ctx, &store_b, "lifecycle_first", 6, 2);
    for i in 0..6usize {
        let expected_a = if i == 1 {
            vec![77, 1]
        } else {
            vec![i as u32, 1]
        };
        let expected_b = if i == 3 {
            vec![0, 0]
        } else {
            vec![500 + i as u32, 2]
        };
        assert_eq!(
            rows_a[ea[i].index() as usize],
            expected_a,
            "world A row {i}"
        );
        assert_eq!(
            rows_b[eb[i].index() as usize],
            expected_b,
            "world B row {i}"
        );
    }
}

#[test]
fn uploads_are_bounded_to_the_rows_that_changed() {
    let Some(ctx) = context() else { return };
    let store = empty_store(&ctx);
    let mut world = mirrored_world(&ctx, &store);
    let entities: Vec<_> = (0..200u32)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Bounded { a: i, b: i });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let row_bytes = store
        .resolve_buffer_handle(BufferKey::of("lifecycle_bounded"))
        .unwrap()
        .row_bytes;

    // An idle frame uploads nothing.
    let idle = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((idle.ranges, idle.bytes), (0, 0), "idle flush");

    // A borrow that writes nothing uploads nothing.
    let _ = world.get_mut::<Bounded>(entities[10]).unwrap().a;
    let read_only = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(read_only.bytes, 0, "a read-only guard");

    // One edited row of 200 uploads one row.
    world.get_mut::<Bounded>(entities[57]).unwrap().b = 9;
    let one = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((one.ranges, one.bytes), (1, row_bytes as u64), "one row");

    // Two distant rows upload two rows, not the span between them.
    world.get_mut::<Bounded>(entities[3]).unwrap().a = 1;
    world.get_mut::<Bounded>(entities[190]).unwrap().a = 1;
    let two = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(two.bytes, 2 * row_bytes as u64, "two scattered rows");
}
