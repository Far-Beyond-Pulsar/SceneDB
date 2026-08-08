//! AAA-scale measurement for the World<->GPU mirror bridge (#24 + follow-ups
//! #25-#29 + hardening #212/#213/#214 + the SceneDB#39 churn/scatter-write
//! fix, all merged as of this file) — spawn throughput, sustained churn,
//! reallocation latency, concurrent buffer access, flush cost vs. dirty
//! fraction, reservation, concurrent `mark_dirty`, and a diagnostic
//! breakdown of where churn's per-op cost actually goes, at realistic
//! entity counts.
//!
//! Originally written and filed as Helio#211 (`crates/helio-scenedb/benches/
//! world_mirror_scale.rs` in the `Helio` repo, upstream of this crate) —
//! duplicated here, in `pulsar_scenedb`'s own `benches/`, specifically to
//! get a reliable run on Windows: Helio's workspace requests wgpu's `dx12`
//! backend explicitly, which currently collides with a separate,
//! already-tracked `windows-core` version conflict in that workspace's
//! `Cargo.lock` (two incompatible versions — 0.58.0 via `gpu-allocator`,
//! 0.62.2 direct in `wgpu-hal` 30.0.0 — coexist and can break compilation
//! once anything touches lockfile resolution). This crate's own `wgpu`
//! dependency has no such explicit backend list and has been reliable for
//! every GPU test in this crate all along. The Helio-side copy stays the
//! canonical, tracked one (Helio#211); this one exists purely so hardening
//! work (#212/#213/#214, SceneDB#39) has a reliable way to be re-measured
//! without waiting on that separate dependency fix.
//!
//! Run: `cargo bench -p pulsar_scenedb --features gpu --bench world_mirror_scale`
//!
//! Set `BENCH_DIAGNOSTIC_ONLY=1` to run only Scenario 8 (the churn
//! cost-breakdown diagnostic) — useful when investigating a churn
//! regression without waiting for the full ~1-2 minute suite.
//!
//! `harness = false`, matching `gpu_timing.rs`/`legacy_model_bench.rs`'s own
//! choice: this measures real wall-clock cost of GPU-submitting operations
//! end to end, not a pure-CPU microbenchmark criterion is tuned for.
use pulsar_scenedb::gpu::{EngineGpuContext, GpuColumnSet, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shaped like Helio's `libhelio::instance::GpuInstanceData` (a realistic
/// AAA per-instance record size, not a toy fixture). `#[gpu(layout =
/// packed)]` so it lands in exactly one buffer.
///
/// `#[gpu(mirror = Once)]`, not the plain `#[gpu]` (`DirtyTracked`) default
/// -- so `register_gpu_columns_growable` routes this through the immediate/
/// growable path (what scenarios 1/3/4/6 are about), not the deferred
/// dirty-tracked one (scenario 5/7 use their own dedicated DirtyTracked
/// fixtures instead).
#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
struct BenchInstance {
    #[gpu(mirror = Once)]
    model: [f32; 16],
    #[gpu(mirror = Once)]
    normal_mat: [f32; 16],
    #[gpu(mirror = Once)]
    bounds: [f32; 16],
    #[gpu(mirror = Once)]
    mesh_material_flags: [f32; 16],
}

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter -- GPU bench needs a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scenedb-world-mirror-scale-bench"),
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

fn sample_instance(i: u32) -> BenchInstance {
    BenchInstance {
        model: [i as f32; 16],
        normal_mat: [i as f32; 16],
        bounds: [i as f32; 16],
        mesh_material_flags: [i as f32; 16],
    }
}

fn fmt_dur(d: Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3} s", d.as_secs_f64())
    } else if d.as_micros() >= 1000 {
        format!("{:.3} ms", d.as_secs_f64() * 1e3)
    } else {
        format!("{:.1} us", d.as_secs_f64() * 1e6)
    }
}

fn spawn_throughput() {
    println!("\n=== Scenario 1: spawn throughput (N entities, one packed #[gpu(mirror = Once)] insert each) ===");
    // SceneDB#39: Once-mode writes are now deferred to `flush_gpu_mirror`
    // instead of an immediate `queue.write_buffer` per insert -- so this
    // scenario measures BOTH halves separately: `insert only` (now just
    // host-side bookkeeping, no GPU work at all) and `insert + flush`
    // (equivalent to the OLD immediate-write behavior -- the real,
    // comparable end-to-end cost of "these N entities' data is actually on
    // the GPU"). Reporting only the first half would be misleading (it
    // would look like an enormous, too-good-to-be-true speedup that's
    // really just "no GPU work happened yet").
    println!(
        "{:>10} | {:>14} | {:>16} | {:>14} | {:>16} | {:>10}",
        "N", "insert only", "insert-only/ent", "insert+flush", "total/entity", "reallocs"
    );
    for &n in &[1_000u32, 10_000, 100_000, 1_000_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.dirty_tracked_epoch_for_id(id).unwrap_or(0);

        let insert_start = Instant::now();
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        let insert_elapsed = insert_start.elapsed();

        let flush_start = Instant::now();
        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
        let total_elapsed = insert_elapsed + flush_start.elapsed();

        let epoch_after = store.dirty_tracked_epoch_for_id(id).unwrap_or(0);
        println!(
            "{:>10} | {:>14} | {:>16} | {:>14} | {:>16} | {:>10}",
            n,
            fmt_dur(insert_elapsed),
            fmt_dur(insert_elapsed / n),
            fmt_dur(total_elapsed),
            fmt_dur(total_elapsed / n),
            epoch_after - epoch_before,
        );
    }
}

fn steady_state_churn() {
    println!("\n=== Scenario 2: steady-state churn (N live entities, F frames, each frame despawns+respawns a fraction, one flush/frame) ===");
    println!("{:>10} | {:>8} | {:>10} | {:>14} | {:>14}", "N", "churn %", "frames", "total", "per-frame");
    const FRAMES: u32 = 30;
    for &n in &[1_000u32, 10_000, 100_000] {
        for &churn_pct in &[1u32, 10, 50] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let mut world = World::new();
            world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

            let mut entities: Vec<_> = (0..n)
                .map(|i| {
                    let e = world.spawn();
                    world.insert(e, sample_instance(i));
                    e
                })
                .collect();

            let churn_count = ((n as u64 * churn_pct as u64) / 100) as usize;
            let start = Instant::now();
            for frame in 0..FRAMES {
                for slot in 0..churn_count {
                    let idx = (slot * 7919 + frame as usize) % entities.len();
                    world.despawn(entities[idx]);
                    let e = world.spawn();
                    world.insert(e, sample_instance(idx as u32));
                    entities[idx] = e;
                }
                world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
            }
            let elapsed = start.elapsed();
            println!(
                "{:>10} | {:>7}% | {:>10} | {:>14} | {:>14}",
                n, churn_pct, FRAMES, fmt_dur(elapsed), fmt_dur(elapsed / FRAMES),
            );
        }
    }
}

fn reallocation_latency() {
    println!("\n=== Scenario 3: single reallocation latency (doubling N -> 2N, write-past-capacity via flush) ===");
    // SceneDB#39: the write that actually crosses the capacity boundary --
    // and therefore the reallocation itself -- now happens at
    // `flush_gpu_mirror`, not at `insert`. Fills to exactly N first (a
    // flush here does NOT grow anything, capacity already fits exactly),
    // queues one more row past capacity, then times the flush that performs
    // the real write-past-capacity + grow-and-copy + GPU work.
    println!("{:>12} | {:>12} | {:>14}", "N (before)", "-> 2N", "realloc time");
    for &n in &[1_000u32, 10_000, 100_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

        let e = world.spawn();
        world.insert(e, sample_instance(n));

        let start = Instant::now();
        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let elapsed = start.elapsed();

        println!("{:>12} | {:>12} | {:>14}", n, n * 2, fmt_dur(elapsed));
    }
}

fn concurrent_buffer_access() {
    println!("\n=== Scenario 4: concurrent access to one shared, already-registered buffer (disjoint rows per thread) ===");
    println!("{:>10} | {:>8} | {:>14} | {:>18}", "N total", "threads", "wall time", "per-write");
    for &n in &[10_000u32, 100_000] {
        for &threads in &[1u32, 4, 8, 16] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            BenchInstance::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let id = BenchInstance::packed_gpu_component_id();
            let per_thread = n / threads;

            // SceneDB#39: BenchInstance's packed Once-mode column is
            // registered dirty-tracked now (same path DirtyTracked fields
            // use) -- writes go through `mark_gpu_row_dirty` (no GPU work,
            // no `queue` needed), not the old immediate
            // `write_row_bytes_growing` call.
            let start = Instant::now();
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let store = Arc::clone(&store);
                    scope.spawn(move || {
                        let base = t * per_thread;
                        let value = sample_instance(t);
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(&value as *const BenchInstance as *const u8, std::mem::size_of::<BenchInstance>())
                        };
                        for row in base..base + per_thread {
                            let _ = store.mark_gpu_row_dirty(id, row, bytes);
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            println!("{:>10} | {:>8} | {:>14} | {:>18}", n, threads, fmt_dur(elapsed), fmt_dur(elapsed / n));
        }
    }
}

fn flush_cost_vs_dirty_fraction() {
    println!("\n=== Scenario 5: flush_gpu_mirror cost vs. dirty fraction (N entities registered dirty-tracked, only a fraction marked dirty) ===");
    println!("{:>10} | {:>10} | {:>14}", "N", "dirty %", "flush time");
    for &n in &[1_000u32, 10_000, 100_000] {
        for &dirty_pct in &[0u32, 1, 10, 100] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            #[derive(SceneStore, Clone, Copy)]
            struct DirtyTrackedBenchField {
                #[gpu]
                value: u32,
            }
            DirtyTrackedBenchField::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let mut world = World::new();
            world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

            let entities: Vec<_> = (0..n)
                .map(|i| {
                    let e = world.spawn();
                    world.insert(e, DirtyTrackedBenchField { value: i });
                    e
                })
                .collect();
            world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

            let dirty_count = ((n as u64 * dirty_pct as u64) / 100) as usize;
            for &e in entities.iter().take(dirty_count) {
                world.insert(e, DirtyTrackedBenchField { value: 999 });
            }

            let start = Instant::now();
            world.flush_gpu_mirror(ctx.queue()).unwrap();
            let elapsed = start.elapsed();
            println!("{:>10} | {:>9}% | {:>14}", n, dirty_pct, fmt_dur(elapsed));
        }
    }
}

fn reservation_eliminates_batch_growth() {
    println!("\n=== Scenario 6 (post-#212): reserve_gpu_mirror_capacity before a known-size batch spawn ===");
    // SceneDB#39: growth now happens at `flush_gpu_mirror` (where the
    // deferred Once-mode writes actually land), not at `insert` -- both
    // variants below flush once, after the batch, to actually trigger (or,
    // for the reserved variant, confirm the absence of) growth.
    println!("{:>10} | {:>18} | {:>18} | {:>14}", "N", "reallocs (cold)", "reallocs (reserved)", "reserved wall time");
    for &n in &[1_000u32, 10_000, 100_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.dirty_tracked_epoch_for_id(id).unwrap_or(0);
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
        let cold_reallocs = store.dirty_tracked_epoch_for_id(id).unwrap_or(0) - epoch_before;

        let ctx2 = test_context();
        let mut store2 = SceneGpuStore::new(&ctx2, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store2, 64, ctx2.device());
        let store2 = Arc::new(store2);
        let mut world2 = World::new();
        world2.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store2), Arc::clone(ctx2.queue())));
        world2
            .reserve_gpu_mirror_capacity(ctx2.queue(), n)
            .expect("mirror attached")
            .expect("reserve succeeds");
        let epoch_before2 = store2.dirty_tracked_epoch_for_id(id).unwrap_or(0);
        let start = Instant::now();
        for i in 0..n {
            let e = world2.spawn();
            world2.insert(e, sample_instance(i));
        }
        world2.flush_gpu_mirror(ctx2.queue()).expect("mirror attached");
        let elapsed = start.elapsed();
        let reserved_reallocs = store2.dirty_tracked_epoch_for_id(id).unwrap_or(0) - epoch_before2;

        println!("{:>10} | {:>18} | {:>18} | {:>14}", n, cold_reallocs, reserved_reallocs, fmt_dur(elapsed));
    }
}

fn concurrent_mark_dirty() {
    println!("\n=== Scenario 7 (post-#213): concurrent mark_dirty on disjoint rows, pre-reserved (fast read-lock path) ===");
    println!("{:>10} | {:>8} | {:>14} | {:>18}", "N total", "threads", "wall time", "per-mark");
    #[derive(SceneStore, Clone, Copy)]
    struct ConcurrentMarkField {
        #[gpu]
        value: u32,
    }
    for &n in &[10_000u32, 100_000] {
        for &threads in &[1u32, 4, 8, 16] {
            let ctx = test_context();
            let mut store = SceneGpuStore::new(&ctx, scene_cfg());
            ConcurrentMarkField::register_gpu_columns_growable(&mut store, n, ctx.device());
            let store = Arc::new(store);
            let id = ConcurrentMarkField::gpu_columns()[0].field_token.id();
            let per_thread = n / threads;

            let start = Instant::now();
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let store = Arc::clone(&store);
                    scope.spawn(move || {
                        let base = t * per_thread;
                        let value_bytes = t.to_ne_bytes();
                        for row in base..base + per_thread {
                            store.mark_gpu_row_dirty(id, row, &value_bytes);
                        }
                    });
                }
            });
            let elapsed = start.elapsed();
            println!("{:>10} | {:>8} | {:>14} | {:>18}", n, threads, fmt_dur(elapsed), fmt_dur(elapsed / n));
        }
    }
}

/// Diagnostic, not a scenario to compare against any other bench file:
/// isolates how much of Scenario 2's per-churn-op cost is GPU-mirror
/// bookkeeping at all, vs. plain `World` archetype/ECS mechanics that would
/// cost the same with no mirror attached. Three variants, same churn
/// pattern as Scenario 2, at 100k/10%:
///
/// (a) no mirror attached at all, inserting a small plain (non-`#[gpu]`)
///     component -- the floor: pure entity_slots + archetype swap-remove +
///     archetype migration cost.
/// (b) no mirror attached, inserting `BenchInstance` (a real 256-byte
///     packed `#[gpu]`-bearing component) -- isolates whether copying a
///     WIDE component into the archetype column costs something on its own,
///     independent of any GPU-mirror dispatch (which never runs here, since
///     `self.gpu_mirror` is `None`).
/// (c) mirror attached, same as Scenario 2 -- the number this whole
///     investigation is about.
fn churn_cost_breakdown() {
    println!("\n=== Scenario 8 (diagnostic): where does churn's per-op cost actually go? ===");
    println!("100,000 entities, 10% churn/frame, 30 frames, one flush/frame where applicable");
    println!("{:<55} | {:>14} | {:>14}", "variant", "total", "per-op");
    const N: u32 = 100_000;
    const CHURN_PCT: u32 = 10;
    const FRAMES: u32 = 30;
    let churn_count = ((N as u64 * CHURN_PCT as u64) / 100) as usize;
    let total_ops = churn_count as u64 * FRAMES as u64;

    struct PlainSmall {
        #[allow(dead_code)]
        value: u32,
    }

    // (a) no mirror, small plain component
    {
        let mut world = World::new();
        let mut entities: Vec<_> = (0..N)
            .map(|i| {
                let e = world.spawn();
                world.insert(e, PlainSmall { value: i });
                e
            })
            .collect();
        let start = Instant::now();
        for frame in 0..FRAMES {
            for slot in 0..churn_count {
                let idx = (slot * 7919 + frame as usize) % entities.len();
                world.despawn(entities[idx]);
                let e = world.spawn();
                world.insert(e, PlainSmall { value: idx as u32 });
                entities[idx] = e;
            }
        }
        let elapsed = start.elapsed();
        println!(
            "{:<55} | {:>14} | {:>14}",
            "(a) no mirror, small plain component",
            fmt_dur(elapsed),
            fmt_dur(elapsed / total_ops as u32)
        );
    }

    // (b) no mirror, wide #[gpu]-bearing component (BenchInstance), but
    // gpu_mirror is never attached -- dispatch never runs.
    {
        let mut world = World::new();
        let mut entities: Vec<_> = (0..N)
            .map(|i| {
                let e = world.spawn();
                world.insert(e, sample_instance(i));
                e
            })
            .collect();
        let start = Instant::now();
        for frame in 0..FRAMES {
            for slot in 0..churn_count {
                let idx = (slot * 7919 + frame as usize) % entities.len();
                world.despawn(entities[idx]);
                let e = world.spawn();
                world.insert(e, sample_instance(idx as u32));
                entities[idx] = e;
            }
        }
        let elapsed = start.elapsed();
        println!(
            "{:<55} | {:>14} | {:>14}",
            "(b) no mirror, wide #[gpu] component (unattached)",
            fmt_dur(elapsed),
            fmt_dur(elapsed / total_ops as u32)
        );
    }

    // (c) mirror attached, same as Scenario 2's 100k/10% row -- split into
    // despawn-only and spawn+insert-only halves (still summed into one
    // total/per-op line for comparability with (a)/(b)/Scenario 2, but
    // printed separately too) to find out which half the extra cost is
    // actually in.
    {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, N, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let mut entities: Vec<_> = (0..N)
            .map(|i| {
                let e = world.spawn();
                world.insert(e, sample_instance(i));
                e
            })
            .collect();
        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
        let mut despawn_total = std::time::Duration::ZERO;
        let mut spawn_insert_total = std::time::Duration::ZERO;
        let mut flush_total = std::time::Duration::ZERO;
        let start = Instant::now();
        for frame in 0..FRAMES {
            for slot in 0..churn_count {
                let idx = (slot * 7919 + frame as usize) % entities.len();
                let t0 = Instant::now();
                world.despawn(entities[idx]);
                despawn_total += t0.elapsed();
                let t1 = Instant::now();
                let e = world.spawn();
                world.insert(e, sample_instance(idx as u32));
                spawn_insert_total += t1.elapsed();
                entities[idx] = e;
            }
            let t2 = Instant::now();
            world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
            flush_total += t2.elapsed();
        }
        let elapsed = start.elapsed();
        println!(
            "{:<55} | {:>14} | {:>14}",
            "(c) mirror attached (Scenario 2's own path)",
            fmt_dur(elapsed),
            fmt_dur(elapsed / total_ops as u32)
        );
        println!(
            "{:<55} | {:>14} | {:>14}",
            "    .. of which: despawn half",
            fmt_dur(despawn_total),
            fmt_dur(despawn_total / total_ops as u32)
        );
        println!(
            "{:<55} | {:>14} | {:>14}",
            "    .. of which: spawn+insert half",
            fmt_dur(spawn_insert_total),
            fmt_dur(spawn_insert_total / total_ops as u32)
        );
        println!(
            "{:<55} | {:>14} | {:>14}",
            "    .. of which: flush_gpu_mirror (30 calls total)",
            fmt_dur(flush_total),
            fmt_dur(flush_total / FRAMES)
        );
    }
}

fn main() {
    println!("World<->GPU mirror bridge AAA-scale benchmark (post #212/#213/#214 hardening)");
    if std::env::var("BENCH_DIAGNOSTIC_ONLY").is_ok() {
        churn_cost_breakdown();
        return;
    }
    spawn_throughput();
    steady_state_churn();
    reallocation_latency();
    concurrent_buffer_access();
    flush_cost_vs_dirty_fraction();
    reservation_eliminates_batch_growth();
    concurrent_mark_dirty();
    churn_cost_breakdown();
}
