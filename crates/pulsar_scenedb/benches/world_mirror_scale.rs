//! AAA-scale measurement for the World<->GPU mirror bridge (#24 + follow-ups
//! #25-#29 + hardening #212/#213/#214, all merged as of this file) — spawn
//! throughput, sustained churn, reallocation latency, concurrent buffer
//! access, flush cost vs. dirty fraction, reservation, and concurrent
//! `mark_dirty`, at realistic entity counts.
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
//! work (#212/#213/#214) has a reliable way to be re-measured without
//! waiting on that separate dependency fix.
//!
//! Run: `cargo bench -p pulsar_scenedb --features gpu --bench world_mirror_scale`
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
    println!("\n=== Scenario 1: spawn throughput (N entities, one packed #[gpu] insert each) ===");
    println!("{:>10} | {:>12} | {:>14} | {:>10}", "N", "wall time", "per-entity", "reallocs");
    for &n in &[1_000u32, 10_000, 100_000, 1_000_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.growable_epoch_for_id(id).unwrap_or(0);

        let start = Instant::now();
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        let elapsed = start.elapsed();

        let epoch_after = store.growable_epoch_for_id(id).unwrap_or(0);
        println!(
            "{:>10} | {:>12} | {:>14} | {:>10}",
            n, fmt_dur(elapsed), fmt_dur(elapsed / n), epoch_after - epoch_before,
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
    println!("\n=== Scenario 3: single reallocation latency (doubling N -> 2N, direct write-past-capacity) ===");
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

        let start = Instant::now();
        let e = world.spawn();
        world.insert(e, sample_instance(n));
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

            let start = Instant::now();
            std::thread::scope(|scope| {
                for t in 0..threads {
                    let store = Arc::clone(&store);
                    let queue = Arc::clone(ctx.queue());
                    scope.spawn(move || {
                        let base = t * per_thread;
                        let value = sample_instance(t);
                        let bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(&value as *const BenchInstance as *const u8, std::mem::size_of::<BenchInstance>())
                        };
                        for row in base..base + per_thread {
                            let _ = store.write_row_bytes_growing(id, &queue, bytes, row);
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
    println!("{:>10} | {:>18} | {:>18} | {:>14}", "N", "reallocs (cold)", "reallocs (reserved)", "reserved wall time");
    for &n in &[1_000u32, 10_000, 100_000] {
        let ctx = test_context();
        let mut store = SceneGpuStore::new(&ctx, scene_cfg());
        BenchInstance::register_gpu_columns_growable(&mut store, 64, ctx.device());
        let store = Arc::new(store);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));
        let id = BenchInstance::packed_gpu_component_id();
        let epoch_before = store.growable_epoch_for_id(id).unwrap_or(0);
        for i in 0..n {
            let e = world.spawn();
            world.insert(e, sample_instance(i));
        }
        let cold_reallocs = store.growable_epoch_for_id(id).unwrap_or(0) - epoch_before;

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
        let epoch_before2 = store2.growable_epoch_for_id(id).unwrap_or(0);
        let start = Instant::now();
        for i in 0..n {
            let e = world2.spawn();
            world2.insert(e, sample_instance(i));
        }
        let elapsed = start.elapsed();
        let reserved_reallocs = store2.growable_epoch_for_id(id).unwrap_or(0) - epoch_before2;

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

fn main() {
    println!("World<->GPU mirror bridge AAA-scale benchmark (post #212/#213/#214 hardening)");
    spawn_throughput();
    steady_state_churn();
    reallocation_latency();
    concurrent_buffer_access();
    flush_cost_vs_dirty_fraction();
    reservation_eliminates_batch_growth();
    concurrent_mark_dirty();
}
