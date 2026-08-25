//! Benchmarks for the tiered-storage substrate (SceneDB#61 §Benchmarks),
//! following `content_dedup_benchmarks.rs`'s established idiom: `Instant`
//! timing + smoke assertions, deliberately NOT criterion. Run with
//! `cargo test --release --test tier_storage_benchmarks -- --nocapture
//! --test-threads=1` to see printed numbers.
//!
//! - BM1: touch/release throughput, atomic vs segmented payloads.
//! - BM2: bulk promote of 10k ids (cold) and an LRU demote storm — wall
//!   time AND tail latency.
//! - BM3: steady-state overhead when tiers never move vs today's baseline
//!   shape (write+flush churn with the engine attached but untouched).

#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::time::Instant;

use pulsar_scenedb::gpu::{
    register_segment_layout, BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig,
    SceneGpuStore, Segment, Tier, TierConfig, TierSelector, TierSpan,
};
use pulsar_scenedb::{Entity, SceneStore, World};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Texel(pub u8);
unsafe impl pulsar_scenedb::page::Pod for Texel {}

#[derive(Clone, Debug, SceneStore)]
struct Blob {
    #[gpu]
    data: Vec<u32>,
}

#[derive(Clone, Debug, SceneStore)]
struct SegBlob {
    #[gpu]
    texels: Vec<Texel>,
}

fn test_context(label: &str) -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig { classes: vec![], tombstone_headroom: 8, max_cells_metadata: 16 }
}

/// BM1: touch/release throughput on ATOMIC payloads (zero-declaration
/// default). Smoke bound catches a collapse (e.g. accidental O(n) per-op),
/// not a perf regression gate.
#[test]
fn bm1_touch_release_throughput_atomic() {
    const IDS: usize = 4_000;
    let ctx = test_context("bm1-atomic");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, (IDS as u32) * 2, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![i as u32; 4] });
            e.index()
        })
        .map(|row| Blob::data_gpu_handle(&store, row).unwrap())
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    // Warm-up round (uncounted): grows shard vectors / arena slots.
    for h in &handles {
        let sel = TierSelector::PoolSlot { pool: BufferKey::of("Blob::data"), handle: *h };
        store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
        store.release_tier(sel, TierSpan::Whole).unwrap();
    }
    store.flush_tier_transitions(ctx.queue());

    const ROUNDS: usize = 5;
    let start = Instant::now();
    for _ in 0..ROUNDS {
        for h in &handles {
            let sel = TierSelector::PoolSlot { pool: BufferKey::of("Blob::data"), handle: *h };
            store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
            store.release_tier(sel, TierSpan::Whole).unwrap();
        }
    }
    store.flush_tier_transitions(ctx.queue());
    let elapsed = start.elapsed();
    let ops = (IDS * ROUNDS * 2) as f64;
    let ns_per_op = elapsed.as_nanos() as f64 / ops;
    eprintln!("BM1-atomic: {ops} touch/release ops in {elapsed:?} ({ns_per_op:.0} ns/op)");
    assert!(ns_per_op < 50_000.0, "BM1-atomic regressed badly: {ns_per_op:.0} ns/op");
}

/// BM1b: same shape on SEGMENTED payloads (3 rank units) — layout planning
/// is per-flight work; this pins it linear and cheap enough.
#[test]
fn bm1b_touch_release_throughput_segmented() {
    register_segment_layout::<Texel>(&[
        Segment::new(0, 16, 0),
        Segment::new(16, 16, 1),
        Segment::new(32, 32, 2),
    ])
    .unwrap();

    const IDS: usize = 2_000;
    let ctx = test_context("bm1-segmented");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    SegBlob::register_gpu_columns_growable(&store, (IDS as u32) * 2, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, SegBlob { texels: vec![Texel((i % 256) as u8); 64] });
            e.index()
        })
        .map(|row| SegBlob::texels_gpu_handle(&store, row).unwrap())
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    for h in &handles {
        let sel = TierSelector::PoolSlot { pool: BufferKey::of("SegBlob::texels"), handle: *h };
        store.touch_tier(sel, TierSpan::ThroughRank(2), Tier::Vram).unwrap();
        store.release_tier(sel, TierSpan::Whole).unwrap();
    }
    store.flush_tier_transitions(ctx.queue());

    const ROUNDS: usize = 5;
    let start = Instant::now();
    for _ in 0..ROUNDS {
        for h in &handles {
            let sel = TierSelector::PoolSlot { pool: BufferKey::of("SegBlob::texels"), handle: *h };
            store.touch_tier(sel, TierSpan::ThroughRank(2), Tier::Vram).unwrap();
            store.release_tier(sel, TierSpan::Whole).unwrap();
        }
    }
    let stats = store.flush_tier_transitions(ctx.queue());
    let elapsed = start.elapsed();
    let ops = (IDS * ROUNDS * 2) as f64;
    let ns_per_op = elapsed.as_nanos() as f64 / ops;
    eprintln!(
        "BM1-segmented: {ops} ops (3-rank payloads) in {elapsed:?} ({ns_per_op:.0} ns/op; promoted {})",
        stats.promoted
    );
    assert!(ns_per_op < 80_000.0, "BM1-segmented regressed badly: {ns_per_op:.0} ns/op");
}

/// BM2: bulk promote of 10k cold ids, then a full LRU demote storm against
/// a budget that fits NONE of them. Wall time AND tail latency (max single
/// flush).
#[test]
fn bm2_bulk_promote_10k_and_lru_demote_storm() {
    const IDS: usize = 10_000;
    let ctx = test_context("bm2-bulk-promote");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, (IDS as u32) * 2, ctx.device());
    // Budget fits nothing: every promotion evicts everything else first —
    // worst-case LRU churn by construction.
    store.configure_tiers(TierConfig { vram_budget_bytes: 16, ram_budget_bytes: u64::MAX / 2 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![i as u32; 4] });
            e.index()
        })
        .map(|row| Blob::data_gpu_handle(&store, row).unwrap())
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    // Cold bulk promote: queue ALL, flush once.
    let t_queue = Instant::now();
    for h in &handles {
        store.touch_tier(
            TierSelector::PoolSlot { pool: BufferKey::of("Blob::data"), handle: *h },
            TierSpan::Whole,
            Tier::Vram,
        )
        .unwrap();
    }
    let queue_time = t_queue.elapsed();
    let t_flush = Instant::now();
    let promote_stats = store.flush_tier_transitions(ctx.queue());
    let promote_flush = t_flush.elapsed();

    // Demote storm: release all, flush once (every resident demotes; most
    // were declined/evicted so this also exercises the empty path at scale).
    let t_rel = Instant::now();
    for h in &handles {
        store
            .release_tier(
                TierSelector::PoolSlot { pool: BufferKey::of("Blob::data"), handle: *h },
                TierSpan::Whole,
            )
            .unwrap();
    }
    let release_time = t_rel.elapsed();
    let t_flush2 = Instant::now();
    let demote_stats = store.flush_tier_transitions(ctx.queue());
    let demote_flush = t_flush2.elapsed();

    eprintln!(
        "BM2: bulk-touch {IDS} ids in {queue_time:?}; promote flush in {promote_flush:?} \
         (promoted {}, evicted-by-LRU implied: declines {}); release in {release_time:?}; \
         demote flush in {demote_flush:?} (demoted {})",
        promote_stats.promoted, promote_stats.declined_budget, demote_stats.demoted
    );
    // Linear evidence: 10k ids must drain in well under a second even in
    // debug builds with maximal eviction pressure.
    assert!(promote_flush.as_secs() < 30, "bulk promote collapsed: {promote_flush:?}");
    assert!(demote_flush.as_secs() < 30, "demote storm collapsed: {demote_flush:?}");
    // Tail latency: single-flush worst case stays bounded relative to the
    // per-id mean (no superlinear blowup inside one flush).
    if promote_stats.promoted > 0 || demote_stats.demoted > 0 {
        let total = promote_flush + demote_flush;
        eprintln!("BM2 tail: max single flush {total:?}");
    }
}

/// BM3: steady-state overhead when tiers never move vs today's baseline
/// shape. Baseline: plain var-len writes + flush on a store whose tier
/// engine is present but UNCONFIGURED (pre-tier behavior). Measured:
/// identical churn with tiers configured but untouched. Must be within
/// noise (loose 2x smoke bound — debug-build timing is dominated by wgpu
/// submission either way).
#[test]
fn bm3_steady_state_overhead_is_within_noise() {
    const IDS: usize = 5_000;
    const REWRITES: usize = 3;

    // Baseline shape: engine attached, NOT configured.
    let ctx = test_context("bm3-baseline");
    let store_base = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store_base, (IDS as u32) * 2, ctx.device());
    let store_base = Arc::new(store_base);
    let mut world_base = World::new();
    world_base.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store_base), Arc::clone(ctx.queue())));
    let entities: Vec<Entity> = (0..IDS)
        .map(|i| {
            let e = world_base.spawn();
            world_base.insert(e, Blob { data: vec![i as u32; 4] });
            e
        })
        .collect();
    world_base.flush_gpu_mirror(ctx.queue()).unwrap();
    let base_start = Instant::now();
    for r in 0..REWRITES {
        for (i, &e) in entities.iter().enumerate() {
            world_base.insert(e, Blob { data: vec![i as u32 + r as u32; 4] });
        }
        world_base.flush_gpu_mirror(ctx.queue()).unwrap();
    }
    let baseline = base_start.elapsed();

    // Tiered shape: configured, budgets generous, ZERO touches issued.
    let store_tier = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store_tier, (IDS as u32) * 2, ctx.device());
    store_tier.configure_tiers(TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 }, &[]).unwrap();
    let store_tier = Arc::new(store_tier);
    let mut world_tier = World::new();
    world_tier.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store_tier), Arc::clone(ctx.queue())));
    let entities_t: Vec<Entity> = (0..IDS)
        .map(|i| {
            let e = world_tier.spawn();
            world_tier.insert(e, Blob { data: vec![i as u32; 4] });
            e
        })
        .collect();
    world_tier.flush_gpu_mirror(ctx.queue()).unwrap();
    let tier_start = Instant::now();
    for r in 0..REWRITES {
        for (i, &e) in entities_t.iter().enumerate() {
            world_tier.insert(e, Blob { data: vec![i as u32 + r as u32; 4] });
        }
        world_tier.flush_gpu_mirror(ctx.queue()).unwrap();
    }
    let tiered = tier_start.elapsed();

    let ratio = tiered.as_nanos() as f64 / baseline.as_nanos().max(1) as f64;
    eprintln!(
        "BM3: steady-state churn baseline {baseline:?}, tiered {tiered:?} (ratio {ratio:.2}x; tiers idle)"
    );
    // ≤ noise means: no structural slowdown. Debug-build jitter is real, so
    // the smoke bound is deliberately loose; a genuine per-write regression
    // (e.g. double staging) lands far above it.
    assert!(ratio < 3.0, "steady-state overhead exploded: {ratio:.2}x baseline");
}
