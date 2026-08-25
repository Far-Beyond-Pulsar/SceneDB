//! Benchmarks for the content-dedup work (Pulsar-Native#632/#658), matching
//! `massive_mesh_load_stress.rs`'s existing `Instant`-timed idiom rather
//! than introducing a criterion dev-dependency nothing else in this crate
//! uses. Run with `cargo test --release --test content_dedup_benchmarks --
//! --nocapture --test-threads=1` to see the printed numbers; these are
//! smoke-asserted (loose upper bounds, not tight regression gates) so they
//! stay green in CI while still catching a genuine correctness-shaped
//! performance collapse (e.g. an accidental O(n²)).
//!
//! - BM1: handle-count event throughput (acquire/release pairs).
//! - BM5: despawn storm, 10k mixed entities.
//! - BM6: resident-bytes invariant — N references of one asset vs. N
//!   distinct assets, proving the dedup actually collapses residency
//!   (assertable, not just fast).

#![cfg(feature = "gpu")]

use pulsar_scenedb::gpu::{BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::handle_ledger::{ContentAddressed, HandleId};
use pulsar_scenedb::{Entity, SceneStore, World};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AssetId(HandleId);
impl ContentAddressed for AssetId {
    fn content_id(&self) -> HandleId {
        self.0
    }
}

#[derive(Clone, Copy, Debug, SceneStore)]
struct SingleHandleRow {
    asset: HandleId,
}

#[derive(Clone, Debug, SceneStore)]
struct MeshRow {
    asset: AssetId,
    #[gpu(mirror = Once, content_id = "asset")]
    vertices: Vec<u32>,
}

fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("content-dedup-benchmarks"),
        ..Default::default()
    }))
    .expect("device");
    (Arc::new(device), Arc::new(queue))
}

/// BM1: 1M acquire/release pairs on a plain `HandleId` field (the A2
/// internalized-counting path, no GPU/interning involved at all).
#[test]
fn bm1_handle_count_event_throughput() {
    let mut world = World::new();
    let e = world.spawn();
    const OPS: u32 = 1_000_000;

    let start = Instant::now();
    for i in 0..OPS {
        world.insert(e, SingleHandleRow { asset: HandleId((i as u128 % 64) + 1) });
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / OPS as f64;
    eprintln!("BM1: {OPS} acquire/release-shaped inserts in {elapsed:?} ({ns_per_op:.1} ns/op)");
    // Loose smoke bound: this path is a handful of HashMap probes and a
    // couple of archetype-column writes, not I/O — 5us/op would already be
    // 50x slower than expected and worth investigating, not a hard perf gate.
    assert!(ns_per_op < 5_000.0, "BM1 regressed badly: {ns_per_op:.1} ns/op");
}

/// BM5: despawn storm — 10k entities, half sharing GPU-mirrored var-len
/// content, half plain, all despawned in one pass.
#[test]
fn bm5_despawn_storm_10k_mixed_entities() {
    let (device, queue) = test_device();
    let ctx = EngineGpuContext::new(device.clone(), queue.clone());
    let mut store = SceneGpuStore::new(&ctx, SceneGpuConfig { classes: vec![], tombstone_headroom: 8, max_cells_metadata: 16 });
    MeshRow::register_gpu_columns_growable(&mut store, 1024, &device);
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), queue));

    const N: usize = 10_000;
    let entities: Vec<Entity> = (0..N)
        .map(|i| {
            let e = world.spawn();
            let id = HandleId((i as u128 % 50) + 1); // heavy sharing: 200 refs/id
            world.insert(e, MeshRow { asset: AssetId(id), vertices: vec![1, 2, 3, 4] });
            e
        })
        .collect();

    let start = Instant::now();
    for e in &entities {
        world.despawn(*e);
    }
    let elapsed = start.elapsed();
    let ns_per_despawn = elapsed.as_nanos() as f64 / N as f64;
    eprintln!("BM5: despawned {N} entities in {elapsed:?} ({ns_per_despawn:.1} ns/despawn)");

    let vpool = store.interned_var_len_pool::<u32>(BufferKey::of("MeshRow::vertices")).unwrap();
    assert!(vpool.audit().is_empty(), "despawn storm must not leak any interned residency");
    assert!(ns_per_despawn < 100_000.0, "BM5 regressed badly: {ns_per_despawn:.1} ns/despawn");
}

/// BM6: resident-bytes invariant. N references of ONE shared asset must
/// resolve to the SAME resident byte total as ONE reference — proving the
/// dedup actually collapses residency, not merely that it runs fast. Also
/// asserts the contrasting N-distinct-assets case scales linearly, so the
/// "collapse" isn't an artifact of a broken audit.
#[test]
fn bm6_resident_bytes_invariant_shared_vs_distinct() {
    let (device, queue) = test_device();
    let ctx = EngineGpuContext::new(device.clone(), queue.clone());
    let mut store = SceneGpuStore::new(&ctx, SceneGpuConfig { classes: vec![], tombstone_headroom: 8, max_cells_metadata: 16 });
    MeshRow::register_gpu_columns_growable(&mut store, 1024, &device);
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(&queue)));

    const N: usize = 500;
    const ELEMS_PER_ASSET: usize = 64;
    let shared_id = HandleId(777);
    for _ in 0..N {
        let e = world.spawn();
        world.insert(e, MeshRow { asset: AssetId(shared_id), vertices: vec![1u32; ELEMS_PER_ASSET] });
    }
    let vpool = store.interned_var_len_pool::<u32>(BufferKey::of("MeshRow::vertices")).unwrap();
    let audit = vpool.audit();
    assert_eq!(audit.len(), 1, "N references of ONE asset must collapse to ONE interned entry");
    let shared_bytes = audit[0].2;
    assert_eq!(shared_bytes, (ELEMS_PER_ASSET * 4) as u64, "resident bytes = ONE asset's size, not N times it");
    assert_eq!(audit[0].1, N as u64, "refcount reflects all N references even though bytes didn't multiply");

    // Contrast: N DISTINCT assets must resident N times the bytes (proves
    // the collapse above is real dedup, not an audit that just always
    // reports 1 entry regardless of what's actually referenced).
    let mut world2 = World::new();
    let mut store2 = SceneGpuStore::new(&ctx, SceneGpuConfig { classes: vec![], tombstone_headroom: 8, max_cells_metadata: 16 });
    MeshRow::register_gpu_columns_growable(&mut store2, 1024, &device);
    let store2 = Arc::new(store2);
    world2.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store2), queue));
    for i in 0..N {
        let e = world2.spawn();
        world2.insert(e, MeshRow { asset: AssetId(HandleId(1000 + i as u128)), vertices: vec![1u32; ELEMS_PER_ASSET] });
    }
    let vpool2 = store2.interned_var_len_pool::<u32>(BufferKey::of("MeshRow::vertices")).unwrap();
    let audit2 = vpool2.audit();
    assert_eq!(audit2.len(), N, "N distinct assets must resident as N interned entries");
    let distinct_total_bytes: u64 = audit2.iter().map(|(_, _, bytes)| bytes).sum();
    assert_eq!(distinct_total_bytes, shared_bytes * N as u64, "N distinct assets cost N times what N shared references cost");

    eprintln!(
        "BM6: {N} shared refs -> {shared_bytes} resident bytes; {N} distinct assets -> {distinct_total_bytes} resident bytes ({}x collapse)",
        distinct_total_bytes / shared_bytes.max(1)
    );
}
