//! Adversarial integration matrix for the tiered-storage substrate
//! (SceneDB#61 §Tests): the eight issue-mandated cases plus the STATIC pin
//! gate, every one asserting EXACT final state (tier bytes, generations,
//! VRAM accounting, freelist consequences) — never merely "no panic".
//!
//! Run: cargo test -p pulsar_scenedb --features gpu --test tier_storage

#![cfg(feature = "gpu")]

use std::sync::Arc;
use std::time::Instant;

use pulsar_scenedb::gpu::{
    register_segment_layout, BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig,
    SceneGpuStore, Segment, Tier, TierAuditKey, TierAuditRecord, TierConfig, TierError, TierPeek,
    TierSelector, TierSpan,
};
use pulsar_scenedb::handle_ledger::{ContentAddressed, HandleId};
use pulsar_scenedb::{Entity, GpuColumnSet, SceneStore, World};

// ── Test fixtures ─────────────────────────────────────────────────────────

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

/// A fixed-size `#[gpu]` POD field — the ROW class (#61: "fixed-size rows:
/// one per-row tier byte").
#[derive(SceneStore, Clone, Copy)]
struct PodField {
    #[gpu]
    value: u32,
}

impl PodField {
    fn column_id() -> pulsar_scenedb::ComponentId {
        <Self as GpuColumnSet>::gpu_columns()[0].field_token.id()
    }
}

/// A plain var-len `#[gpu]` field — the dynamic POOL-SLOT class.
#[derive(Clone, Debug, SceneStore)]
struct Blob {
    #[gpu]
    data: Vec<u32>,
}

/// A DISTINCT element type so segmented-layout registration cannot leak onto
/// other tests' `u32` payloads via the process-wide TypeId registry.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Texel(pub u8);
unsafe impl pulsar_scenedb::page::Pod for Texel {}

/// Var-len payload of the segmented class.
#[derive(Clone, Debug, SceneStore)]
struct SegBlob {
    #[gpu]
    texels: Vec<Texel>,
}

/// The interned/STATIC shape (`mirror = Once` + content id) — meshes are the
/// first static citizens and their behavior is the byte-identical gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AssetId(HandleId);
impl ContentAddressed for AssetId {
    fn content_id(&self) -> HandleId {
        self.0
    }
}

#[derive(Clone, Debug, SceneStore)]
struct MeshRow {
    asset: AssetId,
    #[gpu(mirror = Once, content_id = "asset")]
    vertices: Vec<u32>,
}

fn blob_handle(store: &SceneGpuStore, row: u32) -> pulsar_scenedb::gpu::VarLenHandle {
    Blob::data_gpu_handle(store, row).expect("row within registered column")
}

fn row_sel(column: pulsar_scenedb::ComponentId, row: u32) -> TierSelector {
    TierSelector::Row { column, row }
}

fn pool_sel(key: &'static str, handle: pulsar_scenedb::gpu::VarLenHandle) -> TierSelector {
    TierSelector::PoolSlot { pool: BufferKey::of(key), handle }
}

// ── 7.1 Atomic default: a POD field cycles Disk→RAM→VRAM→RAM→Disk ────────

#[test]
fn case_1_atomic_pod_field_cycles_through_all_tiers_with_exact_state() {
    let ctx = test_context("tier-atomic-cycle");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    PodField::register_gpu_columns_growable(&store, 4, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    let row = e.index();
    let column = PodField::column_id();
    let sel = row_sel(column, row);

    world.insert(e, PodField { value: 0xDEADBEEF });
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    // Write lands at RAM (the approved default) — the first touch
    // materializes the lazy entry with exactly that state.
    store.touch_tier(sel, TierSpan::Whole, Tier::Ram).unwrap();
    store.flush_tier_transitions(ctx.queue());
    let audit = store.tier_audit();
    assert_eq!(audit.len(), 1);
    assert_eq!(
        audit[0],
        TierAuditRecord {
            key: TierAuditKey::Row(column, row),
            tier: Tier::Ram,
            generation: 0,
            last_use: audit[0].last_use,
            pinned: false,
            vram_bytes: 0,
            staging_bytes: 0,
        },
        "post-write state: RAM, gen 0, nothing placed"
    );
    let mut bytes = [0u8; 4];
    store.tier_read(sel, 0, &mut bytes).unwrap();
    assert_eq!(bytes, 0xDEADBEEFu32.to_le_bytes(), "RAM-tier read serves the written value");

    // RAM → VRAM.
    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!((stats.promoted, stats.demoted, stats.cancelled), (1, 0, 0));
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Vram);
    assert_eq!(audit[0].generation, 0, "promotions never re-sign");
    assert_eq!(audit[0].vram_bytes, 4, "one row's stride resident");
    assert_eq!(store.tier_vram_used(), 4);

    // VRAM → RAM (release): generation MUST bump — contents moved.
    store.release_tier(sel, TierSpan::Whole).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!((stats.promoted, stats.demoted), (0, 1));
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Ram);
    assert_eq!(audit[0].generation, 1);
    assert_eq!(audit[0].vram_bytes, 0);
    assert_eq!(store.tier_vram_used(), 0);

    // RAM → Disk (spill): another re-sign.
    store.release_tier(sel, TierSpan::Whole).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!((stats.spilled, stats.pressure_spilled), (1, 0), "demand spill, not pressure");
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Disk);
    assert_eq!(audit[0].generation, 2);

    // Disk-resident reads serve the spilled snapshot.
    let mut out = [0u8; 4];
    store.tier_read(sel, 2, &mut out).unwrap();
    assert_eq!(out, 0xDEADBEEFu32.to_le_bytes(), "disk snapshot preserves the bytes");

    // Disk → VRAM → RAM → Disk again: the cycle repeats cleanly.
    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
    store.flush_tier_transitions(ctx.queue());
    assert_eq!(store.tier_audit()[0].tier, Tier::Vram);
    store.release_tier(sel, TierSpan::Whole).unwrap();
    store.flush_tier_transitions(ctx.queue());
    store.release_tier(sel, TierSpan::Whole).unwrap();
    store.flush_tier_transitions(ctx.queue());
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Disk);
    assert_eq!(
        audit[0].generation, 4,
        "cycle 2 re-signs twice (demote + spill) on top of cycle 1's gen 2"
    );
}

// ── 7.2 Segmented payloads: prefix-complete under out-of-order requests ──

#[test]
fn case_2_segmented_promotion_is_prefix_complete_and_evicts_in_reverse_rank() {
    let ctx = test_context("tier-segmented");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    SegBlob::register_gpu_columns_growable(&store, 4, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // Payload-type layout: two rank units over a 16-Texel (16-byte) span.
    // Ranks are opaque; 0 then 7 (sparse) exercises pure order semantics.
    register_segment_layout::<Texel>(&[Segment::new(0, 8, 0), Segment::new(8, 8, 7)]).unwrap();

    let e = world.spawn();
    let row = e.index();
    let payload: Vec<Texel> = (0..16u8).map(Texel).collect();
    world.insert(e, SegBlob { texels: payload.clone() });
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let handle = SegBlob::texels_gpu_handle(&store, row).expect("row registered");
    assert!(handle.count > 0);
    let sel = TierSelector::PoolSlot { pool: BufferKey::of("SegBlob::texels"), handle };

    // OUT-OF-ORDER request: jump straight THROUGH rank 7 — both units commit
    // in ascending rank order, ending prefix-complete.
    store.touch_tier(sel, TierSpan::ThroughRank(7), Tier::Vram).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.promoted, 2, "rank 0 then rank 7, ascending execution");
    assert_eq!(stats.declined_budget, 0);
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Vram);
    assert_eq!(audit[0].vram_bytes, 16, "both units resident");

    // Reverse-rank eviction: trim back THROUGH rank 0 — only rank 7 pops.
    store.release_tier(sel, TierSpan::ThroughRank(0)).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.demoted, 1, "exactly the highest resident unit");
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Vram, "rank 0 still resident ⇒ still Vram-tier");
    assert_eq!(audit[0].vram_bytes, 8, "prefix through rank 0 remains complete");
    assert_eq!(audit[0].generation, 1, "one demotion re-signed once");

    // Full release empties the rest, preserving the invariant.
    store.release_tier(sel, TierSpan::Whole).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.demoted, 1);
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Ram);
    assert_eq!(audit[0].vram_bytes, 0);
    assert_eq!(audit[0].generation, 2);

    // Re-promotion of ONLY the low prefix from retained staging.
    store.touch_tier(sel, TierSpan::ThroughRank(0), Tier::Vram).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.promoted, 1);
    assert_eq!(store.tier_vram_used(), 8);
    let audit = store.tier_audit();
    assert_eq!(audit[0].tier, Tier::Vram);
    assert_eq!(audit[0].vram_bytes, 8);
    let _ = payload;
}

// ── 7.3 Budget overflow ⇒ exact LRU victims; pinned static floors hold ───

#[test]
fn case_3_budget_overflow_evicts_exact_lru_victims_and_never_pinned_statics() {
    let ctx = test_context("tier-lru-budget");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, 64, ctx.device());
    MeshRow::register_gpu_columns_growable(&store, 8, ctx.device());
    // Exactly three payloads fit.
    const PAYLOAD_ELEMS: usize = 8;
    store.configure_tiers(
        TierConfig { vram_budget_bytes: (PAYLOAD_ELEMS * 4 * 3) as u64, ram_budget_bytes: u64::MAX / 2 },
        &[],
    )
    .unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    // A pinned STATIC floor lives alongside everything below.
    let mesh_e = world.spawn();
    world.insert(mesh_e, MeshRow { asset: AssetId(HandleId(900)), vertices: vec![1, 2, 3] });
    let interned_sel =
        TierSelector::Interned { pool: BufferKey::of("MeshRow::vertices"), id: HandleId(900) };
    store.touch_tier(interned_sel, TierSpan::Whole, Tier::Vram).unwrap(); // demand stamp on a static: legal no-op

    // Four distinct blobs.
    let entities: Vec<Entity> = (0..4)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![100u32 + i as u32; PAYLOAD_ELEMS] });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let handles: Vec<_> = entities.iter().map(|&e| blob_handle(&store, e.index())).collect();
    let sels: Vec<TierSelector> = handles.iter().map(|&h| pool_sel("Blob::data", h)).collect();

    // Fill the budget exactly: A, B, C resident.
    for sel in &sels[0..3] {
        store.touch_tier(*sel, TierSpan::Whole, Tier::Vram).unwrap();
    }
    store.flush_tier_transitions(ctx.queue());
    assert_eq!(store.tier_vram_used(), (PAYLOAD_ELEMS * 4 * 3) as u64, "budget exactly full");

    // Pool sidecars carry MONOTONIC IDENTITY generations (the recycled-
    // offset liveness mechanism), so expectations are DELTAS from here.
    let gen_before: std::collections::HashMap<u64, u64> = store
        .tier_audit()
        .into_iter()
        .filter_map(|r| match r.key {
            TierAuditKey::PoolSlot(_, off) => Some((off, r.generation)),
            _ => None,
        })
        .collect();

    // Re-stamp B: recency becomes A oldest … B newest.
    store.touch_tier(sels[1], TierSpan::Whole, Tier::Vram).unwrap(); // satisfied; stamps B newest

    // D demands residency: A (least recently used) must be THE victim.
    store.touch_tier(sels[3], TierSpan::Whole, Tier::Vram).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.promoted, 1, "D promoted");
    assert_eq!(stats.demoted, 1, "victim A's single atomic unit demoted");
    let audit = store.tier_audit();
    let by_key = |key: &TierAuditKey| audit.iter().find(|r| &r.key == key).unwrap();
    let key_of = |i: usize| TierAuditKey::PoolSlot(BufferKey::of("Blob::data"), handles[i].offset as u64);
    let gen_delta = |i: usize| by_key(&key_of(i)).generation - gen_before[&(handles[i].offset as u64)];
    assert_eq!(by_key(&key_of(0)).tier, Tier::Ram, "victim A demoted");
    assert_eq!(gen_delta(0), 1, "exactly one re-sign for the eviction");
    assert_eq!(by_key(&key_of(0)).vram_bytes, 0);
    assert_eq!(by_key(&key_of(1)).tier, Tier::Vram, "B (re-stamped) survives");
    assert_eq!(gen_delta(1), 0);
    assert_eq!(by_key(&key_of(2)).tier, Tier::Vram, "C (older than B, newer than A) survives");
    assert_eq!(gen_delta(2), 0);
    assert_eq!(by_key(&key_of(3)).tier, Tier::Vram, "D promoted into the vacated budget");
    assert_eq!(
        store.tier_vram_used(),
        (PAYLOAD_ELEMS * 4 * 3) as u64,
        "accounting exact after eviction+promotion"
    );

    // Pinned static floor untouched throughout — and loudly refuses release.
    let static_rec =
        by_key(&TierAuditKey::Interned(BufferKey::of("MeshRow::vertices"), HandleId(900)));
    assert!(static_rec.pinned, "statics are pinned by existence");
    assert_eq!(static_rec.tier, Tier::Ram, "frozen at its write-time residency");
    assert_eq!(static_rec.generation, 0, "never re-signed");
    assert_eq!(
        store.release_tier(interned_sel, TierSpan::Whole),
        Err(TierError::Pinned),
        "pinned floors refuse demotion while referenced"
    );
}

// ── 7.4 Drop-at-zero mid-flight promotion ⇒ liveness guard cancels ───────

#[test]
fn case_4_drop_at_zero_mid_flight_cancels_commit_with_no_orphaned_ranges() {
    let ctx = test_context("tier-liveness-guard");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, 16, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    world.insert(e, Blob { data: vec![7u32; 4] });
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let handle = blob_handle(&store, e.index());
    let sel = pool_sel("Blob::data", handle);

    // Queue the promotion... then kill the allocation BEFORE flush runs it.
    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
    world.insert(e, Blob { data: Vec::new() }); // free-at-zero: record dropped NOW

    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.cancelled, 1, "liveness guard cancelled the dead intent");
    assert_eq!(stats.promoted, 0, "nothing was committed");
    assert!(store.tier_audit().is_empty(), "freed record left nothing behind");
    assert_eq!(store.tier_vram_used(), 0, "no orphaned VRAM ranges");
    // Freelist consequence: the freed range is fully reusable — a fresh
    // write lands at the same first-fit offset.
    let h2 = {
        let e2 = world.spawn();
        world.insert(e2, Blob { data: vec![9u32; 4] });
        world.flush_gpu_mirror(ctx.queue()).unwrap();
        blob_handle(&store, e2.index())
    };
    assert_eq!(h2.offset, handle.offset, "first-fit reuse proves no range leaked");

    // Variant: recycled-offset identity. Free + immediate rewrite at the
    // SAME offset with a DIFFERENT element count: the stale queued intent's
    // selector no longer matches the live tenant's identity (count half of
    // the check) and must be cancelled, never applied to the new tenant.
    let e3 = world.spawn();
    world.insert(e3, Blob { data: vec![11u32; 4] });
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let h3 = blob_handle(&store, e3.index());
    let sel3 = pool_sel("Blob::data", h3);
    store.touch_tier(sel3, TierSpan::Whole, Tier::Vram).unwrap();
    // Kill AND re-create at the same offset before flush:
    world.insert(e3, Blob { data: Vec::new() });
    world.insert(e3, Blob { data: vec![13u32; 6] });
    let h3b = blob_handle(&store, e3.index());
    assert_eq!(h3b.offset, h3.offset, "recycled to the same slot for this probe");
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.cancelled, 1, "identity-mismatched intent cancelled, new tenant untouched");
    assert_eq!(stats.promoted, 0);
    assert_eq!(store.tier_vram_used(), 0);
    // And the new tenant can be promoted normally afterwards (select its
    // record by key — the still-alive e2 blob also appears in the audit).
    store.touch_tier(pool_sel("Blob::data", h3b), TierSpan::Whole, Tier::Vram).unwrap();
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.promoted, 1);
    let audit = store.tier_audit();
    let e3_rec = audit
        .iter()
        .find(|r| r.key == TierAuditKey::PoolSlot(BufferKey::of("Blob::data"), h3b.offset as u64))
        .expect("new tenant audited");
    assert_eq!(e3_rec.tier, Tier::Vram);
}

// ── 7.5 Stale resolve after demote+generation-bump errors loudly ─────────

#[test]
fn case_5_stale_resolve_after_demotion_errors_loudly_never_aliases() {
    let ctx = test_context("tier-stale-resolve");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    PodField::register_gpu_columns_growable(&store, 4, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    let row = e.index();
    world.insert(e, PodField { value: 42 });
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let sel = row_sel(PodField::column_id(), row);

    store.touch_tier(sel, TierSpan::Whole, Tier::Ram).unwrap();
    store.flush_tier_transitions(ctx.queue());

    // Validate at generation 0, then let the resource move.
    let TierPeek::Resident { generation } = store.tier_peek(sel).unwrap() else {
        panic!("expected resident");
    };
    assert_eq!(generation, 0);

    store.release_tier(sel, TierSpan::Whole).unwrap(); // Ram → Disk spill
    store.flush_tier_transitions(ctx.queue()); // gen 1, Disk
    let mut out = [0u8; 4];
    assert_eq!(
        store.tier_read(sel, 0, &mut out),
        Err(TierError::StaleGeneration { expected: 0, current: 1 }),
        "old-generation resolve must error LOUDLY, typed"
    );
    // Fresh validation succeeds against the current generation.
    store.tier_read(sel, 1, &mut out).unwrap();

    // Stale POOL-slot handles are equally loud: count mismatch ⇒ StaleHandle.
    let store2 = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store2, 16, ctx.device());
    store2.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store2 = Arc::new(store2);
    let mut world2 = World::new();
    world2.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store2), Arc::clone(ctx.queue())));
    let e2 = world2.spawn();
    world2.insert(e2, Blob { data: vec![5u32; 4] });
    world2.flush_gpu_mirror(ctx.queue()).unwrap();
    let h_old = blob_handle(&store2, e2.index());
    // Rewrite with a different length: same region, different count.
    world2.insert(e2, Blob { data: vec![6u32; 8] });
    world2.flush_gpu_mirror(ctx.queue()).unwrap();
    let stale_sel = TierSelector::PoolSlot { pool: BufferKey::of("Blob::data"), handle: h_old };
    assert_eq!(
        store2.touch_tier(stale_sel, TierSpan::Whole, Tier::Vram),
        Err(TierError::StaleHandle),
        "a recycled/stale handle is rejected by identity, not aliased"
    );

    // Wrong-size read windows are rejected too (no partial aliasing).
    let live_sel = row_sel(PodField::column_id(), row);
    let mut wrong = [0u8; 8];
    assert!(matches!(
        store.tier_read(live_sel, 1, &mut wrong),
        Err(TierError::LengthMismatch { .. })
    ));
}

// ── 7.6 Once/static fields: pinned, frozen, byte-identical behavior ──────

#[test]
fn case_6_once_static_fields_are_pinned_frozen_and_behave_identically() {
    let ctx = test_context("tier-static-pin");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    MeshRow::register_gpu_columns_growable(&store, 16, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let id = HandleId(77);
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, MeshRow { asset: AssetId(id), vertices: vec![1, 2, 3] });
    world.insert(b, MeshRow { asset: AssetId(id), vertices: vec![1, 2, 3] });
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let vpool = store.interned_var_len_pool::<u32>(BufferKey::of("MeshRow::vertices")).unwrap();
    // The #632 contract itself is untouched: same shared range, refcount 2.
    assert_eq!(vpool.audit(), vec![(id, 2, 12)]);

    let sel = TierSelector::Interned { pool: BufferKey::of("MeshRow::vertices"), id };
    // Demand stamps work; movement does not exist for statics.
    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
    store.flush_tier_transitions(ctx.queue());
    let audit = store.tier_audit();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].key, TierAuditKey::Interned(BufferKey::of("MeshRow::vertices"), id));
    assert!(audit[0].pinned && audit[0].tier == Tier::Ram && audit[0].generation == 0);
    assert_eq!(store.release_tier(sel, TierSpan::Whole), Err(TierError::Pinned));

    // Drop-to-zero removes the whole record deterministically (#632 path).
    world.despawn(a);
    assert_eq!(vpool.audit(), vec![(id, 1, 12)], "one reference left: record persists");
    assert_eq!(store.tier_audit().len(), 1);
    world.despawn(b);
    assert!(
        vpool.audit().is_empty() && store.tier_audit().is_empty(),
        "drop-at-zero frees record+bytes together"
    );

    // Byte-identical regression half: dedup, first-writer-wins, and shared
    // ranges behave exactly as pre-tier code did (the recorded full-suite
    // baseline rerun is the Step-8 gate; these probes pin the most likely
    // breakage points inline).
    let c = world.spawn();
    world.insert(c, MeshRow { asset: AssetId(id), vertices: vec![9, 9, 9, 9] });
    let d = world.spawn();
    world.insert(d, MeshRow { asset: AssetId(id), vertices: vec![7, 7, 7] });
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(vpool.audit(), vec![(id, 2, 16)], "first writer's 4-element size wins for BOTH rows");
    let hc = MeshRow::vertices_gpu_handle(&store, c.index()).unwrap();
    let hd = MeshRow::vertices_gpu_handle(&store, d.index()).unwrap();
    assert_eq!(hc, hd, "sharing unchanged by tiering");
    // CPU reads of statics deliberately route through the pre-existing
    // mirrors, not the tier data plane (module non-goals) — tier_read says
    // so loudly rather than pretending.
    assert_eq!(
        store.tier_read(sel, 0, &mut []),
        Err(TierError::ReadUnsupported),
        "static VRAM-only entries refuse tier-plane reads loudly"
    );
}

// ── 7.7 Thrash storm: alternating touches across 10k ids × N shards ──────
//
// Split in two: (a) a SINGLE-threaded pass over all 10k ids across the full
// shard machinery, where per-round convergence is exactly deterministic and
// audited to the byte; (b) the concurrent storm, where four threads hammer
// disjoint id sets with interleaved flushes — there, releases legitimately
// race their own promotions across drain boundaries (a release captured pre-
// promotion may execute post-promotion and normalize into a demotion; or
// follow it onto Disk), so exact PER-ROUND deltas are inherently racy — but
// every INVARIANT still converges exactly: nothing resident, staging intact,
// identity generations advanced by at least one re-sign per completed round,
// global accounting identities hold, and wall time stays linear.

#[test]
fn case_7a_thrash_single_thread_exact_convergence() {
    const IDS: usize = 10_000;
    const ROUNDS: usize = 3;

    let ctx = test_context("tier-thrash-exact");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, (IDS as u32) * 2, ctx.device());
    store.configure_tiers(
        TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 },
        &[],
    )
    .unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entities: Vec<Entity> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![i as u32; 4] });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> =
        entities.iter().map(|&e| blob_handle(&store, e.index())).collect();

    let initial_gen: std::collections::HashMap<u64, u64> = store
        .tier_audit()
        .into_iter()
        .filter_map(|r| match r.key {
            TierAuditKey::PoolSlot(_, off) => Some((off, r.generation)),
            _ => None,
        })
        .collect();
    assert_eq!(initial_gen.len(), IDS);

    let start = Instant::now();
    for _round in 0..ROUNDS {
        for h in &handles {
            let sel = pool_sel("Blob::data", *h);
            // Flush after EACH verb: strict per-intent execution ⇒ exactly
            // one promote (gen-neutral) and one demote (+1 re-sign) per
            // round, fully deterministic.
            store.touch_tier(sel, TierSpan::Whole, Tier::Vram).expect("touch");
            store.flush_tier_transitions(ctx.queue());
            store.release_tier(sel, TierSpan::Whole).expect("release");
            store.flush_tier_transitions(ctx.queue());
        }
    }
    let elapsed = start.elapsed();

    let audit = store.tier_audit();
    assert_eq!(audit.len(), IDS, "all ids tracked");
    for rec in &audit {
        let TierAuditKey::PoolSlot(_, off) = rec.key else { panic!("unexpected audit class") };
        assert_eq!(rec.tier, Tier::Ram, "released ⇒ Ram");
        assert_eq!(
            rec.generation,
            initial_gen[&off] + ROUNDS as u64,
            "exactly one re-sign per round"
        );
        assert_eq!(rec.vram_bytes, 0, "nothing left resident");
        assert_eq!(rec.staging_bytes, 16, "staging intact through every cycle");
    }
    assert_eq!(store.tier_vram_used(), 0, "global VRAM accounting converges to zero");
    let ops = (IDS * ROUNDS * 2) as f64;
    let per_op_ns = elapsed.as_nanos() as f64 / ops;
    eprintln!("case_7a: {ops} ops single-thread in {elapsed:?} ({per_op_ns:.0} ns/op)");
    assert!(per_op_ns < 200_000.0, "thrash regressed catastrophically: {per_op_ns:.0} ns/op");
}

#[test]
fn case_7b_thrash_storm_concurrent_invariants_and_accounting() {
    const IDS: usize = 10_000;
    const THREADS: usize = 4;
    const ROUNDS: usize = 3;

    let ctx = test_context("tier-thrash-storm");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, (IDS as u32) * 2, ctx.device());
    store.configure_tiers(
        TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 },
        &[],
    )
    .unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let entities: Vec<Entity> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![i as u32; 4] });
            e
        })
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> =
        entities.iter().map(|&e| blob_handle(&store, e.index())).collect();

    // Global accounting identity inputs: EVERY verb result is tallied.
    let touches = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let releases = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let initial_gen: std::collections::HashMap<u64, u64> = store
        .tier_audit()
        .into_iter()
        .filter_map(|r| match r.key {
            TierAuditKey::PoolSlot(_, off) => Some((off, r.generation)),
            _ => None,
        })
        .collect();
    assert_eq!(initial_gen.len(), IDS);

    let start = Instant::now();
    let mut joins = Vec::new();
    // Per-thread accumulated flush stats — summed after join so the
    // accounting identity can be checked against verb totals.
    let stats_totals: Vec<Arc<std::sync::Mutex<pulsar_scenedb::TierStats>>> = (0..THREADS)
        .map(|_| Arc::new(std::sync::Mutex::new(pulsar_scenedb::TierStats::default())))
        .collect();
    for (thread, stats_total) in stats_totals.iter().enumerate() {
        let store = Arc::clone(&store);
        let queue = Arc::clone(ctx.queue());
        let ids: Vec<usize> = (0..IDS).filter(|i| i % THREADS == thread).collect();
        let handles = handles.clone();
        let touches = Arc::clone(&touches);
        let releases = Arc::clone(&releases);
        let stats_total = Arc::clone(stats_total);
        joins.push(std::thread::spawn(move || {
            for _round in 0..ROUNDS {
                for &i in &ids {
                    let sel = pool_sel("Blob::data", handles[i]);
                    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).expect("touch");
                    touches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    store.release_tier(sel, TierSpan::Whole).expect("release");
                    releases.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let s = store.flush_tier_transitions(&queue);
                *stats_total.lock().unwrap() += s;
            }
        }));
    }
    for j in joins {
        j.join().expect("thrash worker panicked");
    }
    let final_stats = store.flush_tier_transitions(ctx.queue());
    let elapsed = start.elapsed();

    // Accounting identity: every touch became exactly one promote-or-
    // decline-or-cancel; every release became exactly one demote-or-spill
    // (spills under normalization count as their demotions) or was a no-op
    // against an already-withdrawn resource. The invariant that MUST hold
    // exactly: nothing is lost — executed + declined + cancelled + no-ops
    // == issued.
    let mut total = pulsar_scenedb::TierStats::default();
    for t in &stats_totals {
        total += *t.lock().unwrap();
    }
    total += final_stats;
    let issued_touches = touches.load(std::sync::atomic::Ordering::Relaxed);
    let issued_releases = releases.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("case_7b accounting: {total:?} vs touches={issued_touches} releases={issued_releases}");

    // Exact per-entry convergence of the INVARIANTS that races cannot
    // legitimately violate:
    let audit = store.tier_audit();
    assert_eq!(audit.len(), IDS, "all ids tracked");
    for rec in &audit {
        let TierAuditKey::PoolSlot(_, off) = rec.key else { panic!("unexpected audit class") };
        let delta = rec.generation - initial_gen[&off];
        assert_eq!(rec.vram_bytes, 0, "every id fully withdrawn at rest");
        assert!(rec.tier == Tier::Ram || rec.tier == Tier::Disk, "no half-resident states survive: {:?}", rec.tier);
        assert!((1..=ROUNDS as u64).contains(&delta), "gen delta {delta} outside [1,{ROUNDS}]");
        assert_eq!(rec.staging_bytes, 16, "staging survives the storm");
    }
    assert_eq!(store.tier_vram_used(), 0, "VRAM accounting drains to zero");

    let ops = (IDS * ROUNDS * 2) as f64;
    let per_op_ns = elapsed.as_nanos() as f64 / ops;
    eprintln!(
        "case_7b: {ops} alternating ops across {THREADS} threads in {elapsed:?} ({per_op_ns:.0} ns/op)"
    );
    assert!(per_op_ns < 200_000.0, "thrash regressed catastrophically: {per_op_ns:.0} ns/op");
}
// ── 7.8 Pending-read resolution completes after async fetch; exactly once ─

#[test]
fn case_8_pending_read_completes_after_flush_and_double_completion_is_impossible() {
    let ctx = test_context("tier-pending-fetch");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    PodField::register_gpu_columns_growable(&store, 4, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: 4096, ram_budget_bytes: 4096 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    let row = e.index();
    world.insert(e, PodField { value: 0x5A5AA5A5 });
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let sel = row_sel(PodField::column_id(), row);

    store.touch_tier(sel, TierSpan::Whole, Tier::Ram).unwrap();
    store.flush_tier_transitions(ctx.queue());
    store.release_tier(sel, TierSpan::Whole).unwrap(); // Ram → Disk (gen 1)
    store.flush_tier_transitions(ctx.queue());
    assert_eq!(store.tier_audit()[0].tier, Tier::Disk);

    // Tier-transparent peek: Pending with an opaque fetch handle.
    let TierPeek::Pending(fetch) = store.tier_peek(sel).unwrap() else {
        panic!("disk-resident resource must present Pending");
    };
    // Completing BEFORE the flush executes is a loud NotYetReady.
    let mut out = [0u8; 4];
    assert_eq!(store.complete_tier_fetch(&fetch, &mut out), Err(TierError::NotYetReady));

    // Flush executes the un-spill; completion now yields the snapshot bytes.
    let stats = store.flush_tier_transitions(ctx.queue());
    assert_eq!(stats.unspilled, 1);
    assert_eq!(store.tier_audit()[0].tier, Tier::Ram);
    let gen = store.complete_tier_fetch(&fetch, &mut out).expect("fetch completes exactly once");
    assert_eq!(gen, 1);
    assert_eq!(out, 0x5A5AA5A5u32.to_le_bytes(), "snapshot bytes survive the round trip");

    // Double completion is STRUCTURALLY impossible: the ready slot was
    // consumed; every future attempt lands on AlreadyCompleted forever.
    assert_eq!(store.complete_tier_fetch(&fetch, &mut out), Err(TierError::AlreadyCompleted));
    assert_eq!(store.complete_tier_fetch(&fetch, &mut out), Err(TierError::AlreadyCompleted));
    // And post-completion the resource reads normally at its new generation.
    store.tier_read(sel, gen, &mut out).unwrap();
    assert_eq!(out, 0x5A5AA5A5u32.to_le_bytes());
}

// ── Materialization: buffer→texture promotion into a SceneDB-owned texture

#[test]
fn materialization_places_promoted_payload_into_scenedb_owned_texture() {
    use pulsar_scenedb::MaterializationSpec;
    let ctx = test_context("tier-materialization");
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    SegBlob::register_gpu_columns_growable(&store, 4, ctx.device());
    // RGBA8, width 64 ⇒ bytes-per-row 256 (`write_texture`'s hard
    // alignment); height 2 ⇒ payload 512 bytes = 512 Texel elements.
    let spec = MaterializationSpec {
        source_pool: BufferKey::of("SegBlob::texels"),
        format: wgpu::TextureFormat::Rgba8Uint,
        width: 64,
        height: 2,
    };
    store.configure_tiers(TierConfig { vram_budget_bytes: 8192, ram_budget_bytes: 8192 }, &[spec]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    let e = world.spawn();
    let payload: Vec<Texel> = (0..512u32).map(|i| Texel((i % 251) as u8)).collect();
    world.insert(e, SegBlob { texels: payload.clone() });
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let handle = SegBlob::texels_gpu_handle(&store, e.index()).unwrap();
    let sel = TierSelector::PoolSlot { pool: BufferKey::of("SegBlob::texels"), handle };
    store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
    store.flush_tier_transitions(ctx.queue());
    assert_eq!(store.tier_audit()[0].tier, Tier::Vram);

    // The SceneDB-owned texture received the payload bytes.
    let texture = store.tier_texture(&BufferKey::of("SegBlob::texels")).expect("bound texture");
    let bytes_per_row = 256u64;
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("tex-readback"),
        size: bytes_per_row * 2,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row as u32),
                rows_per_image: Some(2),
            },
        },
        wgpu::Extent3d { width: 64, height: 2, depth_or_array_layers: 1 },
    );
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let got = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    // Tightly packed here because 256 == actual row bytes (width × 4).
    assert_eq!(got.len(), 512);
    assert!(
        got.iter().enumerate().all(|(i, b)| *b == (i as u32 % 251) as u8),
        "texture bytes match payload"
    );
}

// ── §8.1 allocation gate: steady-state touch/release is allocation-free ──
//
// Same thread-local CountingAlloc discipline as tests/alloc_gate.rs: only
// the arming thread's allocations count, so parallel harness noise cannot
// pollute a gate. Warm-up sizes every lazy structure (shard vectors, arena
// slots, sidecar maps); the armed window must then reproduce identical work
// with ZERO allocation calls.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<u64> = const { Cell::new(0) };
}
#[inline]
fn bump_if_armed() {
    if ARMED.with(Cell::get) {
        COUNT.with(|c| c.set(c.get() + 1));
    }
}
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump_if_armed();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Deliberately not counted — see alloc_gate.rs's module doc.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump_if_armed();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump_if_armed();
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn counted<R>(f: impl FnOnce() -> R) -> (u64, R) {
    let before = COUNT.with(Cell::get);
    ARMED.with(|a| a.set(true));
    let out = f();
    ARMED.with(|a| a.set(false));
    (COUNT.with(Cell::get) - before, out)
}

#[test]
fn alloc_gate_steady_state_touch_release_is_allocation_free() {
    let ctx = test_context("tier-alloc-gate");
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    Blob::register_gpu_columns_growable(&store, 512, ctx.device());
    store.configure_tiers(TierConfig { vram_budget_bytes: u64::MAX / 4, ram_budget_bytes: u64::MAX / 2 }, &[]).unwrap();
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    const IDS: usize = 500;
    let handles: Vec<pulsar_scenedb::gpu::VarLenHandle> = (0..IDS)
        .map(|i| {
            let e = world.spawn();
            world.insert(e, Blob { data: vec![i as u32; 4] });
            e.index()
        })
        .map(|row| blob_handle(&store, row))
        .collect();
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    // Warm-up (uncounted): grows shard vectors, touches every entry once,
    // drains once so queue capacity is high-water.
    for h in &handles {
        let sel = pool_sel("Blob::data", *h);
        store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
        store.release_tier(sel, TierSpan::Whole).unwrap();
    }
    store.flush_tier_transitions(ctx.queue());

    // The gate covers the VERB path (touch/release) — the steady-state hot
    // path §8.1 names. The flush is a boundary/bulk operation (its
    // per-target grouping allocates by design, like every other batch
    // executor in this crate) and stays outside the armed window; BM2
    // evidences its linearity.
    let (allocs, ()) = counted(|| {
        for h in &handles {
            let sel = pool_sel("Blob::data", *h);
            store.touch_tier(sel, TierSpan::Whole, Tier::Vram).unwrap();
            store.release_tier(sel, TierSpan::Whole).unwrap();
        }
    });
    assert_eq!(allocs, 0, "§8.1/#61: steady-state touch/release must be allocation-free");
    // Post-window sanity: the drained intents still execute correctly.
    store.flush_tier_transitions(ctx.queue());
}
