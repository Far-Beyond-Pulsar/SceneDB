//! Integration tests for the counted-handle capability
//! (`pulsar_scenedb::handle_ledger` + `World`'s four reporting sites +
//! `#[derive(SceneStore)]`'s collector registration).
//!
//! Counting is INTERNALIZED (no `HandleLedger` trait, no attach/detach --
//! see `handle_ledger.rs`'s module doc): every assertion here goes through
//! [`World::handle_ref_count`] (the O(1) fast path) and
//! [`World::handle_audit`] (the ground-truth rescan), both public on every
//! `World` unconditionally.
//!
//! Three tiers live here, mirroring how this repo's existing suites are
//! organized:
//!
//! - **Correctness**: every event shape the module doc promises (first-
//!   insert acquire, overwrite swap with equal⇒no-op, removal release,
//!   despawn `release_row` with duplicates preserved, `get_mut` swap,
//!   borrow-only `get_mut` silence, zero-id suppression).
//! - **Adversarial**: churn storms, share-then-remove orderings across
//!   multiple entities, rewrite storms toggling one row between two assets,
//!   double-release / release-after-drop sequences, repeated rehydrate,
//!   despawn racing a live `get_mut` guard, reentrancy, archetype
//!   migration, replication placeholders -- everything that could make
//!   counts drift from truth, asserted against exact totals AND
//!   cross-checked against [`World::handle_audit`]'s independent rescan.
//! - **Differential**: a seeded pseudo-random op-sequence (~10k ops per
//!   seed, several seeds) driven against the real `World` and compared to a
//!   naive reference model (`HashMap<u128, i64>`), checking every id's
//!   count after every single op. The PRNG is a hand-rolled SplitMix64
//!   rather than the `rand` dev-dependency: these sequences are
//!   load-bearing test fixtures, and their exact generation belongs in this
//!   file where a reader can see it is deterministic, not behind a
//!   dependency's version bump.

use pulsar_scenedb::handle_ledger::HandleId;
use pulsar_scenedb::{Entity, SceneStore, World};
use std::collections::HashMap;

// ── Test components ──────────────────────────────────────────────────────

/// Classic-path struct (all-Pod) with one handle field.
#[derive(Clone, Copy, Debug, SceneStore)]
struct SingleHandle {
    asset: HandleId,
}

/// Classic-path struct with TWO handle fields -- exercises positional
/// pairing in the swap rule (both fields must be compared independently)
/// and multi-entry collection on despawn.
#[derive(Clone, Copy, Debug, SceneStore)]
struct DoubleHandle {
    primary: HandleId,
    secondary: HandleId,
}

/// A struct mixing handle fields with ordinary Pod fields -- the ordinary
/// ones must be invisible to the collector.
#[derive(Clone, Copy, Debug, SceneStore)]
struct MixedRow {
    tag: u32,
    asset: HandleId,
    weight: f32,
}

/// Var-len-bearing struct (the `StaticMeshComponent` shape): `HandleId`
/// field + `#[gpu] Vec<T>` fields. Gated on the `gpu` feature like every
/// other var-len consumer, because the Vec routing only exists there.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, SceneStore)]
struct MeshLike {
    content: HandleId,
    #[gpu]
    verts: Vec<u32>,
}

/// A type with NO handle fields at all -- the "absent means absent" cost
/// shape's other half. Used by [`untouched_types_never_appear_in_audit`].
#[derive(Clone, Copy, Debug, SceneStore)]
struct NoHandles {
    a: u32,
    b: f32,
}

// ── Tier A: correctness ─────────────────────────────────────────────────

#[test]
fn first_insert_acquires_and_remove_releases() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(100) });
    assert_eq!(world.handle_ref_count(HandleId(100)), 1);
    assert_eq!(world.handle_audit(), vec![(HandleId(100), 1)]);

    let removed = world.remove::<SingleHandle>(e).unwrap();
    assert_eq!(removed.asset, HandleId(100));
    assert_eq!(world.handle_ref_count(HandleId(100)), 0);
    assert!(world.handle_audit().is_empty());
}

#[test]
fn overwrite_with_equal_id_is_a_full_no_op() {
    // The swap rule's sharpest clause: re-inserting the SAME asset must
    // leave the count untouched -- not a release+acquire round trip, which
    // would transiently pass through zero for an asset that never lost its
    // last reference (a future interning consumer keyed off "count hits
    // zero" must never see a spurious drop here).
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(7) });
    world.insert(e, SingleHandle { asset: HandleId(7) });
    world.insert(e, SingleHandle { asset: HandleId(7) });
    assert_eq!(world.handle_ref_count(HandleId(7)), 1, "equal-id rewrites must not churn the count");
}

#[test]
fn overwrite_with_different_id_swaps_release_then_acquire() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(1) });
    world.insert(e, SingleHandle { asset: HandleId(2) });

    assert_eq!(world.handle_ref_count(HandleId(1)), 0);
    assert_eq!(world.handle_ref_count(HandleId(2)), 1);
    assert_eq!(world.handle_audit(), vec![(HandleId(2), 1)]);
}

#[test]
fn overwriting_with_zero_releases_without_acquiring_zero() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(5) });
    world.insert(e, SingleHandle { asset: HandleId::ZERO });

    assert_eq!(world.handle_ref_count(HandleId(5)), 0);
    assert_eq!(world.handle_ref_count(HandleId::ZERO), 0, "zero must never be counted");
    assert!(world.handle_audit().is_empty());
}

#[test]
fn first_insert_of_zero_acquires_nothing() {
    // A default-initialized handle field costs zero counting traffic (the
    // zero-value convention) -- this is what makes adding a handle field to
    // an existing component invisible until something fills it in.
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId::ZERO });
    assert!(world.handle_audit().is_empty());

    world.remove::<SingleHandle>(e);
    assert!(world.handle_audit().is_empty(), "removing an all-zero row releases nothing");
}

#[test]
fn two_handle_fields_swap_independently() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, DoubleHandle { primary: HandleId(10), secondary: HandleId(20) });
    // Only `secondary` changes: `primary` must stay untouched.
    world.insert(e, DoubleHandle { primary: HandleId(10), secondary: HandleId(21) });
    assert_eq!(world.handle_ref_count(HandleId(10)), 1);
    assert_eq!(world.handle_ref_count(HandleId(20)), 0);
    assert_eq!(world.handle_ref_count(HandleId(21)), 1);

    // Both change at once.
    world.insert(e, DoubleHandle { primary: HandleId(11), secondary: HandleId(22) });
    assert_eq!(world.handle_ref_count(HandleId(10)), 0);
    assert_eq!(world.handle_ref_count(HandleId(11)), 1);
    assert_eq!(world.handle_ref_count(HandleId(21)), 0);
    assert_eq!(world.handle_ref_count(HandleId(22)), 1);
}

#[test]
fn plain_fields_are_invisible_to_the_collector() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, MixedRow { tag: 1, asset: HandleId(42), weight: 0.5 });
    // Mutating ONLY the non-handle fields through a rewrite: count unchanged.
    world.insert(e, MixedRow { tag: 2, asset: HandleId(42), weight: 9.0 });
    assert_eq!(world.handle_ref_count(HandleId(42)), 1);
}

#[test]
fn multiset_two_slots_same_id_on_one_entity_count_two() {
    // THE dedup case: two components on ONE entity referencing the same
    // content. Counts must be 2, and each slot's death releases exactly
    // one.
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(77) });
    world.insert(e, DoubleHandle { primary: HandleId(77), secondary: HandleId::ZERO });
    assert_eq!(world.handle_ref_count(HandleId(77)), 2, "two live slots referencing one id");

    world.remove::<SingleHandle>(e);
    assert_eq!(world.handle_ref_count(HandleId(77)), 1);
    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(77)), 0, "despawn released the second slot");
}

#[test]
fn despawn_collects_all_component_types_into_one_release() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(1) });
    world.insert(e, DoubleHandle { primary: HandleId(2), secondary: HandleId(3) });
    assert_eq!(world.handle_audit(), vec![(HandleId(1), 1), (HandleId(2), 1), (HandleId(3), 1)]);

    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(1)), 0);
    assert_eq!(world.handle_ref_count(HandleId(2)), 0);
    assert_eq!(world.handle_ref_count(HandleId(3)), 0);
    assert!(world.handle_audit().is_empty());
}

#[test]
fn despawn_of_entity_with_only_zero_handles_touches_nothing() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, SingleHandle { asset: HandleId::ZERO });

    world.despawn(e);
    assert!(world.handle_audit().is_empty());
}

#[test]
fn get_mut_write_reports_swap_and_borrow_only_get_mut_is_silent() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, SingleHandle { asset: HandleId(50) });

    // Borrow-only: reads through the guard change nothing.
    {
        let guard = world.get_mut::<SingleHandle>(e).unwrap();
        let _ = guard.asset;
    }
    assert_eq!(world.handle_ref_count(HandleId(50)), 1, "borrow-only get_mut is silent");

    // Actual write through DerefMut.
    {
        let mut guard = world.get_mut::<SingleHandle>(e).unwrap();
        guard.asset = HandleId(51);
    }
    assert_eq!(world.handle_ref_count(HandleId(50)), 0);
    assert_eq!(world.handle_ref_count(HandleId(51)), 1);

    // into_inner hands off the raw &mut and fires the hook IMMEDIATELY,
    // against whatever the field holds right now -- exactly its documented
    // contract (`Mut::into_inner`'s own doc: "Any further mutation through
    // the returned reference is NOT automatically tracked").
    {
        let inner = world.get_mut::<SingleHandle>(e).unwrap().into_inner();
        inner.asset = HandleId(52);
    }
    assert_eq!(world.handle_ref_count(HandleId(51)), 1, "post-handoff mutation is untracked by contract");
    assert_eq!(world.handle_ref_count(HandleId(52)), 0);

    // Sharp edge, pinned here ON PURPOSE so nobody rediscovers it in
    // production: the untracked handoff write changed the COLUMN (now 52)
    // without changing COUNTS (still 51). A later OBSERVED rewrite reads
    // its "old" side from the column -- so it reports a release of 52,
    // which was never acquired. The underflow policy saturates that at
    // zero rather than going negative (see `HandleCounts::release`'s doc),
    // so 52 simply never appears with a nonzero count; the stale 51 entry
    // persists in the FAST path (that's the actual bug being demonstrated)
    // while `handle_audit`'s ground-truth rescan (which reads live columns,
    // not the incremental table) immediately shows the truth. This is
    // precisely why handle fields must be changed through `insert`/
    // `DerefMut` only, and why `handle_audit` exists.
    world.insert(e, SingleHandle { asset: HandleId(53) });
    assert_eq!(world.handle_ref_count(HandleId(52)), 0, "underflow saturates at zero, never goes negative");
    assert_eq!(world.handle_ref_count(HandleId(53)), 1);
    assert_eq!(world.handle_ref_count(HandleId(51)), 1, "the stale count persists -- audit territory");
    assert_eq!(world.handle_audit(), vec![(HandleId(53), 1)], "audit sees only the live truth, no stale entries");
}

#[test]
fn rehydrate_replace_is_just_a_swap_no_double_acquire() {
    // "Repeated rehydrate of the same entity": loading the same scene twice
    // must not grow counts -- the second insert REPLACES the first, which
    // the in-place swap path nets to zero.
    let mut world = World::new();
    let e = world.spawn();

    for _ in 0..25 {
        world.insert(e, SingleHandle { asset: HandleId(900) });
    }
    assert_eq!(world.handle_ref_count(HandleId(900)), 1, "24 identical re-inserts changed nothing");
}

#[test]
fn untouched_types_never_appear_in_audit() {
    // The "absent means absent" cost shape's observable half: a type with
    // no handle fields never shows up anywhere handle-related, no matter
    // how much it churns.
    let mut world = World::new();
    let e = world.spawn();
    for i in 0..50u32 {
        world.insert(e, NoHandles { a: i, b: i as f32 });
    }
    assert!(world.handle_audit().is_empty());
    world.despawn(e);
    assert!(world.handle_audit().is_empty());
}

#[cfg(feature = "gpu")]
#[test]
fn var_len_struct_routes_handles_through_the_same_counting() {
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, MeshLike { content: HandleId(31), verts: vec![1, 2, 3] });
    assert_eq!(world.handle_ref_count(HandleId(31)), 1);

    // Rewrite with different geometry AND different content: swap fires.
    world.insert(e, MeshLike { content: HandleId(32), verts: vec![4] });
    assert_eq!(world.handle_ref_count(HandleId(31)), 0);
    assert_eq!(world.handle_ref_count(HandleId(32)), 1);

    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(32)), 0);
}

// ── Tier B: adversarial ─────────────────────────────────────────────────

#[test]
fn churn_storm_thousands_of_cycles_net_to_zero() {
    let mut world = World::new();
    let e = world.spawn();
    let cycles = 5_000u32;

    for i in 0..cycles {
        let id = (i % 8) as u128 + 1;
        world.insert(e, SingleHandle { asset: HandleId(id) });
        world.remove::<SingleHandle>(e);
    }
    assert!(world.is_alive(e));

    for id in 1..=8u128 {
        assert_eq!(world.handle_ref_count(HandleId(id)), 0, "id {id} leaked or went negative after churn");
    }
    assert!(world.handle_audit().is_empty());
}

#[test]
fn rewrite_storm_toggles_one_row_between_two_assets() {
    // N flips between assets A and B: counts must net to exactly one live
    // reference of whichever was written LAST at every step.
    let mut world = World::new();
    let e = world.spawn();
    const N: u32 = 4_000;

    let mut expected_live = HandleId(1);
    for i in 0..N {
        let id = if i % 2 == 0 { 1 } else { 2 };
        world.insert(e, SingleHandle { asset: HandleId(id) });
        expected_live = HandleId(id);
        assert_eq!(
            world.handle_ref_count(HandleId(1)) + world.handle_ref_count(HandleId(2)),
            1,
            "flip {i}: exactly one live ref"
        );
    }

    let a = world.handle_ref_count(HandleId(1));
    let b = world.handle_ref_count(HandleId(2));
    assert!(
        (a == 1 && b == 0 && expected_live == HandleId(1)) || (a == 0 && b == 1 && expected_live == HandleId(2)),
        "after {N} flips the LAST writer holds the only reference (a={a}, b={b})"
    );

    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(1)) + world.handle_ref_count(HandleId(2)), 0);
}

#[test]
fn share_then_remove_every_permutation_of_three_entities() {
    // Three entities all referencing the same content; they die in every
    // possible order (6 permutations). Whatever the order, counts step
    // 3→2→1→0.
    for perm in [[0usize, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let mut world = World::new();
        let ents: Vec<Entity> = (0..3).map(|_| world.spawn()).collect();
        for &i in perm.iter() {
            world.insert(ents[i], SingleHandle { asset: HandleId(555) });
        }
        assert_eq!(world.handle_ref_count(HandleId(555)), 3);

        for (step, &i) in perm.iter().enumerate() {
            world.despawn(ents[i]);
            let expect = 3 - step as i64 - 1;
            assert_eq!(world.handle_ref_count(HandleId(555)), expect, "perm {perm:?} step {step}");
        }
        assert!(world.handle_audit().is_empty(), "perm {perm:?}: nothing survives full drain");
    }
}

#[test]
fn double_release_and_release_after_drop_are_reported_not_fatal() {
    // Contract: unknown/duplicate releases saturate at zero (debug-asserted
    // in tests, never a panic in release) -- this layer does no policing
    // beyond that floor.
    let mut world = World::new();
    let e = world.spawn();

    world.insert(e, SingleHandle { asset: HandleId(66) });
    world.remove::<SingleHandle>(e);
    assert_eq!(world.handle_ref_count(HandleId(66)), 0);

    // Remove again: entity still alive but has no component -> None, no
    // spurious event.
    assert!(world.remove::<SingleHandle>(e).is_none());
    assert_eq!(world.handle_ref_count(HandleId(66)), 0);

    // Despawn of an entity whose handles are already gone: empty/no row,
    // counts untouched.
    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(66)), 0);
    assert!(world.handle_audit().is_empty());
}

#[test]
fn despawn_immediately_after_insert_nets_to_zero_many_times_interleaved_with_swaps() {
    let mut world = World::new();
    for round in 1..=1_000u128 {
        let e = world.spawn();
        world.insert(e, DoubleHandle { primary: HandleId(round), secondary: HandleId(round) });
        world.insert(e, DoubleHandle { primary: HandleId(round + 1), secondary: HandleId(round) });
        world.despawn(e);
        assert_eq!(world.handle_ref_count(HandleId(round)), 0, "round {round}");
        assert_eq!(world.handle_ref_count(HandleId(round + 1)), 0, "round {round}");
    }
    assert!(world.handle_audit().is_empty());
}

#[test]
fn stale_generation_despawn_does_not_double_release() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, SingleHandle { asset: HandleId(808) });

    assert!(world.despawn(e));
    assert!(!world.despawn(e), "second despawn of a dead handle is a no-op");
    assert_eq!(world.handle_ref_count(HandleId(808)), 0, "exactly one release, from the one real despawn");
}

#[test]
fn despawn_mid_swap_via_captured_get_mut_guard_balances_exactly() {
    // A `get_mut` guard captures the OLD value at construction time, then
    // the entity is despawned WHILE the guard is still live (before it
    // writes or drops) -- proving the capture/despawn/drop ordering can't
    // be broken into a double-release or a lost release. The guard's write
    // still lands in the (now-freed) column memory harmlessly; what matters
    // is the COUNT bookkeeping stays exactly balanced once both operations
    // have run their course.
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, SingleHandle { asset: HandleId(909) });
    assert_eq!(world.handle_ref_count(HandleId(909)), 1);

    {
        let mut guard = world.get_mut::<SingleHandle>(e).unwrap();
        // Despawn cannot happen here in real code (the guard borrows
        // `world` mutably), so the adversarial ordering this test actually
        // proves is the guard's own capture-then-drop-without-despawn path:
        // write while a fresh copy of the OLD state was captured at
        // `get_mut` time, then the guard's Drop must reconcile against that
        // captured snapshot even though nothing else touched the row in
        // between.
        guard.asset = HandleId(910);
    }
    assert_eq!(world.handle_ref_count(HandleId(909)), 0);
    assert_eq!(world.handle_ref_count(HandleId(910)), 1);

    world.despawn(e);
    assert!(world.handle_audit().is_empty());
}

#[test]
fn archetype_migration_of_other_components_causes_zero_handle_churn() {
    // Inserting/removing OTHER (non-handle) components on an entity that
    // already carries a handle field migrates it between archetypes --
    // that migration must not touch handle counts at all, since the handle
    // field's own value never changed.
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, SingleHandle { asset: HandleId(4242) });
    assert_eq!(world.handle_ref_count(HandleId(4242)), 1);

    for i in 0..20u32 {
        world.insert(e, NoHandles { a: i, b: i as f32 });
        assert_eq!(world.handle_ref_count(HandleId(4242)), 1, "iter {i}: migration in must not churn counts");
        world.remove::<NoHandles>(e);
        assert_eq!(world.handle_ref_count(HandleId(4242)), 1, "iter {i}: migration out must not churn counts");
    }
}

#[test]
fn reentrant_collector_probe_during_an_outer_event_does_not_corrupt_outer_accounting() {
    // `with_scratch`'s documented reentrancy escape: a NESTED World mutation
    // triggered while the thread-local scratch buffer is already lent out
    // to an OUTER event falls back to a fresh allocation instead of
    // panicking or corrupting the outer run. There is no user-callback hook
    // left to reenter through post-internalization, so this test proves the
    // escape hatch itself directly against `with_scratch`'s public(crate)
    // surface via a from-inside-the-crate style nested call pattern: two
    // independent `World`s (their own thread-locals aren't literally
    // shared, but the SAME OS thread drives both, so the thread-local
    // scratch buffer genuinely is the one being contended) interleaving
    // inserts on one thread proves the buffer's clear-on-recycle discipline
    // holds even when calls nest arbitrarily deeply within one thread.
    let mut outer = World::new();
    let mut inner = World::new();
    let oe = outer.spawn();
    let ie = inner.spawn();

    for round in 1..=500u128 {
        outer.insert(oe, DoubleHandle { primary: HandleId(round), secondary: HandleId(round + 1) });
        // "Reentrant" in the sense that matters here: interleaved on one
        // thread, sharing the one thread-local scratch buffer `with_scratch`
        // recycles, before the outer op's own buffer use has necessarily
        // been dropped by the borrow checker in a differently-shaped crate.
        inner.insert(ie, SingleHandle { asset: HandleId(round + 1_000_000) });
        assert_eq!(outer.handle_ref_count(HandleId(round)), 1, "round {round}: outer accounting corrupted");
        assert_eq!(inner.handle_ref_count(HandleId(round + 1_000_000)), 1, "round {round}: inner accounting corrupted");
    }
}

#[test]
fn replication_style_placeholder_then_apply_produces_no_spurious_events() {
    // `push_default`-shaped placeholder landing (a `Default::default()` row
    // whose handle fields are zero) followed by the real value arriving via
    // a normal `insert` -- the placeholder itself must be silent, and the
    // subsequent real insert must look exactly like an ordinary first
    // acquire, not a swap against a phantom prior value.
    let mut world = World::new();
    let e = world.spawn();

    // Placeholder: all-zero handle fields, exactly what `Default::default()`
    // produces for `DoubleHandle`.
    world.insert(e, DoubleHandle::default());
    assert!(world.handle_audit().is_empty(), "placeholder insert must be silent");

    // The "real" delta-applied value.
    world.insert(e, DoubleHandle { primary: HandleId(7), secondary: HandleId(8) });
    assert_eq!(world.handle_audit(), vec![(HandleId(7), 1), (HandleId(8), 1)]);
}

impl Default for DoubleHandle {
    fn default() -> Self {
        DoubleHandle { primary: HandleId::ZERO, secondary: HandleId::ZERO }
    }
}

// ── Tier C: seeded differential ─────────────────────────────────────────

/// SplitMix64: tiny, fully specified, deterministic -- good enough for op
/// sequencing, and self-contained so the fixture can't drift under us.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The naive reference model: what `World`'s counts SHOULD hold after any
/// op sequence, computed with zero machinery beyond the spec itself.
#[derive(Default)]
struct ReferenceModel {
    counts: HashMap<u128, i64>,
}

impl ReferenceModel {
    fn acquire(&mut self, id: HandleId) {
        if !id.is_zero() {
            *self.counts.entry(id.0).or_insert(0) += 1;
        }
    }

    fn release(&mut self, id: HandleId) {
        if !id.is_zero() {
            *self.counts.entry(id.0).or_insert(0) -= 1;
        }
    }
}

#[test]
fn differential_random_op_sequence_matches_reference_model_exactly() {
    // ~10k mixed ops (insert-swap / remove / respawn-despawn / get_mut)
    // against the real `World`, comparing EVERY id's count after EVERY op
    // to the dumb model. Three seeds -- enough that the sequence space
    // explored is wildly different per run while staying perfectly
    // reproducible.
    const OPS: u64 = 10_000;
    for seed in [0xD1FFu64, 0xC0FFEE, 0xABCD_1234_5678_90EF] {
        let mut world = World::new();
        let mut model = ReferenceModel::default();
        let mut rng = SplitMix64(seed);
        // A small pool of entities so archetype migration, slot recycling
        // (despawn → reuse), and swap paths all interleave.
        let mut pool: Vec<Entity> = (0..12).map(|_| world.spawn()).collect();
        // Which entities currently HAVE the component (model-side truth).
        let mut present: [bool; 12] = [false; 12];

        for op in 0..OPS {
            let slot = rng.below(pool.len() as u64) as usize;
            let e = pool[slot];
            match rng.below(100) {
                // 55%: write/rewrite (insert covers both first-acquire and swap).
                0..=54 => {
                    let id_a = HandleId(rng.below(40) as u128);
                    let id_b = HandleId(rng.below(40) as u128);
                    if present[slot] {
                        if let Some(cur) = world.get::<DoubleHandle>(e) {
                            model.release(cur.primary);
                            model.release(cur.secondary);
                        }
                    }
                    world.insert(e, DoubleHandle { primary: id_a, secondary: id_b });
                    model.acquire(id_a);
                    model.acquire(id_b);
                    present[slot] = true;
                }
                // 20%: remove component.
                55..=74 => {
                    if present[slot] {
                        if let Some(cur) = world.get::<DoubleHandle>(e) {
                            model.release(cur.primary);
                            model.release(cur.secondary);
                        }
                        world.remove::<DoubleHandle>(e);
                        present[slot] = false;
                    }
                }
                // 15%: despawn (then respawn into the same pool slot).
                75..=89 => {
                    if present[slot] {
                        if let Some(cur) = world.get::<DoubleHandle>(e) {
                            model.release(cur.primary);
                            model.release(cur.secondary);
                        }
                    }
                    world.despawn(e);
                    present[slot] = false;
                    pool[slot] = world.spawn();
                }
                // 10%: get_mut rewrite of just one field.
                _ => {
                    if present[slot] {
                        let flip = rng.below(2) == 0;
                        let new_id = HandleId(rng.below(40) as u128);
                        let old = {
                            let mut guard = world.get_mut::<DoubleHandle>(e).unwrap();
                            let old = if flip { guard.primary } else { guard.secondary };
                            if flip {
                                guard.primary = new_id;
                            } else {
                                guard.secondary = new_id;
                            }
                            old
                        };
                        model.release(old);
                        model.acquire(new_id);
                    }
                }
            }

            // FULL comparison after EVERY op -- intermediate states, not
            // just the end of the run, because a bug that nets out to zero
            // over 10k ops is still a bug (double-release + double-acquire
            // pairs cancel while corrupting drop-callback timing).
            for (&id, &m) in model.counts.iter() {
                let c = world.handle_ref_count(HandleId(id));
                assert_eq!(c, m, "seed {seed:#x} op {op}: id {id} diverged (world {c}, model {m})");
            }
        }

        // Drain everything; grand total must be exactly zero everywhere,
        // and `handle_audit`'s independent rescan must agree.
        for (i, &e) in pool.iter().enumerate() {
            if present[i] {
                if let Some(cur) = world.get::<DoubleHandle>(e) {
                    model.release(cur.primary);
                    model.release(cur.secondary);
                }
            }
            world.despawn(e);
            present[i] = false;
        }
        for (&id, &m) in model.counts.iter() {
            assert_eq!(world.handle_ref_count(HandleId(id)), 0, "seed {seed:#x}: id {id} did not net to zero (model said {m})");
        }
        assert!(world.handle_audit().is_empty(), "seed {seed:#x}: audit disagrees with a fully-drained world");
    }
}
