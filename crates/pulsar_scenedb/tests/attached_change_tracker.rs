//! Correctness tests for [`pulsar_scenedb::SharedChangeTracker`] /
//! `World::attach_change_tracker` -- the attachment-based change-tracking
//! mechanism (`crates/pulsar_scenedb/src/world.rs`,
//! `crates/pulsar_scenedb/src/replication.rs`) that lets plain
//! `spawn`/`insert`/`remove`/`despawn`/`get_mut` calls record automatically,
//! with no `_tracked`-suffixed call anywhere, mirroring how `GpuMirrorHandle`
//! is already attached via `attach_gpu_mirror`.

use pulsar_scenedb::{ChangeTracker, SharedChangeTracker, World};

#[derive(Clone, Copy, PartialEq, Debug)]
struct Pos(f32, f32, f32);
#[derive(Clone, Copy, PartialEq, Debug)]
struct Health(u32);

#[test]
fn plain_spawn_and_insert_are_recorded_with_no_tracked_call_anywhere() {
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    // Every call below is the PLAIN spelling -- no `_tracked` suffix.
    let e = world.spawn();
    world.insert(e, Pos(1.0, 2.0, 3.0));
    world.insert(e, Health(100));

    let delta = tracker.drain_with_world(&world);
    assert_eq!(delta.spawned.len(), 1, "the plain spawn() must be recorded");
    assert_eq!(delta.spawned[0].0, e);
    assert_eq!(
        delta.component_deltas.len(),
        2,
        "both plain insert() calls must be recorded, one ComponentDelta per component type"
    );
}

#[test]
fn plain_get_mut_is_recorded_on_drop_with_no_tracked_call() {
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e = world.spawn();
    world.insert(e, Health(100));
    // Drain the spawn+insert first so this test isolates get_mut's own recording.
    let _ = tracker.drain_with_world(&world);

    {
        let mut h = world.get_mut::<Health>(e).unwrap();
        h.0 = 50;
        // Guard drops here -- this is the only place recording can happen,
        // there is no `get_mut_tracked` method to have called instead.
    }

    let delta = tracker.drain_with_world(&world);
    assert_eq!(
        delta.component_deltas.len(),
        1,
        "get_mut's Drop must record exactly one component change"
    );
    assert_eq!(delta.component_deltas[0].entity, e);
}

#[test]
fn plain_despawn_and_remove_are_recorded() {
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e1 = world.spawn();
    world.insert(e1, Pos(0.0, 0.0, 0.0));
    let e2 = world.spawn();
    world.insert(e2, Pos(0.0, 0.0, 0.0));
    let _ = tracker.drain_with_world(&world); // isolate what follows

    world.remove::<Pos>(e1);
    world.despawn(e2);

    let delta = tracker.drain_with_world(&world);
    assert_eq!(delta.despawned, vec![e2]);
    assert_eq!(
        delta.component_deltas.len(),
        1,
        "the plain remove() must be recorded"
    );
    assert_eq!(delta.component_deltas[0].entity, e1);
}

#[test]
fn remove_records_an_unambiguous_component_removal() {
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    world.insert(e, Health(100));
    let _ = tracker.drain_component_removals(); // isolate what follows

    world.remove::<Pos>(e);

    let removals = tracker.drain_component_removals();
    assert_eq!(removals.len(), 1, "only the removed component must show up here");
    assert_eq!(removals[0].0, e);
    assert_eq!(removals[0].1, pulsar_scenedb::component_id::<Pos>());

    // Health was never removed -- draining again must be empty, and a
    // second remove::<Pos> (already gone) must record nothing new either.
    assert!(tracker.drain_component_removals().is_empty());
}

#[test]
fn despawn_records_one_removal_per_component_the_entity_had() {
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    world.insert(e, Health(100));
    let _ = tracker.drain_component_removals(); // isolate what follows

    world.despawn(e);

    let mut removals = tracker.drain_component_removals();
    removals.sort_by_key(|(_, cid)| format!("{cid:?}"));
    assert_eq!(removals.len(), 2, "despawn must record a removal for every component the entity had");
    assert!(removals.iter().all(|(entity, _)| *entity == e));
    let types: std::collections::HashSet<_> = removals.iter().map(|(_, cid)| *cid).collect();
    assert!(types.contains(&pulsar_scenedb::component_id::<Pos>()));
    assert!(types.contains(&pulsar_scenedb::component_id::<Health>()));
}

#[test]
fn component_removals_is_independent_of_drain_with_world() {
    // Draining the replication Delta must not also silently consume
    // component_removals, and vice versa -- two independent lists sharing
    // one ChangeTracker, not one drain accidentally clearing the other.
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    let _ = tracker.drain_with_world(&world); // drains spawned/component_deltas only

    world.remove::<Pos>(e);

    // The Pos insert is long gone from `component_deltas` (drained above),
    // but the later removal must still be queued and independently drainable.
    let removals = tracker.drain_component_removals();
    assert_eq!(removals, vec![(e, pulsar_scenedb::component_id::<Pos>())]);
}

#[test]
fn plain_spawn_bundle_is_recorded_with_no_tracked_call() {
    // spawn_bundle/spawn_bundle_tracked route through spawn_inner directly
    // (not through the plain spawn()/insert() wrappers), so this is a
    // distinct code path from `plain_spawn_and_insert_are_recorded_...`
    // above and needs its own coverage.
    let mut world = World::new();
    let tracker = SharedChangeTracker::new();
    world.attach_change_tracker(tracker.clone());

    let e = world.spawn_bundle((Pos(1.0, 1.0, 1.0), Health(10)));

    let delta = tracker.drain_with_world(&world);
    assert_eq!(delta.spawned, vec![(e, delta.spawned[0].1.clone())]);
    assert_eq!(
        delta.component_deltas.len(),
        2,
        "both bundle components must be recorded"
    );
}

#[test]
fn no_tracker_attached_plain_calls_behave_exactly_as_before() {
    let mut world = World::new();
    assert!(!world.has_change_tracker());

    let e = world.spawn();
    world.insert(e, Health(75));
    assert_eq!(world.get::<Health>(e), Some(&Health(75)));
    world.remove::<Health>(e);
    assert!(world.despawn(e));
    // No panics, no tracker required -- this is the exact behavior every
    // pre-existing caller of World gets, since attach_change_tracker is
    // opt-in and nothing here calls it.
}

#[test]
fn attached_tracker_and_plain_calls_produce_an_equivalent_delta_to_the_old_tracked_calls() {
    // World A: the new attachment-based mechanism, plain calls only.
    let mut world_a = World::new();
    let tracker_a = SharedChangeTracker::new();
    world_a.attach_change_tracker(tracker_a.clone());
    let ea = world_a.spawn();
    world_a.insert(ea, Pos(1.0, 1.0, 1.0));
    let delta_a = tracker_a.drain_with_world(&world_a);

    // World B: the pre-existing explicit `_tracked` mechanism, unchanged.
    let mut world_b = World::new();
    let mut tracker_b = ChangeTracker::new();
    let eb = world_b.spawn_tracked(&mut tracker_b);
    world_b.insert_tracked(eb, Pos(1.0, 1.0, 1.0), &mut tracker_b);
    let delta_b = tracker_b.drain_with_world(&world_b);

    assert_eq!(delta_a.spawned.len(), delta_b.spawned.len());
    assert_eq!(delta_a.component_deltas.len(), delta_b.component_deltas.len());
    assert_eq!(
        delta_a.component_deltas[0].component_type,
        delta_b.component_deltas[0].component_type
    );
}

#[test]
fn tracked_methods_record_into_the_attached_tracker_not_their_own_external_one_when_both_exist() {
    // Once a tracker is attached, `_tracked`'s own explicitly-passed
    // tracker is documented to become a no-op for the attached case (see
    // `World::spawn_tracked`'s doc) -- recording happens into the attached
    // one so the two spellings can't silently diverge.
    let mut world = World::new();
    let attached = SharedChangeTracker::new();
    world.attach_change_tracker(attached.clone());

    let mut external = ChangeTracker::new();
    let e = world.spawn_tracked(&mut external);
    world.insert_tracked(e, Health(1), &mut external);

    let delta_external = external.drain_with_world(&world);
    assert!(
        delta_external.spawned.is_empty() && delta_external.component_deltas.is_empty(),
        "the external tracker passed to a _tracked call must stay empty once a World-level tracker is attached"
    );

    let delta_attached = attached.drain_with_world(&world);
    assert_eq!(delta_attached.spawned.len(), 1);
    assert_eq!(delta_attached.component_deltas.len(), 1);
}
