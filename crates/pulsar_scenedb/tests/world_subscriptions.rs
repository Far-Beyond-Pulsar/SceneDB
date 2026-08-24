//! Correctness tests for the per-`(Entity, ComponentType)` subscription
//! system (SceneDB#47, `crates/pulsar_scenedb/src/subscriptions.rs` +
//! its hook sites in `world.rs`).
//!
//! Mirrors the GPU mirror's own test shape (`Mut`'s dispatch-on-drop): a
//! subscriber is armed for one key, the World is mutated through every real
//! path (`insert`, re-insert, `get_mut` written through `DerefMut`,
//! borrow-only `get_mut`, `remove`, `despawn`, bundle spawn), and each test
//! asserts exactly what fired -- and what deliberately didn't.

use pulsar_scenedb::{ComponentChangeEvent, ComponentChangeKind, SubscriptionId, World};

#[derive(Clone, Copy, PartialEq, Debug)]
struct Pos(f32, f32, f32);
#[derive(Clone, Copy, PartialEq, Debug)]
struct Health(u32);

fn drain(world: &mut World) -> Vec<ComponentChangeEvent> {
    world.take_component_change_events()
}

/// The one-subscriber-per-key shape the properties panel uses: subscribe,
/// mutate, drain, assert.
#[test]
fn insert_fires_exactly_once_for_the_subscribed_key() {
    let mut world = World::new();
    let e = world.spawn();
    let sub = world.subscribe::<Pos>(e).expect("live entity subscribes");
    assert!(drain(&mut world).is_empty(), "arming alone delivers nothing");

    world.insert(e, Pos(1.0, 2.0, 3.0));

    let events = drain(&mut world);
    assert_eq!(events.len(), 1, "one insert = exactly one event");
    assert_eq!(events[0].subscription, sub);
    assert_eq!(events[0].entity, e);
    assert_eq!(events[0].kind, ComponentChangeKind::Inserted);
    // Drain empties: nothing is redelivered.
    assert!(drain(&mut world).is_empty());
}

/// THE contract a cache-until-signaled consumer depends on: only an actual
/// write through `DerefMut` is a mutation; a borrow-only `get_mut` must fire
/// nothing, or every redraw would re-pull for no reason.
#[test]
fn deref_mut_write_fires_but_borrow_only_get_mut_fires_nothing() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(100));
    let sub = world.subscribe::<Health>(e).unwrap();
    let _ = drain(&mut world); // isolate the get_muts below

    // Borrow-only: reads through Deref, never DerefMut.
    {
        let h = world.get_mut::<Health>(e).unwrap();
        let _read = *h;
        let _again = h.0; // field read via Deref
    }
    assert!(
        drain(&mut world).is_empty(),
        "a get_mut that never wrote through DerefMut fires NOTHING"
    );

    // Real write: through DerefMut, both spellings.
    {
        let mut h = world.get_mut::<Health>(e).unwrap();
        h.0 = 50;
    }
    {
        let mut h = world.get_mut::<Health>(e).unwrap();
        *h = Health(25);
    }

    let events = drain(&mut world);
    assert_eq!(events.len(), 2, "exactly one event per real mutation");
    assert!(events.iter().all(|ev| ev.subscription == sub));
    assert!(
        events
            .iter()
            .all(|ev| ev.kind == ComponentChangeKind::Mutated),
        "get_mut writes report Mutated, got {:?}",
        events.iter().map(|ev| ev.kind).collect::<Vec<_>>()
    );
}

#[test]
fn remove_then_reinsert_reports_removed_then_inserted() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    world.subscribe::<Pos>(e).unwrap();
    let _ = drain(&mut world);

    world.remove::<Pos>(e);
    let events = drain(&mut world);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ComponentChangeKind::Removed);

    // Re-insert after removal fires again -- and the SAME subscription sees
    // it, with no resubscribe needed (the subscription outlives the
    // component's absence).
    world.insert(e, Pos(9.0, 9.0, 9.0));
    let events = drain(&mut world);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ComponentChangeKind::Inserted);

    // Removing a component the entity doesn't have is a no-op, not a change.
    assert!(world.remove::<Health>(e).is_none());
    assert!(drain(&mut world).is_empty());
}

#[test]
fn unrelated_entity_and_component_stay_silent() {
    let mut world = World::new();
    let watched = world.spawn();
    let other_entity = world.spawn();
    world.subscribe::<Pos>(watched).unwrap();
    let _ = drain(&mut world);

    // Same component type, different entity: silent for this subscriber...
    world.insert(other_entity, Pos(1.0, 1.0, 1.0));
    assert!(drain(&mut world).is_empty());

    // Same entity, different component type: silent too -- granularity is
    // the exact `(Entity, ComponentType)` pair.
    world.insert(watched, Health(10));
    assert!(drain(&mut world).is_empty());

    // ...but a second subscriber on that other key does hear it.
    let other_sub = world.subscribe::<Pos>(other_entity).unwrap();
    world.get_mut::<Pos>(other_entity).unwrap().0 = 5.0;
    let events = drain(&mut world);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].subscription, other_sub);
    assert_eq!(events[0].kind, ComponentChangeKind::Mutated);
}

#[test]
fn multiple_subscribers_on_one_key_each_get_their_own_event() {
    let mut world = World::new();
    let e = world.spawn();
    let sub_a = world.subscribe::<Pos>(e).unwrap();
    let sub_b = world.subscribe::<Pos>(e).unwrap();
    assert_ne!(sub_a, sub_b);
    let _ = drain(&mut world);

    world.insert(e, Pos(1.0, 1.0, 1.0));

    let mut subs_seen: Vec<SubscriptionId> =
        drain(&mut world).iter().map(|ev| ev.subscription).collect();
    subs_seen.sort();
    let mut expected = vec![sub_a, sub_b];
    expected.sort();
    assert_eq!(subs_seen, expected, "both watchers of the key were delivered");
}

#[test]
fn unsubscribe_is_idempotent_and_stops_future_events() {
    let mut world = World::new();
    let e = world.spawn();
    let sub = world.subscribe::<Health>(e).unwrap();

    assert!(world.unsubscribe(sub), "first unsubscribe removes it");
    assert!(!world.unsubscribe(sub), "second is a no-op, not an error");

    world.insert(e, Health(7));
    assert!(drain(&mut world).is_empty());
}

#[test]
fn despawn_delivers_removed_per_component_then_disarms_everything() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Pos(1.0, 1.0, 1.0));
    world.insert(e, Health(3));
    let pos_sub = world.subscribe::<Pos>(e).unwrap();
    let health_sub = world.subscribe::<Health>(e).unwrap();
    let _ = drain(&mut world);

    assert!(world.despawn(e));

    let events = drain(&mut world);
    let kinds: Vec<(ComponentChangeKind, SubscriptionId)> = events
        .iter()
        .map(|ev| (ev.kind, ev.subscription))
        .collect();
    assert_eq!(
        kinds.len(),
        2,
        "one Removed per component the entity actually had"
    );
    assert!(kinds.contains(&(ComponentChangeKind::Removed, pos_sub)));
    assert!(kinds.contains(&(ComponentChangeKind::Removed, health_sub)));

    // Everything on the entity is disarmed now: the recycled slot can be
    // respawned and mutated without waking any dead subscription.
    let e2 = world.spawn(); // likely reuses e's slot under a new generation
    world.insert(e2, Pos(0.0, 0.0, 0.0));
    world.get_mut::<Pos>(e2).unwrap().1 = 4.0;
    assert!(
        drain(&mut world).is_empty(),
        "despawn auto-cleaned the subscriptions"
    );
    assert!(!world.unsubscribe(pos_sub), "already disarmed by despawn");
}

#[test]
fn subscribing_before_first_insert_arms_until_that_insert() {
    let mut world = World::new();
    let e = world.spawn();
    let sub = world.subscribe::<Health>(e).unwrap();
    assert!(drain(&mut world).is_empty());

    world.insert(e, Health(1));
    let events = drain(&mut world);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].subscription, sub);
    assert_eq!(events[0].kind, ComponentChangeKind::Inserted);
}

#[test]
fn dead_entities_cannot_subscribe_and_stale_handles_never_fire() {
    let mut world = World::new();
    let e = world.spawn();
    assert!(world.despawn(e));
    assert!(
        world.subscribe::<Pos>(e).is_none(),
        "dead entity: nothing to subscribe to"
    );

    // Respawn into (probably) the same slot under a bumped generation: a
    // hypothetical pre-despawn subscription would be inert anyway, and the
    // fresh entity's changes go only to subscriptions armed for IT.
    let e2 = world.spawn();
    let _sub2 = world.subscribe::<Pos>(e2).unwrap();
    world.insert(e2, Pos(1.0, 1.0, 1.0));
    let events = drain(&mut world);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].entity, e2);
}

/// A consumer that stops draining must not grow the queue without bound:
/// past [`pulsar_scenedb::subscriptions`]' cap the oldest events drop and
/// are counted, and the counter stays visible even after a later drain.
#[test]
fn pending_queue_is_bounded_with_an_accounted_drop_counter() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(0));
    world.subscribe::<Health>(e).unwrap();
    let _ = drain(&mut world); // isolate the flood below

    const OVERFLOW_BY: u32 = 512;
    // One Mutated event per iteration (MAX_PENDING_EVENTS + slack total).
    for i in 0..(66_048 + OVERFLOW_BY) {
        world.get_mut::<Health>(e).unwrap().0 = i;
    }

    assert!(
        world.dropped_component_change_events() > 0,
        "overflow dropped OLDEST events, not silently"
    );
    let drained = drain(&mut world);
    assert!(
        drained.len() <= 65_536,
        "queue never exceeds the cap (got {})",
        drained.len()
    );
    // The drop count is cumulative bookkeeping, not queue state.
    assert!(world.dropped_component_change_events() > 0);
    // Draining caught up: nothing pending anymore.
    assert_eq!(world.pending_component_change_events(), 0);
}

/// `into_inner` hands off a raw `&mut T` with no guard left to observe later
/// writes, so it fires immediately by construction -- the documented escape
/// hatch, tested here so the guarantee can't silently regress.
#[test]
fn into_inner_fires_once_at_handoff() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(100));
    world.subscribe::<Health>(e).unwrap();
    let _ = drain(&mut world);

    let raw = world.get_mut::<Health>(e).unwrap().into_inner();
    *raw = Health(1); // NOT automatically tracked -- but the handoff already fired

    let events = drain(&mut world);
    assert_eq!(events.len(), 1, "into_inner fired exactly once at handoff");
    assert_eq!(events[0].kind, ComponentChangeKind::Mutated);
}

/// Bundles route through `push_new_component`, not `insert_inner` -- same
/// delivery contract, verified here so the second write path can't drift.
#[test]
fn bundle_insert_delivers_one_inserted_per_subscribed_component() {
    let mut world = World::new();
    let e = world.spawn();
    let s_pos = world.subscribe::<Pos>(e).unwrap();
    let s_health = world.subscribe::<Health>(e).unwrap();
    assert!(drain(&mut world).is_empty(), "arming alone delivers nothing");

    // `insert_bundle` pushes each component via `push_new_component`.
    world.insert_bundle(e, (Pos(1.0, 2.0, 3.0), Health(50)));

    let events = drain(&mut world);
    assert_eq!(events.len(), 2, "one Inserted per subscribed bundle component");
    assert!(
        events.iter().all(|ev| ev.kind == ComponentChangeKind::Inserted),
        "got {:?}",
        events.iter().map(|ev| ev.kind).collect::<Vec<_>>()
    );
    assert_eq!(events[0].entity, e);
    assert_eq!(events[1].entity, e);
    assert_ne!(events[0].component, events[1].component);

    let subs_seen: Vec<SubscriptionId> = events.iter().map(|ev| ev.subscription).collect();
    assert!(subs_seen.contains(&s_pos) && subs_seen.contains(&s_health));

    // And a plain `spawn_bundle` BEFORE any subscription arms stays silent,
    // as with every other pre-arm write.
    let e2 = world.spawn_bundle((Pos(0.0, 0.0, 0.0), Health(0)));
    assert!(drain(&mut world).is_empty());
    let _sub_late = world.subscribe::<Pos>(e2).unwrap();
    world.get_mut::<Pos>(e2).unwrap().0 = 7.0;
    let events = drain(&mut world);
    assert_eq!(events.len(), 1, "only post-arm writes fire");
    assert_eq!(events[0].kind, ComponentChangeKind::Mutated);
}
