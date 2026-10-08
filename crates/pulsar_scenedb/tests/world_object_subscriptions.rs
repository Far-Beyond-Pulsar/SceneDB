//! Object subscriptions (`crates/pulsar_scenedb/src/object_subscriptions.rs`
//! and its hook sites in `world.rs` / `mut_guard.rs`): every write to a
//! watched object's components calls back inside the write with the new
//! value; nothing else does.

use std::sync::{Arc, Mutex};

use pulsar_scenedb::{
    component_id, register_component, ComponentChangeKind, Entity, ObjectEvent, World,
};

#[derive(Clone, Copy, PartialEq, Debug)]
struct Pos(f32);
#[derive(Clone, Copy, PartialEq, Debug)]
struct Health(u32);
/// Places a component entity in its object, like a host's owner component.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Owner(Entity);

/// What a subscriber saw, with the values copied out of the callback.
#[derive(Clone, Debug, PartialEq)]
enum Seen {
    Pos(Entity, ComponentChangeKind, Option<Pos>),
    Health(Entity, ComponentChangeKind, Option<Health>),
    Other(Entity, ComponentChangeKind),
    Despawned(Entity),
}

fn recorder(world: &mut World, object: Entity) -> Arc<Mutex<Vec<Seen>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    world
        .subscribe_object(object, move |obj, event| {
            let entry = match event {
                ObjectEvent::Despawned => Seen::Despawned(obj),
                ObjectEvent::Changed(change) => {
                    let value = change.value;
                    if change.component == component_id::<Pos>() {
                        Seen::Pos(change.entity, change.kind, value.and_then(|v| v.downcast_ref().copied()))
                    } else if change.component == component_id::<Health>() {
                        Seen::Health(change.entity, change.kind, value.and_then(|v| v.downcast_ref().copied()))
                    } else {
                        Seen::Other(change.entity, change.kind)
                    }
                }
            };
            log.lock().unwrap().push(entry);
        })
        .expect("live object");
    seen
}

fn take(seen: &Arc<Mutex<Vec<Seen>>>) -> Vec<Seen> {
    std::mem::take(&mut *seen.lock().unwrap())
}

fn owner_resolver(world: &World, entity: Entity) -> Entity {
    world.get::<Owner>(entity).map_or(entity, |owner| owner.0)
}

#[test]
fn every_write_path_calls_back_with_the_new_value() {
    use ComponentChangeKind::*;
    let mut world = World::new();
    let e = world.spawn();
    let seen = recorder(&mut world, e);

    world.insert(e, Pos(1.0)); // first insert
    world.insert(e, Pos(2.0)); // in-place overwrite
    world.get_mut::<Pos>(e).unwrap().0 = 3.0; // written through DerefMut
    let _ = world.get_mut::<Pos>(e).unwrap().0; // borrow only: not a write
    *world.get_mut::<Pos>(e).unwrap().into_inner() = Pos(9.0); // fired at hand-off, before this write
    assert_eq!(world.remove::<Pos>(e), Some(Pos(9.0)));

    assert_eq!(
        take(&seen),
        [
            Seen::Pos(e, Inserted, Some(Pos(1.0))),
            Seen::Pos(e, Inserted, Some(Pos(2.0))),
            Seen::Pos(e, Mutated, Some(Pos(3.0))),
            Seen::Pos(e, Mutated, Some(Pos(3.0))),
            Seen::Pos(e, Removed, Some(Pos(9.0))),
        ]
    );
}

#[test]
fn erased_writes_call_back_like_typed_ones() {
    use ComponentChangeKind::*;
    register_component::<Health>();
    let mut world = World::new();
    let e = world.spawn();
    let seen = recorder(&mut world, e);

    world.insert_dyn(e, Box::new(Health(5))).unwrap();
    {
        let guard = world.get_dyn_mut(e, component_id::<Health>()).unwrap();
        assert_eq!(guard.downcast_ref::<Health>(), Some(&Health(5))); // a read
    }
    world
        .get_dyn_mut(e, component_id::<Health>())
        .unwrap()
        .downcast_mut::<Health>()
        .unwrap()
        .0 = 6;
    world.remove_dyn(e, component_id::<Health>()).unwrap();

    assert_eq!(
        take(&seen),
        [
            Seen::Health(e, Inserted, Some(Health(5))),
            Seen::Health(e, Mutated, Some(Health(6))),
            Seen::Health(e, Removed, Some(Health(6))),
        ]
    );
}

#[test]
fn an_object_covers_its_component_entities_including_later_ones() {
    use ComponentChangeKind::*;
    let mut world = World::new();
    world.set_object_resolver(owner_resolver);
    let object = world.spawn();
    let other = world.spawn();
    let early = world.spawn_bundle((Owner(object), Health(1)));
    let seen = recorder(&mut world, object);

    world.get_mut::<Health>(early).unwrap().0 = 2;
    let later = world.spawn_bundle((Owner(object), Pos(0.5)));
    world.insert(object, Pos(7.0));
    // Another object, and an unowned entity, stay silent.
    let foreign = world.spawn_bundle((Owner(other), Health(1)));
    world.get_mut::<Health>(foreign).unwrap().0 = 9;
    world.insert(other, Pos(1.0));

    assert_eq!(world.object_of(later), object);
    assert_eq!(
        take(&seen),
        [
            Seen::Health(early, Mutated, Some(Health(2))),
            Seen::Other(later, Inserted),
            Seen::Pos(later, Inserted, Some(Pos(0.5))),
            Seen::Pos(object, Inserted, Some(Pos(7.0))),
        ]
    );

    // Despawning a component entity removes its components from the object.
    world.despawn(early);
    let mut removed = take(&seen);
    removed.sort_by_key(|s| format!("{s:?}"));
    assert_eq!(removed, [Seen::Health(early, Removed, None), Seen::Other(early, Removed)]);
}

#[test]
fn despawning_the_object_ends_its_subscriptions() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Pos(1.0));
    let seen = recorder(&mut world, e);
    let id = world.subscribe_object(e, |_, _| {}).unwrap();

    world.despawn(e);
    assert_eq!(
        take(&seen),
        [Seen::Pos(e, ComponentChangeKind::Removed, None), Seen::Despawned(e)]
    );
    assert!(!world.unsubscribe_object(id), "ended with the object");

    // The slot's next occupant is not the despawned object.
    let next = world.spawn();
    world.insert(next, Pos(2.0));
    assert!(take(&seen).is_empty());
    assert!(world.subscribe_object(e, |_, _| {}).is_none(), "a dead object cannot be watched");
}

#[test]
fn unsubscribing_stops_only_that_subscriber() {
    let mut world = World::new();
    let e = world.spawn();
    let first = recorder(&mut world, e);
    let second_seen = Arc::new(Mutex::new(0));
    let count = Arc::clone(&second_seen);
    let second = world
        .subscribe_object(e, move |_, _| *count.lock().unwrap() += 1)
        .unwrap();

    world.insert(e, Health(1));
    assert!(world.unsubscribe_object(second));
    assert!(!world.unsubscribe_object(second));
    world.insert(e, Health(2));

    assert_eq!(take(&first).len(), 2);
    assert_eq!(*second_seen.lock().unwrap(), 1);
}

#[test]
fn an_id_from_a_replaced_world_ends_nothing_on_the_new_one() {
    let mut old = World::new();
    let e = old.spawn();
    let stale = old.subscribe_object(e, |_, _| {}).unwrap();

    let mut world = World::new();
    let e = world.spawn();
    let seen = recorder(&mut world, e);
    assert!(!world.unsubscribe_object(stale));
    world.insert(e, Health(1));
    assert_eq!(take(&seen).len(), 1, "the new world's subscription still delivers");
}

#[derive(pulsar_scenedb_derive::Replicate, Default, Clone, Copy, PartialEq, Debug)]
struct Replicated {
    #[replicate(encoding = Pod, condition = Always)]
    value: f32,
}

#[test]
fn a_replicated_write_calls_back_with_the_new_value() {
    use pulsar_scenedb::{ComponentDelta, Delta, Replicable, ReplicationRegistry};
    let mut registry = ReplicationRegistry::new();
    Replicated::register_replication(&mut registry);

    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Replicated::default());
    let got = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&got);
    world
        .subscribe_object(e, move |_, event| {
            if let ObjectEvent::Changed(change) = event {
                let value = change.value.and_then(|v| v.downcast_ref::<Replicated>().copied());
                log.lock().unwrap().push((change.kind, value));
            }
        })
        .unwrap();

    let mut bytes = Vec::new();
    4.0f32.replicate_encode(&mut bytes);
    let delta = Delta {
        frame: 1,
        base_frame: 0,
        spawned: vec![],
        despawned: vec![],
        component_deltas: vec![ComponentDelta {
            entity: e,
            component_type: component_id::<Replicated>(),
            field_data: vec![bytes],
        }],
        events: vec![],
    };
    assert_eq!(delta.apply(&mut world, &registry), Ok(()));
    assert_eq!(
        *got.lock().unwrap(),
        [(ComponentChangeKind::Mutated, Some(Replicated { value: 4.0 }))]
    );
}

#[test]
fn a_world_without_subscriptions_never_resolves_objects() {
    fn panicking_resolver(_: &World, _: Entity) -> Entity {
        panic!("resolved an object with nobody subscribed");
    }
    let mut world = World::new();
    world.set_object_resolver(panicking_resolver);
    let e = world.spawn();
    world.insert(e, Pos(1.0));
    world.get_mut::<Pos>(e).unwrap().0 = 2.0;
    world.remove::<Pos>(e);
    world.despawn(e);
}
