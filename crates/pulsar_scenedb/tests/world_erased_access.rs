//! Type-erased component access (`has_component`, `component_ids`,
//! `get_dyn`, `get_dyn_mut`) and the `ComponentRef`/`ComponentHandle`
//! reference types built on it.

use pulsar_scenedb::{component_id, ChangeRead, ComponentChangeKind, ComponentHandle, ComponentRef, World};

#[derive(Debug, PartialEq)]
struct Health(u32);
#[derive(Debug, PartialEq)]
struct Name(String);

#[test]
fn erased_reads_see_the_live_components() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(10));
    world.insert(e, Name("a".into()));

    let health = component_id::<Health>();
    assert!(world.has_component(e, health));
    let mut ids: Vec<_> = world.component_ids(e).collect();
    ids.sort_by_key(|id| id.0);
    let mut expected = vec![health, component_id::<Name>()];
    expected.sort_by_key(|id| id.0);
    assert_eq!(ids, expected);
    assert_eq!(world.get_dyn(e, health).unwrap().downcast_ref::<Health>(), Some(&Health(10)));

    world.despawn(e);
    assert!(!world.has_component(e, health));
    assert_eq!(world.component_ids(e).count(), 0);
    assert!(world.get_dyn(e, health).is_none());
}

#[test]
fn erased_writes_fire_the_same_hooks_as_typed_writes() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(10));
    world.subscribe::<Health>(e).unwrap();
    let mut journal = world.open_change_cursor::<Health>();

    world.get_dyn_mut(e, component_id::<Health>()).unwrap().downcast_mut::<Health>().unwrap().0 = 3;
    // A borrow-only erased guard is not a write.
    let _ = world.get_dyn_mut(e, component_id::<Health>()).unwrap().downcast_ref::<Health>().map(|h| h.0);

    assert_eq!(world.get::<Health>(e), Some(&Health(3)));
    let events = world.take_component_change_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ComponentChangeKind::Mutated);
    let mut changes = Vec::new();
    assert_eq!(world.read_changes(&mut journal, &mut changes), ChangeRead::Complete);
    assert_eq!(changes.len(), 1);
}

#[test]
fn references_resolve_only_while_the_target_lives() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Health(1));
    let typed = ComponentHandle::<Health>::new(e);
    let erased = typed.erase();
    assert_eq!(erased, ComponentRef::of::<Health>(e));
    assert_eq!(erased.typed::<Health>(), Some(typed));
    assert_eq!(erased.typed::<Name>(), None);

    typed.get_mut(&mut world).unwrap().0 = 2;
    assert_eq!(erased.get(&world).unwrap().downcast_ref::<Health>(), Some(&Health(2)));

    world.remove::<Health>(e);
    assert!(!typed.is_valid(&world) && !erased.is_valid(&world));

    // A new entity reusing the slot is not reachable through the old handle.
    world.despawn(e);
    let reused = world.spawn();
    world.insert(reused, Health(9));
    assert_eq!(reused.index(), e.index());
    assert!(typed.get(&world).is_none());
    assert!(erased.get_mut(&mut world).is_none());
}
