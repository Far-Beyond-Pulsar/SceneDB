//! Type-erased insertion and removal (`World::insert_dyn`,
//! `World::remove_dyn`) run the same write path as typed `insert`/`remove`:
//! the same stored value, archetype, handle-ledger counts and change-journal
//! entries.

use pulsar_scenedb::handle_ledger::HandleId;
use pulsar_scenedb::{
    component_id, register_component, ChangeRead, ComponentChangeKind, InsertDynError, SceneStore,
    World,
};
use std::any::Any;

#[derive(Clone, Copy, Debug, PartialEq, SceneStore)]
struct Mesh {
    asset: HandleId,
    flags: u32,
}

#[derive(Debug, PartialEq)]
struct Label(String);

#[derive(Debug, PartialEq)]
struct NeverRegistered(u8);

fn boxed<T: Any + Send + Sync>(value: T) -> Box<dyn Any + Send + Sync> {
    Box::new(value)
}

/// Everything a write hook leaves behind, for comparing two worlds.
#[derive(Debug, PartialEq)]
struct Observed {
    mesh: Option<Mesh>,
    label: Option<String>,
    asset_refs: i64,
    journal: Vec<(ComponentChangeKind,)>,
}

fn observe(world: &mut World, e: pulsar_scenedb::Entity, cursor: &mut pulsar_scenedb::ChangeCursor) -> Observed {
    let mut changes = Vec::new();
    assert_eq!(world.read_changes(cursor, &mut changes), ChangeRead::Complete);
    Observed {
        mesh: world.get::<Mesh>(e).copied(),
        label: world.get::<Label>(e).map(|l| l.0.clone()),
        asset_refs: world.handle_ref_count(HandleId(7)) + world.handle_ref_count(HandleId(8)) * 100,
        journal: changes.iter().map(|c| (c.kind,)).collect(),
    }
}

#[test]
fn erased_insert_replace_and_remove_match_typed_writes() {
    register_component::<Mesh>();
    register_component::<Label>();

    let mut typed = World::new();
    let mut erased = World::new();
    let te = typed.spawn();
    let ee = erased.spawn();
    // Same entity index in both worlds, so the observations are comparable.
    assert_eq!(te, ee);
    let mut tc = typed.open_change_cursor::<Mesh>();
    let mut ec = erased.open_change_cursor::<Mesh>();

    // First insert (archetype migration).
    typed.insert(te, Label("a".into()));
    typed.insert(te, Mesh { asset: HandleId(7), flags: 1 });
    assert_eq!(erased.insert_dyn(ee, boxed(Label("a".into()))).unwrap(), component_id::<Label>());
    assert_eq!(
        erased.insert_dyn(ee, boxed(Mesh { asset: HandleId(7), flags: 1 })).unwrap(),
        component_id::<Mesh>()
    );
    let first = observe(&mut typed, te, &mut tc);
    assert_eq!(first, observe(&mut erased, ee, &mut ec));
    assert_eq!(first.asset_refs, 1);
    assert_eq!(first.journal, vec![(ComponentChangeKind::Inserted,)]);

    // In-place replacement with a different handle swaps ledger counts.
    typed.insert(te, Mesh { asset: HandleId(8), flags: 2 });
    erased.insert_dyn(ee, boxed(Mesh { asset: HandleId(8), flags: 2 })).unwrap();
    let replaced = observe(&mut typed, te, &mut tc);
    assert_eq!(replaced, observe(&mut erased, ee, &mut ec));
    assert_eq!(replaced.asset_refs, 100);
    assert_eq!(replaced.mesh, Some(Mesh { asset: HandleId(8), flags: 2 }));

    // Removal returns the owned value and releases its handles.
    let typed_removed = typed.remove::<Mesh>(te).unwrap();
    let erased_removed = erased.remove_dyn(ee, component_id::<Mesh>()).unwrap();
    assert_eq!(erased_removed.downcast_ref::<Mesh>(), Some(&typed_removed));
    let removed = observe(&mut typed, te, &mut tc);
    assert_eq!(removed, observe(&mut erased, ee, &mut ec));
    assert_eq!(removed.mesh, None);
    assert_eq!(removed.asset_refs, 0);
    assert_eq!(removed.journal, vec![(ComponentChangeKind::Removed,)]);
    assert_eq!(removed.label.as_deref(), Some("a"), "other components survive the migration");

    // A second removal finds nothing.
    assert!(erased.remove_dyn(ee, component_id::<Mesh>()).is_none());
}

#[test]
fn erased_and_typed_values_share_one_column() {
    register_component::<Mesh>();
    let mut world = World::new();
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, Mesh { asset: HandleId::default(), flags: 1 });
    world.insert_dyn(b, boxed(Mesh { asset: HandleId::default(), flags: 2 })).unwrap();
    let mut flags: Vec<u32> = world.query::<&Mesh>().map(|(_, m)| m.flags).collect();
    flags.sort();
    assert_eq!(flags, vec![1, 2]);
}

#[test]
fn erased_insert_refuses_unregistered_types_and_dead_entities() {
    register_component::<Label>();
    let mut world = World::new();
    let e = world.spawn();

    let err = world.insert_dyn(e, boxed(NeverRegistered(3))).unwrap_err();
    assert!(matches!(err, InsertDynError::Unregistered { .. }));
    assert_eq!(err.into_value().downcast_ref::<NeverRegistered>(), Some(&NeverRegistered(3)));
    assert_eq!(world.component_ids(e).count(), 0, "a refused insert changes nothing");

    world.despawn(e);
    let err = world.insert_dyn(e, boxed(Label("x".into()))).unwrap_err();
    assert!(matches!(err, InsertDynError::DeadEntity { entity, .. } if entity == e));
    assert_eq!(err.into_value().downcast_ref::<Label>(), Some(&Label("x".into())));
}

#[test]
fn bundle_spawn_acquires_handles_like_insert() {
    let mut world = World::new();
    let e = world.spawn_bundle((Mesh { asset: HandleId(7), flags: 0 }, Label("b".into())));
    assert_eq!(world.handle_ref_count(HandleId(7)), 1);
    world.despawn(e);
    assert_eq!(world.handle_ref_count(HandleId(7)), 0);
}
