//! Correctness tests for multi-reader change journals
//! (`crates/pulsar_scenedb/src/change_journal.rs` + its hook sites in
//! `world.rs`): every real write path is recorded once, a borrow-only
//! `get_mut` is not, and independent readers never consume each other's
//! changes.

use pulsar_scenedb::{ChangeRead, ComponentChange, ComponentChangeKind, World};

#[derive(Clone, Copy, PartialEq, Debug)]
struct Pos(f32, f32, f32);
#[derive(Clone, Copy, PartialEq, Debug)]
struct Health(u32);

fn read(world: &World, cursor: &mut pulsar_scenedb::ChangeCursor) -> Vec<ComponentChange> {
    let mut out = Vec::new();
    assert_eq!(world.read_changes(cursor, &mut out), ChangeRead::Complete);
    out
}

fn kinds(changes: &[ComponentChange]) -> Vec<ComponentChangeKind> {
    changes.iter().map(|c| c.kind).collect()
}

#[test]
fn every_write_path_is_recorded_once_in_order() {
    use ComponentChangeKind::*;
    let mut world = World::new();
    let mut cursor = world.open_change_cursor::<Pos>();

    let a = world.spawn();
    world.insert(a, Pos(0.0, 0.0, 0.0)); // new component: archetype migration
    world.insert(a, Pos(1.0, 0.0, 0.0)); // in-place overwrite
    world.get_mut::<Pos>(a).unwrap().0 = 2.0; // written through DerefMut
    let _ = world.get_mut::<Pos>(a).unwrap().0; // borrow only: not a change
    *world.get_mut::<Pos>(a).unwrap().into_inner() = Pos(3.0, 0.0, 0.0);
    world.remove::<Pos>(a);
    let b = world.spawn_bundle((Pos(4.0, 0.0, 0.0), Health(1)));
    world.despawn(b);

    let changes = read(&world, &mut cursor);
    assert_eq!(
        kinds(&changes),
        [Inserted, Inserted, Mutated, Mutated, Removed, Inserted, Removed]
    );
    let entities: Vec<_> = changes.iter().map(|c| c.entity).collect();
    assert_eq!(entities, [a, a, a, a, a, b, b]);
    assert!(read(&world, &mut cursor).is_empty(), "a read advances the cursor");
}

#[test]
fn journals_are_per_component_type() {
    let mut world = World::new();
    let mut pos = world.open_change_cursor::<Pos>();
    let e = world.spawn();
    world.insert(e, Health(3));
    world.get_mut::<Health>(e).unwrap().0 = 4;
    assert!(read(&world, &mut pos).is_empty(), "Health writes are not Pos changes");

    let mut health = world.open_change_cursor::<Health>();
    assert!(read(&world, &mut health).is_empty(), "a new cursor starts after existing history");
    world.despawn(e);
    assert_eq!(kinds(&read(&world, &mut health)), [ComponentChangeKind::Removed]);
}

#[test]
fn readers_do_not_consume_each_others_changes() {
    let mut world = World::new();
    let mut first = world.open_change_cursor::<Pos>();
    let mut second = world.open_change_cursor::<Pos>();
    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));

    assert_eq!(read(&world, &mut first).len(), 1);
    world.get_mut::<Pos>(e).unwrap().1 = 1.0;
    assert_eq!(read(&world, &mut first).len(), 1);
    assert_eq!(read(&world, &mut second).len(), 2, "the slower reader still sees both");
}

#[test]
fn subscriptions_and_journals_coexist() {
    let mut world = World::new();
    let e = world.spawn();
    world.subscribe::<Pos>(e).unwrap();
    let mut cursor = world.open_change_cursor::<Pos>();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    assert_eq!(world.take_component_change_events().len(), 1);
    assert_eq!(read(&world, &mut cursor).len(), 1, "draining subscriptions leaves journals intact");
}

#[test]
fn a_reader_that_falls_behind_is_told_to_rescan() {
    let mut world = World::new();
    let mut cursor = world.open_change_cursor::<Pos>();
    let e = world.spawn();
    world.insert(e, Pos(0.0, 0.0, 0.0));
    for i in 0..pulsar_scenedb::change_journal::DEFAULT_JOURNAL_CAPACITY {
        world.get_mut::<Pos>(e).unwrap().0 = i as f32;
    }
    let mut out = Vec::new();
    assert_eq!(world.read_changes(&mut cursor, &mut out), ChangeRead::Overflowed);
    assert!(out.is_empty(), "an overflowed read never delivers a gap");
    world.get_mut::<Pos>(e).unwrap().0 = -1.0;
    assert_eq!(read(&world, &mut cursor).len(), 1, "reading resumes after the rescan point");
}
