//! A change cursor belongs to the `World` it was opened on: reading it
//! against a replacement world rescans once instead of reading the new
//! world's journal positions as if they were the old one's.

use pulsar_scenedb::{ChangeRead, World};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Health(u32);

#[test]
fn a_cursor_survives_its_world_being_replaced() {
    let mut old = World::new();
    let e = old.spawn();
    old.insert(e, Health(1));
    let mut cursor = old.open_change_cursor::<Health>();
    old.get_mut::<Health>(e).unwrap().0 = 2;
    let mut out = Vec::new();
    assert_eq!(old.read_changes(&mut cursor, &mut out), ChangeRead::Complete);
    assert_eq!(out.len(), 1);

    // The scene is replaced (level load, undo snapshot swap).
    let mut new = World::new();
    let f = new.spawn();
    new.insert(f, Health(10));
    let _panel = new.open_change_cursor::<Health>();
    new.get_mut::<Health>(f).unwrap().0 = 11;

    out.clear();
    assert_eq!(new.read_changes(&mut cursor, &mut out), ChangeRead::Overflowed);
    assert!(out.is_empty());
    new.get_mut::<Health>(f).unwrap().0 = 12;
    assert_eq!(new.read_changes(&mut cursor, &mut out), ChangeRead::Complete);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].entity, f);
}
