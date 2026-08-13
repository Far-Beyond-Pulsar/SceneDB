//! Correctness proof for the zero-allocation archetype-migration fast path
//! (`World::move_column_row`, `ErasedColumn::swap_remove_into`/`push_from`):
//! every component that migrates via `World::insert`/`remove` must move
//! exactly once, drop exactly once (whenever it's finally dropped), and
//! carry its bytes (including any heap-owned data, e.g. a `String`) intact
//! -- for components that fit the inline scratch fast path AND for ones
//! that don't (oversized/overaligned, forced onto the `Box`-based
//! fallback).
//!
//! This is deliberately a SEPARATE file from the general migration tests:
//! it exists specifically to catch double-drop, leak, or byte-corruption
//! regressions in the raw-pointer move path, which no other test in this
//! crate exercises directly.

use pulsar_scenedb::World;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Increments a shared counter on `Drop`. Migrating this component must
/// never touch the counter -- only an actual, final `Drop` (via `despawn`
/// or `remove`'s returned-and-dropped value) may.
struct DropCounted {
    counter: Arc<AtomicUsize>,
    tag: u32,
}

impl Drop for DropCounted {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
struct Pos(f32, f32, f32);
#[derive(Clone, Copy)]
struct Vel(f32, f32, f32);
#[derive(Clone, Copy)]
struct Tag(u32);

/// A heap-owning component (String's buffer lives on the heap) -- proves
/// the raw byte move doesn't corrupt or duplicate the pointer/len/cap
/// triple, which would show up as a `use-after-free`/double-free or
/// garbled content under Miri/ASan, or silently wrong content otherwise.
struct Name(String);

/// Exactly at the inline scratch boundary: 128 bytes, alignment 8 (still
/// `<= MoveScratch::ALIGN`). Exercises the fast path's edge, not the
/// fallback.
#[derive(Clone, Copy)]
#[repr(C)]
struct Boundary128 {
    data: [u64; 16], // 16 * 8 = 128 bytes, align 8
}

/// Larger than the 128-byte inline scratch capacity -- forces
/// `move_column_row`'s `Box`-based fallback path.
#[derive(Clone, Copy)]
#[repr(C)]
struct Oversized {
    data: [u64; 32], // 256 bytes
}

/// Alignment (32) exceeds `MoveScratch::ALIGN` (16) even though the size
/// (32 bytes) would otherwise fit -- forces the fallback path via the
/// alignment check specifically, not the size check.
#[derive(Clone, Copy)]
#[repr(align(32))]
struct Overaligned {
    data: [u8; 32],
}

#[test]
fn drop_counted_component_never_double_drops_or_leaks_across_repeated_migration() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let e = world.spawn();

    // Every insert here migrates the entity to a new archetype, carrying
    // the DropCounted value through move_column_row on every single one.
    world.insert(e, DropCounted { counter: Arc::clone(&counter), tag: 1 });
    world.insert(e, Pos(1.0, 2.0, 3.0)); // migrates DropCounted along
    world.insert(e, Vel(0.0, 0.0, 0.0)); // migrates DropCounted + Pos along
    world.insert(e, Tag(7)); // migrates DropCounted + Pos + Vel along
    assert_eq!(counter.load(Ordering::SeqCst), 0, "no migration may drop the value early");

    // Overwrite in place (same archetype, no migration) -- the OLD value
    // must drop exactly once here.
    world.insert(e, DropCounted { counter: Arc::clone(&counter), tag: 2 });
    assert_eq!(counter.load(Ordering::SeqCst), 1, "in-place overwrite drops exactly the old value");

    // More migrations carrying the (new) DropCounted value along.
    world.remove::<Tag>(e); // migrates DropCounted + Pos + Vel
    world.remove::<Vel>(e); // migrates DropCounted + Pos
    assert_eq!(counter.load(Ordering::SeqCst), 1, "still just the one prior overwrite drop");
    assert!(world.get::<Tag>(e).is_none(), "Tag was removed");
    assert!(world.get::<Vel>(e).is_none(), "Vel was removed");

    // Final drop via despawn.
    world.despawn(e);
    assert_eq!(counter.load(Ordering::SeqCst), 2, "despawn drops the final live value exactly once");
}

#[test]
fn remove_returns_the_correct_live_value_not_a_stale_or_moved_one() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Pos(1.0, 2.0, 3.0));
    world.insert(e, DropCounted { counter: Arc::clone(&counter), tag: 42 });
    world.insert(e, Vel(9.0, 9.0, 9.0)); // migrates DropCounted along with Pos

    let removed = world.remove::<DropCounted>(e).expect("component present");
    assert_eq!(removed.tag, 42, "remove_inner's direct typed swap_remove must return the SAME value that was migrated, not a stale/corrupted one");
    assert_eq!(counter.load(Ordering::SeqCst), 0, "the returned value hasn't dropped yet -- it's still owned by `removed`");
    drop(removed);
    assert_eq!(counter.load(Ordering::SeqCst), 1, "dropping the returned value now drops exactly once");

    // Pos and Vel must still be intact after DropCounted's removal migrated them again.
    assert_eq!(world.get::<Pos>(e).map(|p| (p.0, p.1, p.2)), Some((1.0, 2.0, 3.0)));
    assert_eq!(world.get::<Vel>(e).map(|v| (v.0, v.1, v.2)), Some((9.0, 9.0, 9.0)));
}

#[test]
fn heap_owning_component_content_survives_repeated_migration_byte_exact() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Name("a genuinely non-trivial heap-allocated string value".to_string()));
    world.insert(e, Pos(1.0, 2.0, 3.0)); // migrates Name along
    world.insert(e, Vel(4.0, 5.0, 6.0)); // migrates Name + Pos along
    world.insert(e, Tag(1)); // migrates Name + Pos + Vel along

    assert_eq!(
        world.get::<Name>(e).map(|n| n.0.as_str()),
        Some("a genuinely non-trivial heap-allocated string value"),
        "String content must be byte-exact after four migrations -- any pointer/len/cap corruption in the raw move would show up here"
    );

    world.remove::<Tag>(e); // migrates Name + Pos + Vel again
    assert_eq!(world.get::<Name>(e).map(|n| n.0.as_str()), Some("a genuinely non-trivial heap-allocated string value"));

    // Drop it via despawn -- if the move corrupted the String's internal
    // pointer, this either segfaults or double-frees under a sanitizer.
    world.despawn(e);
}

#[test]
fn boundary_sized_component_uses_the_fast_path_and_round_trips_exactly() {
    let mut world = World::new();
    let e = world.spawn();
    let payload = Boundary128 { data: std::array::from_fn(|i| i as u64 * 7 + 1) };
    world.insert(e, payload);
    world.insert(e, Pos(1.0, 1.0, 1.0)); // migrates Boundary128 along
    world.insert(e, Vel(2.0, 2.0, 2.0)); // migrates again

    let got = world.get::<Boundary128>(e).expect("present");
    assert_eq!(got.data, payload.data, "128-byte, align-8 component (exactly at the inline cap) must round-trip exactly");
}

#[test]
fn oversized_component_forces_the_fallback_path_and_still_round_trips_exactly() {
    let mut world = World::new();
    let e = world.spawn();
    let payload = Oversized { data: std::array::from_fn(|i| (i as u64) * 31 + 3) };
    world.insert(e, payload);
    world.insert(e, Pos(1.0, 1.0, 1.0)); // migrates the 256-byte Oversized along (fallback path)
    world.insert(e, Vel(2.0, 2.0, 2.0)); // migrates again

    let got = world.get::<Oversized>(e).expect("present");
    assert_eq!(got.data, payload.data, "oversized component (forced onto the Box fallback) must still round-trip exactly");

    world.remove::<Vel>(e); // migrates Oversized + Pos again
    let got2 = world.get::<Oversized>(e).expect("still present");
    assert_eq!(got2.data, payload.data);
}

#[test]
fn overaligned_component_forces_the_fallback_path_via_alignment_and_round_trips_exactly() {
    let mut world = World::new();
    let e = world.spawn();
    let payload = Overaligned { data: std::array::from_fn(|i| i as u8) };
    world.insert(e, payload);
    world.insert(e, Tag(1)); // migrates Overaligned along (align 32 > MoveScratch::ALIGN forces fallback)
    world.insert(e, Pos(1.0, 2.0, 3.0));

    let got = world.get::<Overaligned>(e).expect("present");
    assert_eq!(got.data, payload.data);
    // Verify actual alignment held -- a misaligned Overaligned value would
    // be UB to even read via a reference the way `get::<T>` does.
    let addr = got as *const Overaligned as usize;
    assert_eq!(addr % std::mem::align_of::<Overaligned>(), 0, "returned reference must be properly aligned");
}

#[test]
fn mixed_fast_path_and_fallback_components_migrate_together_correctly() {
    // One entity carrying BOTH an inline-fast-path component (Pos) AND a
    // fallback-path component (Oversized) at once -- proves
    // move_column_row's per-column dispatch (fast vs fallback) doesn't
    // interfere across columns within the same migration call.
    let counter = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Pos(1.0, 2.0, 3.0));
    world.insert(e, Oversized { data: [99u64; 32] });
    world.insert(e, DropCounted { counter: Arc::clone(&counter), tag: 5 });
    world.insert(e, Overaligned { data: [7u8; 32] });
    world.insert(e, Vel(9.0, 9.0, 9.0));

    assert_eq!(world.get::<Pos>(e).map(|p| (p.0, p.1, p.2)), Some((1.0, 2.0, 3.0)));
    assert_eq!(world.get::<Oversized>(e).map(|o| o.data), Some([99u64; 32]));
    assert_eq!(world.get::<Overaligned>(e).map(|o| o.data), Some([7u8; 32]));
    assert_eq!(world.get::<Vel>(e).map(|v| (v.0, v.1, v.2)), Some((9.0, 9.0, 9.0)));
    assert_eq!(counter.load(Ordering::SeqCst), 0);

    world.despawn(e);
    assert_eq!(counter.load(Ordering::SeqCst), 1, "despawn drops DropCounted exactly once even amid mixed fast/fallback columns");
}

#[test]
fn many_entities_churning_through_migrations_never_double_drop() {
    // Broader stress: many entities, many migrations each, verifying total
    // drop count exactly matches total live-component count at the end.
    let counter = Arc::new(AtomicUsize::new(0));
    let mut world = World::new();
    let mut entities = Vec::new();
    for i in 0..200u32 {
        let e = world.spawn();
        world.insert(e, DropCounted { counter: Arc::clone(&counter), tag: i });
        world.insert(e, Pos(i as f32, 0.0, 0.0));
        world.insert(e, Vel(0.0, i as f32, 0.0));
        world.insert(e, Tag(i));
        entities.push(e);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 0);

    // Despawn half of them.
    for &e in &entities[..100] {
        world.despawn(e);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 100, "exactly the despawned half must have dropped");

    // Remaining half: remove Tag (migrates), then despawn.
    for &e in &entities[100..] {
        world.remove::<Tag>(e);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 100, "removing Tag must not touch DropCounted's drop count");
    for &e in &entities[100..] {
        world.despawn(e);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 200, "every DropCounted value drops exactly once, total");
}
