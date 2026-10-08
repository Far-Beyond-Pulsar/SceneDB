//! Per-component-type change journals with independent read cursors.
//!
//! # The problem
//!
//! A system that derives state from a component (a renderer's acceleration
//! structure, a spatial index, a physics proxy) needs to know which entities
//! changed since it last looked. Rescanning every row each frame makes that
//! system O(scene) even when nothing moved, and a queue drained by one owner
//! cannot be shared by two consumers.
//!
//! # The model
//!
//! A journal records every change to one component type in order: insert
//! (including an in-place overwrite), a `get_mut` written through
//! `DerefMut`, remove, the removal implied by despawn, and a replicated
//! write applied by `Delta::apply`. Any number of readers each hold their own
//! [`ChangeCursor`] and read forward from it with
//! [`crate::World::read_changes`]; reading never consumes anything another
//! reader will see.
//!
//! A journal exists only after the first [`crate::World::open_change_cursor`]
//! for its type. Until then that type's writes cost one atomic load. Opening
//! a cursor only sees changes made after it was opened, so a reader does one
//! full scan when it opens (or after an overflow) and applies journal
//! entries from then on.
//!
//! # Bounded memory
//!
//! Each journal is a ring of at most [`DEFAULT_JOURNAL_CAPACITY`] entries.
//! When a reader falls further behind than that, its next read returns
//! [`ChangeRead::Overflowed`] instead of a partial list and its cursor jumps
//! to the newest entry; the reader must rescan. Correctness never depends on
//! a reader keeping up, only its cost does.
//!
//! # Belonging to one world
//!
//! Each `World`'s journals carry a process-unique id, and a cursor records
//! the id it was opened against. Reading with a cursor from another world
//! (a scene that was replaced while the reader held its cursor), or for a
//! type whose journal does not exist yet, rebinds the cursor to this
//! world's newest entry and returns [`ChangeRead::Overflowed`] once: the
//! reader rescans and continues from there. It never reads another world's
//! positions as if they were its own.

use crate::component::ComponentId;
use crate::entity::Entity;
use ahash::AHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Source of journal-set ids; 0 is never handed out.

/// Ring capacity per component type. At 64k entries a reader may skip over
/// a thousand frames of a thousand changes each before it must rescan.
pub const DEFAULT_JOURNAL_CAPACITY: usize = 1 << 16;

/// What kind of write produced a [`ComponentChange`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ComponentChangeKind {
    /// `T` was added to the entity (first insert, re-insert after a remove,
    /// or an in-place overwrite by `insert`).
    Inserted,
    /// `T`'s value was written through `Mut`'s `DerefMut` (or handed out
    /// mutable via `into_inner`), or replaced by a replicated write. A
    /// borrow-only `get_mut` records nothing.
    Mutated,
    /// `T` was taken off the entity (explicit remove, or the entity
    /// despawned while holding `T`).
    Removed,
}

impl ComponentChangeKind {
    /// Stable lowercase name (`"inserted"` / `"mutated"` / `"removed"`),
    /// for logs and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inserted => "inserted",
            Self::Mutated => "mutated",
            Self::Removed => "removed",
        }
    }
}

/// One recorded change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentChange {
    pub entity: Entity,
    pub kind: ComponentChangeKind,
}

/// A reader's position in one component type's journal. Plain data: holding
/// one keeps nothing alive, and dropping one needs no cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangeCursor {
    /// The id of the journal set (one per `World`) this cursor reads.
    journals: u64,
    component: ComponentId,
    next: u64,
}

impl ChangeCursor {
    /// The component type this cursor reads.
    pub fn component(&self) -> ComponentId {
        self.component
    }
}

/// Outcome of [`crate::World::read_changes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "an overflowed read delivered nothing; the reader must rescan"]
pub enum ChangeRead {
    /// Every change since the cursor's previous position was appended.
    Complete,
    /// The journal evicted entries this cursor had not read yet, or the
    /// cursor belonged to another world. Nothing was appended; the cursor
    /// now points at this world's newest entry, so the reader must rebuild
    /// from a full scan and continue from there.
    Overflowed,
}

struct Journal {
    /// Sequence number of `entries[0]`.
    first: u64,
    entries: VecDeque<ComponentChange>,
    capacity: usize,
}

impl Journal {
    fn end(&self) -> u64 {
        self.first + self.entries.len() as u64
    }

    fn push(&mut self, change: ComponentChange) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.first += 1;
        }
        self.entries.push_back(change);
    }
}

pub(crate) struct ChangeJournals {
    /// Process-unique id of this journal set (one per `World`).
    id: u64,
    journals: AHashMap<ComponentId, Journal>,
}

impl Default for ChangeJournals {
    fn default() -> Self {
        Self {
            id: (crate::runtime::runtime().next_journals_id)(),
            journals: AHashMap::default(),
        }
    }
}

pub(crate) type ChangeJournalHandle = Arc<Mutex<ChangeJournals>>;

pub(crate) fn lock(handle: &ChangeJournalHandle) -> std::sync::MutexGuard<'_, ChangeJournals> {
    handle.lock().expect("World change journals: mutex poisoned")
}

impl ChangeJournals {
    pub(crate) fn open(&mut self, component: ComponentId) -> ChangeCursor {
        let end = self.journal(component).end();
        ChangeCursor { journals: self.id, component, next: end }
    }

    fn journal(&mut self, component: ComponentId) -> &mut Journal {
        self.journals.entry(component).or_insert_with(|| Journal {
            first: 0,
            entries: VecDeque::new(),
            capacity: DEFAULT_JOURNAL_CAPACITY,
        })
    }

    /// Records `kind` for `(entity, component)` if that type is journaled.
    #[inline]
    pub(crate) fn record(&mut self, entity: Entity, component: ComponentId, kind: ComponentChangeKind) {
        if let Some(journal) = self.journals.get_mut(&component) {
            journal.push(ComponentChange { entity, kind });
        }
    }

    pub(crate) fn read(&mut self, cursor: &mut ChangeCursor, out: &mut Vec<ComponentChange>) -> ChangeRead {
        let id = self.id;
        let journal = self.journal(cursor.component);
        if cursor.journals != id {
            // A cursor from another world: rebind it here and rescan once.
            cursor.journals = id;
            cursor.next = journal.end();
            return ChangeRead::Overflowed;
        }
        if cursor.next < journal.first || cursor.next > journal.end() {
            cursor.next = journal.end();
            return ChangeRead::Overflowed;
        }
        let skip = (cursor.next - journal.first) as usize;
        out.extend(journal.entries.iter().skip(skip).copied());
        cursor.next = journal.end();
        ChangeRead::Complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(i: u32) -> Entity {
        Entity::new(i, 0)
    }

    #[test]
    fn readers_advance_independently() {
        let cid = crate::component::component_id::<u32>();
        let mut journals = ChangeJournals::default();
        let mut a = journals.open(cid);
        journals.record(entity(1), cid, ComponentChangeKind::Inserted);
        let mut b = journals.open(cid);
        journals.record(entity(2), cid, ComponentChangeKind::Mutated);

        let mut out = Vec::new();
        assert_eq!(journals.read(&mut a, &mut out), ChangeRead::Complete);
        assert_eq!(out.iter().map(|c| c.entity).collect::<Vec<_>>(), [entity(1), entity(2)]);
        out.clear();
        assert_eq!(journals.read(&mut b, &mut out), ChangeRead::Complete);
        assert_eq!(out.iter().map(|c| c.entity).collect::<Vec<_>>(), [entity(2)]);
        out.clear();
        assert_eq!(journals.read(&mut a, &mut out), ChangeRead::Complete);
        assert!(out.is_empty());
    }

    #[test]
    fn untracked_types_record_nothing() {
        let tracked = crate::component::component_id::<u32>();
        let untracked = crate::component::component_id::<u64>();
        let mut journals = ChangeJournals::default();
        let mut cursor = journals.open(tracked);
        journals.record(entity(1), untracked, ComponentChangeKind::Inserted);
        let mut out = Vec::new();
        assert_eq!(journals.read(&mut cursor, &mut out), ChangeRead::Complete);
        assert!(out.is_empty());
    }

    #[test]
    fn a_cursor_from_another_world_rescans_once_then_follows_this_one() {
        let cid = crate::component::component_id::<u32>();
        let mut old = ChangeJournals::default();
        let mut cursor = old.open(cid);
        old.record(entity(1), cid, ComponentChangeKind::Mutated);

        // The scene is replaced: a new journal set at a similar position.
        let mut new = ChangeJournals::default();
        let _other_reader = new.open(cid);
        new.record(entity(7), cid, ComponentChangeKind::Inserted);

        let mut out = Vec::new();
        assert_eq!(new.read(&mut cursor, &mut out), ChangeRead::Overflowed);
        assert!(out.is_empty(), "never reads the new world's entries as the old one's");
        new.record(entity(8), cid, ComponentChangeKind::Mutated);
        assert_eq!(new.read(&mut cursor, &mut out), ChangeRead::Complete);
        assert_eq!(out.iter().map(|c| c.entity).collect::<Vec<_>>(), [entity(8)]);
    }

    #[test]
    fn a_cursor_for_a_type_without_a_journal_starts_one() {
        let cid = crate::component::component_id::<u32>();
        let mut journals = ChangeJournals::default();
        let mut cursor = journals.open(crate::component::component_id::<u64>());
        cursor.component = cid;
        let mut out = Vec::new();
        assert_eq!(journals.read(&mut cursor, &mut out), ChangeRead::Complete);
        journals.record(entity(3), cid, ComponentChangeKind::Inserted);
        assert_eq!(journals.read(&mut cursor, &mut out), ChangeRead::Complete);
        assert_eq!(out.iter().map(|c| c.entity).collect::<Vec<_>>(), [entity(3)]);
    }

    #[test]
    fn a_lagging_reader_overflows_instead_of_reading_a_gap() {
        let cid = crate::component::component_id::<u32>();
        let mut journals = ChangeJournals::default();
        let mut cursor = journals.open(cid);
        journals.journals.get_mut(&cid).unwrap().capacity = 4;
        for i in 0..6 {
            journals.record(entity(i), cid, ComponentChangeKind::Mutated);
        }
        let mut out = Vec::new();
        assert_eq!(journals.read(&mut cursor, &mut out), ChangeRead::Overflowed);
        assert!(out.is_empty());
        journals.record(entity(9), cid, ComponentChangeKind::Mutated);
        assert_eq!(journals.read(&mut cursor, &mut out), ChangeRead::Complete);
        assert_eq!(out.iter().map(|c| c.entity).collect::<Vec<_>>(), [entity(9)]);
    }
}
