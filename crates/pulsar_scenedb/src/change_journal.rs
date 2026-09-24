//! Per-component-type change journals with independent read cursors.
//!
//! # The problem
//!
//! A system that derives state from a component (a renderer's acceleration
//! structure, a spatial index, a physics proxy) needs to know which entities
//! changed since it last looked. Rescanning every row each frame makes that
//! system O(scene) even when nothing moved. [`crate::subscriptions`] answers
//! "did this exact `(Entity, T)` change?" but its queue is drained by one
//! owner ([`crate::World::take_component_change_events`]), so two consumers
//! cannot share it, and it needs a subscription per entity.
//!
//! # The model
//!
//! A journal records every change to one component type in order: insert
//! (including an in-place overwrite), a `get_mut` written through
//! `DerefMut`, remove, and the removal implied by despawn -- the same sites
//! subscriptions fire at. Any number of readers each hold their own
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

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::subscriptions::ComponentChangeKind;
use ahash::AHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Ring capacity per component type. At 64k entries a reader may skip over
/// a thousand frames of a thousand changes each before it must rescan.
pub const DEFAULT_JOURNAL_CAPACITY: usize = 1 << 16;

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
    /// The journal evicted entries this cursor had not read yet. Nothing was
    /// appended; the cursor now points at the newest entry, so the reader
    /// must rebuild from a full scan and continue from there.
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

#[derive(Default)]
pub(crate) struct ChangeJournals {
    journals: AHashMap<ComponentId, Journal>,
}

pub(crate) type ChangeJournalHandle = Arc<Mutex<ChangeJournals>>;

pub(crate) fn lock(handle: &ChangeJournalHandle) -> std::sync::MutexGuard<'_, ChangeJournals> {
    handle.lock().expect("World change journals: mutex poisoned")
}

impl ChangeJournals {
    pub(crate) fn open(&mut self, component: ComponentId) -> ChangeCursor {
        let journal = self.journals.entry(component).or_insert_with(|| Journal {
            first: 0,
            entries: VecDeque::new(),
            capacity: DEFAULT_JOURNAL_CAPACITY,
        });
        ChangeCursor { component, next: journal.end() }
    }

    /// Records `kind` for `(entity, component)` if that type is journaled.
    #[inline]
    pub(crate) fn record(&mut self, entity: Entity, component: ComponentId, kind: ComponentChangeKind) {
        if let Some(journal) = self.journals.get_mut(&component) {
            journal.push(ComponentChange { entity, kind });
        }
    }

    pub(crate) fn read(&self, cursor: &mut ChangeCursor, out: &mut Vec<ComponentChange>) -> ChangeRead {
        let Some(journal) = self.journals.get(&cursor.component) else {
            // Only reachable with a cursor from a different World.
            return ChangeRead::Overflowed;
        };
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
