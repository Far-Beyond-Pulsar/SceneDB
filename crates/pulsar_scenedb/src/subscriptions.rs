//! Per-`(Entity, ComponentType)` component subscriptions -- the push side of
//! "did this exact component change?" (SceneDB#47).
//!
//! # The problem
//!
//! Consumers outside the render-sync loop (Pulsar-Native's properties panel;
//! any live editor UI reading `World` directly) have no way to learn whether
//! a component they already read actually changed. The only correct thing
//! they could do was re-pull every value on every redraw, unconditionally,
//! forever -- polling, at UI-frame rate, for data that usually hasn't moved.
//! A subscription lets such a consumer cache its snapshot and re-pull only
//! when a real change signal fires for the specific `(Entity, ComponentType)`
//! being displayed.
//!
//! # The mutation-detection problem is already solved -- reused, not reinvented
//!
//! `World::get_mut` returns the [`crate::world::Mut`] guard (not a raw
//! `&mut T`) precisely so a drop-hook can react to mutation; its existing
//! hooks are the GPU mirror dispatch (`gpu_hook`) and replication change
//! recording (`change_hook`). This module adds a third hook of the same
//! shape: a subscription registry handle captured at `get_mut` time, fired
//! on drop. `insert`/`remove`/`despawn` fire where their GPU/tracker hooks
//! already run. There is no new "how do we know a mutation happened"
//! mechanism here.
//!
//! # Design decisions (and why)
//!
//! - **Granularity: per-`(Entity, ComponentType)`.** Matches how consumers
//!   actually consume data (one properties-panel card per component), and
//!   avoids whole-`World`/whole-`Entity` broadcasts that would force every
//!   subscriber to re-check on every change anywhere. Multiple subscribers
//!   to the same key each get their own event.
//! - **Delivery: batched, pull-based, never a callback inside `Drop`.** A
//!   synchronous callback during `Mut`'s drop risks reentrancy (a listener
//!   mutating `World` again while the original guard's drop is still
//!   unwinding). Instead, mutations append [`ComponentChangeEvent`]s to a
//!   bounded pending queue; the consumer drains it via
//!   [`World::take_component_change_events`] once per tick/frame at a point
//!   of its own choosing. This is also why there is deliberately **no**
//!   closure-taking API: an event record can't call back into anything.
//! - **A `get_mut` that never wrote through `DerefMut` fires nothing.** The
//!   guard tracks whether `DerefMut` ran; a borrow-only `get_mut` is not a
//!   mutation. (`Mut::into_inner` is the documented escape hatch that fires
//!   immediately by construction -- same as its GPU/change hooks.)
//! - **Subscriber lifetime: explicit [`SubscriptionId`] + unsubscribe**, with
//!   auto-cleanup on despawn (the entity generation makes any surviving
//!   subscription permanently inert anyway, so dropping it then is pure
//!   hygiene, plus it delivers one final `Removed` event per component the
//!   entity had). A consumer that forgets to unsubscribe a live entity keeps
//!   receiving events for it -- same contract as keeping an entity handle.
//! - **Cost when nobody subscribes: one `Option::is_none()` check** per
//!   `insert`/`remove`/`despawn`/`get_mut`/drop, exactly like the attached-
//!   mirror/attached-tracker short-circuits before it. No registry scan, no
//!   allocation, nothing on the drop path.
//! - **Relationship to [`crate::replication::ChangeTracker`]: sits alongside,
//!   does not replace.** `ChangeTracker` captures batched new-state changes
//!   for a specific target (replication delta encoding); a subscription is a
//!   push notification to arbitrary live subscribers about a specific
//!   `(Entity, ComponentId)` key. Different consumers, different retention,
//!   different delivery -- sharing an event stream would couple both to the
//!   union of their needs. They do share the *mutation-detection* layer (the
//!   same hook points in `world.rs`), which is the part worth sharing.
//!
//! # Bounded memory
//!
//! A subscribed-but-never-draining `World` must not grow without bound, so
//! the pending queue is capped ([`MAX_PENDING_EVENTS`]); beyond the cap the
//! OLDEST events are dropped and counted (see
//! [`World::dropped_component_change_events`]). Draining per frame, as
//! documented, never hits the cap.

use crate::component::ComponentId;
use crate::entity::Entity;
use ahash::AHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Soft cap on the pending event queue (see the module doc on bounded
/// memory). Generous by design: a per-frame drain at even 60 Hz means this
/// cap represents over 18 minutes of 60-events-every-tick churn, far past
/// anything a real consumer produces between drains.
pub(crate) const MAX_PENDING_EVENTS: usize = 65_536;

/// Opaque handle identifying one active subscription. Returned by
/// [`World::subscribe`]/[`World::subscribe_id`]; passed to
/// [`World::unsubscribe`]. Cheap to copy and compare; carries no lifetime --
/// a subscription stays armed until explicitly unsubscribed or until its
/// entity despawns.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SubscriptionId(u64);

/// What kind of write produced a [`ComponentChangeEvent`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ComponentChangeKind {
    /// `T` was added to `entity` (first insert, or re-inserted after a
    /// prior remove).
    Inserted,
    /// `T`'s value on `entity` was written through `Mut`'s `DerefMut`
    /// (or handed out mutable via `into_inner`). One event per real
    /// write-through; a borrow-only `get_mut` produces nothing.
    Mutated,
    /// `T` was taken off `entity` (explicit remove, or the entity despawned
    /// while holding `T`).
    Removed,
}

/// One real change to one subscribed `(Entity, ComponentType)` key.
///
/// Delivery is at-least-once per real mutation and in mutation order, but
/// NOT coalesced: two writes between drains produce two events. Consumers
/// that only need "re-pull my cached snapshot" should treat the event set as
/// a dirty mask, not a log.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ComponentChangeEvent {
    /// The subscription this event was delivered to. When several
    /// subscriptions watch the same key, each gets its own event with its
    /// own id here.
    pub subscription: SubscriptionId,
    /// The entity whose component changed.
    pub entity: Entity,
    /// Which component type changed (erased form of `T`; use
    /// `pulsar_scenedb::component_id::<T>()` to compare against).
    pub component: ComponentId,
    /// What kind of write happened.
    pub kind: ComponentChangeKind,
}

/// The subscription bookkeeping itself. Lives behind one
/// `Arc<Mutex<..>>` shared between the `World` that owns it and the
/// `Mut` guards / inline mutation paths that report into it, so a hook can
/// be precomputed at `get_mut` time the same way `GpuMutHook`/
/// `ChangeMutHook` are. Never exposed publicly; all public surface lives on
/// [`World`](crate::World).
#[derive(Default)]
pub(crate) struct SubscriptionRegistry {
    next_id: u64,
    /// Subscription -> its key. Source of truth for unsubscribe.
    entries: AHashMap<SubscriptionId, (Entity, ComponentId)>,
    /// Key -> subscriptions watching it. The mutation-path lookup structure:
    /// one hash probe per real mutation, empty-slot miss costing only the
    /// probe when nothing watches the key.
    by_key: AHashMap<(Entity, ComponentId), Vec<SubscriptionId>>,
    /// Batched delivery queue -- see the module doc on why callbacks-in-Drop
    /// were rejected.
    pending: VecDeque<ComponentChangeEvent>,
    /// Events dropped to enforce [`MAX_PENDING_EVENTS`]. Surfaced so a
    /// consumer that fell behind can notice rather than silently miss.
    dropped_events: u64,
}

impl SubscriptionRegistry {
    /// Arm a new subscription for `(entity, cid)`. Caller has already
    /// checked the entity is alive.
    pub(crate) fn subscribe(&mut self, entity: Entity, cid: ComponentId) -> SubscriptionId {
        self.next_id += 1;
        let id = SubscriptionId(self.next_id);
        self.entries.insert(id, (entity, cid));
        self.by_key.entry((entity, cid)).or_default().push(id);
        id
    }

    /// Disarm `id`. Returns false if it was never armed or already
    /// unsubscribed (idempotent).
    pub(crate) fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        let Some(key) = self.entries.remove(&id) else {
            return false;
        };
        if let Some(watchers) = self.by_key.get_mut(&key) {
            watchers.retain(|&w| w != id);
            if watchers.is_empty() {
                self.by_key.remove(&key);
            }
        }
        true
    }

    /// Disarm every subscription on `entity` (any component type). Used by
    /// despawn cleanup. Returns how many were removed.
    pub(crate) fn unsubscribe_entity(&mut self, entity: Entity) -> usize {
        let dead: Vec<SubscriptionId> = self
            .entries
            .iter()
            .filter(|(_, (e, _))| *e == entity)
            .map(|(id, _)| *id)
            .collect();
        for id in &dead {
            self.unsubscribe(*id);
        }
        dead.len()
    }

    /// Record one change against every subscription watching
    /// `(entity, cid)`. No-op (one hash probe) when nothing watches the key.
    pub(crate) fn record(&mut self, entity: Entity, cid: ComponentId, kind: ComponentChangeKind) {
        // Remove the watcher list, deliver, put it back -- avoids cloning
        // the list on every mutation while staying borrow-checker clean.
        // Nothing else can observe the momentary absence: the caller holds
        // the registry's mutex for the whole call.
        let Some(ids) = self.by_key.remove(&(entity, cid)) else {
            return;
        };
        for &id in &ids {
            self.push(ComponentChangeEvent { subscription: id, entity, component: cid, kind });
        }
        self.by_key.insert((entity, cid), ids);
    }

    /// Append one event, enforcing [`MAX_PENDING_EVENTS`] by dropping the
    /// oldest (see the module doc on bounded memory).
    fn push(&mut self, event: ComponentChangeEvent) {
        if self.pending.len() >= MAX_PENDING_EVENTS {
            self.pending.pop_front();
            self.dropped_events += 1;
        }
        self.pending.push_back(event);
    }

    pub(crate) fn take(&mut self) -> Vec<ComponentChangeEvent> {
        self.pending.drain(..).collect()
    }

    /// Diagnostic read used by `World::pending_component_change_events`.
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Diagnostic read used by `World::dropped_component_change_events`.
    pub(crate) fn dropped_count(&self) -> u64 {
        self.dropped_events
    }
}

/// Shared handle to a World's registry. Cloned into `Mut` guards and used
/// inline on the insert/remove/despawn paths; `None` everywhere when no
/// subscription exists, which is what makes the zero-subscriber cost a
/// single `Option` check.
pub(crate) type SubscriptionRegistryHandle = Arc<Mutex<SubscriptionRegistry>>;

/// Lock the registry. The lock is only ever held for the duration of one
/// bookkeeping operation (record/unsubscribe/drain) and never across user
/// code or another lock acquisition, so a poisoned mutex is unreachable
/// absent a panic mid-bookkeeping -- unwrapping matches
/// [`crate::replication::SharedChangeTracker::lock`]'s precedent.
pub(crate) fn lock(registry: &SubscriptionRegistryHandle) -> std::sync::MutexGuard<'_, SubscriptionRegistry> {
    registry.lock().expect("subscription registry lock poisoned")
}

// ── Public API surface, re-exported through the crate root ──────────────────

impl SubscriptionId {
    /// Raw numeric identity (monotonic per `World`). Exposed for debugging
    /// and logging; treat values as opaque.
    pub fn raw(self) -> u64 {
        self.0
    }
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
