//! Object subscriptions: push every change to one object, with its new
//! value, to the subscribers watching it.
//!
//! # What this is for
//!
//! Data normally flows one way: an edit writes straight into the `World`,
//! the `World` replicates to the GPU, and every system that needs data reads
//! the `World` (or the GPU mirror). Nothing in that direction subscribes.
//!
//! Subscriptions are for the opposite direction: a view that displays an
//! object (the editor's properties panel showing the selection) must follow
//! writes made somewhere else (a gizmo drag, a script, a replicated update)
//! without polling the database. Systems that read large amounts of data,
//! such as renderers, must not use them; they read the `World` and use
//! [`crate::change_journal`] cursors for incremental work.
//!
//! # Scope: an object
//!
//! A subscription watches one **object**: every write to a component of an
//! entity whose object is the subscribed entity. Which object an entity
//! belongs to is decided by the world's object resolver
//! ([`crate::World::set_object_resolver`]); the default maps every entity to
//! itself. A host that stores an object's components on their own entities
//! installs a resolver that maps those entities to their owner, so a
//! subscription covers the object's components, including ones attached
//! after it subscribed.
//!
//! # Delivery
//!
//! The callback runs **inside the write**, after the value is stored, with
//! the component's full new value by reference ([`ObjectChange::value`]).
//! GPU-heavy payloads are not copied: a `GpuHeavy<T>` field holds only its
//! lightweight reference in the component. The callback receives no `World`
//! and must not try to reach one (the write holds it); it copies what it
//! needs and hands it to its own thread (a channel, a notify).
//!
//! - **Inserted**: an insert, re-insert or in-place overwrite. `value` is the
//!   new value.
//! - **Mutated**: a write through `get_mut` / `get_dyn_mut` (a borrow-only
//!   guard is not a write), `into_inner`, or a replicated write applied by
//!   `Delta::apply`. `value` is the new value.
//! - **Removed**: a removal, or the despawn of a component's entity. `value`
//!   is the removed value for `remove`/`remove_dyn`, `None` for a despawn.
//! - [`ObjectEvent::Despawned`] when the object itself despawns; its
//!   subscriptions end with it.
//!
//! Cost: with no subscriptions, one `Option` check per write. With
//! subscriptions, a write resolves its object and probes one map; only a
//! watched object's writes call back.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::change_journal::ComponentChangeKind;
use crate::component::ComponentId;
use crate::entity::Entity;

/// Identifies one subscription; pass it to
/// [`crate::World::unsubscribe_object`]. Ids are unique across every
/// `World` in the process, so an id kept past a world replacement never
/// matches (and never ends) a subscription on the new world.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SubscriptionId(u64);

/// One change to a component of a watched object.
#[derive(Clone, Copy)]
pub struct ObjectChange<'a> {
    /// The entity holding the component: the object itself, or one of its
    /// component entities.
    pub entity: Entity,
    pub component: ComponentId,
    pub kind: ComponentChangeKind,
    /// The full value after the write (`Inserted`, `Mutated`), or the
    /// removed value when the removal returned one. Downcast to the
    /// component's type.
    pub value: Option<&'a dyn Any>,
}

/// What a subscription is told.
#[derive(Clone, Copy)]
pub enum ObjectEvent<'a> {
    Changed(ObjectChange<'a>),
    /// The object despawned; the subscription has ended.
    Despawned,
}

/// A subscriber: called with the watched object and the event.
pub type ObjectCallback = Arc<dyn Fn(Entity, &ObjectEvent<'_>) + Send + Sync>;

/// Maps an entity to the object its components belong to.
pub type ObjectResolver = fn(&crate::World, Entity) -> Entity;

/// The identity resolver: every entity is its own object.
pub fn entity_is_its_own_object(_: &crate::World, entity: Entity) -> Entity {
    entity
}

static NEXT_SUBSCRIPTION: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
pub(crate) struct ObjectSubscriptions {
    by_object: HashMap<Entity, Vec<(SubscriptionId, ObjectCallback)>>,
    objects: HashMap<SubscriptionId, Entity>,
}

impl ObjectSubscriptions {
    pub(crate) fn subscribe(&mut self, object: Entity, callback: ObjectCallback) -> SubscriptionId {
        let id = SubscriptionId(NEXT_SUBSCRIPTION.fetch_add(1, Ordering::Relaxed));
        self.by_object.entry(object).or_default().push((id, callback));
        self.objects.insert(id, object);
        id
    }

    pub(crate) fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        let Some(object) = self.objects.remove(&id) else {
            return false;
        };
        if let Some(subscribers) = self.by_object.get_mut(&object) {
            subscribers.retain(|(sub, _)| *sub != id);
            if subscribers.is_empty() {
                self.by_object.remove(&object);
            }
        }
        true
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_object.is_empty()
    }

    /// The callbacks watching `object`, cloned so they run without the lock
    /// held.
    pub(crate) fn callbacks(&self, object: Entity) -> Vec<ObjectCallback> {
        self.by_object
            .get(&object)
            .map(|subscribers| subscribers.iter().map(|(_, cb)| Arc::clone(cb)).collect())
            .unwrap_or_default()
    }

    /// End every subscription on `object`, returning their callbacks.
    pub(crate) fn end(&mut self, object: Entity) -> Vec<ObjectCallback> {
        let Some(subscribers) = self.by_object.remove(&object) else {
            return Vec::new();
        };
        subscribers
            .into_iter()
            .map(|(id, cb)| {
                self.objects.remove(&id);
                cb
            })
            .collect()
    }
}

pub(crate) type ObjectSubscriptionsHandle = Arc<Mutex<ObjectSubscriptions>>;

pub(crate) fn lock(handle: &ObjectSubscriptionsHandle) -> std::sync::MutexGuard<'_, ObjectSubscriptions> {
    handle.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Call each of `callbacks` with one change.
pub(crate) fn deliver(callbacks: &[ObjectCallback], object: Entity, change: ObjectChange<'_>) {
    let event = ObjectEvent::Changed(change);
    for callback in callbacks {
        callback(object, &event);
    }
}
