//! Typed and erased references to a component on an entity.
//!
//! These are the reference types scripts and other systems hold instead of
//! pointers: plain `Copy` values that name an entity and a component type,
//! and are checked against the live [`World`] on every access. A reference
//! to a despawned entity (or to a component the entity no longer has) simply
//! resolves to `None`; it can never reach another entity that later reuses
//! the slot, because [`Entity`] carries a generation.
//!
//! Both forms are runtime-only: [`ComponentId`] is assigned per process, so
//! anything persisted (level files, script sources) must store a stable
//! identity and resolve to these at load time.

use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::component::{component_id, Component, ComponentId};
use crate::entity::Entity;
use crate::mut_guard::{Mut, MutDyn};
use crate::world::World;

/// Reference to one component of any type on one entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentRef {
    pub entity: Entity,
    pub component: ComponentId,
}

impl ComponentRef {
    pub fn new(entity: Entity, component: ComponentId) -> Self {
        Self { entity, component }
    }

    /// Reference to `entity`'s `T`.
    pub fn of<T: Component>(entity: Entity) -> Self {
        Self::new(entity, component_id::<T>())
    }

    /// Whether the entity is alive and still has the component.
    pub fn is_valid(&self, world: &World) -> bool {
        world.has_component(self.entity, self.component)
    }

    /// The component as `&dyn Any` of its own type.
    pub fn get<'w>(&self, world: &'w World) -> Option<&'w dyn Any> {
        world.get_dyn(self.entity, self.component)
    }

    /// Mutable access; the guard reports the write like `World::get_mut`.
    pub fn get_mut<'w>(&self, world: &'w mut World) -> Option<MutDyn<'w>> {
        world.get_dyn_mut(self.entity, self.component)
    }

    /// Typed view of this reference, if it names a `T`.
    pub fn typed<T: Component>(&self) -> Option<ComponentHandle<T>> {
        (self.component == component_id::<T>()).then(|| ComponentHandle::new(self.entity))
    }
}

/// Reference to `entity`'s component of type `T`.
pub struct ComponentHandle<T> {
    entity: Entity,
    _type: PhantomData<fn() -> T>,
}

impl<T: Component> ComponentHandle<T> {
    pub fn new(entity: Entity) -> Self {
        Self { entity, _type: PhantomData }
    }

    pub fn entity(&self) -> Entity {
        self.entity
    }

    /// Whether the entity is alive and still has a `T`.
    pub fn is_valid(&self, world: &World) -> bool {
        world.get::<T>(self.entity).is_some()
    }

    pub fn get<'w>(&self, world: &'w World) -> Option<&'w T> {
        world.get::<T>(self.entity)
    }

    pub fn get_mut<'w>(&self, world: &'w mut World) -> Option<Mut<'w, T>> {
        world.get_mut::<T>(self.entity)
    }

    /// The erased form of this reference.
    pub fn erase(&self) -> ComponentRef {
        ComponentRef::of::<T>(self.entity)
    }
}

// Manual impls: derives would require `T: Clone` etc., which a reference
// to a `T` does not need.
impl<T> Clone for ComponentHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ComponentHandle<T> {}

impl<T> PartialEq for ComponentHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl<T> Eq for ComponentHandle<T> {}

impl<T> Hash for ComponentHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entity.hash(state);
    }
}

impl<T> fmt::Debug for ComponentHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComponentHandle<{}>({:?})", std::any::type_name::<T>(), self.entity)
    }
}
