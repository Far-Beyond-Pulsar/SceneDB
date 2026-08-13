//! Multi-component insert/spawn in one archetype transition.
//!
//! [`World::spawn`] + N calls to [`World::insert`] is correct but pays for N
//! archetype migrations on a brand-new entity that has never held any
//! component data yet -- each `insert` walks the empty-entity through a new
//! archetype, `migrate_row`-ing every column it already has (zero after the
//! first insert, one after the second, ...), even though there is nothing
//! genuinely "existing" to preserve until the LAST of the N calls. A
//! [`Bundle`] collapses that into: resolve the ONE destination archetype by
//! walking N archetype-graph edges (each edge is a cached `Vec` index read
//! after the first time any entity takes that exact transition -- see
//! [`crate::archetype::Archetype::add_edges`]'s doc), move the entity there
//! once, then push each component's value directly into its column. No
//! column is ever swap-removed and re-pushed, because a freshly spawned
//! entity has no prior column data to move.
//!
//! Implemented for tuples `(A,)` through `(A, B, C, D, E, F, G, H)`.
//!
//! ```
//! use pulsar_scenedb::World;
//! # struct Pos(f32, f32, f32);
//! # struct Vel(f32, f32, f32);
//! # struct Health(u32);
//! let mut world = World::new();
//! let e = world.spawn_bundle((Pos(0.0, 0.0, 0.0), Vel(0.0, 0.0, 0.0), Health(100)));
//! assert!(world.get::<Health>(e).is_some());
//! ```
use crate::archetype::ArchetypeId;
use crate::component::Component;
use crate::entity::Entity;
use crate::replication::ChangeTracker;
use crate::world::World;

/// A fixed set of component types that can be spawned or inserted onto an
/// entity in a single archetype transition. See the [module doc](self) for
/// why this exists. Implemented for tuples of 1 to 8 [`Component`]s --
/// not meant to be implemented directly.
pub trait Bundle: Sized {
    /// Walk this bundle's "add" edge for each component type, in the same
    /// fixed order every time, starting from `start`. Creates any archetype
    /// that doesn't exist yet. Never called with `start` already containing
    /// any of this bundle's component types (see [`World::spawn_bundle`]'s
    /// doc -- a freshly spawned entity starts in the empty archetype).
    #[doc(hidden)]
    fn dest_archetype(world: &mut World, start: ArchetypeId) -> ArchetypeId;

    /// Push every value in this bundle as a brand-new column entry at the
    /// row `entity` already occupies in `arch_id` (the caller pushes
    /// `entity` onto `arch_id`'s entity list before calling this), in the
    /// same fixed order [`Self::dest_archetype`] used to resolve `arch_id`.
    #[doc(hidden)]
    fn push_new(self, world: &mut World, arch_id: ArchetypeId, entity: Entity, tracker: &mut Option<&mut ChangeTracker>);

    /// Fallback used by [`World::insert_bundle`]: `entity` may already carry
    /// some of this bundle's component types (unlike [`World::spawn_bundle`],
    /// which only ever hands a bundle to a just-spawned, componentless
    /// entity), so each component goes through the ordinary, edge-cached
    /// single-component [`World::insert`] path -- which already handles
    /// "overwrite in place if present, migrate if not" correctly -- rather
    /// than [`Self::push_new`]'s "every component is definitely new" fast
    /// path.
    #[doc(hidden)]
    fn insert_each(self, world: &mut World, entity: Entity, tracker: &mut Option<&mut ChangeTracker>);

    /// Reserve capacity for `additional` more entities in each of this
    /// bundle's columns within `arch_id`. Used by [`World::reserve_bundle`].
    #[doc(hidden)]
    fn reserve_columns(world: &mut World, arch_id: ArchetypeId, additional: u32);
}

macro_rules! impl_bundle_tuple {
    ($($Q:ident),+) => {
        impl<$($Q: Component),+> Bundle for ($($Q,)+) {
            #[inline]
            fn dest_archetype(world: &mut World, start: ArchetypeId) -> ArchetypeId {
                let id = start;
                $(let id = World::step_add_edge::<$Q>(world, id);)+
                id
            }

            #[inline]
            fn push_new(self, world: &mut World, arch_id: ArchetypeId, entity: Entity, tracker: &mut Option<&mut ChangeTracker>) {
                #[allow(non_snake_case)]
                let ($($Q,)+) = self;
                $(World::push_new_component::<$Q>(world, arch_id, entity, $Q, tracker);)+
            }

            #[inline]
            fn insert_each(self, world: &mut World, entity: Entity, tracker: &mut Option<&mut ChangeTracker>) {
                #[allow(non_snake_case)]
                let ($($Q,)+) = self;
                $(
                    match tracker.as_deref_mut() {
                        Some(t) => world.insert_tracked(entity, $Q, t),
                        None => world.insert(entity, $Q),
                    }
                )+
            }

            #[inline]
            fn reserve_columns(world: &mut World, arch_id: ArchetypeId, additional: u32) {
                $(world.reserve_component_column::<$Q>(arch_id, additional);)+
            }
        }
    };
}

impl_bundle_tuple!(A);
impl_bundle_tuple!(A, B);
impl_bundle_tuple!(A, B, C);
impl_bundle_tuple!(A, B, C, D);
impl_bundle_tuple!(A, B, C, D, E);
impl_bundle_tuple!(A, B, C, D, E, F);
impl_bundle_tuple!(A, B, C, D, E, F, G);
impl_bundle_tuple!(A, B, C, D, E, F, G, H);

impl World {
    /// Pre-allocate storage for `count` future `spawn_bundle::<B>` calls.
    ///
    /// Resolves (creating if needed) the archetype `B` resolves to from the
    /// empty archetype, then reserves capacity for `count` more entities in
    /// its entity list and in every one of `B`'s columns -- so a tight loop
    /// of `count` `spawn_bundle::<B>(..)` calls that follows doesn't pay for
    /// repeated `Vec` capacity-doubling on any of them. Call alongside
    /// [`World::reserve_entities`] (which only covers the EMPTY archetype's
    /// own entity list, not `B`'s destination archetype) before a batch
    /// spawn of entities that all share the same bundle shape.
    ///
    /// # Example
    /// ```
    /// use pulsar_scenedb::World;
    /// # struct Pos(f32, f32, f32);
    /// # struct Vel(f32, f32, f32);
    /// let mut world = World::new();
    /// world.reserve_entities(10_000);
    /// world.reserve_bundle::<(Pos, Vel)>(10_000);
    /// for _ in 0..10_000 {
    ///     world.spawn_bundle((Pos(0.0, 0.0, 0.0), Vel(0.0, 0.0, 0.0)));
    /// }
    /// ```
    pub fn reserve_bundle<B: Bundle>(&mut self, count: u32) {
        let dest = B::dest_archetype(self, ArchetypeId::EMPTY);
        self.archetypes[dest.0 as usize].entities.reserve(count as usize);
        B::reserve_columns(self, dest, count);
    }

    /// Spawn a new entity with every component in `bundle` already attached,
    /// in a single archetype transition -- see the [module doc](self) for
    /// why this beats `spawn()` + N `insert()` calls.
    ///
    /// # Example
    /// ```
    /// use pulsar_scenedb::World;
    /// # struct Pos(f32, f32, f32);
    /// # struct Vel(f32, f32, f32);
    /// let mut world = World::new();
    /// let e = world.spawn_bundle((Pos(1.0, 2.0, 3.0), Vel(0.0, 0.0, 0.0)));
    /// assert!(world.get::<Pos>(e).is_some());
    /// assert!(world.get::<Vel>(e).is_some());
    /// ```
    pub fn spawn_bundle<B: Bundle>(&mut self, bundle: B) -> Entity {
        self.spawn_bundle_inner(bundle, None)
    }

    /// Like [`spawn_bundle`](Self::spawn_bundle) but also records the spawn
    /// and every component in `bundle` in a [`ChangeTracker`] for
    /// replication.
    pub fn spawn_bundle_tracked<B: Bundle>(&mut self, bundle: B, tracker: &mut ChangeTracker) -> Entity {
        self.spawn_bundle_inner(bundle, Some(tracker))
    }

    fn spawn_bundle_inner<B: Bundle>(&mut self, bundle: B, mut tracker: Option<&mut ChangeTracker>) -> Entity {
        let entity = self.spawn_inner(tracker.as_deref_mut());

        // `spawn_inner` just pushed `entity` as the LAST element of the
        // empty archetype's entity list and touched nothing else -- pop it
        // back off rather than routing through `migrate_row`'s per-column
        // walk, which would find zero columns to move anyway (a freshly
        // spawned entity owns no component data yet). This is what makes
        // `spawn_bundle` a pure entity-list move instead of a migration.
        {
            let empty = &mut self.archetypes[ArchetypeId::EMPTY.0 as usize];
            debug_assert_eq!(
                empty.entities.last().copied(),
                Some(entity),
                "spawn_bundle: entity must be the last one spawned into the empty archetype"
            );
            empty.entities.pop();
        }

        let dest = B::dest_archetype(self, ArchetypeId::EMPTY);
        let row = {
            let arch = &mut self.archetypes[dest.0 as usize];
            let row = arch.entities.len() as u32;
            arch.entities.push(entity);
            row
        };
        self.entity_slots[entity.index() as usize].archetype = dest;
        self.entity_slots[entity.index() as usize].row = row;

        bundle.push_new(self, dest, entity, &mut tracker);
        entity
    }

    /// Insert every component in `bundle` onto an existing `entity`.
    ///
    /// Unlike [`spawn_bundle`](Self::spawn_bundle), `entity` may already own
    /// some of these component types, so this is a convenience wrapper over
    /// one [`World::insert`] per component (each already edge-cached) rather
    /// than a single-migration fast path -- see [`Bundle::insert_each`]'s
    /// doc.
    ///
    /// # Panics
    /// Panics if `entity` is dead.
    pub fn insert_bundle<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        assert!(self.is_alive(entity), "insert_bundle on dead entity {entity}");
        let mut tracker: Option<&mut ChangeTracker> = None;
        bundle.insert_each(self, entity, &mut tracker);
    }

    /// Like [`insert_bundle`](Self::insert_bundle) but also records every
    /// component change in a [`ChangeTracker`] for replication.
    pub fn insert_bundle_tracked<B: Bundle>(&mut self, entity: Entity, bundle: B, tracker: &mut ChangeTracker) {
        assert!(self.is_alive(entity), "insert_bundle_tracked on dead entity {entity}");
        let mut tracker: Option<&mut ChangeTracker> = Some(tracker);
        bundle.insert_each(self, entity, &mut tracker);
    }
}
