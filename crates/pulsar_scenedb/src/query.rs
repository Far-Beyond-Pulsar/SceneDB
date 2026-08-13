use crate::component::component_id;
use crate::archetype::Archetype;
use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;
use std::marker::PhantomData;

/// Types that can be fetched from an archetype row during a query.
///
/// Implementations exist for:
/// - `&T` â€” shared reference to a component
/// - `&mut T` â€” mutable reference to a component
/// - `()` â€” matches every archetype (for counting or iteration without data)
/// - Tuples `(A, B, ...)` up to 8 elements â€” combine multiple fetches
///
/// # Safety
///
/// `fetch` uses `unsafe` because it performs unchecked column access.  The
/// caller guarantees that `matches(archetype)` is `true` and `row` is within
/// the archetype's entity count.
pub trait WorldQuery<'w>: Sized {
    /// The type returned by [`fetch`](Self::fetch).
    type Item;

    /// Returns `true` if the given archetype contains all the components
    /// required by this query.
    fn matches(archetype: &Archetype) -> bool;

    /// Read component data at `row` in `archetype`.
    ///
    /// # Safety
    ///
    /// - `archetype` must satisfy `Self::matches(archetype)`.
    /// - `row` must be < `archetype.entities.len()`.
    unsafe fn fetch(archetype: &'w Archetype, row: usize) -> Self::Item;
}

impl<'w, T: Component> WorldQuery<'w> for &'w T {
    type Item = &'w T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    // SAFETY: caller guarantees archetype matches and row is in bounds.
    #[inline]
    unsafe fn fetch(arch: &'w Archetype, row: usize) -> &'w T {
        let cid = component_id::<T>();
        // SAFETY: caller guarantees the column exists and row is in bounds.
        let col = arch.columns.get_unchecked(cid.0 as usize)
            .as_ref()
            .unwrap_unchecked();
        &*(col.get_raw(row) as *const T)
    }
}

impl<'w, T: Component> WorldQuery<'w> for &'w mut T {
    type Item = &'w mut T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    // SAFETY: caller guarantees archetype matches and row is in bounds.
    #[inline]
    unsafe fn fetch(arch: &'w Archetype, row: usize) -> &'w mut T {
        let cid = component_id::<T>();
        // SAFETY: caller guarantees the column exists and row is in bounds.
        let ptr = arch.columns.get_unchecked(cid.0 as usize)
            .as_ref()
            .unwrap_unchecked()
            .get_raw(row) as *mut T;
        &mut *ptr
    }
}

// â”€â”€ Empty query: matches every archetype (useful for counting entities) â”€â”€â”€â”€â”€â”€

impl<'w> WorldQuery<'w> for () {
    type Item = ();
    #[inline]
    fn matches(_arch: &Archetype) -> bool {
        true
    }
    // SAFETY: caller guarantees row is in bounds.
    #[inline]
    unsafe fn fetch(_arch: &'w Archetype, _row: usize) -> Self::Item {}
}

// â”€â”€ Tuple conbinator macro (1 to 8 components) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

macro_rules! impl_world_query_tuple {
    ($($Q:ident),+) => {
        impl<'w, $($Q: WorldQuery<'w>),+> WorldQuery<'w> for ($($Q,)+) {
            type Item = ($($Q::Item,)+);

            #[inline]
            fn matches(arch: &Archetype) -> bool {
                $($Q::matches(arch))&&+
            }

            // SAFETY: caller guarantees all Q::matches & row in bounds.
            #[inline]
            unsafe fn fetch(arch: &'w Archetype, row: usize) -> Self::Item {
                ($($Q::fetch(arch, row),)+)
            }
        }
    };
}

impl_world_query_tuple!(A);
impl_world_query_tuple!(A, B);
impl_world_query_tuple!(A, B, C);
impl_world_query_tuple!(A, B, C, D);
impl_world_query_tuple!(A, B, C, D, E);
impl_world_query_tuple!(A, B, C, D, E, F);
impl_world_query_tuple!(A, B, C, D, E, F, G);
impl_world_query_tuple!(A, B, C, D, E, F, G, H);

// â”€â”€ QueryIter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Iterator over all entities in the [`World`](crate::World) that match a query
/// pattern `Q`.
///
/// Yields `(Entity, Q::Item)` pairs.  Created by [`World::query`](crate::World::query).
///
/// The iterator scans archetypes in order, skipping those that don't match `Q`.
/// Within each matching archetype it walks rows sequentially.
///
/// `Q::matches` is evaluated exactly ONCE per archetype (in
/// [`Self::advance_to_next_matching_archetype`], called only on an
/// archetype transition), not once per entity. For a query with N
/// components, `Q::matches` chains N `component_id::<T>()` +
/// `Archetype::has_columns` checks — re-running that on every single
/// `next()` call (as an earlier version of this iterator did, by
/// re-checking `Q::matches(arch)` at the top of its per-row loop) meant an
/// archetype with 10,000 entities paid for 10,000 redundant match
/// evaluations instead of 1. `current_len` caches the matching archetype's
/// row count so the hot per-entity path is just an index compare, an
/// unchecked slice read, and `Q::fetch` — no re-validation.
pub struct QueryIter<'w, Q: WorldQuery<'w>> {
    archetypes: &'w [crate::archetype::Archetype],
    arch_idx: usize,
    row: usize,
    /// `archetypes[arch_idx].entities.len()` for the CURRENT matching
    /// archetype, cached at the last archetype transition. `0` once
    /// iteration is exhausted (`arch_idx` has run past the end).
    current_len: usize,
    _marker: PhantomData<Q>,
}

impl<'w, Q: WorldQuery<'w>> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w World) -> Self {
        let mut iter = Self {
            archetypes: &world.archetypes,
            arch_idx: 0,
            row: 0,
            current_len: 0,
            _marker: PhantomData,
        };
        iter.advance_to_next_matching_archetype();
        iter
    }

    /// Scans forward from the current `arch_idx` (inclusive) for the next
    /// archetype that both matches `Q` and has at least one entity, and
    /// positions the iterator there (`row = 0`, `current_len` = its entity
    /// count). If none remain, leaves `arch_idx == archetypes.len()` and
    /// `current_len = 0` — [`Iterator::next`]'s exhaustion check.
    ///
    /// This is the ONLY place `Q::matches` is ever called — once per
    /// archetype visited, never once per entity.
    #[inline]
    fn advance_to_next_matching_archetype(&mut self) {
        while self.arch_idx < self.archetypes.len() {
            // SAFETY: bounds-checked by the loop condition.
            let arch = unsafe { self.archetypes.get_unchecked(self.arch_idx) };
            if !arch.entities.is_empty() && Q::matches(arch) {
                self.current_len = arch.entities.len();
                self.row = 0;
                return;
            }
            self.arch_idx += 1;
        }
        self.current_len = 0;
    }
}

impl<'w, Q: WorldQuery<'w>> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.current_len {
            // Exhausted the current matching archetype (or this is the
            // first call and `new()` found nothing) -- move past it and
            // find the next one. `current_len == 0` after this means no
            // matching archetype remains anywhere ahead.
            self.arch_idx += 1;
            self.advance_to_next_matching_archetype();
            if self.current_len == 0 {
                return None;
            }
        }
        // SAFETY: `advance_to_next_matching_archetype` only ever leaves
        // `arch_idx` pointing at an archetype that matches Q, with
        // `current_len` equal to its entity count -- `arch_idx` is
        // in-bounds and `row < current_len <= arch.entities.len()`.
        let arch = unsafe { self.archetypes.get_unchecked(self.arch_idx) };
        let entity = unsafe { *arch.entities.get_unchecked(self.row) };
        // SAFETY: `arch` matches Q (verified when we last advanced onto
        // it) and `row` is in bounds (checked above).
        let item = unsafe { Q::fetch(arch, self.row) };
        self.row += 1;
        Some((entity, item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Cheap, exact-if-single-archetype lower bound: at least the
        // remaining rows in the current archetype. Computing the true
        // upper bound would require scanning every remaining archetype's
        // `Q::matches` -- exactly the per-call cost this iterator now
        // avoids -- so the upper bound stays `None` rather than paying for
        // it unconditionally on every `size_hint()` call.
        (self.current_len.saturating_sub(self.row), None)
    }
}

impl World {
    /// Iterate all entities whose components match the query pattern `Q`.
    ///
    /// # Example
    ///
    /// ```
    /// use pulsar_scenedb::{World, QueryIter, WorldQuery};
    ///
    /// # struct Pos(f32, f32);
    /// # struct Vel(f32, f32);
    /// # let mut world = World::new();
    /// for (pos, vel) in world.query::<(&Pos, &Vel)>() {
    ///     // ...
    /// }
    /// ```
    ///
    /// An empty tuple `()` matches every archetype and can be used to iterate
    /// all entities without fetching any component data.
    pub fn query<'w, Q: WorldQuery<'w>>(&'w self) -> QueryIter<'w, Q> {
        QueryIter::new(self)
    }
}