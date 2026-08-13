use crate::archetype::Archetype;
use crate::component::{component_id, Column, Component};
use crate::entity::Entity;
use crate::world::World;
use std::marker::PhantomData;

/// Types that can be fetched from an archetype row during a query.
///
/// Implementations exist for:
/// - `&T` — shared reference to a component
/// - `&mut T` — mutable reference to a component
/// - `()` — matches every archetype (for counting or iteration without data)
/// - Tuples `(A, B, ...)` up to 8 elements — combine multiple fetches
///
/// # Design: `init_fetch` once per archetype, `fetch` once per row
///
/// Resolving "where does `T`'s data live in this archetype" (looking up
/// `component_id::<T>()`, indexing into `Archetype::columns`, downcasting the
/// type-erased column) is an ARCHETYPE-level fact — every row in a matching
/// archetype shares the exact same answer. [`Self::init_fetch`] computes
/// that once, into [`Self::Fetch`] (typically a raw base pointer);
/// [`Self::fetch`] then reads a single row from it via pure pointer
/// arithmetic — no `component_id` lookup, no `dyn ErasedColumn` virtual
/// dispatch, on the per-entity hot path. An earlier version of this trait
/// had a single `fetch(archetype, row)` method that repeated the
/// column-resolution work on every row — for an N-component query over an
/// archetype with M entities, that's N*M redundant lookups instead of N.
///
/// # Safety
///
/// Both methods are `unsafe`: the caller (always [`QueryIter`] in this
/// crate) guarantees `matches(archetype)` was checked before calling
/// `init_fetch`, that any `Self::Fetch` passed to `fetch` came from
/// `init_fetch` on the SAME archetype, and that `row` is within that
/// archetype's entity count.
pub trait WorldQuery<'w>: Sized {
    /// The type returned by [`fetch`](Self::fetch).
    type Item;

    /// Per-archetype resolved fetch state — e.g. a raw base pointer into a
    /// column's backing storage. Computed once per archetype by
    /// [`Self::init_fetch`], then reused for every row via [`Self::fetch`].
    type Fetch;

    /// Returns `true` if the given archetype contains all the components
    /// required by this query.
    fn matches(archetype: &Archetype) -> bool;

    /// Resolve this query's per-archetype fetch state. Called exactly once
    /// per archetype the iterator visits — never once per row.
    ///
    /// # Safety
    /// - `archetype` must satisfy `Self::matches(archetype)`.
    unsafe fn init_fetch(archetype: &'w Archetype) -> Self::Fetch;

    /// Read component data at `row`, given the per-archetype state from
    /// [`Self::init_fetch`].
    ///
    /// # Safety
    /// - `fetch` must have come from [`Self::init_fetch`] called on the
    ///   same archetype `row` belongs to.
    /// - `row` must be `<` that archetype's entity count.
    unsafe fn fetch(fetch: &Self::Fetch, row: usize) -> Self::Item;
}

impl<'w, T: Component> WorldQuery<'w> for &'w T {
    type Item = &'w T;
    /// Base pointer into the column's `Vec<T>` backing storage — `fetch`
    /// becomes `*base.add(row)`, no lookup, no dispatch.
    type Fetch = *const T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    // SAFETY: caller guarantees `matches(archetype)` -- the column exists
    // and (crate-wide invariant: `Archetype::columns` is dense-indexed by
    // `ComponentId`, and a given id always names the same concrete type
    // everywhere in the crate -- see `column_mut`'s identical assumption)
    // is a `Column<T>`.
    #[inline]
    unsafe fn init_fetch(arch: &'w Archetype) -> Self::Fetch {
        let cid = component_id::<T>();
        let col = arch
            .columns
            .get(cid.0 as usize)
            .and_then(|c| c.as_ref())
            .expect("WorldQuery::init_fetch: matches() must have verified this column exists");
        col.as_any()
            .downcast_ref::<Column<T>>()
            .expect("WorldQuery::init_fetch: column type mismatch -- ComponentId<->T invariant violated")
            .data
            .as_ptr()
    }

    // SAFETY: caller guarantees `fetch` came from `init_fetch` on this
    // archetype and `row` is in bounds -- `fetch.add(row)` is therefore a
    // valid, initialized `T`.
    #[inline]
    unsafe fn fetch(fetch: &Self::Fetch, row: usize) -> &'w T {
        unsafe { &*fetch.add(row) }
    }
}

impl<'w, T: Component> WorldQuery<'w> for &'w mut T {
    type Item = &'w mut T;
    type Fetch = *mut T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    // SAFETY: same column-resolution contract as `&T`'s `init_fetch`. The
    // `*const T -> *mut T` cast here mirrors what the single-method
    // version of this trait already did per-row (via
    // `ErasedColumn::get_raw`, also an immutable read cast to `*mut`) --
    // this redesign moves WHEN that happens (once per archetype, not once
    // per row), not the aliasing contract itself: the caller
    // (`QueryIter`/`World::query`) is responsible for the same "no two live
    // `&mut T` ever alias the same row, and no `&T`/`&mut T` from this
    // query outlives independent access to the same World data" discipline
    // it always was.
    #[inline]
    unsafe fn init_fetch(arch: &'w Archetype) -> Self::Fetch {
        let cid = component_id::<T>();
        let col = arch
            .columns
            .get(cid.0 as usize)
            .and_then(|c| c.as_ref())
            .expect("WorldQuery::init_fetch: matches() must have verified this column exists");
        col.as_any()
            .downcast_ref::<Column<T>>()
            .expect("WorldQuery::init_fetch: column type mismatch -- ComponentId<->T invariant violated")
            .data
            .as_ptr() as *mut T
    }

    // SAFETY: caller guarantees `fetch` came from `init_fetch` on this
    // archetype and `row` is in bounds.
    #[inline]
    unsafe fn fetch(fetch: &Self::Fetch, row: usize) -> &'w mut T {
        unsafe { &mut *fetch.add(row) }
    }
}

// ── Empty query: matches every archetype (useful for counting entities) ────

impl<'w> WorldQuery<'w> for () {
    type Item = ();
    type Fetch = ();
    #[inline]
    fn matches(_arch: &Archetype) -> bool {
        true
    }
    #[inline]
    unsafe fn init_fetch(_arch: &'w Archetype) -> Self::Fetch {}
    #[inline]
    unsafe fn fetch(_fetch: &Self::Fetch, _row: usize) -> Self::Item {}
}

// ── Tuple conbinator macro (1 to 8 components) ──────────────────────────

macro_rules! impl_world_query_tuple {
    ($($Q:ident),+) => {
        impl<'w, $($Q: WorldQuery<'w>),+> WorldQuery<'w> for ($($Q,)+) {
            type Item = ($($Q::Item,)+);
            type Fetch = ($($Q::Fetch,)+);

            #[inline]
            fn matches(arch: &Archetype) -> bool {
                $($Q::matches(arch))&&+
            }

            // SAFETY: caller guarantees `matches(archetype)`, which chains
            // every $Q::matches -- so each $Q::init_fetch's own contract is
            // satisfied too.
            #[inline]
            unsafe fn init_fetch(arch: &'w Archetype) -> Self::Fetch {
                unsafe { ($($Q::init_fetch(arch),)+) }
            }

            // SAFETY: caller guarantees `fetch` came from this impl's
            // `init_fetch` on the row's archetype and `row` is in bounds --
            // so each element of the destructured tuple is a valid
            // $Q::Fetch for that same archetype, satisfying $Q::fetch's
            // contract.
            #[inline]
            unsafe fn fetch(fetch: &Self::Fetch, row: usize) -> Self::Item {
                #[allow(non_snake_case)]
                let ($($Q,)+) = fetch;
                unsafe { ($($Q::fetch($Q, row),)+) }
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

// ── QueryIter ────────────────────────────────────────────────────────────

/// Iterator over all entities in the [`World`](crate::World) that match a query
/// pattern `Q`.
///
/// Yields `(Entity, Q::Item)` pairs.  Created by [`World::query`](crate::World::query).
///
/// The iterator scans archetypes in order, skipping those that don't match `Q`.
/// Within each matching archetype it walks rows sequentially.
///
/// `Q::matches` and `Q::init_fetch` are evaluated exactly ONCE per
/// archetype (in [`Self::advance_to_next_matching_archetype`], called only
/// on an archetype transition) — never once per entity. The per-entity hot
/// path in [`Iterator::next`] is an index compare, an unchecked slice read
/// for the `Entity`, and `Q::fetch` against the cached per-archetype state
/// — pure pointer arithmetic for `&T`/`&mut T`/tuples thereof, no column
/// lookup, no dispatch.
pub struct QueryIter<'w, Q: WorldQuery<'w>> {
    archetypes: &'w [crate::archetype::Archetype],
    arch_idx: usize,
    row: usize,
    /// `archetypes[arch_idx].entities.len()` for the CURRENT matching
    /// archetype, cached at the last archetype transition. `0` once
    /// iteration is exhausted (`arch_idx` has run past the end).
    current_len: usize,
    /// `Q`'s resolved per-archetype fetch state (see [`WorldQuery::Fetch`]),
    /// cached at the same transition as `current_len`. `None` only before
    /// the first archetype is found and after iteration is exhausted --
    /// [`Iterator::next`] never reads it in either of those states (it
    /// returns `None` first, via the `current_len == 0` check).
    fetch: Option<Q::Fetch>,
    _marker: PhantomData<Q>,
}

impl<'w, Q: WorldQuery<'w>> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w World) -> Self {
        let mut iter = Self {
            archetypes: &world.archetypes,
            arch_idx: 0,
            row: 0,
            current_len: 0,
            fetch: None,
            _marker: PhantomData,
        };
        iter.advance_to_next_matching_archetype();
        iter
    }

    /// Scans forward from the current `arch_idx` (inclusive) for the next
    /// archetype that both matches `Q` and has at least one entity, and
    /// positions the iterator there (`row = 0`, `current_len` = its entity
    /// count, `fetch` = `Q::init_fetch` resolved against it). If none
    /// remain, leaves `arch_idx == archetypes.len()`, `current_len = 0`,
    /// `fetch = None` — [`Iterator::next`]'s exhaustion check.
    ///
    /// This is the ONLY place `Q::matches`/`Q::init_fetch` are ever called
    /// — once per archetype visited, never once per entity.
    #[inline]
    fn advance_to_next_matching_archetype(&mut self) {
        while self.arch_idx < self.archetypes.len() {
            // SAFETY: bounds-checked by the loop condition.
            let arch = unsafe { self.archetypes.get_unchecked(self.arch_idx) };
            if !arch.entities.is_empty() && Q::matches(arch) {
                self.current_len = arch.entities.len();
                self.row = 0;
                // SAFETY: just verified `Q::matches(arch)` on this exact
                // archetype, immediately above.
                self.fetch = Some(unsafe { Q::init_fetch(arch) });
                return;
            }
            self.arch_idx += 1;
        }
        self.current_len = 0;
        self.fetch = None;
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
        // SAFETY: `self.fetch` is `Some` whenever `current_len > 0` (the
        // invariant `advance_to_next_matching_archetype` maintains, and we
        // already returned `None` above if it were 0), and it was computed
        // via `Q::init_fetch` on THIS exact archetype at the last
        // transition. `row` is in bounds, checked above.
        let fetch = unsafe { self.fetch.as_ref().unwrap_unchecked() };
        // SAFETY: `fetch` came from `Q::init_fetch` on this archetype (see
        // above) and `row` is in bounds.
        let item = unsafe { Q::fetch(fetch, self.row) };
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
