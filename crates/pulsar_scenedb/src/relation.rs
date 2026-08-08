//! Columnar relational indexing over `World` components (design plan §4:
//! "For relational patterns (like `PortalComponent` links or multi-body
//! attachments)... provide a Columnar Relation View").
//!
//! One correction versus the plan's sketch: the pair type is
//! [`crate::Entity`], not `Handle`. `Handle` is this crate's GPU-cell row
//! identity (`registry.rs`/`gpu`); the CPU archetype ECS (`World`,
//! `query.rs`) that actually holds component data like a portal's link
//! field identifies rows with `Entity`. A relation built by scanning
//! `World` components has `Entity` pairs, not `Handle` pairs, by
//! construction — there is no `Handle` in scope to produce one from.
//!
//! [`RelationIndex::build`] takes a link-extractor closure rather than
//! assuming a fixed `PortalComponent` type or field name, per the plan's
//! own decoupling principle (§1.3: "SceneDB remains agnostic to specific
//! subsystems") — it doesn't know what a portal is, only how to ask a
//! caller-supplied component for the `Entity` it points at.

use std::collections::{HashMap, HashSet};

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

/// Why a source entity's link didn't resolve into a confirmed pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictReason {
    /// `source` links to `target`, but `target` either has no link back to
    /// `source` (points elsewhere, or has no relevant component) — the
    /// wrapped `Entity` is what `target` links to instead, if anything
    /// (`Entity::DANGLING` if `target` isn't in the index at all).
    NotReciprocated(Entity),
}

/// One source entity whose link did not resolve into a confirmed pair.
#[derive(Debug, Clone, Copy)]
pub struct ConflictEntry {
    pub source: Entity,
    pub target: Entity,
    pub reason: ConflictReason,
}

/// Owns the dense pair/unmatched/conflict buffers a [`RelationView`]
/// borrows from. Rebuilt on demand via [`RelationIndex::build`] (typically
/// once per boundary, not per read) — reads during harvest borrow the
/// already-built buffers through [`RelationIndex::view`], no allocation or
/// per-row dispatch on the read path (design plan §4.3).
#[derive(Default)]
pub struct RelationIndex {
    pairs: Vec<(Entity, Entity)>,
    unmatched: Vec<Entity>,
    conflicts: Vec<ConflictEntry>,
}

impl RelationIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the index from every `C` component currently in `world`,
    /// using `link` to read each component's target `Entity`.
    ///
    /// A pair `(a, b)` is confirmed when `a`'s component links to `b` *and*
    /// `b`'s component links back to `a` — reciprocal by construction, so
    /// each confirmed pair is emitted exactly once (in the order its first
    /// member was encountered by `World::query`, not by entity index).
    /// Everything else is either `unmatched` (the target has no `C`, or no
    /// link back to this source) or a `conflicts` entry carrying what the
    /// target links to instead, for the caller to decide how to resolve
    /// (log it, drop the stale link, etc.) — `RelationIndex` only reports.
    pub fn build<C: Component>(&mut self, world: &World, link: impl Fn(&C) -> Entity) {
        self.pairs.clear();
        self.unmatched.clear();
        self.conflicts.clear();

        let mut links: HashMap<Entity, Entity> = HashMap::new();
        for (entity, component) in world.query::<&C>() {
            links.insert(entity, link(component));
        }

        let mut resolved: HashSet<Entity> = HashSet::with_capacity(links.len());
        for (&source, &target) in &links {
            if resolved.contains(&source) {
                continue;
            }
            match links.get(&target) {
                Some(&back) if back == source => {
                    self.pairs.push((source, target));
                    resolved.insert(source);
                    resolved.insert(target);
                }
                Some(&back) => {
                    self.conflicts.push(ConflictEntry {
                        source,
                        target,
                        reason: ConflictReason::NotReciprocated(back),
                    });
                    resolved.insert(source);
                }
                None => {
                    self.unmatched.push(source);
                    resolved.insert(source);
                }
            }
        }
    }

    /// Borrow the current buffers as dense, contiguous slices — the read
    /// path subsystems actually use during harvest.
    pub fn view(&self) -> RelationView<'_> {
        RelationView {
            pairs: &self.pairs,
            unmatched: &self.unmatched,
            conflicts: &self.conflicts,
        }
    }
}

/// Dense, contiguous read-only view over a [`RelationIndex`]'s current
/// buffers — no allocation, no per-row dynamic dispatch. Borrowed, not
/// owned: build (or rebuild) a `RelationIndex` first, then call
/// [`RelationIndex::view`] each time a subsystem's harvest hook needs the
/// current relation state.
pub struct RelationView<'a> {
    pub pairs: &'a [(Entity, Entity)],
    pub unmatched: &'a [Entity],
    pub conflicts: &'a [ConflictEntry],
}

#[cfg(test)]
mod tests {
    use super::*;

    // `Component` is blanket-implemented for every `Any + Send + Sync +
    // 'static` type (see `component.rs`) — no manual impl needed here.
    #[derive(Debug)]
    struct PortalLink {
        linked_to: Entity,
    }

    #[test]
    fn reciprocal_links_pair_once_each() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        world.insert(a, PortalLink { linked_to: b });
        world.insert(b, PortalLink { linked_to: a });

        let mut index = RelationIndex::new();
        index.build::<PortalLink>(&world, |c| c.linked_to);
        let view = index.view();

        assert_eq!(view.pairs.len(), 1);
        let (p0, p1) = view.pairs[0];
        assert!((p0 == a && p1 == b) || (p0 == b && p1 == a));
        assert!(view.unmatched.is_empty());
        assert!(view.conflicts.is_empty());
    }

    #[test]
    fn one_sided_link_is_unmatched_and_broken_reciprocity_is_a_conflict() {
        let mut world = World::new();
        let lonely = world.spawn();
        let target_without_component = world.spawn();
        world.insert(
            lonely,
            PortalLink {
                linked_to: target_without_component,
            },
        );

        let wants_c = world.spawn();
        let wants_a_instead = world.spawn();
        let actually_linked = world.spawn();
        world.insert(
            wants_c,
            PortalLink {
                linked_to: actually_linked,
            },
        );
        world.insert(
            actually_linked,
            PortalLink {
                linked_to: wants_a_instead,
            },
        );
        world.insert(
            wants_a_instead,
            PortalLink {
                linked_to: actually_linked,
            },
        );

        let mut index = RelationIndex::new();
        index.build::<PortalLink>(&world, |c| c.linked_to);
        let view = index.view();

        assert_eq!(view.pairs.len(), 1);
        let (p0, p1) = view.pairs[0];
        assert!(
            (p0 == actually_linked && p1 == wants_a_instead)
                || (p0 == wants_a_instead && p1 == actually_linked)
        );

        assert_eq!(view.unmatched, &[lonely]);

        assert_eq!(view.conflicts.len(), 1);
        assert_eq!(view.conflicts[0].source, wants_c);
        assert_eq!(view.conflicts[0].target, actually_linked);
        assert_eq!(
            view.conflicts[0].reason,
            ConflictReason::NotReciprocated(wants_a_instead)
        );
    }
}
