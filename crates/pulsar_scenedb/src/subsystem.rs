//! External subsystem registration, hooked into SceneDB's real phase
//! machine (`gpu::phase`) instead of an invented one.
//!
//! This module intentionally does NOT introduce a parallel reflection
//! layer. Dynamic (script/event) dispatch is meant to reuse
//! `pulsar_reflection`'s existing link-time method registry
//! (`MethodMetadata`/`MethodCaller`/`inventory`, see
//! `pulsar_reflection::registry::ComponentMethodRegistration`) — see the
//! module-level doc for why that needs one small upstream change
//! (`MethodCaller`'s receiver generalized from `&mut dyn EngineClass` to
//! `&mut dyn Any`) before a `SubsystemMethodRegistration` can plug into it
//! verbatim. That change is out of scope here; this module ships the half
//! that only needs SceneDB's own crate: the phase-gated hook trait and the
//! typed registry.
//!
//! ## Corrections versus the speculative "SceneDB Subsystem Registry" plan
//!
//! - There is no unified `SceneDB` type to hang `subsystem_mut::<T>(witness)`
//!   off of. CPU state is [`crate::World`]; GPU residency is
//!   [`crate::gpu::SceneGpuStore`] (+ `CellSlot`). A subsystem that needs
//!   both takes both, explicitly.
//! - `SimulateWitness` is **sealed** (`gpu::phase::private::Sealed`): only
//!   `SimulateA`/`SimulateB` implement it, and no downstream crate — this
//!   one included — can add a third. That's fine for *taking* `&SimulateA`/
//!   `&SimulateB` params (sealing blocks new impls, not consumption), but it
//!   is fatal to a `dyn Subsystem` with a **generic** `simulate<W:
//!   SimulateWitness>(&mut self, ...)` method: a trait with a non-`Sized`-
//!   exempt generic method cannot be called through a trait object at all,
//!   which breaks the entire point of a registry that dispatches to boxed
//!   subsystems uniformly. This module uses two concrete, object-safe hooks
//!   (`simulate_a`/`simulate_b`) instead — which also matches the crate's
//!   own documented A=gameplay / B=physics-writeback split.
//! - There is no single boundary witness spanning retire → compact → sync;
//!   each stage consumes the previous witness and produces the next
//!   (`BoundaryPhase::retire` → [`RetiredPhase`] → `compact` →
//!   `CompactedPhase` → `sync`). [`RetiredPhase`] is the one pause point
//!   already exposed publicly (for exactly this reason — see its doc), so
//!   the `boundary` hook is gated on that, not on a fictional `RetiredPhase`
//!   that means "the whole boundary".

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::gpu::{HarvestPhase, RetiredPhase, SceneGpuStore, SimulateA, SimulateB};
use crate::World;

/// A named engine subsystem hooked into SceneDB's phase machine.
///
/// All hooks default to no-ops — implement only the phases a given
/// subsystem actually needs. `Any + Send + Sync` (not just `Any`) matches
/// the rest of the crate's threading assumptions (`SystemFn` in
/// `schedule.rs` carries the same bounds).
pub trait Subsystem: Any + Send + Sync {
    /// Stable registry key. Also used as the dynamic-dispatch name once
    /// method dispatch lands (see module docs).
    fn name(&self) -> &'static str;

    /// Runs during the gameplay simulate sub-phase (C3 "A"). Mutation of
    /// `World` is permitted.
    fn simulate_a(&mut self, _world: &mut World, _witness: &SimulateA) {}

    /// Runs during the physics-writeback simulate sub-phase (C3 "B").
    fn simulate_b(&mut self, _world: &mut World, _witness: &SimulateB) {}

    /// Runs during the harvest phase. Read-only by construction — neither
    /// `HarvestPhase` nor `&SceneGpuStore` grants a mutation path.
    fn harvest(&mut self, _store: &SceneGpuStore, _phase: &HarvestPhase) {}

    /// Runs at the retire/compact pause point inside the frame boundary
    /// (see module docs on why this is `RetiredPhase` and not "the whole
    /// boundary").
    fn boundary(&mut self, _phase: &RetiredPhase) {}

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Owned subsystem instances, indexed by Rust `TypeId` for the static path
/// and by name for the dynamic path. SceneDB stays agnostic to which
/// concrete subsystems exist — registration is the caller's job (typically
/// once at startup).
#[derive(Default)]
pub struct SubsystemRegistry {
    instances: HashMap<TypeId, Box<dyn Subsystem>>,
    name_to_type: HashMap<&'static str, TypeId>,
}

impl SubsystemRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subsystem instance. Replaces any prior instance of the
    /// same concrete type (last-write-wins), matching `HashMap::insert`.
    pub fn register<T: Subsystem + 'static>(&mut self, instance: T) {
        let type_id = TypeId::of::<T>();
        self.name_to_type.insert(instance.name(), type_id);
        self.instances.insert(type_id, Box::new(instance));
    }

    /// Static path (hot loop): zero-cost typed borrow by concrete type.
    pub fn get<T: Subsystem + 'static>(&self) -> Option<&T> {
        self.instances
            .get(&TypeId::of::<T>())
            .and_then(|b| b.as_any().downcast_ref::<T>())
    }

    /// Static path (hot loop): zero-cost typed mutable borrow.
    pub fn get_mut<T: Subsystem + 'static>(&mut self) -> Option<&mut T> {
        self.instances
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.as_any_mut().downcast_mut::<T>())
    }

    /// Dynamic path (scripts/events): borrow by registered name.
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut dyn Subsystem> {
        let type_id = *self.name_to_type.get(name)?;
        self.instances.get_mut(&type_id).map(|b| b.as_mut())
    }

    /// Runs every registered subsystem's `simulate_a` hook, in registration
    /// order is NOT guaranteed (`HashMap` iteration) — subsystems that need
    /// ordering against each other should say so explicitly; SceneDB's own
    /// `Schedule` (see `schedule.rs`) documents the same non-goal for its
    /// closure-based systems.
    pub fn simulate_a(&mut self, world: &mut World, witness: &SimulateA) {
        for subsystem in self.instances.values_mut() {
            subsystem.simulate_a(world, witness);
        }
    }

    pub fn simulate_b(&mut self, world: &mut World, witness: &SimulateB) {
        for subsystem in self.instances.values_mut() {
            subsystem.simulate_b(world, witness);
        }
    }

    pub fn harvest(&mut self, store: &SceneGpuStore, phase: &HarvestPhase) {
        for subsystem in self.instances.values_mut() {
            subsystem.harvest(store, phase);
        }
    }

    pub fn boundary(&mut self, phase: &RetiredPhase) {
        for subsystem in self.instances.values_mut() {
            subsystem.boundary(phase);
        }
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::FrameDriver;

    struct Counter {
        simulate_a_calls: u32,
        boundary_calls: u32,
    }

    impl Subsystem for Counter {
        fn name(&self) -> &'static str {
            "counter"
        }

        fn simulate_a(&mut self, _world: &mut World, _witness: &SimulateA) {
            self.simulate_a_calls += 1;
        }

        fn boundary(&mut self, _phase: &RetiredPhase) {
            self.boundary_calls += 1;
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn static_and_dynamic_paths_reach_the_same_instance() {
        let mut registry = SubsystemRegistry::new();
        registry.register(Counter {
            simulate_a_calls: 0,
            boundary_calls: 0,
        });

        let mut world = World::new();
        let mut driver = FrameDriver::new();
        let sim_a = driver.begin();
        registry.simulate_a(&mut world, &sim_a);

        assert_eq!(registry.get::<Counter>().unwrap().simulate_a_calls, 1);
        assert!(registry.get_by_name_mut("counter").is_some());
        assert!(registry.get_by_name_mut("missing").is_none());
    }
}
