//! `SceneDb`: the facade the subsystem-registry plan asked for
//! (`SceneDB::step()`, `scene_db.subsystem_mut::<T>()`), built as new code
//! on top of what already exists rather than assumed to be sitting there.
//!
//! Two real constraints from the rest of the crate shape this:
//!
//! - **C0 ("the core stays graphics-free")**: `gpu::SceneGpuStore` needs a
//!   real `EngineGpuContext` (device + queue) to exist at all — see
//!   `gpu::scene_store::SceneGpuStore::new`. `SceneDb` has no business
//!   owning a GPU device, so it does not own a `SceneGpuStore`; callers
//!   that have one pass it into [`SceneDb::step_gpu`] explicitly, same as
//!   every other GPU-phase API in this crate.
//! - **Phase tokens are per-call capability proofs, not a frame counter**:
//!   `FrameDriver::begin` (see `gpu::phase`) doesn't track whether a
//!   previous chain ever reached `BoundaryPhase` — nothing in the type
//!   system requires a caller to run the GPU phases every "frame". So
//!   `step` (CPU: SimulateA/SimulateB) and `step_gpu` (Harvest/Boundary)
//!   are independent entry points a host can call at different
//!   cadences, not two halves of one call that must always run together.
//!
//! `subsystem_mut::<T>()` intentionally does not take a witness parameter
//! the way the original sketch had it (`subsystem_mut::<T>(witness)`):
//! phase gating here happens by *which method you're inside*
//! (`step`/`step_gpu` hand the witness to the subsystem trait's hooks
//! directly), not by a value threaded through an unrelated accessor. A
//! witness parameter on a plain getter would be decorative, not load-bearing
//! — it wouldn't stop you from calling `subsystem_mut` outside a phase.

use crate::gpu::FrameDriver;
use crate::World;

use super::subsystem::{Subsystem, SubsystemRegistry};

#[cfg(feature = "gpu")]
use crate::gpu::{CellSlot, SceneGpuStore, SyncStats};

/// Owns CPU scene state (`World`) and the subsystem registry; drives both
/// through SceneDB's real phase machine. See module docs for why this does
/// *not* also own a `SceneGpuStore`.
pub struct SceneDb {
    pub world: World,
    subsystems: SubsystemRegistry,
    driver: FrameDriver,
}

impl SceneDb {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            subsystems: SubsystemRegistry::new(),
            driver: FrameDriver::new(),
        }
    }

    /// Register a subsystem instance. See [`SubsystemRegistry::register`].
    pub fn register_subsystem<T: Subsystem + 'static>(&mut self, instance: T) {
        self.subsystems.register(instance);
    }

    /// Static path (hot loop): zero-cost typed borrow by concrete type.
    pub fn subsystem<T: Subsystem + 'static>(&self) -> Option<&T> {
        self.subsystems.get::<T>()
    }

    /// Static path (hot loop): zero-cost typed mutable borrow.
    pub fn subsystem_mut<T: Subsystem + 'static>(&mut self) -> Option<&mut T> {
        self.subsystems.get_mut::<T>()
    }

    /// Dynamic path (scripts/events): borrow by registered name.
    pub fn subsystem_by_name_mut(&mut self, name: &str) -> Option<&mut dyn Subsystem> {
        self.subsystems.get_by_name_mut(name)
    }

    /// Dynamic path (scripts/events): invoke a subsystem method by string
    /// name through Pulsar's reflection database. See
    /// [`SubsystemRegistry::dispatch`].
    pub fn dispatch(
        &mut self,
        subsystem_name: &str,
        method_name: &str,
        args: pulsar_reflection::DynMethodArgs,
    ) -> Result<pulsar_reflection::DynMethodReturnValue, super::subsystem::SubsystemDispatchError>
    {
        self.subsystems.dispatch(subsystem_name, method_name, args)
    }

    /// Direct registry access, for callers that want to run a phase hook
    /// across every subsystem themselves instead of going through `step`/
    /// `step_gpu` (e.g. a custom frame loop).
    pub fn subsystems(&mut self) -> &mut SubsystemRegistry {
        &mut self.subsystems
    }

    /// CPU-only simulate step: `SimulateA` → `SimulateB`, dispatching every
    /// registered subsystem's `simulate_a`/`simulate_b` hook. Does not
    /// touch the GPU phases — see [`Self::step_gpu`] and the module docs
    /// on why those are separate.
    pub fn step(&mut self) {
        let sim_a = self.driver.begin();
        self.subsystems.simulate_a(&mut self.world, &sim_a);
        let sim_b = sim_a.end();
        self.subsystems.simulate_b(&mut self.world, &sim_b);
    }

    /// GPU-phase step: `Harvest` → `Boundary` (retire → compact → sync),
    /// dispatching every registered subsystem's `harvest`/`boundary` hook.
    /// Takes the caller's real `SceneGpuStore`/`CellSlot`s (see module docs
    /// on C0). Runs its own `SimulateA`→`SimulateB` internally purely to
    /// reach a `HarvestPhase` witness — those sub-phases carry no mutation
    /// here (`world` is not touched by this method) so no subsystem
    /// `simulate_a`/`simulate_b` hook is invoked for it; call [`Self::step`]
    /// for that.
    #[cfg(feature = "gpu")]
    pub fn step_gpu(&mut self, store: &mut SceneGpuStore, cells: &mut [CellSlot<'_>]) -> SyncStats {
        let sim_a = self.driver.begin();
        let sim_b = sim_a.end();
        let harvest = sim_b.end();
        self.subsystems.harvest(store, &harvest);
        let boundary = harvest.end();
        let (retired, _drained) = boundary.retire(store, cells);
        self.subsystems.boundary(&retired);
        retired.compact(store, cells).sync(store, cells)
    }
}

impl Default for SceneDb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{SimulateA, SimulateB};
    use std::any::Any;

    struct Ticker {
        simulate_a_calls: u32,
        simulate_b_calls: u32,
    }

    impl Subsystem for Ticker {
        fn name(&self) -> &'static str {
            "ticker"
        }

        fn simulate_a(&mut self, _world: &mut World, _witness: &SimulateA) {
            self.simulate_a_calls += 1;
        }

        fn simulate_b(&mut self, _world: &mut World, _witness: &SimulateB) {
            self.simulate_b_calls += 1;
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn step_drives_both_simulate_sub_phases_on_every_registered_subsystem() {
        let mut db = SceneDb::new();
        db.register_subsystem(Ticker {
            simulate_a_calls: 0,
            simulate_b_calls: 0,
        });

        db.step();
        db.step();

        let ticker = db.subsystem::<Ticker>().expect("registered");
        assert_eq!(ticker.simulate_a_calls, 2);
        assert_eq!(ticker.simulate_b_calls, 2);

        assert!(db.subsystem_by_name_mut("ticker").is_some());
        assert!(db.subsystem_by_name_mut("missing").is_none());
    }
}
