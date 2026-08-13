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
//!   `step` (CPU: SimulateA/SimulateB) and `step_gpu` (Harvest/Boundary,
//!   cell-mirrored path) remain independent entry points a host can call
//!   at different cadences, not two halves of one call that must always
//!   run together — a host with no cell-mirrored `SceneGpuStore` never
//!   calls `step_gpu` at all. What's no longer split across them is the
//!   `World`-mirror flush (issue #41): both `step` and `step_gpu`
//!   perform it automatically when a mirror is attached, so a host whose
//!   GPU-mirrored state is entirely `World`-side gets full, automatic GPU
//!   sync from `step()` alone — see that method's doc.
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
use crate::gpu::{CellSlot, GpuMirrorHandle, SceneGpuStore, SyncStats};

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

    /// The normal init path for a GPU-mirrored `SceneDb`: build it already
    /// wired to `mirror`, instead of `SceneDb::new()` followed by a
    /// separate `db.world.attach_gpu_mirror(mirror)` call. This is the ONE
    /// wiring step that can't disappear entirely (C0: the graphics-free
    /// core can never conjure a `wgpu::Device`/`SceneGpuStore` on its own —
    /// someone external has to hand it one), but it now happens exactly
    /// once, at construction, as an ordinary constructor parameter — not a
    /// recurring per-frame step, and not a separate call a host has to
    /// remember to make between `new()` and its first `step()`.
    ///
    /// Everything downstream of this call is automatic: registration on
    /// first insert and flush inside [`Self::step`]/[`Self::step_gpu`] both
    /// need no further manual calls (see [`Self::step`]'s doc).
    ///
    /// `#[cfg(feature = "gpu")]`-gated — [`GpuMirrorHandle`] doesn't exist
    /// without the `gpu` feature (CONTRACTS C0), so this constructor can't
    /// either; [`Self::new`] remains the feature-agnostic entry point for a
    /// `SceneDb` with no GPU wiring (or for attaching one later via
    /// `db.world.attach_gpu_mirror`, still available for a host that only
    /// knows its `GpuMirrorHandle` after construction).
    #[cfg(feature = "gpu")]
    pub fn new_with_gpu_mirror(mirror: GpuMirrorHandle) -> Self {
        let mut db = Self::new();
        db.world.attach_gpu_mirror(mirror);
        db
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
    /// registered subsystem's `simulate_a`/`simulate_b` hook.
    ///
    /// # Automatic World-mirror flush (issue #41: no manual steps, and no
    /// GPU-specific call the host has to know to make)
    ///
    /// If [`Self::world`] has an attached [`crate::gpu::GpuMirrorHandle`]
    /// (`World::attach_gpu_mirror`, called once at setup — the one wiring
    /// step C0 makes unavoidable, see the module doc), this call ALSO
    /// flushes it: every `#[gpu(mirror = DirtyTracked)]`/`#[gpu(mirror =
    /// Once)]` field marked dirty since the last call, plus the
    /// entity-liveness generation mirror. A host whose entire GPU-mirrored
    /// state lives on `World` (no cell-mirrored `SceneGpuStore` at all)
    /// gets fully automatic GPU sync from `step()` ALONE — one call, no
    /// GPU-typed argument, no second GPU-named method to remember. Combined
    /// with auto-registration on first use (`gpu::world_mirror`'s per-type
    /// dispatch function), neither registration nor flush ever needs a
    /// manual call for this path, regardless of whether [`Self::step_gpu`]
    /// is ever called at all.
    ///
    /// If the `gpu` feature is off, or no mirror is attached, this flush is
    /// skipped entirely — `step()` is byte-for-byte the CPU-only call it
    /// always was in that case.
    ///
    /// [`Self::step_gpu`] ALSO performs this same flush (see its doc) —
    /// calling both in one frame is harmless (the second flush is a
    /// near-no-op, since nothing new was marked dirty between them) and is
    /// exactly what a host that mutates `World` again inside a
    /// `harvest`/`boundary` subsystem hook wants: freshly dirtied data still
    /// reaches the GPU before `step_gpu` returns.
    pub fn step(&mut self) {
        let sim_a = self.driver.begin();
        self.subsystems.simulate_a(&mut self.world, &sim_a);
        let sim_b = sim_a.end();
        self.subsystems.simulate_b(&mut self.world, &sim_b);
        #[cfg(feature = "gpu")]
        self.flush_world_gpu_mirror();
    }

    /// GPU-phase step: `Harvest` → `Boundary` (retire → compact → sync),
    /// dispatching every registered subsystem's `harvest`/`boundary` hook.
    /// Takes the caller's real `SceneGpuStore`/`CellSlot`s (see module docs
    /// on C0) — only needed by a host that ALSO uses the cell-mirrored
    /// (`CellStorage`-based) residency path; a host whose GPU-mirrored state
    /// is entirely `World`-side needs only [`Self::step`] and never calls
    /// this at all. Runs its own `SimulateA`→`SimulateB` internally purely
    /// to reach a `HarvestPhase` witness — those sub-phases carry no
    /// mutation here (`world` is not touched by this method) so no
    /// subsystem `simulate_a`/`simulate_b` hook is invoked for it; call
    /// [`Self::step`] for that.
    ///
    /// Also flushes the World mirror, same as [`Self::step`] does — see
    /// that method's doc for why calling both in one frame is harmless
    /// (and sometimes exactly what you want). Returns the combined
    /// [`SyncStats`] of the cell-mirrored boundary sync and the World-mirror
    /// flush.
    ///
    /// The World-mirror flush here reaches the mirror's OWN `SceneGpuStore`
    /// (`mirror.store()`, an `Arc` clone reached through
    /// [`crate::World::gpu_mirror`]) — NOT the `store` parameter below. The
    /// two are necessarily different `SceneGpuStore` instances: `store`
    /// here needs `&mut` access (for the cell-mirrored register/write/sync
    /// path), while a `GpuMirrorHandle` needs `Arc`-shared access (so
    /// `World::insert`'s dispatch can reach it without exclusive ownership)
    /// — `Arc<T>` can never hand out `&mut T` once cloned, so no single
    /// `SceneGpuStore` can satisfy both callers at once. World-mirrored
    /// component fields and cell-mirrored scene state are therefore always
    /// two separate buffer pools when both are used through `SceneDb` — a
    /// real, structural split, not an oversight to paper over here.
    #[cfg(feature = "gpu")]
    pub fn step_gpu(&mut self, store: &mut SceneGpuStore, cells: &mut [CellSlot<'_>]) -> SyncStats {
        let sim_a = self.driver.begin();
        let sim_b = sim_a.end();
        let harvest = sim_b.end();
        self.subsystems.harvest(store, &harvest);
        let boundary = harvest.end();
        let (retired, _drained) = boundary.retire(store, cells);
        self.subsystems.boundary(&retired);
        let mut stats = retired.compact(store, cells).sync(store, cells);

        let world_stats = self.flush_world_gpu_mirror();
        stats.ranges += world_stats.ranges;
        stats.bytes += world_stats.bytes;
        stats
    }

    /// Shared implementation behind the automatic flush both [`Self::step`]
    /// and [`Self::step_gpu`] perform — see [`Self::step`]'s doc for the
    /// full rationale. `SyncStats::default()`-shaped (zero ranges/bytes) if
    /// no mirror is attached; never panics either way.
    #[cfg(feature = "gpu")]
    fn flush_world_gpu_mirror(&self) -> SyncStats {
        match self.world.gpu_mirror() {
            Some(mirror) => {
                let queue = mirror.queue();
                mirror.generations().flush(queue);
                mirror.store().flush_gpu_mirror(queue)
            }
            None => SyncStats { ranges: 0, bytes: 0 },
        }
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
