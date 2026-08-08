//! Integration test for the subsystem-registry design (see `src/subsystem.rs`,
//! `src/scene_db.rs`, `pulsar_scenedb_derive::scenedb_subsystem`):
//!
//! - `PhysicsSubsystem::apply_impulse` is registered into Pulsar's central
//!   reflection database (`pulsar_reflection::DYN_METHOD_REGISTRY`) at link
//!   time by `#[scenedb_subsystem(name = "physics")]` /
//!   `#[subsystem_method]` — no hand-written registration code below.
//! - Dynamic invocation by string names (`"physics"`, `"apply_impulse"`)
//!   is verified through `SceneDb::dispatch`.
//! - Zero-overhead static access is verified through
//!   `SceneDb::subsystem_mut::<PhysicsSubsystem>()`.

use pulsar_scenedb::gpu::{HarvestPhase, RetiredPhase, SceneGpuStore, SimulateA, SimulateB};
use pulsar_scenedb::{Handle, SceneDb, Subsystem, World};
use pulsar_scenedb_derive::{scenedb_subsystem, subsystem_method};
use std::any::Any;
use std::collections::HashMap;

#[derive(Default)]
struct PhysicsSubsystem {
    // Keyed by `Handle::index()` (a plain `u64`) rather than `Handle`
    // itself: reflection-visible method parameters must implement
    // `pulsar_reflection::Reflectable`, which `Handle` does not (yet —
    // that's a real gap worth its own registration decision, tracked
    // separately rather than folded into this test). `u64`/`[f32; 3]`
    // are already registered primitives (`prims/core`), so the dispatch
    // path below is exercised for real, just with entity identity
    // narrowed to the slot index for this test.
    impulses: HashMap<u64, [f32; 3]>,
    simulate_a_calls: u64,
}

#[scenedb_subsystem(name = "physics")]
impl PhysicsSubsystem {
    /// Reflection-visible: registered into `DYN_METHOD_REGISTRY` under
    /// `("physics", "apply_impulse")` by the `#[scenedb_subsystem]` +
    /// `#[subsystem_method]` macros above, with no manual registration.
    #[subsystem_method]
    pub fn apply_impulse(&mut self, entity_index: u64, impulse: [f32; 3]) {
        self.impulses.insert(entity_index, impulse);
    }

    /// Not reflection-visible (no `#[subsystem_method]`) — a plain method
    /// the phase hooks below call directly, same as any other Rust code.
    fn total_impulse_magnitude(&self) -> f32 {
        self.impulses
            .values()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .sum()
    }
}

impl Subsystem for PhysicsSubsystem {
    fn name(&self) -> &'static str {
        "physics"
    }

    fn simulate_a(&mut self, _world: &mut World, _witness: &SimulateA) {
        self.simulate_a_calls += 1;
    }

    fn simulate_b(&mut self, _world: &mut World, _witness: &SimulateB) {}
    fn harvest(&mut self, _store: &SceneGpuStore, _phase: &HarvestPhase) {}
    fn boundary(&mut self, _phase: &RetiredPhase) {}

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[test]
fn dynamic_dispatch_reaches_the_real_subsystem_by_string_name() {
    let mut db = SceneDb::new();
    db.register_subsystem(PhysicsSubsystem::default());

    let entity = Handle::new(1, 1);

    db.dispatch(
        "physics",
        "apply_impulse",
        vec![Box::new(entity.index() as u64), Box::new([1.0f32, 2.0, 3.0])],
    )
    .expect("dispatch succeeds");

    let physics = db
        .subsystem::<PhysicsSubsystem>()
        .expect("registered under its Rust type");
    assert_eq!(physics.impulses.get(&(entity.index() as u64)), Some(&[1.0, 2.0, 3.0]));

    // Dispatching a method that was never marked #[subsystem_method] is a
    // reflection-layer miss, not a panic -- it's simply not in the DB.
    assert!(db
        .dispatch("physics", "total_impulse_magnitude", vec![])
        .is_err());

    // Dispatching against an unregistered subsystem name is likewise a
    // clean error, not a panic.
    assert!(db.dispatch("audio", "play", vec![]).is_err());
}

#[test]
fn static_path_is_a_zero_overhead_typed_borrow() {
    let mut db = SceneDb::new();
    db.register_subsystem(PhysicsSubsystem::default());

    // The static path never touches the reflection database at all --
    // it's a plain TypeId-keyed downcast (see SubsystemRegistry::get_mut).
    db.step();
    db.step();

    let physics = db.subsystem_mut::<PhysicsSubsystem>().expect("registered");
    assert_eq!(physics.simulate_a_calls, 2);

    physics
        .impulses
        .insert(Handle::new(2, 1).index() as u64, [0.0, 0.0, 1.0]);
    assert!((physics.total_impulse_magnitude() - 1.0).abs() < f32::EPSILON);
}
