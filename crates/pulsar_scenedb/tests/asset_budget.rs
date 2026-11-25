//! A frame thread's view of asset streaming: a budgeted pump that leaves work
//! for later calls and reports it, loads started nearest first within an
//! in-flight cap, and coarse-first tiers that are bound as they arrive.
//!
//! The registries are process-wide and tests run in parallel, so each test
//! declares its own asset kind and sources.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pulsar_scenedb::{
    register_binding, register_loader, register_tiered_loader, AssetBudget, AssetId, AssetPayload,
    AssetView, SceneStore, TierLoad, World,
};

/// Names an asset; `rank` is what the priority function reads.
#[derive(Clone, Copy, Debug, SceneStore)]
struct Thing {
    asset: AssetId,
    rank: u32,
    /// The tier last bound, plus one (0 = nothing bound).
    bound_tier: u32,
    complete: u32,
    binds: u32,
}

impl Thing {
    fn new(asset: AssetId, rank: u32) -> Self {
        Self {
            asset,
            rank,
            bound_tier: 0,
            complete: 0,
            binds: 0,
        }
    }
}

fn bind_thing(thing: &mut Thing, view: &AssetView) {
    thing.bound_tier = u32::from(view.tier) + 1;
    thing.complete = u32::from(view.complete);
    thing.binds += 1;
}

fn settle(world: &mut World) {
    assert!(
        world.resolve_assets_blocking(Duration::from_secs(10)),
        "assets settle"
    );
}

#[test]
fn a_budgeted_pump_binds_a_few_assets_a_call_and_reports_what_is_left() {
    register_loader("b-budget", |_| Ok(Arc::new(1u32) as AssetPayload));
    register_binding::<Thing>("b-budget", bind_thing);
    let mut world = World::new();
    let entities: Vec<_> = (0..20)
        .map(|index| {
            let id = AssetId::declare("b-budget", &format!("budget/{index}"));
            let entity = world.spawn();
            world.insert(entity, Thing::new(id, index));
            entity
        })
        .collect();

    let budget = AssetBudget {
        max_assets: 3,
        time: None,
    };
    let (mut bound, mut calls) = (0, 0);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let report = world.pump_assets_within(budget);
        assert!(report.bound <= 3, "{report:?}");
        assert!(report.completed <= 3, "{report:?}");
        bound += report.bound;
        calls += 1;
        if report.pending == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "never settled: {report:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(bound, 20);
    assert!(calls >= 7, "20 assets at 3 a call took {calls} calls");
    for entity in entities {
        assert_eq!(world.get::<Thing>(entity).unwrap().binds, 1);
    }
    assert!(!world.asset_work_pending());
}

#[test]
fn a_zero_budget_still_makes_progress() {
    register_loader("b-zero", |_| Ok(Arc::new(1u32) as AssetPayload));
    register_binding::<Thing>("b-zero", bind_thing);
    let mut world = World::new();
    let id = AssetId::declare("b-zero", "zero/a");
    let entity = world.spawn();
    world.insert(entity, Thing::new(id, 0));
    let budget = AssetBudget {
        max_assets: 0,
        time: Some(Duration::ZERO),
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while world.get::<Thing>(entity).unwrap().binds == 0 {
        world.pump_assets_within(budget);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn loads_start_nearest_first_within_the_in_flight_cap() {
    static ORDER: Mutex<Vec<String>> = Mutex::new(Vec::new());
    register_loader("b-priority", |source| {
        ORDER.lock().unwrap().push(source.to_owned());
        Ok(Arc::new(1u32) as AssetPayload)
    });
    register_binding::<Thing>("b-priority", bind_thing);
    let mut world = World::new();
    world.set_max_loads_in_flight(1);
    world.set_asset_priority(Some(|world: &World, entity| {
        world.get::<Thing>(entity).map(|thing| thing.rank as f32)
    }));
    // Spawned far to near, so a first-come order would be the reverse.
    for rank in (0..12u32).rev() {
        let id = AssetId::declare("b-priority", &format!("priority/{rank:02}"));
        let entity = world.spawn();
        world.insert(entity, Thing::new(id, rank));
    }
    settle(&mut world);
    let order = ORDER.lock().unwrap().clone();
    let expected: Vec<String> = (0..12).map(|rank| format!("priority/{rank:02}")).collect();
    assert_eq!(order, expected, "one load at a time, nearest first");
}

#[test]
fn a_tiered_asset_binds_coarse_first_then_upgrades() {
    static FINE_OPEN: AtomicBool = AtomicBool::new(false);
    register_tiered_loader("b-tiers", |_, tier| {
        if tier >= 1 {
            while !FINE_OPEN.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(TierLoad {
            payload: Arc::new(u32::from(tier)) as AssetPayload,
            last: tier == 1,
        })
    });
    register_binding::<Thing>("b-tiers", bind_thing);
    let mut world = World::new();
    let id = AssetId::declare("b-tiers", "tiers/a");
    let entity = world.spawn();
    world.insert(entity, Thing::new(id, 0));

    // The coarse tier arrives and is bound while the fine one is held back.
    let deadline = Instant::now() + Duration::from_secs(10);
    while world.get::<Thing>(entity).unwrap().binds == 0 {
        world.pump_assets();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let thing = *world.get::<Thing>(entity).unwrap();
    assert_eq!((thing.bound_tier, thing.complete), (1, 0), "coarse, not complete");
    assert_eq!(world.asset_version(id), 1);
    assert!(world.asset_work_pending(), "the upgrade is outstanding");

    FINE_OPEN.store(true, Ordering::SeqCst);
    settle(&mut world);
    let thing = *world.get::<Thing>(entity).unwrap();
    assert_eq!((thing.bound_tier, thing.complete), (2, 1), "fine and complete");
    assert_eq!(thing.binds, 2);
    assert_eq!(world.asset_version(id), 2);
    assert!(!world.asset_work_pending());
}

#[test]
fn finer_tiers_wait_for_every_first_tier() {
    static ORDER: Mutex<Vec<(String, u8)>> = Mutex::new(Vec::new());
    register_tiered_loader("b-tier-order", |source, tier| {
        ORDER.lock().unwrap().push((source.to_owned(), tier));
        Ok(TierLoad {
            payload: Arc::new(0u32) as AssetPayload,
            last: tier == 1,
        })
    });
    register_binding::<Thing>("b-tier-order", bind_thing);
    let mut world = World::new();
    world.set_max_loads_in_flight(1);
    for index in 0..4 {
        let id = AssetId::declare("b-tier-order", &format!("order/{index}"));
        let entity = world.spawn();
        world.insert(entity, Thing::new(id, index));
    }
    settle(&mut world);
    let order = ORDER.lock().unwrap().clone();
    assert_eq!(order.len(), 8);
    assert!(
        order[..4].iter().all(|(_, tier)| *tier == 0) && order[4..].iter().all(|(_, tier)| *tier == 1),
        "every coarse tier before any fine tier: {order:?}"
    );
}

#[test]
fn an_unreferenced_asset_is_not_upgraded() {
    static FINE_LOADS: Mutex<u32> = Mutex::new(0);
    register_tiered_loader("b-no-upgrade", |_, tier| {
        if tier >= 1 {
            *FINE_LOADS.lock().unwrap() += 1;
        }
        Ok(TierLoad {
            payload: Arc::new(0u32) as AssetPayload,
            last: false,
        })
    });
    register_binding::<Thing>("b-no-upgrade", bind_thing);
    let mut world = World::new();
    world.set_max_loads_in_flight(1);
    let id = AssetId::declare("b-no-upgrade", "no-upgrade/a");
    let entity = world.spawn();
    world.insert(entity, Thing::new(id, 0));
    // Start the load, drop the only reference, then let the coarse tier land.
    world.pump_assets();
    world.despawn(entity);
    settle(&mut world);
    assert_eq!(*FINE_LOADS.lock().unwrap(), 0);
}

/// Loads that finish faster than the frame binds them must not pile up: what
/// loaded and is not bound yet counts against the in-flight cap, so the
/// loaders wait for the frame and what they cooked is still hot when it binds.
#[test]
fn loads_wait_for_the_frame_to_bind_what_already_loaded() {
    static LOADED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    register_loader("b-backlog", |_| {
        LOADED.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(1u32) as AssetPayload)
    });
    register_binding::<Thing>("b-backlog", bind_thing);
    let mut world = World::new();
    world.set_max_loads_in_flight(2);
    for index in 0..30 {
        let id = AssetId::declare("b-backlog", &format!("backlog/{index}"));
        let entity = world.spawn();
        world.insert(entity, Thing::new(id, index));
    }
    let budget = AssetBudget {
        max_assets: 1,
        time: None,
    };
    let (mut bound, mut worst_backlog) = (0usize, 0usize);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        // Let any started load finish, as a slow frame would.
        std::thread::sleep(Duration::from_millis(2));
        let report = world.pump_assets_within(budget);
        bound += report.bound;
        worst_backlog = worst_backlog.max(LOADED.load(Ordering::SeqCst).saturating_sub(bound));
        if report.pending == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "never settled: {report:?}");
    }
    assert_eq!(bound, 30);
    assert!(
        worst_backlog <= 3,
        "{worst_backlog} loaded assets waited unbound at once; the cap is 2"
    );
}

/// A bind that costs milliseconds must not be repeated a chunk at a time
/// without looking at the clock: once the pump knows what a bind costs it
/// binds only as many as the time left holds.
#[test]
fn a_time_budget_limits_binds_by_what_they_cost() {
    register_loader("b-cost", |_| Ok(Arc::new(1u32) as AssetPayload));
    register_binding::<Thing>("b-cost", |thing, view| {
        std::thread::sleep(Duration::from_millis(6));
        bind_thing(thing, view);
    });
    let mut world = World::new();
    for index in 0..12 {
        let id = AssetId::declare("b-cost", &format!("cost/{index}"));
        let entity = world.spawn();
        world.insert(entity, Thing::new(id, index));
    }
    let budget = AssetBudget {
        max_assets: 8,
        time: Some(Duration::from_millis(8)),
    };
    let mut per_call = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        std::thread::sleep(Duration::from_millis(3));
        let report = world.pump_assets_within(budget);
        if report.bound > 0 {
            per_call.push(report.bound);
        }
        if report.pending == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "never settled: {report:?}");
    }
    assert_eq!(per_call.iter().sum::<usize>(), 12);
    // Once the cost is known (after the first calls) a call binds at most two
    // 6 ms assets inside the 8 ms budget, never the 8 the count would allow.
    assert!(
        per_call.iter().skip(2).all(|&n| n <= 2),
        "binds per call: {per_call:?}"
    );
}

/// One asset shared by thousands of entities (a mesh used by every chair in a
/// level) is not one indivisible step: a pump that runs out of time binds as
/// many as it reached, and the next ones the rest.
#[test]
fn one_asset_shared_by_many_entities_is_bound_over_several_pumps() {
    register_loader("b-shared", |_| Ok(Arc::new(1u32) as AssetPayload));
    register_binding::<Thing>("b-shared", |thing, view| {
        // A bind that costs something, so the clock is reached.
        std::thread::sleep(Duration::from_micros(40));
        bind_thing(thing, view);
    });
    let mut world = World::new();
    let id = AssetId::declare("b-shared", "shared/chair");
    let entities: Vec<_> = (0..2000)
        .map(|rank| {
            let entity = world.spawn();
            world.insert(entity, Thing::new(id, rank));
            entity
        })
        .collect();
    let budget = AssetBudget {
        max_assets: 8,
        time: Some(Duration::from_millis(5)),
    };
    let (mut pumps, mut largest) = (0, 0);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let report = world.pump_assets_within(budget);
        pumps += 1;
        largest = largest.max(report.bound);
        if report.pending == 0 && !world.asset_work_pending() {
            break;
        }
        assert!(Instant::now() < deadline, "never settled: {report:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(pumps >= 4, "2000 binds of 40 us at 5 ms a pump took {pumps} pumps");
    assert!(largest < 1500, "a pump bound {largest} of them");
    for entity in entities {
        assert_eq!(world.get::<Thing>(entity).unwrap().binds, 1, "bound exactly once");
    }
}
