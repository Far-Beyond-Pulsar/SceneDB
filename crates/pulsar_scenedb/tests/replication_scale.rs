//! Scale integration tests for the replication module (HARDENING.md item 8).
//!
//! Run with: `cargo test -p pulsar_scenedb --test replication_scale`.

use pulsar_scenedb::replication::events_to_batch;
use pulsar_scenedb::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ── Component types ─────────────────────────────────────────────────────
//
// `Position` is always present; combined with one of 10 `Shard*` marker
// types and one of 5 `Group*` marker types, entities land in exactly
// 10 * 5 = 50 distinct archetypes.

macro_rules! marker_types {
    ($($name:ident),+ $(,)?) => {
        $( #[derive(Clone, Copy, Debug, Default)] struct $name; )+
    };
}

marker_types!(Shard0, Shard1, Shard2, Shard3, Shard4, Shard5, Shard6, Shard7, Shard8, Shard9);
marker_types!(Group0, Group1, Group2, Group3, Group4);

#[derive(Clone, Copy, Debug, Default)]
struct Position(f32, f32, f32);
// SAFETY: three `f32`s, no padding, no niches — trivially safe to
// zero-init/byte-reinterpret. Needed so `Position: Replicable` via the
// blanket `impl<T: Pod> Replicable for T` (see `replication.rs`).
unsafe impl Pod for Position {}

fn insert_shard(world: &mut World, e: Entity, idx: usize) {
    match idx {
        0 => { world.insert(e, Shard0); }
        1 => { world.insert(e, Shard1); }
        2 => { world.insert(e, Shard2); }
        3 => { world.insert(e, Shard3); }
        4 => { world.insert(e, Shard4); }
        5 => { world.insert(e, Shard5); }
        6 => { world.insert(e, Shard6); }
        7 => { world.insert(e, Shard7); }
        8 => { world.insert(e, Shard8); }
        9 => { world.insert(e, Shard9); }
        _ => unreachable!(),
    }
}

fn insert_group(world: &mut World, e: Entity, idx: usize) {
    match idx {
        0 => { world.insert(e, Group0); }
        1 => { world.insert(e, Group1); }
        2 => { world.insert(e, Group2); }
        3 => { world.insert(e, Group3); }
        4 => { world.insert(e, Group4); }
        _ => unreachable!(),
    }
}

const ENTITY_COUNT: usize = 10_000;
const ARCHETYPE_COUNT: usize = 50;

// ── 10,000 entities / 50 archetypes / 4-client relevance ───────────────

#[test]
fn ten_thousand_entities_fifty_archetypes_four_client_relevance() {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(ENTITY_COUNT);
    for i in 0..ENTITY_COUNT {
        let e = world.spawn();
        world.insert(e, Position(i as f32, 0.0, 0.0));
        // `i % 10` and `(i / 10) % 5` are independent — plain `i % 5` would
        // always equal `(i % 10) % 5`, collapsing the two axes down to only
        // 10 distinct combinations instead of the intended 50.
        insert_shard(&mut world, e, i % 10);
        insert_group(&mut world, e, (i / 10) % 5);
        entities.push(e);
    }

    // Intermediate archetypes from the insert-migration chain (empty ->
    // {Position} -> {Position, ShardX} -> {Position, ShardX, GroupY}) stay
    // registered but end up with zero live entities; only the 50 final
    // combinations should hold any.
    let non_empty = world.archetypes.iter().filter(|a| !a.entities.is_empty()).count();
    assert_eq!(non_empty, ARCHETYPE_COUNT, "expected exactly 50 archetypes holding live entities");

    let pos_cid = component_id::<Position>();
    let component_deltas = entities
        .iter()
        .map(|&e| ComponentDelta { entity: e, component_type: pos_cid, field_data: vec![vec![1u8]] })
        .collect();
    let delta = Delta {
        frame: 0,
        base_frame: 0,
        spawned: vec![],
        despawned: vec![],
        component_deltas,
        events: vec![],
    };

    let mut reg = ReplicationRegistry::new();
    let builder = reg.register::<Position>();
    reg.insert(builder.whole_field("pos", ReplicationEncoding::Pod, ReplicationCondition::Always));

    let authority = AuthorityTable::new();

    // 4 clients, each relevant to a disjoint quarter of the 10,000 entities.
    for client_idx in 0..4u64 {
        let mut set = RelevanceSet::new();
        for (i, &e) in entities.iter().enumerate() {
            if i % 4 == client_idx as usize {
                set.add(e);
            }
        }
        let view = set.filter(&delta, &authority, &reg, ClientId(client_idx));
        assert_eq!(view.component_deltas.len(), ENTITY_COUNT / 4);

        // Spot-check: an entity from a different quarter is excluded.
        let other_quarter_entity = entities[(client_idx as usize + 1) % 4];
        assert!(!view.component_deltas.iter().any(|cd| cd.entity == other_quarter_entity));
    }
}

// ── 1,000 events: ordering + direction enforcement ──────────────────────

#[test]
fn one_thousand_events_ordering_and_direction_enforcement() {
    let mut tracker = ChangeTracker::new();
    let world = World::new();
    let e = Entity::from_bits(0);
    let cid = component_id::<Position>();

    let mut reg = ReplicationRegistry::new();
    let builder = reg.register::<Position>();
    let builder = builder
        .event("to_server", ReplicationCondition::ClientToServer, EventChannel::ReliableOrdered)
        .event("to_client", ReplicationCondition::ServerToClient, EventChannel::ReliableOrdered)
        .event("multicast", ReplicationCondition::Multicast, EventChannel::Unreliable);
    reg.insert(builder);

    for i in 0..1000u32 {
        tracker.record_event(ReplicatedEvent {
            entity: e,
            component_type: cid,
            event_field: i % 3,
            payload: i.to_le_bytes().to_vec(),
            channel: if i % 3 == 2 { EventChannel::Unreliable } else { EventChannel::ReliableOrdered },
            target_client: if i % 3 == 1 { Some(ClientId(7)) } else { None },
        });
    }

    let delta = tracker.drain_with_world(&world);
    assert_eq!(delta.events.len(), 1000);

    // Ordering preserved exactly as recorded.
    for (i, ev) in delta.events.iter().enumerate() {
        assert_eq!(ev.payload, (i as u32).to_le_bytes().to_vec());
    }

    let view = DeltaView {
        spawned: &delta.spawned,
        despawned: &delta.despawned,
        component_deltas: delta.component_deltas.iter().collect(),
        events: delta.events.iter().collect(),
    };

    let count0 = (0..1000u32).filter(|i| i % 3 == 0).count();
    let count1 = (0..1000u32).filter(|i| i % 3 == 1).count();
    let count2 = (0..1000u32).filter(|i| i % 3 == 2).count();

    let assert_ordered_subsequence = |events: &[ReplicatedEvent]| {
        let mut last: i64 = -1;
        for ev in events {
            let bytes: [u8; 4] = ev.payload.clone().try_into().unwrap();
            let val = u32::from_le_bytes(bytes) as i64;
            assert!(val > last, "direction-filtered events must preserve relative order");
            last = val;
        }
    };

    // Client(1) -> Server(0): ClientToServer always passes; ServerToClient
    // needs sender==recipient (false here); Multicast needs sender!=recipient (true).
    let batch = events_to_batch(&view, 0, &reg, ClientId(1), ClientId(0)).unwrap();
    assert_eq!(batch.events.len(), count0 + count2);
    assert_ordered_subsequence(&batch.events);

    // Server -> Client(5), sender==recipient==5: ServerToClient passes;
    // Multicast blocked (sender==recipient); ClientToServer always passes.
    let batch = events_to_batch(&view, 0, &reg, ClientId(5), ClientId(5)).unwrap();
    assert_eq!(batch.events.len(), count0 + count1);
    assert_ordered_subsequence(&batch.events);

    // Sender(2) -> a different peer(3): ClientToServer always passes;
    // ServerToClient needs sender==recipient (false here, so blocked);
    // Multicast needs sender!=recipient (true here, so passes).
    let batch = events_to_batch(&view, 0, &reg, ClientId(2), ClientId(3)).unwrap();
    assert_eq!(batch.events.len(), count0 + count2);
    assert_ordered_subsequence(&batch.events);
}

// ── 64-frame reconciliation cycle: convergence under random inputs ─────

#[test]
fn sixty_four_frame_reconciliation_converges() {
    let mut rng = StdRng::seed_from_u64(0xB0BA);
    let mut reconciler = Reconciler::new();
    let mut world = World::new();
    let e = world.spawn();
    let cid = component_id::<Position>();

    // Acknowledgment lags the live frame by a fixed window; the pending
    // queue should never exceed that window, and only inputs newer than
    // the acknowledged base_frame should ever be replayed.
    const ACK_LAG: u64 = 5;

    for frame in 1..=64u64 {
        let value: u8 = rng.gen();
        reconciler.push_input(ClientInput {
            frame,
            entity: e,
            component: cid,
            field_data: vec![(0, vec![value])],
        });

        let base_frame = frame.saturating_sub(ACK_LAG);
        let server_delta = Delta {
            frame,
            base_frame,
            spawned: vec![],
            despawned: vec![],
            component_deltas: vec![],
            events: vec![],
        };

        let mut replayed = Vec::new();
        reconciler.reconcile(&server_delta, &mut world, |_, input| replayed.push(input.frame));

        assert!(
            reconciler.pending_inputs().len() as u64 <= ACK_LAG,
            "pending window must stay bounded by the ack lag (frame {frame})"
        );
        for r in &replayed {
            assert!(*r > base_frame, "only unacknowledged inputs may be replayed");
        }
    }

    // A final delta acknowledging every frame drains the queue to zero —
    // the convergence property this test exists to verify.
    let final_delta = Delta { frame: 65, base_frame: 64, spawned: vec![], despawned: vec![], component_deltas: vec![], events: vec![] };
    reconciler.reconcile(&final_delta, &mut world, |_, _| {});
    assert!(reconciler.pending_inputs().is_empty(), "reconciliation must converge to zero pending inputs");
}

// ── Snapshot capture/restore at scale, byte-exact ───────────────────────

#[test]
fn snapshot_capture_full_ten_thousand_entities_is_deterministic() {
    let mut world = World::new();
    for i in 0..ENTITY_COUNT {
        let e = world.spawn();
        world.insert(e, Position(i as f32, i as f32 * 2.0, i as f32 * 3.0));
    }

    let mut reg = ReplicationRegistry::new();
    let builder = reg.register::<Position>();
    reg.insert(builder.whole_field("pos", ReplicationEncoding::Pod, ReplicationCondition::Always));

    let snap1 = Snapshot::capture_full(&world, &reg, 1);
    let snap2 = Snapshot::capture_full(&world, &reg, 1);
    assert_eq!(snap1.entities.len(), ENTITY_COUNT);
    assert_eq!(snap1.entities.len(), snap2.entities.len());
    for (a, b) in snap1.entities.iter().zip(snap2.entities.iter()) {
        assert_eq!(a.entity, b.entity);
        assert_eq!(a.components, b.components, "capture must be byte-exact and repeatable");
    }
}

#[test]
fn snapshot_capture_restore_cells_ten_thousand_rows_byte_exact() {
    const CELLS: usize = 50;
    const ROWS_PER_CELL: usize = ENTITY_COUNT / CELLS;

    let mut cells: Vec<SpatialCell> = (0..CELLS)
        .map(|_| SpatialCell::with_transform(ROWS_PER_CELL as u32).unwrap())
        .collect();

    let mut rng = StdRng::seed_from_u64(0x5CA1E);
    for cell in &mut cells {
        for _ in 0..ROWS_PER_CELL {
            let h = cell.alloc(Aabb { min: [0.0; 3], max: [1.0; 3] }).unwrap();
            let row = cell.row_of(h).unwrap() as usize;
            cell.storage_mut().column_for_mut::<InstanceInfo>().unwrap()[row] =
                InstanceInfo { mesh_index: rng.gen(), flags: rng.gen() };
        }
    }

    let mut reg = ReplicationRegistry::new();
    let builder = reg.register::<InstanceInfo>();
    reg.insert(builder.whole_field("info", ReplicationEncoding::Pod, ReplicationCondition::Always));

    let snap = Snapshot::capture_cells(&cells, &reg, 1);
    assert_eq!(snap.cell_rows.len(), ENTITY_COUNT);

    let mut restored: Vec<SpatialCell> = (0..CELLS)
        .map(|_| SpatialCell::with_transform(ROWS_PER_CELL as u32).unwrap())
        .collect();
    assert_eq!(snap.restore_to_cells(&mut restored, &reg), Ok(()));

    let snap2 = Snapshot::capture_cells(&restored, &reg, 1);
    assert_eq!(snap2.cell_rows.len(), snap.cell_rows.len());
    for (a, b) in snap.cell_rows.iter().zip(snap2.cell_rows.iter()) {
        assert_eq!(a.cell_index, b.cell_index);
        assert_eq!(a.components, b.components, "restore must reproduce byte-exact component data");
    }
}
