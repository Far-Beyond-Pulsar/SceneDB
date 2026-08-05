//! Benchmarks for the replication module's data-path claims (HARDENING plan
//! Phase D): does `apply_with_scratch` actually help, how expensive is
//! `Snapshot` capture at scale, and what does the generic `Replicable` path
//! (`String`/`Vec<T>`) cost relative to the `Pod` fast path.
//!
//! Run with: `cargo bench -p pulsar_scenedb --bench replication_bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pulsar_scenedb::*;

#[derive(Clone, Copy, Default)]
struct Position {
    xyz: [f32; 3],
}
unsafe impl Pod for Position {}

fn registry_with_position() -> ReplicationRegistry {
    let mut reg = ReplicationRegistry::new();
    let builder = reg.register::<Position>();
    reg.insert(builder.whole_field("xyz", ReplicationEncoding::Pod, ReplicationCondition::Always));
    reg
}

fn world_with_positions(n: usize) -> (World, ReplicationRegistry, Vec<Entity>) {
    let mut world = World::new();
    let reg = registry_with_position();
    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        let e = world.spawn();
        world.insert(e, Position { xyz: [i as f32, 0.0, 0.0] });
        entities.push(e);
    }
    (world, reg, entities)
}

fn bench_delta_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_apply");
    for &n in &[1_000usize, 10_000] {
        let (mut world, reg, entities) = world_with_positions(n);
        let cid = component_id::<Position>();
        let component_deltas: Vec<ComponentDelta> = entities
            .iter()
            .map(|&e| {
                let mut bytes = Vec::new();
                Position { xyz: [1.0, 2.0, 3.0] }.xyz.replicate_encode(&mut bytes);
                ComponentDelta { entity: e, component_type: cid, field_data: vec![bytes] }
            })
            .collect();
        let delta = Delta {
            frame: 0, base_frame: 0,
            spawned: vec![], despawned: vec![],
            component_deltas,
            events: vec![],
        };

        group.bench_function(format!("apply_{n}"), |b| {
            b.iter(|| black_box(delta.apply(&mut world, &reg)))
        });

        let mut scratch = Vec::new();
        group.bench_function(format!("apply_with_scratch_{n}"), |b| {
            b.iter(|| black_box(delta.apply_with_scratch(&mut world, &reg, &mut scratch)))
        });
    }
    group.finish();
}

fn bench_snapshot_capture(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_capture");
    for &n in &[1_000usize, 10_000] {
        let (world, reg, _) = world_with_positions(n);
        group.bench_function(format!("capture_full_{n}"), |b| {
            b.iter(|| black_box(Snapshot::capture_full(&world, &reg, 0)))
        });
    }
    group.finish();
}

fn bench_replicable_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("replicable_encode_decode");

    let pos = Position { xyz: [1.0, 2.0, 3.0] };
    group.bench_function("pod_f32x3_encode", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            black_box(&pos).xyz.replicate_encode(&mut buf);
            black_box(buf)
        })
    });
    let pos_bytes = {
        let mut buf = Vec::new();
        pos.xyz.replicate_encode(&mut buf);
        buf
    };
    group.bench_function("pod_f32x3_decode", |b| {
        b.iter(|| black_box(<[f32; 3]>::replicate_decode(black_box(&pos_bytes))))
    });

    let name = "a moderately-sized entity name".to_string();
    group.bench_function("string_encode", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            black_box(&name).replicate_encode(&mut buf);
            black_box(buf)
        })
    });
    let name_bytes = {
        let mut buf = Vec::new();
        name.replicate_encode(&mut buf);
        buf
    };
    group.bench_function("string_decode", |b| {
        b.iter(|| black_box(String::replicate_decode(black_box(&name_bytes))))
    });

    let tags: Vec<u32> = (0..16).collect();
    group.bench_function("vec_u32x16_encode", |b| {
        b.iter(|| {
            let mut buf = Vec::new();
            black_box(&tags).replicate_encode(&mut buf);
            black_box(buf)
        })
    });
    let tags_bytes = {
        let mut buf = Vec::new();
        tags.replicate_encode(&mut buf);
        buf
    };
    group.bench_function("vec_u32x16_decode", |b| {
        b.iter(|| black_box(Vec::<u32>::replicate_decode(black_box(&tags_bytes))))
    });

    group.finish();
}

criterion_group!(benches, bench_delta_apply, bench_snapshot_capture, bench_replicable_encode_decode);
criterion_main!(benches);
