//! Head-to-head criterion benchmark: `pulsar_scenedb::World` vs
//! `bevy_ecs::World`, on matched scenarios (spawn, archetype migration,
//! query iteration). Single-threaded on both sides -- neither `World` is
//! driven through any parallel scheduler/executor here, just raw,
//! synchronous `World`/`Query` operations called directly inside each
//! `b.iter()` closure, same as `ecs_bench.rs`'s own methodology.
//!
//! Run with:
//!   cargo bench --bench vs_bevy_ecs
//!
//! Read the output as `pulsar_scenedb/<case>` vs `bevy_ecs/<case>` pairs
//! within each criterion group -- criterion reports each independently;
//! compare the printed `time:`/`thrpt:` lines directly.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// â”€â”€ Shared component shapes, one definition per side (same fields/sizes) â”€â”€

mod psdb_components {
    #[derive(Clone, Copy)]
    pub struct Pos(pub f32, pub f32, pub f32);
    #[derive(Clone, Copy)]
    pub struct Vel(pub f32, pub f32, pub f32);
    #[derive(Clone, Copy)]
    pub struct Health(pub u32);
    #[derive(Clone, Copy)]
    pub struct Tag(pub u32);
}

mod bevy_components {
    use bevy_ecs::component::Component;
    #[derive(Component, Clone, Copy)]
    pub struct Pos(pub f32, pub f32, pub f32);
    #[derive(Component, Clone, Copy)]
    pub struct Vel(pub f32, pub f32, pub f32);
    #[derive(Component, Clone, Copy)]
    pub struct Health(pub u32);
    #[derive(Component, Clone, Copy)]
    pub struct Tag(pub u32);
}

// â”€â”€ spawn: N entities, each with 4 components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn spawn_4_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_bevy/spawn_4_components");
    for &n in &[1_000u64, 10_000] {
        group.throughput(Throughput::Elements(n));

        group.bench_with_input(BenchmarkId::new("pulsar_scenedb", n), &n, |b, &n| {
            use psdb_components::*;
            use pulsar_scenedb::World;
            b.iter(|| {
                let mut world = World::new();
                world.reserve_entities(n as u32);
                for _ in 0..n {
                    let e = world.spawn();
                    world.insert(e, Pos(1.0, 2.0, 3.0));
                    world.insert(e, Vel(0.0, 0.0, 0.0));
                    world.insert(e, Health(100));
                    world.insert(e, Tag(0));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("bevy_ecs", n), &n, |b, &n| {
            use bevy_components::*;
            use bevy_ecs::world::World;
            b.iter(|| {
                let mut world = World::new();
                for _ in 0..n {
                    world.spawn((Pos(1.0, 2.0, 3.0), Vel(0.0, 0.0, 0.0), Health(100), Tag(0)));
                }
            });
        });
    }
    group.finish();
}

// â”€â”€ archetype_migration: N entities, each migrated empty -> {Pos} ->
// {Pos,Health} -> then Health removed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn archetype_migration(c: &mut Criterion) {
    let n: u64 = 10_000;
    let mut group = c.benchmark_group("vs_bevy/archetype_migration");
    group.throughput(Throughput::Elements(n));

    group.bench_function("pulsar_scenedb", |b| {
        use psdb_components::*;
        use pulsar_scenedb::World;
        b.iter(|| {
            let mut world = World::new();
            let entities: Vec<_> = (0..n)
                .map(|_| {
                    let e = world.spawn();
                    world.insert(e, Pos(1.0, 2.0, 3.0));
                    world.insert(e, Health(100));
                    e
                })
                .collect();
            for &e in &entities {
                world.remove::<Health>(e);
            }
        });
    });

    group.bench_function("bevy_ecs", |b| {
        use bevy_components::*;
        use bevy_ecs::world::World;
        b.iter(|| {
            let mut world = World::new();
            let entities: Vec<_> = (0..n)
                .map(|_| {
                    let mut e = world.spawn(Pos(1.0, 2.0, 3.0));
                    e.insert(Health(100));
                    e.id()
                })
                .collect();
            for &e in &entities {
                world.entity_mut(e).remove::<Health>();
            }
        });
    });

    group.finish();
}

// â”€â”€ query: iterate N entities matching a 2-component and an 8-field-total
// pattern â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn query_two_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_bevy/query_2_components");
    for &n in &[1_000u64, 10_000, 50_000] {
        group.throughput(Throughput::Elements(n));

        group.bench_with_input(BenchmarkId::new("pulsar_scenedb", n), &n, |b, &n| {
            use psdb_components::*;
            use pulsar_scenedb::World;
            let mut world = World::new();
            world.reserve_entities(n as u32);
            for _ in 0..n {
                let e = world.spawn();
                world.insert(e, Pos(1.0, 2.0, 3.0));
                world.insert(e, Vel(1.0, 1.0, 1.0));
            }
            b.iter(|| {
                let mut sum = 0.0f32;
                for (_e, (pos, vel)) in world.query::<(&Pos, &Vel)>() {
                    sum += pos.0 + vel.0;
                }
                std::hint::black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("bevy_ecs", n), &n, |b, &n| {
            use bevy_components::*;
            use bevy_ecs::world::World;
            let mut world = World::new();
            for _ in 0..n {
                world.spawn((Pos(1.0, 2.0, 3.0), Vel(1.0, 1.0, 1.0)));
            }
            let mut query = world.query::<(&Pos, &Vel)>();
            b.iter(|| {
                let mut sum = 0.0f32;
                for (pos, vel) in query.iter(&world) {
                    sum += pos.0 + vel.0;
                }
                std::hint::black_box(sum);
            });
        });
    }
    group.finish();
}

fn query_four_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_bevy/query_4_components");
    let n: u64 = 10_000;
    group.throughput(Throughput::Elements(n));

    group.bench_function("pulsar_scenedb", |b| {
        use psdb_components::*;
        use pulsar_scenedb::World;
        let mut world = World::new();
        world.reserve_entities(n as u32);
        for _ in 0..n {
            let e = world.spawn();
            world.insert(e, Pos(1.0, 2.0, 3.0));
            world.insert(e, Vel(1.0, 1.0, 1.0));
            world.insert(e, Health(100));
            world.insert(e, Tag(0));
        }
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_e, (pos, vel, hp, tag)) in world.query::<(&Pos, &Vel, &Health, &Tag)>() {
                sum += pos.0 + vel.0 + hp.0 as f32 + tag.0 as f32;
            }
            std::hint::black_box(sum);
        });
    });

    group.bench_function("bevy_ecs", |b| {
        use bevy_components::*;
        use bevy_ecs::world::World;
        let mut world = World::new();
        for _ in 0..n {
            world.spawn((Pos(1.0, 2.0, 3.0), Vel(1.0, 1.0, 1.0), Health(100), Tag(0)));
        }
        let mut query = world.query::<(&Pos, &Vel, &Health, &Tag)>();
        b.iter(|| {
            let mut sum = 0.0f32;
            for (pos, vel, hp, tag) in query.iter(&world) {
                sum += pos.0 + vel.0 + hp.0 as f32 + tag.0 as f32;
            }
            std::hint::black_box(sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    spawn_4_components,
    archetype_migration,
    query_two_components,
    query_four_components
);
criterion_main!(benches);
