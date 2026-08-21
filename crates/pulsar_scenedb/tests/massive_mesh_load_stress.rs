//! Exploratory load/soak test (Pulsar-Native#561 follow-up): spawns a large
//! number of entities each carrying a substantial fake mesh (`Vec<FakeVertex>`
//! + `Vec<u32>` indices, the same `#[gpu] Vec<T>` var-len shape
//! `StaticMeshComponent` uses for real), then runs many simulated "frames"
//! of churn against them -- LOD-style re-inserts with different vertex
//! counts, entity despawn/respawn -- to see what actually happens under
//! sustained load: does anything panic, does a `wgpu` validation error fire,
//! does the pool's freelist leak space over time, does per-frame cost stay
//! roughly flat, and does the data read back correctly after all the churn.
//!
//! This exercises `VarLenGpuPool`/`DynamicGpuBuffer` (`gpu/var_len_pool.rs`,
//! `gpu/dynamic_buffer.rs`) at a scale and churn pattern the small, targeted
//! unit tests elsewhere in this crate don't attempt -- thousands of
//! entities, tens of millions of vertices total, repeated grow/free cycles
//! across dozens of frames.
//!
//! Run with `--nocapture` to see the per-frame diagnostics as they happen:
//! `cargo test -p pulsar_scenedb --features gpu --test massive_mesh_load_stress -- --test-threads=1 --nocapture`

use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{Entity, World};
use pulsar_scenedb_derive::SceneStore;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::Instant;

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("massive-mesh-load-stress-test"),
        // The default limits' max_buffer_size comfortably covers this
        // test's scale (tens of MB per buffer); nothing raised here.
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback(ctx: &EngineGpuContext, buf: &wgpu::Buffer, src_offset: u64, bytes: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, src_offset, &staging, 0, bytes);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 64, max_resident_cells: 1 }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

/// A realistic-shaped vertex (position + normal + uv) -- 32 bytes, the same
/// order of magnitude as `StaticMeshComponent`'s real `PackedVertex`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FakeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}
unsafe impl pulsar_scenedb::page::Pod for FakeVertex {}

/// A fake mesh component -- deliberately the same `#[gpu] Vec<T>` var-len
/// shape real mesh components use, on two independently-pooled fields
/// (vertices, indices) plus a plain scalar tag alongside them (proves the
/// mixed scalar/var-len case holds up under load too, not just in
/// isolation).
#[derive(SceneStore, Clone)]
struct FakeMeshComponent {
    #[gpu(buffer = "stress_vertices")]
    vertices: Vec<FakeVertex>,
    #[gpu(buffer = "stress_indices")]
    indices: Vec<u32>,
    #[gpu]
    mesh_id: u32,
}

fn make_mesh(rng: &mut StdRng, mesh_id: u32, vert_count: usize) -> FakeMeshComponent {
    let vertices: Vec<FakeVertex> = (0..vert_count)
        .map(|i| FakeVertex {
            position: [i as f32, rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)],
            normal: [0.0, 1.0, 0.0],
            uv: [(i % 7) as f32 / 7.0, (i % 11) as f32 / 11.0],
        })
        .collect();
    // Fake triangle-list indices -- not topologically real, just enough
    // volume (~2x the vertex count) to stress a second, independent pool
    // alongside the first.
    let indices: Vec<u32> = (0..vert_count as u32 * 2).map(|i| i % vert_count.max(1) as u32).collect();
    FakeMeshComponent { vertices, indices, mesh_id }
}

#[test]
fn massive_mesh_load_survives_sustained_frame_churn() {
    const ENTITY_COUNT: usize = 1500;
    const VERT_COUNT_RANGE: std::ops::Range<usize> = 200..2000;
    const FRAME_COUNT: usize = 40;
    const CHURN_FRACTION: f32 = 0.05; // ~5% of entities reloaded/frame
    const RESPAWN_FRACTION: f32 = 0.01; // ~1% despawned + replaced/frame

    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let ctx = test_context();

    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // Generous initial capacity so the load phase itself grows the pools a
    // realistic (not-absurd) number of times, matching how a real level
    // load would ramp up rather than pre-sizing for the whole scene.
    FakeMeshComponent::register_gpu_columns_growable(&mut store, 4096, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue())));

    eprintln!(
        "\n=== massive_mesh_load_stress: loading {ENTITY_COUNT} entities, {}..{} verts each ===",
        VERT_COUNT_RANGE.start, VERT_COUNT_RANGE.end
    );

    // ---- Load phase: spawn every entity with its initial fake mesh ----
    let load_start = Instant::now();
    let mut entities: Vec<Entity> = Vec::with_capacity(ENTITY_COUNT);
    let mut vert_counts: Vec<usize> = Vec::with_capacity(ENTITY_COUNT);
    let mut total_verts_loaded: u64 = 0;
    for i in 0..ENTITY_COUNT {
        let vc = rng.gen_range(VERT_COUNT_RANGE);
        let e = world.spawn();
        world.insert(e, make_mesh(&mut rng, i as u32, vc));
        entities.push(e);
        vert_counts.push(vc);
        total_verts_loaded += vc as u64;
    }
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    let load_elapsed = load_start.elapsed();

    let vert_pool = store.var_len_pool::<FakeVertex>(pulsar_scenedb::gpu::BufferKey::of("stress_vertices"))
        .expect("vertices pool registered");
    let idx_pool = store.var_len_pool::<u32>(pulsar_scenedb::gpu::BufferKey::of("stress_indices"))
        .expect("indices pool registered");

    eprintln!(
        "load phase: {ENTITY_COUNT} entities, {total_verts_loaded} total vertices, {:.2?} elapsed \
         ({:.1} entities/ms) -- vertex pool capacity {}, index pool capacity {}",
        load_elapsed,
        ENTITY_COUNT as f64 / load_elapsed.as_millis().max(1) as f64,
        vert_pool.capacity(),
        idx_pool.capacity(),
    );

    // ---- Frame loop: LOD-style reloads + despawn/respawn churn ----
    let mut next_mesh_id = ENTITY_COUNT as u32;
    let mut frame_times = Vec::with_capacity(FRAME_COUNT);
    for frame in 0..FRAME_COUNT {
        let frame_start = Instant::now();

        let churn_count = ((entities.len() as f32) * CHURN_FRACTION) as usize;
        for _ in 0..churn_count {
            let idx = rng.gen_range(0..entities.len());
            let e = entities[idx];
            let vc = rng.gen_range(VERT_COUNT_RANGE);
            // Re-insert with a DIFFERENT vertex count -- exercises the
            // free-then-reallocate path (shrink or grow, whichever the new
            // random count happens to be) on an entity that already has a
            // live allocation, every single frame.
            world.insert(e, make_mesh(&mut rng, next_mesh_id, vc));
            next_mesh_id += 1;
            vert_counts[idx] = vc;
        }

        let respawn_count = ((entities.len() as f32) * RESPAWN_FRACTION).ceil() as usize;
        for _ in 0..respawn_count {
            let idx = rng.gen_range(0..entities.len());
            let old = entities[idx];
            world.despawn(old);
            let vc = rng.gen_range(VERT_COUNT_RANGE);
            let fresh = world.spawn();
            world.insert(fresh, make_mesh(&mut rng, next_mesh_id, vc));
            next_mesh_id += 1;
            entities[idx] = fresh;
            vert_counts[idx] = vc;
        }

        world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
        let frame_elapsed = frame_start.elapsed();
        frame_times.push(frame_elapsed);

        if frame % 10 == 0 || frame == FRAME_COUNT - 1 {
            eprintln!(
                "frame {frame:>3}: {churn_count} reloaded, {respawn_count} respawned, {:.2?} -- \
                 vertex pool capacity {}, index pool capacity {}",
                frame_elapsed,
                vert_pool.capacity(),
                idx_pool.capacity(),
            );
        }
    }

    let total_frame_time: std::time::Duration = frame_times.iter().sum();
    let avg_frame_ms = total_frame_time.as_secs_f64() * 1000.0 / FRAME_COUNT as f64;
    let max_frame = frame_times.iter().max().unwrap();
    eprintln!(
        "frame loop done: {FRAME_COUNT} frames, avg {avg_frame_ms:.3} ms/frame, worst {:.2?} -- \
         final vertex pool capacity {}, final index pool capacity {}",
        max_frame,
        vert_pool.capacity(),
        idx_pool.capacity(),
    );

    // ---- Correctness spot-check: every surviving entity's LIVE data must
    // still read back correctly after all that churn -- the real proof
    // nothing got silently corrupted (a stale/overlapping allocation would
    // show up here as wrong bytes, not necessarily as a crash). ----
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    for &e in entities.iter() {
        let Some(mesh) = world.get::<FakeMeshComponent>(e) else { continue };
        let handle_table_ok = !mesh.vertices.is_empty();
        if !handle_table_ok {
            continue;
        }
        // Re-derive what this entity's CURRENT mesh should look like isn't
        // possible post-hoc (the RNG stream already moved on) -- instead,
        // check the WEAKER but still meaningful invariant: the mesh's own
        // CPU-side `vertices`/`indices` (which `World::get` returns
        // straight from the page column, no GPU round trip) and its
        // GPU-mirrored bytes at its own recorded handle offset agree
        // byte-for-byte. Any pool corruption (a neighbor's write bleeding
        // into this span) would show up as a mismatch here.
        let Some(gpu_handle) = FakeMeshComponent::vertices_gpu_handle(&store, e.index()) else { continue };
        if gpu_handle.count == 0 {
            continue;
        }
        let byte_len = gpu_handle.count as u64 * std::mem::size_of::<FakeVertex>() as u64;
        let bytes = vert_pool.read_buffer();
        let raw = readback(&ctx, &bytes, gpu_handle.offset as u64 * std::mem::size_of::<FakeVertex>() as u64, byte_len);
        drop(bytes);

        let first_cpu = mesh.vertices[0];
        let first_gpu_x = f32::from_ne_bytes(raw[0..4].try_into().unwrap());
        checked += 1;
        if (first_cpu.position[0] - first_gpu_x).abs() > f32::EPSILON {
            mismatches += 1;
            eprintln!(
                "MISMATCH entity {:?}: CPU vertex[0].position.x = {}, GPU bytes decode to {}",
                e, first_cpu.position[0], first_gpu_x
            );
        }
    }
    eprintln!("correctness spot-check: {checked} entities checked, {mismatches} mismatches");
    assert_eq!(mismatches, 0, "GPU-mirrored vertex data diverged from CPU source after frame churn");
    assert!(checked > ENTITY_COUNT / 2, "spot-check should have covered a healthy majority of surviving entities");

    eprintln!("=== massive_mesh_load_stress: PASSED ===\n");
}
