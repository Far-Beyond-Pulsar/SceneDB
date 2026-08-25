//! End-to-end integration tests for content-id-interned var-len `#[gpu]`
//! fields (`#[gpu(mirror = Once, content_id = "sibling")]`) — the actual
//! mesh-dedup mechanism (Pulsar-Native#632), exercised through the REAL
//! `#[derive(SceneStore)]` codegen path + a real `World` + a real GPU
//! device, not just `InternedVarLenPool` in isolation (that tier lives in
//! `gpu::interned_pool`'s own unit tests).

#![cfg(feature = "gpu")]

use pulsar_scenedb::gpu::{BufferKey, EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::handle_ledger::{ContentAddressed, HandleId};
use pulsar_scenedb::{Entity, SceneStore, World};
use std::sync::Arc;

/// A minimal "content id" wrapper standing in for Pulsar-Native's
/// `MeshAssetPath` — carries a `HandleId` directly rather than a real file
/// path, since this crate has no domain knowledge of assets and shouldn't
/// need any to test the mechanism.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AssetId(HandleId);

impl ContentAddressed for AssetId {
    fn content_id(&self) -> HandleId {
        self.0
    }
}

/// The `StaticMeshComponent` shape: an identity field plus two interned
/// var-len payload fields sharing it.
#[derive(Clone, Debug, SceneStore)]
struct MeshComponent {
    asset: AssetId,
    #[gpu(mirror = Once, content_id = "asset")]
    vertices: Vec<u32>,
    #[gpu(mirror = Once, content_id = "asset")]
    indices: Vec<u32>,
}

fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("interned-var-len-integration-test"),
        ..Default::default()
    }))
    .expect("device");
    (Arc::new(device), Arc::new(queue))
}

/// Wires a fresh `World` to a real `SceneGpuStore` + `GpuMirrorHandle`,
/// exactly like `helio_bridge::attach_gpu_render_seam` does in
/// Pulsar-Native, minus everything renderer-specific.
fn mirrored_world() -> (World, Arc<SceneGpuStore>, Arc<wgpu::Queue>) {
    let (device, queue) = test_device();
    let ctx = EngineGpuContext::new(device.clone(), queue.clone());
    let gpu_cfg = SceneGpuConfig {
        classes: vec![],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    };
    let store = Arc::new(SceneGpuStore::new(&ctx, gpu_cfg));
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(&queue));
    let mut world = World::new();
    world.attach_gpu_mirror(mirror);
    (world, store, queue)
}

fn vertices_pool<'a>(store: &'a SceneGpuStore) -> Arc<pulsar_scenedb::gpu::InternedVarLenPool<u32>> {
    store
        .interned_var_len_pool::<u32>(BufferKey::of("MeshComponent::vertices"))
        .expect("registered on first insert")
}

fn indices_pool<'a>(store: &'a SceneGpuStore) -> Arc<pulsar_scenedb::gpu::InternedVarLenPool<u32>> {
    store
        .interned_var_len_pool::<u32>(BufferKey::of("MeshComponent::indices"))
        .expect("registered on first insert")
}

#[test]
fn ten_entities_one_asset_share_one_gpu_allocation() {
    let (mut world, store, _queue) = mirrored_world();
    let id = HandleId(1);
    let entities: Vec<Entity> = (0..10)
        .map(|_| {
            let e = world.spawn();
            world.insert(
                e,
                MeshComponent { asset: AssetId(id), vertices: vec![1, 2, 3], indices: vec![0, 1, 2] },
            );
            e
        })
        .collect();

    let vpool = vertices_pool(&store);
    let ipool = indices_pool(&store);
    assert_eq!(vpool.audit(), vec![(id, 10, 12)], "one shared vertex allocation, refcounted by all 10 entities");
    assert_eq!(ipool.audit(), vec![(id, 10, 12)]);

    // Every entity's row-indexed handle resolves to the SAME shared range —
    // this is the whole point: existing `..._gpu_handle` accessors need no
    // API change to observe sharing.
    let handles: Vec<_> = entities
        .iter()
        .map(|e| MeshComponent::vertices_gpu_handle(&store, e.index()).unwrap())
        .collect();
    assert!(handles.iter().all(|h| *h == handles[0]), "all ten entities resolve to the identical shared range");
    assert!(handles[0].count > 0);
}

#[test]
fn despawning_all_referencing_entities_frees_the_shared_allocation() {
    let (mut world, store, _queue) = mirrored_world();
    let id = HandleId(2);
    let entities: Vec<Entity> = (0..5)
        .map(|_| {
            let e = world.spawn();
            world.insert(e, MeshComponent { asset: AssetId(id), vertices: vec![9, 9], indices: vec![0] });
            e
        })
        .collect();

    let vpool = vertices_pool(&store);
    assert_eq!(vpool.audit().len(), 1);

    for (i, &e) in entities.iter().enumerate() {
        world.despawn(e);
        let remaining = 5 - i - 1;
        if remaining > 0 {
            assert_eq!(vpool.audit(), vec![(id, remaining as u64, 8)], "after despawning {} of 5", i + 1);
        } else {
            assert!(vpool.audit().is_empty(), "last reference despawned -- fully torn down");
            assert_eq!(vpool.resolve(id), None);
        }
    }
}

#[test]
fn removing_the_component_without_despawning_also_frees_correctly() {
    let (mut world, store, _queue) = mirrored_world();
    let id = HandleId(3);
    let a = world.spawn();
    let b = world.spawn();
    world.insert(a, MeshComponent { asset: AssetId(id), vertices: vec![1], indices: vec![1] });
    world.insert(b, MeshComponent { asset: AssetId(id), vertices: vec![1], indices: vec![1] });

    let vpool = vertices_pool(&store);
    assert_eq!(vpool.audit(), vec![(id, 2, 4)]);

    // `remove` (entity stays alive) must release exactly like despawn does.
    world.remove::<MeshComponent>(a);
    assert_eq!(vpool.audit(), vec![(id, 1, 4)]);
    assert!(world.is_alive(a), "remove must not despawn the entity");

    world.despawn(b);
    assert!(vpool.audit().is_empty());
}

#[test]
fn switching_an_entity_to_a_different_asset_decrefs_the_old_one() {
    let (mut world, store, _queue) = mirrored_world();
    let a = HandleId(10);
    let b = HandleId(20);
    let e1 = world.spawn();
    let e2 = world.spawn();
    world.insert(e1, MeshComponent { asset: AssetId(a), vertices: vec![1, 1], indices: vec![0] });
    world.insert(e2, MeshComponent { asset: AssetId(a), vertices: vec![1, 1], indices: vec![0] });

    let vpool = vertices_pool(&store);
    assert_eq!(vpool.audit(), vec![(a, 2, 8)]);

    // Rewrite e1 onto a DIFFERENT asset -- old (a) drops to 1 ref, new (b)
    // gets its own fresh allocation.
    world.insert(e1, MeshComponent { asset: AssetId(b), vertices: vec![2, 2, 2], indices: vec![0] });
    let mut audit = vpool.audit();
    audit.sort_by_key(|(id, ..)| id.0);
    assert_eq!(audit, vec![(a, 1, 8), (b, 1, 12)]);
}

#[test]
fn equal_asset_rewrite_of_the_same_entity_does_not_double_count() {
    let (mut world, store, _queue) = mirrored_world();
    let id = HandleId(42);
    let e = world.spawn();
    for _ in 0..10 {
        world.insert(e, MeshComponent { asset: AssetId(id), vertices: vec![1, 2], indices: vec![0] });
    }
    let vpool = vertices_pool(&store);
    assert_eq!(vpool.audit(), vec![(id, 1, 8)], "10 identical rewrites of the SAME entity must still net to refcount 1");
}

#[test]
fn zero_asset_still_writes_privately_but_never_shares() {
    // HandleId::ZERO means "no identity available" (e.g. an unresolvable
    // mesh_asset), not "discard this data" -- the geometry a component
    // already has must still land on the GPU, just as a private,
    // non-shared allocation (exactly what a plain, non-interned Vec<T>
    // #[gpu] field would have done).
    let (mut world, store, _queue) = mirrored_world();
    let e1 = world.spawn();
    let e2 = world.spawn();
    world.insert(e1, MeshComponent { asset: AssetId::default(), vertices: vec![1, 2, 3], indices: vec![0] });
    world.insert(e2, MeshComponent { asset: AssetId::default(), vertices: vec![9, 9], indices: vec![0] });

    let vpool = vertices_pool(&store);
    assert!(vpool.audit().is_empty(), "a private allocation is never interned/shared, so it's invisible to audit");

    let h1 = MeshComponent::vertices_gpu_handle(&store, e1.index()).unwrap();
    let h2 = MeshComponent::vertices_gpu_handle(&store, e2.index()).unwrap();
    assert_eq!(h1.count, 3, "the data was actually written, not discarded");
    assert_eq!(h2.count, 2);
    assert_ne!(h1, h2, "two zero-id rows must NOT accidentally share one allocation");

    world.despawn(e1);
    world.despawn(e2);
    assert!(vpool.audit().is_empty(), "still empty -- private allocations were never counted here in the first place");
}

#[test]
fn leak_loop_repeated_spawn_and_despawn_of_many_entities_leaves_nothing_resident() {
    // The mission's mandatory leak-loop shape (Pulsar-Native#632, #661):
    // spawn a batch, despawn it all, repeat — audit must show empty
    // residency every single time, not just eventually.
    let (mut world, store, _queue) = mirrored_world();
    // Pools only exist once the derive's auto-registration-on-first-insert
    // has run -- fetched fresh each round rather than cached up front, both
    // to sidestep that and because it's what a real long-running caller
    // (e.g. a test harness reading residency between frames) would do
    // anyway.
    let vpool = || vertices_pool(&store);
    let ipool = || indices_pool(&store);

    for round in 0..50u128 {
        let entities: Vec<Entity> = (0..20)
            .map(|k| {
                let e = world.spawn();
                // 4 distinct assets per round, 5 entities each.
                let id = HandleId(round * 10 + (k % 4) as u128 + 1);
                world.insert(
                    e,
                    MeshComponent { asset: AssetId(id), vertices: vec![1, 2, 3, 4], indices: vec![0, 1, 2] },
                );
                e
            })
            .collect();

        assert_eq!(vpool().audit().len(), 4, "round {round}: 4 distinct assets resident mid-round");

        for e in entities {
            world.despawn(e);
        }
        assert!(vpool().audit().is_empty(), "round {round}: vertices leaked");
        assert!(ipool().audit().is_empty(), "round {round}: indices leaked");
    }
}

#[test]
fn same_id_different_bytes_collision_keeps_first_writers_bytes() {
    let (mut world, store, _queue) = mirrored_world();
    let id = HandleId(99);
    let e1 = world.spawn();
    let e2 = world.spawn();
    world.insert(e1, MeshComponent { asset: AssetId(id), vertices: vec![1, 2, 3], indices: vec![0] });
    // Same id, DIFFERENT vertex data -- a real content-id function should
    // never produce this, but the pool must not silently corrupt state if
    // it happens (upstream bug, not a supported path).
    world.insert(e2, MeshComponent { asset: AssetId(id), vertices: vec![7, 7, 7, 7, 7], indices: vec![0] });

    let vpool = vertices_pool(&store);
    let audit = vpool.audit();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0], (id, 2, 12), "byte size reflects e1's 3 elements (first writer), not e2's 5");

    let h1 = MeshComponent::vertices_gpu_handle(&store, e1.index()).unwrap();
    let h2 = MeshComponent::vertices_gpu_handle(&store, e2.index()).unwrap();
    assert_eq!(h1, h2, "both entities resolve to the SAME (first-writer) range");
}
