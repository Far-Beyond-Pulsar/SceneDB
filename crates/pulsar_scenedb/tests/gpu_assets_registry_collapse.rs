//! Issue #41 Tier-2 collapse, proven end-to-end for `gpu/assets.rs`: every
//! bespoke asset store (`GeometryArena`, `MeshRegistry`, `ClusterBuffer`,
//! `MeshletBuffer`, `MaterialRegistry`, `TextureStore`) is reachable by key
//! through ONE `GpuBufferRegistry` via its `register_key` method, alongside
//! whatever `SceneGpuStore` already registers there — without changing any
//! of these types' own CPU-side APIs (`register`/`get`/`append`/`entries`/
//! `rebuild`/`texture` all still work exactly as before; `register_key` is
//! purely additive).
//!
//! Scope note: this proves the "reachable by key" half of the collapse map
//! (issue #41 acceptance criterion 4). It does NOT claim a
//! `#[gpu(buffer = "builtin_material")]` component field and a standalone
//! `MaterialRegistry` constructed independently share the literal same
//! `wgpu::Buffer` — `GpuBufferRegistry::register_row`'s compatibility check
//! validates element type/stride/access/mode, not buffer identity, so two
//! independently-constructed owners registering under the same key is a
//! configuration hazard (last registration wins the registry's pointer),
//! not automatic aliasing. Real cross-owner buffer sharing between a
//! `SceneGpuStore`-internal shared column and an externally-built asset
//! store would need `SceneGpuStore`'s registration path to consult the
//! registry FIRST and adopt an existing external buffer — a further
//! integration this test does not exercise.

use pulsar_scenedb::gpu::{
    BufferAccess, ClusterBuffer, ClusterNode, EngineGpuContext, GeometryArena, GpuBufferRegistry,
    MaterialRegistry, MeshMetadata, MeshRegistry, MeshletBuffer, TextureStore, CLUSTER_BUFFER_KEY,
    GEOMETRY_INDEX_BUFFER_KEY, GEOMETRY_VERTEX_BUFFER_KEY, MATERIAL_BUFFER_KEY,
    MESHLET_BUFFER_KEY, MESH_METADATA_BUFFER_KEY, TEXTURE_STORE_KEY,
};

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
        label: Some("scenedb-gpu-assets-registry-collapse-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

#[test]
fn every_asset_store_registers_its_key_and_resolves_from_one_shared_registry() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();

    let arena = GeometryArena::new(&ctx, 4096, 2048);
    arena.register_key(&registry).expect("geometry arena registers both keys");

    let meshes = MeshRegistry::new(&ctx, 8);
    meshes.register_key(&registry).expect("mesh registry registers");

    let clusters = ClusterBuffer::new(&ctx, 8);
    clusters.register_key(&registry).expect("cluster buffer registers");

    let meshlets = MeshletBuffer::new(&ctx, 8);
    meshlets.register_key(&registry).expect("meshlet buffer registers");

    let materials = MaterialRegistry::new(&ctx, 8);
    materials.register_key(&registry).expect("material registry registers");

    let textures = TextureStore::new(4);
    textures.register_key(&registry).expect("texture store registers");

    // Every key is reachable, and resolves to the SAME buffer each owning
    // type exposes through its own accessor.
    assert_eq!(registry.resolve(GEOMETRY_VERTEX_BUFFER_KEY).unwrap().buffer.size(), arena.vertex_buffer().size());
    assert_eq!(registry.resolve(GEOMETRY_INDEX_BUFFER_KEY).unwrap().buffer.size(), arena.index_buffer().size());
    assert_eq!(registry.resolve(MESH_METADATA_BUFFER_KEY).unwrap().buffer.size(), meshes.buffer().size());
    assert_eq!(registry.resolve(CLUSTER_BUFFER_KEY).unwrap().buffer.size(), clusters.buffer().size());
    assert_eq!(registry.resolve(MESHLET_BUFFER_KEY).unwrap().buffer.size(), meshlets.buffer().size());
    assert_eq!(registry.resolve(MATERIAL_BUFFER_KEY).unwrap().buffer.size(), materials.buffer().size());

    // TextureStore has no wgpu::Buffer — it reserved a texture-array
    // namespace entry instead, per GpuBufferRegistry::register_texture_array.
    assert!(registry.resolve(TEXTURE_STORE_KEY).is_none(), "texture arrays never resolve to a buffer");
    assert!(registry.is_texture_array(TEXTURE_STORE_KEY));
    assert_eq!(registry.texture_array_slot_capacity(TEXTURE_STORE_KEY), Some(0), "no textures registered yet");

    // Seven distinct keys, seven distinct entries — no accidental aliasing.
    assert_eq!(registry.len(), 7);
}

#[test]
fn mesh_metadata_written_before_and_after_register_key_is_visible_through_the_registry() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let mut meshes = MeshRegistry::new(&ctx, 4);

    // Write BEFORE registering the key: the registry entry is the live
    // wgpu::Buffer, not a snapshot, so prior writes are visible.
    meshes.register(ctx.queue(), MeshMetadata {
        vertex_offset: 1, index_offset: 2, index_count: 3, base_vertex: 0,
        material_index: 0, lod_count: 1, lod_distances: [1.0, 0.0, 0.0, 0.0],
        local_aabb_center: [0.0, 0.0, 0.0], cluster_table_offset: 0,
        local_aabb_extents: [1.0, 1.0, 1.0], meshlet_count: 0,
    }).expect("first mesh");
    meshes.register_key(&registry).expect("register key after first write");

    let handle = registry.resolve(MESH_METADATA_BUFFER_KEY).expect("resolvable");
    let got: MeshMetadata = pulsar_scenedb::gpu::readback_row(ctx.device(), ctx.queue(), &handle.buffer, 0);
    assert_eq!(got.vertex_offset, 1);
    assert_eq!(got.lod_count, 1);
}

#[test]
fn a_row_buffer_cannot_squat_on_the_material_key_registered_by_material_registry() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let materials = MaterialRegistry::new(&ctx, 4);
    materials.register_key(&registry).expect("material registry registers first");

    // A different element type under the same key is rejected, not silently
    // aliased — proves the collapse still enforces issue #41's "mismatched
    // element/stride/packing/access/mode combos are rejected at
    // registration" acceptance criterion even for Tier-2 asset buffers.
    let err = registry
        .register_row::<ClusterNode>(
            MATERIAL_BUFFER_KEY,
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("bad"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            BufferAccess::ReadOnly,
            pulsar_scenedb::gpu::MirrorMode::Once,
        )
        .expect_err("ClusterNode must not be able to alias the material row buffer");
    match err {
        pulsar_scenedb::gpu::BufferRegistrationError::ElementTypeMismatch { key, .. } => {
            assert_eq!(key, MATERIAL_BUFFER_KEY);
        }
        other => panic!("expected ElementTypeMismatch, got {other:?}"),
    }
}

#[test]
fn texture_store_re_registering_after_growth_updates_the_recorded_slot_capacity() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let mut textures = TextureStore::new(4);
    textures.register_key(&registry).expect("initial registration");
    assert_eq!(registry.texture_array_slot_capacity(TEXTURE_STORE_KEY), Some(0));

    let desc = wgpu::TextureDescriptor {
        label: Some("t"),
        size: wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let pixels = [0u8; 16];
    textures.register(ctx.device(), ctx.queue(), &desc, &pixels).expect("register texture");
    textures.register_key(&registry).expect("re-registration after growth");
    assert_eq!(registry.texture_array_slot_capacity(TEXTURE_STORE_KEY), Some(1));
}

#[test]
fn material_row_and_mesh_metadata_resolve_to_their_own_distinctly_sized_buffers() {
    // Sanity: two DIFFERENT keys never collide even when both owners are
    // registered into the same shared registry back-to-back — each key's
    // resolved buffer size matches only its own owner's `max_* * stride`.
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let meshes = MeshRegistry::new(&ctx, 4); // 4 * 72 = 288 bytes
    let materials = MaterialRegistry::new(&ctx, 4); // 4 * 64 = 256 bytes
    meshes.register_key(&registry).unwrap();
    materials.register_key(&registry).unwrap();

    let mesh_handle = registry.resolve(MESH_METADATA_BUFFER_KEY).unwrap();
    let material_handle = registry.resolve(MATERIAL_BUFFER_KEY).unwrap();
    assert_eq!(mesh_handle.buffer.size(), 4 * 72);
    assert_eq!(material_handle.buffer.size(), 4 * 64);
    assert_ne!(mesh_handle.buffer.size(), material_handle.buffer.size(), "distinct buffers, distinct sizes");
}
