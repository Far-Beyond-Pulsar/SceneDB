//! Issue #41 "Systems are declaratively passed buffers", end to end:
//! `SceneGpuStore::buffer_registry()` + an asset store's `register_key` +
//! `GpuSystemContext::bind::<E>("key")` — a "system" (here, an ordinary
//! function taking `&GpuSystemContext`, matching `Schedule`'s own
//! `FnMut(&mut World, GameTime)` closure shape rather than requiring a new
//! parameter-injection framework) resolves exactly the `wgpu::Buffer` +
//! epoch it declared, with a setup-time error — never a runtime panic — for
//! a missing key or a mismatched element type.

use pulsar_scenedb::gpu::{
    BufferKey, BufferResolveError, EngineGpuContext, GpuSystemContext, MaterialRegistry,
    MaterialRow, RegionClassConfig, SceneGpuConfig, SceneGpuStore, MATERIAL_BUFFER_KEY,
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
        label: Some("scenedb-gpu-system-binding-e2e-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: 16, max_resident_cells: 1 }],
        tombstone_headroom: 4,
        max_cells_metadata: 4,
    }
}

/// A "system" in the sense issue #41 means: it declares its GPU dependency
/// by key + type and gets back exactly the wgpu object it needs to bind —
/// nothing about rows, presence, tombstones, or mirror modes leaks through.
fn material_binding_system(ctx: &GpuSystemContext) -> Result<wgpu::Buffer, BufferResolveError> {
    let binding = ctx.bind::<MaterialRow>(MATERIAL_BUFFER_KEY, false)?;
    Ok(binding.buffer().clone())
}

#[test]
fn a_system_resolves_the_material_buffer_registered_by_material_registry() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    let materials = MaterialRegistry::new(&ctx, 4);
    materials.register_key(store.buffer_registry()).expect("material registry registers into the store's registry");

    let sys_ctx = GpuSystemContext::new(store.buffer_registry());
    let bound = material_binding_system(&sys_ctx).expect("system resolves the material buffer");
    assert_eq!(bound.size(), materials.buffer().size());
}

#[test]
fn a_system_gets_a_setup_time_error_not_a_panic_when_the_buffer_is_missing() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    // Deliberately never registered.
    let sys_ctx = GpuSystemContext::new(store.buffer_registry());
    let err = material_binding_system(&sys_ctx).unwrap_err();
    assert_eq!(err, BufferResolveError::Missing { key: MATERIAL_BUFFER_KEY });
}

#[test]
fn a_system_gets_a_setup_time_error_for_a_key_registered_under_a_different_element_type() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    // A shared column of a DIFFERENT element type claims the key first —
    // simulates a misconfigured system declaring the wrong type for a real
    // key, which must fail loudly at bind time, not by silently reading the
    // wrong stride.
    store.buffer_registry()
        .register_row::<u32>(
            MATERIAL_BUFFER_KEY,
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("wrong-type"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            pulsar_scenedb::gpu::BufferAccess::ReadOnly,
            pulsar_scenedb::gpu::MirrorMode::Once,
        )
        .expect("register a decoy u32 buffer under the material key");

    let sys_ctx = GpuSystemContext::new(store.buffer_registry());
    let err = material_binding_system(&sys_ctx).unwrap_err();
    match err {
        BufferResolveError::ElementTypeMismatch { key, .. } => assert_eq!(key, MATERIAL_BUFFER_KEY),
        other => panic!("expected ElementTypeMismatch, got {other:?}"),
    }
}

#[test]
fn can_bind_lets_setup_code_probe_without_unwinding() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    let sys_ctx = GpuSystemContext::new(store.buffer_registry());
    assert!(!sys_ctx.can_bind::<MaterialRow>(MATERIAL_BUFFER_KEY, false));

    let materials = MaterialRegistry::new(&ctx, 4);
    materials.register_key(store.buffer_registry()).unwrap();
    assert!(sys_ctx.can_bind::<MaterialRow>(MATERIAL_BUFFER_KEY, false));
    assert!(!sys_ctx.can_bind::<u32>(MATERIAL_BUFFER_KEY, false), "wrong element type still fails");
}

#[test]
fn a_key_never_registered_anywhere_is_unbindable_by_any_system() {
    let ctx = test_context();
    let store = SceneGpuStore::new(&ctx, scene_cfg());
    let sys_ctx = GpuSystemContext::new(store.buffer_registry());
    assert!(sys_ctx.bind::<u32>(BufferKey::of("totally-unregistered"), false).is_err());
}
