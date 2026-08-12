//! Issue #41 Tier-2 collapse for `gpu/view_upload.rs`: `ViewTokenBuffers`'
//! token/expected-gens pair is reachable by key through a shared
//! `GpuBufferRegistry` via `register_keys`, idempotently re-syncable every
//! frame, and its epoch only advances on a real reallocation — never on a
//! same-capacity re-sync.

use pulsar_scenedb::gpu::{BufferKey, EngineGpuContext, GpuBufferRegistry, HarvestStaging, MeshClass, ViewTokenBuffers};

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
        label: Some("scenedb-gpu-view-upload-registry-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

fn staging_with(tokens: &[u32]) -> HarvestStaging {
    HarvestStaging {
        traditional: tokens.to_vec(),
        traditional_gens: vec![0u32; tokens.len()],
        ..Default::default()
    }
}

const TOKENS_KEY: BufferKey = BufferKey::of("builtin_view_token_traditional");
const GENS_KEY: BufferKey = BufferKey::of("builtin_view_token_traditional_gens");

#[test]
fn register_keys_resolves_both_buffers_and_epoch_starts_at_zero() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let mut buffers = ViewTokenBuffers::new(&ctx, "reg-test", 0);
    buffers.upload(&ctx, &staging_with(&[1, 2, 3, 4]), MeshClass::Traditional);
    assert_eq!(buffers.epoch(), 1, "capacity 0 -> 4 is a real growth");

    buffers.register_keys(&registry, TOKENS_KEY, GENS_KEY).expect("register");
    let tokens_handle = registry.resolve(TOKENS_KEY).expect("tokens resolvable");
    let gens_handle = registry.resolve(GENS_KEY).expect("gens resolvable");
    assert_eq!(tokens_handle.buffer.size(), buffers.tokens_buffer().size());
    assert_eq!(gens_handle.buffer.size(), buffers.expected_gens_buffer().size());
}

#[test]
fn register_keys_is_idempotent_and_only_bumps_the_registry_epoch_on_real_growth() {
    let ctx = test_context();
    let registry = GpuBufferRegistry::new();
    let mut buffers = ViewTokenBuffers::new(&ctx, "reg-test-2", 0);

    buffers.upload(&ctx, &staging_with(&[1, 2, 3, 4]), MeshClass::Traditional);
    buffers.register_keys(&registry, TOKENS_KEY, GENS_KEY).expect("first registration");
    let epoch0 = registry.epoch(TOKENS_KEY).unwrap();

    // Re-sync with no growth in between (same capacity): epoch must NOT bump.
    buffers.upload(&ctx, &staging_with(&[5, 6]), MeshClass::Traditional);
    buffers.register_keys(&registry, TOKENS_KEY, GENS_KEY).expect("re-sync, no growth");
    assert_eq!(registry.epoch(TOKENS_KEY).unwrap(), epoch0, "no growth happened -> no epoch bump");

    // Force growth, then re-sync: epoch DOES bump, and the registry's buffer
    // is the NEW (grown) one.
    buffers.upload(&ctx, &staging_with(&(0..64).collect::<Vec<u32>>()), MeshClass::Traditional);
    assert!(buffers.epoch() > 1, "grew past initial small capacity");
    buffers.register_keys(&registry, TOKENS_KEY, GENS_KEY).expect("re-sync after growth");
    assert!(registry.epoch(TOKENS_KEY).unwrap() > epoch0, "growth must bump the registry epoch");
    assert_eq!(
        registry.resolve(TOKENS_KEY).unwrap().buffer.size(),
        buffers.tokens_buffer().size(),
        "registry now points at the grown buffer"
    );
}
