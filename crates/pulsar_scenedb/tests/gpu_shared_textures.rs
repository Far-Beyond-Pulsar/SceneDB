//! `gpu::shared_textures`: one upload per device and asset variant, one slot
//! per texture store, and slots given back when released.

use pulsar_scenedb::gpu::shared_textures::{
    self, SharedTextureError, TextureImage, TextureRequest, TextureUpload,
};
use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuMirrorHandle, MipSource, SceneGpuConfig, SceneGpuStore, TextureStore,
    UploadBudget,
};
use pulsar_scenedb::AssetId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

struct Texels(Vec<u8>);

impl MipSource for Texels {
    fn level_count(&self) -> u32 {
        1
    }

    fn level(&self, _level: u32) -> &[u8] {
        &self.0
    }
}

fn context() -> Option<EngineGpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).ok()?;
    Some(EngineGpuContext::new(Arc::new(device), Arc::new(queue)))
}

fn scene(ctx: &EngineGpuContext, slots: u32) -> GpuMirrorHandle {
    let store = Arc::new(SceneGpuStore::new(
        ctx,
        SceneGpuConfig {
            classes: vec![],
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    ));
    GpuMirrorHandle::new(store, Arc::clone(ctx.queue()))
        .with_texture_store(Arc::new(RwLock::new(TextureStore::new(slots))))
        .expect("a texture store on the mirror's device")
}

fn image() -> TextureImage {
    TextureImage {
        descriptor: wgpu::TextureDescriptor {
            label: Some("shared-textures-test"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        texels: Arc::new(Texels(vec![7; 64])),
        native_bytes: 64,
        published: None,
    }
}

/// Requests `request`, counting how often the owner is asked for texels.
fn take(
    scene: &GpuMirrorHandle,
    request: TextureRequest,
    upload: TextureUpload,
    made: &AtomicUsize,
) -> Result<u32, SharedTextureError> {
    shared_textures::acquire(scene, request, upload, |_| {
        made.fetch_add(1, Ordering::Relaxed);
        Ok(image())
    })
}

fn request(asset: &str, variant: u64, revision: u64) -> TextureRequest {
    TextureRequest {
        asset: AssetId::of("shared-textures-test", asset),
        variant,
        revision,
    }
}

#[test]
fn a_variant_is_made_once_and_shared_between_the_scenes_of_a_device() {
    let Some(ctx) = context() else {
        return;
    };
    let (first, second) = (scene(&ctx, 8), scene(&ctx, 8));
    let made = AtomicUsize::new(0);
    let wanted = request("shared", 0, 1);

    let slot = take(&first, wanted, TextureUpload::Whole, &made).unwrap();
    assert_eq!(
        take(&first, wanted, TextureUpload::Whole, &made).unwrap(),
        slot
    );
    assert!(shared_textures::holds(&second, wanted));
    take(&second, wanted, TextureUpload::Whole, &made).unwrap();
    assert_eq!(made.load(Ordering::Relaxed), 1);

    // Another variant of the same asset is other texels.
    let data = request("shared", 1, 1);
    assert!(!shared_textures::holds(&first, data));
    assert_ne!(
        take(&first, data, TextureUpload::Whole, &made).unwrap(),
        slot
    );
    assert_eq!(made.load(Ordering::Relaxed), 2);
}

#[test]
fn a_new_revision_takes_the_old_ones_place_and_release_gives_slots_back() {
    let Some(ctx) = context() else {
        return;
    };
    let scene = scene(&ctx, 2);
    let made = AtomicUsize::new(0);

    let old = take(
        &scene,
        request("reloaded", 0, 1),
        TextureUpload::Whole,
        &made,
    )
    .unwrap();
    let new = take(
        &scene,
        request("reloaded", 0, 2),
        TextureUpload::Whole,
        &made,
    )
    .unwrap();
    assert_eq!(new, old, "the replaced slot is reused");
    take(
        &scene,
        request("reloaded", 1, 2),
        TextureUpload::Whole,
        &made,
    )
    .unwrap();

    let other = request("other", 0, 1);
    let full = take(&scene, other, TextureUpload::Whole, &made);
    assert!(
        matches!(full, Err(SharedTextureError::Store { slots: 2, .. })),
        "{full:?}"
    );
    assert_eq!(
        shared_textures::release(&scene, request("reloaded", 0, 2).asset),
        2
    );
    take(&scene, other, TextureUpload::Whole, &made).unwrap();
    // Released and held by no other scene: made again when asked for.
    assert!(!shared_textures::holds(&scene, request("reloaded", 0, 2)));
}

#[test]
fn a_staged_texture_is_shared_only_once_it_is_whole() {
    let Some(ctx) = context() else {
        return;
    };
    let (first, second) = (scene(&ctx, 4), scene(&ctx, 4));
    let made = AtomicUsize::new(0);
    let wanted = request("staged", 0, 1);

    take(&first, wanted, TextureUpload::Staged, &made).unwrap();
    assert!(!shared_textures::holds(&second, wanted));
    let store = first.texture_store().unwrap();
    while store.read().unwrap().has_staged_uploads() {
        store.write().unwrap().advance_uploads(
            first.queue(),
            UploadBudget {
                bytes: usize::MAX,
                time: None,
            },
            None,
        );
    }
    assert!(shared_textures::holds(&second, wanted));
    take(&second, wanted, TextureUpload::Staged, &made).unwrap();
    assert_eq!(made.load(Ordering::Relaxed), 1);
}

#[test]
fn an_owner_that_cannot_make_the_texels_takes_no_slot() {
    let Some(ctx) = context() else {
        return;
    };
    let scene = scene(&ctx, 1);
    let wanted = request("broken", 0, 1);
    let failed = shared_textures::acquire(&scene, wanted, TextureUpload::Whole, |_| {
        Err("not an image".into())
    });
    assert!(matches!(failed, Err(SharedTextureError::Produce(_))));
    take(&scene, wanted, TextureUpload::Whole, &AtomicUsize::new(0)).unwrap();
}
