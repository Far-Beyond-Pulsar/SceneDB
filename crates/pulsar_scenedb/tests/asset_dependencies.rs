//! Assets that need assets ([`register_composer`]): what a composed asset
//! needs is loaded, shared, counted, followed through tiers and reloads, and
//! let go with it.
//!
//! The registries are process-wide and tests run in parallel, so each test
//! has its own kinds and sources.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use pulsar_scenedb::{
    register_binding, register_composer, register_loader, register_payload_size,
    register_tiered_loader, AssetDependency, AssetId, AssetPayload, AssetState, AssetView, Entity,
    SceneStore, TierLoad, World,
};

/// A component showing a "material": the words of the images it needs.
#[derive(Clone, Copy, Debug, SceneStore)]
struct Surface {
    asset: AssetId,
    sum: u32,
    /// 0 unresolved, 1 ready, 2 failed.
    status: u32,
    complete: u32,
    binds: u64,
}

impl Surface {
    fn of(asset: AssetId) -> Self {
        Self {
            asset,
            sum: 0,
            status: 0,
            complete: 0,
            binds: 0,
        }
    }
}

/// What a material composes into: its images' values, added up.
struct Composed(u32);

fn bind_surface(surface: &mut Surface, view: &AssetView) {
    surface.binds += 1;
    match view.payload::<Composed>() {
        Some(composed) => {
            surface.sum = composed.0;
            surface.status = 1;
            surface.complete = u32::from(view.complete);
        }
        None => {
            assert!(matches!(view.state, AssetState::Failed(_)));
            surface.status = 2;
        }
    }
}

/// Each source's text: for a material, the image sources it needs, space
/// separated; for an image, its value.
fn files() -> &'static Mutex<HashMap<String, String>> {
    static FILES: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    FILES.get_or_init(Default::default)
}

fn write(source: &str, text: &str) {
    files()
        .lock()
        .unwrap()
        .insert(source.to_string(), text.to_string());
}

fn read(source: &str) -> Result<String, String> {
    files()
        .lock()
        .unwrap()
        .get(source)
        .cloned()
        .ok_or_else(|| format!("{source} is missing"))
}

/// Registers a material kind that needs images of `image`. An image has two
/// tiers: its coarse one reads as a tenth of its value. Returns how often
/// images were loaded.
fn kinds(material: &'static str, image: &'static str) -> Arc<AtomicUsize> {
    let loads = Arc::new(AtomicUsize::new(0));
    let counted = loads.clone();
    register_tiered_loader(image, move |source, tier| {
        counted.fetch_add(1, Ordering::SeqCst);
        let value: u32 = read(source)?.trim().parse().map_err(|_| "not a number")?;
        Ok(TierLoad {
            payload: Arc::new(if tier == 0 { value / 10 } else { value }),
            last: tier == 1,
        })
    });
    register_loader(material, |source| {
        Ok(Arc::new(read(source)?) as AssetPayload)
    });
    register_composer(
        material,
        move |raw| {
            let text = raw.downcast_ref::<String>().expect("the loader's text");
            text.split_whitespace()
                .map(|need| match need.strip_prefix('@') {
                    // Another material, for the cycle test.
                    Some(other) => AssetId::declare(material, other),
                    None => AssetId::declare(image, need),
                })
                .collect()
        },
        |_, needs: &[AssetDependency]| {
            let mut sum = 0;
            for need in needs {
                match need.payload::<u32>() {
                    Some(value) => sum += *value,
                    None => return Err(format!("an image is unavailable: {:?}", need.state)),
                }
            }
            Ok(Arc::new(Composed(sum)) as AssetPayload)
        },
    );
    register_payload_size(material, |_| 1);
    register_payload_size(image, |_| 1);
    register_binding::<Surface>(material, bind_surface);
    loads
}

fn settle(world: &mut World) {
    assert!(
        world.resolve_assets_blocking(Duration::from_secs(10)),
        "assets did not settle"
    );
    // Binds what the last pump completed.
    world.pump_assets();
}

fn surface(world: &World, entity: Entity) -> Surface {
    *world.get::<Surface>(entity).unwrap()
}

#[test]
fn what_an_asset_needs_is_loaded_once_and_shared() {
    let loads = kinds("dep-share-material", "dep-share-image");
    write("share/a", "100");
    write("share/b", "200");
    write("share/one", "share/a share/b");
    write("share/two", "share/b");
    let mut world = World::new();
    let one = world.spawn();
    let two = world.spawn();
    let first = AssetId::declare("dep-share-material", "share/one");
    let second = AssetId::declare("dep-share-material", "share/two");
    world.insert(one, Surface::of(first));
    world.insert(two, Surface::of(second));
    settle(&mut world);

    assert_eq!(
        (surface(&world, one).status, surface(&world, one).sum),
        (1, 300)
    );
    assert_eq!(surface(&world, two).sum, 200);
    assert_eq!(surface(&world, one).complete, 1);
    // Two tiers of each of two images, whoever needs them.
    assert_eq!(loads.load(Ordering::SeqCst), 4);
    let shared = AssetId::of("dep-share-image", "share/b");
    assert_eq!(world.asset_ref_count(shared), 2);
}

#[test]
fn a_parent_composes_coarse_first_and_again_as_its_needs_sharpen() {
    kinds("dep-tier-material", "dep-tier-image");
    write("tier/image", "500");
    write("tier/material", "tier/image");
    let mut world = World::new();
    let entity = world.spawn();
    world.insert(
        entity,
        Surface::of(AssetId::declare("dep-tier-material", "tier/material")),
    );
    let mut seen = Vec::new();
    for _ in 0..10_000 {
        let report = world.pump_assets();
        let now = surface(&world, entity);
        if now.status == 1 && seen.last() != Some(&(now.sum, now.complete)) {
            seen.push((now.sum, now.complete));
        }
        if report.pending == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(seen, vec![(50, 0), (500, 1)], "coarse first, then complete");
}

#[test]
fn a_changed_need_recomposes_what_needs_it() {
    kinds("dep-reload-material", "dep-reload-image");
    write("reload/image", "100");
    write("reload/other", "40");
    write("reload/material", "reload/image");
    let mut world = World::new();
    let entity = world.spawn();
    let material = AssetId::declare("dep-reload-material", "reload/material");
    world.insert(entity, Surface::of(material));
    settle(&mut world);

    write("reload/image", "900");
    assert_eq!(world.asset_source_changed("reload/image"), 1);
    settle(&mut world);
    assert_eq!(surface(&world, entity).sum, 900);

    // The material itself changes what it needs: the old one is let go.
    write("reload/material", "reload/other");
    assert_eq!(world.asset_source_changed("reload/material"), 1);
    settle(&mut world);
    assert_eq!(surface(&world, entity).sum, 40);
    let old = AssetId::of("dep-reload-image", "reload/image");
    assert_eq!(world.asset_ref_count(old), 0);
}

#[test]
fn needs_are_let_go_when_what_needs_them_is_dropped() {
    kinds("dep-evict-material", "dep-evict-image");
    write("evict/image", "100");
    write("evict/material", "evict/image");
    let mut world = World::new();
    let entity = world.spawn();
    let material = AssetId::declare("dep-evict-material", "evict/material");
    let image = AssetId::of("dep-evict-image", "evict/image");
    world.insert(entity, Surface::of(material));
    settle(&mut world);

    world.despawn(entity);
    // The cached material still holds its image.
    assert_eq!(world.asset_ref_count(image), 1);
    assert_eq!(
        world.evict_unreferenced_assets(),
        2,
        "the material, then its image"
    );
    assert_eq!(world.asset_ref_count(image), 0);
    assert_eq!(world.asset_state(image), AssetState::Unresolved);

    // And it all comes back when referenced again.
    let again = world.spawn();
    world.insert(again, Surface::of(material));
    settle(&mut world);
    assert_eq!(surface(&world, again).sum, 100);
}

#[test]
fn the_owner_decides_what_a_missing_need_means() {
    kinds("dep-missing-material", "dep-missing-image");
    write("missing/material", "missing/nowhere");
    let mut world = World::new();
    let entity = world.spawn();
    world.insert(
        entity,
        Surface::of(AssetId::declare("dep-missing-material", "missing/material")),
    );
    settle(&mut world);
    assert_eq!(surface(&world, entity).status, 2);

    // The need appears: what needs it recovers.
    write("missing/nowhere", "70");
    world.asset_source_changed("missing/nowhere");
    settle(&mut world);
    assert_eq!(
        (surface(&world, entity).status, surface(&world, entity).sum),
        (1, 70)
    );
}

#[test]
fn an_asset_that_needs_itself_fails_instead_of_waiting() {
    kinds("dep-cycle-material", "dep-cycle-image");
    write("cycle/a", "@cycle/b");
    write("cycle/b", "@cycle/a");
    let mut world = World::new();
    let entity = world.spawn();
    world.insert(
        entity,
        Surface::of(AssetId::declare("dep-cycle-material", "cycle/a")),
    );
    settle(&mut world);
    assert_eq!(surface(&world, entity).status, 2);
}

/// GPU textures of assets out of use are given back.
#[cfg(feature = "gpu")]
mod release {
    use super::*;
    use pulsar_scenedb::gpu::shared_textures::{self, TextureImage, TextureRequest, TextureUpload};
    use pulsar_scenedb::gpu::{
        EngineGpuContext, GpuMirrorHandle, MipSource, SceneGpuConfig, SceneGpuStore, TextureStore,
    };
    use std::sync::RwLock;

    struct Texel;

    impl MipSource for Texel {
        fn level_count(&self) -> u32 {
            1
        }

        fn level(&self, _level: u32) -> &[u8] {
            &[1, 2, 3, 4]
        }
    }

    fn world_with_textures() -> Option<(World, GpuMirrorHandle)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).ok()?;
        let ctx = EngineGpuContext::new(Arc::new(device), Arc::new(queue));
        let store = Arc::new(SceneGpuStore::new(
            &ctx,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        ));
        let mirror = GpuMirrorHandle::new(store, Arc::clone(ctx.queue()))
            .with_texture_store(Arc::new(RwLock::new(TextureStore::new(8))))
            .ok()?;
        let mut world = World::new();
        world.attach_gpu_mirror(mirror.clone());
        world.set_asset_release_delay(3);
        Some((world, mirror))
    }

    fn wanted(asset: AssetId) -> TextureRequest {
        TextureRequest {
            asset,
            variant: 0,
            revision: 1,
        }
    }

    /// What an owner does when a row first shows `asset`.
    fn show(mirror: &GpuMirrorHandle, asset: AssetId) {
        shared_textures::acquire(mirror, wanted(asset), TextureUpload::Whole, |_| {
            Ok(TextureImage {
                descriptor: wgpu::TextureDescriptor {
                    label: Some("release-test"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                texels: Arc::new(Texel),
                native_bytes: 4,
                published: None,
            })
        })
        .unwrap();
    }

    fn pump(world: &mut World, times: usize) -> usize {
        (0..times).map(|_| world.pump_assets().released).sum()
    }

    #[test]
    fn textures_are_given_back_once_nothing_in_use_needs_them() {
        let Some((mut world, mirror)) = world_with_textures() else {
            return;
        };
        kinds("dep-release-material", "dep-release-image");
        write("release/own", "100");
        write("release/both", "200");
        write("release/material", "release/own release/both");
        write("release/direct", "release/both");
        let material = AssetId::declare("dep-release-material", "release/material");
        let other = AssetId::declare("dep-release-material", "release/direct");
        let (own, both) = (
            AssetId::of("dep-release-image", "release/own"),
            AssetId::of("dep-release-image", "release/both"),
        );
        let first = world.spawn();
        let second = world.spawn();
        world.insert(first, Surface::of(material));
        world.insert(second, Surface::of(other));
        settle(&mut world);
        show(&mirror, own);
        show(&mirror, both);
        assert!(world.asset_in_use(own) && world.asset_in_use(both));

        world.despawn(first);
        assert!(
            !world.asset_in_use(own),
            "only the cached material holds it"
        );
        assert!(
            world.asset_in_use(both),
            "the other material still needs it"
        );
        assert!(world.asset_work_pending(), "a release is due");
        assert_eq!(pump(&mut world, 3), 0, "not before the delay");
        assert!(shared_textures::holds(&mirror, wanted(own)));
        assert_eq!(pump(&mut world, 2), 1);
        assert!(!shared_textures::holds(&mirror, wanted(own)));
        assert!(shared_textures::holds(&mirror, wanted(both)));

        world.despawn(second);
        assert_eq!(pump(&mut world, 5), 1);
        assert!(!shared_textures::holds(&mirror, wanted(both)));
        assert!(!world.asset_work_pending());
    }

    #[test]
    fn an_asset_referenced_again_within_the_delay_keeps_its_textures() {
        let Some((mut world, mirror)) = world_with_textures() else {
            return;
        };
        kinds("dep-keep-material", "dep-keep-image");
        write("keep/image", "100");
        write("keep/material", "keep/image");
        let material = AssetId::declare("dep-keep-material", "keep/material");
        let image = AssetId::of("dep-keep-image", "keep/image");
        let first = world.spawn();
        world.insert(first, Surface::of(material));
        settle(&mut world);
        show(&mirror, image);

        world.despawn(first);
        pump(&mut world, 1);
        let again = world.spawn();
        world.insert(again, Surface::of(material));
        assert_eq!(pump(&mut world, 8), 0);
        assert!(shared_textures::holds(&mirror, wanted(image)));
    }
}
