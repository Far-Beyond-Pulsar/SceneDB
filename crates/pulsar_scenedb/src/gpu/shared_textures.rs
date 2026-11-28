//! GPU textures made from assets: where an asset's GPU forms live.
//!
//! An asset can have several GPU forms, its *variants* (the same image
//! sampled as a colour and as data is different texels on the GPU). The
//! variant's number is the owner's: SceneDB only tells variants apart. The
//! owner also makes the texels ([`TextureImage`]); this module decides when
//! it has to, and shares the result:
//!
//! - **one upload per device** for each (asset, variant): scenes on one
//!   device (two viewports of a level) show one GPU allocation;
//! - **one slot per texture store** for each (asset, variant), which a
//!   caller reads into the rows it writes. A new revision of the content
//!   takes the old one's place;
//! - a **revision** is the owner's too: whatever would change the texels
//!   (the asset's content version, a detail level);
//! - nothing is made or uploaded twice, and no lock is held while the owner
//!   makes texels, which can take a while.
//!
//! The table is process-wide, in the asset registries, so a plugin library's
//! copy of this crate shares it.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

use crate::assets::AssetId;
use crate::gpu::{GpuMirrorHandle, MipSource, TextureError, TextureStore};

/// Which GPU form of which asset, at which revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureRequest {
    pub asset: AssetId,
    /// The owner's number for this GPU form of the asset.
    pub variant: u64,
    /// The owner's number for what the texels are made from. A slot holding
    /// another revision is replaced.
    pub revision: u64,
}

/// When a new GPU texture is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureUpload {
    /// Before the call returns.
    Whole,
    /// Over the following frames, under the store's upload budget
    /// ([`TextureStore::advance_uploads`]): the call returns the slot at
    /// once. The slot a newer revision replaces keeps its texture until the
    /// new one is whole; a new slot shows nothing until then.
    Staged,
}

/// Called when a texture becomes shareable on its device, with the bytes the
/// shared textures would take in full ([`TextureImage::native_bytes`]).
pub type PublishedFn = Box<dyn FnOnce(u64) + Send + Sync>;

/// The texels of one variant, as its owner made them.
pub struct TextureImage {
    /// A single-layer, single-sample 2D texture.
    pub descriptor: wgpu::TextureDescriptor<'static>,
    /// The levels the descriptor names.
    pub texels: Arc<dyn MipSource>,
    /// Bytes the image takes at full detail, for an owner's memory budget.
    pub native_bytes: u64,
    pub published: Option<PublishedFn>,
}

/// Why a request got no slot.
#[derive(Debug)]
pub enum SharedTextureError {
    /// The scene has no texture store.
    NoTextureStore,
    /// A lock was poisoned.
    Poisoned,
    /// The owner could not make the texels.
    Produce(String),
    /// The store refused the texture; `slots` is how many it has.
    Store { error: TextureError, slots: u32 },
}

type Store = RwLock<TextureStore>;

/// A slot one texture store holds.
struct Held {
    owner: Weak<Store>,
    revision: u64,
    slot: u32,
}

/// A texture on one device.
struct Shared {
    revision: u64,
    native_bytes: u64,
    texture: wgpu::Texture,
    /// The stores holding a clone; the entry goes when none is left.
    users: Vec<Weak<Store>>,
}

/// (the store's or device's address, asset, variant).
type Key = (usize, AssetId, u64);

#[derive(Default)]
pub(crate) struct Table {
    slots: HashMap<Key, Held>,
    devices: HashMap<Key, Shared>,
}

impl Table {
    /// Forgets what dropped stores held, so their textures can be freed.
    fn prune(&mut self) {
        self.slots.retain(|_, held| held.owner.strong_count() != 0);
        self.devices.retain(|_, shared| {
            shared.users.retain(|user| user.strong_count() != 0);
            !shared.users.is_empty()
        });
    }

    fn held(&self, key: Key, revision: u64) -> Option<u32> {
        self.slots
            .get(&key)
            .filter(|held| held.revision == revision)
            .map(|held| held.slot)
    }

    /// The device's texture at `revision`, with `store` recorded as a user.
    fn share(&mut self, key: Key, revision: u64, store: &Arc<Store>) -> Option<wgpu::Texture> {
        let shared = self.devices.get_mut(&key)?;
        if shared.revision != revision {
            return None;
        }
        let me = Arc::downgrade(store);
        if !shared.users.iter().any(|user| user.ptr_eq(&me)) {
            shared.users.push(me);
        }
        Some(shared.texture.clone())
    }
}

fn table() -> Result<std::sync::MutexGuard<'static, Table>, SharedTextureError> {
    crate::assets::registries()
        .textures
        .lock()
        .map_err(|_| SharedTextureError::Poisoned)
}

fn keys(mirror: &GpuMirrorHandle, store: &Arc<Store>, request: TextureRequest) -> (Key, Key) {
    let device = mirror.store().device_arc();
    (
        (Arc::as_ptr(store) as usize, request.asset, request.variant),
        (
            Arc::as_ptr(&device) as usize,
            request.asset,
            request.variant,
        ),
    )
}

/// Whether [`acquire`] would need no texels from the owner: this scene's
/// store holds the revision, or its device already has it.
pub fn holds(mirror: &GpuMirrorHandle, request: TextureRequest) -> bool {
    let Some(store) = mirror.texture_store() else {
        return false;
    };
    let (store_key, device_key) = keys(mirror, &store, request);
    table().is_ok_and(|table| {
        table.held(store_key, request.revision).is_some()
            || table
                .devices
                .get(&device_key)
                .is_some_and(|shared| shared.revision == request.revision)
    })
}

/// The slot `request` holds in `mirror`'s texture store. `produce` is called
/// only when neither the store nor its device has that revision.
pub fn acquire(
    mirror: &GpuMirrorHandle,
    request: TextureRequest,
    upload: TextureUpload,
    produce: impl FnOnce(&wgpu::Device) -> Result<TextureImage, String>,
) -> Result<u32, SharedTextureError> {
    profiling::profile_scope!("SharedTextures::acquire");
    let texture_store = mirror
        .texture_store()
        .ok_or(SharedTextureError::NoTextureStore)?;
    let device = mirror.store().device_arc();
    let (store_key, device_key) = keys(mirror, &texture_store, request);
    let shared = {
        let mut table = table()?;
        table.prune();
        if let Some(slot) = table.held(store_key, request.revision) {
            return Ok(slot);
        }
        table.share(device_key, request.revision, &texture_store)
    };
    // No lock is held while the owner makes the texels.
    let image = match shared {
        Some(_) => None,
        None => Some(produce(&device).map_err(SharedTextureError::Produce)?),
    };

    let mut table = table()?;
    if let Some(slot) = table.held(store_key, request.revision) {
        return Ok(slot);
    }
    let mut store = texture_store
        .write()
        .map_err(|_| SharedTextureError::Poisoned)?;
    let old = table.slots.remove(&store_key).map(|held| held.slot);
    let mut publish = None;
    let slot = match (shared, image) {
        (Some(texture), _) => {
            if let Some(old) = old {
                let _ = store.unregister(old);
            }
            store.register_shared(texture)
        }
        (None, Some(image)) => match upload {
            TextureUpload::Whole => {
                if let Some(old) = old {
                    let _ = store.unregister(old);
                }
                let levels: Vec<&[u8]> = (0..image.texels.level_count())
                    .map(|level| image.texels.level(level))
                    .collect();
                let slot = store.register_mips(&device, mirror.queue(), &image.descriptor, &levels);
                if let Ok(slot) = slot {
                    publish = store
                        .texture(slot)
                        .cloned()
                        .map(|texture| (texture, image.native_bytes, image.published));
                }
                slot
            }
            TextureUpload::Staged => {
                // Published when whole, not now: another scene must not
                // share half a texture.
                let owner = Arc::downgrade(&texture_store);
                let (native_bytes, published) = (image.native_bytes, image.published);
                store.register_mips_staged(
                    &device,
                    &image.descriptor,
                    image.texels,
                    old,
                    request.asset.0,
                    Some(Box::new(move |texture: &wgpu::Texture| {
                        publish_shared(
                            device_key,
                            request.revision,
                            native_bytes,
                            texture.clone(),
                            owner,
                            published,
                        );
                    })),
                )
            }
        },
        (None, None) => unreachable!("texels are made whenever nothing is shared"),
    };
    let slot = slot.map_err(|error| SharedTextureError::Store {
        error,
        slots: store.slot_count(),
    })?;
    table.slots.insert(
        store_key,
        Held {
            owner: Arc::downgrade(&texture_store),
            revision: request.revision,
            slot,
        },
    );
    drop(store);
    drop(table);
    if let Some((texture, native_bytes, published)) = publish {
        publish_shared(
            device_key,
            request.revision,
            native_bytes,
            texture,
            Arc::downgrade(&texture_store),
            published,
        );
    }
    Ok(slot)
}

/// Whether `mirror`'s store holds a slot for any variant of `asset`.
pub fn holds_asset(mirror: &GpuMirrorHandle, asset: AssetId) -> bool {
    let Some(store) = mirror.texture_store() else {
        return false;
    };
    let address = Arc::as_ptr(&store) as usize;
    table().is_ok_and(|table| {
        table
            .slots
            .keys()
            .any(|(owner, held, _)| *owner == address && *held == asset)
    })
}

/// Gives back every slot `mirror`'s store holds for `asset`, whatever the
/// variant. The device's texture goes once no other store shows it. Returns
/// how many slots were freed.
pub fn release(mirror: &GpuMirrorHandle, asset: AssetId) -> usize {
    let Some(texture_store) = mirror.texture_store() else {
        return 0;
    };
    let Ok(mut table) = table() else {
        return 0;
    };
    let store_address = Arc::as_ptr(&texture_store) as usize;
    let keys: Vec<Key> = table
        .slots
        .keys()
        .filter(|(store, held, _)| *store == store_address && *held == asset)
        .copied()
        .collect();
    if keys.is_empty() {
        return 0;
    }
    let Ok(mut store) = texture_store.write() else {
        return 0;
    };
    for key in &keys {
        if let Some(held) = table.slots.remove(key) {
            let _ = store.unregister(held.slot);
        }
    }
    let me = Arc::downgrade(&texture_store);
    table.devices.retain(|(_, shared_asset, _), shared| {
        if *shared_asset == asset {
            shared.users.retain(|user| !user.ptr_eq(&me));
        }
        !shared.users.is_empty()
    });
    keys.len()
}

/// Records `texture` as the device's one copy of the variant.
fn publish_shared(
    device_key: Key,
    revision: u64,
    native_bytes: u64,
    texture: wgpu::Texture,
    owner: Weak<Store>,
    published: Option<PublishedFn>,
) {
    let total = {
        let Ok(mut table) = table() else {
            return;
        };
        table.devices.insert(
            device_key,
            Shared {
                revision,
                native_bytes,
                texture,
                users: vec![owner],
            },
        );
        table
            .devices
            .values()
            .map(|shared| shared.native_bytes)
            .sum()
    };
    if let Some(published) = published {
        published(total);
    }
}
