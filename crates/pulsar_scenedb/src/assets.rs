//! Asset references, owned by the world (SceneDB#84).
//!
//! A component names an asset with an [`AssetId`] field: a stable identity
//! for the asset's *source* (a kind and a source string, such as a mesh path),
//! not for its content. The world resolves every asset its components
//! reference, with no caller-side loading:
//!
//! - **Declaring** an id ([`AssetId::declare`]) records which kind and source
//!   it names, process-wide, so any world can resolve it.
//! - **Writing** a component that references a declared id (insert,
//!   overwrite, `get_mut`) marks the id in the world, through the same
//!   reference counting [`crate::handle_ledger`] does for content handles. The
//!   derive treats an `AssetId` field exactly like a `HandleId` field.
//! - **[`World::pump_assets`]** starts loads for referenced ids the world has
//!   not resolved, off the write path (a worker thread runs the kind's
//!   loader), and applies finished loads: each records the payload and its
//!   state and *binds* every component that references the id. A binding
//!   ([`register_binding`]) is a kind's way to derive component data from the
//!   payload (a mesh component's geometry, say), written through `get_mut`
//!   so GPU mirrors, change tracking and journals see it like any write.
//! - A load result is **generation-checked**: a request carries a token, and
//!   a reload or a newer request for the same id makes older results stale,
//!   so they are dropped. Bindings look up the *current* references when
//!   they run, so a component that was removed, or that now names another
//!   asset, is never bound to a result it no longer asked for.
//! - **Reload** ([`World::reload_asset`], [`World::asset_source_changed`])
//!   bumps the asset's content version and rebinds every dependent when the
//!   new data arrives. No editor subscription is needed for this.
//! - The world keeps each asset's payload (its CPU data) until it is
//!   evicted ([`World::evict_unreferenced_assets`]), so a rebuild has
//!   something to rebuild from.
//!
//! Loaders and bindings are registered process-wide, once, by the crate that
//! defines the asset kind; every world uses them.
//!
//! # Readiness
//!
//! [`World::asset_state`] answers whether an asset is unresolved, loading,
//! ready or failed. A binding runs for ready and failed results alike, so a
//! component can show that its asset is missing or corrupt.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};

use crate::component::{component_id, Component, ComponentId};
use crate::entity::Entity;
use crate::handle_ledger::HandleId;
use crate::World;

/// A stable identity for an asset source: what a component stores to
/// reference an asset. `repr(C)` `Pod` newtype over `u128`, like
/// [`HandleId`]; [`Self::ZERO`] means "no asset".
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct AssetId(pub u128);

// SAFETY: #[repr(C)], one plain u128, no padding, every bit pattern valid --
// the same argument `HandleId`'s impl makes.
unsafe impl crate::page::Pod for AssetId {}

impl AssetId {
    /// The "no asset" sentinel.
    pub const ZERO: AssetId = AssetId(0);

    /// Whether this is the "no asset" sentinel.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The id of `source` as an asset of `kind`. Deterministic across runs
    /// and processes (a fixed 128-bit FNV-1a over both strings), so it can be
    /// persisted. An empty `source` is [`Self::ZERO`].
    pub fn of(kind: &str, source: &str) -> AssetId {
        if source.is_empty() {
            return AssetId::ZERO;
        }
        let mut hash: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
        let prime: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        for byte in kind.bytes().chain([0]).chain(source.bytes()) {
            hash ^= u128::from(byte);
            hash = hash.wrapping_mul(prime);
        }
        // Never ZERO for a real source.
        AssetId(hash.max(1))
    }

    /// [`Self::of`], recording which kind and source the id names so every
    /// world can resolve it. Idempotent.
    pub fn declare(kind: &'static str, source: &str) -> AssetId {
        let id = Self::of(kind, source);
        if !id.is_zero() {
            let mut sources = sources().write().expect("asset sources poisoned");
            sources.entry(id).or_insert_with(|| Declared {
                kind,
                source: Arc::from(source),
            });
        }
        id
    }

    /// The kind and source a declared id names.
    pub fn source(self) -> Option<(&'static str, Arc<str>)> {
        sources()
            .read()
            .expect("asset sources poisoned")
            .get(&self)
            .map(|declared| (declared.kind, declared.source.clone()))
    }

    /// The same bits as a [`HandleId`], under which the world counts
    /// references to this asset.
    #[inline]
    pub fn handle(self) -> HandleId {
        HandleId(self.0)
    }
}

/// Where an asset stands in a world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetState {
    /// Referenced, not yet requested (the next [`World::pump_assets`] starts
    /// it), or not referenced at all.
    Unresolved,
    /// A load is in flight.
    Loading,
    /// Loaded; the payload is available.
    Ready,
    /// The last load failed, with the loader's message. A reload retries.
    Failed(Arc<str>),
}

/// A loaded asset's data, shared by every component that references it.
pub type AssetPayload = Arc<dyn Any + Send + Sync>;

/// What a binding sees of an asset.
pub struct AssetView<'a> {
    pub id: AssetId,
    pub kind: &'static str,
    pub source: &'a str,
    pub state: &'a AssetState,
    /// Bumped by every completed (re)load: a component can store it to skip
    /// binding the same content twice.
    pub version: u64,
    payload: Option<&'a AssetPayload>,
}

impl AssetView<'_> {
    /// The payload as `T`, when the asset is ready and the kind's loader
    /// produced a `T`.
    pub fn payload<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.payload
            .cloned()
            .and_then(|payload| payload.downcast::<T>().ok())
    }
}

/// Loads one asset of a kind from its source: runs on a worker thread.
pub type LoadFn = Arc<dyn Fn(&str) -> Result<AssetPayload, String> + Send + Sync>;

/// Register the loader for `kind`, replacing any earlier one.
pub fn register_loader(
    kind: &'static str,
    load: impl Fn(&str) -> Result<AssetPayload, String> + Send + Sync + 'static,
) {
    loaders()
        .write()
        .expect("asset loaders poisoned")
        .insert(kind, Arc::new(load));
}

/// Register how components of type `T` take in assets of `kind`: `bind`
/// runs, through `get_mut`, on every `T` whose `AssetId` fields name an
/// asset of `kind` when that asset finishes loading (ready or failed), and
/// when a `T` starts referencing an asset that is already resolved. `bind`
/// receives the asset it is bound to; with several `AssetId` fields of the
/// same kind, it decides which field the asset belongs to.
///
/// `T` must have at least one `AssetId` field (`#[derive(SceneStore)]`
/// counts it); otherwise its references are invisible and it is never bound.
pub fn register_binding<T: Component>(kind: &'static str, bind: fn(&mut T, &AssetView)) {
    let binding = Binding {
        component: component_id::<T>,
        apply: Arc::new(move |world: &mut World, entity: Entity, view: &AssetView| {
            if let Some(mut component) = world.get_mut::<T>(entity) {
                bind(&mut component, view);
            }
        }),
    };
    bindings()
        .write()
        .expect("asset bindings poisoned")
        .entry(kind)
        .or_default()
        .push(binding);
}

struct Declared {
    kind: &'static str,
    source: Arc<str>,
}

type ApplyFn = Arc<dyn Fn(&mut World, Entity, &AssetView) + Send + Sync>;

#[derive(Clone)]
struct Binding {
    component: fn() -> ComponentId,
    apply: ApplyFn,
}

/// The declared sources, loaders and bindings: process-wide, and shared by
/// every copy of this crate through [`crate::runtime`], so an asset a
/// plugin declares, or a loader it registers, serves the editor's worlds.
/// (A payload crosses copies as `dyn Any`: [`AssetView::payload`] finds it
/// only from the copy whose `T` the loader built.)
#[derive(Default)]
pub(crate) struct AssetRegistries {
    sources: RwLock<HashMap<AssetId, Declared>>,
    loaders: RwLock<HashMap<&'static str, LoadFn>>,
    bindings: RwLock<HashMap<&'static str, Vec<Binding>>>,
}

fn registries() -> &'static AssetRegistries {
    (crate::runtime::runtime().asset_registries)()
}

fn sources() -> &'static RwLock<HashMap<AssetId, Declared>> {
    &registries().sources
}

fn loaders() -> &'static RwLock<HashMap<&'static str, LoadFn>> {
    &registries().loaders
}

fn bindings() -> &'static RwLock<HashMap<&'static str, Vec<Binding>>> {
    &registries().bindings
}

/// Whether `id` names a declared asset: the handle ledger records references
/// to these for [`World::pump_assets`].
pub(crate) fn is_declared(id: HandleId) -> bool {
    sources()
        .read()
        .expect("asset sources poisoned")
        .contains_key(&AssetId(id.0))
}

struct Entry {
    kind: &'static str,
    source: Arc<str>,
    state: AssetState,
    version: u64,
    /// The request a completion must carry to be applied.
    token: u64,
    payload: Option<AssetPayload>,
}

enum LoadStart {
    Started,
    Failed,
    Undeclared,
}

struct Completion {
    id: AssetId,
    token: u64,
    result: Result<AssetPayload, String>,
}

/// A world's assets: the [`AssetState`], version and payload of every asset
/// it has resolved, and its in-flight loads.
pub(crate) struct AssetTable {
    entries: HashMap<AssetId, Entry>,
    next_token: u64,
    sender: Sender<Completion>,
    receiver: Mutex<Receiver<Completion>>,
    /// Ids to (re)load at the next pump.
    requested: HashSet<AssetId>,
}

impl Default for AssetTable {
    fn default() -> Self {
        let (sender, receiver) = channel();
        Self {
            entries: HashMap::new(),
            next_token: 0,
            sender,
            receiver: Mutex::new(receiver),
            requested: HashSet::new(),
        }
    }
}

/// What one [`World::pump_assets`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetPumpReport {
    /// Loads started.
    pub started: usize,
    /// Loads applied (ready or failed).
    pub completed: usize,
    /// Results dropped because a newer request superseded them.
    pub stale: usize,
    /// Components bound.
    pub bound: usize,
    /// Loads still in flight.
    pub loading: usize,
}

impl World {
    /// Advance asset resolution: apply finished loads (dropping stale ones),
    /// start loads for newly referenced or reloaded assets, and bind the
    /// components that reference assets which resolved or which they started
    /// referencing since the last pump. Cheap when nothing changed; call it
    /// once per frame (or tick).
    pub fn pump_assets(&mut self) -> AssetPumpReport {
        let mut report = AssetPumpReport::default();
        let mut to_bind: HashSet<AssetId> = HashSet::new();

        // Finished loads.
        let completions: Vec<Completion> = {
            let receiver = self
                .assets
                .receiver
                .lock()
                .expect("asset receiver poisoned");
            receiver.try_iter().collect()
        };
        for completion in completions {
            let Some(entry) = self.assets.entries.get_mut(&completion.id) else {
                report.stale += 1;
                continue;
            };
            if entry.token != completion.token || entry.state != AssetState::Loading {
                report.stale += 1;
                continue;
            }
            entry.version += 1;
            match completion.result {
                Ok(payload) => {
                    entry.state = AssetState::Ready;
                    entry.payload = Some(payload);
                }
                Err(message) => {
                    tracing::warn!(kind = entry.kind, source = %entry.source, %message, "asset failed to load");
                    entry.state = AssetState::Failed(Arc::from(message));
                    entry.payload = None;
                }
            }
            report.completed += 1;
            to_bind.insert(completion.id);
        }

        // Assets that components started referencing since the last pump.
        let touched = self
            .handle_counts
            .lock()
            .expect("World handle_counts: mutex poisoned")
            .take_touched_assets();
        for handle in touched {
            let id = AssetId(handle.0);
            // Referenced and released again before this pump: nothing to do.
            if self.asset_ref_count(id) == 0 {
                continue;
            }
            match self.assets.entries.get(&id).map(|entry| &entry.state) {
                None | Some(AssetState::Unresolved) => {
                    self.assets.requested.insert(id);
                }
                Some(AssetState::Loading) => {}
                Some(AssetState::Ready | AssetState::Failed(_)) => {
                    to_bind.insert(id);
                }
            }
        }

        // Start loads.
        let requested: Vec<AssetId> = self.assets.requested.drain().collect();
        for id in requested {
            match self.start_load(id) {
                LoadStart::Started => report.started += 1,
                // No loader: failed on the spot, which its dependents see.
                LoadStart::Failed => {
                    to_bind.insert(id);
                }
                LoadStart::Undeclared => {}
            }
        }

        // Bind.
        for id in to_bind {
            report.bound += self.bind_dependents(id);
        }
        report.loading = self
            .assets
            .entries
            .values()
            .filter(|entry| entry.state == AssetState::Loading)
            .count();
        report
    }

    /// Pump until no load is in flight, or `timeout` passes: for headless
    /// loading and tests. Returns `false` on timeout.
    pub fn resolve_assets_blocking(&mut self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let report = self.pump_assets();
            if report.loading == 0 && self.assets.requested.is_empty() {
                // One more pump binds anything the last one completed.
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Where `id` stands in this world.
    pub fn asset_state(&self, id: AssetId) -> AssetState {
        self.assets
            .entries
            .get(&id)
            .map_or(AssetState::Unresolved, |entry| entry.state.clone())
    }

    /// How many times `id` has finished (re)loading in this world.
    pub fn asset_version(&self, id: AssetId) -> u64 {
        self.assets
            .entries
            .get(&id)
            .map_or(0, |entry| entry.version)
    }

    /// `id`'s payload as `T`, when ready.
    pub fn asset_payload<T: Any + Send + Sync>(&self, id: AssetId) -> Option<Arc<T>> {
        self.assets
            .entries
            .get(&id)
            .and_then(|entry| entry.payload.clone())
            .and_then(|payload| payload.downcast::<T>().ok())
    }

    /// How many component fields reference `id`.
    pub fn asset_ref_count(&self, id: AssetId) -> i64 {
        self.handle_ref_count(id.handle())
    }

    /// Load `id` again at the next pump; its dependents are rebound when the
    /// new data arrives. A load already in flight becomes stale.
    pub fn reload_asset(&mut self, id: AssetId) {
        if id.is_zero() {
            return;
        }
        if let Some(entry) = self.assets.entries.get_mut(&id) {
            // A newer request supersedes the one in flight.
            entry.token = entry.token.wrapping_add(1);
            entry.state = AssetState::Unresolved;
        }
        self.assets.requested.insert(id);
    }

    /// The source `source` changed: reload every asset this world resolved
    /// from it. Returns how many were queued.
    pub fn asset_source_changed(&mut self, source: &str) -> usize {
        let ids: Vec<AssetId> = self
            .assets
            .entries
            .iter()
            .filter(|(_, entry)| &*entry.source == source)
            .map(|(id, _)| *id)
            .collect();
        for &id in &ids {
            self.reload_asset(id);
        }
        ids.len()
    }

    /// Drop the payloads of assets no component references, keeping their
    /// identity (a later reference loads them again). Returns how many.
    pub fn evict_unreferenced_assets(&mut self) -> usize {
        let unreferenced: Vec<AssetId> = self
            .assets
            .entries
            .iter()
            .filter(|(id, entry)| {
                entry.state != AssetState::Loading && self.asset_ref_count(**id) == 0
            })
            .map(|(id, _)| *id)
            .collect();
        for id in &unreferenced {
            self.assets.entries.remove(id);
        }
        unreferenced.len()
    }

    /// Queue a worker load of `id`. An id whose kind has no loader is marked
    /// failed instead; an undeclared one is left unresolved.
    fn start_load(&mut self, id: AssetId) -> LoadStart {
        let Some((kind, source)) = id.source() else {
            tracing::warn!(?id, "asset id was never declared; it cannot be loaded");
            return LoadStart::Undeclared;
        };
        let token = self.assets.next_token;
        self.assets.next_token += 1;
        let entry = self.assets.entries.entry(id).or_insert_with(|| Entry {
            kind,
            source: source.clone(),
            state: AssetState::Unresolved,
            version: 0,
            token,
            payload: None,
        });
        entry.token = token;
        let loader = loaders()
            .read()
            .expect("asset loaders poisoned")
            .get(kind)
            .cloned();
        let Some(loader) = loader else {
            entry.version += 1;
            entry.state =
                AssetState::Failed(Arc::from(format!("no loader for asset kind `{kind}`")));
            entry.payload = None;
            return LoadStart::Failed;
        };
        entry.state = AssetState::Loading;
        let sender = self.assets.sender.clone();
        std::thread::Builder::new()
            .name(format!("scenedb-asset-{kind}"))
            .spawn(move || {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader(&source)))
                        .unwrap_or_else(|_| Err("the loader panicked".to_owned()));
                let _ = sender.send(Completion { id, token, result });
            })
            .expect("spawn asset loader thread");
        LoadStart::Started
    }

    /// Bind every component that currently references `id` to its entry.
    fn bind_dependents(&mut self, id: AssetId) -> usize {
        let Some(entry) = self.assets.entries.get(&id) else {
            return 0;
        };
        let kind = entry.kind;
        let source = entry.source.clone();
        let state = entry.state.clone();
        let version = entry.version;
        let payload = entry.payload.clone();
        let view = AssetView {
            id,
            kind,
            source: &source,
            state: &state,
            version,
            payload: payload.as_ref(),
        };
        let kind_bindings = bindings()
            .read()
            .expect("asset bindings poisoned")
            .get(kind)
            .cloned()
            .unwrap_or_default();
        let mut bound = 0;
        for binding in kind_bindings {
            let component = (binding.component)();
            for entity in self.entities_referencing(component, id.handle()) {
                (binding.apply)(self, entity, &view);
                bound += 1;
            }
        }
        bound
    }

    /// The entities whose `component` holds `handle` in a counted field.
    fn entities_referencing(&self, component: ComponentId, handle: HandleId) -> Vec<Entity> {
        let Some(collect) = crate::handle_ledger::collect_fn_for(component) else {
            return Vec::new();
        };
        let mut entities = Vec::new();
        let mut scratch = Vec::new();
        for arch in &self.archetypes {
            let Some(Some(col)) = arch.columns.get(component.0 as usize) else {
                continue;
            };
            for row in 0..col.len() {
                // SAFETY: `row` is in `[0, col.len())` of this column, the
                // contract `handle_audit`'s identical walk relies on.
                let ptr = unsafe { col.get_raw(row) };
                scratch.clear();
                (collect)(ptr, &mut scratch);
                if scratch.contains(&handle) {
                    entities.push(arch.entities[row]);
                }
            }
        }
        entities
    }
}
