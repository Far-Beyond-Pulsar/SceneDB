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
//!   not resolved, off the write path (one of a fixed set of worker threads
//!   runs the kind's loader; more loads than workers wait their turn), and
//!   applies finished loads: each records the payload and its
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
//! - The world keeps each asset's payload (its CPU data) while components
//!   reference it, so a rebuild has something to rebuild from. Payloads no
//!   component references any more stay cached, up to a byte budget
//!   ([`World::set_asset_cache_budget`]): past it, [`World::pump_assets`]
//!   drops the ones referenced longest ago. A kind reports its payloads'
//!   size with [`register_payload_size`]; payloads of a kind that does not
//!   are never evicted by the budget. [`World::evict_unreferenced_assets`]
//!   drops them all.
//!
//! # Budgets, priority and quality tiers
//!
//! [`World::pump_assets`] does all of its work in one call. A frame thread
//! calls [`World::pump_assets_within`] with an [`AssetBudget`] instead: at most
//! that many finished loads are applied, and that many assets bound, per call
//! (or until the time budget is spent); the rest wait in the world and
//! [`AssetPumpReport::pending`] says how many. Loads are not all started at
//! once: at most [`World::set_max_loads_in_flight`] run, chosen nearest first
//! by the score [`World::set_asset_priority`]'s function gives the entities
//! that reference the asset (lower is sooner). A kind registered with
//! [`register_tiered_loader`] loads coarse first: each finished tier is bound
//! (a component sees [`AssetView::tier`] and [`AssetView::complete`]) and the
//! next tier is requested, after every first-tier load, again nearest first.
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
use std::time::{Duration, Instant};

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
    /// The tier of the payload (0 = coarsest; see [`register_tiered_loader`]).
    pub tier: u8,
    /// Whether the payload is the finest tier: no upgrade is coming.
    pub complete: bool,
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

/// One tier of an asset, as a tiered loader returns it.
pub struct TierLoad {
    pub payload: AssetPayload,
    /// No finer tier exists: the asset is complete.
    pub last: bool,
}

/// Loads one tier of an asset of a kind from its source: runs on a worker
/// thread. Tier 0 is the coarsest; the world asks for tier N+1 after tier N
/// finished with `last == false`.
pub type TieredLoadFn = Arc<dyn Fn(&str, u8) -> Result<TierLoad, String> + Send + Sync>;

/// Register the loader for `kind`, replacing any earlier one. The asset is
/// loaded at full quality in one step.
pub fn register_loader(
    kind: &'static str,
    load: impl Fn(&str) -> Result<AssetPayload, String> + Send + Sync + 'static,
) {
    register_tiered_loader(kind, move |source, _| {
        load(source).map(|payload| TierLoad {
            payload,
            last: true,
        })
    });
}

/// Register a loader that produces `kind`'s assets coarse first (low mips, a
/// reduced mesh): the world binds each tier as it arrives and requests the
/// next one while the asset stays referenced. Replaces any earlier loader.
pub fn register_tiered_loader(
    kind: &'static str,
    load: impl Fn(&str, u8) -> Result<TierLoad, String> + Send + Sync + 'static,
) {
    loaders()
        .write()
        .expect("asset loaders poisoned")
        .insert(kind, Arc::new(load));
}

/// How many bytes a loaded payload holds, for the cache budget.
pub type SizeFn = fn(&AssetPayload) -> usize;

/// Register how to measure `kind`'s payloads, replacing any earlier one.
/// The world counts them against its cache budget once no component
/// references them ([`World::set_asset_cache_budget`]).
pub fn register_payload_size(kind: &'static str, size: SizeFn) {
    registries()
        .sizes
        .write()
        .expect("asset sizes poisoned")
        .insert(kind, size);
}

/// Register how components of type `T` take in assets of `kind`, replacing
/// an earlier registration for the same kind and component: `bind`
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
    let mut bindings = bindings().write().expect("asset bindings poisoned");
    let kind_bindings = bindings.entry(kind).or_default();
    // Each statically linked copy of the defining crate registers its own; the
    // component id is shared, so the last one replaces the others.
    let component = component_id::<T>();
    kind_bindings.retain(|existing| (existing.component)() != component);
    kind_bindings.push(binding);
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
    loaders: RwLock<HashMap<&'static str, TieredLoadFn>>,
    bindings: RwLock<HashMap<&'static str, Vec<Binding>>>,
    sizes: RwLock<HashMap<&'static str, SizeFn>>,
    /// The residency cache (see [`crate::residency`]).
    pub(crate) residency: Mutex<crate::residency::State>,
    /// The derived-data cache (see [`crate::derived`]).
    pub(crate) derived: Mutex<crate::derived::State>,
}

pub(crate) fn registries() -> &'static AssetRegistries {
    (crate::runtime::runtime().asset_registries)()
}

fn sources() -> &'static RwLock<HashMap<AssetId, Declared>> {
    &registries().sources
}

fn loaders() -> &'static RwLock<HashMap<&'static str, TieredLoadFn>> {
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
    /// The payload's size, when its kind reports one.
    bytes: usize,
    /// The last pump that saw a component reference it.
    last_used: u64,
    /// The tier `payload` holds.
    tier: u8,
    /// `payload` is the finest tier.
    complete: bool,
    /// A finer tier is loading while the coarser payload stays bound.
    upgrading: bool,
}

enum LoadStart {
    Started,
    Failed,
    Undeclared,
}

struct Completion {
    id: AssetId,
    token: u64,
    tier: u8,
    result: Result<TierLoad, String>,
}

/// How an entity ranks for loading: lower loads sooner (a camera distance,
/// say). `None` ranks last.
pub type AssetPriorityFn = Arc<dyn Fn(&World, Entity) -> Option<f32> + Send + Sync>;

const UNRANKED: f32 = f32::MAX;
/// Ids bound per column walk while a pump has time left.
const BIND_CHUNK: usize = 16;
/// Pumps between rescoring requests that still wait.
const RESCORE_PUMPS: u64 = 30;

/// What one [`World::pump_assets_within`] call may spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssetBudget {
    /// Finished loads applied, and assets bound, per call. A call always
    /// makes progress, so a budget of 0 behaves as 1.
    pub max_assets: usize,
    /// Wall time after which the call stops taking more work, checked
    /// between assets (one asset's dependents are bound together).
    pub time: Option<Duration>,
}

impl AssetBudget {
    /// No limit: [`World::pump_assets`].
    pub const UNLIMITED: AssetBudget = AssetBudget {
        max_assets: usize::MAX,
        time: None,
    };

    /// A frame thread's share: 8 assets or 2 ms.
    pub const FRAME: AssetBudget = AssetBudget {
        max_assets: 8,
        time: Some(Duration::from_millis(2)),
    };
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
    /// Loads started whose completion has not been received yet (stale ones
    /// included).
    outstanding: usize,
    /// Pumps run, for [`Entry::last_used`].
    pumps: u64,
    /// Bytes of unreferenced payloads kept cached.
    budget: usize,
    /// Finished loads received but not applied yet (over budget).
    arrived: Vec<Completion>,
    /// Applied assets whose dependents are not bound yet.
    bind_queue: Vec<AssetId>,
    /// Assets whose next tier should load once no first tier waits.
    upgrades: HashSet<AssetId>,
    /// At most this many loads run at once.
    max_in_flight: usize,
    priority: Option<AssetPriorityFn>,
    scores: HashMap<AssetId, f32>,
    scored_at: u64,
}

/// The default [`World::set_asset_cache_budget`]: 256 MiB of payloads no
/// component references.
pub const DEFAULT_ASSET_CACHE_BUDGET: usize = 256 << 20;

impl Default for AssetTable {
    fn default() -> Self {
        let (sender, receiver) = channel();
        Self {
            entries: HashMap::new(),
            next_token: 0,
            sender,
            receiver: Mutex::new(receiver),
            requested: HashSet::new(),
            outstanding: 0,
            pumps: 0,
            budget: DEFAULT_ASSET_CACHE_BUDGET,
            arrived: Vec::new(),
            bind_queue: Vec::new(),
            upgrades: HashSet::new(),
            max_in_flight: asset_load_workers() * 2,
            priority: None,
            scores: HashMap::new(),
            scored_at: 0,
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
    /// Unreferenced payloads dropped to stay within the cache budget.
    pub evicted: usize,
    /// Work left for later pumps: loads not started, in flight or finished
    /// but not applied, finer tiers not loaded, assets not bound. Zero means
    /// the world is settled.
    pub pending: usize,
}

impl World {
    /// Advance asset resolution without a limit: [`Self::pump_assets_within`]
    /// with [`AssetBudget::UNLIMITED`]. For headless loading and tests; a
    /// frame thread uses the budgeted form.
    pub fn pump_assets(&mut self) -> AssetPumpReport {
        self.pump_assets_within(AssetBudget::UNLIMITED)
    }

    /// Advance asset resolution within `budget`: apply finished loads
    /// (dropping stale ones), start loads for newly referenced or reloaded
    /// assets (nearest first, at most [`Self::set_max_loads_in_flight`] at a
    /// time), and bind the components that reference assets which resolved or
    /// which they started referencing. What the budget leaves undone waits for
    /// the next call; [`AssetPumpReport::pending`] counts it. Cheap when
    /// nothing changed; call it once per frame (or tick).
    pub fn pump_assets_within(&mut self, budget: AssetBudget) -> AssetPumpReport {
        let began = Instant::now();
        let max = budget.max_assets.max(1);
        let out_of_time = |done: usize| {
            done >= 1 && budget.time.is_some_and(|limit| began.elapsed() >= limit)
        };
        let mut report = AssetPumpReport::default();
        self.assets.pumps += 1;

        // Finished loads: take them all off the channel, apply the nearest
        // first while the budget lasts.
        let received: Vec<Completion> = {
            let receiver = self
                .assets
                .receiver
                .lock()
                .expect("asset receiver poisoned");
            receiver.try_iter().collect()
        };
        self.assets.outstanding -= received.len();
        self.assets.arrived.extend(received);
        if self.assets.arrived.len() > 1 {
            // Best last, so `pop` takes it.
            let scores = &self.assets.scores;
            self.assets.arrived.sort_by(|a, b| {
                let key = |c: &Completion| scores.get(&c.id).copied().unwrap_or(UNRANKED);
                key(b).total_cmp(&key(a))
            });
        }
        let mut applied = 0usize;
        while applied < max && !out_of_time(applied) {
            let Some(completion) = self.assets.arrived.pop() else {
                break;
            };
            let id = completion.id;
            let Some(entry) = self.assets.entries.get_mut(&id) else {
                report.stale += 1;
                continue;
            };
            if entry.token != completion.token
                || !(entry.state == AssetState::Loading || entry.upgrading)
            {
                report.stale += 1;
                continue;
            }
            let was_upgrade = entry.upgrading;
            entry.upgrading = false;
            match completion.result {
                Ok(load) => {
                    entry.version += 1;
                    entry.bytes = payload_size(entry.kind, &load.payload);
                    entry.state = AssetState::Ready;
                    entry.payload = Some(load.payload);
                    entry.tier = completion.tier;
                    entry.complete = load.last;
                    if !load.last {
                        self.assets.upgrades.insert(id);
                    }
                }
                Err(message) if was_upgrade => {
                    // The coarse tier stays; there is nothing finer to wait for.
                    tracing::warn!(kind = entry.kind, source = %entry.source, %message, "asset upgrade failed to load");
                    entry.complete = true;
                }
                Err(message) => {
                    entry.version += 1;
                    tracing::warn!(kind = entry.kind, source = %entry.source, %message, "asset failed to load");
                    entry.state = AssetState::Failed(Arc::from(message));
                    entry.payload = None;
                    entry.bytes = 0;
                    entry.complete = true;
                }
            }
            if entry.complete {
                self.assets.scores.remove(&id);
            }
            report.completed += 1;
            applied += 1;
            self.assets.bind_queue.push(id);
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
                    self.assets.bind_queue.push(id);
                }
            }
        }

        self.start_pending_loads(&mut report);

        // Bind, nearest first, in chunks (one column walk per chunk and kind).
        if !self.assets.bind_queue.is_empty() {
            let mut seen = HashSet::new();
            let mut queue: Vec<AssetId> = std::mem::take(&mut self.assets.bind_queue)
                .into_iter()
                .filter(|id| seen.insert(*id))
                .collect();
            let scores = &self.assets.scores;
            queue.sort_by(|a, b| {
                let key = |id: &AssetId| scores.get(id).copied().unwrap_or(UNRANKED);
                key(a).total_cmp(&key(b))
            });
            let mut next = 0usize;
            while next < queue.len() && next < max && !out_of_time(next) {
                let end = (next + BIND_CHUNK).min(queue.len()).min(max);
                let chunk: HashSet<AssetId> = queue[next..end].iter().copied().collect();
                report.bound += self.bind_dependents(&chunk);
                next = end;
            }
            self.assets.bind_queue = queue.split_off(next);
        }

        report.loading = self
            .assets
            .entries
            .values()
            .filter(|entry| entry.state == AssetState::Loading)
            .count();
        report.evicted = self.enforce_asset_cache_budget();
        report.pending = self.assets.outstanding
            + self.assets.arrived.len()
            + self.assets.bind_queue.len()
            + self.assets.requested.len()
            + self.assets.upgrades.len();
        report
    }

    /// Start the nearest requested loads, up to the in-flight cap; finer
    /// tiers only once no first tier waits.
    fn start_pending_loads(&mut self, report: &mut AssetPumpReport) {
        let free = self
            .assets
            .max_in_flight
            .saturating_sub(self.assets.outstanding);
        if free == 0 {
            return;
        }
        let upgrading = self.assets.requested.is_empty();
        let mut candidates: Vec<AssetId> = if upgrading {
            self.assets.upgrades.iter().copied().collect()
        } else {
            self.assets.requested.iter().copied().collect()
        };
        if candidates.is_empty() {
            return;
        }
        if candidates.len() > free {
            self.refresh_scores(&candidates);
            let scores = &self.assets.scores;
            candidates.sort_by(|a, b| {
                let key = |id: &AssetId| scores.get(id).copied().unwrap_or(UNRANKED);
                key(a).total_cmp(&key(b))
            });
            candidates.truncate(free);
        }
        for id in candidates {
            if upgrading {
                self.assets.upgrades.remove(&id);
                let ready = self
                    .assets
                    .entries
                    .get(&id)
                    .is_some_and(|entry| entry.state == AssetState::Ready && !entry.complete);
                // Nothing shows it any more: keep the coarse payload as it is.
                if !ready || self.asset_ref_count(id) == 0 {
                    continue;
                }
                let tier = self.assets.entries[&id].tier + 1;
                if matches!(self.start_load(id, tier), LoadStart::Started) {
                    report.started += 1;
                }
            } else {
                self.assets.requested.remove(&id);
                match self.start_load(id, 0) {
                    LoadStart::Started => report.started += 1,
                    // No loader: failed on the spot, which its dependents see.
                    LoadStart::Failed => self.assets.bind_queue.push(id),
                    LoadStart::Undeclared => {}
                }
            }
        }
    }

    /// Score `ids` for loading order: the lowest score the priority function
    /// gives an entity that references the asset. Scores are kept until the
    /// asset completes and rescored every [`RESCORE_PUMPS`] pumps, since the
    /// camera moves; one walk of each bound component's column serves every
    /// id of its kind.
    fn refresh_scores(&mut self, ids: &[AssetId]) {
        let Some(priority) = self.assets.priority.clone() else {
            return;
        };
        let stale = self.assets.pumps - self.assets.scored_at >= RESCORE_PUMPS;
        let wanted: Vec<AssetId> = ids
            .iter()
            .copied()
            .filter(|id| stale || !self.assets.scores.contains_key(id))
            .collect();
        if wanted.is_empty() {
            return;
        }
        self.assets.scored_at = self.assets.pumps;
        let mut by_kind: HashMap<&'static str, HashSet<HandleId>> = HashMap::new();
        let mut scored: HashMap<AssetId, f32> = wanted.iter().map(|id| (*id, UNRANKED)).collect();
        for id in &wanted {
            if let Some((kind, _)) = id.source() {
                by_kind.entry(kind).or_default().insert(id.handle());
            }
        }
        for (kind, handles) in by_kind {
            let kind_bindings = bindings()
                .read()
                .expect("asset bindings poisoned")
                .get(kind)
                .cloned()
                .unwrap_or_default();
            for binding in &kind_bindings {
                for (handle, entities) in self.entities_referencing((binding.component)(), &handles)
                {
                    let best = entities
                        .iter()
                        .filter_map(|&entity| priority(self, entity))
                        .fold(UNRANKED, f32::min);
                    let slot = scored.entry(AssetId(handle.0)).or_insert(UNRANKED);
                    *slot = slot.min(best);
                }
            }
        }
        self.assets.scores.extend(scored);
    }

    /// Rank loads by `priority`: it scores an entity that references an asset
    /// (lower loads sooner; a camera distance), and an asset takes the lowest
    /// score of its dependents. Without one, loads start in no set order.
    pub fn set_asset_priority(
        &mut self,
        priority: Option<impl Fn(&World, Entity) -> Option<f32> + Send + Sync + 'static>,
    ) {
        self.assets.priority = priority.map(|f| Arc::new(f) as AssetPriorityFn);
        self.assets.scores.clear();
    }

    /// At most `max` loads run at once (default twice the loader threads);
    /// the rest wait, nearest first. At least 1.
    pub fn set_max_loads_in_flight(&mut self, max: usize) {
        self.assets.max_in_flight = max.max(1);
    }


    /// Pump until no load is in flight, or `timeout` passes: for headless
    /// loading and tests. Superseded loads still running count as in
    /// flight, so every load started has finished when this returns `true`.
    /// Returns `false` on timeout.
    pub fn resolve_assets_blocking(&mut self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let report = self.pump_assets();
            // Settled: nothing loading, waiting, unapplied, unbound or left to
            // upgrade (every tier is loaded).
            if report.pending == 0 {
                // One more pump binds anything the last one completed.
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Whether [`Self::pump_assets`] has anything to do: a load in flight,
    /// one requested, or an asset newly referenced. Cheap and `&self`, so a
    /// frame loop can skip taking the world mutably when it is `false`.
    pub fn asset_work_pending(&self) -> bool {
        self.assets.outstanding > 0
            || !self.assets.requested.is_empty()
            || !self.assets.arrived.is_empty()
            || !self.assets.bind_queue.is_empty()
            || !self.assets.upgrades.is_empty()
            || self
                .handle_counts
                .lock()
                .expect("World handle_counts: mutex poisoned")
                .has_touched_assets()
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
            entry.upgrading = false;
        }
        self.assets.upgrades.remove(&id);
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

    /// Keep at most `bytes` of payloads no component references (see
    /// [`register_payload_size`]); the next [`Self::pump_assets`] that has
    /// work drops the ones referenced longest ago until they fit. Payloads
    /// components reference are never evicted. Defaults to
    /// [`DEFAULT_ASSET_CACHE_BUDGET`].
    pub fn set_asset_cache_budget(&mut self, bytes: usize) {
        self.assets.budget = bytes;
    }

    /// The budget [`Self::set_asset_cache_budget`] set.
    pub fn asset_cache_budget(&self) -> usize {
        self.assets.budget
    }

    /// Bytes of cached payloads no component references.
    pub fn unreferenced_asset_bytes(&self) -> usize {
        self.assets
            .entries
            .iter()
            .filter(|(id, _)| self.asset_ref_count(**id) == 0)
            .map(|(_, entry)| entry.bytes)
            .sum()
    }

    /// Note which assets components still reference, then drop unreferenced
    /// payloads, least recently referenced first, until they fit the budget.
    /// Returns how many were dropped.
    fn enforce_asset_cache_budget(&mut self) -> usize {
        let pump = self.assets.pumps;
        let mut unreferenced: Vec<(u64, AssetId, usize)> = Vec::new();
        let mut cached = 0usize;
        let ids: Vec<AssetId> = self.assets.entries.keys().copied().collect();
        for id in ids {
            let referenced = self.asset_ref_count(id) > 0;
            let entry = self.assets.entries.get_mut(&id).expect("listed entry");
            if referenced {
                entry.last_used = pump;
            } else if entry.bytes > 0 && entry.state != AssetState::Loading {
                cached += entry.bytes;
                unreferenced.push((entry.last_used, id, entry.bytes));
            }
        }
        if cached <= self.assets.budget {
            return 0;
        }
        unreferenced.sort_unstable();
        let mut evicted = 0;
        for (_, id, bytes) in unreferenced {
            if cached <= self.assets.budget {
                break;
            }
            self.assets.entries.remove(&id);
            cached -= bytes;
            evicted += 1;
        }
        evicted
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
    fn start_load(&mut self, id: AssetId, tier: u8) -> LoadStart {
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
            bytes: 0,
            last_used: 0,
            tier: 0,
            complete: false,
            upgrading: false,
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
        if tier == 0 {
            entry.state = AssetState::Loading;
            entry.upgrading = false;
            entry.complete = false;
        } else {
            // The coarser payload stays bound while the finer tier loads.
            entry.upgrading = true;
        }
        self.assets.outstanding += 1;
        let sender = self.assets.sender.clone();
        run_load(Box::new(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loader(&source, tier)))
                    .unwrap_or_else(|_| Err("the loader panicked".to_owned()));
            let _ = sender.send(Completion {
                id,
                token,
                tier,
                result,
            });
        }));
        LoadStart::Started
    }

    /// Bind every component that references any of `ids` to its entry.
    ///
    /// One walk of each bound component's column serves every asset of the
    /// kind, so a pump that completes N assets costs one pass over the
    /// entities, not N. The pump runs under the world's write lock: with each
    /// asset scanning every row itself, a level with thousands of assets and
    /// entities froze every reader for seconds.
    fn bind_dependents(&mut self, ids: &HashSet<AssetId>) -> usize {
        let mut by_kind: HashMap<&'static str, Vec<AssetId>> = HashMap::new();
        for &id in ids {
            if let Some(entry) = self.assets.entries.get(&id) {
                by_kind.entry(entry.kind).or_default().push(id);
            }
        }
        let mut bound = 0;
        for (kind, kind_ids) in by_kind {
            let kind_bindings = bindings()
                .read()
                .expect("asset bindings poisoned")
                .get(kind)
                .cloned()
                .unwrap_or_default();
            let wanted: HashSet<HandleId> = kind_ids.iter().map(|id| id.handle()).collect();
            let found: Vec<HashMap<HandleId, Vec<Entity>>> = kind_bindings
                .iter()
                .map(|binding| self.entities_referencing((binding.component)(), &wanted))
                .collect();
            for id in kind_ids {
                let Some(entry) = self.assets.entries.get(&id) else {
                    continue;
                };
                let source = entry.source.clone();
                let state = entry.state.clone();
                let version = entry.version;
                let (tier, complete) = (entry.tier, entry.complete);
                let payload = entry.payload.clone();
                let view = AssetView {
                    id,
                    kind,
                    source: &source,
                    state: &state,
                    version,
                    tier,
                    complete,
                    payload: payload.as_ref(),
                };
                for (binding, found) in kind_bindings.iter().zip(&found) {
                    let Some(entities) = found.get(&id.handle()) else {
                        continue;
                    };
                    for &entity in entities {
                        (binding.apply)(self, entity, &view);
                        bound += 1;
                    }
                }
            }
        }
        bound
    }

    /// The entities whose `component` holds each of `wanted` in a counted
    /// field, found in one walk of the column.
    fn entities_referencing(
        &self,
        component: ComponentId,
        wanted: &HashSet<HandleId>,
    ) -> HashMap<HandleId, Vec<Entity>> {
        let mut found: HashMap<HandleId, Vec<Entity>> = HashMap::new();
        let Some(collect) = crate::handle_ledger::collect_fn_for(component) else {
            return found;
        };
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
                let entity = arch.entities[row];
                for handle in scratch.iter().filter(|h| wanted.contains(*h)) {
                    let list = found.entry(*handle).or_default();
                    if list.last() != Some(&entity) {
                        list.push(entity);
                    }
                }
            }
        }
        found
    }
}

/// `payload`'s size as its kind reports it, or 0.
fn payload_size(kind: &str, payload: &AssetPayload) -> usize {
    registries()
        .sizes
        .read()
        .expect("asset sizes poisoned")
        .get(kind)
        .map_or(0, |size| size(payload))
}

pub(crate) type LoadJob = Box<dyn FnOnce() + Send>;

/// How many threads run loaders: the machine's parallelism, at least 4, so
/// one slow load does not hold up the rest. Loads beyond it queue.
pub fn asset_load_workers() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .max(4)
}

/// Queue `job` on the loader threads, starting them on first use.
pub(crate) fn run_load(job: LoadJob) {
    static JOBS: std::sync::OnceLock<Sender<LoadJob>> = std::sync::OnceLock::new();
    let jobs = JOBS.get_or_init(|| {
        let (sender, receiver) = channel::<LoadJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..asset_load_workers() {
            let receiver = receiver.clone();
            std::thread::Builder::new()
                .name(format!("scenedb-asset-load-{index}"))
                .spawn(move || loop {
                    let job = receiver.lock().expect("asset load queue poisoned").recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                })
                .expect("spawn asset loader thread");
        }
        sender
    });
    jobs.send(job).expect("asset loader threads are running");
}
