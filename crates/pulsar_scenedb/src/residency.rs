//! Budgeted residency for data that can be rebuilt (the RAM tier of an asset).
//!
//! A decoded image, a parsed mesh or a cooked chunk is *derived* from a source
//! that stays on disk. Holding every such payload for as long as a component
//! references it makes memory grow with the number of assets, not with what is
//! in use. [`Residency<T>`] holds the value in a process-wide, byte-budgeted,
//! least-recently-used cache instead:
//!
//! - [`Residency::get`] returns the value, rebuilding it from the source the
//!   owner supplied if it was dropped to make room. Concurrent callers share
//!   one rebuild (single flight): the first builds, the rest wait for it;
//! - [`Residency::prefetch`] starts that rebuild on the asset loader threads
//!   and returns a handle, so a caller that knows it will need the value soon
//!   can ask early and collect later, and several rebuilds run in parallel;
//! - the cache never drops a value a caller is using (they hold an `Arc`); it
//!   only drops *its own* reference, so a value in use outlives eviction and is
//!   counted against the budget again if it is touched;
//! - the budget ([`set_residency_budget`]) bounds what the cache itself keeps.
//!   One value larger than the whole budget is still kept while it is the
//!   most recently used, so a single huge asset never thrashes.
//!
//! Nothing here knows what `T` is. A kind decides how to rebuild its data and
//! how big it is (`size`); the world's asset layer and the tier engine decide
//! what to ask for.
//!
//! # Recency
//!
//! Recency is a monotonic counter. Entries are also indexed by it in an
//! ordered map, so the least recently used value is the first key: choosing a
//! victim, touching a value and inserting one are all `O(log n)`, whatever the
//! number of resident values.
//!
//! # Locking
//!
//! Three kinds of lock exist: the cache's, a value's slot (the `Arc<T>`), and a
//! value's flight (the rebuild in progress). The cache's is only ever taken
//! *before* a slot's, never while holding one; a flight's lock is never held
//! while taking the cache's or a slot's except for the brief
//! flight-then-slot check in `join`. Values are always dropped after every
//! lock is released, so neither a drop that re-enters the cache nor two
//! threads taking locks in opposite orders can deadlock.
//!
//! A prefetch queued on the loader threads can be *claimed* by a `get` that
//! reaches the value first, which then runs the rebuild itself. A loader
//! thread that calls `get` therefore never waits on a queue it is blocking.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, Weak};

/// Default cache budget: 512 MiB of rebuildable data kept in memory.
pub const DEFAULT_RESIDENCY_BUDGET: usize = 512 << 20;

/// What the cache can do to a value without knowing its type.
trait Slot: Send + Sync {
    /// Drop the cached value (if any); the value is returned so the caller can
    /// release it after unlocking.
    fn take(&self) -> Option<Box<dyn std::any::Any + Send + Sync>>;
}

struct Resident {
    last_use: u64,
    bytes: usize,
    slot: Weak<dyn Slot>,
}

/// The cache: process-wide, reached through the asset registries so every
/// statically linked copy of this crate shares one (see [`crate::runtime`]).
pub(crate) struct State {
    budget: usize,
    resident_bytes: usize,
    tick: u64,
    next_id: u64,
    entries: HashMap<u64, Resident>,
    /// `last_use` tick -> value id. Ticks are unique, so this is a total order
    /// over the resident values by recency; the first key is the victim.
    order: BTreeMap<u64, u64>,
    evictions: u64,
    rebuilds: u64,
    prefetches: u64,
    joined: u64,
}

impl Default for State {
    fn default() -> Self {
        Self {
            budget: DEFAULT_RESIDENCY_BUDGET,
            resident_bytes: 0,
            tick: 0,
            next_id: 1,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            evictions: 0,
            rebuilds: 0,
            prefetches: 0,
            joined: 0,
        }
    }
}

impl State {
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// Mark `id` as just used. A value the cache no longer lists is left alone.
    fn touch(&mut self, id: u64) {
        let tick = self.next_tick();
        if let Some(entry) = self.entries.get_mut(&id) {
            self.order.remove(&entry.last_use);
            entry.last_use = tick;
            self.order.insert(tick, id);
        }
    }

    /// List `id` as resident, or refresh it if it already is.
    fn insert_or_touch(&mut self, id: u64, bytes: usize, slot: Weak<dyn Slot>) {
        if self.entries.contains_key(&id) {
            self.touch(id);
            return;
        }
        let tick = self.next_tick();
        self.entries.insert(id, Resident { last_use: tick, bytes, slot });
        self.order.insert(tick, id);
        self.resident_bytes += bytes;
    }

    /// Stop listing `id`, returning what the cache knew of it.
    fn remove(&mut self, id: u64) -> Option<Resident> {
        let entry = self.entries.remove(&id)?;
        self.order.remove(&entry.last_use);
        self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
        Some(entry)
    }

    /// The least recently used value other than `except`.
    fn victim(&self, except: u64) -> Option<u64> {
        // At most one key is skipped, so this reads one or two map entries.
        self.order.values().copied().find(|id| *id != except)
    }
}

fn cache() -> &'static Mutex<State> {
    &crate::assets::registries().residency
}

fn lock() -> std::sync::MutexGuard<'static, State> {
    cache().lock().unwrap_or_else(|poison| poison.into_inner())
}

/// A snapshot of the cache, for diagnostics (Mission Control).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub budget: usize,
    /// Bytes the cache itself is keeping resident.
    pub resident_bytes: usize,
    /// Values currently resident.
    pub resident_values: usize,
    /// Values dropped to stay within the budget.
    pub evictions: u64,
    /// Values rebuilt from their source after being dropped (or first built).
    pub rebuilds: u64,
    /// Rebuilds started ahead of need with [`Residency::prefetch`].
    pub prefetches: u64,
    /// Callers that joined a rebuild already in flight instead of starting one.
    pub joined: u64,
}

pub fn residency_stats() -> ResidencyStats {
    let state = lock();
    ResidencyStats {
        budget: state.budget,
        resident_bytes: state.resident_bytes,
        resident_values: state.entries.len(),
        evictions: state.evictions,
        rebuilds: state.rebuilds,
        prefetches: state.prefetches,
        joined: state.joined,
    }
}

pub fn residency_budget() -> usize {
    lock().budget
}

/// Set the budget. Takes effect at the next insertion or touch; lowering it
/// does not drop anything by itself.
pub fn set_residency_budget(bytes: usize) {
    lock().budget = bytes;
}

type RebuildFn<T> = dyn Fn() -> Result<T, String> + Send + Sync;
type SizeFn<T> = dyn Fn(&T) -> usize + Send + Sync;

enum FlightState<T> {
    /// Created by a prefetch and waiting in the loader queue: any caller may
    /// claim it.
    Queued,
    Running,
    Done(Result<Arc<T>, String>),
}

/// One rebuild in progress, shared by everyone who wants the value meanwhile.
struct Flight<T> {
    state: Mutex<FlightState<T>>,
    done: Condvar,
}

impl<T> Flight<T> {
    fn new(state: FlightState<T>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(state),
            done: Condvar::new(),
        })
    }

    fn wait(&self) -> Result<Arc<T>, String> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let FlightState::Done(result) = &*state {
                return result.clone();
            }
            state = self.done.wait(state).unwrap_or_else(|p| p.into_inner());
        }
    }

    fn is_done(&self) -> bool {
        matches!(
            *self.state.lock().unwrap_or_else(|p| p.into_inner()),
            FlightState::Done(_)
        )
    }
}

struct Inner<T: Send + Sync + 'static> {
    id: u64,
    value: Mutex<Option<Arc<T>>>,
    flight: Mutex<Option<Arc<Flight<T>>>>,
    size: Box<SizeFn<T>>,
    rebuild: Box<RebuildFn<T>>,
}

impl<T: Send + Sync + 'static> Slot for Inner<T> {
    fn take(&self) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        let taken = self
            .value
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()?;
        Some(Box::new(taken))
    }
}

impl<T: Send + Sync + 'static> Drop for Inner<T> {
    fn drop(&mut self) {
        // The owner went away: stop counting its bytes. (Nothing holds the
        // cache's lock while dropping an `Inner`, see the module docs.)
        lock().remove(self.id);
    }
}

/// Publishes the outcome of a rebuild, and does so even if the rebuild
/// panics, so nobody waits forever on a flight whose builder is gone.
struct FlightGuard<'a, T: Send + Sync + 'static> {
    inner: &'a Inner<T>,
    flight: &'a Arc<Flight<T>>,
    finished: bool,
}

impl<T: Send + Sync + 'static> FlightGuard<'_, T> {
    fn finish(mut self, result: Result<Arc<T>, String>) {
        self.publish(result);
        self.finished = true;
    }

    fn publish(&self, result: Result<Arc<T>, String>) {
        // Clear the slot for new flights first, so a caller arriving after the
        // value is installed finds the value and not a finished flight.
        *self.inner.flight.lock().unwrap_or_else(|p| p.into_inner()) = None;
        *self.flight.state.lock().unwrap_or_else(|p| p.into_inner()) = FlightState::Done(result);
        self.flight.done.notify_all();
    }
}

impl<T: Send + Sync + 'static> Drop for FlightGuard<'_, T> {
    fn drop(&mut self) {
        if !self.finished {
            self.publish(Err("rebuild panicked".into()));
        }
    }
}

/// Rebuildable data kept under the residency budget. Cloning is cheap and
/// shares the same cached value.
pub struct Residency<T: Send + Sync + 'static> {
    inner: Arc<Inner<T>>,
}

impl<T: Send + Sync + 'static> Clone for Residency<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + Sync + 'static> std::fmt::Debug for Residency<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Residency")
            .field("resident", &self.is_resident())
            .finish()
    }
}

/// A rebuild started with [`Residency::prefetch`].
pub struct Prefetch<T: Send + Sync + 'static> {
    residency: Residency<T>,
    flight: Option<Arc<Flight<T>>>,
}

impl<T: Send + Sync + 'static> Prefetch<T> {
    /// Whether the value is ready, so `wait` returns at once.
    pub fn is_ready(&self) -> bool {
        match &self.flight {
            Some(flight) => flight.is_done(),
            None => true,
        }
    }

    /// The value, waiting for the rebuild if it is still running. If it has
    /// not started (it is queued behind other loads), this thread runs it.
    pub fn wait(self) -> Result<Arc<T>, String> {
        match self.flight {
            Some(flight) => {
                self.residency.run_flight(&flight);
                flight.wait()
            }
            None => self.residency.get(),
        }
    }
}

/// What a caller found when it went to rebuild.
enum Join<T> {
    Resident(Arc<T>),
    Flight(Arc<Flight<T>>, bool),
}

impl<T: Send + Sync + 'static> Residency<T> {
    /// Data that is built on first [`Self::get`] by `rebuild`, whose resident
    /// size is `size`.
    pub fn lazy(
        size: impl Fn(&T) -> usize + Send + Sync + 'static,
        rebuild: impl Fn() -> Result<T, String> + Send + Sync + 'static,
    ) -> Self {
        let id = {
            let mut state = lock();
            let id = state.next_id;
            state.next_id += 1;
            id
        };
        Self {
            inner: Arc::new(Inner {
                id,
                value: Mutex::new(None),
                flight: Mutex::new(None),
                size: Box::new(size),
                rebuild: Box::new(rebuild),
            }),
        }
    }

    /// Data that is already built (`value`), resident now, and rebuilt by
    /// `rebuild` if it is ever dropped to make room.
    pub fn with_value(
        value: T,
        size: impl Fn(&T) -> usize + Send + Sync + 'static,
        rebuild: impl Fn() -> Result<T, String> + Send + Sync + 'static,
    ) -> Self {
        let this = Self::lazy(size, rebuild);
        this.install(Arc::new(value), false);
        this
    }

    /// Whether the cache is holding the value right now.
    pub fn is_resident(&self) -> bool {
        self.resident_value().is_some()
    }

    fn resident_value(&self) -> Option<Arc<T>> {
        self.inner
            .value
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    /// The value, rebuilt from its source if it was dropped. The returned
    /// `Arc` keeps it alive for as long as the caller uses it, whatever the
    /// cache does meanwhile. Callers that arrive while a rebuild is in flight
    /// wait for it instead of starting another.
    pub fn get(&self) -> Result<Arc<T>, String> {
        if let Some(value) = self.resident_value() {
            lock().touch(self.inner.id);
            return Ok(value);
        }
        match self.join() {
            Join::Resident(value) => {
                lock().touch(self.inner.id);
                Ok(value)
            }
            Join::Flight(flight, created) => {
                if !created {
                    lock().joined += 1;
                }
                // Builds it here unless somebody is already building it, in
                // which case this returns at once and `wait` waits.
                self.run_flight(&flight);
                flight.wait()
            }
        }
    }

    /// Start rebuilding the value on the asset loader threads, if it is not
    /// resident and not already being rebuilt. Collect it with
    /// [`Prefetch::wait`]; a [`Self::get`] meanwhile joins the same rebuild.
    /// Several prefetches run in parallel, up to the loader thread count.
    pub fn prefetch(&self) -> Prefetch<T> {
        if self.is_resident() {
            return Prefetch {
                residency: self.clone(),
                flight: None,
            };
        }
        match self.join() {
            Join::Resident(_) => Prefetch {
                residency: self.clone(),
                flight: None,
            },
            Join::Flight(flight, created) => {
                if created {
                    lock().prefetches += 1;
                    let this = self.clone();
                    let queued = flight.clone();
                    crate::assets::run_load(Box::new(move || this.run_flight(&queued)));
                }
                Prefetch {
                    residency: self.clone(),
                    flight: Some(flight),
                }
            }
        }
    }

    /// The rebuild in flight, a new one if there is none, or the value if it
    /// appeared meanwhile.
    fn join(&self) -> Join<T> {
        let mut flight = self
            .inner
            .flight
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(existing) = flight.as_ref() {
            return Join::Flight(existing.clone(), false);
        }
        // A builder installs the value before it clears the flight, so a value
        // that is here now means there is nothing to rebuild.
        if let Some(value) = self.resident_value() {
            return Join::Resident(value);
        }
        let created = Flight::new(FlightState::Queued);
        *flight = Some(created.clone());
        Join::Flight(created, true)
    }

    /// Run `flight`'s rebuild unless another thread already has (or has
    /// finished it). The first caller to find it `Queued` claims it.
    fn run_flight(&self, flight: &Arc<Flight<T>>) {
        {
            let mut state = flight.state.lock().unwrap_or_else(|p| p.into_inner());
            match *state {
                FlightState::Queued => *state = FlightState::Running,
                _ => return,
            }
        }
        let guard = FlightGuard {
            inner: &self.inner,
            flight,
            finished: false,
        };
        // Rebuild with no lock held: it can take a while (a decode).
        match (self.inner.rebuild)() {
            Ok(rebuilt) => {
                let value = self.install(Arc::new(rebuilt), true);
                guard.finish(Ok(value));
            }
            Err(error) => guard.finish(Err(error)),
        }
    }

    /// Make `value` the resident value (unless another thread installed one
    /// first, which wins), account for it, and evict others over budget.
    fn install(&self, value: Arc<T>, rebuilt: bool) -> Arc<T> {
        let value = {
            let mut slot = self
                .inner
                .value
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            match slot.as_ref() {
                Some(existing) => existing.clone(),
                None => {
                    *slot = Some(value.clone());
                    value
                }
            }
        };
        let bytes = (self.inner.size)(&value);
        let slot: Arc<dyn Slot> = self.inner.clone();

        let mut dropped: Vec<Box<dyn std::any::Any + Send + Sync>> = Vec::new();
        let mut keep_alive: Vec<Arc<dyn Slot>> = Vec::new();
        {
            let mut state = lock();
            if rebuilt {
                state.rebuilds += 1;
            }
            state.insert_or_touch(self.inner.id, bytes, Arc::downgrade(&slot));
            // Evict least-recently-used others until within budget; the value
            // just used stays even if it alone exceeds the budget.
            while state.resident_bytes > state.budget {
                let Some(victim) = state.victim(self.inner.id) else {
                    break;
                };
                let entry = state.remove(victim).expect("victim is listed");
                state.evictions += 1;
                if let Some(slot) = entry.slot.upgrade() {
                    if let Some(taken) = slot.take() {
                        dropped.push(taken);
                    }
                    // Dropping the last handle to an owner would re-enter the
                    // cache; hold it until the lock is released.
                    keep_alive.push(slot);
                }
            }
        }
        // Values and owners are released here, with no lock held.
        drop(dropped);
        drop(keep_alive);
        value
    }
}
