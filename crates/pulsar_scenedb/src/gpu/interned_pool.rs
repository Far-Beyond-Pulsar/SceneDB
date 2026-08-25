//! Content-addressed, reference-counted sharing for `Vec<T>`-typed `#[gpu]`
//! fields (design goal: "Arc-semantics asset sharing inside SceneDB" —
//! Pulsar-Native#632). This is the actual dedup mechanism: ten components
//! all naming the same mesh asset must upload and retain that geometry
//! ONCE, not once per component, and free it the instant the last one stops
//! referencing it.
//!
//! # Why this wraps [`VarLenGpuPool`] instead of replacing it
//!
//! [`VarLenGpuPool<T>`] already solves "a growable, suballocated GPU buffer
//! shared across many rows, freeing and reusing space as rows come and go"
//! — the byte-range allocator ([`super::freelist::RangeList`]) and the
//! grow-and-copy backing buffer ([`super::dynamic_buffer::DynamicGpuBuffer`])
//! are exactly the mechanics an interning layer needs too. The ONLY thing
//! missing is keying allocations by CONTENT IDENTITY instead of by row, so
//! this module adds that as a thin layer on top: [`InternedVarLenPool<T>`]
//! owns a `VarLenGpuPool<T>` unchanged and adds an id → allocation intern
//! table plus a per-row "what id is resident here" shadow. Every actual
//! byte of storage still flows through the pool's own proven
//! alloc/write/free path — this module never touches a `wgpu::Buffer`
//! directly.
//!
//! # Contract
//!
//! - **Identity source**: a value's [`crate::handle_ledger::ContentAddressed::
//!   content_id`] — this module (and this crate) never hashes bytes or
//!   parses any domain format. [`HandleId::ZERO`] opts a row out of
//!   interning entirely (falls back to a private, unshared allocation for
//!   that write — the zero-value convention, extended to this feature).
//! - **First writer wins**: the FIRST `upsert_row` call for a given id
//!   allocates and uploads; every subsequent call for the SAME id, from any
//!   row, is a refcount bump and a byte-for-byte no-op upload. This crate
//!   never re-uploads to "fix" a later writer's differing bytes.
//! - **Collision policy**: a later `upsert_row` for an id ALREADY interned,
//!   whose bytes hash to a different [`fingerprint`] than what's on file,
//!   is a same-id-different-bytes collision — first-writer-wins holds (the
//!   ORIGINAL resident bytes are never touched), and a `tracing::warn!`
//!   fires exactly once per id (never once per occurrence — a caller stuck
//!   in a bug loop must not flood logs). This is a defensive backstop, not
//!   a normal-path feature: legitimate content-addressed ids are expected
//!   to change whenever bytes do (see `ContentAddressed`'s own doc), so a
//!   collision here means an upstream id-computation bug, not a supported
//!   "update in place" workflow — there is no update-in-place API, on
//!   purpose (never silently alias old readers onto new bytes).
//! - **Deterministic drop**: refcount 1→0 frees the range back to the
//!   underlying pool's freelist IMMEDIATELY, synchronously, inside whichever
//!   `upsert_row`/`release_row` call caused it — no deferred GC pass, no
//!   epoch to wait out. The freed range becomes available to the very next
//!   allocation.
//! - **Generation guard, presence-based**: [`InternedVarLenPool::resolve`]
//!   returns `None` once an id's refcount reaches zero. `HandleId`s are
//!   never reassigned to different content in this design (see the
//!   collision policy above), so "is this id still resident" is a complete
//!   staleness check on its own — no wrapping generation counter needed on
//!   top of it (contrast `gpu::generation`'s slot-reuse guard, which exists
//!   specifically because ITS keys — GPU slot indices — legitimately DO get
//!   reassigned to different content over a `World`'s lifetime; content ids
//!   structurally never do).
//!
//! # What is NOT covered (deliberately)
//!
//! No knowledge of what `T` represents (a vertex, an index, anything else)
//! — this module compares opaque [`HandleId`]s and moves opaque `T` bytes,
//! nothing more. No cross-pool coordination: `StaticMeshComponent`'s
//! `vertices`/`indices` fields intern into TWO separate
//! `InternedVarLenPool`s (one per buffer key, same as the plain var-len
//! path already splits vertex/index storage) that happen to be keyed by the
//! same content id — this module has no notion that they're "the same
//! asset", and doesn't need one; each pool's own refcount is independently
//! correct.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use super::var_len_pool::VarLenGpuPool;
use super::VarLenHandle;
use crate::handle_ledger::HandleId;
use crate::page::Pod;

/// One interned id's bookkeeping: where its bytes live in the underlying
/// pool, how many rows currently reference it, and a cheap fingerprint of
/// what was actually uploaded (for the collision diagnostic — see the
/// module doc's "Collision policy"; NOT a retained copy of the payload
/// itself, matching `GeometryArena`'s "no CPU copy" precedent).
struct InternEntry {
    range: VarLenHandle,
    refcount: u64,
    fingerprint: u64,
}

#[derive(Default)]
struct InternState {
    entries: HashMap<HandleId, InternEntry>,
    /// `entity.index()` -> the content id currently resident at that row
    /// (`HandleId::ZERO` = "not interned" -- see `private_by_row` below for
    /// what that row actually holds instead). Grown lazily, exactly like
    /// every other row-indexed CPU shadow in this crate (`GenerationMirror`'s
    /// `gpu_mirrored_rows`, `DirtyTrackedSceneBuffer`'s shadow column).
    id_by_row: Vec<HandleId>,
    /// A row whose content id is `HandleId::ZERO` (no identity available --
    /// e.g. no project path to resolve against, or a genuinely-empty
    /// `mesh_asset`) still needs its data written SOMEWHERE if `data` is
    /// non-empty: zero means "not shareable", not "discard this write".
    /// Such a row gets an ordinary PRIVATE allocation from the same
    /// underlying pool, tracked here by row instead of by id (zero is not a
    /// usable map key -- every private row would collide on it). Meaningful
    /// only where the corresponding `id_by_row` entry is zero; a row that
    /// currently holds an interned id has `VarLenHandle::default()` here.
    private_by_row: Vec<VarLenHandle>,
    /// Ids that have already fired the collision diagnostic — see the
    /// module doc. Never shrinks; see `HandleCounts::warned_underflow`'s
    /// doc for the identical "not worth the complexity" reasoning.
    warned_collision: HashSet<HandleId>,
}

impl InternState {
    fn id_at_row(&self, row: usize) -> HandleId {
        self.id_by_row.get(row).copied().unwrap_or(HandleId::ZERO)
    }

    fn private_at_row(&self, row: usize) -> VarLenHandle {
        self.private_by_row.get(row).copied().unwrap_or_default()
    }

    /// Sets both shadows for `row` together — always called as a pair (see
    /// `upsert_row`/`release_row`), so the two Vecs never drift out of
    /// length-sync with each other.
    fn set_row(&mut self, row: usize, id: HandleId, private: VarLenHandle) {
        if row >= self.id_by_row.len() {
            self.id_by_row.resize(row + 1, HandleId::ZERO);
            self.private_by_row.resize(row + 1, VarLenHandle::default());
        }
        self.id_by_row[row] = id;
        self.private_by_row[row] = private;
    }
}

/// Content-addressed wrapper over a [`VarLenGpuPool<T>`] — see the module
/// doc. One instance per (struct, field) buffer key, exactly like the plain
/// pool it wraps; registered via
/// [`super::SceneGpuStore::register_interned_var_len_gpu_pool`].
pub struct InternedVarLenPool<T: Pod> {
    pool: Arc<VarLenGpuPool<T>>,
    state: RwLock<InternState>,
}

/// Cheap, non-cryptographic fingerprint of a `Pod` slice's bytes — good
/// enough to catch "these two uploads under the same id clearly aren't the
/// same content" (the collision diagnostic), not a security property. Reuses
/// `ahash` (already a workspace dependency, see `AHashMap`'s use elsewhere
/// in this crate) rather than adding a new hashing dependency for a
/// diagnostic-only check.
fn fingerprint<T: Pod>(data: &[T]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    super::as_bytes(data).hash(&mut hasher);
    hasher.finish()
}

impl<T: Pod + Send + Sync + 'static> InternedVarLenPool<T> {
    pub fn new(pool: Arc<VarLenGpuPool<T>>) -> Self {
        Self { pool, state: RwLock::new(InternState::default()) }
    }

    /// The underlying plain pool — for buffer-binding call sites that only
    /// need the `wgpu::Buffer` (e.g. Helio binding shared geometry directly
    /// as a vertex/index buffer), unaware of and unaffected by interning.
    pub fn underlying(&self) -> &Arc<VarLenGpuPool<T>> {
        &self.pool
    }

    /// Write `row`'s content: `new_id` is the value's current
    /// `ContentAddressed::content_id()`. Returns the [`VarLenHandle`] to
    /// store in the row-indexed handle-table column — the SAME shared
    /// `{offset, count}` for every row currently referencing `new_id`, so
    /// existing row-indexed readers (e.g. the derive's `..._gpu_handle`
    /// accessors) see the dedup transparently, with no API change on their
    /// side.
    ///
    /// - `new_id == HandleId::ZERO`: NOT shareable, but `data` still gets
    ///   written if non-empty — zero means "no identity available" (an
    ///   unresolvable `mesh_asset`, no project path to resolve against,
    ///   etc.), not "discard this write". Falls back to an ordinary
    ///   PRIVATE, per-row allocation via the underlying pool's own
    ///   free-then-write cycle — exactly what a plain (non-interned)
    ///   `Vec<T>` `#[gpu]` field already does, so a row that can't supply
    ///   an id never silently loses its data. If the row previously held
    ///   an INTERNED id, that gets released first (switching from shared to
    ///   private).
    /// - `new_id` unchanged and nonzero: true no-op, not even a pool call —
    ///   matches the swap-equality rule `handle_ledger::report_value_swap`
    ///   already established for plain `HandleId` fields, extended here.
    ///   (Two ZERO writes are deliberately NOT treated as equal — zero is a
    ///   shared "no identity" sentinel, not a real repeated identity, so
    ///   each zero-id write goes through the private path independently;
    ///   see the private-write branch above.)
    /// - Otherwise (a genuine id change, old possibly zero or a different
    ///   nonzero id): releases whatever the row held before, then either
    ///   bumps an existing entry's refcount (byte-for-byte no-op upload; a
    ///   fingerprint mismatch fires the once-per-id collision diagnostic
    ///   and leaves the ORIGINAL resident bytes alone) or allocates+writes
    ///   fresh via the underlying pool.
    pub fn upsert_row(&self, queue: &wgpu::Queue, row: u32, new_id: HandleId, data: &[T]) -> VarLenHandle {
        let row = row as usize;
        let mut state = self.state.write().expect("InternedVarLenPool lock poisoned");
        let old_id = state.id_at_row(row);

        if new_id.is_zero() {
            if !old_id.is_zero() {
                Self::decref_locked(&mut state, &self.pool, old_id);
            }
            let old_private = state.private_at_row(row);
            let range = self
                .pool
                .write_var_row(queue, old_private, data)
                .expect("interned var-len pool's underlying VarLenGpuPool never has a capacity ceiling");
            state.set_row(row, HandleId::ZERO, range);
            return range;
        }

        if old_id == new_id {
            return state.entries.get(&new_id).map(|e| e.range).unwrap_or_default();
        }

        if old_id.is_zero() {
            // Row is switching from a private allocation to an interned
            // one -- free the private range (a no-op if it was never
            // written, `VarLenHandle::default()`'s `count == 0`).
            let old_private = state.private_at_row(row);
            self.pool.free_handle(old_private);
        } else {
            Self::decref_locked(&mut state, &self.pool, old_id);
        }

        let new_range = if let Some((range, mismatch)) = {
            // Named-field destructure (not `state.entries.get_mut(..)`
            // directly): gives `entries`/`warned_collision` disjoint local
            // `&mut` bindings so the borrow checker can see the two field
            // touches below don't alias, without needing a second lock
            // round-trip.
            let InternState { entries, warned_collision, .. } = &mut *state;
            entries.get_mut(&new_id).map(|entry| {
                entry.refcount += 1;
                let fp = fingerprint(data);
                let mismatch = fp != entry.fingerprint && warned_collision.insert(new_id);
                (entry.range, mismatch)
            })
        } {
            if mismatch {
                tracing::warn!(
                    "InternedVarLenPool: content id {new_id:?} uploaded with DIFFERENT bytes than \
                     its first writer -- first-writer-wins, original resident bytes kept unchanged. \
                     This means an upstream content-id computation produced the same id for two \
                     different payloads (a real bug there), not a supported update-in-place path. \
                     This diagnostic fires once per id."
                );
            }
            range
        } else {
            let range = self
                .pool
                .write_var_row(queue, VarLenHandle::default(), data)
                .expect("interned var-len pool grows transparently, same no-capacity-ceiling contract as VarLenGpuPool itself");
            state.entries.insert(new_id, InternEntry { range, refcount: 1, fingerprint: fingerprint(data) });
            range
        };

        state.set_row(row, new_id, VarLenHandle::default());
        new_range
    }

    /// Despawn/removal path: release whatever id `row` currently holds (if
    /// any), decref/free-at-zero, and clear the row's shadow entry. Safe to
    /// call on a row that was never written (no-op) — mirrors
    /// `VarLenGpuPool::free_handle`'s own `count == 0` tolerance.
    pub fn release_row(&self, row: u32) {
        let row = row as usize;
        let mut state = self.state.write().expect("InternedVarLenPool lock poisoned");
        let old_id = state.id_at_row(row);
        if old_id.is_zero() {
            // A private (non-interned) allocation, if any -- free it the
            // same way a plain `VarLenGpuPool` consumer's despawn path
            // would (`VarLenHandle::default()`'s `count == 0` makes this a
            // harmless no-op for a row that was never written).
            let old_private = state.private_at_row(row);
            self.pool.free_handle(old_private);
        } else {
            Self::decref_locked(&mut state, &self.pool, old_id);
        }
        if row < state.id_by_row.len() {
            state.set_row(row, HandleId::ZERO, VarLenHandle::default());
        }
    }

    fn decref_locked(state: &mut InternState, pool: &VarLenGpuPool<T>, id: HandleId) {
        let Some(entry) = state.entries.get_mut(&id) else {
            // Already gone (or never interned) -- tolerated, same
            // saturate-at-floor philosophy as `HandleCounts::release`.
            return;
        };
        entry.refcount = entry.refcount.saturating_sub(1);
        if entry.refcount == 0 {
            let range = entry.range;
            state.entries.remove(&id);
            pool.free_handle(range);
        }
    }

    /// Generation-guarded lookup by raw id — for a consumer that holds a
    /// [`HandleId`] directly rather than an entity row (see the module
    /// doc's "Generation guard, presence-based"). `None` once the id's
    /// refcount has reached zero and its range was freed.
    pub fn resolve(&self, id: HandleId) -> Option<(u32, u32)> {
        let state = self.state.read().expect("InternedVarLenPool lock poisoned");
        state.entries.get(&id).map(|e| (e.range.offset, e.range.count))
    }

    /// Every currently-interned id, its live refcount, and its resident
    /// byte size — the audit surface the mission's "leak loop" adversarial
    /// test and the resident-bytes-invariant benchmark both read directly.
    /// O(unique interned ids); not on any hot path.
    pub fn audit(&self) -> Vec<(HandleId, u64, u64)> {
        let state = self.state.read().expect("InternedVarLenPool lock poisoned");
        let elem_size = std::mem::size_of::<T>() as u64;
        state
            .entries
            .iter()
            .map(|(&id, e)| (id, e.refcount, e.range.count as u64 * elem_size))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    /// One process-lifetime device/queue, shared by every test in this
    /// module instead of each test creating its own (the pattern the rest
    /// of this crate's GPU test files use). `wgpu::Device`/`Queue` are
    /// `Send + Sync` by design specifically so concurrent callers can share
    /// one — each test still creates its OWN `InternedVarLenPool`/buffers
    /// on top, so there is no cross-test state to contaminate. This exists
    /// because this module alone adds nine GPU-context-touching tests to
    /// the lib test binary; on constrained hardware/driver combinations,
    /// `cargo test`'s default parallelism creating that many concurrent
    /// `wgpu::Instance::request_adapter`/`request_device` calls (on top of
    /// every OTHER GPU test file's own fresh devices, all in the same
    /// process) can stall badly enough to look hung rather than merely
    /// slow — sharing one device here removes this module's contribution
    /// to that pressure without changing what each test actually proves.
    fn test_device() -> (StdArc<wgpu::Device>, StdArc<wgpu::Queue>) {
        static SHARED: std::sync::OnceLock<(StdArc<wgpu::Device>, StdArc<wgpu::Queue>)> = std::sync::OnceLock::new();
        SHARED.get_or_init(create_device).clone()
    }

    fn create_device() -> (StdArc<wgpu::Device>, StdArc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("interned-var-len-pool-test"),
            ..Default::default()
        }))
        .expect("device");
        (StdArc::new(device), StdArc::new(queue))
    }

    fn pool(device: &StdArc<wgpu::Device>) -> InternedVarLenPool<u32> {
        InternedVarLenPool::new(Arc::new(VarLenGpuPool::new(StdArc::clone(device), "test", 8)))
    }

    #[test]
    fn two_rows_same_id_share_one_allocation() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let id = HandleId(1);

        let h1 = p.upsert_row(&queue, 0, id, &[1, 2, 3]);
        let h2 = p.upsert_row(&queue, 1, id, &[1, 2, 3]);
        assert_eq!(h1, h2, "same content id must resolve to the SAME shared range");
        assert_eq!(p.audit(), vec![(id, 2, 3 * 4)]);
    }

    #[test]
    fn last_reference_dropping_frees_the_range() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let id = HandleId(1);

        p.upsert_row(&queue, 0, id, &[1, 2, 3]);
        p.upsert_row(&queue, 1, id, &[1, 2, 3]);
        p.release_row(0);
        assert_eq!(p.audit(), vec![(id, 1, 12)], "one reference still live");
        p.release_row(1);
        assert!(p.audit().is_empty(), "last reference released -- fully torn down");
        assert_eq!(p.resolve(id), None, "resolve after free must be None, never stale data");
    }

    #[test]
    fn equal_id_rewrite_is_a_true_no_op() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let id = HandleId(7);
        let h1 = p.upsert_row(&queue, 0, id, &[9, 9]);
        let h2 = p.upsert_row(&queue, 0, id, &[9, 9]);
        assert_eq!(h1, h2);
        assert_eq!(p.audit(), vec![(id, 1, 8)], "refcount must not double-count a same-id rewrite of the SAME row");
    }

    #[test]
    fn switching_a_row_to_a_new_id_decrefs_the_old_one() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let a = HandleId(1);
        let b = HandleId(2);
        p.upsert_row(&queue, 0, a, &[1]);
        p.upsert_row(&queue, 0, b, &[2]);
        assert_eq!(p.resolve(a), None, "old id fully released once its only row moved away");
        assert!(p.resolve(b).is_some());
    }

    #[test]
    fn same_id_different_bytes_is_first_writer_wins() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let id = HandleId(5);
        let h1 = p.upsert_row(&queue, 0, id, &[1, 2, 3]);
        // Different row, SAME id, DIFFERENT bytes -- a collision by this
        // module's contract (upstream should never produce this for a
        // correct content-id function; simulated here directly).
        let h2 = p.upsert_row(&queue, 1, id, &[9, 9, 9, 9]);
        assert_eq!(h1, h2, "first writer's range wins; second writer shares it unchanged");
        assert_eq!(p.audit(), vec![(id, 2, 12)], "byte size reflects the FIRST writer's 3 elements, not the second's 4");
    }

    #[test]
    fn zero_id_still_writes_data_privately_just_never_shared() {
        // Zero means "no identity available", not "discard this write" --
        // a component whose content-id resolution failed (no project path,
        // an unresolvable asset, ...) must not silently lose real geometry
        // it already has. It gets an ordinary private allocation instead,
        // exactly like a plain (non-interned) Vec<T> #[gpu] field would.
        let (device, queue) = test_device();
        let p = pool(&device);
        let h = p.upsert_row(&queue, 0, HandleId::ZERO, &[1, 2]);
        assert_eq!(h, VarLenHandle { offset: 0, count: 2 }, "zero id still allocates and writes, just privately");
        assert!(p.audit().is_empty(), "a private allocation is never interned/shared, so it's invisible to audit");

        // A second, unrelated zero-id row on a DIFFERENT row gets its OWN
        // private allocation, not accidentally shared with the first.
        let h2 = p.upsert_row(&queue, 1, HandleId::ZERO, &[9, 9, 9]);
        assert_ne!(h, h2);

        // Releasing a private row frees it like any other allocation.
        p.release_row(0);
        let h3 = p.upsert_row(&queue, 2, HandleId::ZERO, &[5, 5]);
        assert_eq!(h3.offset, h.offset, "freed private space is reusable");
    }

    #[test]
    fn zero_id_with_truly_empty_data_allocates_nothing() {
        let (device, queue) = test_device();
        let p = pool(&device);
        let h = p.upsert_row(&queue, 0, HandleId::ZERO, &[]);
        assert_eq!(h, VarLenHandle::default());
    }

    #[test]
    fn switching_from_private_to_interned_frees_the_private_allocation() {
        let (device, queue) = test_device();
        let p = pool(&device);
        p.upsert_row(&queue, 0, HandleId::ZERO, &[1, 2, 3]);
        let id = HandleId(1);
        let shared = p.upsert_row(&queue, 0, id, &[9, 9]);
        assert_eq!(p.audit(), vec![(id, 1, 8)]);
        // The private range [0,3) must be back in the freelist -- proven by
        // a fresh allocation reusing offset 0 (the pool is first-fit).
        assert_eq!(shared.offset, 0, "the freed private space at offset 0 was reused for the new interned entry");
    }

    #[test]
    fn release_of_a_never_written_row_is_a_harmless_no_op() {
        let (device, _queue) = test_device();
        let p = pool(&device);
        p.release_row(999); // never written, way past id_by_row's length
        assert!(p.audit().is_empty());
    }
}
