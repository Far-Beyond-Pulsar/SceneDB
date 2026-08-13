//! The global dense slot allocator — issue #41's "Shared buffers use a
//! global dense slot allocator" section:
//!
//! > A named shared buffer... does not map 1:1 to local archetype row
//! > indices... Tier 1 attaches a global dense slot allocator (free-list +
//! > generational handles) to each named `GpuBuffer<T>`... The allocator's
//! > generational handles keep compaction and swap-remove sound: freed
//! > slots are reused via the free-list, and stale handles are caught by
//! > generation mismatch.
//!
//! [`SlotAllocator`] is exactly that primitive: a thread-safe, dense,
//! LIFO-reusing index allocator whose handles ([`SlotHandle`]) carry a
//! generation that is bumped on every free, so a caller holding a handle
//! from before a slot was freed-and-reused can detect the staleness with
//! [`SlotAllocator::is_valid`] instead of silently reading (or writing) a
//! different logical owner's data at the same physical index.
//!
//! # Scope: available Tier 1 infrastructure, not wired into `World`'s own
//! shared-buffer row scheme
//!
//! [`crate::world::World`]'s automatic GPU mirroring
//! ([`super::world_mirror::write_gpu_columns_at_row`]) already writes every
//! `#[gpu]` field — INCLUDING fields declaring a shared
//! `#[gpu(buffer = "key")]` — at `row = Entity::index()`, unconditionally,
//! and this is a deliberate, DOCUMENTED, and extensively tested contract
//! (`tests/world_gpu_mirror.rs`'s module doc: "registered GPU buffer at row
//! = `entity.index()`, automatically... is really `entity.index()`, not
//! just 'row 0 always works'"; every `tests/world_gpu_mirror_*.rs` file
//! asserts it directly). That is not an oversight this type needs to paper
//! over: [`crate::registry::HandleRegistry`] backing `Entity` allocation
//! ALREADY IS a free-list + generational-handle dense allocator (see
//! `registry.rs`'s own `allocate_starts_at_generation_one`/
//! `stale_handle_rejected_after_free` tests) — reusing `Entity::index()`
//! as the World-mirror's "global slot" gets the exact free-list-reuse +
//! generation-staleness properties this module provides, for free, backed
//! by machinery `World` already maintains for every entity regardless of
//! whether it carries any `#[gpu]` field at all
//! ([`super::world_mirror::GenerationMirror`] mirrors that same generation
//! to the GPU for staleness checks). Introducing a SECOND, independent
//! slot allocator into that write path would duplicate this bookkeeping,
//! not fix a gap, and — because `row = entity.index()` is a hard,
//! test-asserted contract — silently changing what row a World-mirrored
//! field lands at would be a breaking, not additive, change.
//!
//! Where this type earns its place is anything ELSE that needs a shared,
//! dense, reusable row space that is NOT naturally keyed by an existing
//! generational allocator — e.g. a future non-`Entity`-indexed consumer of
//! a named `GpuBuffer<T>` (an asset-side pool, a renderer-owned instance
//! table). It is exported as ordinary, directly-usable, fully-tested Tier 1
//! infrastructure for exactly that case, matching the issue's specified
//! shape (free-list + generational handles, dense, LIFO reuse) precisely.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, RwLock};

/// A handle into a [`SlotAllocator`]: a dense `index` plus the `generation`
/// it was valid for at allocation time. Two handles with the same `index`
/// but different `generation`s refer to two different logical claims on
/// that physical slot — the earlier one is stale the instant the later one
/// exists (see [`SlotAllocator::is_valid`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotHandle {
    pub index: u32,
    pub generation: u32,
}

/// Thread-safe global dense slot allocator: LIFO free-list reuse over a
/// monotonically-extending index space, with a per-index generation counter
/// that invalidates any handle issued before the most recent free of that
/// index. See the module doc for the full rationale and scope.
///
/// Cost, matching the issue's stated bound: one `u32` generation per
/// ever-issued index (4 bytes), plus the free list's `Vec<u32>` (bounded by
/// the live-freed count, never larger than the index space itself).
pub struct SlotAllocator {
    free: Mutex<Vec<u32>>,
    generations: RwLock<Vec<AtomicU32>>,
    next: AtomicU32,
    live: AtomicU32,
}

impl Default for SlotAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotAllocator {
    /// A fresh, empty allocator — no slots issued, nothing on the free list.
    pub fn new() -> Self {
        Self {
            free: Mutex::new(Vec::new()),
            generations: RwLock::new(Vec::new()),
            next: AtomicU32::new(0),
            live: AtomicU32::new(0),
        }
    }

    /// Claim a slot: reuses the most-recently-freed index if the free list
    /// is non-empty (LIFO — same reuse discipline
    /// [`crate::gpu::TextureStore`]'s slot table already uses), otherwise
    /// extends the index space by one. The returned handle's `generation`
    /// is this index's CURRENT generation (0 the first time an index is
    /// ever issued; bumped by exactly one on every [`Self::free`] of that
    /// index since).
    pub fn alloc(&self) -> SlotHandle {
        let index = {
            let mut free = self.free.lock().expect("SlotAllocator free-list lock poisoned");
            free.pop()
        }
        .unwrap_or_else(|| self.next.fetch_add(1, Ordering::Relaxed));

        let generation = self.ensure_and_read_generation(index);
        self.live.fetch_add(1, Ordering::Relaxed);
        SlotHandle { index, generation }
    }

    /// Release `handle`'s slot back to the free list, bumping its
    /// generation so any OTHER copy of this same handle (a stale clone held
    /// elsewhere) is immediately caught by [`Self::is_valid`]. Returns
    /// `false` — a no-op, not a panic — if `handle`'s generation no longer
    /// matches the index's current generation (already freed, or `handle`
    /// is simply invalid): a double-free is therefore always safe to call,
    /// unlike a bare `Vec`-backed free list would be.
    pub fn free(&self, handle: SlotHandle) -> bool {
        let generations = self.generations.read().expect("SlotAllocator generations lock poisoned");
        let Some(cell) = generations.get(handle.index as usize) else { return false };
        // Compare-and-bump: only the caller holding the CURRENT generation's
        // handle can free it, and doing so atomically invalidates that
        // generation for anyone else — a concurrent double-free from two
        // threads racing on the same stale handle can only ever succeed
        // once.
        let current = cell.load(Ordering::Acquire);
        if current != handle.generation {
            return false;
        }
        if cell
            .compare_exchange(current, current.wrapping_add(1), Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            // Lost the race to another freer of the exact same (index,
            // generation) pair — the other side's free already succeeded,
            // so this one correctly reports "no, you didn't free it".
            return false;
        }
        drop(generations);
        self.free.lock().expect("SlotAllocator free-list lock poisoned").push(handle.index);
        self.live.fetch_sub(1, Ordering::Relaxed);
        true
    }

    /// Whether `handle` still refers to a live, unfreed claim — i.e. its
    /// `generation` matches the index's current generation. `false` for an
    /// index past [`Self::capacity`] (never issued) as well as a freed one.
    pub fn is_valid(&self, handle: SlotHandle) -> bool {
        let generations = self.generations.read().expect("SlotAllocator generations lock poisoned");
        generations
            .get(handle.index as usize)
            .is_some_and(|cell| cell.load(Ordering::Acquire) == handle.generation)
    }

    /// Number of currently-live (allocated, not yet freed) slots.
    pub fn live_count(&self) -> u32 {
        self.live.load(Ordering::Relaxed)
    }

    /// One past the highest index ever issued — the dense extent a
    /// consumer's own backing storage (e.g. a `GpuBuffer<T>` row count)
    /// must cover, NOT the live count (freed-but-not-yet-reused indices
    /// still count toward this).
    pub fn capacity(&self) -> u32 {
        self.next.load(Ordering::Relaxed)
    }

    fn ensure_and_read_generation(&self, index: u32) -> u32 {
        let idx = index as usize;
        {
            let generations = self.generations.read().expect("SlotAllocator generations lock poisoned");
            if let Some(cell) = generations.get(idx) {
                return cell.load(Ordering::Acquire);
            }
        }
        let mut generations = self.generations.write().expect("SlotAllocator generations lock poisoned");
        if idx >= generations.len() {
            generations.resize_with(idx + 1, || AtomicU32::new(0));
        }
        generations[idx].load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn fresh_allocator_issues_dense_zero_based_indices() {
        let alloc = SlotAllocator::new();
        let a = alloc.alloc();
        let b = alloc.alloc();
        let c = alloc.alloc();
        assert_eq!((a.index, b.index, c.index), (0, 1, 2));
        assert_eq!(a.generation, 0);
        assert_eq!(alloc.capacity(), 3);
        assert_eq!(alloc.live_count(), 3);
    }

    #[test]
    fn free_then_alloc_reuses_the_index_lifo_with_a_bumped_generation() {
        let alloc = SlotAllocator::new();
        let a = alloc.alloc();
        let b = alloc.alloc();
        assert!(alloc.free(b), "free b");
        assert_eq!(alloc.live_count(), 1);
        let c = alloc.alloc();
        assert_eq!(c.index, b.index, "LIFO reuse: freed index b comes back first");
        assert_ne!(c.generation, b.generation, "generation bumped on free");
        assert_eq!(alloc.capacity(), 2, "reuse never extends capacity");
        assert!(alloc.is_valid(a));
        assert!(alloc.is_valid(c));
        assert!(!alloc.is_valid(b), "b's old handle is stale now that c reused its index");
    }

    #[test]
    fn double_free_is_a_safe_no_op_not_a_corruption() {
        let alloc = SlotAllocator::new();
        let a = alloc.alloc();
        assert!(alloc.free(a));
        assert!(!alloc.free(a), "second free of the same handle must fail cleanly");
        assert_eq!(alloc.live_count(), 0);
        // The free list must not have gained a duplicate entry from the
        // failed second free — reusing the index once must not somehow
        // hand it out twice concurrently.
        let b = alloc.alloc();
        let c = alloc.alloc();
        assert_ne!(b.index, c.index, "no duplicate index from the double-free");
    }

    #[test]
    fn is_valid_is_false_for_an_index_never_issued() {
        let alloc = SlotAllocator::new();
        assert!(!alloc.is_valid(SlotHandle { index: 0, generation: 0 }));
        alloc.alloc();
        assert!(!alloc.is_valid(SlotHandle { index: 1, generation: 0 }), "index 1 never issued");
    }

    #[test]
    fn stale_handle_from_before_a_free_is_rejected_even_with_a_matching_index() {
        let alloc = SlotAllocator::new();
        let a = alloc.alloc();
        assert!(alloc.free(a));
        let b = alloc.alloc(); // reuses a.index with a new generation
        assert_eq!(a.index, b.index);
        assert!(!alloc.is_valid(a), "a is stale");
        assert!(alloc.is_valid(b), "b is the live claim on this index now");
        // The stale handle can't be used to free the NEW claim either.
        assert!(!alloc.free(a), "freeing with a's stale generation must not free b's live slot");
        assert!(alloc.is_valid(b), "b must still be live after the rejected stale free");
    }

    #[test]
    fn many_alloc_free_cycles_never_produce_two_live_handles_for_one_index() {
        let alloc = SlotAllocator::new();
        let mut live: Vec<SlotHandle> = Vec::new();
        for i in 0..500u32 {
            live.push(alloc.alloc());
            if i % 3 == 0 && !live.is_empty() {
                let h = live.remove(0);
                assert!(alloc.free(h));
            }
        }
        // Every remaining live handle must be valid, and no two share an
        // index (which would mean two logical owners think they hold the
        // same physical slot).
        let mut seen = HashSet::new();
        for h in &live {
            assert!(alloc.is_valid(*h));
            assert!(seen.insert(h.index), "duplicate live index {}", h.index);
        }
        assert_eq!(alloc.live_count() as usize, live.len());
    }

    #[test]
    fn concurrent_allocations_never_hand_out_the_same_index_twice() {
        let alloc = Arc::new(SlotAllocator::new());
        const THREADS: usize = 8;
        const PER_THREAD: usize = 200;
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let alloc = Arc::clone(&alloc);
                std::thread::spawn(move || {
                    (0..PER_THREAD).map(|_| alloc.alloc()).collect::<Vec<_>>()
                })
            })
            .collect();

        let mut all_indices = HashSet::new();
        for h in handles {
            for slot in h.join().expect("thread panicked") {
                assert!(all_indices.insert(slot.index), "index {} issued twice concurrently", slot.index);
            }
        }
        assert_eq!(all_indices.len(), THREADS * PER_THREAD);
        assert_eq!(alloc.live_count() as usize, THREADS * PER_THREAD);
    }

    #[test]
    fn default_matches_new() {
        let alloc = SlotAllocator::default();
        assert_eq!(alloc.capacity(), 0);
        assert_eq!(alloc.live_count(), 0);
    }
}
