//! Growable, variable-length-per-row GPU pool for `#[gpu]` fields whose
//! Rust type is `Vec<T>` (`T: Pod`) — e.g. a component storing its own
//! vertex array directly, instead of an index into a separate asset
//! registry. Two pieces work together, both reusing already-proven
//! mechanics rather than adding a new one:
//!
//! - A per-entity, row-indexed (`entity.index()`-keyed) handle table
//!   (`{offset, count}` — [`VarLenHandle`], itself a plain fixed-size `Pod`
//!   value). This is nothing new: it's the exact same
//!   `GrowableSceneBuffer<T>` mechanism every scalar `#[gpu]` field already
//!   uses, with `T = VarLenHandle`.
//! - ONE shared, growable pool (this module's [`VarLenGpuPool<T>`], built on
//!   [`super::DynamicGpuBuffer`]'s existing grow-and-copy mechanics) that
//!   every entity's variable-length payload is suballocated from, tracked by
//!   [`super::freelist::RangeList`] — the same allocator
//!   [`super::assets::GeometryArena`] already proved, shared rather than
//!   reimplemented (see that module for the extraction).
//!
//! Each write frees the entity's PREVIOUS allocation (if any — the vec's
//! length may differ from last time) before allocating fresh space, so the
//! pool never accumulates orphaned space from a shrinking/growing field.
//! Despawn (or overwriting with an empty `Vec`) frees the same way — see
//! `world_mirror.rs`'s despawn hook, the only other caller of
//! [`VarLenGpuPool::free_handle`] besides this module's own re-write path.
//!
//! No new `#[gpu(...)]` attribute syntax: a `Vec<T>`-typed `#[gpu]` field is
//! detected by its Rust type alone (`pulsar_scenedb_derive`'s codegen) and
//! routed through this pool automatically. Every OTHER `#[gpu]` field
//! (scalars, fixed-size arrays, structs) is completely unaffected — same
//! generated code, same performance, as before this module existed.
//!
//! `T` is allowed to be any `Pod` element, including ones smaller than
//! `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes) or whose size doesn't evenly
//! divide it — e.g. `pulsar_world_registry::GpuRepr<bool>`, 1 byte. Every
//! allocation this pool hands out is therefore reserved in units of
//! [`super::dynamic_buffer::elem_align::<T>()`] elements rather than raw
//! element count, and written through [`DynamicGpuBuffer::write_padded`] —
//! together these guarantee every `wgpu::Queue::write_buffer` call this
//! pool issues has a 4-byte-aligned offset AND a 4-byte-aligned length, no
//! matter how small or oddly-sized `T` is. [`VarLenHandle::count`] always
//! stays the true, unpadded element count regardless — the padding is
//! purely a GPU-buffer-layout implementation detail, invisible to callers.

use super::dynamic_buffer::{elem_align, CapacityError, DynamicGpuBuffer};
use super::freelist::RangeList;
use super::tier::{ByteRange, TierAuditKey, TierAuditRecord, TierEngine, TieredPoolHooks, NO_SLOT};
use crate::page::Pod;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Rounds `count` up to the nearest multiple of `align` — the element count
/// actually reserved/freed in the freelist for a `count`-element logical
/// row, once `T` is small enough that individual elements don't line up on
/// `wgpu::COPY_BUFFER_ALIGNMENT` (4-byte) boundaries on their own. See
/// [`elem_align`] for where `align` itself comes from. A no-op multiplier
/// (`align == 1`) for every `T` that existed before `#[gpu]` fields allowed
/// arbitrary `Copy` types — zero behavior change for the common case.
fn padded_alloc_len(count: u32, align: u64) -> u64 {
    (count as u64).next_multiple_of(align.max(1))
}

/// A `Pod`, per-entity-row handle into a [`VarLenGpuPool`] — what actually
/// lands in the row-indexed growable column for a `Vec<T>`-typed `#[gpu]`
/// field. `count == 0` means "no allocation" (an empty or never-written
/// `Vec`) — `offset` is meaningless in that case, never dereferenced.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct VarLenHandle {
    pub offset: u32,
    pub count: u32,
}

// SAFETY: #[repr(C)], two plain u32s, no padding, every bit pattern valid.
unsafe impl Pod for VarLenHandle {}

struct Inner<T: Pod> {
    buf: DynamicGpuBuffer<T>,
    free: RangeList,
    /// Tier sidecar (#61 §1): one residency record per LIVE allocation,
    /// keyed by the allocation's pool offset — parallel to the freelist
    /// bookkeeping exactly as the issue specifies. `Off` until
    /// [`VarLenGpuPool::set_tier_participation`] flips it on (the store's
    /// plain-pool registration path does; interned underlays and standalone
    /// pools stay Off, costing nothing beyond one enum discriminant).
    tier: PoolTierState<T>,
}

/// One live allocation's residency record — the extended entry from #61 §1:
/// `{offset(=key), count, tier, resident_ranges, generation, last_use}`.
pub(crate) struct PoolTierRecord {
    /// True element count of the live allocation (identity half: a recycled
    /// offset whose new tenant declares a different count is detectably NOT
    /// the same resource).
    pub count: u32,
    /// Monotonic identity generation from a pool-level counter that never
    /// resets — strictly increases across offset reuse, which is what makes
    /// the flush liveness guard sound (a queued intent captured before a
    /// free can NEVER validate against the recycled tenant).
    pub generation: u64,
    /// Last-demand stamp (engine clock).
    pub last_use: u64,
    /// Newest executed intent-sequence stamp for this slot (freshness
    /// filter for cross-batch stale intents).
    pub last_seq: u64,
    /// Current authoritative residency of the payload bytes.
    pub tier: super::tier::Tier,
    /// RAM-arena staging slot (`NO_SLOT` = unstaged), or disk-arena export
    /// slot once spilled (`spilled == true`).
    pub staging: u32,
    pub spilled: bool,
    /// Resident VRAM rank-prefix watermark (index into the flight plan,
    /// not a rank value — plans are rebuilt deterministically).
    pub resident_units: Vec<(u32, ByteRange)>,
}

enum PoolTierState<T: Pod> {
    Off,
    On {
        engine: Arc<TierEngine>,
        /// This pool's own registration key — audit labels + texture
        /// materialization lookup.
        key: crate::gpu::BufferKey,
        /// Identity-generation counter (monotonic, never reset).
        next_generation: u64,
        recs: HashMap<u64 /*offset*/, PoolTierRecord>,
        _elem: std::marker::PhantomData<T>,
    },
}

/// Shared growable pool backing every entity's `Vec<T>`-typed `#[gpu]`
/// field for one (struct, field) column (or one shared `#[gpu(buffer =
/// "key")]` group, same sharing rule scalar `#[gpu]` fields already have).
///
/// `RwLock`-guarded for the same reason [`super::GrowableSceneBuffer`] is —
/// growth needs `&mut`, but the derive-generated dispatch only ever has
/// `&self` to call through (see that type's doc for the full rationale;
/// identical here).
pub struct VarLenGpuPool<T: Pod> {
    device: Arc<wgpu::Device>,
    inner: RwLock<Inner<T>>,
}

impl<T: Pod + Send + Sync + 'static> VarLenGpuPool<T> {
    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32) -> Self {
        Self::new_with_usage(device, label, initial_capacity, wgpu::BufferUsages::STORAGE)
    }

    /// Same as [`Self::new`], but with an explicit base buffer usage instead
    /// of the default `STORAGE` — for pools bound a different way than a
    /// `#[gpu] Vec<T>` field's usual shader-storage-buffer read (e.g. a
    /// fixed-function `VERTEX`/`INDEX` buffer). Every derive-generated
    /// `#[gpu] Vec<T>` field still goes through [`Self::new`] (`STORAGE`,
    /// unchanged) — this is for direct, non-derive callers like Helio's mesh
    /// pool that want the SAME pool/freelist mechanics with a different
    /// underlying buffer usage.
    pub fn new_with_usage(
        device: Arc<wgpu::Device>,
        label: &str,
        initial_capacity: u32,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buf = DynamicGpuBuffer::new_with_usage(&device, label, initial_capacity, usage);
        let free = RangeList::new(initial_capacity as u64);
        Self { device, inner: RwLock::new(Inner { buf, free, tier: PoolTierState::Off }) }
    }

    /// Attaches the tier engine to this pool (the store's plain-pool
    /// registration path calls this; interned underlays deliberately do not
    /// — statics have no tier data plane). Idempotent; the LAST call wins.
    /// Registration-time only — never reachable on a per-frame path.
    pub(crate) fn set_tier_participation(&self, engine: Arc<TierEngine>, key: crate::gpu::BufferKey) {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        if matches!(&guard.tier, PoolTierState::On { key: existing, .. } if *existing == key) {
            return;
        }
        guard.tier = PoolTierState::On {
            engine,
            key,
            next_generation: 0,
            recs: HashMap::new(),
            _elem: std::marker::PhantomData,
        };
    }

    /// Removes and drops the record at `offset` together with its staging
    /// bytes — the fate of an emptied-out or freed allocation.
    fn tier_drop_record(tier: &mut PoolTierState<T>, offset: u64) {
        let PoolTierState::On { engine, recs, .. } = tier else { return };
        if let Some(prev) = recs.remove(&offset) {
            if prev.staging != NO_SLOT {
                if prev.spilled {
                    engine.discard_disk_export(prev.staging);
                } else {
                    engine.discard_ram(prev.staging);
                }
            }
        }
    }

    /// Detaches the record at `offset` WITHOUT dropping its staging,
    /// returning the RAM slot for in-place reuse by the next staging of a
    /// same-or-smaller payload — the mechanism that keeps steady-state
    /// rewrites allocation-free. Spilled exports are dropped (their owner
    /// is being rewritten; the frozen snapshot is obsolete).
    fn tier_detach_for_reuse(tier: &mut PoolTierState<T>, offset: u64) -> Option<u32> {
        let PoolTierState::On { engine, recs, .. } = tier else { return None };
        match recs.remove(&offset) {
            Some(prev) => {
                if prev.spilled {
                    engine.discard_disk_export(prev.staging);
                    None
                } else {
                    Some(prev.staging).filter(|&s| s != NO_SLOT)
                }
            }
            None => None,
        }
    }

    /// Stages `data`'s bytes and inserts the sidecar record for the live
    /// allocation at `offset` — the write-path half of #61 §1's "default
    /// tier on write is RAM". `reuse` is a detached staging slot eligible
    /// for in-place overwrite (`Self::tier_detach_for_reuse`).
    fn tier_insert_record(
        tier: &mut PoolTierState<T>,
        offset: u64,
        count: u32,
        data: &[T],
        reuse: Option<u32>,
    ) {
        let PoolTierState::On { engine, recs, next_generation, .. } = tier else { return; };
        debug_assert!(
            !recs.contains_key(&offset),
            "stale tier record at a freshly allocated pool offset -- prior tenant was never freed"
        );
        let byte_len = std::mem::size_of::<T>() as u64 * data.len() as u64;
        let generation = *next_generation;
        *next_generation += 1;
        let last_use = engine.stamp();
        let staging = engine.restage_ram(reuse.filter(|_| byte_len > 0), super::as_bytes(data));
        recs.insert(
            offset,
            PoolTierRecord {
                count,
                generation,
                last_use,
                tier: super::tier::Tier::Ram,
                staging: if byte_len == 0 { NO_SLOT } else { staging },
                spilled: false,
                resident_units: Vec::new(),
                last_seq: 0,
            },
        );
    }

    /// Drops the sidecar record for a freed allocation (if any), returning
    /// its staging bytes to the arena's free pool with them.
    fn tier_record_free(tier: &mut PoolTierState<T>, offset: u64) {
        Self::tier_drop_record(tier, offset)
    }

    /// Frees `prev`'s allocation (if any — `count == 0` is a no-op, so this
    /// is safe to call with a fresh entity's default handle) and, if `data`
    /// is non-empty, allocates fresh space and writes it, growing the
    /// backing buffer first if the freelist can't satisfy the request.
    /// Returns the new handle to store in the row-indexed handle table —
    /// `VarLenHandle::default()` (count 0) if `data` was empty.
    pub fn write_var_row(
        &self,
        queue: &wgpu::Queue,
        prev: VarLenHandle,
        data: &[T],
    ) -> Result<VarLenHandle, CapacityError> {
        let align = elem_align::<T>();
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        if prev.count > 0 {
            guard.free.free(prev.offset as u64, padded_alloc_len(prev.count, align));
        }
        if data.is_empty() {
            Self::tier_record_free(&mut guard.tier, prev.offset as u64);
            return Ok(VarLenHandle::default());
        }

        // Reserve `align`-rounded-up space, not just `data.len()` — the
        // extra tail (if any) is what `DynamicGpuBuffer::write_padded`
        // pads the actual write into so its byte length stays a multiple
        // of `wgpu::COPY_BUFFER_ALIGNMENT`; reserving it here (rather than
        // at write time) is what guarantees that padding never overlaps a
        // neighboring row's live allocation. `VarLenHandle::count` below
        // stays the true, unpadded element count — callers/shaders never
        // see the padding as real data.
        let len = data.len() as u64;
        let alloc_len = padded_alloc_len(data.len() as u32, align);
        let offset = match guard.free.alloc(alloc_len, align) {
            Some(offset) => offset,
            None => {
                // Exhausted -- grow the backing buffer (existing
                // DynamicGpuBuffer grow-and-copy mechanics, not
                // reimplemented) then extend the freelist's tracked total
                // to match the ACTUAL new capacity (which may be larger
                // than the bare minimum requested -- `ensure_capacity`
                // doubles), and retry. Must succeed: the new capacity is
                // guaranteed >= old_total + alloc_len by construction.
                let old_total = guard.buf.capacity() as u64;
                let min_capacity = old_total.saturating_add(alloc_len).min(u32::MAX as u64) as u32;
                guard.buf.ensure_capacity(&self.device, queue, min_capacity)?;
                let new_total = guard.buf.capacity() as u64;
                guard.free.extend_total(old_total, new_total);
                guard
                    .free
                    .alloc(alloc_len, align)
                    .expect("freelist must satisfy an allocation immediately after extending past it")
            }
        };

        // Detach the previous tenant's staging for in-place reuse BEFORE
        // the new allocation — wherever it lived (same offset first-fit
        // reuse, or a relocated one), its box can serve the new payload.
        // ONLY for an actual previous allocation (`prev.count > 0`): a
        // never-written row's default handle names offset 0, and detaching
        // THAT would steal whoever legitimately owns offset 0 today.
        let reuse = if prev.count > 0 {
            Self::tier_detach_for_reuse(&mut guard.tier, prev.offset as u64)
        } else {
            None
        };

        guard.buf.write_padded(queue, offset as u32, data);
        Self::tier_insert_record(&mut guard.tier, offset, len as u32, data, reuse);
        Ok(VarLenHandle { offset: offset as u32, count: len as u32 })
    }

    /// Frees `handle`'s allocation without writing a replacement — the
    /// despawn/removal path (an entity going away, or its `Vec` field being
    /// removed outright, has nothing new to write, only old space to give
    /// back). No-op if `handle.count == 0`. Also drops the allocation's
    /// tier sidecar record together with its staging bytes, so a freed
    /// range never leaves orphaned residency behind (#61 test-4 contract).
    pub fn free_handle(&self, handle: VarLenHandle) {
        if handle.count == 0 {
            return;
        }
        let align = elem_align::<T>();
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        guard.free.free(handle.offset as u64, padded_alloc_len(handle.count, align));
        Self::tier_record_free(&mut guard.tier, handle.offset as u64);
    }

    /// Overwrites `data` at `offset` in place — no allocation, no freelist
    /// interaction, offset never changes. For a caller that already holds a
    /// valid, still-allocated `VarLenHandle` and wants to replace its
    /// contents WITHOUT the free-then-reallocate cycle [`Self::write_var_row`]
    /// does (which offers no guarantee the same offset comes back — a caller
    /// that baked the old offset into other GPU-side state, e.g. a draw
    /// command's `vertex_offset`, would silently start reading garbage).
    /// Panics (same contract as the underlying `DynamicGpuBuffer::write`) if
    /// `offset + data.len()` exceeds the pool's current capacity, or if
    /// `data.len()` doesn't match `offset`'s original allocation size —
    /// neither of which this call can check on its own (it has no notion of
    /// "whose" allocation `offset` belongs to), so callers must pass an
    /// `offset`/length pair known to fit inside an allocation they already
    /// hold the matching `VarLenHandle` for.
    pub fn write_at_offset(&self, queue: &wgpu::Queue, offset: u32, data: &[T]) {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        guard.buf.write_padded(queue, offset, data);
    }

    pub fn epoch(&self) -> u64 {
        self.inner.read().expect("VarLenGpuPool lock poisoned").buf.epoch()
    }

    pub fn capacity(&self) -> u32 {
        self.inner.read().expect("VarLenGpuPool lock poisoned").buf.capacity()
    }

    /// Runs `f` against the pool's current `wgpu::Buffer` — the lock-safe
    /// way to reach it for bind-group construction. Do not stash the
    /// `&wgpu::Buffer` past this call (same contract as
    /// [`super::GrowableGpuBufferDispatch::with_buffer`]).
    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        f(guard.buf.buffer());
    }

    /// Same buffer access as [`Self::with_buffer`], but as a held guard
    /// instead of a closure — for callers that need a real `&wgpu::Buffer`-
    /// shaped struct field (e.g. a per-frame bind-group descriptor threaded
    /// through several passes) rather than a callback. Holds the SAME read
    /// lock `with_buffer` takes for the guard's lifetime; a concurrent
    /// `write_var_row`/`free_handle` blocks until it's dropped, exactly like
    /// any other `RwLock` reader — callers should drop it once the frame's
    /// draw calls are recorded, not hold it indefinitely.
    pub fn read_buffer(&self) -> VarLenBufferRef<'_, T> {
        VarLenBufferRef(self.inner.read().expect("VarLenGpuPool lock poisoned"))
    }

    pub fn as_any(&self) -> &dyn Any
    where
        Self: 'static,
    {
        self
    }
}

/// A read-locked handle to a [`VarLenGpuPool`]'s current `wgpu::Buffer`.
/// `Deref`s straight to it, so `pool.read_buffer().slice(..)` etc. work the
/// same as if a bare `&wgpu::Buffer` had been borrowed.
pub struct VarLenBufferRef<'a, T: Pod>(std::sync::RwLockReadGuard<'a, Inner<T>>);

impl<'a, T: Pod> std::ops::Deref for VarLenBufferRef<'a, T> {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.0.buf.buffer()
    }
}

/// The flush executor's type-erased view over one participating pool — see
/// [`TieredPoolHooks`] (in `tier.rs`) for the seam contract. Granularity
/// note: rank-unit slices against a POOL payload must be 4-byte aligned in
/// payload-relative terms (the strictest `wgpu::Queue::write_buffer`
/// demands, matched exactly); misaligned layouts are a registration-author
/// error and panic loudly at first flight rather than silently rounding.
impl<T: Pod + Send + Sync + 'static> TieredPoolHooks for VarLenGpuPool<T> {
    fn payload_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn slot_state(&self, offset: u64) -> Option<super::tier::SlotState> {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        match &guard.tier {
            PoolTierState::On { recs, .. } => {
                let rec = recs.get(&offset)?;
                Some(super::tier::SlotState {
                    generation: rec.generation,
                    count: rec.count,
                    tier: rec.tier,
                    staged_len: rec.count as u64 * std::mem::size_of::<T>() as u64,
                    spilled: rec.spilled,
                    vram_bytes: rec.resident_units.iter().map(|(_, rng)| rng.len).sum(),
                    last_use: rec.last_use,
                })
            }
            PoolTierState::Off => None,
        }
    }

    fn stamp_slot(&self, offset: u64) -> u64 {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { engine, recs, .. } = &mut guard.tier else { return 0 };
        let stamp = engine.stamp();
        if let Some(rec) = recs.get_mut(&offset) {
            rec.last_use = stamp;
        }
        stamp
    }

    fn claim_seq(&self, offset: u64, seq: u64) -> bool {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &mut guard.tier else { return false };
        match recs.get_mut(&offset) {
            Some(rec) if seq > rec.last_seq => {
                rec.last_seq = seq;
                true
            }
            _ => false,
        }
    }

    fn copy_staged_window(&self, offset: u64, rel: u64, out: &mut [u8]) -> bool {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { engine, recs, .. } = &guard.tier else { return false };
        let Some(rec) = recs.get(&offset) else { return false };
        if rec.staging == NO_SLOT {
            return false;
        }
        if rec.spilled {
            engine.copy_disk_window(rec.staging, rel, out)
        } else {
            engine.copy_ram_window(rec.staging, rel, out)
        }
    }

    fn admit_rank_unit(
        &self,
        queue: &wgpu::Queue,
        offset: u64,
        rank: u32,
        packed: &[u8],
        slices: &[(u64, u64)],
    ) -> ByteRange {
        let elem_size = std::mem::size_of::<T>().max(1) as u64;
        let align = elem_align::<T>();
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        // Reserve element units exactly like any payload of the same size,
        // so the freelist never mixes unit systems with ordinary writes.
        let elems = (packed.len() as u64).div_ceil(elem_size);
        let alloc_len = padded_alloc_len(elems as u32, align);
        let alloc_off = match guard.free.alloc(alloc_len, align) {
            Some(off) => off,
            None => {
                let old_total = guard.buf.capacity() as u64;
                let min_capacity = old_total.saturating_add(alloc_len).min(u32::MAX as u64) as u32;
                guard
                    .buf
                    .ensure_capacity(&self.device, queue, min_capacity)
                    .expect("tier promotion growth cannot fail (pools have no capacity ceiling)");
                let new_total = guard.buf.capacity() as u64;
                guard.free.extend_total(old_total, new_total);
                guard
                    .free
                    .alloc(alloc_len, align)
                    .expect("freelist must satisfy an allocation immediately after extending past it")
            }
        };
        // Upload the unit's bytes at their payload-relative positions.
        let base_bytes = alloc_off * elem_size;
        let buf = guard.buf.buffer();
        for &(rel, len) in slices {
            assert!(
                rel % 4 == 0 && len % 4 == 0 && (rel + len) as usize <= packed.len(),
                "rank-unit slice [{rel},+{len}) violates the 4-byte granularity contract for \
                 pool-payload segment layouts (see tier.rs module doc) -- fix the registered layout"
            );
            queue.write_buffer(buf, base_bytes + rel, &packed[rel as usize..(rel + len) as usize]);
        }
        let range = ByteRange { offset: base_bytes, len: alloc_len * elem_size };
        let rec = match &mut guard.tier {
            PoolTierState::On { recs, .. } => recs.get_mut(&offset).expect("live record validated before flight"),
            PoolTierState::Off => unreachable!("admit_rank_unit on a non-participating pool"),
        };
        rec.resident_units.push((rank, range));
        rec.tier = super::tier::Tier::Vram;
        range
    }

    fn withdraw_top_unit(&self, offset: u64) -> Option<(u32, ByteRange)> {
        let elem_size = std::mem::size_of::<T>().max(1) as u64;
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &mut guard.tier else { return None };
        let rec = recs.get_mut(&offset)?;
        let idx = rec
            .resident_units
            .iter()
            .enumerate()
            .max_by_key(|(_, (r, _))| *r)
            .map(|(i, _)| i)?;
        let (rank, range) = rec.resident_units.remove(idx);
        rec.generation += 1;
        if rec.resident_units.is_empty() {
            rec.tier = super::tier::Tier::Ram;
        }
        // `range.len` was reserved in padded element units — free the exact
        // same span so the freelist never drifts from its own bookkeeping.
        guard.free.free(range.offset / elem_size, range.len / elem_size);
        Some((rank, range))
    }

    fn spill_staging(&self, offset: u64) {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { engine, recs, .. } = &mut guard.tier else { return };
        if let Some(rec) = recs.get_mut(&offset) {
            if !rec.spilled {
                let (disk_slot, _) = engine.spill_ram_to_disk(rec.staging);
                rec.staging = disk_slot;
                rec.spilled = true;
                rec.tier = super::tier::Tier::Disk;
                rec.generation += 1; // demotion Ram->Disk re-signs identity
            }
        }
    }

    fn unspill_staging(&self, offset: u64) {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { engine, recs, .. } = &mut guard.tier else { return };
        if let Some(rec) = recs.get_mut(&offset) {
            if rec.spilled {
                let (ram_slot, _) = engine.unspill_disk_to_ram(rec.staging);
                rec.staging = ram_slot;
                rec.spilled = false;
                rec.tier = super::tier::Tier::Ram; // promotion Disk->Ram: no gen bump
            }
        }
    }

    fn harvest_vram_candidates(&self) -> Vec<(u64, u64, u64)> {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &guard.tier else { return Vec::new() };
        recs.iter()
            .filter(|(_, r)| !r.resident_units.is_empty())
            .map(|(&off, r)| (off, r.last_use, r.resident_units.iter().map(|(_, rng)| rng.len).sum()))
            .collect()
    }

    fn resident_ranks(&self, offset: u64) -> Vec<(u32, ByteRange)> {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &guard.tier else { return Vec::new() };
        let mut units = recs.get(&offset).map(|r| r.resident_units.clone()).unwrap_or_default();
        units.sort_by_key(|(rank, _)| *rank);
        units
    }

    fn harvest_staging_candidates(&self) -> Vec<(u64, u64)> {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &guard.tier else { return Vec::new() };
        // VRAM-resident slots are EXCLUDED: spilling a resident slot's
        // staging would leave its tier byte (Vram) inconsistent with its
        // backing location, and the resident copy doesn't need its staging
        // to stay alive — only future re-promotions do. Pressure spills
        // target Ram-resident slots, whose residency IS their staging.
        recs.iter()
            .filter(|(_, r)| !r.spilled && r.staging != NO_SLOT && r.resident_units.is_empty())
            .map(|(&off, r)| (off, r.last_use))
            .collect()
    }

    fn audit_records(&self, key: crate::gpu::BufferKey) -> Vec<TierAuditRecord> {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        let PoolTierState::On { recs, .. } = &guard.tier else { return Vec::new() };
        let mut out: Vec<TierAuditRecord> = recs
            .iter()
            .map(|(&off, r)| TierAuditRecord {
                key: TierAuditKey::PoolSlot(key, off),
                tier: r.tier,
                generation: r.generation,
                last_use: r.last_use,
                pinned: false,
                vram_bytes: r.resident_units.iter().map(|(_, rng)| rng.len).sum(),
                staging_bytes: if r.staging == NO_SLOT {
                    0
                } else {
                    r.count as u64 * std::mem::size_of::<T>() as u64
                },
            })
            .collect();
        out.sort_by_key(|rec| match rec.key {
            TierAuditKey::PoolSlot(_, off) => off,
            _ => 0,
        });
        out
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("var-len-gpu-pool-test"),
            ..Default::default()
        }))
        .expect("device");
        (Arc::new(device), Arc::new(queue))
    }

    fn readback(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        data
    }

    #[test]
    fn write_then_readback_round_trips() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let handle = pool
            .write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3])
            .expect("fits initial capacity");
        assert_eq!(handle, VarLenHandle { offset: 0, count: 3 });

        let mut bytes = Vec::new();
        pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 3 * 4));
        let got: Vec<u32> = bytes.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn re_write_with_a_different_length_frees_the_old_slot_and_reuses_it() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 16);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3, 4]).unwrap();
        assert_eq!(h1.offset, 0);

        // Shrink: re-write with fewer elements -- must free [0,4) and
        // re-allocate a smaller [0,2) slot, not leak the tail.
        let h2 = pool.write_var_row(&queue, h1, &[9, 9]).unwrap();
        assert_eq!(h2, VarLenHandle { offset: 0, count: 2 }, "reuses the freed slot from the same offset");

        // A second entity's write must now be able to reuse [2,4) (freed by
        // the shrink above), proving the old tail wasn't orphaned.
        let h3 = pool.write_var_row(&queue, VarLenHandle::default(), &[7, 7]).unwrap();
        assert_eq!(h3, VarLenHandle { offset: 2, count: 2 });
    }

    #[test]
    fn exhaustion_grows_the_backing_buffer_transparently() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 2);
        assert_eq!(pool.capacity(), 2);

        // 10 elements doesn't fit the initial capacity of 2 -- must grow,
        // not fail.
        let handle = pool.write_var_row(&queue, VarLenHandle::default(), &(0..10).collect::<Vec<u32>>()).unwrap();
        assert_eq!(handle, VarLenHandle { offset: 0, count: 10 });
        assert!(pool.capacity() >= 10);

        let mut bytes = Vec::new();
        pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 10 * 4));
        let got: Vec<u32> = bytes.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn writing_an_empty_slice_frees_the_previous_allocation_and_allocates_nothing() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3]).unwrap();
        let h2 = pool.write_var_row(&queue, h1, &[]).unwrap();
        assert_eq!(h2, VarLenHandle::default(), "empty write yields the no-allocation sentinel");

        // The freed [0,3) must be fully reusable by a fresh write.
        let h3 = pool.write_var_row(&queue, VarLenHandle::default(), &[5, 5, 5]).unwrap();
        assert_eq!(h3.offset, 0);
    }

    #[test]
    fn free_handle_reclaims_space_with_no_replacement_write() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3, 4]).unwrap();
        pool.free_handle(h1);

        let h2 = pool.write_var_row(&queue, VarLenHandle::default(), &[9, 9, 9, 9]).unwrap();
        assert_eq!(h2.offset, 0, "the space freed by free_handle must be reusable");
    }

    #[test]
    fn read_buffer_derefs_to_the_same_buffer_with_buffer_reaches() {
        // Proves `read_buffer()` isn't a second, divergent path -- it must
        // observe the exact same bytes `with_buffer` (the pre-existing,
        // already-proven-correct accessor) sees, since both just borrow the
        // same lock-guarded `wgpu::Buffer`. This is the accessor a caller
        // like Helio's `MeshBuffers<'a>` (which needs a real `&wgpu::Buffer`-
        // shaped struct field, not a callback) will hold instead.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);
        pool.write_var_row(&queue, VarLenHandle::default(), &[11, 22, 33]).unwrap();

        let via_guard = readback(&device, &queue, &pool.read_buffer(), 3 * 4);
        let mut via_closure = Vec::new();
        pool.with_buffer(&mut |b| via_closure = readback(&device, &queue, b, 3 * 4));
        assert_eq!(via_guard, via_closure);

        let got: Vec<u32> =
            via_guard.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![11, 22, 33]);
    }

    #[test]
    fn write_at_offset_overwrites_in_place_without_touching_the_freelist() {
        // The property Helio's dynamic-mesh vertex updates depend on:
        // overwriting an existing allocation's contents must NEVER change
        // its offset (unlike write_var_row's free-then-reallocate), since
        // other GPU-side state (a draw command's vertex_offset) may have
        // already baked in the old offset and would silently read garbage
        // if a same-size "update" ever relocated the data.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3, 4]).unwrap();

        // A second, unrelated allocation right after h1 -- if write_at_offset
        // touched the freelist at all, this would be corrupted by it.
        let h2 = pool.write_var_row(&queue, VarLenHandle::default(), &[100, 200]).unwrap();
        assert_eq!(h2.offset, 4, "second alloc lands right after the first");

        pool.write_at_offset(&queue, h1.offset, &[9, 9, 9, 9]);

        let bytes = readback(&device, &queue, &pool.read_buffer(), 6 * 4);
        let got: Vec<u32> = bytes.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(&got[0..4], &[9, 9, 9, 9], "h1's slot must be overwritten in place, same offset");
        assert_eq!(&got[4..6], &[100, 200], "h2's data, right after h1, must be completely untouched");

        // A fresh allocation afterward must NOT be able to reuse h1's
        // offset -- write_at_offset must not have freed it.
        let h3 = pool.write_var_row(&queue, VarLenHandle::default(), &[7]).unwrap();
        assert_eq!(h3.offset, 6, "h1's range was never freed, so h3 must append past both live allocations");
    }

    // --- Odd-sized-element stress tests -------------------------------
    //
    // `T` is allowed to be ANY `Pod` element, not just 4-byte-or-larger
    // ones -- these push genuinely awkward shapes through the pool to prove
    // `elem_align`/`write_padded` (dynamic_buffer.rs) hold up: 1-byte and
    // 3-byte elements (neither divides 4 evenly), and a small fixed-size
    // "2D block" per row as the honest GPU-representable stand-in for
    // jagged/nested data. A real `Vec<Vec<T>>` field can never reach this
    // pool at all -- `Vec` isn't `Copy`, so it fails `GpuRepr<T>`'s `T:
    // Copy` bound at COMPILE time, before any of this runs. That's Rust's
    // own rule, not a gap: heap-recursive data has no fixed byte layout to
    // copy in the first place. A `Vec<[T; N]>` (this pool already supports
    // arbitrarily many, arbitrarily-shaped fixed rows) is the correct way
    // to put row-major 2D/3D data on the GPU through this mechanism.

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    struct OneByte(u8);
    unsafe impl Pod for OneByte {}

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    struct ThreeBytes([u8; 3]);
    unsafe impl Pod for ThreeBytes {}

    #[test]
    fn one_byte_elements_survive_every_count_from_zero_through_nine() {
        // 1-byte T: elem_align::<OneByte>() == 4 (see dynamic_buffer.rs's
        // own unit test) -- every count here either divides evenly into
        // that or doesn't, deliberately walking both cases.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<OneByte> = VarLenGpuPool::new(Arc::clone(&device), "test", 4);

        let mut prev = VarLenHandle::default();
        for count in 0..=9u8 {
            let data: Vec<OneByte> = (0..count).map(OneByte).collect();
            prev = pool.write_var_row(&queue, prev, &data).unwrap_or_else(|e| {
                panic!("count {count} failed to allocate/write: {e}")
            });
            assert_eq!(prev.count, count as u32);

            if count > 0 {
                // `copy_buffer_to_buffer` (what `readback` uses) has the
                // exact same COPY_BUFFER_ALIGNMENT size requirement as
                // `write_buffer` -- round the READBACK request up too; this
                // is a test-harness constraint, unrelated to the pool's own
                // (already-padded) internal storage.
                let readback_len = ((prev.offset + prev.count) as u64).next_multiple_of(4);
                let mut bytes = Vec::new();
                pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, readback_len));
                let got = &bytes[prev.offset as usize..(prev.offset + prev.count) as usize];
                let expected: Vec<u8> = (0..count).collect();
                assert_eq!(got, expected.as_slice(), "count {count} round-tripped the wrong bytes");
            }
        }
    }

    #[test]
    fn three_byte_elements_never_corrupt_a_neighboring_allocation() {
        // The sharpest version of the risk this fix exists for: T's size
        // (3) shares no common factor with 4 beyond 1, so naive offsets
        // land unaligned almost immediately. Two INTERLEAVED entities each
        // repeatedly resize -- if padding ever escaped its own reserved
        // span, this cross-contaminates the other entity's live bytes.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<ThreeBytes> = VarLenGpuPool::new(Arc::clone(&device), "test", 4);

        let row = |tag: u8, count: u8| -> Vec<ThreeBytes> {
            (0..count).map(|i| ThreeBytes([tag, i, tag.wrapping_add(i)])).collect()
        };

        let mut a = pool.write_var_row(&queue, VarLenHandle::default(), &row(0xAA, 3)).unwrap();
        let mut b = pool.write_var_row(&queue, VarLenHandle::default(), &row(0xBB, 1)).unwrap();

        for round in 0..12u8 {
            let a_count = 1 + (round * 7) % 5; // 1..=5, deliberately non-monotonic
            let b_count = 1 + (round * 3) % 4; // 1..=4
            a = pool.write_var_row(&queue, a, &row(0xAA, a_count)).unwrap();
            b = pool.write_var_row(&queue, b, &row(0xBB, b_count)).unwrap();

            let mut bytes = Vec::new();
            pool.with_buffer(&mut |buf| {
                bytes = readback(&device, &queue, buf, pool.capacity() as u64 * 3);
            });

            let a_bytes = &bytes[a.offset as usize * 3..(a.offset as usize + a.count as usize) * 3];
            let expected_a: Vec<u8> = row(0xAA, a_count).iter().flat_map(|e| e.0).collect();
            assert_eq!(a_bytes, expected_a.as_slice(), "round {round}: entity A's live bytes got clobbered");

            let b_bytes = &bytes[b.offset as usize * 3..(b.offset as usize + b.count as usize) * 3];
            let expected_b: Vec<u8> = row(0xBB, b_count).iter().flat_map(|e| e.0).collect();
            assert_eq!(b_bytes, expected_b.as_slice(), "round {round}: entity B's live bytes got clobbered");
        }
    }

    #[test]
    fn fixed_size_2d_blocks_per_row_stand_in_for_jagged_nested_data() {
        // `Vec<[u16; 3]>` -- each logical element is itself a small
        // fixed-shape row (a triangle's 3 UV components, a bone's 3
        // weights, whatever). 6 bytes/element, not a multiple of 4, so
        // this exercises the same padding machinery with a compound
        // element type rather than a single-field newtype.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<[u16; 3]> = VarLenGpuPool::new(Arc::clone(&device), "test", 4);

        let blocks: Vec<[u16; 3]> = vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let handle = pool.write_var_row(&queue, VarLenHandle::default(), &blocks).unwrap();
        assert_eq!(handle.count, 3);

        // Same COPY_BUFFER_ALIGNMENT rounding as the 1-byte-element test
        // above -- this is `readback`'s own copy size, not the pool's.
        let readback_len = ((handle.offset + handle.count) as u64 * 6).next_multiple_of(4);
        let mut bytes = Vec::new();
        pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, readback_len));
        let region = &bytes[handle.offset as usize * 6..(handle.offset as usize + 3) * 6];
        let got: Vec<[u16; 3]> = region
            .chunks_exact(6)
            .map(|c| [
                u16::from_ne_bytes([c[0], c[1]]),
                u16::from_ne_bytes([c[2], c[3]]),
                u16::from_ne_bytes([c[4], c[5]]),
            ])
            .collect();
        assert_eq!(got, blocks);

        // A second entity's block must land right after, not overlapping
        // the first's padded tail.
        let h2 = pool.write_var_row(&queue, VarLenHandle::default(), &[[10, 11, 12]]).unwrap();
        assert!(h2.offset >= handle.offset + handle.count, "second allocation must not overlap the first");
    }
}
