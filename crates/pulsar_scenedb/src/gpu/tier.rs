//! Tiered-storage substrate (SceneDB#61 §4–§5): intrinsic Disk ⇄ RAM ⇄ VRAM
//! residency for any field, fully internal to SceneDB. This module owns the
//! SHARED machinery every storage class rides — the demand verbs' bookkeeping,
//! the transition queue flush executes, the LRU/budget policy, the two
//! SceneDB-owned byte arenas (RAM staging + Disk spill), the pending-fetch
//! registry behind value-or-[`Pending`] CPU reads, and the SceneDB-owned
//! texture bindings promotions materialize into. Per-resource RESIDENCY
//! RECORDS deliberately do NOT live here: they live in each class's own
//! entry table, exactly where the issue puts them —
//! [`super::interned_pool::InternedVarLenPool`]'s per-id entries (static
//! class), [`super::var_len_pool::VarLenGpuPool`]'s freelist-parallel
//! sidecar (dynamic pool payloads), and this module's own lazy-grown
//! per-row tables (fixed-size mirrored rows — grown exactly like
//! `GenerationMirror`'s `gpu_mirrored_rows`).
//!
//! # Contract
//!
//! - **Tiers are exclusive, metadata-authoritative**: an entry's `tier` byte
//!   names the ONE tier the engine considers authoritative for its bytes.
//!   **Default tier on write = RAM** for all three storage classes: a write
//!   lands its resource at `Ram` (for pool payloads, a staging copy is taken
//!   at the write — the cheapest moment the bytes exist on the CPU; for
//!   fixed-size rows the existing dirty-tracked CPU shadow IS the RAM home,
//!   zero marginal bytes; for static/interned entries the record is
//!   metadata-only — see below). Absence of an entry READS AS `Ram`
//!   generation 0, so untouched resources cost nothing.
//! - **Transitions execute only at flush**: [`crate::gpu::SceneGpuStore::
//!   touch_tier`]/[`release_tier`](crate::gpu::SceneGpuStore::release_tier)
//!   enqueue intents (updating `last_use` immediately — that part is
//!   demand, not movement); the next flush executes them in queue order,
//!   liveness-guarded, budget-admitted. Touches issued before a submit are
//!   complete before it — the ordering promise downstream consumers rely on
//!   is simply flush ordering discipline: `World::flush_gpu_mirror` runs
//!   dirty-column uploads first, THEN tier transitions, all issued to the
//!   same in-order `wgpu::Queue`; anything queued before that call has its
//!   final tier state committed before the call returns (GPU-side effects
//!   are ordered behind it in queue submission order, like every other
//!   write this crate issues).
//! - **Liveness-guarded commits**: before committing any intent, flush
//!   re-validates the target's IDENTITY — the record still exists and, for
//!   pool slots, the selector's element count still matches. A resource
//!   freed mid-flight (an interned entry dropping to zero references, a
//!   var-len allocation freed, or its offset recycled to a different-size
//!   tenant) CANCELS the intent — nothing is uploaded, freed, or recorded;
//!   no orphaned ranges survive a cancelled transition (counted in
//!   [`TierStats::cancelled`]). Generation EQUALITY is deliberately NOT
//!   part of transition guarding: generations advance via the resource's
//!   OWN queued transitions, so strict equality would cancel benign
//!   same-id sequences racing their own execution (cross-thread flushes
//!   drain shards concurrently). Generations guard RESOLVES instead, where
//!   a moved-bytes aliasing hazard actually exists (#61 test 5).
//! - **Generation guards**: every committed DEMOTION bumps the entry's
//!   generation. Resolves carry the generation they validated against;
//!   resolving with a stale one errors loudly
//!   ([`TierError::StaleGeneration`]) — never silently aliases moved bytes
//!   (`tests/stale_handle.rs` precedent, applied to residency instead of
//!   slots).
//! - **Stable mapping**: promoting/demoting never invalidates a published
//!   bind-slot. Pool ranges come and go behind the pool's existing
//!   freelist; column buffers keep their identities; contents change behind
//!   generation-guarded handles. A promoted pool payload occupies a fresh
//!   range whose location readers discover through the same handles they
//!   already hold (`VarLenHandle` columns / `resolve`), not through addresses
//!   baked in before the flight.
//! - **LRU + budgets**: `last_use` is stamped on every touch/write from a
//!   process-wide monotonic clock. Budget admission for VRAM placements is
//!   checked per rank-unit at flush; overflow evicts least-recently-used
//!   DYNAMIC residents (never pinned static floors — statics are not
//!   candidates by construction) until it fits, declining loudly
//!   ([`TierError::BudgetExhausted`] surfaces as a declined transition in
//!   [`TierStats::declined_budget`]) only if even a fully-evicted budget
//!   cannot admit the unit. RAM-budget overflow is reconciled at flush by
//!   spilling least-recently-used STAGING to the disk arena (write-path
//!   staging itself never fails — it is best-effort cache; budgets converge
//!   at the boundary).
//! - **Static vs dynamic comes from mirror modes** (#61 §2): interned /
//!   `mirror = Once` entries are STATIC — recorded once at intern, pinned
//!   while referenced, demote only at drop-to-zero (which is #632's
//!   deterministic free removing the whole record). Their ordinary mirror
//!   upload is untouched — byte-identical behavior is the hard gate this
//!   design is shaped around; the tier engine owns NOTHING physical for a
//!   static entry, it only audits and pins. `DirtyTracked`-provenance
//!   resources are DYNAMIC — full runtime movement per the verbs.
//! - **CPU reads are tier-transparent**: [`TierPeek`] answers
//!   `Resident { generation }` or [`Pending`](TierPeek::Pending); completing
//!   a pending fetch after the flush executes its un-spill yields the bytes
//!   EXACTLY ONCE ([`TierError::AlreadyCompleted`] forever after — the
//!   fetch's completion slot is consumed, making double completion
//!   structurally impossible, not merely forbidden).
//! - **Concurrency shape**: arenas, the fetch registry, and the pending
//!   queue are sharded/locked independently; pool sidecars ride each pool's
//!   existing inner lock; row tables are per-column locks. No lock ordered
//!   engine-class ↔ pool-class ever nests (the engine holds no storage
//!   references at all — the store mediates every cross-class step), so
//!   there is no deadlock lattice to reason about beyond "leaf locks only".
//!
//! # Cost model
//!
//! - **Write paths**: pool payloads pay one memcpy + one sidecar-map op at
//!   `write_var_row` (the bytes are in registers anyway); fixed-size rows
//!   pay ZERO (lazy entries; absence = Ram); statics pay one small record
//!   fill at intern. Steady-state touch/release on existing entries: one
//!   clock increment, one entry-field write, one `Vec` push into a
//!   pre-warmed shard — allocation-free (amortized; shard vectors are
//!   reserved at configure time and recycle).
//! - **Flush**: linear in queued transitions + one global LRU candidate
//!   sort when budgets overflow (bulk boundary operation, deliberately NOT
//!   allocation-free — it is the batch path, mirroring how
//!   `execute_transitions` treats its own boundary work).
//! - **Memory**: RAM staging ≈ one copy per dynamic pool payload (rows ride
//!   shadows; statics ride nothing); disk arena grows only under spill;
//!   VRAM accounting tracks exactly the ranges the engine placed.
//!
//! # What is NOT covered (deliberately)
//!
//! No format parsing, no "mip" concept, no rendering concepts — ranks and
//! ranges only (see [`super::tier_layout`]); texture materialization treats
//! a bound texture purely as "a bound resource bytes land in", with
//! dimensions/format supplied wholesale by the one consumer config call.
//! No real file I/O: the Disk tier is a SceneDB-owned in-process spill
//! arena — pluggable remote storage is a downstream concern this crate
//! deliberately does not model. No GPU→CPU readback in this substrate:
//! CPU reads serve RAM/Disk-tier data (a static VRAM-only entry reads
//! through the pre-existing row-indexed mirrors, not through here).
//! Demotion does not scrub GPU bytes it leaves behind — it withdraws the
//! guarantee (generation bump + loud stale resolves); the next promotion
//! re-signs the contents.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::buffer_registry::BufferKey;
use super::tier_layout::{execution_plan, RankUnit};
use crate::component::ComponentId;
use crate::handle_ledger::HandleId;

/// Number of independent pending-transition shards. Sharding is the
/// contention story for the thrash regime (#61 test 7: alternating touches
/// across thousands of ids from many threads): two threads hashing to
/// different shards never serialize against each other at all. 64 makes a
/// collision unlikely at any plausible thread count without being worth
/// tuning.
const PENDING_SHARDS: usize = 64;

/// Sentinel arena-slot value meaning "no slot".
pub(crate) const NO_SLOT: u32 = u32::MAX;

// ── Public vocabulary ─────────────────────────────────────────────────────

/// A storage tier. Ordered along the promotion ladder: `Disk < Ram < Vram`,
/// so comparisons read naturally (`current < target` = needs promoting).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Tier {
    /// SceneDB-owned spill arena (process-lifetime bytes; see non-goals).
    Disk,
    /// RAM staging (arena-held for pool payloads; the existing dirty-tracked
    /// CPU shadow for fixed-size rows).
    Ram,
    /// GPU residency — a pool range, column row, or bound texture the
    /// engine placed and accounts.
    Vram,
}

impl Tier {
    /// One rung DOWN the ladder (`Vram -> Ram -> Disk`); `Disk` stays put.
    #[inline]
    pub fn lower(self) -> Option<Tier> {
        match self {
            Tier::Vram => Some(Tier::Ram),
            Tier::Ram => Some(Tier::Disk),
            Tier::Disk => None,
        }
    }

    /// The compact form stored in per-row/per-entry tier bytes.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn from_u8(b: u8) -> Option<Tier> {
        match b {
            0 => Some(Tier::Disk),
            1 => Some(Tier::Ram),
            2 => Some(Tier::Vram),
            _ => None,
        }
    }
}

/// Consumer-provided capacities — the ONE configuration call maps whatever
/// engine settings a consumer has into these two numbers. Zero elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierConfig {
    /// Ceiling on bytes the engine will place in VRAM (pool ranges it
    /// allocates + rows/columns it re-signs + texture materializations,
    /// all accounted identically).
    pub vram_budget_bytes: u64,
    /// Ceiling on bytes held as RAM staging (pool payloads; row shadows are
    /// the mirror's own memory and are not double-counted here).
    pub ram_budget_bytes: u64,
}

/// Which resource a verb targets — the `id | row` axis of the demand API.
/// Every variant names an EXISTING resource (created by an ordinary write);
/// verbs never create residency for something no field wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TierSelector {
    /// An interned content-addressed entry (the `id` form) — STATIC class.
    Interned {
        pool: BufferKey,
        id: HandleId,
    },
    /// One variable-length payload allocation in a plain (non-interned)
    /// pool — DYNAMIC class. `handle` is the same `VarLenHandle` the
    /// row-indexed handle-table column already carries (`..._gpu_handle`
    /// accessors hand it out); `count` participates in the identity check
    /// so a recycled offset cannot be mistaken for the old tenant.
    PoolSlot {
        pool: BufferKey,
        handle: super::var_len_pool::VarLenHandle,
    },
    /// One row of a fixed-size mirrored column — DYNAMIC class (the `row`
    /// form). `column` is the ComponentId the field was registered under
    /// (`component_id::<FieldTypeWrapper>` — the same id every other
    /// per-row API in this crate keys on).
    Row { column: ComponentId, row: u32 },
}

/// How much of a segmented payload a verb covers — the `range | whole`
/// axis, expressed in the only vocabulary this crate has: ranks.
///
/// [`TierSpan::Whole`] covers every rank; [`TierSpan::ThroughRank`]
/// extends/trimms the residency PREFIX through one rank (inclusive),
/// preserving the prefix-complete invariant regardless of request order.
/// For an atomic (zero-declaration) payload both forms are identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierSpan {
    Whole,
    ThroughRank(u32),
}

/// Loud typed errors — stale resolves error, never silently alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierError {
    /// No `configure_tiers` call yet (or the named pool/column was never
    /// registered through a tier-participating path).
    NotConfigured,
    /// The selector names something that does not exist (never written,
    /// already freed at zero references, or a stale recycled offset/count
    /// pair). Loud on purpose.
    UnknownResource,
    /// A STATIC (interned/Once-provenance) entry was asked to move while
    /// its reference count is above zero — pinned floors are never chosen,
    /// by request or by eviction.
    Pinned,
    /// Resolve guarded by `expected` found `current` — the resource was
    /// demoted (and/or its slot reused) since the caller validated.
    StaleGeneration { expected: u64, current: u64 },
    /// A [`PoolSlot`](TierSelector::PoolSlot) selector's `handle.count`
    /// disagrees with the live allocation it names — a recycled/stale
    /// handle, rejected before anything else consults it.
    StaleHandle,
    /// `tier_read`'s output window doesn't match the payload's byte length.
    LengthMismatch { expected: u64, provided: u64 },
    /// Even after evicting every eligible LRU victim, the tier cannot admit
    /// the requested unit (everything left is pinned, or the budget is
    /// smaller than the unit). Reported per declined transition.
    BudgetExhausted { needed: u64, budget: u64 },
    /// This tier fetch was already completed once — completion is
    /// exactly-once by construction; the second attempt lands here forever.
    AlreadyCompleted,
    /// The fetch's transition has not executed yet — flush first.
    NotYetReady,
    /// CPU reads are not served for this entry's current state (e.g. a
    /// static VRAM-only entry — see module non-goals).
    ReadUnsupported,
}

impl std::fmt::Display for TierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TierError::NotConfigured => write!(f, "tier subsystem not configured (call configure_tiers first)"),
            TierError::UnknownResource => write!(f, "selector names no live tiered resource"),
            TierError::Pinned => write!(f, "entry is a pinned static (referenced); demotion refused"),
            TierError::StaleGeneration { expected, current } => {
                write!(f, "stale resolve: expected generation {expected}, current is {current}")
            }
            TierError::StaleHandle => write!(f, "stale var-len handle: count disagrees with the live allocation"),
            TierError::LengthMismatch { expected, provided } => {
                write!(f, "tier_read output window is {provided} bytes; payload is {expected}")
            }
            TierError::BudgetExhausted { needed, budget } => {
                write!(f, "budget exhausted: need {needed} bytes against budget {budget} with nothing evictable left")
            }
            TierError::AlreadyCompleted => write!(f, "tier fetch already completed (completion is exactly-once)"),
            TierError::NotYetReady => write!(f, "tier fetch not yet executed -- flush first"),
            TierError::ReadUnsupported => write!(f, "CPU reads unsupported for this entry's residency (VRAM-only static)"),
        }
    }
}

impl std::error::Error for TierError {}

/// What [`SceneGpuStore::tier_peek`](super::SceneGpuStore::tier_peek)
/// answers — the tier-transparent access contract's value-or-Pending shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierPeek {
    /// Bytes are readable now; `generation` is what [`tier_read`] must be
    /// handed to stay guarded.
    Resident { generation: u64 },
    /// Bytes sit below RAM (disk-spilled); the un-spill is queued and the
    /// next flush executes it. Complete the carried fetch afterwards.
    Pending(TierFetch),
}

/// Opaque handle to one queued lower-tier fetch. Completion is
/// exactly-once: the ready-state slot it names is consumed by the first
/// successful completion and can never be handed out twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TierFetch {
    pub(crate) ticket: u64,
}

/// Outcome tally for one flush's transition execution — the audit surface
/// the adversarial matrix asserts exact values against.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TierStats {
    /// Promotions committed (whole flights count once per rank-unit
    /// actually committed).
    pub promoted: u32,
    /// Demotions committed (one rung each; reverse-rank order within a
    /// flight).
    pub demoted: u32,
    /// Staging spills RAM-arena → disk-arena.
    pub spilled: u32,
    /// Spills reversed (demand returned, or a pending fetch executing).
    pub unspilled: u32,
    /// Intents cancelled by the liveness guard (target died / identity
    /// changed mid-flight). No ranges, no records touched.
    pub cancelled: u32,
    /// Promotions declined by budget admission after LRU evacuation ran.
    pub declined_budget: u32,
    /// Staging entries spilled by RAM-budget reconciliation at end of
    /// flush (subset of `spilled` — counted separately so callers can
    /// tell demand-driven from pressure-driven movement).
    pub pressure_spilled: u32,
}

impl std::ops::AddAssign for TierStats {
    fn add_assign(&mut self, rhs: Self) {
        self.promoted += rhs.promoted;
        self.demoted += rhs.demoted;
        self.spilled += rhs.spilled;
        self.unspilled += rhs.unspilled;
        self.cancelled += rhs.cancelled;
        self.declined_budget += rhs.declined_budget;
        self.pressure_spilled += rhs.pressure_spilled;
    }
}

// ── Internal records ──────────────────────────────────────────────────────

/// A half-open byte range in whichever address space the holder says
/// (payload-relative offsets, pool-buffer offsets, …). Unit-agnostic like
/// [`super::freelist::RangeList`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

/// One pending intent, captured at verb time and committed-or-cancelled at
/// flush.
/// (creation-tolerant row intents), `Some(g)` = "must exist WITH that
/// exact generation".
#[derive(Clone, Debug)]
pub(crate) struct PendingTransition {
    pub target: TierSelector,
    pub kind: TransitionKind,
    /// Verb-time engine-clock stamp — the per-record freshness ticket the
    /// executor claims before committing (cross-batch stale rejection).
    pub seq: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransitionKind {
    /// Extend the VRAM residency prefix THROUGH this rank (per-rank
    /// admission; ascending execution).
    Promote { through_rank: u32 },
    /// Withdraw VRAM residency — how far is the extent:
    Demote(DemoteExtent),
    /// Move staging from the RAM arena to the disk arena.
    Spill,
    /// Move staging back from the disk arena to the RAM arena.
    Unspill,
    /// ABSOLUTE demand captured at verb time: move the resource to exactly
    /// this tier (through_rank applies when tier == Vram). Because the
    /// target is absolute, batches of these fold by LAST-WINS regardless
    /// of cross-batch execution order.
    SetTo { tier: Tier, through_rank: u32 },
}

/// Folds one target's drained intent sequence into its net effect. Under
/// concurrent flushes, per-id FIFO across BATCH boundaries cannot hold
/// (drain-then-execute lets batch-B execute an intent while batch-A still
/// holds an earlier sibling), so raw sequential replay is unsound.
/// Absolute-demand intents ([`TransitionKind::SetTo`]) commute under
/// LAST-WINS — whichever verb issued last observed the freshest live state,
/// so its target is authoritative. Relative intents (rank-scoped partial
/// demotes) do NOT commute; groups containing them execute sequentially,
/// unfolded, with the documented limitation that their cross-batch ordering
/// relies on same-id FIFO within a single drain.
pub(crate) fn fold_kinds(kinds: &[TransitionKind]) -> Vec<TransitionKind> {
    if kinds.len() <= 1 {
        return kinds.to_vec();
    }
    let all_absolute = kinds.iter().all(|k| matches!(k, TransitionKind::SetTo { .. }));
    if all_absolute {
        match kinds.last() {
            Some(last) => vec![*last],
            None => Vec::new(),
        }
    } else {
        kinds.to_vec()
    }
}
/// How much VRAM residency a demotion withdraws. All extents pop rank
/// units in DESCENDING-rank order (reverse-rank eviction, #61 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DemoteExtent {
    /// Pop every resident rank unit (full demote to the backing tier).
    All,
    /// Pop units ranked ABOVE `rank`, leaving the prefix through it.
    AboveRank(u32),
}

/// One row's tier state in a per-column lazy-grow table. Fields are
/// parallel `Vec`s grown exactly like every other row-indexed CPU shadow
/// in this crate (`GenerationMirror.gpu_mirrored_rows` idiom); a row past
/// the current length READS AS `{Ram, gen 0}` without materializing.
pub(crate) struct RowTierTable {
    /// `Tier::as_u8()` per row.
    pub tier: Vec<u8>,
    /// Bumped on every committed demotion/spill touching the row.
    pub generation: Vec<u64>,
    /// Last-demand stamp (engine clock).
    pub last_use: Vec<u64>,
    /// Resident VRAM rank-prefix watermark per row (0 = none resident…
    /// for ATOMIC rows residency is binary: `seg_ranges` empty or one
    /// rank-0 range).
    pub resident_through: Vec<u32>,
    /// Committed VRAM ranges per row, `(rank, range)` ascending by rank —
    /// empty for non-`Vram` rows. Small per-row allocation only for rows
    /// that actually promoted (a boundary event, never hot-path).
    pub seg_ranges: Vec<Vec<(u32, ByteRange)>>,
    /// Whether the row has a disk-arena export (`disk_slot` valid).
    pub spilled: Vec<bool>,
    pub disk_slot: Vec<u32>,
    /// True once ANY tier op touched this row (distinguishes "explicitly
    /// known Ram" from "past-the-end default" for audit exactness).
    pub present: Vec<bool>,
    /// Newest executed intent-sequence stamp per row (freshness filter
    /// for cross-batch stale intents).
    pub last_seq: Vec<u64>,
}

impl RowTierTable {
    pub(crate) fn new() -> Self {
        Self {
            tier: Vec::new(),
            generation: Vec::new(),
            last_use: Vec::new(),
            resident_through: Vec::new(),
            seg_ranges: Vec::new(),
            spilled: Vec::new(),
            disk_slot: Vec::new(),
            present: Vec::new(),
            last_seq: Vec::new(),
        }
    }

    /// Grows all parallel columns to cover `row`, seeding defaults
    /// (`Ram`, gen 0, nothing resident). Returns the row index.
    pub(crate) fn grow_to(&mut self, row: usize) {
        if row >= self.tier.len() {
            let n = row + 1 - self.tier.len();
            self.tier.extend(std::iter::repeat_n(Tier::Ram.as_u8(), n));
            self.generation.extend(std::iter::repeat_n(0u64, n));
            self.last_use.extend(std::iter::repeat_n(0u64, n));
            self.resident_through.extend(std::iter::repeat_n(0u32, n));
            self.seg_ranges.extend(std::iter::repeat_n(Vec::new(), n));
            self.spilled.extend(std::iter::repeat_n(false, n));
            self.disk_slot.extend(std::iter::repeat_n(NO_SLOT, n));
            self.present.extend(std::iter::repeat_n(false, n));
            self.last_seq.extend(std::iter::repeat_n(0u64, n));
        }
    }
}

/// A consumer-declared "materialize promoted pool payloads of this pool
/// into a SceneDB-owned texture" binding. The engine CREATES and OWNS the
/// `wgpu::Texture` (C0/§10-G4 ownership precedent: SceneDB-owned textures
/// survive renderer teardown); the binding is declared ONCE inside the one
/// `configure_tiers` call — no per-use ceremony anywhere.
pub struct MaterializationSpec {
    /// The pool whose promoted payloads materialize into the texture.
    pub source_pool: BufferKey,
    /// Texel format. Only uncompressed (1×1-block) formats are accepted —
    /// same scope line `TextureStore` draws (block-compressed formats need
    /// block-aware row arithmetic this substrate deliberately does not
    /// model).
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

pub(crate) struct TextureBinding {
    pub texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
    /// Bytes per texel row (format's copy size × width) — validated a
    /// multiple of 256 at bind time (`write_texture`'s own requirement).
    pub bytes_per_texel_row: u64,
}

enum FetchState {
    /// Un-spill queued, not yet executed.
    Queued { target: TierSelector },
    /// Executed: bytes readable from the named place. Consumed on first
    /// completion — the exactly-once mechanism.
    Ready { target: TierSelector },
    /// Completed once; permanently terminal.
    Consumed,
}

// ── Arenas ────────────────────────────────────────────────────────────────

/// SceneDB-owned slab arena holding boxed byte payloads behind recycled
/// u32 slots. Identical shape for the RAM-staging and Disk-spill homes;
/// which is which is the holder's business. `put`/`take` move Boxes — a
/// spill is a pointer handoff between two of these, never a copy.
pub(crate) struct ByteArena {
    slots: Vec<Option<Box<[u8]>>>,
    free: Vec<u32>,
    pub used: u64,
}

impl ByteArena {
    fn new() -> Self {
        Self { slots: Vec::new(), free: Vec::new(), used: 0 }
    }

    fn put(&mut self, bytes: Box<[u8]>) -> u32 {
        self.used += bytes.len() as u64;
        match self.free.pop() {
            Some(idx) => {
                self.slots[idx as usize] = Some(bytes);
                idx
            }
            None => {
                self.slots.push(Some(bytes));
                (self.slots.len() - 1) as u32
            }
        }
    }

    fn take(&mut self, slot: u32) -> Option<Box<[u8]>> {
        let bytes = self.slots.get_mut(slot as usize)?.take()?;
        self.used -= bytes.len() as u64;
        self.free.push(slot);
        Some(bytes)
    }

    /// Borrows the slot's full byte slice (windowed copies build on this).
    fn slot_slice(&self, slot: u32) -> Option<&[u8]> {
        self.slots.get(slot as usize)?.as_ref().map(|b| b.as_ref())
    }

    /// Frees without yielding the bytes (drop-on-floor for cancelled work).
    fn discard(&mut self, slot: u32) -> Option<u64> {
        let bytes = self.slots.get_mut(slot as usize)?.take()?;
        self.used -= bytes.len() as u64;
        self.free.push(slot);
        Some(bytes.len() as u64)
    }
}

// ── The engine ────────────────────────────────────────────────────────────

/// Shared tier machinery — see the module doc for the division of labor
/// with the per-class entry tables. Holds NO storage-class references (the
/// store mediates every cross-class step), which is what keeps the lock
/// discipline trivial: every lock this type touches is a leaf.
pub(crate) struct TierEngine {
    config: RwLock<Option<TierConfig>>,
    /// Monotonic last-use clock. `Relaxed`: stamps feed only LRU ORDERING
    /// within one engine; no other synchronization depends on them.
    clock: AtomicU64,
    ram: Mutex<ByteArena>,
    disk: Mutex<ByteArena>,
    /// Pending intents, sharded by target hash. Reserved generously at
    /// configure; steady-state pushes recycle capacity (allocation-free).
    shards: [Mutex<Vec<PendingTransition>>; PENDING_SHARDS],
    fetches: Mutex<HashMap<u64, FetchState>>,
    next_ticket: AtomicU64,
    /// SceneDB-owned materialization targets, keyed by SOURCE POOL.
    textures: RwLock<HashMap<BufferKey, Arc<TextureBinding>>>,
    /// Bytes the engine currently accounts as VRAM-placed.
    vram_used: AtomicU64,
    ram_used_target: AtomicU64,
    /// Fixed-size mirrored rows' residency tables, per column. This IS the
    /// "one tier byte per row" entry table of #61 §1 for fixed-size
    /// fields — grown lazily exactly like every other row-indexed CPU
    /// shadow in this crate (`GenerationMirror.gpu_mirrored_rows` idiom),
    /// with the remaining per-row fields in parallel columns.
    rows: RwLock<HashMap<ComponentId, Arc<Mutex<RowTierTable>>>>,
}

impl TierEngine {
    pub(crate) fn new() -> Self {
        Self {
            config: RwLock::new(None),
            clock: AtomicU64::new(1),
            ram: Mutex::new(ByteArena::new()),
            disk: Mutex::new(ByteArena::new()),
            shards: std::array::from_fn(|_| Mutex::new(Vec::new())),
            fetches: Mutex::new(HashMap::new()),
            next_ticket: AtomicU64::new(1),
            textures: RwLock::new(HashMap::new()),
            vram_used: AtomicU64::new(0),
            ram_used_target: AtomicU64::new(0),
            rows: RwLock::new(HashMap::new()),
        }
    }

    /// Installs the one-time configuration. Reconfiguration REPLACES the
    /// budgets (documented escape hatch for consumers re-sizing mid-run);
    /// it never touches live entries.
    pub(crate) fn configure(&self, cfg: TierConfig, shard_warmup: usize) {
        *self.config.write().expect("tier config lock poisoned") = Some(cfg);
        for shard in &self.shards {
            shard.lock().expect("tier shard lock poisoned").reserve(shard_warmup);
        }
    }

    pub(crate) fn configured(&self) -> Option<TierConfig> {
        *self.config.read().expect("tier config lock poisoned")
    }

    /// Demand stamp — called on EVERY touch/write-path staging so LRU
    /// reflects true recency.
    #[inline]
    pub(crate) fn stamp(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Freshness ticket for an outgoing intent (`seq`). Same clock as
    /// [`Self::stamp`] — strictly increasing, globally comparable.
    #[inline]
    pub(crate) fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    // ── Arena plumbing (leaf-locked; never calls outward) ────────────────

    fn stage_ram(&self, bytes: Box<[u8]>) -> u32 {
        let mut ram = self.ram.lock().expect("tier ram arena poisoned");
        let slot = ram.put(bytes);
        self.ram_used_target.store(ram.used, Ordering::Relaxed);
        slot
    }

    fn take_ram(&self, slot: u32) -> Option<Box<[u8]>> {
        let mut ram = self.ram.lock().expect("tier ram arena poisoned");
        let out = ram.take(slot);
        self.ram_used_target.store(ram.used, Ordering::Relaxed);
        out
    }

    /// Drops a RAM staging slot outright — the fate of a freed allocation's
    /// staged copy (the allocation is gone; nobody can ever demand it).
    pub(crate) fn discard_ram(&self, ram_slot: u32) {
        let mut ram = self.ram.lock().expect("tier ram arena poisoned");
        let _ = ram.discard(ram_slot);
        self.ram_used_target.store(ram.used, Ordering::Relaxed);
    }

    /// Drops a disk export outright (un-spill consumed it / record freed).
    pub(crate) fn discard_disk_export(&self, disk_slot: u32) {
        let mut disk = self.disk.lock().expect("tier disk arena poisoned");
        let _ = disk.discard(disk_slot);
    }

    /// Write-path staging with box reuse: if `prev` names a live RAM slot
    /// whose physical capacity fits `bytes`, overwrite it in place — no
    /// allocation, which is what keeps steady-state pool rewrites
    /// allocation-free. Otherwise discard `prev` and put fresh. Returns the
    /// slot now holding (at least) `bytes`.
    pub(crate) fn restage_ram(&self, prev: Option<u32>, bytes: &[u8]) -> u32 {
        let mut ram = self.ram.lock().expect("tier ram arena poisoned");
        if let Some(idx) = prev {
            if idx < ram.slots.len() as u32 {
                if let Some(Some(existing)) = ram.slots.get_mut(idx as usize) {
                    if existing.len() >= bytes.len() {
                        existing[..bytes.len()].copy_from_slice(bytes);
                        return idx;
                    }
                }
            }
            let _ = ram.discard(idx);
        }
        let out = ram.put(bytes.to_vec().into_boxed_slice());
        self.ram_used_target.store(ram.used, Ordering::Relaxed);
        out
    }

    /// Windowed copy out of RAM staging (payload-relative offset).
    pub(crate) fn copy_ram_window(&self, slot: u32, rel: u64, out: &mut [u8]) -> bool {
        let ram = self.ram.lock().expect("tier ram arena poisoned");
        let Some(bytes) = ram.slot_slice(slot) else { return false };
        let start = rel as usize;
        let end = start + out.len();
        if end > bytes.len() {
            return false;
        }
        out.copy_from_slice(&bytes[start..end]);
        true
    }

    /// Windowed copy out of a DISK export (the spilled-staging read path).
    pub(crate) fn copy_disk_window(&self, slot: u32, rel: u64, out: &mut [u8]) -> bool {
        let disk = self.disk.lock().expect("tier disk arena poisoned");
        let Some(bytes) = disk.slot_slice(slot) else { return false };
        let start = rel as usize;
        let end = start + out.len();
        if end > bytes.len() {
            return false;
        }
        out.copy_from_slice(&bytes[start..end]);
        true
    }

    // ── Row-table access ─────────────────────────────────────────────────

    /// `column`'s row-tier table, creating it if absent. The Arc lets the
    /// store hold the inner lock briefly while engine bookkeeping stays
    /// lock-free on this side.
    pub(crate) fn row_table(&self, column: ComponentId) -> Arc<Mutex<RowTierTable>> {
        {
            let rows = self.rows.read().expect("tier row-table map poisoned");
            if let Some(t) = rows.get(&column) {
                return Arc::clone(t);
            }
        }
        let mut rows = self.rows.write().expect("tier row-table map poisoned");
        rows.entry(column).or_insert_with(|| Arc::new(Mutex::new(RowTierTable::new()))).clone()
    }

    pub(crate) fn row_tables(&self) -> Vec<(ComponentId, Arc<Mutex<RowTierTable>>)> {
        self.rows.read().expect("tier row-table map poisoned").iter().map(|(&c, t)| (c, Arc::clone(t))).collect()
    }

    /// Test-only staging entry (production callers go through the pool
    /// write path's `restage_ram`).
    #[cfg(test)]
    pub(crate) fn stage_ram_for_test(&self, bytes: Vec<u8>) -> u32 {
        self.stage_ram(bytes.into_boxed_slice())
    }

    pub(crate) fn spill_ram_to_disk(&self, ram_slot: u32) -> (u32, u64) {
        let bytes = self.take_ram(ram_slot).expect("spill of a live staging slot");
        let len = bytes.len() as u64;
        let mut disk = self.disk.lock().expect("tier disk arena poisoned");
        (disk.put(bytes), len)
    }

    pub(crate) fn unspill_disk_to_ram(&self, disk_slot: u32) -> (u32, u64) {
        let bytes = {
            let mut disk = self.disk.lock().expect("tier disk arena poisoned");
            disk.take(disk_slot).expect("unspill of a live export slot")
        };
        let len = bytes.len() as u64;
        let slot = self.stage_ram(bytes);
        (slot, len)
    }

    /// Exports a fixed-size row's demotion-time snapshot into the disk
    /// arena (rows' RAM home — the shadow — stays where it is; the export
    /// preserves the value AS OF the spill against later mutations).
    pub(crate) fn export_row_snapshot(&self, bytes: Vec<u8>) -> (u32, u64) {
        let len = bytes.len() as u64;
        let mut disk = self.disk.lock().expect("tier disk arena poisoned");
        let slot = disk.put(bytes.into_boxed_slice());
        (slot, len)
    }

    /// Drops a row's disk export (un-spill consumed it).
    pub(crate) fn drop_row_export(&self, disk_slot: u32) {
        let mut disk = self.disk.lock().expect("tier disk arena poisoned");
        let _ = disk.discard(disk_slot);
    }

    // ── VRAM accounting ──────────────────────────────────────────────────

    /// Attempts to reserve `bytes` of VRAM headroom against the configured
    /// budget WITHOUT evicting anyone — the caller (store flush) layers LRU
    /// evacuation around retries of this.
    pub(crate) fn try_reserve_vram(&self, bytes: u64) -> Result<(), TierError> {
        let budget = self.configured().ok_or(TierError::NotConfigured)?.vram_budget_bytes;
        let mut cur = self.vram_used.load(Ordering::Relaxed);
        loop {
            if cur + bytes <= budget {
                match self.vram_used.compare_exchange_weak(
                    cur,
                    cur + bytes,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Ok(()),
                    Err(actual) => cur = actual,
                }
            } else {
                return Err(TierError::BudgetExhausted { needed: bytes, budget });
            }
        }
    }

    pub(crate) fn release_vram(&self, bytes: u64) {
        let prev = self.vram_used.fetch_sub(bytes, Ordering::Relaxed);
        debug_assert!(prev >= bytes, "vram accounting underflow");
    }

    pub(crate) fn vram_used(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    pub(crate) fn ram_used(&self) -> u64 {
        self.ram_used_target.load(Ordering::Relaxed)
    }

    // ── Pending-intent queue ─────────────────────────────────────────────

    fn shard_of(target: &TierSelector) -> usize {
        let mut h = target_hash_seed(target);
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        (h % PENDING_SHARDS as u64) as usize
    }

    pub(crate) fn queue_transition(&self, t: PendingTransition) {
        let shard = &self.shards[Self::shard_of(&t.target)];
        shard.lock().expect("tier shard lock poisoned").push(t);
    }

    /// Drains every shard, invoking `f` on each non-empty batch WHILE THE
    /// SHARD LOCK IS HELD. Holding through execution is load-bearing: it
    /// guarantees strict per-id FIFO across overlapping flushes (a later
    /// flush cannot drain and execute a shard's newer intents before an
    /// earlier flush finishes an older sibling), which is what makes
    /// absolute-target intents sound without per-record sequence tracking
    /// beyond the belt-and-braces seq filter. Lock order stays leaf-only:
    /// callbacks may take pool/class locks, never another shard's.
    pub(crate) fn for_each_shard_batch(&self, mut f: impl FnMut(Vec<PendingTransition>)) {
        for shard in &self.shards {
            let mut guard = shard.lock().expect("tier shard lock poisoned");
            if guard.is_empty() {
                continue;
            }
            // `drain`, not `mem::take`: taking would discard the shard
            // vector's CAPACITY every flush, turning steady-state pushes
            // back into reallocations (§8.1 violation). Draining keeps the
            // allocation resident and reusable.
            let batch: Vec<PendingTransition> = guard.drain(..).collect();
            f(batch);
        }
    }

    // ── Fetch registry (exactly-once completions) ────────────────────────

    pub(crate) fn open_fetch(&self, target: TierSelector) -> TierFetch {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        self.fetches
            .lock()
            .expect("tier fetch registry poisoned")
            .insert(ticket, FetchState::Queued { target });
        TierFetch { ticket }
    }

    /// Marks every fetch targeting `sel` ready (its un-spill executed).
    pub(crate) fn mark_fetches_ready(&self, sel: &TierSelector) {
        let mut fetches = self.fetches.lock().expect("tier fetch registry poisoned");
        for state in fetches.values_mut() {
            if let FetchState::Queued { target } = state {
                if target == sel {
                    *state = FetchState::Ready { target: *target };
                }
            }
        }
    }

    /// Exactly-once completion: `Queued` → [`TierError::NotYetReady`],
    /// `Ready` → consumed + Ok, `Consumed`/unknown → [`TierError::
    /// AlreadyCompleted`].
    pub(crate) fn claim_fetch(&self, fetch: &TierFetch) -> Result<TierSelector, TierError> {
        let mut fetches = self.fetches.lock().expect("tier fetch registry poisoned");
        match fetches.get_mut(&fetch.ticket) {
            Some(FetchState::Queued { .. }) => Err(TierError::NotYetReady),
            Some(state @ FetchState::Ready { .. }) => {
                let FetchState::Ready { target } = std::mem::replace(state, FetchState::Consumed) else {
                    unreachable!("matched Ready arm above")
                };
                Ok(target)
            }
            Some(FetchState::Consumed) | None => Err(TierError::AlreadyCompleted),
        }
    }

    // ── Materialization bindings ─────────────────────────────────────────

    pub(crate) fn texture_binding(&self, pool: &BufferKey) -> Option<Arc<TextureBinding>> {
        self.textures.read().expect("tier textures lock poisoned").get(pool).cloned()
    }

    /// Validates and installs one binding, creating the SceneDB-owned
    /// texture. Loud on inconsistent geometry (see [`MaterializationSpec`]).
    pub(crate) fn bind_texture(
        &self,
        device: &wgpu::Device,
        spec: &MaterializationSpec,
        max_payload_bytes: Option<u64>,
    ) -> Result<(), TierError> {
        if spec.format.block_dimensions() != (1, 1) {
            // Same scope line TextureStore draws — see its register() doc.
            return Err(TierError::ReadUnsupported);
        }
        let block_size = spec
            .format
            .block_copy_size(None)
            .ok_or(TierError::ReadUnsupported)?;
        let bpr = block_size as u64 * spec.width as u64;
        const ROW_ALIGN: u64 = 256; // == wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        if bpr & (ROW_ALIGN - 1) != 0 {
            // `write_texture`'s own hard requirement — reject the binding at
            // configure time (loud) rather than panicking mid-flight later.
            return Err(TierError::LengthMismatch { expected: bpr.next_multiple_of(ROW_ALIGN), provided: bpr });
        }
        let total = bpr * spec.height as u64;
        if let Some(maxb) = max_payload_bytes {
            if total != maxb {
                return Err(TierError::ReadUnsupported);
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scenedb-tier-materialization"),
            size: wgpu::Extent3d { width: spec.width, height: spec.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: spec.format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let binding = TextureBinding {
            texture,
            width: spec.width,
            height: spec.height,
            bytes_per_texel_row: bpr,
        };
        self.textures
            .write()
            .expect("tier textures lock poisoned")
            .insert(spec.source_pool, Arc::new(binding));
        Ok(())
    }
}

/// Cheap deterministic seed for shard selection — mixes the selector's
/// discriminant + payload words directly (no HashMap dependency; the same
/// FxHash-style finalizer [`TierEngine::shard_of`] applies).
fn target_hash_seed(t: &TierSelector) -> u64 {
    fn mix_str(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
    match *t {
        TierSelector::Interned { pool, id } => {
            0x1000 ^ (mix_str(pool.as_str()) ^ ((id.0 as u64).wrapping_mul(0x9E3779B97F4A7C15)))
        }
        TierSelector::PoolSlot { pool, handle } => {
            0x2000 ^ (mix_str(pool.as_str()) ^ (handle.offset as u64).wrapping_mul(0xBF58476D1CE4E5B9))
        }
        TierSelector::Row { column, row } => {
            0x4000 ^ ((column.0 as u64).wrapping_mul(0x94D049BB133111EB) ^ row as u64)
        }
    }
}

/// Builds the rank-unit plan for one promotion flight: registered layout if
/// the payload type has one, atomic default otherwise. Shared by pool and
/// row executors.
pub(crate) fn flight_plan(payload_type: TypeId, payload_bytes: u64) -> Vec<RankUnit> {
    execution_plan(payload_type, payload_bytes)
}

// ── Audit vocabulary + class seams ────────────────────────────────────────

/// Which resource an audit record describes — mirrors [`TierSelector`] but
/// with plain owned values (no handle needed once identity is settled).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TierAuditKey {
    Interned(BufferKey, HandleId),
    PoolSlot(BufferKey, u64),
    Row(ComponentId, u32),
}

/// One resource's exact residency state — what every adversarial audit
/// asserts against. Produced by `SceneGpuStore::tier_audit`, sorted by key
/// for deterministic comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierAuditRecord {
    pub key: TierAuditKey,
    pub tier: Tier,
    pub generation: u64,
    pub last_use: u64,
    /// STATIC entries pinned while referenced (never true for dynamics).
    pub pinned: bool,
    /// Bytes the engine accounts as placed in VRAM for this entry.
    pub vram_bytes: u64,
    /// Bytes held as RAM staging for this entry (0 for rows — their RAM
    /// home is the mirror's own shadow).
    pub staging_bytes: u64,
}

/// A pool-slot/row LRU candidate harvested during budget reconciliation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VictimCandidate {
    pub last_use: u64,
    pub vram_bytes: u64,
    /// Identity of the eviction callback — resolved by the store (which
    /// owns the class registries), never by the engine.
    pub class: VictimClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VictimClass {
    PoolSlot { pool: BufferKey, offset: u64 },
    Row { column: ComponentId, row: u32 },
}
/// Snapshot of a static (interned) entry's tier-relevant state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StaticSnapshot {
    pub last_use: u64,
    pub byte_len: u64,
}

/// One dynamic pool-slot's full residency state — the single probe verbs
/// and the liveness guard both consume (one lock acquisition, not four).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotState {
    /// Monotonic identity generation (never resets across offset reuse).
    pub generation: u64,
    /// True element count of the live allocation (selector identity half).
    pub count: u32,
    /// Current authoritative residency.
    pub tier: Tier,
    /// Staged payload bytes (wherever staging currently lives).
    pub staged_len: u64,
    /// Staging currently in the disk arena rather than the RAM arena.
    pub spilled: bool,
    /// Bytes currently placed in VRAM for this slot.
    pub vram_bytes: u64,
    /// Last-demand stamp.
    pub last_use: u64,
}

/// The type-erased seam between the store's flush/verb layer and one
/// registered PLAIN var-len pool's sidecar. Object-safe by construction;
/// implemented by `VarLenGpuPool<T>` for its concrete `T`. The engine never
/// sees these — the store mediates, which is what keeps lock ordering
/// leaf-only (pool inner locks are leaves; engine locks are leaves; nothing
/// nests them in the opposite order anywhere).
pub(crate) trait TieredPoolHooks: Send + Sync {
    /// The payload TYPE layouts are authored against (the element type).
    fn payload_type(&self) -> TypeId;

    /// One-lock snapshot of the slot's residency, or `None` when no live
    /// record exists at `offset` (the liveness guard's "already dead").
    fn slot_state(&self, offset: u64) -> Option<SlotState>;

    /// Re-stamps the slot's `last_use` with the engine clock (demand
    /// observed) and RETURNS the stamp for use as the intent's freshness
    /// ticket. 0 if the record vanished.
    fn stamp_slot(&self, offset: u64) -> u64;

    /// Claims freshness for an intent stamped `seq`: succeeds (recording
    /// `seq` as the newest applied) iff `seq` is newer than anything already
    /// executed for this slot — the cross-batch stale-intent filter.
    fn claim_seq(&self, offset: u64, seq: u64) -> bool;

    /// Copies `out.len()` bytes of staged payload starting at payload-
    /// relative `rel` into `out`, from wherever staging currently lives
    /// (RAM arena or disk export). `false` = window out of bounds.
    fn copy_staged_window(&self, offset: u64, rel: u64, out: &mut [u8]) -> bool;

    /// Admits ONE rank unit for the slot: allocates a pool range sized
    /// for `packed`, uploads it at the payload-relative positions named by
    /// `slices` (each `(start, len)` byte-aligned to 4), records
    /// `(rank, range)` in the sidecar, and returns the reserved range.
    /// Caller has already passed budget admission and validated liveness.
    fn admit_rank_unit(
        &self,
        queue: &wgpu::Queue,
        offset: u64,
        rank: u32,
        packed: &[u8],
        slices: &[(u64, u64)],
    ) -> ByteRange;

    /// Withdraws the highest resident rank unit (reverse-rank demotion):
    /// frees its pool range, clears its record entry, bumps the generation.
    /// Returns the freed range so the caller can release VRAM accounting.
    fn withdraw_top_unit(&self, offset: u64) -> Option<(u32, ByteRange)>;

    /// Marks the slot's backing tier moved RAM→Disk (`staging` swaps to the
    /// disk arena); returns nothing — accounting lives engine-side.
    fn spill_staging(&self, offset: u64);

    /// Marks the slot's backing tier moved Disk→RAM.
    fn unspill_staging(&self, offset: u64);

    /// LRU candidates: `(offset, last_use, vram_bytes)` per VRAM-resident
    /// slot. Bulk boundary operation — a small Vec clone is expected here.
    fn harvest_vram_candidates(&self) -> Vec<(u64, u64, u64)>;

    /// The slot's committed `(rank, range)` units, ascending by rank — the
    /// read-only view the extent logic needs without mutating anything.
    fn resident_ranks(&self, offset: u64) -> Vec<(u32, ByteRange)>;

    /// LRU candidates for STAGING SPILL: `(offset, last_use)` per Ram-tier
    /// slot holding RAM-arena staging (VRAM residents excluded — spilling
    /// under residency would strand placed ranges).
    fn harvest_staging_candidates(&self) -> Vec<(u64, u64)>;

    /// Exact audit records for every live sidecar entry under `key`.
    fn audit_records(&self, key: BufferKey) -> Vec<TierAuditRecord>;
}

/// The type-erased seam for INTERNED pools' static records — read-mostly:
/// statics never transition, they only get stamped, peeked, and audited.
pub(crate) trait TieredInternedHooks: Send + Sync {
    /// The static snapshot for `id`, if that id is currently interned
    /// (referenced). Entries exist ONLY while referenced, so presence IS
    /// liveness for the static class.
    fn static_snapshot(&self, id: HandleId) -> Option<StaticSnapshot>;

    /// Re-stamps `id`'s last-use (demand observed). No-op if absent.
    fn stamp_static(&self, id: HandleId);

    /// Exact audit records for every interned entry under `key`.
    fn audit_records(&self, key: BufferKey) -> Vec<TierAuditRecord>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ladder_orders_disk_ram_vram() {
        assert!(Tier::Disk < Tier::Ram && Tier::Ram < Tier::Vram);
        assert_eq!(Tier::Vram.lower(), Some(Tier::Ram));
        assert_eq!(Tier::Ram.lower(), Some(Tier::Disk));
        assert_eq!(Tier::Disk.lower(), None);
        assert_eq!(Tier::from_u8(Tier::Vram.as_u8()), Some(Tier::Vram));
        assert_eq!(Tier::from_u8(9), None);
    }

    #[test]
    fn arenas_put_take_recycle_slots_and_account_bytes() {
        let mut arena = ByteArena::new();
        let a = arena.put(vec![1u8; 100].into_boxed_slice());
        let b = arena.put(vec![2u8; 50].into_boxed_slice());
        assert_eq!(arena.used, 150);
        let got = arena.take(a).unwrap();
        assert_eq!(arena.used, 50);
        let c = arena.put(vec![3u8; 100].into_boxed_slice());
        assert_eq!(c, a, "recycled slot index");
        assert_eq!(got.len(), 100);
        assert!(arena.take(b).is_some());
        assert_eq!(arena.used, 100, "only c's bytes remain");
        assert!(arena.take(c).is_some());
        assert_eq!(arena.used, 0);
    }

    #[test]
    fn fetch_completion_is_exactly_once() {
        let engine = TierEngine::new();
        let sel = TierSelector::Row { column: ComponentId(7), row: 3 };
        let f = engine.open_fetch(sel);
        assert_eq!(engine.claim_fetch(&f), Err(TierError::NotYetReady), "queued ≠ ready");
        engine.mark_fetches_ready(&sel);
        // A DIFFERENT selector's fetch must not be marked by this one's.
        let other_sel = TierSelector::Row { column: ComponentId(7), row: 4 };
        let f_other = engine.open_fetch(other_sel);
        engine.mark_fetches_ready(&other_sel);
        assert!(engine.claim_fetch(&f).is_ok(), "matching selector becomes completable");
        assert_eq!(engine.claim_fetch(&f), Err(TierError::AlreadyCompleted), "double completion impossible");
        assert!(
            engine.claim_fetch(&f_other).is_ok(),
            "second fetch unaffected by the first's consumption"
        );
        let f2 = engine.open_fetch(sel);
        assert_eq!(engine.claim_fetch(&f2), Err(TierError::NotYetReady));
    }

    #[test]
    fn transitions_shard_deterministically_and_drain_in_shard_order() {
        let engine = TierEngine::new();
        let mk = |row| PendingTransition {
            target: TierSelector::Row { column: ComponentId(1), row },
            kind: TransitionKind::Demote(DemoteExtent::All),
            seq: 0,
        };
        engine.queue_transition(mk(1));
        engine.queue_transition(mk(2));
        engine.queue_transition(mk(3));
        let mut rows: Vec<u32> = Vec::new();
        engine.for_each_shard_batch(|batch| {
            for t in batch {
                match t.target {
                    TierSelector::Row { row, .. } => rows.push(row),
                    _ => unreachable!(),
                }
            }
        });
        rows.sort_unstable();
        assert_eq!(rows, vec![1, 2, 3]);
        let mut seen_again = false;
        engine.for_each_shard_batch(|_| seen_again = true);
        assert!(!seen_again, "batch walk empties every shard");
    }

    #[test]
    fn vram_reservation_is_cas_exact_and_releases_cleanly() {
        let engine = TierEngine::new();
        engine.configure(TierConfig { vram_budget_bytes: 100, ram_budget_bytes: 100 }, 4);
        engine.try_reserve_vram(60).unwrap();
        assert!(matches!(
            engine.try_reserve_vram(41),
            Err(TierError::BudgetExhausted { needed: 41, budget: 100 })
        ));
        engine.try_reserve_vram(40).unwrap(); // exactly fills the budget
        engine.release_vram(100);
        assert_eq!(engine.vram_used(), 0);
    }

    #[test]
    fn spill_moves_boxes_between_arenas_without_growing_total_memory_twice() {
        let engine = TierEngine::new();
        let slot = engine.stage_ram_for_test(vec![9u8; 32]);
        assert_eq!(engine.ram_used(), 32);
        let (disk_slot, len) = engine.spill_ram_to_disk(slot);
        assert_eq!(len, 32);
        assert_eq!(engine.ram_used(), 0, "spill MOVES ownership, RAM account drops");
        let (back, len2) = engine.unspill_disk_to_ram(disk_slot);
        assert_eq!(len2, 32);
        let mut out = [0u8; 32];
        assert!(engine.copy_ram_window(back, 0, &mut out));
        assert!(out.iter().all(|&b| b == 9));
    }
}
