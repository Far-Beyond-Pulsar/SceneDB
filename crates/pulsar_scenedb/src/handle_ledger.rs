//! Counted-handle plumbing: the CONCEPT of an asset identity whose lifetime
//! is tracked by reference counting, with zero domain knowledge. This crate
//! never learns what a texture, mesh, or material is — it only knows that
//! some `u128`-shaped field values ("handles") name shared resources, and
//! that every insert/overwrite/removal/despawn of a component carrying one
//! is a reference-count transition worth recording. The split is precise:
//!
//! - **Codegen** (`pulsar_scenedb_derive`) detects fields whose Rust type is
//!   syntactically [`HandleId`] (the exact same approach `Vec<T>` detection
//!   already uses for the variable-length `#[gpu]` path — a last-path-
//!   segment name match, because macro expansion has no real type
//!   information; see `scene_store.rs`'s `as_vec_elem_type` doc for the
//!   established reasoning) and generates ONE pure collector function per
//!   struct that copies the current values of those fields out of a value
//!   pointer, plus an inventory-submitted [`HandleLedgerRegistration`] under
//!   the struct's own `ComponentId` — the identical link-time-dispatch shape
//!   `GpuMirrorRegistration` proved (see `gpu::world_mirror`'s "Dispatch
//!   mechanism" section for why compile-time specialization cannot work
//!   inside `World::insert_inner`'s generic body; the same argument applies
//!   verbatim here).
//! - **[`World`](crate::World)** fires the events, at exactly four sites:
//!   an insert that OVERWRITES an existing component reports a swap (release
//!   old unless zero, acquire new unless zero — equal ⇒ nothing), a first
//!   insert acquires, a component removal releases what it took away, and a
//!   despawn collects every dying row's handles into one slice and reports
//!   them as one batch. A `get_mut` write through `DerefMut` reports a swap
//!   against values captured at `get_mut` time, closing the same gap `Mut`'s
//!   GPU hook originally closed for `#[gpu]` fields (see `world.rs`'s `Mut`
//!   doc — without this, mutating a handle field through `get_mut` would
//!   silently bypass accounting, which is a leak or a premature free
//!   depending on direction, i.e. the worst kind of bug: both).
//! - **This module** maintains the counts. Multiset semantics live here:
//!   two components referencing the same content produce two acquires for
//!   one id, and the count reflects 2 — nothing upstream aggregates,
//!   deduplicates, or otherwise interprets before reaching this layer.
//!
//! # Internalized, not attached (SceneDB content-dedup mission)
//!
//! An earlier revision of this module exposed counting through an
//! externally-implemented `HandleLedger` trait, attached to a `World` via
//! `attach_handle_ledger` — the same shape `attach_gpu_mirror`/
//! `attach_change_tracker` use for genuinely optional, swappable
//! capabilities. That shape was the wrong one for handle counting
//! specifically: every consumer of this crate that cares about content
//! identity at all wants counting to simply be correct, always, with zero
//! setup — not a seam some downstream crate implements and might forget to
//! attach (the WIP's own doc even had to name this as a hazard: entities
//! hydrated before attachment would leak accounting silently). So counting
//! moved INTO `World` as always-present state (`world.rs`'s `handle_counts`
//! field, no `Option`, no attach/detach API) and the `HandleLedger` trait
//! was deleted outright rather than kept as an optional observation seam —
//! nothing in this crate or any downstream consumer needs to observe raw
//! acquire/release events from outside `World`; anything that needs to know
//! "is this id still referenced" asks [`crate::World::handle_ref_count`] or
//! [`crate::World::handle_audit`] instead of subscribing to a stream of them.
//! If a real external-observer need ever materializes, add it back as a
//! genuinely optional `Vec` of weak callbacks costing nothing when empty —
//! don't resurrect the trait speculatively.
//!
//! # Zero-value convention
//!
//! `HandleId::ZERO` (all bits clear) means "no asset". Generated collectors
//! copy zero values faithfully (they are pure field snapshots), but every
//! event helper in this module filters zeros AT THE BOUNDARY -- the count
//! table never observes [`HandleId::ZERO`] from any `World` event, ever. A
//! default-initialized component therefore costs zero counting traffic,
//! which is what makes adding a handle field to an existing component
//! invisible until hydrate actually fills it in.
//!
//! # Cost model
//!
//! - **`T` has no `HandleId`-typed fields**: `insert`/`remove`/`despawn`/
//!   `get_mut` each pay one `HashMap` probe (`collect_fn_for`) that misses —
//!   no allocation, no counting work. This is the same "absent means
//!   absent" shape every optional `World` capability already established
//!   (change tracker, subscriptions, GPU mirror), just without the extra
//!   `Option` check an attach-based capability would also pay, since
//!   counting is unconditional now.
//! - **`T` has handle fields**: each event is O(fields-of-the-type-being-
//!   touched) with NO allocation on the steady-state path (the scratch
//!   buffer the collectors fill is a reused thread-local — see
//!   `with_scratch`), plus one `HashMap` entry op per distinct id touched.
//!
//! # What is NOT covered (deliberately)
//!
//! Replication's `force_spawn_in_archetype` path pushes placeholder rows and
//! lets `Delta::apply` drive subsequent writes through `insert_inner` —
//! those inserts ARE covered, but a peer-authored value landing via
//! `push_default` itself is not reported (it is a `Default::default()`
//! placeholder, whose handle fields are zero by the convention above, and
//! zero produces no events anyway — so in practice nothing escapes; this
//! sentence exists so the next reader doesn't have to re-derive it).

use crate::component::ComponentId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A plain, fixed-size asset-identity handle — what lands in a component
/// field to say "this slot references content X". `repr(C)` `Pod` newtype
/// over `u128` (same shape and same SAFETY argument as `VarLenHandle`: no
/// padding, every bit pattern valid), so a struct containing one stays a
/// plain `Pod` row on the classic `SceneStore` path AND composes freely
/// with the variable-length path (a struct can carry both a `Vec<T>`
/// `#[gpu]` field and a `HandleId` field — Pulsar-Native's
/// `StaticMeshComponent` is exactly that shape).
///
/// The bits themselves are opaque HERE. Producers (content hashing) and
/// consumers (whatever gives an id meaning) agree on what they mean; this
/// crate only ever compares them for equality and tests them against
/// [`Self::ZERO`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct HandleId(pub u128);

impl HandleId {
    /// The "no asset" sentinel — see the module-level "Zero-value
    /// convention" section for why it exists and who skips it.
    pub const ZERO: HandleId = HandleId(0);

    /// Whether this is the "no asset" sentinel.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

// SAFETY: #[repr(C)], one plain u128, no padding, every bit pattern valid —
// byte-for-byte the same safety argument `VarLenHandle`'s impl makes.
unsafe impl crate::page::Pod for HandleId {}

/// A field type that can supply a stable content identity for a SIBLING
/// `Vec<T>`-typed `#[gpu]` field it's paired with via
/// `#[gpu(mirror = Once, content_id = "that_field")]` — see
/// `gpu::interned_pool`'s module doc for the full mechanism this exists to
/// drive (content-addressed dedup of GPU-resident variable-length payloads,
/// e.g. mesh geometry).
///
/// Semantic identity is owned by downstream implementors -- asset wrapper
/// types like Pulsar-Native's `MeshAssetPath` own the hashing and any file
/// I/O it needs; this trait is just the narrow seam the derive macro calls
/// through. (The ONE mechanical id helper in this crate,
/// `gpu::placement::structural_content_id`, deliberately does NOT flow
/// through this trait: it byte-folds a row's own reference list for
/// Heavy-placement fields with no semantic source to name, which is
/// structural equality, not asset identity.)
/// identity does this value carry" without needing to know the concrete
/// type beyond "it implements `ContentAddressed`" (enforced as a normal
/// trait bound on the generated call, the same way `GpuUploadSource`
/// already is for the fixed-size heavy/handle split).
///
/// Two values that are content-IDENTICAL must return the SAME id (so
/// duplicate assets actually dedup); two values that differ must return
/// DIFFERENT ids (or the intern layer's collision policy kicks in — see
/// `gpu::interned_pool`'s doc for what happens then). Returning
/// [`HandleId::ZERO`] opts a value out of interning entirely for that
/// write: the field falls back to behaving like an ordinary (non-shared)
/// `Vec<T>` `#[gpu]` field for that one row, matching the zero-value
/// convention every other handle-shaped field in this crate already
/// follows.
pub trait ContentAddressed {
    fn content_id(&self) -> HandleId;
}

/// Signature of the per-type, macro-generated collector: copies every
/// `HandleId`-typed field's CURRENT value out of `value` (pointing at a
/// live, correctly-aligned instance of the concrete struct the registration
/// was submitted for — the sole caller guarantees this, exactly like
/// `GpuMirrorRegistration::dispatch`'s `data` contract) onto `out`, in field
/// declaration order. Pure: no locks, no user callbacks, no allocation
/// beyond appending to the caller's buffer. Declaration order matters —
/// `report_value_swap` pairs two collections positionally, so both sides of
/// a swap are the SAME field ordering by construction.
pub type CollectHandlesFn = fn(value: *const (), out: &mut Vec<HandleId>);

/// One `#[derive(SceneStore)]` struct's entry in the handle-counting
/// dispatch table, submitted via `inventory` (same mechanism, same
/// reasoning as `GpuMirrorRegistration` — see `gpu::world_mirror`'s module
/// doc for why a link-time registry beats autoref specialization inside
/// generic `insert_inner` bodies; that argument transfers verbatim). Structs
/// with no `HandleId`-typed fields submit NOTHING and are indistinguishable
/// from types the derive never saw: one `HashMap` miss, once, every event.
pub struct HandleLedgerRegistration {
    /// Resolved lazily, not at `submit!` const-eval time — same reason
    /// `GpuMirrorRegistration::component_id` is a `fn` pointer.
    pub component_id: fn() -> ComponentId,
    /// See [`CollectHandlesFn`].
    pub collect_from_value: CollectHandlesFn,
}

pulsar_reflection::inventory::collect!(HandleLedgerRegistration);

fn registry_map() -> &'static HashMap<ComponentId, CollectHandlesFn> {
    static MAP: OnceLock<HashMap<ComponentId, CollectHandlesFn>> = OnceLock::new();
    MAP.get_or_init(|| {
        pulsar_reflection::inventory::iter::<HandleLedgerRegistration>()
            .map(|r| ((r.component_id)(), r.collect_from_value))
            .collect()
    })
}

/// Looks up the collector for `cid`'s component type, if the derive
/// generated one (i.e. the struct has ≥1 `HandleId`-typed field, or, from
/// the content-dedup feature onward, ≥1 `content_id = "..."`-linked `Vec<T>`
/// field — see `gpu::interned_pool`'s doc). Callers have `cid` in hand
/// already (archetype indexing computed it), so this is one `HashMap`
/// probe, not a second `TypeId` resolution — mirrors
/// `gpu::world_mirror::dispatch_for`'s cost shape exactly.
#[inline]
pub(crate) fn collect_fn_for(id: ComponentId) -> Option<CollectHandlesFn> {
    registry_map().get(&id).copied()
}

/// Per-thread reusable scratch buffer for collector output. The hot path
/// (`World::insert`/`remove`/`despawn`) clears, fills, consumes, and
/// recycles one `Vec` instead of allocating per event; the `try_borrow_mut`
/// failure arm is the documented reentrancy escape: if a nested `World`
/// operation somehow re-enters while the buffer is still lent out, we fall
/// back to a fresh allocation for the nested operation rather than
/// panicking or corrupting the outer run — correctness first, and the
/// nested case is rare enough that its allocation is irrelevant.
pub(crate) fn with_scratch<R>(f: impl FnOnce(&mut Vec<HandleId>) -> R) -> R {
    thread_local! {
        static SCRATCH: RefCell<Vec<HandleId>> = const { RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|cell| {
        match cell.try_borrow_mut() {
            Ok(mut buf) => {
                buf.clear();
                let r = f(&mut buf);
                buf.clear(); // don't hold dead ids across unrelated events
                r
            }
            Err(_) => {
                // Reentrant call while the thread-local is already lent out.
                // Fresh buffer, no shared state touched — see doc above.
                let mut buf = Vec::new();
                f(&mut buf)
            }
        }
    })
}

/// The always-present per-`World` multiset of live handle references, keyed
/// by content identity. Owned directly by `World` (`world.rs`'s
/// `handle_counts` field) — no lock, because every `World` mutator already
/// requires `&mut World`, so there is no concurrent access to this table to
/// guard against (contrast `gpu::interned_pool::InternedVarLenPool`, which
/// genuinely is reachable from multiple threads via `Arc<SceneGpuStore>` and
/// is `RwLock`-guarded for exactly that reason).
///
/// A count reaching zero REMOVES the entry rather than leaving a `0` behind
/// — otherwise a long-running `World` that churns through many distinct
/// transient ids would grow this table forever. This also makes
/// [`HandleCounts::get`] and "is this id referenced at all" the same
/// question (`contains_key`), for free.
#[derive(Default)]
pub(crate) struct HandleCounts {
    counts: HashMap<HandleId, i64>,
    /// Ids that have already triggered the release-with-no-prior-acquire
    /// diagnostic below — warned ONCE per id, not once per occurrence, so a
    /// genuinely buggy caller hammering the same id doesn't flood output.
    /// Small and self-cleaning is not worth the complexity here: the
    /// scenario this guards is a caller bug (or `Mut::into_inner`'s
    /// documented escape hatch, exercised deliberately in tests), not a
    /// steady-state path, so an entry lingering here for the `World`'s
    /// lifetime is immaterial.
    warned_underflow: std::collections::HashSet<HandleId>,
}

impl HandleCounts {
    /// One more live reference to `id`. `id.is_zero()` must already be
    /// filtered by the caller (every reporting fn below does this at the
    /// boundary — see the module doc's "Zero-value convention").
    fn acquire(&mut self, id: HandleId) {
        *self.counts.entry(id).or_insert(0) += 1;
    }

    /// One fewer live reference to `id`. Releasing an id that was never
    /// acquired, or releasing past zero, is tolerated (not always a caller
    /// bug — `Mut::into_inner`'s documented escape hatch can legitimately
    /// produce this, see `tests/handle_ledger.rs`'s
    /// `get_mut_write_reports_swap_and_borrow_only_get_mut_is_silent`) —
    /// saturates at zero (no entry is ever left negative; an id with no
    /// live references simply has no entry) and logs a once-per-id
    /// diagnostic so a REAL accounting bug is still visible somewhere.
    /// Never panics, in any build profile — see the adversarial test
    /// matrix's explicit "no panic" requirement for this exact scenario.
    fn release(&mut self, id: HandleId) {
        use std::collections::hash_map::Entry;
        match self.counts.entry(id) {
            Entry::Occupied(mut e) => {
                let count = e.get_mut();
                *count -= 1;
                if *count <= 0 {
                    e.remove();
                }
            }
            Entry::Vacant(_) => {
                if self.warned_underflow.insert(id) {
                    tracing::warn!(
                        "handle {id:?} released with no prior acquire (or already at zero) -- \
                         tolerated, saturating at zero; this diagnostic fires once per id"
                    );
                }
            }
        }
    }

    /// Current live reference count for `id` — `0` for an id never acquired
    /// or already fully released. O(1).
    pub(crate) fn get(&self, id: HandleId) -> i64 {
        self.counts.get(&id).copied().unwrap_or(0)
    }
}

/// Runs `collect` over `old` and `new` (both pointing at live instances of
/// the same concrete struct, in declaration-order collection), then reports
/// the differences: fields whose old and new ids are EQUAL produce nothing
/// (re-writing a component with unchanged assets must not churn counts),
/// unequal fields release the old id unless zero and acquire the new id
/// unless zero. This is THE swap rule from the task contract, kept in ONE
/// place so the three swap-shaped reporting sites (in-place insert,
/// rehydrate-replace — which IS an in-place insert — and `get_mut` mutation)
/// cannot drift apart.
pub(crate) fn report_value_swap(
    counts: &mut HandleCounts,
    collect: CollectHandlesFn,
    old: *const (),
    new: *const (),
) {
    // ONE buffer, two passes: the old ids land at [0..n), the new ids at
    // [n..2n). Deliberately NOT two nested `with_scratch` calls — the inner
    // one would see its own thread-local already borrowed and fall into the
    // reentrant-fresh-allocation arm EVERY time, silently converting the
    // steady-state no-alloc guarantee back into a per-swap heap hit.
    with_scratch(|buf| {
        (collect)(old, buf);
        let n = buf.len();
        if n == 0 {
            return;
        }
        (collect)(new, buf);
        debug_assert_eq!(
            buf.len(),
            2 * n,
            "handle-field count changed between two reads of the same type — collector bug"
        );
        for i in 0..n.min(buf.len() - n) {
            let o = buf[i];
            let v = buf[n + i];
            if o == v {
                continue;
            }
            if !o.is_zero() {
                counts.release(o);
            }
            if !v.is_zero() {
                counts.acquire(v);
            }
        }
    });
}

/// First-insert path: acquire every nonzero handle `collect` finds. No
/// comparison, no release — there is nothing to release (the entity did not
/// have this component before; migration of OTHER components moves values
/// verbatim and touches no counts, which is exactly why handles are keyed by
/// CONTENT and not by row position).
pub(crate) fn report_value_acquire(counts: &mut HandleCounts, collect: CollectHandlesFn, value: *const ()) {
    with_scratch(|scratch| {
        (collect)(value, scratch);
        for &id in scratch.iter() {
            if !id.is_zero() {
                counts.acquire(id);
            }
        }
    });
}

/// Removal path: release every nonzero handle `collect` finds — the exact
/// inverse of [`report_value_acquire`], called with the value being taken
/// OUT of the world.
pub(crate) fn report_value_release(counts: &mut HandleCounts, collect: CollectHandlesFn, value: *const ()) {
    with_scratch(|scratch| {
        (collect)(value, scratch);
        for &id in scratch.iter() {
            if !id.is_zero() {
                counts.release(id);
            }
        }
    });
}

/// `get_mut`-write path: same swap rule as [`report_value_swap`], but the
/// OLD side arrives already collected (`captured`, snapshotted at
/// `get_mut` time by `World::get_mut` -- after that point there is no way
/// back at the old value, since the caller is about to mutate through the
/// guard), and only the NEW side is collected live. Same single-buffer
/// discipline as [`report_value_swap`]: one scratch fill, positional zip
/// against `captured`.
pub(crate) fn report_captured_swap(
    counts: &mut HandleCounts,
    collect: CollectHandlesFn,
    captured_old: &[HandleId],
    new: *const (),
) {
    with_scratch(|buf| {
        (collect)(new, buf);
        debug_assert_eq!(
            buf.len(),
            captured_old.len(),
            "handle-field count changed between get_mut capture and drop -- collector bug"
        );
        for (i, &v) in buf.iter().enumerate() {
            let o = captured_old[i];
            if o == v {
                continue;
            }
            if !o.is_zero() {
                counts.release(o);
            }
            if !v.is_zero() {
                counts.acquire(v);
            }
        }
    });
}

/// Despawn fast path: every handle collected from a dying row's components,
/// released in one batch. Duplicates in `row` are meaningful (two slots
/// referenced the same id) — released individually, multiset semantics.
/// `row` empty is legal and a no-op.
pub(crate) fn release_row(counts: &mut HandleCounts, row: &[HandleId]) {
    for &id in row {
        counts.release(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_the_default_and_is_zero() {
        assert_eq!(HandleId::default(), HandleId::ZERO);
        assert!(HandleId::ZERO.is_zero());
        assert!(!HandleId(1).is_zero());
    }

    #[test]
    fn handle_id_is_a_plain_pod_u128_shape() {
        assert_eq!(std::mem::size_of::<HandleId>(), 16);
        assert_eq!(
            std::mem::align_of::<HandleId>(),
            std::mem::align_of::<u128>()
        );
        // Every bit pattern is a valid value (Pod contract) — round-trip the
        // extremes through bytes to prove there's no niche being violated.
        let extremes = [u128::MAX, u128::MIN, 0xDEAD_BEEF_CAFE_F00D];
        for v in extremes {
            let h = HandleId(v);
            let bytes =
                unsafe { std::slice::from_raw_parts(&h as *const HandleId as *const u8, 16) };
            let back = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const HandleId) };
            assert_eq!(back, h);
        }
    }

    #[test]
    fn acquire_then_release_returns_to_zero_and_drops_the_entry() {
        let mut counts = HandleCounts::default();
        counts.acquire(HandleId(7));
        counts.acquire(HandleId(7));
        assert_eq!(counts.get(HandleId(7)), 2);
        counts.release(HandleId(7));
        assert_eq!(counts.get(HandleId(7)), 1);
        counts.release(HandleId(7));
        assert_eq!(counts.get(HandleId(7)), 0);
        assert!(!counts.counts.contains_key(&HandleId(7)), "zero count must not linger as a map entry");
    }

    #[test]
    fn release_row_reports_duplicates_individually() {
        // Two slots in one row referencing the same id must release TWO
        // references — multiset semantics are this module's job.
        let mut counts = HandleCounts::default();
        counts.acquire(HandleId(7));
        counts.acquire(HandleId(7));
        counts.acquire(HandleId(9));
        release_row(&mut counts, &[HandleId(7), HandleId(7), HandleId(9)]);
        assert_eq!(counts.get(HandleId(7)), 0);
        assert_eq!(counts.get(HandleId(9)), 0);
    }

    #[test]
    fn empty_release_row_is_a_no_op() {
        let mut counts = HandleCounts::default();
        release_row(&mut counts, &[]);
        assert!(counts.counts.is_empty());
    }

    /// `World::handle_counts` is `Arc<Mutex<HandleCounts>>` purely so
    /// [`super::super::world::Mut`]'s drop-time hook can hold an owned
    /// handle -- see that field's doc. Nothing in the internalized design
    /// actually needs `HandleCounts` to be contended from multiple threads
    /// (every `World` mutator is `&mut self`), but nothing prevents someone
    /// from wrapping one in a shared `Arc` directly either, so this proves
    /// the plain data type itself balances correctly under real concurrent
    /// access if it's ever used that way -- the same exactly-once-accounting
    /// property the deleted `HandleLedger` trait's own concurrency test used
    /// to prove at the trait boundary, now proved at the data-structure one.
    #[test]
    fn handle_counts_balances_exactly_under_concurrent_access() {
        use std::sync::{Arc, Barrier, Mutex};
        let counts = Arc::new(Mutex::new(HandleCounts::default()));
        let barrier = Arc::new(Barrier::new(8));
        let mut joins = Vec::new();
        for thread in 0..8u64 {
            let counts = Arc::clone(&counts);
            let barrier = Arc::clone(&barrier);
            joins.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..2_000u64 {
                    // Threads 0-3 share ids 0..16 (hot contention); threads
                    // 4-7 own disjoint ranges (zero contention).
                    let id = HandleId(if thread < 4 {
                        (i % 16) as u128
                    } else {
                        1_000 + (thread as u128) * 100_000 + (i % 64) as u128
                    });
                    let mut c = counts.lock().unwrap();
                    c.acquire(id);
                    c.release(id);
                    c.acquire(id);
                }
            }));
        }
        for j in joins {
            j.join().expect("worker thread panicked");
        }
        let final_counts = counts.lock().unwrap();
        // Every hot id: acquire-release-acquire per touching iteration nets
        // to +1 per touch, summed across all 4 threads that share it.
        for i in 0..16u128 {
            let expected = (0..2_000u64).filter(|x| x % 16 == i as u64).count() as i64 * 4;
            assert_eq!(final_counts.get(HandleId(i)), expected, "hot id {i} diverged under contention");
        }
    }
}