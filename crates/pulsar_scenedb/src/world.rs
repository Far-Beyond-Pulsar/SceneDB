use crate::archetype::{Archetype, ArchetypeId, ArchetypeKey};
use crate::component::{Column, Component, ComponentId, ErasedColumn};
use crate::entity::{Entity, EntitySlot};
use crate::replication::ChangeTracker;
use ahash::AHashMap;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

/// Fixed-capacity, over-aligned scratch buffer for a single erased-component
/// move (swap-remove -> push) during archetype migration
/// ([`World::move_column_row`]), without a heap allocation.
///
/// `CAP = 128` bytes at `ALIGN = 16` covers the overwhelming majority of
/// real ECS components -- a 4x4 `f32` transform matrix is 64 bytes at
/// 4-byte alignment; ordinary game components (positions, velocities,
/// health, tags, small structs) are typically well under this. SIMD-shaped
/// types (`[f32; 4]`, etc.) never exceed 16-byte natural alignment on any
/// target this crate supports. A component whose size or alignment
/// requirement genuinely exceeds these bounds is rare, and
/// `move_column_row` falls back to the original heap-allocating path for
/// it -- correctness holds either way; only the common case skips the
/// allocator.
#[repr(align(16))]
struct MoveScratch {
    bytes: [MaybeUninit<u8>; Self::CAP],
}

impl MoveScratch {
    const CAP: usize = 128;
    const ALIGN: usize = 16;

    #[inline]
    fn new() -> Self {
        // SAFETY-adjacent note: `MaybeUninit<u8>` needs no initialization
        // (it is explicitly "possibly uninitialized"), so leaving every
        // byte uninitialized here is not itself unsafe -- only reading the
        // bytes before they've been written (which `move_column_row` never
        // does; it always writes via `swap_remove_into` before reading via
        // `push_from`) would be.
        Self { bytes: [MaybeUninit::uninit(); Self::CAP] }
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr() as *mut u8
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr() as *const u8
    }
}

/// The central ECS store: owns all entities, their component data, and the
/// archetype graph.
///
/// # Entity lifecycle
///
/// 1. [`World::spawn`] allocates a slot and places the entity in the empty
///    archetype (no components).
/// 2. [`World::insert`] adds a component, migrating the entity to a new
///    archetype.
/// 3. [`World::remove`] strips a component, migrating back.
/// 4. [`World::despawn`] frees the slot and swap-removes the entity from its
///    archetype.
///
/// # Queries
///
/// Use [`World::query`] to iterate entities matching a component pattern.
/// Queries scan all archetypes, using the `u64` bitmask to skip non-matching
/// archetypes in constant time.
pub struct World {
    pub entity_slots: Vec<EntitySlot>,
    pub free_slots: Vec<u32>,
    pub archetypes: Vec<Archetype>,
    pub archetype_index: AHashMap<ArchetypeKey, ArchetypeId>,
    /// GPU-mirror wiring for `#[gpu]`-tagged component fields (see
    /// `crate::gpu::world_mirror`). `None` (the default) means `insert`
    /// behaves exactly as it does without the `gpu` feature at all — this
    /// field, and the one check against it in `insert_inner`, are the
    /// entire surface area this crate's GPU layer adds to `World` itself.
    /// Compiled out completely without the `gpu` feature (CONTRACTS C0:
    /// `--no-default-features` never depends on `wgpu`).
    #[cfg(feature = "gpu")]
    gpu_mirror: Option<crate::gpu::GpuMirrorHandle>,
    /// Change-tracking wiring, attached the same way as `gpu_mirror` above
    /// (see [`Self::attach_change_tracker`]). `None` (the default) means
    /// `spawn`/`insert`/`remove`/`despawn`/`get_mut` behave exactly as they
    /// always have -- no tracking, no cost beyond the `Option::is_none()`
    /// check. Once attached, every one of those calls records into it
    /// automatically -- no `_tracked`-suffixed call needed anywhere. Not
    /// `gpu`-feature-gated: replication is always available (CONTRACTS C0).
    change_tracker: Option<crate::replication::SharedChangeTracker>,
}

/// A mutable borrow of component `T` on some entity, returned by
/// [`World::get_mut`]. `Deref`/`DerefMut` to `T`, so every existing call
/// site (`*guard = value`, `guard.field += 1`, method calls) keeps working
/// unchanged — the only observable difference from the old `&mut T` is that
/// dropping this guard, not the mutation itself, is when a `#[gpu]`-bearing
/// component's fields reach the GPU mirror (see the struct's field doc).
///
/// This exists to close a real gap: before this type, `World::insert` had a
/// GPU dispatch hook but `get_mut` had none at all — mutating a `#[gpu]`
/// field through `get_mut` silently never reached the GPU, for EITHER
/// `MirrorMode`, not just `Once`. `Mut` gives `get_mut` the same hook
/// `insert_inner` already has, reusing the identical link-time dispatch
/// registry (`crate::gpu::world_mirror::dispatch_for`) — no new registration
/// mechanism.
pub struct Mut<'a, T> {
    value: &'a mut T,
    /// Precomputed at [`World::get_mut`] time (not resolved again in
    /// [`Drop::drop`]): `None` whenever the `gpu` feature is off, no mirror
    /// is attached, or `T` has no `#[gpu]` fields — the exact same
    /// short-circuit `insert_inner` already applies, so a `get_mut` on a
    /// plain (non-GPU) component costs one `Option`/`HashMap`-miss check on
    /// construction and nothing at all on drop.
    #[cfg(feature = "gpu")]
    gpu_hook: Option<GpuMutHook>,
    /// Precomputed at [`World::get_mut`] time, same shape as `gpu_hook`
    /// above: `None` unless a [`crate::replication::SharedChangeTracker`]
    /// is attached ([`World::attach_change_tracker`]). Lets `get_mut`
    /// mutations record automatically on drop, the same way `insert`
    /// already does when a tracker is attached — no `_tracked` call, no
    /// separate `get_mut_tracked` method to remember.
    change_hook: Option<ChangeMutHook>,
}

#[cfg(feature = "gpu")]
struct GpuMutHook {
    mirror: crate::gpu::GpuMirrorHandle,
    row: u32,
    dispatch: crate::gpu::world_mirror::DispatchFn,
}

/// See [`Mut::change_hook`]'s doc. Records into the SAME `SharedChangeTracker`
/// `insert`/`spawn`/`remove`/`despawn` already record into when one is
/// attached to the `World` this entity belongs to — captured at `get_mut`
/// time (not re-resolved in `Drop`) since `Mut` doesn't keep a `&World`
/// borrow alive across its own lifetime.
struct ChangeMutHook {
    tracker: crate::replication::SharedChangeTracker,
    entity: Entity,
    component_id: ComponentId,
}

impl<'a, T> Deref for Mut<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.value
    }
}

impl<'a, T> DerefMut for Mut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<'a, T> Drop for Mut<'a, T> {
    fn drop(&mut self) {
        #[cfg(feature = "gpu")]
        if let Some(hook) = &self.gpu_hook {
            // `is_new_insert = true`: from `write_gpu_columns_at_row`'s
            // perspective this bool means "write `Once` fields too, don't
            // skip them" — exactly right here. `Once`'s "never re-write
            // after the first insert" pinning is specifically an
            // INSERT-path behavior (a routine re-insert of the same
            // component shouldn't silently re-upload static data); an
            // explicit `get_mut` mutation is, by construction, the caller
            // deliberately changing the value, so `Once` fields re-upload
            // here exactly like `DirtyTracked` ones do (see the module doc
            // on `MirrorMode::Once` / `GpuUploadSource` for the full
            // contract this is the write-side half of).
            (hook.dispatch)(&hook.mirror, hook.row, self.value as *const T as *const (), true);
        }
        if let Some(hook) = &self.change_hook {
            // Same "0, empty bytes" shape `insert_inner`'s tracked path
            // already uses (see its own call to `record_component_change`
            // below) -- the actual field-level bytes are reconstructed by
            // the replication schema encoder from the live component at
            // encode time, not captured here.
            hook.tracker
                .record_component_change(hook.entity, hook.component_id, 0, Vec::new());
        }
    }
}

impl World {
    /// Create an empty world with one empty archetype and no entities.
    pub fn new() -> Self {
        let empty = Archetype::new_empty(ArchetypeId::EMPTY);
        let mut archetype_index = AHashMap::default();
        archetype_index.insert(ArchetypeKey(vec![]), ArchetypeId::EMPTY);
        Self {
            entity_slots: Vec::new(),
            free_slots: Vec::new(),
            archetypes: vec![empty],
            archetype_index,
            #[cfg(feature = "gpu")]
            gpu_mirror: None,
            change_tracker: None,
        }
    }

    /// Like [`Self::new`], already wired to `tracker` — the constructor
    /// counterpart of [`Self::attach_change_tracker`], for a caller that
    /// knows its tracker up front (mirrors [`crate::SceneDb::new_with_gpu_mirror`]'s
    /// shape for the GPU mirror).
    pub fn new_with_change_tracker(tracker: crate::replication::SharedChangeTracker) -> Self {
        let mut world = Self::new();
        world.attach_change_tracker(tracker);
        world
    }

    /// Attach a [`crate::replication::SharedChangeTracker`] so that every
    /// future `spawn`/`insert`/`remove`/`despawn`/`get_mut` call
    /// automatically records into it — no `_tracked`-suffixed call needed
    /// anywhere. Exactly the same shape as [`Self::attach_gpu_mirror`],
    /// applied to change tracking instead of GPU mirroring: before this,
    /// every mutating call site had to remember, on its own, whether *this*
    /// write needed tracking; now it's a property of whether a tracker is
    /// attached to the `World` at all.
    ///
    /// Existing `_tracked` methods (`insert_tracked`, etc, which take an
    /// explicit `&mut ChangeTracker` parameter) are unaffected by this —
    /// they keep recording into whatever tracker the caller passes them,
    /// independent of whatever is or isn't attached here.
    ///
    /// Idempotent: calling this again replaces the previous handle, it does
    /// not stack.
    pub fn attach_change_tracker(&mut self, tracker: crate::replication::SharedChangeTracker) {
        self.change_tracker = Some(tracker);
    }

    /// Detach the change tracker, if any — subsequent plain
    /// `spawn`/`insert`/`remove`/`despawn`/`get_mut` calls stop recording
    /// (existing `_tracked` call sites are unaffected either way).
    pub fn detach_change_tracker(&mut self) -> Option<crate::replication::SharedChangeTracker> {
        self.change_tracker.take()
    }

    /// Whether a change tracker is currently attached (see
    /// [`Self::attach_change_tracker`]).
    pub fn has_change_tracker(&self) -> bool {
        self.change_tracker.is_some()
    }

    /// The currently-attached change tracker, if any —
    /// [`crate::replication::SharedChangeTracker`] is cheap to `Clone`, so
    /// callers that already kept their own copy from before
    /// [`Self::attach_change_tracker`] don't need this — it exists for the
    /// case where they didn't (e.g. draining it once per frame from
    /// wherever the frame loop lives, without also having to thread the
    /// original handle all the way there separately).
    pub fn change_tracker(&self) -> Option<&crate::replication::SharedChangeTracker> {
        self.change_tracker.as_ref()
    }

    /// Attach a [`crate::gpu::GpuMirrorHandle`] so that every future
    /// `insert`/`insert_tracked` call automatically mirrors any `#[gpu]`
    /// fields of the inserted component to their registered GPU buffer, at
    /// row = `entity.index()` — no per-call opt-in needed once this is set.
    ///
    /// Call this once during setup, after registering every GPU-mirrored
    /// component type's buffers (`T::register_gpu_columns(&store, capacity,
    /// device)`, generated by `#[derive(SceneStore)]`). Components inserted
    /// before their buffer is registered are not an error — the write is
    /// silently skipped (see [`crate::gpu::SceneGpuStore::write_row_bytes`])
    /// — but they also won't retroactively appear on the GPU once the
    /// buffer *is* registered; insert (or re-insert) them again afterward.
    ///
    /// Idempotent: calling this again replaces the previous handle, it does
    /// not stack. Passing a handle backed by a different, smaller-capacity
    /// `SceneGpuStore` than a previous one is the caller's responsibility to
    /// avoid — this method does no capacity reconciliation of its own.
    #[cfg(feature = "gpu")]
    pub fn attach_gpu_mirror(&mut self, mirror: crate::gpu::GpuMirrorHandle) {
        self.gpu_mirror = Some(mirror);
    }

    /// Detach the GPU mirror, if any — subsequent inserts stop writing to
    /// the GPU (existing GPU-side data is left as-is, now unmaintained).
    #[cfg(feature = "gpu")]
    pub fn detach_gpu_mirror(&mut self) -> Option<crate::gpu::GpuMirrorHandle> {
        self.gpu_mirror.take()
    }

    /// Whether a GPU mirror is currently attached (see
    /// [`Self::attach_gpu_mirror`]).
    #[cfg(feature = "gpu")]
    pub fn has_gpu_mirror(&self) -> bool {
        self.gpu_mirror.is_some()
    }

    /// The currently-attached GPU mirror, if any — e.g. to reach
    /// [`crate::gpu::GpuMirrorHandle::generations`] for binding the
    /// liveness/generation buffer into a shader. `GpuMirrorHandle` is cheap
    /// to `Clone`, so callers that already kept their own copy from before
    /// [`Self::attach_gpu_mirror`] don't need this — it exists for the case
    /// where they didn't.
    #[cfg(feature = "gpu")]
    pub fn gpu_mirror(&self) -> Option<&crate::gpu::GpuMirrorHandle> {
        self.gpu_mirror.as_ref()
    }

    /// Uploads every row queued since the last call — both `#[gpu(mirror =
    /// DirtyTracked)]` World-mirrored fields (the default `#[gpu]` mode) and,
    /// as of SceneDB#39, `#[gpu(mirror = Once)]` fields and the liveness/
    /// generation mirror, all of which now defer their writes here too
    /// instead of writing immediately inline with `spawn`/`insert`/`despawn`
    /// — coalesced into as few GPU writes as row adjacency allows. Call once
    /// per frame — the World-mirror analogue of the cell-mirrored path's own
    /// boundary-phase sync. `None` if no mirror is attached;
    /// `Some(SyncStats::default()-shaped)` (zero ranges/bytes) if one is
    /// attached but nothing was pending.
    #[cfg(feature = "gpu")]
    pub fn flush_gpu_mirror(&self, queue: &wgpu::Queue) -> Option<crate::gpu::SyncStats> {
        self.gpu_mirror.as_ref().map(|m| {
            m.generations().flush(queue);
            m.store().flush_gpu_mirror(queue)
        })
    }

    /// Reserves capacity `n` on every registered World-mirrored GPU buffer
    /// right now, ahead of a known-size batch of upcoming inserts — moves
    /// what would otherwise be an unpredictable, mid-batch reallocation
    /// (a real GPU-to-GPU copy, potentially tens of milliseconds at
    /// AAA-relevant scale — see Helio#211's benchmark findings) off the
    /// per-insert critical path and onto this one, caller-controlled call.
    /// `None` if no mirror is attached (nothing to reserve on); `Some(Err(..))`
    /// if some registered column can't grow that far (e.g. the device's own
    /// `wgpu::Limits::max_buffer_size` — see
    /// `gpu::DynamicGpuBuffer::ensure_capacity`'s doc).
    ///
    /// Call this before a batch, not instead of sizing
    /// `register_gpu_columns_growable`'s `initial_capacity` sensibly — the
    /// two aren't redundant: `initial_capacity` is what a *fresh* `World`
    /// starts with, `reserve` is for growing an *already-running* `World`
    /// ahead of a specific future batch.
    #[cfg(feature = "gpu")]
    pub fn reserve_gpu_mirror_capacity(&self, queue: &wgpu::Queue, n: u32) -> Option<Result<(), crate::gpu::CapacityError>> {
        self.gpu_mirror.as_ref().map(|m| m.store().reserve_world_mirror_capacity(queue, n))
    }

    /// Shrinks every registered World-mirrored GPU buffer to the smallest
    /// capacity that still covers `highest_live_row`, with `slack_factor`
    /// headroom (e.g. `1.5` = 50% extra room before the next growth).
    /// `highest_live_row` must come from the caller — `World` doesn't
    /// currently expose a ready-made "highest live `Entity::index()`" query
    /// on its own; a caller can compute one from `entity_slots.len()` minus
    /// its own trailing-freed-slot bookkeeping, or from whatever tracks
    /// object counts already. Call at a natural boundary (a level unload, a
    /// large despawn batch settling), not every frame — this is a real
    /// GPU-to-GPU copy per buffer that actually shrinks. No-op if no mirror
    /// is attached.
    #[cfg(feature = "gpu")]
    pub fn shrink_gpu_mirror_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) {
        if let Some(mirror) = &self.gpu_mirror {
            mirror.store().shrink_world_mirror_to_fit(queue, highest_live_row, slack_factor);
        }
    }

    /// Debug assertion: every archetype's column lengths must equal its entity
    /// count.  Panics on the first mismatch.  Compiled out in release builds
    /// (the loop body becomes a no-op).
    #[inline]
    pub fn assert_archetype_consistency(&self) {
        #[cfg(debug_assertions)]
        for arch in &self.archetypes {
            let elen = arch.entities.len();
            for (cidx, col) in arch.columns.iter().enumerate() {
                if let Some(c) = col {
                    assert_eq!(
                        c.len(),
                        elen,
                        "ArchetypeId({}) column[{}] len {} != entities.len {} (key={:?})",
                        arch.id.0,
                        cidx,
                        c.len(),
                        elen,
                        arch.key.0,
                    );
                }
            }
        }
    }

    // â”€â”€ Entity lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Pre-allocate storage for `count` entities.  Call before a batch spawn
    /// loop to avoid repeated capacity-doubling reallocations of the slot vec
    /// and the empty archetype's entity vec.
    pub fn reserve_entities(&mut self, count: u32) {
        self.entity_slots.reserve(count as usize);
        self.archetypes[ArchetypeId::EMPTY.0 as usize]
            .entities
            .reserve(count as usize);
    }

    /// Allocate a new entity in the empty archetype.
    ///
    /// Recycles a free slot if one is available; otherwise extends the slot
    /// vec.  The returned handle includes a generation counter that allows
    /// [`is_alive`](Self::is_alive) to detect stale handles after despawn.
    ///
    /// Records into the attached [`crate::replication::SharedChangeTracker`]
    /// automatically, if one is attached ([`Self::attach_change_tracker`]) —
    /// no separate `_tracked` call needed for that; see [`Self::spawn_tracked`]
    /// only if you additionally need to record into a *different*,
    /// explicitly-held tracker.
    pub fn spawn(&mut self) -> Entity {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.spawn_inner(Some(&mut guard))
        } else {
            self.spawn_inner(None)
        }
    }

    /// Like [`spawn`](Self::spawn) but also records the spawn in `tracker`.
    ///
    /// Redundant with plain [`Self::spawn`] once a change tracker is
    /// attached via [`Self::attach_change_tracker`] — in that case this
    /// records into the ATTACHED tracker (same as `spawn` would), not
    /// `tracker`, so the two spellings can't silently diverge once a
    /// `World` opts into automatic tracking. `tracker` is only actually
    /// used when nothing is attached, preserving this method's exact prior
    /// behavior for a `World` that never calls `attach_change_tracker`.
    pub fn spawn_tracked(&mut self, tracker: &mut ChangeTracker) -> Entity {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.spawn_inner(Some(&mut guard))
        } else {
            self.spawn_inner(Some(tracker))
        }
    }

    pub(crate) fn spawn_inner(&mut self, tracker: Option<&mut ChangeTracker>) -> Entity {
        let (idx, gen) = if let Some(idx) = self.free_slots.pop() {
            let slot = &mut self.entity_slots[idx as usize];
            slot.generation = slot.generation.wrapping_add(1);
            slot.archetype = ArchetypeId::EMPTY;
            (idx, slot.generation)
        } else {
            let idx = self.entity_slots.len() as u32;
            self.entity_slots.push(EntitySlot::empty(0));
            (idx, 0)
        };

        let entity = Entity::new(idx, gen);
        let empty = &mut self.archetypes[ArchetypeId::EMPTY.0 as usize];
        let row = empty.entities.len() as u32;
        empty.entities.push(entity);
        self.entity_slots[idx as usize].row = row;

        // GPU liveness mirror: deliberately NOT touched here (SceneDB#39).
        // A freshly spawned entity has no components yet, so at this point
        // there is no way to know whether it will ever carry a `#[gpu]`
        // field -- writing (or even queuing) a generation entry for every
        // spawn regardless would mean every entity in the World pays a
        // GPU-mirror cost the instant a mirror is attached, whether or not
        // it ever touches GPU-mirrored data. Instead, `insert_inner` queues
        // this row's generation the first time (if ever) it actually
        // receives a `#[gpu]`-bearing component -- see
        // `crate::gpu::world_mirror::GenerationMirror::note_gpu_bearing_insert`'s
        // doc for the full contract this splits across spawn/insert/despawn.

        if let Some(t) = tracker {
            t.record_spawn(entity);
        }

        entity
    }

    /// Remove an entity and all its components from the world.
    ///
    /// Returns `false` if the entity was already dead (generation mismatch or
    /// out-of-bounds index).
    ///
    /// The entity's slot is recycled: the generation is incremented and the
    /// index is pushed onto the free list.  The entity's data is
    /// swap-removed from its archetype.
    ///
    /// Records into the attached change tracker automatically, same as
    /// [`Self::spawn`] — see that method's doc.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.despawn_inner(entity, Some(&mut guard))
        } else {
            self.despawn_inner(entity, None)
        }
    }

    /// Like [`despawn`](Self::despawn) but also records the despawn in
    /// `tracker`. Redundant with plain [`Self::despawn`] once a change
    /// tracker is attached — see [`Self::spawn_tracked`]'s doc for why.
    pub fn despawn_tracked(&mut self, entity: Entity, tracker: &mut ChangeTracker) -> bool {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            return self.despawn_inner(entity, Some(&mut guard));
        }
        self.despawn_inner(entity, Some(tracker))
    }

    fn despawn_inner(&mut self, entity: Entity, tracker: Option<&mut ChangeTracker>) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let swapped = self.archetypes[arch_id.0 as usize].remove_row(row);
        if let Some(moved) = swapped {
            self.entity_slots[moved.index() as usize].row = row as u32;
        }
        let slot = &mut self.entity_slots[entity.index() as usize];
        slot.generation = slot.generation.wrapping_add(1);
        let new_generation = slot.generation;
        self.free_slots.push(entity.index());

        // GPU liveness mirror: queue the FRESHLY-BUMPED generation, not
        // `entity`'s own (now-dead) one -- a reader still holding `entity`
        // must see a mismatch against this row going forward, exactly what
        // CPU-side `is_alive` already guarantees for `entity.generation()`
        // vs. `entity_slots[idx].generation`. A genuine no-op (SceneDB#39)
        // if this row never received a `#[gpu]`-bearing component in the
        // first place -- see `GenerationMirror::note_despawn`'s doc. Queued
        // for the next `flush_gpu_mirror`, not written immediately; the
        // "reader must see a mismatch going forward" guarantee only needs to
        // hold by the next point anything actually reads GPU-mirrored state,
        // which is already gated on a flush having happened (same standing
        // assumption every other World-mirrored write in this crate rests
        // on).
        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            mirror.generations().note_despawn(entity.index(), new_generation);
        }

        if let Some(t) = tracker {
            t.record_despawn(entity);
        }

        true
    }

    /// Returns `true` if `entity` is still alive.
    ///
    /// Checks that the slot exists (index in bounds) and that the stored
    /// generation matches the entity handle's generation â€” meaning the slot
    /// hasn't been recycled since the handle was created.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entity_slots
            .get(entity.index() as usize)
            .map(|s| s.generation == entity.generation())
            .unwrap_or(false)
    }

    // â”€â”€ Component helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Fast path: check whether archetype `arch_id` has a column at `cid`.
    #[inline]
    fn has_column_id(arch: &Archetype, cid: ComponentId) -> bool {
        let idx = cid.0 as usize;
        idx < arch.columns.len() && arch.columns[idx].is_some()
    }

    /// Get a mutable reference to the `ErasedColumn` at `cid` in `arch`.
    #[inline]
    fn get_erased_mut(arch: &mut Archetype, cid: ComponentId) -> Option<&mut Box<dyn ErasedColumn>> {
        arch.columns.get_mut(cid.0 as usize).and_then(|c| c.as_mut())
    }

    /// Get a shared reference to the `ErasedColumn` at `cid` in `arch`.
    #[inline]
    fn get_erased(arch: &Archetype, cid: ComponentId) -> Option<&Box<dyn ErasedColumn>> {
        arch.columns.get(cid.0 as usize).and_then(|c| c.as_ref())
    }

    /// Ensure the columns vec is large enough for `cid`, then set it.
    #[inline]
    fn set_column(arch: &mut Archetype, cid: ComponentId, col: Box<dyn ErasedColumn>) {
        let idx = cid.0 as usize;
        for _ in arch.columns.len()..=idx {
            arch.columns.push(None);
        }
        arch.columns[idx] = Some(col);
    }

    /// Collect all CIDs that have a column in this archetype (for migration).
    fn collect_cids(arch: &Archetype) -> Vec<ComponentId> {
        arch.columns
            .iter()
            .enumerate()
            .filter(|(_, col)| col.is_some())
            .map(|(i, _)| ComponentId(i as u32))
            .collect()
    }

    /// Collect all CIDs except `skip` (for migration skip).
    fn collect_cids_skip(arch: &Archetype, skip: ComponentId) -> Vec<ComponentId> {
        arch.columns
            .iter()
            .enumerate()
            .filter(|(i, col)| col.is_some() && ComponentId(*i as u32) != skip)
            .map(|(i, _)| ComponentId(i as u32))
            .collect()
    }

    // â”€â”€ Component operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Add a component to an entity, migrating it to a new archetype if needed.
    ///
    /// If the entity already has a component of type `T`, the value is
    /// overwritten in place (no migration).  Otherwise the entity is moved to
    /// an archetype that includes `T`, preserving all existing component data.
    ///
    /// # Panics
    ///
    /// Panics if `entity` is dead.
    ///
    /// Records into the attached change tracker automatically, same as
    /// [`Self::spawn`] — see that method's doc.
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.insert_inner(entity, value, Some(&mut guard));
        } else {
            self.insert_inner(entity, value, None);
        }
    }

    /// Like [`insert`](Self::insert) but also records the change in
    /// `tracker`. Redundant with plain [`Self::insert`] once a change
    /// tracker is attached — see [`Self::spawn_tracked`]'s doc for why.
    pub fn insert_tracked<T: Component>(&mut self, entity: Entity, value: T, tracker: &mut ChangeTracker) {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.insert_inner(entity, value, Some(&mut guard));
        } else {
            self.insert_inner(entity, value, Some(tracker));
        }
    }

    fn insert_inner<T: Component>(&mut self, entity: Entity, value: T, mut tracker: Option<&mut ChangeTracker>) {
        let cid = crate::component::component_id::<T>();
        assert!(self.is_alive(entity), "insert on dead entity {entity}");

        let (old_arch_id, old_row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let is_new_insert = !Self::has_column_id(&self.archetypes[old_arch_id.0 as usize], cid);

        // GPU mirror: automatic, via a link-time dispatch registry keyed by
        // `cid` (see `crate::gpu::world_mirror`'s module docs for why this
        // is a registry lookup and not compile-time specialization — the
        // short version: `insert_inner` is itself generic, and Rust cannot
        // specialize a method call inside an unconstrained generic body on
        // the caller's substituted type). Reuses `cid`, already computed
        // above for archetype indexing — no extra `TypeId` resolution.
        // Must run before `value` is moved into a column below (both
        // branches move it), and covers BOTH the in-place-update and the
        // new-archetype-migration paths with one call, since either way
        // this is the new authoritative value for `entity`'s `T`. Passes
        // `is_new_insert` through so `Once`-mode `#[gpu]` fields know
        // whether this is the one time they should actually write. A no-op
        // (one `Option::is_none()` check, then at most one `HashMap` lookup
        // that misses for any type `#[derive(SceneStore)]` never touched)
        // when no mirror is attached or `T` has no `#[gpu]` fields.
        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            if let Some(dispatch) = crate::gpu::world_mirror::dispatch_for(cid) {
                // First time this entity gets ANY `#[gpu]`-bearing component
                // (checked via `is_new_insert`, not re-derived here): tell
                // the liveness mirror there is now something on the GPU at
                // this row that needs a generation entry -- and, eventually,
                // invalidating on despawn. See
                // `GenerationMirror::note_gpu_bearing_insert`'s doc; this is
                // the other half of SceneDB#39's "non-GPU entities pay
                // nothing" fix (the first half is that `spawn_inner` no
                // longer touches the liveness mirror at all).
                if is_new_insert {
                    mirror.generations().note_gpu_bearing_insert(entity.index(), entity.generation());
                }
                dispatch(mirror, entity.index(), &value as *const T as *const (), is_new_insert);
            }
        }

        // In-place update: entity already has this component in this archetype.
        if !is_new_insert {
            if let Some(t) = tracker.as_deref_mut() {
                // Capture bytes before the value is moved into the column.
                let len = std::mem::size_of::<T>();
                let bytes = unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, len) };
                t.record_component_change(entity, cid, 0, bytes.to_vec());
            }
            let col = self.archetypes[old_arch_id.0 as usize].column_mut::<T>();
            col.data[old_row] = value;
            return;
        }

        // Archetype-graph edge cache (see `Archetype::add_edges`'s doc): a
        // transition already taken before is two Vec index reads, zero
        // allocation, zero hashing -- the common case for any entity that
        // repeatedly gains/loses the same component shape (tag toggling,
        // pooled respawn, etc). Only a never-before-seen (old_arch_id, cid)
        // pair pays for rebuilding the key and hashing it into
        // `archetype_index`, and that cost is paid exactly once per pair,
        // ever.
        let new_arch_id = match self.archetypes[old_arch_id.0 as usize].add_edge(cid) {
            Some(cached) => cached,
            None => {
                let new_key = self.archetypes[old_arch_id.0 as usize].key.with::<T>();
                let id = self.get_or_create_archetype(new_key);
                self.archetypes[old_arch_id.0 as usize].set_add_edge(cid, id);
                id
            }
        };

        // Ensure Column<T> exists in the destination (may be empty).
        let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
        let idx = cid.0 as usize;
        if let Some(existing) = new_arch.columns.get(idx).and_then(|c| c.as_ref()) {
            debug_assert_eq!(
                ErasedColumn::type_id(existing.as_ref()),
                std::any::TypeId::of::<T>(),
                "insert column type collision at {:?}",
                cid,
            );
        } else {
            Self::set_column(new_arch, cid, Box::new(Column::<T>::new()));
        }

        // Phase 1: push entity + migrate all existing components.
        // migrate_row pushes the entity to the destination first, then
        // transfers every column from the source, then updates all slots.
        self.migrate_row(entity, old_arch_id, old_row, new_arch_id);

        // Phase 2: push the new value.  The destination entity vec has
        // already grown by one, so this keeps all column lengths in sync.
        let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
        new_arch.columns[idx]
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .unwrap()
            .data
            .push(value);

        if let Some(t) = tracker {
            // The value was moved into the column, so we can't read it anymore.
            // Record the change without field data — R2 schema encoding will handle it.
            t.record_component_change(entity, cid, 0, Vec::new());
        }
    }

    /// Remove a component from an entity, returning its value.
    ///
    /// The entity is migrated to an archetype without `T`.  All other
    /// components are preserved.
    ///
    /// Returns `None` if the entity is dead or does not have component `T`.
    ///
    /// Records into the attached change tracker automatically, same as
    /// [`Self::spawn`] — see that method's doc.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            self.remove_inner(entity, Some(&mut guard))
        } else {
            self.remove_inner(entity, None)
        }
    }

    /// Like [`remove`](Self::remove) but also records the change in
    /// `tracker`. Redundant with plain [`Self::remove`] once a change
    /// tracker is attached — see [`Self::spawn_tracked`]'s doc for why.
    pub fn remove_tracked<T: Component>(&mut self, entity: Entity, tracker: &mut ChangeTracker) -> Option<T> {
        if let Some(shared) = self.change_tracker.clone() {
            let mut guard = shared.lock();
            return self.remove_inner(entity, Some(&mut guard));
        }
        self.remove_inner(entity, Some(tracker))
    }

    fn remove_inner<T: Component>(&mut self, entity: Entity, tracker: Option<&mut ChangeTracker>) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let (old_arch_id, old_row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let cid = crate::component::component_id::<T>();
        if !Self::has_column_id(&self.archetypes[old_arch_id.0 as usize], cid) {
            return None;
        }

        // Pull the value out of the column. `remove_inner<T>` already knows
        // the concrete type -- unlike `migrate_row`'s type-erased column
        // carry-over (which genuinely doesn't know the OTHER components'
        // types at compile time), there is no reason to route the value
        // being returned through the erased `swap_remove_erased`/
        // `Box::from_raw` path at all. That path heap-allocates a `Box`
        // just to hand back a type-erased pointer this call immediately
        // downcasts and moves out of anyway -- a wasted allocator round
        // trip for every single `World::remove` call. A direct, safe,
        // typed `Vec::swap_remove` on the concrete `Column<T>` is both
        // simpler and allocation-free.
        let removed_val = self.archetypes[old_arch_id.0 as usize]
            .column_mut::<T>()
            .data
            .swap_remove(old_row);

        // Archetype-graph edge cache -- see `insert_inner`'s identical use
        // of `add_edge`/`Archetype::add_edges`'s doc for the shared
        // rationale.
        let new_arch_id = match self.archetypes[old_arch_id.0 as usize].remove_edge(cid) {
            Some(cached) => cached,
            None => {
                let new_key = self.archetypes[old_arch_id.0 as usize].key.without::<T>();
                let id = self.get_or_create_archetype(new_key);
                self.archetypes[old_arch_id.0 as usize].set_remove_edge(cid, id);
                id
            }
        };

        // Migrate everything except the removed component.
        // migrate_row_skip pushes the entity first, migrates all columns
        // except the skipped one, then updates all slots.
        self.migrate_row_skip(entity, old_arch_id, old_row, new_arch_id, cid);

        if let Some(t) = tracker {
            t.record_component_change(entity, cid, 0, Vec::new());
        }

        Some(removed_val)
    }

    /// Returns a shared reference to component `T` on `entity`, if present.
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let s = &self.entity_slots[entity.index() as usize];
        let arch = &self.archetypes[s.archetype.0 as usize];
        let cid = crate::component::component_id::<T>();
        Self::get_erased(arch, cid).and_then(|c| {
            c.as_any()
                .downcast_ref::<Column<T>>()
                .map(|col| &col.data[s.row as usize])
        })
    }

    /// Returns a mutable borrow of component `T` on `entity`, if present.
    ///
    /// The returned [`Mut<T>`] derefs to `&mut T` exactly like the old raw
    /// `&mut T` this used to return — every ordinary call site keeps working
    /// unchanged. The difference is what happens when it drops: if `T` has
    /// `#[gpu]` fields and a mirror is attached, the mutated value is written
    /// through to the GPU mirror then, the same way `World::insert` already
    /// does on every insert. See [`Mut`]'s doc for why this exists.
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<Mut<'_, T>> {
        if !self.is_alive(entity) {
            return None;
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let cid = crate::component::component_id::<T>();
        let value = Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid).and_then(|c| {
            c.as_any_mut()
                .downcast_mut::<Column<T>>()
                .map(|col| &mut col.data[row])
        })?;

        #[cfg(feature = "gpu")]
        let gpu_hook = self.gpu_mirror.as_ref().and_then(|mirror| {
            crate::gpu::world_mirror::dispatch_for(cid).map(|dispatch| GpuMutHook {
                mirror: mirror.clone(),
                row: entity.index(),
                dispatch,
            })
        });

        let change_hook = self.change_tracker.as_ref().map(|tracker| ChangeMutHook {
            tracker: tracker.clone(),
            entity,
            component_id: cid,
        });

        Some(Mut {
            value,
            #[cfg(feature = "gpu")]
            gpu_hook,
            change_hook,
        })
    }

    // â”€â”€ Archetype graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Spawn `entity` at its EXACT wire index+generation, directly into the
    /// archetype identified by `key`, using `row_ops` to construct any
    /// column the archetype doesn't already have and to fill a placeholder
    /// row in every column of that archetype.
    ///
    /// Used by [`crate::replication::Delta::apply`]: replicated entity
    /// handles are shared verbatim between peers (see `Entity::bits`/
    /// `from_bits` and the replication module doc's "Endianness is a
    /// non-concern" tenet), so a replicated spawn must land at the same
    /// slot the wire value encodes, not the next locally-available one. If
    /// that slot is already live under a different occupant, the incoming
    /// spawn is authoritative and the old occupant is despawned first.
    ///
    /// Every column of the destination archetype (not just the ones newly
    /// created) is grown by one row via its
    /// [`crate::replication::RowOps::push_default`] — a real `T::default()`
    /// push, not a raw byte fill, so this is sound for ANY component type
    /// that implements `Default` (not just `Pod` ones).
    /// [`crate::replication::Delta::apply`] overwrites the real values
    /// afterward via `component_deltas`.
    ///
    /// Returns `None` (leaving the archetype registered but without the
    /// entity) if `row_ops` can't produce ops for one of `key`'s component
    /// ids, **or** if placing `entity` would grow `entity_slots` by more
    /// than `MAX_SLOT_GROWTH_PER_SPAWN` in this one call (see that local
    /// constant's doc, below — a wire-supplied index near `u32::MAX` must
    /// not be able to force a multi-gigabyte allocation).
    pub(crate) fn force_spawn_in_archetype(
        &mut self,
        entity: Entity,
        key: ArchetypeKey,
        mut row_ops: impl FnMut(ComponentId) -> Option<crate::replication::RowOps>,
    ) -> Option<Entity> {
        let idx = entity.index();
        let gen = entity.generation();

        // `idx` is wire-supplied (`Entity::index()` off a replicated,
        // attacker-controlled `Delta::spawned` entry) and the loop below
        // must grow `entity_slots`/`free_slots` to at least `idx + 1` to
        // place the entity there. Unbounded, that lets a single spawn with
        // an index near `u32::MAX` force a huge allocation before a single
        // real entity has been created — confirmed by fuzzing
        // (`delta_apply`'s `oom-*` crash artifacts): a wire index of
        // ~4.06e9 drove `entity_slots`'s `Vec<EntitySlot>` (12 bytes/elem)
        // to a real `malloc(6_442_450_944)` — a single ~6 GiB reallocation
        // (`Vec`'s doubling growth landing on 2^29 elements) — before any
        // of the earlier, smaller reallocations even had a chance to fail
        // gracefully.
        //
        // Bounding *this call's* growth (not the world's total size) is
        // the right invariant: a legitimately large, long-lived world
        // still reaches millions of entity slots over its lifetime just
        // fine, because that growth happens incrementally — one entity at
        // a time, across many separate `Delta`s — never as one call
        // demanding millions of new slots at once the way a single
        // malicious spawn does.
        const MAX_SLOT_GROWTH_PER_SPAWN: u32 = 1 << 20; // ~1M slots, ~12 MiB of EntitySlot
        let current_len = self.entity_slots.len() as u32;
        if idx > current_len && idx - current_len > MAX_SLOT_GROWTH_PER_SPAWN {
            return None;
        }

        while (self.entity_slots.len() as u32) <= idx {
            let new_idx = self.entity_slots.len() as u32;
            self.entity_slots.push(EntitySlot::empty(0));
            self.free_slots.push(new_idx);
        }
        if !self.free_slots.contains(&idx) {
            // `idx` is currently live under some other occupant — the
            // incoming delta is authoritative.
            let old_gen = self.entity_slots[idx as usize].generation;
            self.despawn(Entity::new(idx, old_gen));
        }
        self.free_slots.retain(|&s| s != idx);
        self.entity_slots[idx as usize] = EntitySlot {
            generation: gen,
            archetype: ArchetypeId::EMPTY,
            row: 0,
        };

        let arch_id = self.get_or_create_archetype(key.clone());
        for &cid in &key.0 {
            if !Self::has_column_id(&self.archetypes[arch_id.0 as usize], cid) {
                let ops = row_ops(cid)?;
                let col = (ops.new_column)();
                Self::set_column(&mut self.archetypes[arch_id.0 as usize], cid, col);
            }
        }

        let row = self.archetypes[arch_id.0 as usize].entities.len() as u32;
        self.archetypes[arch_id.0 as usize].entities.push(entity);
        self.entity_slots[idx as usize].archetype = arch_id;
        self.entity_slots[idx as usize].row = row;

        let n = self.archetypes[arch_id.0 as usize].active_cids.len();
        for i in 0..n {
            let cid = self.archetypes[arch_id.0 as usize].active_cids[i];
            let ops = row_ops(cid)?;
            let col = Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid).unwrap();
            (ops.push_default)(col.as_mut());
        }

        Some(entity)
    }

    /// Write one field's value into `entity`'s existing column for `cid` at
    /// its row, via the field's own
    /// [`crate::replication::FieldOps::decode_into`] closure — no raw
    /// pointers, no byte-width guessing; the closure downcasts to the
    /// concrete column type itself.
    ///
    /// A dead entity or a missing column is a silent no-op (`Ok(())`) —
    /// expected in ordinary operation (e.g. a stale delta arriving after a
    /// despawn, or a relevance change) — but a genuine decode failure from
    /// `decode_into` itself (malformed bytes) is propagated as `Err`.
    pub(crate) fn write_component_field(
        &mut self,
        entity: Entity,
        cid: ComponentId,
        decode_into: &(dyn Fn(&mut dyn ErasedColumn, usize, &[u8]) -> Result<(), crate::replication::ErrorCode>
              + Send
              + Sync),
        bytes: &[u8],
    ) -> Result<(), crate::replication::ErrorCode> {
        if !self.is_alive(entity) {
            return Ok(());
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let Some(col) = Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid) else {
            return Ok(());
        };
        decode_into(col.as_mut(), row, bytes)
    }

    pub(crate) fn get_or_create_archetype(&mut self, key: ArchetypeKey) -> ArchetypeId {
        if let Some(&id) = self.archetype_index.get(&key) {
            return id;
        }
        let id = ArchetypeId(self.archetypes.len() as u32);
        self.archetypes.push(Archetype::new(id, key.clone()));
        self.archetype_index.insert(key, id);
        id
    }

    // â”€â”€ Bundle support â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Edge-cached "insert `T`" archetype-graph step -- the exact same
    /// cache-or-rebuild-key logic `insert_inner` uses for a single
    /// component, extracted so [`crate::bundle::Bundle`] impls can chain it
    /// once per component without duplicating the branch. Each call after
    /// the first for a given `(from, T)` pair is two `Vec` index reads, no
    /// allocation, no hashing -- see [`Archetype::add_edges`]'s doc for why.
    pub(crate) fn step_add_edge<T: Component>(&mut self, from: ArchetypeId) -> ArchetypeId {
        let cid = crate::component::component_id::<T>();
        match self.archetypes[from.0 as usize].add_edge(cid) {
            Some(cached) => cached,
            None => {
                let new_key = self.archetypes[from.0 as usize].key.with::<T>();
                let id = self.get_or_create_archetype(new_key);
                self.archetypes[from.0 as usize].set_add_edge(cid, id);
                id
            }
        }
    }

    /// Push one [`crate::bundle::Bundle`] component's value as a BRAND NEW
    /// column entry onto `arch_id`, at the row `entity` already occupies
    /// there (the caller -- [`World::spawn_bundle_inner`] -- pushes `entity`
    /// onto `arch_id.entities` before calling this for any of the bundle's
    /// components, so every column's `Vec::push` here keeps that same row
    /// index in sync across every column, exactly like `insert_inner`'s
    /// "phase 1 migrate, phase 2 push new value" ordering does for a single
    /// component).
    ///
    /// Ensures the destination column exists (creating an empty one on first
    /// use, same as `insert_inner`), runs the identical GPU-mirror dispatch
    /// and liveness-mirror bookkeeping `insert_inner` runs for a
    /// new-component insert (`is_new_insert = true` unconditionally --
    /// every component in a bundle handed to a freshly spawned entity is,
    /// by construction, new to it), then records the change in `tracker`
    /// if present.
    pub(crate) fn push_new_component<T: Component>(
        &mut self,
        arch_id: ArchetypeId,
        entity: Entity,
        value: T,
        tracker: &mut Option<&mut ChangeTracker>,
    ) {
        let cid = crate::component::component_id::<T>();

        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            if let Some(dispatch) = crate::gpu::world_mirror::dispatch_for(cid) {
                mirror.generations().note_gpu_bearing_insert(entity.index(), entity.generation());
                dispatch(mirror, entity.index(), &value as *const T as *const (), true);
            }
        }

        let arch = &mut self.archetypes[arch_id.0 as usize];
        let idx = cid.0 as usize;
        if arch.columns.get(idx).and_then(|c| c.as_ref()).is_none() {
            Self::set_column(arch, cid, Box::new(Column::<T>::new()));
        }
        self.archetypes[arch_id.0 as usize].columns[idx]
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .unwrap()
            .data
            .push(value);

        if let Some(t) = tracker.as_deref_mut() {
            t.record_component_change(entity, cid, 0, Vec::new());
        }
    }

    /// Reserve capacity for `additional` more `T` values in `arch_id`'s
    /// column, creating the (empty) column first if this archetype doesn't
    /// have one yet. Used by [`crate::bundle::Bundle::reserve_columns`]
    /// (via [`World::reserve_bundle`]) so a tight `spawn_bundle` loop over a
    /// known entity count doesn't pay for repeated `Vec` capacity-doubling
    /// on every column the bundle touches -- the same reason
    /// [`World::reserve_entities`] exists for the empty archetype's own
    /// entity list.
    pub(crate) fn reserve_component_column<T: Component>(&mut self, arch_id: ArchetypeId, additional: u32) {
        let cid = crate::component::component_id::<T>();
        let arch = &mut self.archetypes[arch_id.0 as usize];
        let idx = cid.0 as usize;
        if arch.columns.get(idx).and_then(|c| c.as_ref()).is_none() {
            Self::set_column(arch, cid, Box::new(Column::<T>::new()));
        }
        self.archetypes[arch_id.0 as usize].columns[idx]
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .unwrap()
            .data
            .reserve(additional as usize);
    }

    // â”€â”€ Archetype migration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Moves one column's element at `old_row` in `old_arch_id` into a
    /// (possibly newly-created) column of the same `cid` in `new_arch_id`.
    /// Shared by [`Self::migrate_row`]/[`Self::migrate_row_skip`].
    ///
    /// Uses the zero-allocation [`ErasedColumn::swap_remove_into`]/
    /// [`ErasedColumn::push_from`] path (through a stack-allocated
    /// [`MoveScratch`]) whenever the element fits it -- true for the
    /// overwhelming majority of real components (anything up to 128 bytes
    /// at up to 16-byte alignment; a 4x4 `[f32; 16]` transform matrix is
    /// exactly 64 bytes at 4-byte alignment, well inside it). Before this,
    /// EVERY column moved during EVERY migration went through
    /// [`ErasedColumn::swap_remove_erased`]/[`ErasedColumn::push_erased`] --
    /// a `Box::new` heap allocation on the way out and a matching
    /// deallocation on the way in, for every single component on every
    /// single entity that migrates. An entity with N components paid for
    /// 2N allocator round trips on every `insert`/`remove` that changed its
    /// archetype. Only a component whose size/alignment genuinely exceeds
    /// the inline scratch capacity (rare) still pays that cost, via the
    /// original path kept as a correctness fallback.
    #[inline]
    fn move_column_row(&mut self, old_arch_id: ArchetypeId, old_row: usize, new_arch_id: ArchetypeId, cid: ComponentId) {
        if !Self::has_column_id(&self.archetypes[new_arch_id.0 as usize], cid) {
            let proto = Self::get_erased(&self.archetypes[old_arch_id.0 as usize], cid)
                .unwrap()
                .new_empty();
            Self::set_column(&mut self.archetypes[new_arch_id.0 as usize], cid, proto);
        }

        let (elem_size, elem_align) = {
            let src = Self::get_erased(&self.archetypes[old_arch_id.0 as usize], cid).unwrap();
            (src.element_size(), src.element_align())
        };

        if elem_size <= MoveScratch::CAP && elem_align <= MoveScratch::ALIGN {
            let mut scratch = MoveScratch::new();
            // SAFETY: `scratch` is `MoveScratch::CAP` bytes, `repr(align(16))`
            // (>= `MoveScratch::ALIGN`), and the check above confirms this
            // column's element fits both bounds -- `swap_remove_into`'s
            // safety contract (valid for writes, sufficiently aligned)
            // holds. `old_row` is in bounds because the caller only ever
            // reaches here for a real occupied row of `old_arch_id`.
            unsafe {
                Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .swap_remove_into(old_row, scratch.as_mut_ptr());
            }
            // SAFETY: `scratch` now holds a valid, initialized, properly
            // aligned instance of this column's element type -- written by
            // `swap_remove_into` immediately above, with no intervening use
            // of `scratch`. `push_from` is called exactly once per
            // `swap_remove_into` on this path, upholding the "logical
            // ownership handoff, exactly once" contract both methods
            // document -- no double-move, no leak.
            unsafe {
                Self::get_erased_mut(&mut self.archetypes[new_arch_id.0 as usize], cid)
                    .unwrap()
                    .push_from(scratch.as_ptr());
            }
        } else {
            // Fallback: an element too large or too strictly aligned for
            // the inline scratch buffer (rare) -- the original Box-based
            // path. Still fully correct, just not allocation-free.
            let ptr = unsafe {
                Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .swap_remove_erased(old_row)
            };
            unsafe {
                Self::get_erased_mut(&mut self.archetypes[new_arch_id.0 as usize], cid)
                    .unwrap()
                    .push_erased(ptr);
            }
        }
    }


    /// Move the entity and all component data from `old_arch_id`/`old_row`
    /// into `new_arch_id`.
    ///
    /// Order of operations (single cohesive window):
    /// 1. Push entity to destination `.entities` (first).
    /// 2. For each component in `active_cids` of the source archetype:
    ///    swap-remove from the source column, ensure the destination column
    ///    exists, and push into it.
    /// 3. Swap-remove the entity from the source archetype and fix the
    ///    swapped-in entity's slot row.
    /// 4. Update the migrated entity's slot.
    ///
    /// The caller is responsible for pushing any *new* component value (not
    /// present in the source archetype) *after* this returns.
    fn migrate_row(
        &mut self,
        entity: Entity,
        old_arch_id: ArchetypeId,
        old_row: usize,
        new_arch_id: ArchetypeId,
    ) {
        // Phase 1: push entity to destination first.
        let new_row = self.archetypes[new_arch_id.0 as usize]
            .entities
            .len() as u32;
        self.archetypes[new_arch_id.0 as usize]
            .entities
            .push(entity);

        // Phase 2: broadcast each source column into the destination using
        // the pre-computed `active_cids` slice (no heap allocation for the
        // slice itself, and -- via `move_column_row`'s inline scratch path
        // -- none for the moved element's bytes either, for the
        // overwhelming majority of component types).
        let n = self.archetypes[old_arch_id.0 as usize].active_cids.len();
        for i in 0..n {
            let cid = {
                // isolated immutable borrow â€” released before the mutable one
                let src = &self.archetypes[old_arch_id.0 as usize];
                src.active_cids[i]
            };
            self.move_column_row(old_arch_id, old_row, new_arch_id, cid);
        }

        // Phase 3: remove entity from old archetype; fix swapped-in slot.
        let moved = {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            old_arch.entities.swap_remove(old_row);
            if old_row < old_arch.entities.len() {
                Some(old_arch.entities[old_row])
            } else {
                None
            }
        };
        if let Some(m) = moved {
            self.entity_slots[m.index() as usize].row = old_row as u32;
        }

        // Phase 4: update the migrated entity's slot.
        let slot = &mut self.entity_slots[entity.index() as usize];
        slot.archetype = new_arch_id;
        slot.row = new_row;
    }

    /// Move all components EXCEPT `skip_cid` and push the entity into the
    /// destination archetype.
    ///
    /// Same ordering as [`migrate_row`]: entity first, then columns, then
    /// slot updates.
    fn migrate_row_skip(
        &mut self,
        entity: Entity,
        old_arch_id: ArchetypeId,
        old_row: usize,
        new_arch_id: ArchetypeId,
        skip_cid: ComponentId,
    ) {
        // Phase 1: push entity to destination first.
        let new_row = self.archetypes[new_arch_id.0 as usize]
            .entities
            .len() as u32;
        self.archetypes[new_arch_id.0 as usize]
            .entities
            .push(entity);

        // Phase 2: migrate all columns except `skip_cid` (see
        // `migrate_row`'s identical use of `move_column_row` for the
        // zero-allocation rationale).
        let n = self.archetypes[old_arch_id.0 as usize].active_cids.len();
        for i in 0..n {
            let cid = {
                let src = &self.archetypes[old_arch_id.0 as usize];
                src.active_cids[i]
            };
            if cid == skip_cid {
                continue;
            }
            self.move_column_row(old_arch_id, old_row, new_arch_id, cid);
        }

        // Phase 3: remove entity from old archetype; fix swapped-in slot.
        let moved = {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            old_arch.entities.swap_remove(old_row);
            if old_row < old_arch.entities.len() {
                Some(old_arch.entities[old_row])
            } else {
                None
            }
        };
        if let Some(m) = moved {
            self.entity_slots[m.index() as usize].row = old_row as u32;
        }

        // Phase 4: update the migrated entity's slot.
        let slot = &mut self.entity_slots[entity.index() as usize];
        slot.archetype = new_arch_id;
        slot.row = new_row;
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}