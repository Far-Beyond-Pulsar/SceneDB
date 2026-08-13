//! Per-view token + expected-generation GPU upload (M3-β T1, design §3.1):
//! the device-side landing point for [`super::HarvestStaging`]'s per-class
//! token/gens columns. Helio's M3-β cull compute pass (a later task) is the
//! consumer — it fetches `tokens[i]` (a global row) and validates it against
//! `generations[slot_mirror[tokens[i]]]` using `expected_gens[i]` as the
//! snapshot-time comparand (§3.1's fails-closed generation check).
//!
//! **Placement note:** this is a NEW module (`gpu::view_upload`), not folded
//! into `harvest.rs` or `scene_store.rs`. Rationale: `harvest.rs` is
//! CPU-only host-side routing (no `wgpu` import anywhere in that file) and
//! stays that way by design (its own doc: "One cell, one view, one scan" is
//! a pure-Rust scan over `SpatialCell`); bolting a `wgpu::Buffer`-owning type
//! onto it would be the first `wgpu` dependency in an otherwise
//! graphics-adjacent-but-CPU-only file. `scene_store.rs` owns the *scene's*
//! persistent, per-cell, region-partitioned SSBOs (`SceneGpuStore`) —
//! `ViewTokenBuffers` is neither persistent-scene state nor region-scoped:
//! it is **per-view, per-frame scratch** that is cleared and re-populated
//! every harvest (mirroring `HarvestStaging`'s own frame-scratch nature,
//! just moved one step further, onto the device). A new sibling module,
//! parallel to how `assets.rs` (write-once residency) sits beside
//! `scene_store.rs` (per-frame scene SSBOs) as a distinct ownership shape,
//! keeps this distinction visible at the file level rather than overloading
//! either existing file with a third.
//!
//! **Growth model — deliberately NOT `SceneBuffer`'s.** `SceneBuffer`/
//! `SceneGpuStore` regions are fixed-capacity-at-registration and hard-error
//! on overflow ("scene buffers never reallocate" — design §8's frame-scratch
//! sizing philosophy for the SCENE side). `ViewTokenBuffers` instead mirrors
//! `HarvestStaging`'s OWN Vecs one layer down: `traditional`/`vg`/`hlod` (and
//! their gens columns) are plain `Vec<u32>` that grow via ordinary
//! amortized-growth `Vec` semantics and never shrink. A per-view token
//! buffer's natural sizing worst case (every resident row in view) is not
//! knowable up front the way a cell's row capacity is, so this module grows
//! the device buffer on demand with slack (1.5x, floored at the needed
//! count), never reallocating once a session's high-water mark is reached —
//! the GPU-side continuation of the same "grow like a Vec, then go quiet"
//! discipline the §8.1 alloc gates already hold `HarvestStaging` to.

use super::{
    BufferAccess, BufferKey, BufferRegistrationError, EngineGpuContext, GpuBufferRegistry,
    HarvestStaging, MeshClass, MirrorMode,
};

/// Owns one class's per-view token buffer + its positionally-aligned
/// expected-generation buffer (design §3.1): both `STORAGE | COPY_DST |
/// COPY_SRC` `u32` SSBOs, uploaded from a [`HarvestStaging`] column pair via
/// [`Self::upload`]. One instance per `(view, class)` the driver cares to
/// keep resident; frame-scratch in nature (re-uploaded every harvest), but
/// the wgpu buffers themselves persist across frames — only their content
/// and logical `count` change.
pub struct ViewTokenBuffers {
    tokens: wgpu::Buffer,
    expected_gens: wgpu::Buffer,
    /// Element (u32) capacity of BOTH buffers — they are always resized in
    /// lockstep (see module doc: one pair, one owner, one growth decision).
    capacity: u32,
    /// Logical count of valid entries as of the last [`Self::upload`] (may
    /// be less than `capacity`; the tail beyond `count` holds stale bytes
    /// from a prior, larger upload — callers must dispatch/read over `count`
    /// only, never `capacity`, mirroring `SceneGpuStore::slot_mirror_buffer`'s
    /// own "never index past the harvested count" contract).
    count: u32,
    /// Test 13 instrumentation (see `upload_count` below): counts
    /// `queue.write_buffer` calls — one for the tokens column, one for the
    /// gens column, per non-empty [`Self::upload`] (the T7 per-`write_buffer`
    /// convention this crate's other upload counters already follow).
    upload_count: u64,
    label: String,
    /// Bumped by exactly one on every [`Self::ensure_capacity`] reallocation
    /// — the same "epoch" concept `DynamicGpuBuffer`/`GrowableSceneBuffer`
    /// already expose, added here so [`Self::register_keys`] can sync a
    /// registered [`GpuBufferRegistry`] entry idempotently (bump the
    /// registry's epoch only when a real reallocation happened, not on
    /// every call).
    growth_epoch: u64,
}

impl ViewTokenBuffers {
    /// Amortized growth factor applied when `upload`'s needed count exceeds
    /// current capacity: `next = max(needed, capacity * 3 / 2)`, mirroring
    /// `Vec`'s own ~1.5x growth curve (module doc).
    const GROWTH_NUMER: u32 = 3;
    const GROWTH_DENOM: u32 = 2;

    /// `initial_capacity` may be 0 (a zero-size wgpu buffer is valid; the
    /// first non-empty upload grows it). `label` is a debug label prefix —
    /// `"{label}-tokens"` / `"{label}-expected-gens"` name the two buffers.
    #[must_use]
    pub fn new(ctx: &EngineGpuContext, label: &str, initial_capacity: u32) -> Self {
        let (tokens, expected_gens) = Self::alloc_pair(ctx, label, initial_capacity);
        Self {
            tokens,
            expected_gens,
            capacity: initial_capacity,
            count: 0,
            upload_count: 0,
            label: label.to_string(),
            growth_epoch: 0,
        }
    }

    fn alloc_pair(ctx: &EngineGpuContext, label: &str, capacity: u32) -> (wgpu::Buffer, wgpu::Buffer) {
        let size = capacity as u64 * 4;
        let make = |suffix: &str| {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}-{suffix}")),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        (make("tokens"), make("expected-gens"))
    }

    /// Grow (never shrink) to hold at least `needed` elements, if necessary.
    /// A no-op — zero `create_buffer` calls, zero device work — once `needed
    /// <= capacity`, which is the steady state after warm-up (the alloc-gate
    /// test asserts exactly this: identical `needed` across two calls issues
    /// no growth on the second).
    fn ensure_capacity(&mut self, ctx: &EngineGpuContext, needed: u32) {
        if needed <= self.capacity {
            return;
        }
        let grown = (self.capacity * Self::GROWTH_NUMER / Self::GROWTH_DENOM).max(needed);
        let (tokens, expected_gens) = Self::alloc_pair(ctx, &self.label, grown);
        self.tokens = tokens;
        self.expected_gens = expected_gens;
        self.capacity = grown;
        self.growth_epoch = self.growth_epoch.wrapping_add(1);
    }

    /// Upload `class`'s token + expected-gen columns from `staging` (§3.1's
    /// data path arriving at the device boundary). `debug_assert`s the T8
    /// invariant (`tokens.len() == gens.len()`) one more time here, at the
    /// last point before it becomes two independent `write_buffer` calls —
    /// a future edit that broke pairing between `harvest_cell` and this call
    /// site would otherwise surface only as a silently-misaligned GPU read.
    ///
    /// **Empty-upload decision (documented, per plan):** an empty column
    /// (`staging`'s class array has zero entries — no cell/view combination
    /// harvested anything into this class this frame) issues ZERO
    /// `write_buffer` calls and does NOT increment [`Self::upload_count`].
    /// `count()` still updates to 0 so a consumer dispatching over `count()`
    /// naturally dispatches zero threads — there is nothing to upload and
    /// nothing to validate, so treating it as "an upload of zero bytes"
    /// would only add busywork (and a misleading counter bump) for no
    /// observable effect.
    pub fn upload(&mut self, ctx: &EngineGpuContext, staging: &HarvestStaging, class: MeshClass) {
        let (tokens, gens): (&[u32], &[u32]) = match class {
            MeshClass::Traditional => (&staging.traditional, &staging.traditional_gens),
            MeshClass::VirtualGeometry => (&staging.vg, &staging.vg_gens),
            MeshClass::HlodProxy => (&staging.hlod, &staging.hlod_gens),
        };
        debug_assert_eq!(
            tokens.len(),
            gens.len(),
            "ViewTokenBuffers::upload: tokens/gens must stay positionally aligned (§3.1) \
             at the device boundary"
        );
        self.count = tokens.len() as u32;
        if tokens.is_empty() {
            return;
        }
        self.ensure_capacity(ctx, self.count);
        ctx.queue().write_buffer(&self.tokens, 0, super::as_bytes(tokens));
        self.upload_count += 1;
        ctx.queue().write_buffer(&self.expected_gens, 0, super::as_bytes(gens));
        self.upload_count += 1;
    }

    pub fn tokens_buffer(&self) -> &wgpu::Buffer {
        &self.tokens
    }

    pub fn expected_gens_buffer(&self) -> &wgpu::Buffer {
        &self.expected_gens
    }

    /// Valid-entry count as of the last [`Self::upload`] — dispatch/readback
    /// bound, never `capacity()` (see the field doc: the tail beyond `count`
    /// may hold stale bytes from an earlier, larger upload).
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Current element capacity of both buffers (equal for both — see the
    /// field doc). Test-observable to assert the grow-then-quiet discipline.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Test 13 instrumentation: the teardown gate asserts these do not move
    /// across the renderer drop/rebind window.
    #[doc(hidden)]
    pub fn upload_count(&self) -> u64 {
        self.upload_count
    }

    /// Bump count since construction, incremented exactly once per
    /// [`Self::ensure_capacity`] reallocation (i.e. per grow triggered by
    /// [`Self::upload`]). Compare against a previously cached value to know
    /// whether [`Self::tokens_buffer`]/[`Self::expected_gens_buffer`]
    /// changed identity since the last observation.
    pub fn epoch(&self) -> u64 {
        self.growth_epoch
    }

    /// Issue #41 Tier-2 collapse ("grid/view-token... buffers reachable by
    /// key through one registry"): register (first call) or idempotently
    /// re-sync (every later call) this instance's token/expected-gens
    /// buffers under caller-chosen keys in `registry`.
    ///
    /// Per-view/per-class instances have no single well-known key the way
    /// the singleton asset stores in `gpu/assets.rs` do (module doc: this
    /// type is deliberately per-`(view, class)` scratch, constructed
    /// per-instance with a caller-supplied `label`), so — unlike
    /// `MeshRegistry::register_key` etc. — the caller supplies both keys
    /// (e.g. `BufferKey::of("builtin_view_token_traditional")` for a fixed
    /// small set of well-known views/classes the integrator names ahead of
    /// time).
    ///
    /// Safe and cheap to call every frame after [`Self::upload`]: the first
    /// call registers; every later call is a `registry.sync` (idempotent,
    /// no-op unless [`Self::epoch`] actually advanced since the last sync —
    /// see [`GpuBufferRegistry::sync`]'s doc), so a caller that always calls
    /// this right after `upload` keeps the registry's view current without
    /// spuriously bumping the registry's own epoch on frames where nothing
    /// grew.
    pub fn register_keys(
        &self,
        registry: &GpuBufferRegistry,
        tokens_key: BufferKey,
        expected_gens_key: BufferKey,
    ) -> Result<(), BufferRegistrationError> {
        // `register_row`'s freshly-created entry always starts at epoch 0,
        // regardless of `self.growth_epoch`'s value at first-registration
        // time (a caller may register after this instance already grew a
        // few times). Always following registration with a `sync` call
        // brings the registry's epoch up to `self.growth_epoch` in that
        // case too, so a SUBSEQUENT same-growth_epoch call is the no-op
        // `sync` is documented to be — without this, the first post-
        // registration `sync` would spuriously look like a "growth" (0 !=
        // self.growth_epoch) even when nothing grew between the two calls.
        if !registry.contains_key(tokens_key) {
            registry.register_row::<u32>(tokens_key, self.tokens.clone(), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)?;
        }
        registry.sync(tokens_key, self.tokens.clone(), self.growth_epoch);

        if !registry.contains_key(expected_gens_key) {
            registry.register_row::<u32>(expected_gens_key, self.expected_gens.clone(), BufferAccess::ReadOnly, MirrorMode::DirtyTracked)?;
        }
        registry.sync(expected_gens_key, self.expected_gens.clone(), self.growth_epoch);
        Ok(())
    }
}
