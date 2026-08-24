//! Concentric streaming grid — pure logic (design Rev 2 §4, spec §5/§5.3/§5.5).
//!
//! Classifies each tracked cell into a residency [`Domain`] (`Outer` →
//! `Warm`* → `Margin` → `Inner`) from the observer set, with §5.5 hysteresis
//! to damp boundary jitter, and tracks a per-cell cross-fade `alpha` (§5.2).
//! (*`Warm` — a system-RAM residency tier between "not tracked anywhere" and
//! "GPU-resident" — is OPT-IN via [`GridConfig::warm`]; without it the
//! machine is exactly the legacy `Outer`/`Margin`/`Inner` ladder.)
//! Classification itself (`classify`, `commit_transition`, `advance_crossfade`)
//! stays PURE LOGIC: it decides *what* should transition and queues the
//! decision as a [`Transition`], touching neither `SceneGpuStore` nor wgpu.
//! [`execute_transitions`] (M2b-β T5) is this module's one exception: it
//! drains [`StreamingGrid::take_transitions`] and wires each transition to
//! `SceneGpuStore::register_cell`/`unregister_cell`, and
//! [`StreamingGrid::write_cell_metadata`] packs the per-cell α/domain SSBO
//! straight from wgpu — both live here because the grid's committed
//! domain/α state is exactly what they read. This module lives under `gpu`
//! because it depends on [`super::RegionClassConfig`] for the budget check
//! below and, now, on the store/wgpu types those two items touch.
//!
//! ## β simplification: grid is XZ-planar
//!
//! A cell's bounds are unbounded on Y (`[-inf, inf]`): observer altitude
//! never affects classification. Spec §5 allows a `[min_y, max_y]` cell
//! extent; that's out of scope for β and is a documented simplification, not
//! an oversight.
//!
//! ## Classification: the hysteresis band machine (authoritative)
//!
//! A cell's **base bounds** are its world AABB from `coord × cell_width`
//! (XZ only, Y unbounded — see above). All AABB tests below use **closed**
//! intervals: touching faces count as intersecting (crate-wide §8.2
//! discipline; see `spatial.rs`). Let `pad = pad_fraction × cell_width`
//! (§5.5 Δpad) and `hyst = hysteresis` (§5.5 δhyst).
//!
//! Per cell, up to six concentric zones are derived from the base bounds,
//! each tested for intersection against the observer union (any-of):
//!
//! | zone             | base grown by                       | role                        |
//! |------------------|-------------------------------------|-----------------------------|
//! | `inner_promote`  | `pad`                               | Margin→Inner trigger        |
//! | `inner_demote`   | `pad + hyst`                        | Inner→Margin hold zone      |
//! | `margin_promote` | `margin_radius + pad`               | Outer/Warm→Margin trigger   |
//! | `margin_demote`  | `margin_radius + pad + hyst`        | Margin→{Outer,Warm} hold    |
//! | `warm_promote`*  | `warm.radius + warm.pad_fraction·w` | Outer→Warm trigger          |
//! | `warm_demote`*   | `warm.radius + warm pad + warm hyst`| Warm→Outer hold zone        |
//!
//! (*opt-in RAM tier only — see the tier section below.)
//!
//! Transition rules — **at most one step per `classify` call**, evaluated
//! from the cell's *committed* domain only. With the RAM tier OFF:
//!
//! - `Outer`: intersects `margin_promote` → queue `→Margin`. Else nothing.
//! - `Margin`: intersects `inner_promote` → queue `→Inner`; else if NOT
//!   intersecting `margin_demote` → queue `→Outer`; else nothing.
//! - `Inner`: NOT intersecting `inner_demote` → queue `→Margin`. Else
//!   nothing.
//!
//! With the RAM tier ON (full rules in the tier section below): `Outer`
//! additionally checks `warm_promote` (after `margin_promote`, which keeps
//! its fast path), `Warm` promotes on `margin_promote` and cools to `Outer`
//! past `warm_demote`, and `Margin` demotes to `Warm` instead of `Outer`.
//! Every GPU-residency boundary (`margin_promote`, `margin_demote`,
//! `inner_*`) keeps its exact legacy threshold either way — the tier only
//! changes what happens *between* those boundaries.
//!
//! The promotion boundary stands `pad` proud of the unpadded region edge
//! (§5.5 PromotionBoundary = CellBounds + Δpad: an observer promotes
//! *earlier* than the plain edge), and the demotion boundary stands a
//! further `hyst` beyond that. The gap between them IS the §5.5 hysteresis
//! band: an observer parked (or jittering) anywhere inside it triggers no
//! transition in either direction. Multi-ring promotion (`Outer→Inner`)
//! therefore takes two `classify` calls — one step each — as does the
//! symmetric demotion cascade.
//!
//! **α**: a transition crossing INTO GPU residency (`to` ∈ {`Margin`,
//! `Inner`}, more resident than `from`) sets `alpha_target = 1.0`; every
//! other transition — demotions, and promotions into the RAM tier, which
//! has no on-screen content — sets `alpha_target = 0.0` (applied in
//! [`StreamingGrid::commit_transition`], never in `classify`).
//! [`StreamingGrid::advance_crossfade`] moves `alpha` linearly toward
//! `alpha_target` by `distance / fade_distance`, clamped to `[0, 1]`.
//!
//! ## The system-RAM residency tier (`Warm`) — opt-in (issue #43)
//!
//! Between "not tracked anywhere" (`Outer`: no GPU buffer, nothing staged)
//! and "GPU-resident" (`Margin`/`Inner`) sits an optional third residency
//! state: `Warm`, a cell whose source data is loaded into **system RAM**,
//! ready for a cheap GPU promote, occupying no VRAM. Design decisions
//! (resolving the open questions in issue #43):
//!
//! - **One state machine, not an orthogonal flag.** `Warm` is a [`Domain`]
//!   variant (`Outer < Warm < Margin < Inner` by [`domain_rank`]), so the
//!   existing single-step/hysteresis/drain machinery governs the RAM tier
//!   too; no parallel promotion path to keep in sync.
//! - **Band geometry.** `warm_promote`/`warm_demote` reuse the §5.5 pad +
//!   hysteresis construction with their OWN radius/pad/hysteresis pair
//!   ([`WarmTierConfig`] — typically wider than the GPU pair). Two
//!   deliberate asymmetries preserve legacy GPU timing exactly:
//!   *promotion fast path* — `Outer` checks `margin_promote` FIRST, so an
//!   observer sprinting in skips RAM staging entirely and registers straight
//!   away (`Outer→Margin→Inner`, two steps, as today); *graded teardown* —
//!   leaving drops GPU first at the unchanged `margin_demote`
//!   (`Margin→Warm`, VRAM freed, RAM retained), then RAM one wider band
//!   later (`Warm→Outer`). A player who steps just outside the margin and
//!   back re-promotes from the warm copy without any reload.
//! - **SceneDB owns accounting, not bytes.** The grid tracks the state and
//!   a count ([`StreamingGrid::ram_cached_count`]); the actual cached bytes
//!   stay in an engine-side asset cache, reached through caller-supplied
//!   hooks ([`RamHooks`]) passed to [`execute_transitions_with_ram`] — the
//!   same callback shape as `execute_transitions`' `class_of`. Disk/network
//!   asset delivery stays out of scope exactly as before.
//! - **Budget.** [`StreamingBudget::ram_budget`] /
//!   [`StreamingBudget::max_ram_cached_cells`] are validated against each
//!   other (and `max_materialized_cells`) once at construction, mirroring
//!   the VRAM checks. At runtime, `classify` refuses to queue `Outer→Warm`
//!   while `max_ram_cached_cells` warm cells exist (graceful hold-and-retry,
//!   §8-style); LRU-style eviction of warm cells under pressure is future
//!   work. Pins bypass the cap, as they bypass every concentric rule.
//! - **Shader-visible encoding is unchanged.** [`StreamingGrid::
//!   write_cell_metadata`] encodes `Warm` as `0` — indistinguishable from
//!   `Outer`, correct since neither has VRAM content — and α stays 0, so
//!   the M3 stipple pass needs no knowledge of the tier.
//!
//! With `GridConfig::warm = None` (the default shape for callers that don't
//! opt in) the machine, thresholds, executor behavior, and metadata bytes
//! are bit-for-bit identical to the pre-tier implementation. One documented
//! asymmetry when the tier IS on: a cell that sprinted in via the fast path
//! never had its copy staged by this tier, yet its eventual `Warm→Outer`
//! still fires `RamHooks::evict` — engines treat evict as a hint and no-op
//! on cache misses (the hook contract below).
//!
//! ## Drain-every-boundary contract
//!
//! `classify` never mutates a cell's committed `domain`/`alpha_target` — it
//! only queues [`Transition`]s. The caller MUST drain the queue via
//! [`StreamingGrid::take_transitions`] once per boundary, execute, and
//! report success via [`StreamingGrid::commit_transition`] (a declined
//! transition simply isn't committed; the next `classify` re-evaluates from
//! the unchanged committed state and re-queues it). Calling `classify` with
//! an undrained queue is a contract violation: a stale queued transition
//! could contradict what the newer classification would decide (e.g. a
//! queued `Inner→Margin` surviving a frame in which the observer moved back
//! inside), and the executor would apply it. `classify` debug-asserts the
//! queue is empty, and — belt and braces for release builds — drops any
//! stale queued transition for a cell before queueing that cell's new one,
//! so the queue never holds two transitions for the same coord.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{CellId, RegionClassConfig, RetiredPhase, SceneGpuStore};
use crate::spatial::{Aabb, SpatialCell};

/// §5.5 tunables. `pad_fraction` default is 0.10 (§5.5 Δpad); `hysteresis`
/// is δhyst, additional world units layered on top of the pad for the
/// demotion test only.
#[derive(Clone, Copy, Debug)]
pub struct GridConfig {
    pub cell_width: f32,
    /// World units beyond the inner union that count as `Margin`.
    pub margin_radius: f32,
    /// §5.5 Δpad fraction of `cell_width`; default 0.10.
    pub pad_fraction: f32,
    /// §5.5 δhyst, world units beyond the pad, demotion-only.
    pub hysteresis: f32,
    /// Opt-in system-RAM residency tier between `Outer` and `Margin`
    /// (`None` = disabled: the machine is exactly the legacy
    /// `Outer`/`Margin`/`Inner` ladder). See the module-doc tier section.
    pub warm: Option<WarmTierConfig>,
}

/// The opt-in RAM-tier's own §5.5-style tunables ([`GridConfig::warm`]).
/// Deliberately a separate pair from the GPU pad/hysteresis: the warm band
/// sits further out and typically wants wider jitter damping — its promote
/// boundary must lie strictly beyond the GPU one
/// (`radius > margin_radius`, validated at [`StreamingGrid::new`]).
#[derive(Clone, Copy, Debug)]
pub struct WarmTierConfig {
    /// World units beyond the base bounds that count as `Warm`.
    pub radius: f32,
    /// Δpad fraction of `cell_width` for the `warm_promote` boundary;
    /// same construction as [`GridConfig::pad_fraction`].
    pub pad_fraction: f32,
    /// δhyst, world units beyond the warm pad, demotion-only.
    pub hysteresis: f32,
}

/// Dense grid coordinate: cell `(x, z)` spans world
/// `[x * cell_width, (x+1) * cell_width) × [z * cell_width, (z+1) * cell_width)`
/// (Y unbounded — see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: i32,
    pub z: i32,
}

/// Residency domain, ordered `Outer < Warm < Margin < Inner` (least to most
/// resident; `Warm` is the opt-in system-RAM tier — see the module-doc
/// tier section). The enum's declared variant order is documentation-only —
/// [`domain_rank`] is the authoritative ordering used for the α-target
/// promotion/demotion distinction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Domain {
    Inner,
    Margin,
    Warm,
    Outer,
}

fn domain_rank(d: Domain) -> u8 {
    match d {
        Domain::Outer => 0,
        Domain::Warm => 1,
        Domain::Margin => 2,
        Domain::Inner => 3,
    }
}

impl Domain {
    /// Whether a cell in this domain is GPU-resident (`Margin`/`Inner`):
    /// registered with `SceneGpuStore` and carrying VRAM content. `Warm`
    /// cells hold system-RAM copies only, and `Outer` cells hold nothing.
    pub fn is_gpu_resident(self) -> bool {
        matches!(self, Domain::Margin | Domain::Inner)
    }
}

/// A single queued domain change for one cell. `from` is the committed
/// domain at queue time; under the drain-every-boundary contract (module
/// docs) it is always the cell's current domain when the executor sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    pub coord: CellCoord,
    pub from: Domain,
    pub to: Domain,
}

/// §5.3 VRAM budget inputs, checked once at construction (α-audit
/// bounded-extent input: `max_materialized_cells` bounds the HLOD term
/// regardless of how large the world actually is). The `ram_*` fields bound
/// the opt-in system-RAM tier the same way; they are validated even when
/// [`GridConfig::warm`] is `None`, mirroring how the VRAM fields are
/// validated unconditionally.
#[derive(Clone, Copy, Debug)]
pub struct StreamingBudget {
    pub vram_hlod_budget: u64,
    pub vram_geometry_budget: u64,
    /// Bounded worst-case count of simultaneously materialized cells.
    pub max_materialized_cells: u32,
    pub proxy_mesh_bytes: u64,
    pub mean_cell_geometry_bytes: u64,
    /// System-RAM ceiling for the warm tier's cached cell data.
    /// Accounting-only: SceneDB tracks the bookkeeping (see
    /// [`StreamingGrid::ram_cached_count`]) while an engine-side asset cache
    /// owns the actual memory (module-doc tier section).
    pub ram_budget: u64,
    /// Bounded worst-case count of simultaneously warm (RAM-cached) cells;
    /// enforced at runtime by `classify` as a hold-and-retry cap.
    pub max_ram_cached_cells: u32,
}

/// §5.3 budget-validation failures, surfaced at [`StreamingGrid::new`].
#[derive(Debug, PartialEq)]
pub enum BudgetError {
    HlodOverBudget,
    GeometryOverBudget,
    /// `max_ram_cached_cells × mean_cell_geometry_bytes` exceeds
    /// `ram_budget`.
    RamOverBudget,
    /// `max_ram_cached_cells` exceeds `max_materialized_cells`.
    RamCapExceedsMaterialized,
    /// The warm tier's promote boundary would sit at or inside the GPU
    /// margin boundary (`warm.radius ≤ margin_radius`) — the ladder must be
    /// strictly ordered `Outer < Warm < Margin` for the band machine to
    /// terminate sensibly.
    WarmRadiusNotBeyondMargin,
}

#[derive(Debug)]
struct GridCellState {
    domain: Domain,
    /// If `Some`, the cell is pinned to this domain — concentric
    /// classification is bypassed and the grid forces residency at
    /// this level regardless of observer distance. `None` (default)
    /// means the cell follows the standard concentric rules.
    pinned_domain: Option<Domain>,
    dense_id: u32,
    alpha: f32,
    alpha_target: f32,
    /// The store-side region assignment while GPU-resident (`Margin`/
    /// `Inner`); `None` while `Outer` or `Warm`. Set by
    /// [`execute_transitions`] at a successful promotion into GPU residency,
    /// cleared at a demotion out of it — the grid never allocates or frees
    /// this itself, only records what the executor reports (module docs:
    /// `execute_transitions` is the one place this module touches
    /// `SceneGpuStore`).
    gpu_id: Option<CellId>,
}

/// Pure-logic concentric streaming grid — see module docs for the full
/// band-machine/cross-fade contract.
#[derive(Debug)]
pub struct StreamingGrid {
    cfg: GridConfig,
    cells: HashMap<CellCoord, GridCellState>,
    next_dense_id: u32,
    transitions: Vec<Transition>,
    /// Runtime warm-tier cap, copied from `StreamingBudget::
    /// max_ram_cached_cells` at construction: `classify` holds `Outer→Warm`
    /// promotions while this many cells are warm (hold-and-retry, §8-style).
    max_ram_cached_cells: u32,
    /// Current count of `Warm`-domain cells — the live side of the
    /// `max_ram_cached_cells` accounting. Maintained only by
    /// [`StreamingGrid::commit_transition`] (the single writer of committed
    /// domain state).
    warm_count: u32,
    /// Test 13 instrumentation (see `upload_count` below). `write_cell_metadata`
    /// takes `&self` (it only reads `cells`/`next_dense_id`, no mutation), so
    /// this needs interior mutability — `AtomicU64` with `Relaxed` ordering,
    /// unlike the plain `u64` counters on the other asset stores (whose
    /// write methods already take `&mut self`). `Relaxed` is sufficient
    /// because this is monotonic instrumentation only: the count itself
    /// carries no data other stores/threads synchronize on, so there is no
    /// ordering relationship to preserve with any other memory operation.
    upload_count: AtomicU64,
}

impl StreamingGrid {
    /// Validates the §5.3 budget once, up front: `max_materialized_cells ×
    /// proxy_mesh_bytes ≤ vram_hlod_budget` (HLOD/proxy term) and
    /// `(Σ inner_classes.max_resident_cells) × mean_cell_geometry_bytes ≤
    /// vram_geometry_budget` (resident-geometry term). The warm-tier
    /// equivalents (`max_ram_cached_cells` against `max_materialized_cells`
    /// and `ram_budget`) are validated unconditionally, as is the tier
    /// ladder ordering when [`GridConfig::warm`] is enabled.
    pub fn new(
        cfg: GridConfig,
        budget: StreamingBudget,
        inner_classes: &[RegionClassConfig],
    ) -> Result<Self, BudgetError> {
        let hlod_used = budget.max_materialized_cells as u64 * budget.proxy_mesh_bytes;
        if hlod_used > budget.vram_hlod_budget {
            return Err(BudgetError::HlodOverBudget);
        }
        let resident_cells: u64 = inner_classes
            .iter()
            .map(|c| c.max_resident_cells as u64)
            .sum();
        let geometry_used = resident_cells * budget.mean_cell_geometry_bytes;
        if geometry_used > budget.vram_geometry_budget {
            return Err(BudgetError::GeometryOverBudget);
        }
        if budget.max_ram_cached_cells > budget.max_materialized_cells {
            return Err(BudgetError::RamCapExceedsMaterialized);
        }
        let ram_used = budget.max_ram_cached_cells as u64 * budget.mean_cell_geometry_bytes;
        if ram_used > budget.ram_budget {
            return Err(BudgetError::RamOverBudget);
        }
        if let Some(warm) = &cfg.warm {
            if warm.radius <= cfg.margin_radius {
                return Err(BudgetError::WarmRadiusNotBeyondMargin);
            }
        }
        Ok(Self {
            cfg,
            cells: HashMap::new(),
            next_dense_id: 0,
            transitions: Vec::new(),
            max_ram_cached_cells: budget.max_ram_cached_cells,
            warm_count: 0,
            upload_count: AtomicU64::new(0),
        })
    }

    /// Track a content-bearing cell (assigns a dense id, starts `Outer`).
    /// Idempotent: re-materializing an already-tracked coord returns its
    /// existing dense id and leaves its state untouched.
    pub fn materialize(&mut self, coord: CellCoord) -> u32 {
        if let Some(state) = self.cells.get(&coord) {
            return state.dense_id;
        }
        let id = self.next_dense_id;
        self.next_dense_id += 1;
        self.cells.insert(
            coord,
            GridCellState {
                domain: Domain::Outer,
                pinned_domain: None,
                dense_id: id,
                alpha: 0.0,
                alpha_target: 0.0,
                gpu_id: None,
            },
        );
        id
    }

    pub fn domain(&self, coord: CellCoord) -> Option<Domain> {
        self.cells.get(&coord).map(|s| s.domain)
    }

    pub fn alpha(&self, coord: CellCoord) -> Option<f32> {
        self.cells.get(&coord).map(|s| s.alpha)
    }

    pub fn dense_id(&self, coord: CellCoord) -> Option<u32> {
        self.cells.get(&coord).map(|s| s.dense_id)
    }

    /// Pin a cell to a specific domain, bypassing concentric classification.
    /// The cell will be forced to `domain` regardless of observer distance.
    /// Returns `false` if the coord is not tracked (call `materialize` first).
    pub fn pin(&mut self, coord: CellCoord, domain: Domain) -> bool {
        if let Some(state) = self.cells.get_mut(&coord) {
            state.pinned_domain = Some(domain);
            true
        } else {
            false
        }
    }

    /// Remove a pin, returning the cell to standard concentric classification.
    /// A no-op for an untracked coord or one that was never pinned.
    pub fn unpin(&mut self, coord: CellCoord) {
        if let Some(state) = self.cells.get_mut(&coord) {
            state.pinned_domain = None;
        }
    }

    /// Returns the pinned domain override, if any. `None` means the cell
    /// follows standard concentric rules.
    pub fn pinned_domain(&self, coord: CellCoord) -> Option<Option<Domain>> {
        self.cells.get(&coord).map(|s| s.pinned_domain)
    }

    /// The store-side region assignment for a resident cell — `None` for an
    /// `Outer` cell or an untracked coord. Set/cleared by
    /// [`execute_transitions`] only.
    pub fn gpu_id(&self, coord: CellCoord) -> Option<CellId> {
        self.cells.get(&coord).and_then(|s| s.gpu_id)
    }

    /// Executor-only: record (`Some`) or clear (`None`) a cell's store-side
    /// region assignment. A no-op for an untracked coord.
    pub fn set_gpu_id(&mut self, coord: CellCoord, id: Option<CellId>) {
        if let Some(state) = self.cells.get_mut(&coord) {
            state.gpu_id = id;
        }
    }

    /// Whether a tracked coord currently holds a warm (system-RAM-cached,
    /// not GPU-resident) copy — i.e. its domain is [`Domain::Warm`].
    /// `None` for an untracked coord. Accounting only: whether real bytes
    /// are cached is the engine-side cache's truth (module-doc tier
    /// section); the grid tracks the residency *decision*.
    pub fn ram_cached(&self, coord: CellCoord) -> Option<bool> {
        self.cells.get(&coord).map(|s| s.domain == Domain::Warm)
    }

    /// Current count of warm (RAM-cached) cells — the runtime side of
    /// `StreamingBudget::max_ram_cached_cells`. Maintained by
    /// [`Self::commit_transition`] only.
    pub fn ram_cached_count(&self) -> u32 {
        self.warm_count
    }

    /// §5 classification via the §5.5 hysteresis band machine (module docs).
    /// Queues at most one single-step [`Transition`] per cell; applies NO
    /// state change to `domain`/`alpha_target` — that happens only in
    /// [`Self::commit_transition`].
    ///
    /// CONTRACT: the caller must drain [`Self::take_transitions`] every
    /// boundary, before the next `classify` — an undrained queue can hold a
    /// transition the newer observer positions would no longer justify.
    /// Debug builds assert this; release builds additionally self-heal by
    /// evicting any stale queued transition for a cell before queueing that
    /// cell's new one.
    pub fn classify(&mut self, observer_aabbs: &[Aabb]) {
        debug_assert!(
            self.transitions.is_empty(),
            "classify() called with undrained transitions — drain via take_transitions() every boundary"
        );
        let pad = self.cfg.pad_fraction * self.cfg.cell_width;
        let hyst = self.cfg.hysteresis;
        let mr = self.cfg.margin_radius;
        let cell_width = self.cfg.cell_width;

        for (&coord, state) in self.cells.iter() {
            // Pinned cells bypass concentric classification entirely:
            // queue a forced transition if they don't match their pin target.
            if let Some(pin) = state.pinned_domain {
                if state.domain != pin {
                    let from = state.domain;
                    // Belt and braces: evict stale queued transitions for this coord.
                    self.transitions.retain(|t| t.coord != coord);
                    self.transitions.push(Transition { coord, from, to: pin });
                }
                continue;
            }
            let base = base_bounds(coord, cell_width);
            let to = match state.domain {
                Domain::Outer => {
                    // margin_promote first: an observer sprinting in keeps
                    // the legacy fast path and skips RAM staging entirely.
                    if any_intersect(&grow(base, mr + pad), observer_aabbs) {
                        Some(Domain::Margin)
                    } else if let Some(warm) = self.cfg.warm {
                        // warm_promote: base + (warm.radius + warm pad).
                        // The runtime cap holds the promotion (stay Outer,
                        // retry on a later boundary) — §8-style graceful
                        // degradation for the RAM tier.
                        let wpad = warm.pad_fraction * cell_width;
                        if self.warm_count < self.max_ram_cached_cells
                            && any_intersect(&grow(base, warm.radius + wpad), observer_aabbs)
                        {
                            Some(Domain::Warm)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                Domain::Warm => match self.cfg.warm {
                    Some(warm) => {
                        let wpad = warm.pad_fraction * cell_width;
                        // margin_promote: GPU promotion straight from the
                        // warm copy.
                        if any_intersect(&grow(base, mr + pad), observer_aabbs) {
                            Some(Domain::Margin)
                        // warm_demote: base + (warm.radius + wpad + whyst)
                        } else if !any_intersect(
                            &grow(base, warm.radius + wpad + warm.hysteresis),
                            observer_aabbs,
                        ) {
                            Some(Domain::Outer)
                        } else {
                            None // inside the Warm band: hold
                        }
                    }
                    // Tier withdrawn while cells are warm (pin path): cool
                    // them immediately rather than panicking on the missing
                    // config. Unreachable via concentric flow — Warm is only
                    // queued when the tier is on.
                    None => Some(Domain::Outer),
                },
                Domain::Margin => {
                    // inner_promote: base + pad
                    if any_intersect(&grow(base, pad), observer_aabbs) {
                        Some(Domain::Inner)
                    // margin_demote: base + (margin_radius + pad + hyst) —
                    // the UNCHANGED legacy GPU-teardown boundary; with the
                    // tier on it drops to Warm (RAM retained) instead of
                    // all the way to Outer.
                    } else if !any_intersect(&grow(base, mr + pad + hyst), observer_aabbs) {
                        Some(if self.cfg.warm.is_some() { Domain::Warm } else { Domain::Outer })
                    } else {
                        None // inside the Margin band: hold
                    }
                }
                Domain::Inner => {
                    // inner_demote: base + (pad + hyst)
                    if !any_intersect(&grow(base, pad + hyst), observer_aabbs) {
                        Some(Domain::Margin)
                    } else {
                        None // inside the Inner band (or still inside): hold
                    }
                }
            };
            if let Some(to) = to {
                // Belt and braces for release builds (the debug_assert above
                // is the contract): never leave two queued transitions for
                // the same coord — the newer classification wins.
                self.transitions.retain(|t| t.coord != coord);
                self.transitions.push(Transition { coord, from: state.domain, to });
            }
        }
    }

    /// Drain queued transitions (caller executes them at the boundary).
    pub fn take_transitions(&mut self) -> Vec<Transition> {
        std::mem::take(&mut self.transitions)
    }

    /// Confirm an executed transition (caller reports success/decline by
    /// simply not calling this for a declined one). Sets the cell's domain
    /// to `t.to` and its α target: 1.0 only when crossing INTO GPU
    /// residency (`to` ∈ {`Margin`, `Inner`} and more resident than `from`)
    /// — every other transition, including promotions into the RAM tier,
    /// targets 0.0 (module-doc tier section: `Warm` has no on-screen
    /// content). Also maintains [`Self::ram_cached_count`].
    pub fn commit_transition(&mut self, t: Transition) {
        if let Some(state) = self.cells.get_mut(&t.coord) {
            if state.domain == Domain::Warm {
                self.warm_count = self
                    .warm_count
                    .checked_sub(1)
                    .expect("warm_count underflow: committed transition out of Warm with count 0");
            }
            state.domain = t.to;
            if state.domain == Domain::Warm {
                self.warm_count += 1;
            }
            state.alpha_target =
                if t.to.is_gpu_resident() && domain_rank(t.to) > domain_rank(t.from) { 1.0 } else { 0.0 };
        }
    }

    /// §5.2: advance cross-fade by observer world-distance travelled,
    /// linearly, clamped to `[0, 1]`.
    pub fn advance_crossfade(&mut self, distance: f32, fade_distance: f32) {
        let step = distance / fade_distance;
        for state in self.cells.values_mut() {
            let target = state.alpha_target;
            if target > state.alpha {
                state.alpha = (state.alpha + step).min(target);
            } else if target < state.alpha {
                state.alpha = (state.alpha - step).max(target);
            }
            state.alpha = state.alpha.clamp(0.0, 1.0);
        }
    }

    /// Packs `(f32 alpha, u32 domain)` for every materialized cell into
    /// `buf` at byte offset `dense_id * 8` — the M3 stipple-pass contract.
    /// Domain encoding: `Outer` = 0, `Margin` = 1, `Inner` = 2 — and the
    /// RAM-tier's `Warm` ALSO encodes as 0: it has no VRAM content, so
    /// shaders see exactly what an `Outer` cell looks like (module-doc tier
    /// section; the pass needs no knowledge of the tier).
    ///
    /// Simple full rewrite of every materialized entry's 8 bytes on every
    /// call (bounded by `next_dense_id ≤ max_cells_metadata`, §8);
    /// delta-tracking (skipping unchanged entries) is a recorded future
    /// optimization, not built here.
    pub fn write_cell_metadata(&self, queue: &wgpu::Queue, buf: &wgpu::Buffer) {
        let mut data = vec![0u8; self.next_dense_id as usize * 8];
        for state in self.cells.values() {
            let domain_code: u32 = match state.domain {
                Domain::Outer | Domain::Warm => 0,
                Domain::Margin => 1,
                Domain::Inner => 2,
            };
            let offset = state.dense_id as usize * 8;
            data[offset..offset + 4].copy_from_slice(&state.alpha.to_le_bytes());
            data[offset + 4..offset + 8].copy_from_slice(&domain_code.to_le_bytes());
        }
        assert!(
            data.len() as u64 <= buf.size(),
            "materialized cell count {} needs {} bytes, exceeding the cell-metadata buffer's {} bytes (max_cells_metadata too small)",
            self.next_dense_id,
            data.len(),
            buf.size()
        );
        queue.write_buffer(buf, 0, &data);
        self.upload_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Test 13 instrumentation: the teardown gate asserts these do not move
    /// across the renderer drop/rebind window. `Relaxed` load — see the
    /// `upload_count` field doc for why no stronger ordering is needed.
    #[doc(hidden)]
    pub fn upload_count(&self) -> u64 {
        self.upload_count.load(Ordering::Relaxed)
    }
}

/// Outcome tally for one [`execute_transitions`]/
/// [`execute_transitions_with_ram`] call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TransitionStats {
    /// GPU registrations that succeeded (`register_cell` returned `Ok`) —
    /// both legacy `Outer→Margin` promotions and RAM-tier `Warm→Margin`
    /// promotions (which register straight from the warm copy, no reload).
    pub promoted: u32,
    /// GPU teardowns executed (`unregister_cell`): `Margin→Outer` (legacy)
    /// and, with the tier on, `Margin→Warm` (VRAM freed, RAM retained).
    pub demoted: u32,
    /// GPU registrations that `register_cell` declined
    /// (`Err(RegionError)`, §8 graceful degradation) — the cell stays in its
    /// current domain and is re-classified on the next `classify` call.
    pub declined: u32,
    /// Drained transitions dropped because the grid's committed domain no
    /// longer matched `t.from` at execution time (T3 reviewer: release-build
    /// stale-queue hole). A dropped transition is safe by construction —
    /// the next `classify()` re-derives intent from committed state.
    pub dropped_stale: u32,
    /// RAM-tier: `Outer→Warm` transitions whose [`RamHooks::load`] reported
    /// success. The engine-side cache now holds the cell's source bytes.
    pub warmed: u32,
    /// RAM-tier: `Warm→Outer` transitions whose [`RamHooks::evict`] ran.
    pub cooled: u32,
    /// RAM-tier: `Outer→Warm` transitions refused — no hooks were supplied
    /// to the executor, `load` returned `false`, or the runtime warm cap
    /// held the promotion at `classify` time (the last case never reaches
    /// the executor). The cell stays in its current domain; the next
    /// `classify` re-queues when conditions allow.
    pub ram_declined: u32,
}

/// Caller-supplied system-RAM residency hooks for
/// [`execute_transitions_with_ram`] — the RAM-tier analog of the existing
/// `class_of` callback shape. SceneDB owns the *decisions* and the
/// *bookkeeping* ([`StreamingGrid::ram_cached_count`]); these hooks own the
/// bytes. Disk/network asset delivery stays the engine's job exactly as
/// before (module-doc tier section).
pub struct RamHooks<'a> {
    /// Populate the RAM-resident source data for `coord` in the engine's
    /// cache. Return `false` to refuse (cache miss, pressure, …): the
    /// transition is not committed and the cell stays in its current
    /// domain, re-queued on a later boundary.
    pub load: &'a dyn Fn(CellCoord) -> bool,
    /// Drop the RAM-resident copy for `coord`. Infallible hint semantics:
    /// fired on every committed `Warm→Outer` demotion, which — via the
    /// sprint-in fast path — can occur for a coord this tier never loaded
    /// (module-doc tier section). Engines no-op the miss.
    pub evict: &'a dyn Fn(CellCoord),
}

/// Boundary transition executor (M2b-β T5): drains
/// [`StreamingGrid::take_transitions`] and applies each against `store` and
/// `cells`, reporting outcomes via [`TransitionStats`]. Legacy entry point:
/// identical to [`execute_transitions_with_ram`] with no [`RamHooks`] — an
/// `Outer→Warm` transition can then only be *declined* (`stats.
/// ram_declined`), so callers that never enable [`GridConfig::warm`] see
/// bit-for-bit legacy behavior through this function.
///
/// - **Into GPU residency** (`from` ∈ {`Outer`, `Warm`} → `to` ∈
///   {`Margin`, `Inner`}, including pin jumps): `store.register_cell(cell.
///   storage(), class_of(coord))`. From `Warm` this registers straight from
///   the already-RAM-resident copy — **no loader round-trip**, the point of
///   the tier. `Ok(id)` → commit the transition and record `id` via
///   [`StreamingGrid::set_gpu_id`] (`stats.promoted += 1`). `Err(RegionError)`
///   → DECLINE: the transition is not committed (the cell stays in its
///   current domain, grid state unchanged), `stats.declined += 1`, and a
///   `tracing::warn!` records the exhaustion (§8 graceful degradation).
/// - **Out of GPU residency into `Warm`**: `store.unregister_cell(id, …)`
///   using the id recorded at promotion, cleared via `set_gpu_id(coord,
///   None)` (`stats.demoted += 1`). The RAM copy is deliberately NOT
///   evicted — that retention is the feature.
/// - **`Margin/Inner → Outer`**: same unregister, full teardown (legacy
///   shape, still produced by pins and by tier-off grids).
/// - **`Outer → Warm`** (`stats.warmed += 1` on success): `(hooks.load)(
///   coord)`; `false`/absent hooks → DECLINE (`stats.ram_declined += 1`,
///   warn), cell unchanged.
/// - **`Warm → Outer`**: `(hooks.evict)(coord)` (hint — see [`RamHooks::
///   evict`]), commit (`stats.cooled += 1`).
/// - **`Margin↔Inner`**: commit-only — a domain-flag change with no store
///   interaction.
///
/// `_w: &RetiredPhase` is a witness, not a value read here: it proves the
/// caller has run this frame's `retire` boundary stage (so eviction serials
/// and pending-retire drains are already resolved) before promoting or
/// evicting any cell this boundary.
///
/// Before applying ANY drained transition, its coord's *current* committed
/// domain is checked against `t.from`; a mismatch means the queue held a
/// transition from before some other write invalidated it (T3 reviewer,
/// release-build stale-queue hole) — it is silently dropped and counted in
/// `stats.dropped_stale`. A dropped transition is safe by construction — the
/// next `classify()` re-derives intent from committed state.
///
/// **Trust contract on `class_of`:** its return value is used as-is to index
/// `SceneGpuStore`'s internal per-class region pools (`register_cell`'s
/// `row_pools[class]`/`slot_pools[class]`); this function does not validate
/// it against the store's configured class count. A `class_of` that returns
/// an index outside the range the `SceneGpuStore` was constructed with (its
/// `SceneGpuConfig::classes` length) panics inside `register_cell` on the
/// next promotion into GPU residency, not here — the caller owns keeping
/// `class_of`'s range in sync with the store's class configuration.
pub fn execute_transitions(
    grid: &mut StreamingGrid,
    store: &mut SceneGpuStore,
    cells: &mut HashMap<CellCoord, SpatialCell>,
    class_of: &dyn Fn(CellCoord) -> usize,
    eviction_serial: u64,
    _w: &RetiredPhase,
) -> TransitionStats {
    execute_transitions_with_ram(grid, store, cells, class_of, None, eviction_serial, _w)
}

/// [`execute_transitions`] with the opt-in RAM tier's [`RamHooks`] attached
/// (`ram: None` ≡ [`execute_transitions`]). See that function's doc for the
/// per-transition contract; the hooks only ever fire on `Outer→Warm`
/// (`load`) and `Warm→Outer` (`evict`) — GPU registrations/unregistrations
/// are unchanged store interactions.
pub fn execute_transitions_with_ram(
    grid: &mut StreamingGrid,
    store: &mut SceneGpuStore,
    cells: &mut HashMap<CellCoord, SpatialCell>,
    class_of: &dyn Fn(CellCoord) -> usize,
    ram: Option<&RamHooks<'_>>,
    eviction_serial: u64,
    _w: &RetiredPhase,
) -> TransitionStats {
    let mut stats = TransitionStats::default();
    for t in grid.take_transitions() {
        if grid.domain(t.coord) != Some(t.from) {
            // A dropped transition is safe by construction — the next
            // classify() re-derives intent from committed state.
            stats.dropped_stale += 1;
            continue;
        }
        let cell = cells
            .get_mut(&t.coord)
            .expect("execute_transitions: materialized coord must have a tracked SpatialCell");
        match (t.from, t.to) {
            // ── Into GPU residency (incl. pin jumps like Outer→Inner):
            // register from the cell's CPU-side storage. From Warm this
            // consumes the already-resident RAM copy — no loader round-trip.
            (from, to) if !from.is_gpu_resident() && to.is_gpu_resident() => {
                let class = class_of(t.coord);
                match store.register_cell(cell.storage(), class) {
                    Ok(id) => {
                        grid.set_gpu_id(t.coord, Some(id));
                        grid.commit_transition(t);
                        stats.promoted += 1;
                    }
                    Err(err) => {
                        stats.declined += 1;
                        tracing::warn!(
                            coord = ?t.coord,
                            error = ?err,
                            "region exhausted — declining promotion into GPU residency; cell stays where it is"
                        );
                    }
                }
            }
            // ── Out of GPU residency into the RAM tier: free VRAM, keep the
            // RAM copy (that retention is the feature). No evict hook here.
            (from, Domain::Warm) if from.is_gpu_resident() => {
                let id = grid
                    .gpu_id(t.coord)
                    .expect("GPU-resident cell must carry a gpu_id assigned at its promotion");
                store.unregister_cell(id, cell.storage_mut(), eviction_serial);
                grid.set_gpu_id(t.coord, None);
                grid.commit_transition(t);
                stats.demoted += 1;
            }
            // ── Full GPU teardown (tier-off Margin→Outer, or pin-to-Outer):
            // legacy shape — unregister, no RAM-tier interaction.
            (from, Domain::Outer) if from.is_gpu_resident() => {
                let id = grid
                    .gpu_id(t.coord)
                    .expect("GPU-resident cell must carry a gpu_id assigned at its promotion");
                store.unregister_cell(id, cell.storage_mut(), eviction_serial);
                grid.set_gpu_id(t.coord, None);
                grid.commit_transition(t);
                stats.demoted += 1;
            }
            // ── RAM population: stage the source bytes via the engine's
            // cache. Refusal (or no hooks at all) leaves the cell put.
            (Domain::Outer, Domain::Warm) => {
                match ram {
                    Some(hooks) if (hooks.load)(t.coord) => {
                        grid.commit_transition(t);
                        stats.warmed += 1;
                    }
                    Some(_) => {
                        stats.ram_declined += 1;
                        tracing::debug!(
                            coord = ?t.coord,
                            "RAM-tier load refused — cell stays Outer; will retry on a later boundary"
                        );
                    }
                    None => {
                        stats.ram_declined += 1;
                        tracing::warn!(
                            coord = ?t.coord,
                            "RAM-tier transition queued but no RamHooks supplied — \
                             pass them to execute_transitions_with_ram; cell stays Outer"
                        );
                    }
                }
            }
            // ── RAM eviction: drop the staged copy (hint semantics — may be
            // a cache miss after a sprint-in fast path; engines no-op it).
            (Domain::Warm, Domain::Outer) => {
                if let Some(hooks) = ram {
                    (hooks.evict)(t.coord);
                }
                grid.commit_transition(t);
                stats.cooled += 1;
            }
            // ── Margin↔Inner: domain-flag change only, no store
            // interaction (module docs).
            _ => {
                grid.commit_transition(t);
            }
        }
    }
    stats
}

/// A cell's world AABB from `coord × cell_width`. XZ-planar (β
/// simplification): Y is unbounded so observer altitude never affects
/// classification.
fn base_bounds(coord: CellCoord, cell_width: f32) -> Aabb {
    let x0 = coord.x as f32 * cell_width;
    let z0 = coord.z as f32 * cell_width;
    Aabb {
        min: [x0, f32::NEG_INFINITY, z0],
        max: [x0 + cell_width, f32::INFINITY, z0 + cell_width],
    }
}

/// Grow an AABB by `r` in every axis (Y stays effectively unbounded: ±inf ±
/// r is still ±inf).
fn grow(a: Aabb, r: f32) -> Aabb {
    Aabb {
        min: [a.min[0] - r, a.min[1] - r, a.min[2] - r],
        max: [a.max[0] + r, a.max[1] + r, a.max[2] + r],
    }
}

/// Closed-interval AABB intersection (crate-wide §8.2 discipline: touching
/// faces count as a hit).
fn aabb_intersect(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|i| a.min[i] <= b.max[i] && a.max[i] >= b.min[i])
}

fn any_intersect(region: &Aabb, observers: &[Aabb]) -> bool {
    observers.iter().any(|o| aabb_intersect(region, o))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GridConfig {
        GridConfig { cell_width: 100.0, margin_radius: 150.0, pad_fraction: 0.10, hysteresis: 20.0, warm: None }
    }

    fn budget() -> StreamingBudget {
        StreamingBudget {
            vram_hlod_budget: u64::MAX,
            vram_geometry_budget: u64::MAX,
            max_materialized_cells: 1024,
            proxy_mesh_bytes: 1024,
            mean_cell_geometry_bytes: 1 << 20,
            ram_budget: u64::MAX,
            max_ram_cached_cells: 1024,
        }
    }

    fn observer_at(x: f32) -> Aabb {
        Aabb { min: [x - 10.0, -10.0, -10.0], max: [x + 10.0, 10.0, 10.0] }
    }

    // ── Test 11 gate: threshold derivations (closed intervals) ────────────
    //
    // cfg: cell_width 100, margin_radius 150, pad = 0.10 × 100 = 10,
    // hyst = 20. Cell (5,0): base x ∈ [500, 600]. Observer half-width 10.
    //
    //   margin_promote = base ± (150+10)    = [340, 760] on x
    //     → intersects when center + 10 ≥ 340  ⟺  center ≥ 330
    //     (the UNPADDED margin edge would be [350, 750] ⟺ center ≥ 340:
    //      promotion at center 331 < 340 is possible ONLY because Δpad
    //      advanced the boundary — §5.5 PromotionBoundary = bounds + Δpad)
    //   margin_demote  = base ± (150+10+20) = [320, 780] on x
    //     → holds while center + 10 ≥ 320  ⟺  center ≥ 310;
    //       demotes when center < 310
    //   inner_promote  = base ± 10          = [490, 610] on x
    //     → needs center ≥ 480; never touched by these positions
    //
    //   ⇒ Margin band (cell held Margin, zero transitions either way):
    //     center ∈ [310, 330). Jitter range [312, 328] sits inside it.

    #[test]
    fn test11_pad_advances_promotion_band_holds_and_hysteresis_delays_demotion() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let far = CellCoord { x: 5, z: 0 }; // base x ∈ [500, 600]
        g.materialize(far);

        // (a) Just below the padded promote threshold (center 329 < 330):
        // no transition.
        g.classify(&[observer_at(329.0)]);
        assert!(g.take_transitions().is_empty(), "329 < padded threshold 330: no promotion");
        assert_eq!(g.domain(far), Some(Domain::Outer));

        // (a) Just past it (center 331 ≥ 330, yet well short of the UNPADDED
        // threshold 340): promotes — proof that Δpad advances the boundary.
        g.classify(&[observer_at(331.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1, "exactly one transition at the padded boundary");
        assert_eq!(ts[0], Transition { coord: far, from: Domain::Outer, to: Domain::Margin });
        g.commit_transition(ts[0]);

        // (b) Jitter inside the band [312, 328] — past the demote-hold
        // threshold (310), short of the promote threshold (330): the §5.5
        // band must hold with ZERO transitions in either direction.
        for i in 0..200 {
            let center = 320.0 + ((i % 9) as f32 - 4.0) * 2.0; // ∈ [312, 328]
            g.classify(&[observer_at(center)]);
            assert!(
                g.take_transitions().is_empty(),
                "band frame {i} (center {center}) caused a transition"
            );
        }
        assert_eq!(g.domain(far), Some(Domain::Margin), "held Margin through the band");

        // (c) Retreat past the demotion boundary (center 305 < 310):
        // exactly one demotion — hysteresis delayed it 20 units beyond
        // where the padded promote boundary sits.
        g.classify(&[observer_at(305.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0], Transition { coord: far, from: Domain::Margin, to: Domain::Outer });
    }

    #[test]
    fn test11_decisive_crossing_promotes_exactly_once_and_demotion_lags_by_hysteresis() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let far = CellCoord { x: 5, z: 0 }; // cell spanning x ∈ [500, 600]
        g.materialize(far);
        g.classify(&[observer_at(0.0)]);
        assert!(g.take_transitions().is_empty());
        assert_eq!(g.domain(far), Some(Domain::Outer));
        // Decisive move deep into margin range of the far cell:
        g.classify(&[observer_at(480.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1, "exactly one transition — single step, no skip past Margin");
        assert_eq!(ts[0], Transition { coord: far, from: Domain::Outer, to: Domain::Margin });
        g.commit_transition(ts[0]);
        // Retreat to just inside the demotion boundary → NO demotion (hysteresis):
        g.classify(&[observer_at(480.0 - cfg().hysteresis + 1.0)]);
        assert!(g.take_transitions().is_empty(), "inside hysteresis band: no demotion");
        // Retreat past it → demotion:
        g.classify(&[observer_at(300.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].to, Domain::Outer);
    }

    #[test]
    fn cascade_promotes_one_step_per_classify_as_observer_converges() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 }; // base x ∈ [0, 100]
        g.materialize(c);
        // Far away: stays Outer.
        g.classify(&[observer_at(500.0)]);
        assert!(g.take_transitions().is_empty());
        // Call N — converge into margin_promote ([-160, 260] ⟺ center ≤ 270):
        // ONE step only, Outer→Margin — no skip past Margin even though the
        // observer will keep closing.
        g.classify(&[observer_at(200.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: c, from: Domain::Outer, to: Domain::Margin }]);
        g.commit_transition(ts[0]);
        // Call N+1 — converge into inner_promote ([-10, 110] ⟺ center ≤ 120):
        // second step, Margin→Inner.
        g.classify(&[observer_at(105.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: c, from: Domain::Margin, to: Domain::Inner }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.domain(c), Some(Domain::Inner));
    }

    #[test]
    #[should_panic(expected = "undrained")]
    fn classify_with_undrained_transitions_panics_in_debug() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        g.materialize(CellCoord { x: 0, z: 0 });
        g.classify(&[observer_at(50.0)]); // queues Outer→Margin
        g.classify(&[observer_at(50.0)]); // undrained — contract violation
    }

    #[test]
    fn budget_violation_fails_construction() {
        let mut b = budget();
        b.vram_hlod_budget = 10; // 1024 cells × 1 KiB proxies ≫ 10 bytes
        assert_eq!(StreamingGrid::new(cfg(), b, &[]).unwrap_err(), BudgetError::HlodOverBudget);
    }

    #[test]
    fn geometry_budget_violation_fails_construction() {
        let b = budget();
        let classes = [RegionClassConfig { capacity: 64, max_resident_cells: 10 }];
        // 10 resident cells × 1 MiB (mean_cell_geometry_bytes) ≫ this tiny cap.
        let mut b2 = b;
        b2.vram_geometry_budget = 1024;
        assert_eq!(
            StreamingGrid::new(cfg(), b2, &classes).unwrap_err(),
            BudgetError::GeometryOverBudget
        );
    }

    #[test]
    fn crossfade_advances_by_world_distance_and_clamps() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        g.classify(&[observer_at(50.0)]);
        for t in g.take_transitions() {
            g.commit_transition(t);
        }
        // Now heading resident (Outer→Margin promotion): α target 1.
        g.advance_crossfade(25.0, 100.0);
        assert!((g.alpha(c).unwrap() - 0.25).abs() < 1e-6);
        g.advance_crossfade(1000.0, 100.0);
        assert_eq!(g.alpha(c).unwrap(), 1.0, "clamped");
    }

    #[test]
    fn materialize_is_idempotent_and_starts_outer_with_zero_alpha() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 3, z: -2 };
        let id_a = g.materialize(c);
        let id_b = g.materialize(c); // re-materialize: same coord
        assert_eq!(id_a, id_b, "re-materializing returns the existing dense id");
        assert_eq!(g.domain(c), Some(Domain::Outer));
        assert_eq!(g.alpha(c), Some(0.0));
        // A second, distinct cell gets a distinct dense id.
        let other = g.materialize(CellCoord { x: 3, z: -1 });
        assert_ne!(id_a, other);
        // Untracked coord: everything reads back None.
        let untracked = CellCoord { x: 99, z: 99 };
        assert_eq!(g.domain(untracked), None);
        assert_eq!(g.alpha(untracked), None);
        assert_eq!(g.dense_id(untracked), None);
    }

    #[test]
    fn far_cell_with_no_observers_stays_outer() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 40, z: 40 };
        g.materialize(c);
        g.classify(&[]);
        assert!(g.take_transitions().is_empty());
        assert_eq!(g.domain(c), Some(Domain::Outer));
    }

    // ── Persistent (pinned) domain tests ──────────────────────────────────

    #[test]
    fn pin_starts_none_and_can_be_set_and_read() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        assert_eq!(g.pinned_domain(c), Some(None), "fresh cell has no pin");
        assert!(g.pin(c, Domain::Inner), "pin returns true for tracked coord");
        assert_eq!(g.pinned_domain(c), Some(Some(Domain::Inner)));
        g.unpin(c);
        assert_eq!(g.pinned_domain(c), Some(None), "unpin clears the override");
    }

    #[test]
    fn pin_on_untracked_coord_returns_false() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let untracked = CellCoord { x: 99, z: 99 };
        assert!(!g.pin(untracked, Domain::Inner), "untracked coord returns false");
        assert_eq!(g.pinned_domain(untracked), None, "untracked coord has no state");
    }

    #[test]
    fn pinned_cell_bypasses_concentric_classification() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        // Pin to Inner before any observer gets close.
        assert!(g.pin(c, Domain::Inner));
        // Observer is at 500 — well past the demotion threshold for an unpinned cell.
        g.classify(&[observer_at(500.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1, "pinned cell queues a forced transition to its pin target");
        assert_eq!(ts[0], Transition { coord: c, from: Domain::Outer, to: Domain::Inner });
        g.commit_transition(ts[0]);
        assert_eq!(g.domain(c), Some(Domain::Inner));
    }

    #[test]
    fn pinned_cell_does_not_demote_when_observer_retreats() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        // Promote to Inner via concentric rules first (observer close enough
        // for inner_promote: base x ∈ [0,100], pad=10 → inner_promote x ∈ [-10,110].
        // Observer at 50 (AABB x ∈ [40,60]) intersects → Margin→Inner).
        g.classify(&[observer_at(50.0)]);
        for t in g.take_transitions() { g.commit_transition(t); }
        // Observer at 0 (AABB x ∈ [-10,10]) intersects inner_promote [-10,110] → promoting.
        g.classify(&[observer_at(-5.0)]);
        for t in g.take_transitions() { g.commit_transition(t); }
        assert_eq!(g.domain(c), Some(Domain::Inner));
        // Now pin it to Inner.
        assert!(g.pin(c, Domain::Inner));
        // Move observer far away — would demote an unpinned cell.
        g.classify(&[observer_at(10000.0)]);
        let ts = g.take_transitions();
        assert!(ts.is_empty(), "pinned Inner cell does not demote when observer retreats");
        assert_eq!(g.domain(c), Some(Domain::Inner));
    }

    #[test]
    fn unpinned_cell_resumes_concentric_classification() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        // Pin to Inner.
        assert!(g.pin(c, Domain::Inner));
        g.classify(&[observer_at(10000.0)]);
        let t = g.take_transitions().into_iter().next().unwrap();
        g.commit_transition(t);
        assert_eq!(g.domain(c), Some(Domain::Inner));
        // Unpin — next classify with far observer demotes one step (Inner→Margin).
        g.unpin(c);
        g.classify(&[observer_at(10000.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1, "unpinned cell demotes one step via concentric rules");
        assert_eq!(ts[0].to, Domain::Margin);
        g.commit_transition(ts[0]);
        // Second classify with far observer demotes the rest of the way (Margin→Outer).
        g.classify(&[observer_at(10000.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts.len(), 1, "unpinned cell demotes second step");
        assert_eq!(ts[0].to, Domain::Outer);
    }

    #[test]
    fn pin_outer_keeps_cell_unloaded_while_observer_is_close() {
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        // Pin to Outer — should never promote even if observer is right on top.
        assert!(g.pin(c, Domain::Outer));
        g.classify(&[observer_at(0.0)]);
        let ts = g.take_transitions();
        assert!(ts.is_empty(), "pinned Outer cell does not promote despite nearby observer");
        assert_eq!(g.domain(c), Some(Domain::Outer));
    }

    // ── System-RAM residency tier (`Warm`, opt-in) tests ──────────────────
    //
    // warm_cfg(): cell_width 100, margin_radius 150, pad 10, hyst 20;
    // warm: radius 300, pad_fraction 0.10 (wpad = 10), hysteresis 40.
    // Cell (5,0): base x ∈ [500, 600]; observer half-width 10.
    //
    //   warm_promote   = base ± (300+10)     = [190, 910]  ⟺ center ≥ 180
    //   margin_promote = base ± (150+10)     = [340, 760]  ⟺ center ≥ 330
    //   margin_demote  = base ± (150+10+20)  = [320, 780]  ⟺ holds ≥ 310
    //   warm_demote    = base ± (300+10+40)  = [150, 950]  ⟺ holds ≥ 140

    fn warm_cfg() -> GridConfig {
        GridConfig {
            cell_width: 100.0,
            margin_radius: 150.0,
            pad_fraction: 0.10,
            hysteresis: 20.0,
            warm: Some(WarmTierConfig { radius: 300.0, pad_fraction: 0.10, hysteresis: 40.0 }),
        }
    }

    #[test]
    fn warm_tier_stages_holds_and_cools_on_its_own_bands() {
        let mut g = StreamingGrid::new(warm_cfg(), budget(), &[]).unwrap();
        let far = CellCoord { x: 5, z: 0 }; // base x ∈ [500, 600]
        g.materialize(far);

        // Just below warm_promote (center 179 < 180): nothing.
        g.classify(&[observer_at(179.0)]);
        assert!(g.take_transitions().is_empty(), "179 < padded warm threshold 180");

        // Cross it (center 200): exactly one step — Outer→Warm, NOT Margin.
        g.classify(&[observer_at(200.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Outer, to: Domain::Warm }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.domain(far), Some(Domain::Warm));
        assert_eq!(g.ram_cached(far), Some(true));
        assert_eq!(g.ram_cached_count(), 1);
        assert_eq!(g.alpha(far), Some(0.0), "RAM residency has no on-screen content");

        // Jitter inside the warm hold band [140, 330): zero transitions.
        for i in 0..100 {
            let center = 160.0 + ((i % 5) as f32 - 2.0) * 8.0; // ∈ [144, 176]
            g.classify(&[observer_at(center)]);
            assert!(
                g.take_transitions().is_empty(),
                "warm-band frame {i} (center {center}) caused a transition"
            );
        }

        // Sprint in past margin_promote (center 340 ≥ 330): Warm→Margin, one
        // step — GPU promotion straight off the already-resident warm copy.
        g.classify(&[observer_at(340.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Warm, to: Domain::Margin }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.domain(far), Some(Domain::Margin));
        assert_eq!(g.gpu_id(far), None, "executor records ids; pure grid leaves it unset");
        assert_eq!(g.ram_cached_count(), 0, "GPU-resident cells leave the Warm pool count");
        // α NOW lights up: crossing into GPU residency targets 1.0.
        g.advance_crossfade(50.0, 100.0);
        assert!((g.alpha(far).unwrap() - 0.5).abs() < 1e-6);

        // Retreat past the UNCHANGED margin_demote floor (305 < 310): drops
        // to Warm — RAM retained — instead of all the way to Outer.
        g.classify(&[observer_at(305.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Margin, to: Domain::Warm }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached(far), Some(true));

        // Retreat past warm_demote (139 < 140): Warm→Outer, the cool-down.
        g.classify(&[observer_at(139.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Warm, to: Domain::Outer }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached_count(), 0);
    }

    #[test]
    fn sprint_in_skips_warm_staging_entirely() {
        let mut g = StreamingGrid::new(warm_cfg(), budget(), &[]).unwrap();
        let far = CellCoord { x: 5, z: 0 };
        g.materialize(far);
        // Observer jumps straight into the margin zone (center 400 ≥ 330),
        // never touching the warm-only ring [180, 330).
        g.classify(&[observer_at(400.0)]);
        let ts = g.take_transitions();
        assert_eq!(
            ts,
            vec![Transition { coord: far, from: Domain::Outer, to: Domain::Margin }],
            "fast path: sprint-in registers straight away — no RAM staging step, legacy latency"
        );
        g.commit_transition(ts[0]);
        // The legacy two-step cascade continues unchanged from there.
        g.classify(&[observer_at(505.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Margin, to: Domain::Inner }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.domain(far), Some(Domain::Inner));
    }

    #[test]
    fn warm_cap_holds_staging_until_room_frees() {
        let mut b = budget();
        b.max_ram_cached_cells = 1;
        let mut g = StreamingGrid::new(warm_cfg(), b, &[]).unwrap();
        // Rings kept disjoint along the observer path: `a`'s warm_promote
        // triggers at center ≥ 180; `c` (mirrored across the origin) at
        // center ≤ −300 — so any position qualifying one disqualifies the
        // other and assertions stay order-independent.
        let a = CellCoord { x: 5, z: 0 }; // base [500, 600]
        let c = CellCoord { x: -6, z: 0 }; // base [-700, -600]: warm ⟺ center ≤ -300
        g.materialize(a);
        g.materialize(c);

        // Only `a` is inside its warm ring (center 200): warms, cap 1/1.
        g.classify(&[observer_at(200.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: a, from: Domain::Outer, to: Domain::Warm }]);
        g.commit_transition(ts[0]);

        // `a` holds (300 ≥ 140, below margin_promote) and `c` is not yet
        // eligible (300 > −300): the boundary produces nothing anyway.
        g.classify(&[observer_at(300.0)]);
        assert!(g.take_transitions().is_empty());

        // Walk left past a's warm_demote floor (130 < 140): a cools and
        // frees the slot; c still ineligible (130 > −300).
        g.classify(&[observer_at(130.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: a, from: Domain::Warm, to: Domain::Outer }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached_count(), 0);

        // Now c stages (−310 ≤ −300; a back to Outer but −310 < 180 keeps
        // it ineligible): the freed cap admits exactly the new tenant.
        g.classify(&[observer_at(-310.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: c, from: Domain::Outer, to: Domain::Warm }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached_count(), 1);
    }

    #[test]
    fn warm_budget_validation_fails_construction() {
        // Cap exceeding materialized cells cannot be bounded.
        let mut b = budget();
        b.max_ram_cached_cells = b.max_materialized_cells + 1;
        assert_eq!(
            StreamingGrid::new(cfg(), b, &[]).unwrap_err(),
            BudgetError::RamCapExceedsMaterialized
        );

        // Worst-case warm bytes over the RAM ceiling.
        let mut b = budget(); // mean_cell_geometry_bytes = 1 MiB
        b.max_ram_cached_cells = 1024;
        b.ram_budget = 1023 * (1 << 20);
        assert_eq!(StreamingGrid::new(cfg(), b, &[]).unwrap_err(), BudgetError::RamOverBudget);

        // Ladder ordering: the warm boundary must sit strictly BEYOND the
        // margin boundary, else the concentric zones interleave.
        let bad = GridConfig {
            warm: Some(WarmTierConfig { radius: 150.0, pad_fraction: 0.10, hysteresis: 40.0 }),
            ..cfg()
        };
        assert_eq!(
            StreamingGrid::new(bad, budget(), &[]).unwrap_err(),
            BudgetError::WarmRadiusNotBeyondMargin
        );
    }

    #[test]
    fn pin_to_warm_forces_ram_residency_past_any_observer() {
        let mut g = StreamingGrid::new(warm_cfg(), budget(), &[]).unwrap();
        let c = CellCoord { x: 0, z: 0 };
        g.materialize(c);
        assert!(g.pin(c, Domain::Warm));
        // Pins bypass concentric rules (and the warm cap, like every rule):
        // forced staging regardless of distance.
        g.classify(&[observer_at(10_000.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: c, from: Domain::Outer, to: Domain::Warm }]);
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached(c), Some(true));
        g.classify(&[observer_at(10_000.0)]);
        assert!(g.take_transitions().is_empty(), "stays pinned Warm");
    }

    #[test]
    fn tier_off_margin_drops_straight_to_outer_no_warm_rung() {
        // Same geometry as the tier-on walk, but warm: None — the demote
        // target at the unchanged margin_demote floor must be Outer, and no
        // Warm state may appear anywhere.
        let mut g = StreamingGrid::new(cfg(), budget(), &[]).unwrap();
        let far = CellCoord { x: 5, z: 0 };
        g.materialize(far);
        g.classify(&[observer_at(480.0)]);
        let ts = g.take_transitions();
        assert_eq!(ts, vec![Transition { coord: far, from: Domain::Outer, to: Domain::Margin }]);
        g.commit_transition(ts[0]);
        g.classify(&[observer_at(305.0)]);
        let ts = g.take_transitions();
        assert_eq!(
            ts,
            vec![Transition { coord: far, from: Domain::Margin, to: Domain::Outer }],
            "tier off: full teardown at the legacy floor, no intermediate Warm"
        );
        g.commit_transition(ts[0]);
        assert_eq!(g.ram_cached(far), Some(false));
    }
}
