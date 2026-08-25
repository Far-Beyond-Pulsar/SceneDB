//! Segment-layout registry for tiered payloads (SceneDB#61 §3): the
//! "arbitrary-POD answer" to *which parts of a value move together* between
//! Disk/RAM/VRAM. A consumer whose payload has internal structure worth
//! streaming separately (think: coarse-then-fine byte regions of one asset —
//! named here only to motivate; see non-goals) registers ONE layout per
//! payload TYPE; everyone else gets the default and needs nothing.
//!
//! # Contract
//!
//! - **Default = atomic tiering**: an unregistered payload type tiers as ONE
//!   implicit segment covering the entire value at one rank. Works for every
//!   POD type with ZERO declarations — this is the zero-ceremony path, and
//!   it is the only path anything in this crate exercises unless a consumer
//!   explicitly registers a layout.
//! - **Opt-in layout**: [`register_segment_layout::<T>`] attaches an ordered
//!   segment list `[(offset, len, rank)]` to `T`'s [`TypeId`]. Offsets/lens
//!   are BYTE ranges within one logical payload's byte span; `rank` is an
//!   OPAQUE integer whose only meaning is total order. This crate never
//!   interprets a rank beyond sorting it.
//! - **Prefix-complete invariant**: residency through rank `k` implies every
//!   segment of rank ≤ `k` is resident. Promotion executes ranks in
//!   ASCENDING order regardless of the order requests arrived in; demotion
//!   executes in DESCENDING (reverse-rank) order; budget admission is
//!   evaluated PER RANK UNIT before that unit's commit, so a declined unit
//!   leaves the prefix below it intact and complete — never a hole.
//! - **Clamping**: a layout registered for a payload TYPE also sees
//!   variable-length payloads typed by that element. A segment extending
//!   past a concrete payload's actual byte span is CLAMPED (possibly to
//!   empty); empty-clamped rank units are skipped entirely — they cost zero
//!   bytes and never break the prefix property. Layouts should be authored
//!   against realistic payload sizes; clamping is a safety net, not a
//!   feature.
//! - **Loud validation**: overlapping segments (two segments claiming the
//!   same byte — admission math would double-count) and zero-length segments
//!   are rejected at registration with [`LayoutError`], not silently
//!   tolerated. Duplicate ranks are LEGAL: two segments sharing a rank form
//!   one admission/demotion unit (they always move together).
//!
//! # Cost model
//!
//! Registration is O(seg log seg) (validation sort), paid once per type at
//! consumer setup. Building an execution plan for one promotion flight is
//! O(seg log seg) (sort unique ranks + clamp), paid ONCE when the flight is
//! queued at a flush boundary — never on the per-frame write/touch hot path.
//! The atomic default builds a one-unit plan with zero registry traffic
//! (one `HashMap` miss).
//!
//! # What is NOT covered (deliberately)
//!
//! No format parsing of any kind — this module never looks inside a payload;
//! offsets/lens/ranks arrive pre-computed from whoever registers them. No
//! "mip" concept, no texture/layout/rendering concepts — ranks and ranges
//! only (SceneDB#61's hard constraint). No per-element repetition semantics:
//! a layout applies to the WHOLE payload byte span (clamped); authoring one
//! layout per element of a `Vec<T>` is a downstream concern this crate
//! deliberately does not model. No persistence: registrations live for the
//! process lifetime (`Arc`-shared, immutable once set).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::any::TypeId;

/// One segment of a payload's byte span: `len` bytes starting at byte
/// `offset`, admitted/demoted as part of rank-unit `rank`. See the module
/// doc for the full contract; `rank` is opaque here — total order only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub offset: u64,
    pub len: u64,
    pub rank: u32,
}

impl Segment {
    pub const fn new(offset: u64, len: u64, rank: u32) -> Self {
        Self { offset, len, rank }
    }
}

/// Registration-time validation failure — loud by design (see module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    /// A segment declared `len == 0` — dead weight that could only ever be
    /// clamped away; declare fewer segments instead.
    ZeroLengthSegment,
    /// Two segments overlap in the payload's byte span — per-rank admission
    /// math would account shared bytes twice.
    OverlappingSegments,
    /// Re-registering a type whose existing layout differs from the new one.
    /// Identical re-registration is tolerated (idempotent), matching how
    /// every other registration path in this crate treats "already
    /// registered with compatible contents".
    ConflictingLayout,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::ZeroLengthSegment => write!(f, "segment layout contains a zero-length segment"),
            LayoutError::OverlappingSegments => write!(f, "segment layout contains overlapping byte ranges"),
            LayoutError::ConflictingLayout => write!(f, "segment layout conflicts with an already-registered layout for this type"),
        }
    }
}

impl std::error::Error for LayoutError {}

/// The registry: payload `TypeId` -> validated, rank-sorted segments.
/// Runtime-`RwLock`-guarded like `SceneGpuStore`'s owner maps — registration
/// is startup-time churn; the read side sits behind every promotion flight.
/// `Arc<[Segment]>` so readers clone a cheap handle, never the contents.
fn registry() -> &'static RwLock<HashMap<TypeId, Arc<[Segment]>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<TypeId, Arc<[Segment]>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Validates `segments` and attaches them as `T`'s payload layout. Idempotent
/// when re-registering byte-identical contents (see [`LayoutError::
/// ConflictingLayout`]); otherwise replaces nothing and errors loudly.
///
/// There is no unregister API on purpose: layouts are process-lifetime
/// facts, exactly like the inventory-submitted registrations elsewhere in
/// this crate. Tests that need a fresh slate use fresh types (new `TypeId`).
pub fn register_segment_layout<T: 'static>(segments: &[Segment]) -> Result<(), LayoutError> {
    let arc: Arc<[Segment]> = segments.into();
    validate(&arc)?;
    let mut reg = registry().write().expect("segment-layout registry lock poisoned");
    match reg.get(&TypeId::of::<T>()) {
        // Idempotent identical re-registration (same reason
        // `register_gpu_columns_growable` may be called repeatedly).
        Some(existing) if **existing == *arc => Ok(()),
        Some(_) => Err(LayoutError::ConflictingLayout),
        None => {
            reg.insert(TypeId::of::<T>(), arc);
            Ok(())
        }
    }
}

/// Registers a layout against a `TypeId` resolved at runtime — for callers
/// holding the payload type only as type-erased metadata (e.g. a var-len
/// pool registering against its ELEMENT type without naming it statically).
/// Same validation and idempotence contract as [`register_segment_layout`].
pub fn register_segment_layout_for_type(
    ty: TypeId,
    segments: &[Segment],
) -> Result<(), LayoutError> {
    let arc: Arc<[Segment]> = segments.into();
    validate(&arc)?;
    let mut reg = registry().write().expect("segment-layout registry lock poisoned");
    match reg.get(&ty) {
        Some(existing) if **existing == *arc => Ok(()),
        Some(_) => Err(LayoutError::ConflictingLayout),
        None => {
            reg.insert(ty, arc);
            Ok(())
        }
    }
}

/// `T`'s registered layout, if one exists. `None` = atomic default.
pub(crate) fn layout_for(ty: TypeId) -> Option<Arc<[Segment]>> {
    registry()
        .read()
        .expect("segment-layout registry lock poisoned")
        .get(&ty)
        .cloned()
}

fn validate(segments: &[Segment]) -> Result<(), LayoutError> {
    let mut sorted: Vec<Segment> = segments.to_vec();
    sorted.sort_by_key(|s| (s.offset, s.rank));
    for s in segments {
        if s.len == 0 {
            return Err(LayoutError::ZeroLengthSegment);
        }
    }
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b.offset < a.offset + a.len {
            return Err(LayoutError::OverlappingSegments);
        }
    }
    Ok(())
}

/// One rank-unit of an execution plan: all segments sharing `rank`, clamped
/// to the concrete payload's byte span, with their COMBINED clamped byte
/// footprint (what per-rank budget admission accounts).
#[derive(Clone, Debug)]
pub(crate) struct RankUnit {
    pub rank: u32,
    /// Clamped `(offset_in_payload, len)` slices, ascending by offset.
    pub slices: Vec<(u64, u64)>,
    /// Σ `len` of `slices` — the admission footprint of this unit.
    pub bytes: u64,
}

/// Builds the promotion/demotion execution plan for a payload of
/// `payload_bytes` actual bytes: registered layout if one exists for `ty`,
/// otherwise the single implicit atomic unit covering the whole span at
/// rank 0. Units ascend by rank (promotion order; reverse it for demotion).
pub(crate) fn execution_plan(ty: TypeId, payload_bytes: u64) -> Vec<RankUnit> {
    let Some(layout) = layout_for(ty) else {
        if payload_bytes == 0 {
            return Vec::new();
        }
        return vec![RankUnit { rank: 0, slices: vec![(0, payload_bytes)], bytes: payload_bytes }];
    };
    plan_from_segments(&layout, payload_bytes)
}

/// The plan-building half of [`execution_plan`], factored out so the atomic
/// default and registered layouts share the clamping/sorting code path.
pub(crate) fn plan_from_segments(layout: &[Segment], payload_bytes: u64) -> Vec<RankUnit> {
    if layout.is_empty() || payload_bytes == 0 {
        return Vec::new();
    }
    let mut by_rank: Vec<(u32, Vec<(u64, u64)>)> = Vec::new();
    for seg in layout {
        // Clamp to the concrete payload span (module-doc contract).
        let start = seg.offset.min(payload_bytes);
        let end = seg.offset.saturating_add(seg.len).min(payload_bytes);
        if end <= start {
            continue; // empty-clamped: skipped entirely, never breaks the prefix
        }
        match by_rank.iter_mut().find(|(r, _)| *r == seg.rank) {
            Some((_, slices)) => slices.push((start, end - start)),
            None => by_rank.push((seg.rank, vec![(start, end - start)])),
        }
    }
    by_rank.sort_by_key(|(r, _)| *r);
    by_rank
        .into_iter()
        .map(|(rank, mut slices)| {
            slices.sort_by_key(|&(off, _)| off);
            let bytes = slices.iter().map(|&(_, len)| len).sum();
            RankUnit { rank, slices, bytes }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unregistered_type_plans_one_atomic_unit() {
        struct Unregistered;
        // Zero-declaration default: the WHOLE value at rank 0, one unit.
        let plan = execution_plan(TypeId::of::<Unregistered>(), 512);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].rank, 0);
        assert_eq!(plan[0].bytes, 512);
        assert_eq!(plan[0].slices, vec![(0, 512)]);
    }

    #[test]
    fn registered_layout_plans_units_in_rank_order_and_clamps() {
        struct Segmented;
        // Disjoint byte spans: A=[0,128)@r0, B=[128,384)@r2, C=[384,512)@r7.
        register_segment_layout::<Segmented>(&[
            Segment::new(384, 128, 7), // higher rank declared FIRST — order must not matter
            Segment::new(0, 128, 0),
            Segment::new(128, 256, 2),
        ])
        .expect("valid layout");

        let plan = execution_plan(TypeId::of::<Segmented>(), 512);
        assert_eq!(plan.iter().map(|u| u.rank).collect::<Vec<_>>(), vec![0, 2, 7], "ascending rank order");
        assert_eq!(plan.iter().map(|u| u.bytes).collect::<Vec<_>>(), vec![128, 256, 128]);

        // A SHORTER concrete payload clamps/truncates later-rank segments —
        // the surviving prefix stays complete.
        let short = execution_plan(TypeId::of::<Segmented>(), 200);
        assert_eq!(short.len(), 2, "rank 7 clamps to empty and vanishes");
        assert_eq!(short[0].rank, 0);
        assert_eq!(short[0].slices, vec![(0, 128)], "rank 0's slice survives intact");
        assert_eq!(short[1].rank, 2);
        assert_eq!(short[1].slices, vec![(128, 72)], "truncated to what fits");

        let mid = execution_plan(TypeId::of::<Segmented>(), 320);
        assert_eq!(mid.len(), 2);
        assert_eq!(mid[1].bytes, 192);
    }

    #[test]
    fn duplicate_ranks_form_one_admission_unit() {
        struct Paired;
        register_segment_layout::<Paired>(&[
            Segment::new(0, 100, 0),
            Segment::new(100, 50, 1),
            Segment::new(150, 50, 1), // same rank as its neighbor: one unit
        ])
        .expect("valid layout");
        let plan = execution_plan(TypeId::of::<Paired>(), 200);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[1].rank, 1);
        assert_eq!(plan[1].bytes, 100, "both same-rank segments admitted together");
        assert_eq!(plan[1].slices, vec![(100, 50), (150, 50)]);
    }

    #[test]
    fn invalid_layouts_are_rejected_loudly() {
        struct Bad;
        assert_eq!(
            register_segment_layout::<Bad>(&[Segment::new(0, 0, 0)]),
            Err(LayoutError::ZeroLengthSegment)
        );
        struct Overlap;
        assert_eq!(
            register_segment_layout::<Overlap>(&[Segment::new(0, 16, 0), Segment::new(8, 16, 1)]),
            Err(LayoutError::OverlappingSegments)
        );
        // Adjacent (non-overlapping) is fine.
        struct Adjacent;
        assert_eq!(
            register_segment_layout::<Adjacent>(&[Segment::new(0, 8, 0), Segment::new(8, 8, 1)]),
            Ok(())
        );
        // Identical re-registration is idempotent; differing is loud.
        assert_eq!(
            register_segment_layout::<Adjacent>(&[Segment::new(0, 8, 0), Segment::new(8, 8, 1)]),
            Ok(())
        );
        assert_eq!(
            register_segment_layout::<Adjacent>(&[Segment::new(0, 4, 0), Segment::new(8, 8, 1)]),
            Err(LayoutError::ConflictingLayout)
        );
    }
}
