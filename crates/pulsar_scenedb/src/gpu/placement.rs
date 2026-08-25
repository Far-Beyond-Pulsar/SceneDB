//! Placement annotations for `#[gpu]` var-len fields: the vocabulary that
//! tells the derive WHERE a cache/dedup boundary sits inside a field's type
//! signature, and what the GPU-side reference for it looks like.
//!
//! - [`GpuHeavy<T>`] marks the boundary itself. The real resource lives
//!   behind an interned store keyed by content identity; only the
//!   lightweight reference occupies this position in the CPU column.
//!   `repr(transparent)` over a `Pod` `T` makes a
//!   `Vec<GpuHeavy<H>>`'s bytes byte-for-byte a `[H]` — which is what lets
//!   the derive-generated mirror write such a field with ZERO per-element
//!   work and ZERO allocation (a pointer cast to `&[H]`, nothing more).
//! - [`GpuRef`] is what an optional position requires of its handle:
//!   `Option<GpuHeavy<H>>` lowers to a chain of `H` where absence is
//!   [`GpuRef::NULL`] — stored in place, never aliased, never skipped.
//! - [`structural_content_id`] gives a Heavy-placement field WITHOUT an
//!   explicit `content_id = "sibling"` declaration a deterministic fallback
//!   identity: a byte-fold over the row's own reference list, so two rows
//!   holding identical reference lists share one interned allocation.
//!
//! # Performance contract
//!
//! All of this is O(n) in the row's own element count — the same order as
//! the GPU upload every mirror write performs anyway — and the dominant
//! shapes allocate nothing: bare `Vec<GpuHeavy<H>>` writes reinterpret the
//! existing buffer in place; only `Option`/`Result` positions build a mapped
//! copy (one small allocation per edit-driven write, never per frame).

use crate::handle_ledger::HandleId;
use crate::page::Pod;

/// Marks a `#[property]`/`#[gpu]` field position as a cache/dedup boundary:
/// `T` stays a lightweight Pod reference (an asset id, typically 4-16
/// bytes); the actual GPU-resident payload is interned elsewhere, keyed by
/// content identity, shared by refcount across every row naming it.
///
/// Written so the derive can detect it SYNTACTICALLY (last path segment,
/// same detection philosophy as `Vec<T>`) — a proc macro cannot ask "does
/// this type implement some trait", so the wrapper's presence in the type
/// signature IS the annotation.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct GpuHeavy<T>(pub T);

impl<T> GpuHeavy<T> {
    /// Unwrap to the lightweight reference.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for GpuHeavy<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for GpuHeavy<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Copy> From<T> for GpuHeavy<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

// SAFETY: `#[repr(transparent)]` over `T` — byte-for-byte `T`'s own layout,
// the identical argument `HandleId`'s transparent-Pod impl makes.
unsafe impl<T: Pod> Pod for GpuHeavy<T> {}

/// What an OPTIONAL heavy position requires of its handle type: a NULL
/// sentinel distinguishing "slot deliberately empty" from any real
/// reference. Absent slots are STORED IN PLACE in the lowered chain (they
/// occupy their position; they are never aliased or compacted away).
pub trait GpuRef: Pod {
    const NULL: Self;
    fn is_null(self) -> bool;
}

macro_rules! impl_gpu_ref_for_int_primitives {
    ($($t:ty),*) => { $(
        impl GpuRef for $t {
            const NULL: Self = 0;
            fn is_null(self) -> bool { self == 0 }
        }
    )* };
}

impl_gpu_ref_for_int_primitives!(u32, i32, u64);

/// Deterministic structural content id over a row's raw reference list —
/// the fallback identity for a Heavy-placement field WITHOUT an explicit
/// `content_id = "sibling"`.
///
/// Deliberately distinct from [`crate::handle_ledger::ContentAddressed`]:
/// that trait carries SEMANTIC identity owned by downstream asset-wrapper
/// types ("this path names this asset"); this function is MECHANICAL
/// equality of the reference list itself ("these two rows name exactly the
/// same set of resources, in exactly the same order") — no domain knowledge,
/// no file I/O, stable within a process, which is the only lifetime an
/// intern table needs (the table dies with the store).
///
/// FNV-1a evaluated twice with distinct 64-bit bases, folded into a
/// `u128`: two independent diffusion lanes so adjacent-element collisions
/// need both lanes to collide simultaneously. O(n · size_of::<H>()) over
/// the row's own bytes — the same order as the upload it accompanies, and
/// it runs once per edit-driven mirror write, never per frame.
pub fn structural_content_id<H: Pod>(handles: &[H]) -> HandleId {
    const FNV_OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_OFFSET_B: u64 = 0x6c62_272e_07bb_0142;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let bytes = unsafe {
        // SAFETY: `H: Pod` and the slice came from a live `[H]`/`Vec<H>` —
        // every byte in range is initialized, no padding exists between
        // elements.
        ::std::slice::from_raw_parts(handles.as_ptr() as *const u8, handles.len() * ::std::mem::size_of::<H>())
    };

    let mut lane_a = FNV_OFFSET_A;
    let mut lane_b = FNV_OFFSET_B;
    for (index, byte) in bytes.iter().enumerate() {
        lane_a ^= u64::from(*byte);
        lane_a = lane_a.wrapping_mul(FNV_PRIME);
        // Lane B mixes the element INDEX too, so [h1, h2] and [h2, h1]
        // cannot fold to the same id merely by lane-cancellation luck.
        lane_b ^= u64::from(*byte).rotate_right((index % 8) as u32);
        lane_b = lane_b.wrapping_mul(FNV_PRIME);
    }
    HandleId((u128::from(lane_a) << 64) | u128::from(lane_b))
}
