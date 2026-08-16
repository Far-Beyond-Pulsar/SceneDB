//! Shared first-fit range suballocator (extracted from `gpu::assets`'s
//! `GeometryArena`, where this exact algorithm was already proven — see
//! that module for the original design note). Unit-agnostic: callers decide
//! whether `len`/`align` mean bytes or elements; this type only tracks free
//! spans and never touches a buffer itself.
//!
//! Two real consumers now share this: [`super::assets::GeometryArena`]'s
//! write-once-at-load geometry residency (a hard ceiling, no regrow — see
//! that module), and [`super::var_len_pool::VarLenGpuPool`]'s growable
//! variable-length `#[gpu]` field pool (grows the backing buffer and extends
//! this allocator's free space to match, rather than ever failing outright).

/// Sorted, non-adjacent free spans over one linear range `[0, total)`.
/// First-fit allocation, free-span coalescing on free.
pub(crate) struct RangeList {
    /// Sorted, non-adjacent free spans: (offset, len).
    free: Vec<(u64, u64)>,
}

impl RangeList {
    pub(crate) fn new(total: u64) -> Self {
        Self { free: vec![(0, total)] }
    }

    pub(crate) fn alloc(&mut self, len: u64, align: u64) -> Option<u64> {
        debug_assert!(align.is_power_of_two());
        for i in 0..self.free.len() {
            let (off, span) = self.free[i];
            let aligned = (off + align - 1) & !(align - 1);
            let pad = aligned - off;
            if pad + len <= span {
                // Split: [off, aligned) stays free (alignment pad),
                // [aligned+len, off+span) stays free (tail).
                let tail = span - pad - len;
                self.free.remove(i);
                if tail > 0 {
                    self.free.insert(i, (aligned + len, tail));
                }
                if pad > 0 {
                    self.free.insert(i, (off, pad));
                }
                return Some(aligned);
            }
        }
        None
    }

    pub(crate) fn free(&mut self, offset: u64, len: u64) {
        if len == 0 {
            // A zero-length allocation was never really an allocation (see
            // `VarLenGpuPool::write_var_row`'s empty-slice short-circuit) --
            // freeing it is a harmless no-op, not a double-free.
            return;
        }
        debug_assert!(
            self.free.iter().all(|&(o, l)| offset + len <= o || o + l <= offset),
            "double-free or overlapping free range"
        );
        let idx = self.free.partition_point(|&(o, _)| o < offset);
        self.free.insert(idx, (offset, len));
        // Coalesce with next, then with previous.
        if idx + 1 < self.free.len() && self.free[idx].0 + self.free[idx].1 == self.free[idx + 1].0 {
            self.free[idx].1 += self.free[idx + 1].1;
            self.free.remove(idx + 1);
        }
        if idx > 0 && self.free[idx - 1].0 + self.free[idx - 1].1 == self.free[idx].0 {
            self.free[idx - 1].1 += self.free[idx].1;
            self.free.remove(idx);
        }
    }

    /// Extends the tracked range to a new, larger `total`, adding the newly
    /// covered span `[old_total, new_total)` as free. Used by growable
    /// consumers ([`super::var_len_pool::VarLenGpuPool`]) after growing
    /// their backing buffer, to keep this allocator's view in sync -- the
    /// fixed-ceiling consumer ([`super::assets::GeometryArena`]) never calls
    /// this, by design (`ArenaError::Exhausted` is deliberate there, not a
    /// gap this method exists to paper over).
    pub(crate) fn extend_total(&mut self, old_total: u64, new_total: u64) {
        debug_assert!(new_total >= old_total);
        if new_total == old_total {
            return;
        }
        let added = new_total - old_total;
        // Coalesce into the last free span if it already ends exactly at
        // `old_total` (the overwhelmingly common case -- nothing was
        // allocated past the previous tail); otherwise the new span simply
        // becomes its own trailing free entry.
        if let Some(last) = self.free.last_mut() {
            if last.0 + last.1 == old_total {
                last.1 += added;
                return;
            }
        }
        self.free.push((old_total, added));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_round_trips() {
        let mut rl = RangeList::new(100);
        let a = rl.alloc(10, 1).unwrap();
        let b = rl.alloc(20, 1).unwrap();
        assert_eq!(a, 0);
        assert_eq!(b, 10);
        rl.free(a, 10);
        // Freeing `a` back must make room for a same-size allocation again.
        let c = rl.alloc(10, 1).unwrap();
        assert_eq!(c, 0);
    }

    #[test]
    fn exhaustion_returns_none_not_a_panic() {
        let mut rl = RangeList::new(10);
        assert!(rl.alloc(11, 1).is_none());
        assert!(rl.alloc(10, 1).is_some());
        assert!(rl.alloc(1, 1).is_none());
    }

    #[test]
    fn extend_total_coalesces_with_the_trailing_free_span() {
        let mut rl = RangeList::new(10);
        let _ = rl.alloc(10, 1).unwrap(); // fully consumed, no free span left
        rl.extend_total(10, 20);
        // The newly-added [10, 20) must be allocatable now.
        let a = rl.alloc(10, 1).unwrap();
        assert_eq!(a, 10);
    }

    #[test]
    fn extend_total_after_a_free_hole_still_finds_the_new_space() {
        let mut rl = RangeList::new(10);
        let a = rl.alloc(5, 1).unwrap();
        let _b = rl.alloc(5, 1).unwrap();
        rl.free(a, 5); // hole at [0, 5)
        rl.extend_total(10, 20);
        // Both the old hole and the new tail must be usable.
        let c = rl.alloc(5, 1).unwrap();
        assert_eq!(c, 0, "reuses the freed hole first (first-fit)");
        let d = rl.alloc(10, 1).unwrap();
        assert_eq!(d, 10, "then the newly-extended tail");
    }
}
