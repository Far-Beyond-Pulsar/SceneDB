//! Per-cell row dirty bitmask (design Rev 2 §2): dirty state lives beside the
//! CELL, not inside the global buffer — the same atomic-word shape as M1's
//! LivenessMask. Relaxed ordering per the M2a contract: the frame-boundary
//! join provides the happens-before edge between column writes and sync.

use std::sync::atomic::{AtomicU64, Ordering};

pub struct DirtyMask {
    words: Vec<AtomicU64>,
    capacity: u32,
}

impl DirtyMask {
    pub fn new(capacity: u32) -> Self {
        let words = (0..capacity.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Self { words, capacity }
    }

    #[inline]
    pub fn mark(&self, row: u32) {
        debug_assert!(row < self.capacity, "row {row} beyond mask capacity {}", self.capacity);
        self.words[(row / 64) as usize].fetch_or(1u64 << (row % 64), Ordering::Relaxed);
    }

    #[inline]
    pub fn is_marked(&self, row: u32) -> bool {
        self.words[(row / 64) as usize].load(Ordering::Relaxed) & (1u64 << (row % 64)) != 0
    }

    /// Mark rows `0..rows` (promotion warm-up: full-region resync, §4.1).
    pub fn mark_range(&self, rows: u32) {
        for row in 0..rows {
            self.mark(row);
        }
    }

    pub fn clear_all(&self) {
        for w in &self.words {
            w.store(0, Ordering::Relaxed);
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Calls `f(row)` for every marked row, in ascending order, skipping
    /// all-zero 64-bit words entirely (one atomic load + comparison per
    /// *empty* word, instead of 64 individual bit checks) — `O(capacity/64
    /// + marked_count)` instead of `O(capacity)`.
    ///
    /// Added for [`crate::gpu::DirtyTrackedSceneBuffer::flush`] (Helio#214):
    /// a real-device benchmark found its previous row-by-row scan cost
    /// nearly the same whether 0% or 10% of rows were dirty at 100,000
    /// capacity (~38.5 µs either way) — the scan itself, not the actual
    /// writes, dominated. Exposed here rather than kept private to that one
    /// caller because [`super::SceneBuffer::sync_region`]'s own row-by-row
    /// scan has the identical shape and could benefit the same way; not
    /// switched over in this change (see `dirty_tracked_scene_buffer.rs`'s
    /// module doc on why sharing `sync_region`'s implementation wasn't
    /// taken on without its own benchmarking pass) — but the primitive is
    /// here, ready, rather than duplicated privately.
    #[inline]
    pub fn for_each_marked(&self, mut f: impl FnMut(u32)) {
        for (word_idx, word) in self.words.iter().enumerate() {
            let bits = word.load(Ordering::Relaxed);
            if bits == 0 {
                continue;
            }
            let base = (word_idx as u32) * 64;
            let mut remaining = bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                f(base + bit);
                remaining &= remaining - 1; // clear the lowest set bit
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_marks_and_clears() {
        let m = DirtyMask::new(130);
        m.mark(0);
        m.mark(129);
        assert!(m.is_marked(0) && m.is_marked(129) && !m.is_marked(64));
        m.clear_all();
        assert!(!m.is_marked(0) && !m.is_marked(129));
        m.mark_range(65);
        assert!(m.is_marked(64) && !m.is_marked(65));
    }

    #[test]
    fn for_each_marked_visits_exactly_the_marked_rows_in_ascending_order() {
        let m = DirtyMask::new(200);
        m.mark(0);
        m.mark(63);
        m.mark(64); // adjacent word boundary -- must not be skipped or double-counted
        m.mark(150);

        let mut seen = Vec::new();
        m.for_each_marked(|row| seen.push(row));
        assert_eq!(seen, vec![0, 63, 64, 150]);

        m.clear_all();
        let mut seen_after_clear = Vec::new();
        m.for_each_marked(|row| seen_after_clear.push(row));
        assert!(seen_after_clear.is_empty(), "an all-zero mask must visit nothing, not even word 0");
    }
}
