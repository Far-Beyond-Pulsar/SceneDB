//! Growable, dirty-tracked GPU column for World-mirrored
//! `#[gpu(mirror = DirtyTracked)]` fields (the default `#[gpu]` mode) —
//! writes are recorded (marked dirty, CPU-side) rather than uploaded
//! immediately; [`World::flush_gpu_mirror`](crate::world::World::flush_gpu_mirror)
//! performs the actual, coalesced upload once per call.
//!
//! # Why a CPU-side shadow, not a re-read from `World`'s own storage
//!
//! The cell-mirrored path's `SceneBuffer::sync_region` coalesces dirty rows
//! by reading straight from `CellStorage`'s own per-field SoA columns — those
//! columns are already indexed by the exact row the GPU buffer uses.
//! `World`'s archetype storage has no equivalent: a component's data lives
//! in `Column<T>` at an ARCHETYPE ROW, which moves on migration/compaction
//! and has no relationship to `Entity::index()` (the row every World-mirrored
//! GPU buffer is keyed by). There is no existing "read component `T` for
//! every entity, in `Entity::index()` order" primitive to read from.
//!
//! So this type keeps its own CPU-side shadow — `Vec<T>`, indexed by
//! `Entity::index()` directly, by construction — as the coalescing scan's
//! source of truth. This costs real CPU memory (one shadow row per mirrored
//! row, same size as the GPU element), which is the honest price of
//! deferring/coalescing writes for data that isn't naturally row-indexed on
//! the CPU side the way `CellStorage` already is.
//!
//! # Scope decision: a second coalescing-scan implementation, not a shared one
//!
//! [`Self::flush`]'s run-detection loop mirrors `SceneBuffer::sync_region`'s
//! algorithm (strict adjacency — no `GAP_MERGE_THRESHOLD`-style gap bridging;
//! see that constant's own extensively-benchmarked doc for why strict
//! adjacency is the right default) rather than sharing an implementation
//! with it. Refactoring `SceneBuffer::sync_region` — already shipped,
//! already perf-validated — to serve both this type and the cell-mirrored
//! path wasn't judged worth the risk to land alongside this feature without
//! its own benchmarking pass; a reasonable follow-up, not done here.
use crate::gpu::dynamic_buffer::CapacityError;
use crate::gpu::{DirtyMask, SyncStats};
use crate::page::Pod;
use std::sync::{Arc, RwLock};

struct DirtyTrackedState<T: Pod> {
    buf: crate::gpu::DynamicGpuBuffer<T>,
    shadow: Vec<T>,
    dirty: DirtyMask,
}

/// Grows `state.shadow`/`state.dirty` (pure CPU, no GPU work) to cover
/// `new_len`, if it doesn't already — shared by [`DirtyTrackedSceneBuffer::mark_dirty`]
/// (grows lazily, one row at a time as marks arrive) and
/// [`DirtyTrackedSceneBuffer::reserve`] (grows eagerly, ahead of a known
/// batch). `DirtyMask` has no resize primitive of its own, so growing it
/// means rebuilding at the new capacity and re-marking whatever was already
/// dirty — only runs when `new_len` actually exceeds the current shadow
/// length, not on every call.
fn grow_shadow_to<T: Pod>(state: &mut DirtyTrackedState<T>, new_len: usize) {
    if new_len <= state.shadow.len() {
        return;
    }
    // SAFETY: `T: Pod`'s own safety contract (`page.rs`) guarantees
    // all-zero bytes are a valid `T`.
    state.shadow.resize(new_len, unsafe { std::mem::zeroed::<T>() });
    let new_dirty = DirtyMask::new(new_len as u32);
    for r in 0..state.dirty.capacity() {
        if state.dirty.is_marked(r) {
            new_dirty.mark(r);
        }
    }
    state.dirty = new_dirty;
}

/// Type-erased counterpart, mirroring [`super::GrowableGpuBufferDispatch`]'s
/// own reasoning (the buffer lives behind a lock, so `&wgpu::Buffer` can't be
/// returned directly — [`Self::with_buffer`] is the lock-safe replacement).
pub trait DirtyTrackedGpuBufferDispatch: Send + Sync {
    /// Records `data` as row `row`'s new value and marks it dirty — no GPU
    /// work at all (not even a device/queue borrow) unless growth is needed,
    /// in which case only the CPU-side shadow/mask grow; the GPU buffer
    /// itself grows lazily, at the next [`Self::flush`].
    fn mark_dirty_bytes(&self, row: u32, data: &[u8]);

    /// Uploads every row marked dirty since the last flush, coalesced into
    /// as few `queue.write_buffer` calls as adjacency allows. Grows the GPU
    /// buffer to match the shadow's capacity first, if needed — this is the
    /// only point this type's growth can (in principle) fail, and it can't
    /// in practice, since these buffers are never registered with a
    /// `max_capacity` ceiling (see `SceneGpuStore::register_dirty_tracked_gpu_buffer`).
    fn flush(&self, queue: &wgpu::Queue) -> SyncStats;

    /// Grows the shadow and GPU buffer to `capacity` right now. See
    /// [`DirtyTrackedSceneBuffer::reserve`].
    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError>;

    /// Shrinks the shadow and GPU buffer to fit `highest_live_row`. See
    /// [`DirtyTrackedSceneBuffer::shrink_to_fit`].
    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool;

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer));

    fn as_any(&self) -> &dyn std::any::Any;
}

pub struct DirtyTrackedSceneBuffer<T: Pod> {
    device: Arc<wgpu::Device>,
    state: RwLock<DirtyTrackedState<T>>,
}

impl<T: Pod + Send + Sync + 'static> DirtyTrackedSceneBuffer<T> {
    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32) -> Self {
        let buf = crate::gpu::DynamicGpuBuffer::new(&device, label, initial_capacity);
        // SAFETY: `T: Pod`'s own safety contract (`page.rs`) guarantees
        // all-zero bytes are a valid `T` -- the same invariant `CellStorage`
        // itself relies on for freshly-allocated, not-yet-written rows.
        let shadow = vec![unsafe { std::mem::zeroed::<T>() }; initial_capacity as usize];
        let dirty = DirtyMask::new(initial_capacity);
        Self { device, state: RwLock::new(DirtyTrackedState { buf, shadow, dirty }) }
    }

    pub fn mark_dirty(&self, row: u32, value: T) {
        let mut state = self.state.write().expect("DirtyTrackedSceneBuffer lock poisoned");
        if row as usize >= state.shadow.len() {
            let mut new_len = state.shadow.len().max(1);
            while new_len <= row as usize {
                new_len = new_len.saturating_mul(2);
            }
            grow_shadow_to(&mut state, new_len);
        }
        state.shadow[row as usize] = value;
        state.dirty.mark(row);
    }

    /// Grows the CPU-side shadow (and `DirtyMask`) to cover `capacity` right
    /// now, and the underlying GPU buffer to match, ahead of any
    /// `mark_dirty`/`flush` call that would otherwise trigger either
    /// lazily. See [`crate::gpu::DynamicGpuBuffer::reserve`]'s doc for the
    /// intent — moving a known-size batch's reallocation cost off the
    /// per-insert/per-flush critical path. Growing the GPU buffer here too
    /// (not just the shadow) means a `flush()` right after a reserved batch
    /// doesn't ALSO need to grow it at that point.
    pub fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        let mut state = self.state.write().expect("DirtyTrackedSceneBuffer lock poisoned");
        grow_shadow_to(&mut state, capacity as usize);
        if state.buf.capacity() < capacity {
            state.buf.reserve(&self.device, queue, capacity)?;
        }
        Ok(())
    }

    /// Shrinks both the CPU-side shadow and the underlying GPU buffer to
    /// the smallest size that still covers `highest_live_row` (plus
    /// `slack_factor` headroom). See
    /// [`crate::gpu::DynamicGpuBuffer::shrink_to_fit`]'s doc — same
    /// semantics, applied to both halves of this type's storage together
    /// (shrinking only the GPU buffer while leaving a larger shadow around
    /// would just waste CPU memory for no benefit).
    pub fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        let mut state = self.state.write().expect("DirtyTrackedSceneBuffer lock poisoned");
        let target = (((highest_live_row as u64 + 1) as f64 * slack_factor.max(1.0) as f64).ceil() as u64)
            .min(u32::MAX as u64) as usize;
        if target >= state.shadow.len() {
            return false;
        }
        state.shadow.truncate(target);
        state.shadow.shrink_to_fit();
        let new_dirty = DirtyMask::new(target as u32);
        for r in 0..(target as u32).min(state.dirty.capacity()) {
            if state.dirty.is_marked(r) {
                new_dirty.mark(r);
            }
        }
        state.dirty = new_dirty;
        state.buf.shrink_to_fit(&self.device, queue, highest_live_row, slack_factor)
    }

    pub fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        let mut state = self.state.write().expect("DirtyTrackedSceneBuffer lock poisoned");
        let shadow_len = state.shadow.len() as u32;
        if state.buf.capacity() < shadow_len {
            state
                .buf
                .ensure_capacity(&self.device, queue, shadow_len)
                .expect("DirtyTrackedSceneBuffer never sets a max_capacity -- growth cannot fail");
        }
        let DirtyTrackedState { buf, shadow, dirty } = &mut *state;
        let mut stats = SyncStats { ranges: 0, bytes: 0 };
        let mut run_start: Option<u32> = None;
        let flush_run = |start: u32, end: u32, stats: &mut SyncStats| {
            buf.write(queue, start, &shadow[start as usize..end as usize]);
            stats.ranges += 1;
            stats.bytes += ((end - start) as usize * std::mem::size_of::<T>()) as u64;
        };
        for row in 0..shadow_len {
            if dirty.is_marked(row) {
                if run_start.is_none() {
                    run_start = Some(row);
                }
            } else if let Some(start) = run_start.take() {
                flush_run(start, row, &mut stats);
            }
        }
        if let Some(start) = run_start {
            flush_run(start, shadow_len, &mut stats);
        }
        dirty.clear_all();
        stats
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let state = self.state.read().expect("DirtyTrackedSceneBuffer lock poisoned");
        f(state.buf.buffer());
    }
}

impl<T: Pod + Send + Sync + 'static> DirtyTrackedGpuBufferDispatch for DirtyTrackedSceneBuffer<T> {
    fn mark_dirty_bytes(&self, row: u32, data: &[u8]) {
        assert_eq!(data.len(), std::mem::size_of::<T>(), "byte slice length must equal one element");
        // SAFETY: `T: Pod`, and `data` is exactly one element's worth of
        // bytes (asserted above) -- a valid, aligned read of `T` by value.
        let value: T = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const T) };
        self.mark_dirty(row, value);
    }

    fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        DirtyTrackedSceneBuffer::flush(self, queue)
    }

    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        DirtyTrackedSceneBuffer::reserve(self, queue, capacity)
    }

    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        DirtyTrackedSceneBuffer::shrink_to_fit(self, queue, highest_live_row, slack_factor)
    }

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        DirtyTrackedSceneBuffer::with_buffer(self, f)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
            label: Some("dirty-tracked-scene-buffer-test"),
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
    fn marking_dirty_does_not_touch_the_gpu_until_flush() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> = DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 8);

        buf.mark_dirty(0, 111);
        buf.mark_dirty(1, 222);
        buf.mark_dirty(5, 555);

        // Before flush: the GPU buffer must still read as all-zero -- the
        // marks are pure CPU-side bookkeeping so far.
        let mut before = Vec::new();
        buf.with_buffer(&mut |b| before = readback(&device, &queue, b, 8 * 4));
        assert!(before.iter().all(|&b| b == 0), "no GPU write must have happened before flush");

        let stats = buf.flush(&queue);
        // Two runs: [0,1] (adjacent, one coalesced write) and [5,5] --
        // strict adjacency (no gap bridging), matching SceneBuffer::sync_region's
        // GAP_MERGE_THRESHOLD == 0 default.
        assert_eq!(stats.ranges, 2, "row 0-1 coalesce into one run, row 5 is a separate run (gap at 2,3,4)");

        let mut after = Vec::new();
        buf.with_buffer(&mut |b| after = readback(&device, &queue, b, 8 * 4));
        let at = |row: usize| u32::from_ne_bytes(after[row * 4..row * 4 + 4].try_into().unwrap());
        assert_eq!(at(0), 111);
        assert_eq!(at(1), 222);
        assert_eq!(at(2), 0, "never marked -- must stay zero");
        assert_eq!(at(5), 555);

        // A second flush with nothing newly dirty must be a true no-op.
        let stats2 = buf.flush(&queue);
        assert_eq!(stats2.ranges, 0);
        assert_eq!(stats2.bytes, 0);
    }

    #[test]
    fn mark_dirty_past_capacity_grows_the_shadow_and_the_flush_still_lands_correctly() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> = DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);

        buf.mark_dirty(0, 1);
        buf.mark_dirty(50, 999); // far past initial capacity of 2

        let stats = buf.flush(&queue);
        assert_eq!(stats.ranges, 2);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 51 * 4));
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 1);
        assert_eq!(u32::from_ne_bytes(bytes[200..204].try_into().unwrap()), 999);
    }

    #[test]
    fn reserve_grows_both_shadow_and_gpu_buffer_so_flush_needs_no_further_growth() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> = DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);

        buf.reserve(&queue, 500).expect("reserve");
        for row in 0..500u32 {
            buf.mark_dirty(row, row * 2);
        }
        let stats = buf.flush(&queue);
        assert_eq!(stats.ranges, 1, "500 contiguous dirty rows must coalesce into one range");

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 500 * 4));
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 0);
        assert_eq!(u32::from_ne_bytes(bytes[996..1000].try_into().unwrap()), 498);
    }

    #[test]
    fn shrink_to_fit_reclaims_both_shadow_and_gpu_buffer() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> = DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);
        buf.mark_dirty(999, 42);
        buf.flush(&queue);

        let shrank = buf.shrink_to_fit(&queue, 999, 1.0);
        assert!(shrank);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 1000 * 4));
        assert_eq!(bytes.len(), 4000, "buffer must have actually shrunk to exactly 1000 rows");
        assert_eq!(u32::from_ne_bytes(bytes[999 * 4..999 * 4 + 4].try_into().unwrap()), 42);

        // Further marks/flushes must still work correctly post-shrink.
        buf.mark_dirty(500, 777);
        buf.flush(&queue);
        let mut bytes2 = Vec::new();
        buf.with_buffer(&mut |b| bytes2 = readback(&device, &queue, b, 1000 * 4));
        assert_eq!(u32::from_ne_bytes(bytes2[500 * 4..500 * 4 + 4].try_into().unwrap()), 777);
    }
}
