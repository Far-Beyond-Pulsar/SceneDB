//! `ComponentId`-addressable growable GPU columns, for `#[gpu]` fields
//! mirrored through [`crate::gpu::world_mirror`] — `World`'s entity count has
//! no ceiling, so a buffer registered at a fixed capacity (`SceneBuffer<T>`,
//! via [`super::SceneGpuStore::register_gpu_buffer`]) is a guaranteed
//! eventual panic for long-running or entity-heavy worlds. This module gives
//! that registration path a growable alternative built on
//! [`super::DynamicGpuBuffer`]'s grow-and-copy mechanics.
//!
//! Cell-mirrored `#[gpu]` columns (the `CellStorage`/`Handle` path) are
//! unaffected — `RegionPool`-based cells are already capacity-planned
//! deliberately, and keep using [`super::SceneGpuStore::register_gpu_buffer`]
//! exactly as before. This is additive, scoped to World-mirrored columns.
use crate::gpu::dynamic_buffer::{CapacityError, DynamicGpuBuffer};
use crate::page::Pod;
use std::any::Any;
use std::sync::{Arc, RwLock};

/// Type-erased counterpart to [`super::GpuBufferDispatch`] for growable
/// columns. A separate trait, not an extension of `GpuBufferDispatch`,
/// because `GpuBufferDispatch::buffer(&self) -> &wgpu::Buffer` returns a
/// bare reference tied to `&self`'s lifetime — impossible to implement
/// soundly once the buffer lives behind an `RwLock` (there is no live
/// buffer to hand out a reference to without holding the lock guard, which
/// the trait signature has nowhere to put). [`Self::with_buffer`] is the
/// lock-safe replacement: it hands the current buffer to a closure for the
/// duration of one call, never leaking a guard past that.
pub trait GrowableGpuBufferDispatch: Send + Sync {
    /// Grows if necessary (see [`GrowableSceneBuffer::write_row_growing`])
    /// then writes. `data`'s length must be a multiple of the element size;
    /// mismatches are a logic error in the caller (the derive-generated
    /// dispatch always passes exactly one element's worth), asserted rather
    /// than silently truncated or padded.
    fn write_row_growing(&self, queue: &wgpu::Queue, row: u32, data: &[u8]) -> Result<(), CapacityError>;

    fn epoch(&self) -> u64;
    fn capacity(&self) -> u32;
    fn element_size(&self) -> usize;

    /// Grows to at least `capacity` right now, ahead of any write that
    /// would otherwise trigger it lazily — see
    /// [`super::DynamicGpuBuffer::reserve`]'s doc for the intent this
    /// exists to support (moving a known-size batch's reallocation cost off
    /// the per-insert critical path).
    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError>;

    /// Shrinks to the smallest capacity that still covers `highest_live_row`
    /// (plus `slack_factor` headroom) — see
    /// [`super::DynamicGpuBuffer::shrink_to_fit`]'s doc.
    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool;

    /// Runs `f` against the buffer's current `wgpu::Buffer` — the lock-safe
    /// way to reach it for e.g. bind group construction. Do not attempt to
    /// stash the `&wgpu::Buffer` past this call; a concurrent growth on
    /// another handle to the same registration is exactly the case this
    /// exists to make safe, and a stashed reference would defeat that.
    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer));

    fn as_any(&self) -> &dyn Any;
}

/// Growable, `ComponentId`-addressable counterpart to [`super::SceneBuffer`],
/// for `#[gpu]` columns registered through [`super::SceneGpuStore::register_growable_gpu_buffer`].
///
/// Shares [`DynamicGpuBuffer`]'s grow-and-copy mechanics (that's the
/// non-`ComponentId`-addressable version of the same idea, for pipeline-owned
/// data — see its module docs for the boundary between the two use cases)
/// rather than reimplementing them; this type only adds the `RwLock` needed
/// to grow safely behind a `&self`-only trait method (see
/// [`GrowableGpuBufferDispatch`]'s doc for why that's required), and the
/// device handle growth itself needs (`write_rows_raw`-style methods only
/// receive a `&wgpu::Queue`, not a `&wgpu::Device`, so this type keeps its
/// own).
pub struct GrowableSceneBuffer<T: Pod> {
    device: Arc<wgpu::Device>,
    inner: RwLock<DynamicGpuBuffer<T>>,
}

impl<T: Pod + Send + Sync + 'static> GrowableSceneBuffer<T> {
    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32, max_capacity: Option<u32>) -> Self {
        let mut buf = DynamicGpuBuffer::new(&device, label, initial_capacity);
        if let Some(max) = max_capacity {
            buf = buf.with_max_capacity(max);
        }
        Self { device, inner: RwLock::new(buf) }
    }

    /// Grows if `row` doesn't fit the current capacity, then writes one
    /// element at `row`. Double-checked: a cheap read-lock capacity check
    /// first (the common case — no growth needed, matching
    /// `DynamicGpuBuffer::write`'s own `&self` cost exactly), escalating to
    /// the write lock only when growth is actually required, and re-checking
    /// capacity under the write lock in case another writer grew it first
    /// (the classic double-checked-locking shape — avoids two concurrent
    /// writers both reallocating for the same shortfall).
    pub fn write_row_growing(&self, queue: &wgpu::Queue, row: u32, data: &[T]) -> Result<(), CapacityError> {
        let min_capacity = row.saturating_add(data.len() as u32);
        {
            let guard = self.inner.read().expect("GrowableSceneBuffer lock poisoned");
            if min_capacity <= guard.capacity() {
                guard.write(queue, row, data);
                return Ok(());
            }
        }
        let mut guard = self.inner.write().expect("GrowableSceneBuffer lock poisoned");
        if min_capacity > guard.capacity() {
            guard.ensure_capacity(&self.device, queue, min_capacity)?;
        }
        guard.write(queue, row, data);
        Ok(())
    }

    pub fn epoch(&self) -> u64 {
        self.inner.read().expect("GrowableSceneBuffer lock poisoned").epoch()
    }

    pub fn capacity(&self) -> u32 {
        self.inner.read().expect("GrowableSceneBuffer lock poisoned").capacity()
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let guard = self.inner.read().expect("GrowableSceneBuffer lock poisoned");
        f(guard.buffer());
    }

    /// See [`GrowableGpuBufferDispatch::reserve`].
    pub fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        let mut guard = self.inner.write().expect("GrowableSceneBuffer lock poisoned");
        guard.reserve(&self.device, queue, capacity)
    }

    /// See [`GrowableGpuBufferDispatch::shrink_to_fit`].
    pub fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        let mut guard = self.inner.write().expect("GrowableSceneBuffer lock poisoned");
        guard.shrink_to_fit(&self.device, queue, highest_live_row, slack_factor)
    }
}

impl<T: Pod + Send + Sync + 'static> GrowableGpuBufferDispatch for GrowableSceneBuffer<T> {
    fn write_row_growing(&self, queue: &wgpu::Queue, row: u32, data: &[u8]) -> Result<(), CapacityError> {
        assert_eq!(
            data.len() % std::mem::size_of::<T>(),
            0,
            "byte slice length not a multiple of element size"
        );
        let typed: &[T] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const T, data.len() / std::mem::size_of::<T>())
        };
        GrowableSceneBuffer::write_row_growing(self, queue, row, typed)
    }

    fn epoch(&self) -> u64 {
        GrowableSceneBuffer::epoch(self)
    }

    fn capacity(&self) -> u32 {
        GrowableSceneBuffer::capacity(self)
    }

    fn element_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        GrowableSceneBuffer::reserve(self, queue, capacity)
    }

    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        GrowableSceneBuffer::shrink_to_fit(self, queue, highest_live_row, slack_factor)
    }

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        GrowableSceneBuffer::with_buffer(self, f)
    }

    fn as_any(&self) -> &dyn Any {
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
            label: Some("growable-scene-buffer-test"),
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
    fn write_past_capacity_grows_transparently_and_preserves_prior_rows() {
        let (device, queue) = test_device();
        let buf: GrowableSceneBuffer<u32> = GrowableSceneBuffer::new(Arc::clone(&device), "test", 2, None);
        assert_eq!(buf.capacity(), 2);
        assert_eq!(buf.epoch(), 0);

        buf.write_row_growing(&queue, 0, &[111]).expect("row 0 fits initial capacity");
        buf.write_row_growing(&queue, 1, &[222]).expect("row 1 fits initial capacity");

        // Row 100 is way past capacity 2 -- must grow transparently, not panic.
        buf.write_row_growing(&queue, 100, &[999]).expect("must grow, not fail (no max_capacity set)");
        assert!(buf.capacity() > 100);
        assert!(buf.epoch() >= 1);

        let mut readback_bytes = Vec::new();
        buf.with_buffer(&mut |b| {
            readback_bytes = readback(&device, &queue, b, (buf.capacity() as u64) * 4);
        });
        let at = |row: usize| -> u32 {
            u32::from_ne_bytes(readback_bytes[row * 4..row * 4 + 4].try_into().unwrap())
        };
        assert_eq!(at(0), 111, "row 0 must survive the growth-triggering write");
        assert_eq!(at(1), 222, "row 1 must survive the growth-triggering write");
        assert_eq!(at(100), 999);
    }

    #[test]
    fn max_capacity_ceiling_is_still_honored_through_the_growing_write_path() {
        let (device, queue) = test_device();
        let buf: GrowableSceneBuffer<u32> = GrowableSceneBuffer::new(Arc::clone(&device), "test", 2, Some(4));

        buf.write_row_growing(&queue, 3, &[1]).expect("row 3 fits within max_capacity 4");
        let err = buf
            .write_row_growing(&queue, 10, &[1])
            .expect_err("row 10 exceeds max_capacity 4 -- must error, not silently drop the write");
        assert_eq!(err.max, 4);
    }

    #[test]
    fn reserve_moves_growth_off_the_write_path() {
        let (device, queue) = test_device();
        let buf: GrowableSceneBuffer<u32> = GrowableSceneBuffer::new(Arc::clone(&device), "test", 2, None);

        buf.reserve(&queue, 500).expect("reserve");
        assert!(buf.capacity() >= 500);
        let epoch_after_reserve = buf.epoch();

        // Every write in the reserved batch must land with zero further growth.
        for row in 0..500u32 {
            buf.write_row_growing(&queue, row, &[row]).expect("within reserved capacity");
        }
        assert_eq!(buf.epoch(), epoch_after_reserve, "a fully-reserved batch must trigger no further reallocation");
    }

    #[test]
    fn shrink_to_fit_reclaims_capacity_through_the_dispatch_trait() {
        let (device, queue) = test_device();
        let buf: GrowableSceneBuffer<u32> = GrowableSceneBuffer::new(Arc::clone(&device), "test", 2, None);
        buf.write_row_growing(&queue, 999, &[42]).expect("grow");
        let grown_capacity = buf.capacity();

        let dispatch: &dyn GrowableGpuBufferDispatch = &buf;
        let shrank = dispatch.shrink_to_fit(&queue, 999, 1.0);
        assert!(shrank);
        assert!(buf.capacity() < grown_capacity);
        assert_eq!(buf.capacity(), 1000);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 1000 * 4));
        assert_eq!(u32::from_ne_bytes(bytes[999 * 4..999 * 4 + 4].try_into().unwrap()), 42);
    }
}
