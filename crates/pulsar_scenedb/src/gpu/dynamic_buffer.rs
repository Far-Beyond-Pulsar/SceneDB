//! A plain, row-count-agnostic growable SSBO — no `ComponentId`, no
//! `Entity`, no dirty-tracking, no region/cell concept. For pipeline-owned
//! GPU data whose length is decided by a compute pass each frame
//! (visibility lists, compacted indices, indirect draw args) rather than by
//! `World`'s entity count — see the "GPU-native fields on World entities"
//! README section for the full boundary this type exists to formalize:
//! that class of data is explicitly **not** what `#[gpu]` component fields
//! (`gpu::world_mirror`) are for, and previously had no supported home in
//! this crate at all, forcing every consumer to hand-roll their own growable
//! buffer (e.g. Helio's own `GrowableBuffer<T>`, built with no relationship
//! to this crate).
//!
//! A separate, `ComponentId`-addressable growable column type for
//! World-mirrored `#[gpu]` fields (tracked upstream) is expected to share
//! this type's grow-and-copy mechanics rather than reimplementing them — the
//! two differ only in whether they're reachable through `SceneGpuStore` by
//! type, not in how they grow.
use crate::page::Pod;
use std::marker::PhantomData;

/// Returned by [`DynamicGpuBuffer::ensure_capacity`] when growing to cover
/// `requested` would exceed the buffer's configured `max_capacity` — a real,
/// intentional ceiling (e.g. a hard VRAM budget), not a silent truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityError {
    pub requested: u32,
    pub max: u32,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requested capacity {} exceeds configured max_capacity {}",
            self.requested, self.max
        )
    }
}

impl std::error::Error for CapacityError {}

/// Growable counterpart to [`super::SceneBuffer`], for callers whose
/// required row count isn't known ahead of time and isn't tied to any
/// `Entity`/`ComponentId`. Doubles on demand (capped by an optional
/// `max_capacity`), GPU-to-GPU copying existing contents on every
/// reallocation.
///
/// The buffer's `wgpu::Buffer` **identity changes** on reallocation — any
/// bind group built against `buffer()` before a growth becomes stale.
/// [`Self::epoch`] exists specifically so callers holding such bind groups
/// can detect this cheaply (compare against a cached value each frame)
/// instead of re-querying `buffer()` and comparing pointers, or rebuilding
/// bind groups unconditionally every frame.
pub struct DynamicGpuBuffer<T: Pod> {
    buf: wgpu::Buffer,
    capacity: u32,
    max_capacity: Option<u32>,
    epoch: u64,
    label: String,
    _elem: PhantomData<T>,
}

impl<T: Pod> DynamicGpuBuffer<T> {
    pub fn new(device: &wgpu::Device, label: &str, initial_capacity: u32) -> Self {
        let buf = Self::alloc(device, label, initial_capacity);
        Self {
            buf,
            capacity: initial_capacity,
            max_capacity: None,
            epoch: 0,
            label: label.to_string(),
            _elem: PhantomData,
        }
    }

    /// Sets a hard ceiling `ensure_capacity` will never grow past — growth
    /// requests beyond it fail with [`CapacityError`] instead of silently
    /// clamping (a caller that hits this needs to know, not get a
    /// mysteriously undersized buffer).
    pub fn with_max_capacity(mut self, max_capacity: u32) -> Self {
        self.max_capacity = Some(max_capacity);
        self
    }

    fn alloc(device: &wgpu::Device, label: &str, capacity: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Ensures `capacity() >= min_capacity`, growing (doubling from the
    /// current capacity, or starting from 1 if it's currently 0, until the
    /// requirement is met) and GPU-to-GPU copying the buffer's existing
    /// contents into the new allocation if a reallocation is needed.
    ///
    /// Returns `Ok(true)` if a reallocation happened (bumping [`Self::epoch`]),
    /// `Ok(false)` if the existing capacity already sufficed, and
    /// `Err(CapacityError)` if `min_capacity` exceeds a configured
    /// [`Self::with_max_capacity`] ceiling — in the error case the buffer is
    /// left completely unchanged (no partial growth).
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        min_capacity: u32,
    ) -> Result<bool, CapacityError> {
        if min_capacity <= self.capacity {
            return Ok(false);
        }
        if let Some(max) = self.max_capacity {
            if min_capacity > max {
                return Err(CapacityError { requested: min_capacity, max });
            }
        }
        let mut new_capacity = self.capacity.max(1);
        while new_capacity < min_capacity {
            new_capacity = new_capacity.saturating_mul(2);
        }
        if let Some(max) = self.max_capacity {
            new_capacity = new_capacity.min(max);
        }

        let new_buf = Self::alloc(device, &self.label, new_capacity);
        // Copy the live prefix (the old buffer's full capacity — every row
        // up to `self.capacity`, since this type has no notion of which
        // rows within that range are actually "in use"; that's the
        // caller's concern, this just preserves bytes faithfully).
        let copy_bytes = self.capacity as u64 * std::mem::size_of::<T>() as u64;
        if copy_bytes > 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dynamic-gpu-buffer-grow-copy"),
            });
            encoder.copy_buffer_to_buffer(&self.buf, 0, &new_buf, 0, copy_bytes);
            queue.submit([encoder.finish()]);
        }

        self.buf = new_buf;
        self.capacity = new_capacity;
        self.epoch += 1;
        Ok(true)
    }

    /// Writes `data` starting at row `offset`. Panics if `offset + data.len()`
    /// exceeds the current capacity — call [`Self::ensure_capacity`] first if
    /// the write might not fit; this type never grows implicitly on write
    /// (growth needs `&mut self`/a device, `write` deliberately doesn't, to
    /// stay usable from contexts that only have a shared reference).
    pub fn write(&self, queue: &wgpu::Queue, offset: u32, data: &[T]) {
        assert!(
            offset as u64 + data.len() as u64 <= self.capacity as u64,
            "write [{offset}, +{}) exceeds capacity {} — call ensure_capacity first",
            data.len(),
            self.capacity
        );
        if !data.is_empty() {
            queue.write_buffer(&self.buf, offset as u64 * std::mem::size_of::<T>() as u64, super::as_bytes(data));
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Bumped by exactly one on every reallocation caused by
    /// [`Self::ensure_capacity`]. Compare against a previously-cached value
    /// to know whether a bind group built against [`Self::buffer`] needs
    /// rebuilding.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("dynamic-gpu-buffer-test"),
            ..Default::default()
        }))
        .expect("device")
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
    fn grow_preserves_existing_contents_and_bumps_epoch_exactly_once() {
        let (device, queue) = test_device();
        let mut buf: DynamicGpuBuffer<u32> = DynamicGpuBuffer::new(&device, "test", 4);
        assert_eq!(buf.capacity(), 4);
        assert_eq!(buf.epoch(), 0);

        buf.write(&queue, 0, &[10, 20, 30, 40]);

        // Grow past capacity -- must copy the existing 4 rows forward.
        let grew = buf.ensure_capacity(&device, &queue, 10).expect("grow");
        assert!(grew, "10 > capacity 4 -- must have reallocated");
        assert!(buf.capacity() >= 10);
        assert_eq!(buf.epoch(), 1);

        let bytes = readback(&device, &queue, buf.buffer(), 4 * 4);
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![10, 20, 30, 40], "grow must preserve existing rows");

        // A second call within the already-grown capacity must be a no-op.
        let grew_again = buf.ensure_capacity(&device, &queue, 8).expect("no-op grow");
        assert!(!grew_again, "8 <= already-grown capacity -- must not reallocate again");
        assert_eq!(buf.epoch(), 1, "epoch must not bump on a no-op ensure_capacity");

        // New rows are writable and readable after growth.
        buf.write(&queue, 4, &[50]);
        let bytes = readback(&device, &queue, buf.buffer(), 5 * 4);
        let last = u32::from_ne_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(last, 50);
    }

    #[test]
    fn max_capacity_is_a_hard_ceiling_not_a_silent_clamp() {
        let (device, queue) = test_device();
        let mut buf: DynamicGpuBuffer<u32> = DynamicGpuBuffer::new(&device, "test", 4).with_max_capacity(8);

        let err = buf
            .ensure_capacity(&device, &queue, 100)
            .expect_err("100 > max_capacity 8 -- must error, not clamp silently");
        assert_eq!(err.requested, 100);
        assert_eq!(err.max, 8);
        assert_eq!(buf.capacity(), 4, "a failed grow must leave the buffer completely unchanged");
        assert_eq!(buf.epoch(), 0);

        // A request within the ceiling still succeeds and is clamped to
        // exactly max_capacity when doubling would otherwise overshoot it.
        buf.ensure_capacity(&device, &queue, 6).expect("within ceiling");
        assert_eq!(buf.capacity(), 8, "doubling 4->8 exactly hits max_capacity");
    }
}
