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

/// The element-count alignment that guarantees a `T`-element index's BYTE
/// address (`index * size_of::<T>()`) is always a multiple of
/// `wgpu::COPY_BUFFER_ALIGNMENT` (4) — e.g. 4 for a 1-byte `T`, 2 for a
/// 2-byte `T`, 1 (no constraint at all) for any `T` whose own size is
/// already a multiple of 4, which covers every `#[gpu]` field type that
/// existed before `T` was allowed to be an arbitrary `Copy` type.
///
/// Exists because `wgpu::Queue::write_buffer` requires BOTH the destination
/// offset and the byte length of every write to be 4-byte aligned, and
/// `pulsar_world_registry::GpuRepr<T>` deliberately supports `T` smaller
/// than 4 bytes (`bool`) or whose size doesn't evenly divide 4. Padding the
/// WRITE SIZE alone ([`DynamicGpuBuffer::write_padded`]) only helps once
/// every row/allocation already starts on a 4-byte boundary — this is the
/// piece that guarantees that. [`super::var_len_pool::VarLenGpuPool`] is
/// the one caller that needs it, allocating every row's element count
/// rounded up to a multiple of this value.
pub(crate) fn elem_align<T>() -> u64 {
    let size = std::mem::size_of::<T>().max(1) as u64;
    4 / gcd(size, 4)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod elem_align_tests {
    use super::elem_align;

    #[test]
    fn matches_hand_worked_cases() {
        assert_eq!(elem_align::<u8>(), 4, "1-byte T needs 4 elements to guarantee a 4-byte-aligned byte address");
        assert_eq!(elem_align::<[u8; 2]>(), 2, "2-byte T needs 2 elements");
        assert_eq!(elem_align::<[u8; 3]>(), 4, "3-byte T: gcd(3,4)=1, needs 4 elements (12 bytes)");
        assert_eq!(elem_align::<u32>(), 1, "4-byte T is already always aligned — no padding needed");
        assert_eq!(elem_align::<[u8; 16]>(), 1, "any multiple of 4 is already always aligned");
    }
}

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
    /// Usage flags every (re)allocation is created with, always implicitly
    /// carrying `COPY_DST | COPY_SRC` (load-bearing for `write`/grow-copy —
    /// not caller-configurable). Defaults to `STORAGE` via [`Self::new`],
    /// matching every existing caller's behavior exactly; override with
    /// [`Self::with_usage`] for buffers bound a different way (e.g. a
    /// fixed-function `VERTEX`/`INDEX` buffer that has no need to also be
    /// `STORAGE`-bindable).
    usage: wgpu::BufferUsages,
    _elem: PhantomData<T>,
}

impl<T: Pod> DynamicGpuBuffer<T> {
    pub fn new(device: &wgpu::Device, label: &str, initial_capacity: u32) -> Self {
        Self::new_with_usage(device, label, initial_capacity, wgpu::BufferUsages::STORAGE)
    }

    /// Same as [`Self::new`], but with an explicit base usage instead of the
    /// default `STORAGE` — e.g. `VERTEX`/`INDEX` for a buffer that's only
    /// ever bound through the fixed-function vertex/index pipeline stages,
    /// not read by a shader as a storage buffer. `COPY_DST | COPY_SRC` are
    /// always added on top (grow-and-copy and `write` depend on them
    /// unconditionally); no need to include them here.
    pub fn new_with_usage(
        device: &wgpu::Device,
        label: &str,
        initial_capacity: u32,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buf = Self::alloc(device, label, initial_capacity, usage);
        Self {
            buf,
            capacity: initial_capacity,
            max_capacity: None,
            epoch: 0,
            label: label.to_string(),
            usage,
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

    fn alloc(device: &wgpu::Device, label: &str, capacity: u32, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * std::mem::size_of::<T>() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
    /// `Err(CapacityError)` if `min_capacity` exceeds the *effective* ceiling
    /// — the smaller of a configured [`Self::with_max_capacity`] and the
    /// device's own `Limits::max_buffer_size` — in the error case the
    /// buffer is left completely unchanged (no partial growth).
    ///
    /// The device-limit half of that ceiling exists because, without it,
    /// growth would keep doubling past what the device can actually
    /// allocate and `device.create_buffer` would reject the request via a
    /// `wgpu` *validation-error panic* — not a catchable `Result` at all.
    /// Confirmed reachable in practice, not a theoretical concern: a
    /// 256-byte packed row hits the default 256 MiB `max_buffer_size` at
    /// 1,048,576 rows in one buffer (Helio#211's benchmark harness). This
    /// check converts that hard panic into the same `CapacityError` an
    /// explicit `max_capacity` ceiling already produces — callers that
    /// already handle `CapacityError` (including
    /// `gpu::world_mirror::write_gpu_columns_at_row`'s panic-with-context
    /// path for World-mirrored columns) get a clearer, SceneDB-authored
    /// error instead of an opaque `wgpu` one, even when they never set an
    /// explicit `max_capacity` at all.
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        min_capacity: u32,
    ) -> Result<bool, CapacityError> {
        if min_capacity <= self.capacity {
            return Ok(false);
        }
        let device_limit_rows =
            (device.limits().max_buffer_size / std::mem::size_of::<T>().max(1) as u64) as u32;
        let effective_max = match self.max_capacity {
            Some(max) => max.min(device_limit_rows),
            None => device_limit_rows,
        };
        if min_capacity > effective_max {
            return Err(CapacityError { requested: min_capacity, max: effective_max });
        }
        let mut new_capacity = self.capacity.max(1);
        while new_capacity < min_capacity {
            new_capacity = new_capacity.saturating_mul(2);
        }
        new_capacity = new_capacity.min(effective_max);

        let new_buf = Self::alloc(device, &self.label, new_capacity, self.usage);
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

    /// Grows to at least `capacity` right now, if it doesn't already have
    /// that much — the exact same reallocation [`Self::ensure_capacity`]
    /// performs lazily on a crossing write, just performed on the caller's
    /// own schedule instead of whichever write happens to trigger it. Call
    /// ahead of a known-size batch operation (streaming in a sublevel with
    /// M known objects, spawning a wave of enemies from a design-time-known
    /// count) to move the reallocation cost off any per-frame critical
    /// path. A thin alias for `ensure_capacity` — kept as a separate,
    /// intention-revealing name because "ensure capacity because I'm about
    /// to need it right now, at this exact call site" and "reserve capacity
    /// ahead of a batch I haven't started yet" are different caller intents
    /// even though the mechanism is identical.
    pub fn reserve(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        self.ensure_capacity(device, queue, capacity)?;
        Ok(())
    }

    /// Shrinks to the smallest capacity that still covers
    /// `(highest_live_row + 1) * slack_factor`, rounded up — a no-op
    /// (returns `false`) if the current capacity is already at or below
    /// that target. `highest_live_row` must come from the caller (this
    /// type has no notion of which rows are "live" vs. stale-but-allocated
    /// — that's `World`'s/the caller's concern, via `entity_slots`/liveness
    /// tracking, not this buffer's). `slack_factor` should be `>= 1.0`
    /// (e.g. `1.5` leaves 50% headroom before the *next* growth is needed);
    /// passing exactly `1.0` shrinks as tight as possible, trading VRAM for
    /// a higher chance of needing to grow again soon.
    ///
    /// Reallocates as a real GPU-to-GPU copy of the surviving prefix, same
    /// cost profile as growth — call this at a natural boundary (a level
    /// unload, a large despawn batch settling), not every frame.
    pub fn shrink_to_fit(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        debug_assert!(slack_factor >= 1.0, "slack_factor < 1.0 would shrink below the caller's own stated live-row watermark");
        let target = (((highest_live_row as u64 + 1) as f64 * slack_factor.max(1.0) as f64).ceil() as u64).min(u32::MAX as u64) as u32;
        if target >= self.capacity {
            return false;
        }
        let new_buf = Self::alloc(device, &self.label, target, self.usage);
        // Only the surviving prefix needs copying -- rows past `target`
        // are, by the caller's own claim (`highest_live_row`), not live,
        // so their bytes are not owed any preservation guarantee.
        let copy_bytes = target as u64 * std::mem::size_of::<T>() as u64;
        if copy_bytes > 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dynamic-gpu-buffer-shrink-copy"),
            });
            encoder.copy_buffer_to_buffer(&self.buf, 0, &new_buf, 0, copy_bytes);
            queue.submit([encoder.finish()]);
        }
        self.buf = new_buf;
        self.capacity = target;
        self.epoch += 1;
        true
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

    /// Same as [`Self::write`], but tolerates `data`'s byte length not being
    /// a multiple of `wgpu::COPY_BUFFER_ALIGNMENT` (4) — required once `T`
    /// is allowed to be ANY `Copy` type (`pulsar_world_registry::GpuRepr<T>`
    /// wraps it for exactly this reason), including ones smaller than 4
    /// bytes or whose size doesn't evenly divide 4 (a `GpuRepr<bool>` is 1
    /// byte). `wgpu::Queue::write_buffer` rejects a write whose byte length
    /// isn't 4-aligned outright — this pads the write with zero bytes up to
    /// the next 4-byte boundary before issuing it. The padding bytes' VALUE
    /// never matters (nothing reads past the caller's own logical element
    /// count — see [`super::var_len_pool::VarLenHandle::count`], which
    /// always holds the true count, never the padded one).
    ///
    /// This only fixes the SIZE half of `write_buffer`'s alignment
    /// requirement. Callers MUST separately ensure `offset`'s byte address
    /// (`offset * size_of::<T>()`) is itself a multiple of 4 — a 3-byte
    /// write at byte offset 5 is just as invalid as one at offset 4, for a
    /// different reason this method has no way to detect on its own. See
    /// [`elem_align`] — [`super::var_len_pool::VarLenGpuPool`] (the one
    /// caller that needs this) allocates every row on an `elem_align`-
    /// element boundary specifically so every offset it passes here is
    /// already byte-aligned.
    ///
    /// # Panics
    ///
    /// If the padded write would extend past this buffer's actual
    /// allocation (`capacity * size_of::<T>()` bytes) — i.e. the caller
    /// must reserve capacity for the PADDED element count, not just the
    /// logical one, or this has no safe room to round up into. Same
    /// "call `ensure_capacity` first" contract as [`Self::write`].
    pub fn write_padded(&self, queue: &wgpu::Queue, offset: u32, data: &[T]) {
        assert!(
            offset as u64 + data.len() as u64 <= self.capacity as u64,
            "write [{offset}, +{}) exceeds capacity {} — call ensure_capacity first",
            data.len(),
            self.capacity
        );
        if data.is_empty() {
            return;
        }
        let elem_size = std::mem::size_of::<T>() as u64;
        let byte_offset = offset as u64 * elem_size;
        let bytes = super::as_bytes(data);
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let padded_len = (bytes.len() as u64).next_multiple_of(align);
        if padded_len == bytes.len() as u64 {
            // The overwhelmingly common case (any T whose size is already a
            // multiple of 4) — no padding needed, identical to `write`.
            queue.write_buffer(&self.buf, byte_offset, bytes);
            return;
        }
        debug_assert_eq!(
            byte_offset % align,
            0,
            "write_padded requires a 4-byte-aligned offset — see this method's own doc"
        );
        let buffer_end_bytes = self.capacity as u64 * elem_size;
        assert!(
            byte_offset + padded_len <= buffer_end_bytes,
            "write_padded [{byte_offset}, +{padded_len}) exceeds this buffer's real allocation \
             ({buffer_end_bytes} bytes) — the caller must reserve capacity padded to a 4-byte \
             multiple, not just the logical element count"
        );
        let mut padded = vec![0u8; padded_len as usize];
        padded[..bytes.len()].copy_from_slice(bytes);
        queue.write_buffer(&self.buf, byte_offset, &padded);
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

    #[test]
    fn device_max_buffer_size_is_a_real_ceiling_not_a_hard_panic() {
        let (device, queue) = test_device();
        // No explicit max_capacity -- this is exactly the World-mirror
        // default (register_gpu_columns_growable never sets one). Request
        // something absurdly larger than any real device's max_buffer_size
        // (u32 elements at 4 bytes each -> ~16 TiB) -- must come back as a
        // normal Err, never a wgpu validation panic.
        let mut buf: DynamicGpuBuffer<u32> = DynamicGpuBuffer::new(&device, "test", 4);
        let device_limit_rows = device.limits().max_buffer_size / 4;
        let err = buf
            .ensure_capacity(&device, &queue, u32::MAX)
            .expect_err("must error via the device's own max_buffer_size, not panic inside wgpu");
        assert_eq!(err.requested, u32::MAX);
        assert_eq!(err.max as u64, device_limit_rows.min(u32::MAX as u64));
        assert_eq!(buf.capacity(), 4, "a failed grow must leave the buffer completely unchanged");
    }

    #[test]
    fn reserve_grows_ahead_of_any_write_and_is_a_true_no_op_once_satisfied() {
        let (device, queue) = test_device();
        let mut buf: DynamicGpuBuffer<u32> = DynamicGpuBuffer::new(&device, "test", 4);
        buf.reserve(&device, &queue, 1000).expect("reserve");
        assert!(buf.capacity() >= 1000);
        let epoch_after_reserve = buf.epoch();

        // Writes within the reserved capacity must never trigger growth.
        buf.write(&queue, 999, &[42]);
        assert_eq!(buf.epoch(), epoch_after_reserve, "a write within reserved capacity must not grow");

        // A second reserve for less than what's already there is a no-op.
        buf.reserve(&device, &queue, 10).expect("no-op reserve");
        assert_eq!(buf.epoch(), epoch_after_reserve);
    }

    #[test]
    fn shrink_to_fit_reclaims_capacity_and_preserves_the_surviving_prefix() {
        let (device, queue) = test_device();
        let mut buf: DynamicGpuBuffer<u32> = DynamicGpuBuffer::new(&device, "test", 4);
        buf.ensure_capacity(&device, &queue, 1000).expect("grow to 1000+");
        let grown_capacity = buf.capacity();
        buf.write(&queue, 0, &[111, 222, 333]);

        let shrank = buf.shrink_to_fit(&device, &queue, 2, 1.0); // highest live row = 2 (rows 0,1,2 survive), no slack
        assert!(shrank, "capacity was {grown_capacity}, well above the shrink target");
        assert_eq!(buf.capacity(), 3, "target = (highest_live_row + 1) * slack_factor = 3 * 1.0");

        let bytes = readback(&device, &queue, buf.buffer(), 3 * 4);
        let values: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(values, vec![111, 222, 333], "shrink must preserve the surviving prefix's bytes");

        // Already at (or below) the target -- a true no-op.
        assert!(!buf.shrink_to_fit(&device, &queue, 2, 1.0));
    }

    #[test]
    fn new_with_usage_sets_the_requested_usage_and_survives_growth() {
        let (device, queue) = test_device();
        // VERTEX instead of the default STORAGE -- the shape Helio's mesh
        // pool needs (bound only via the fixed-function vertex stage, never
        // read as a shader storage buffer).
        let mut buf: DynamicGpuBuffer<u32> =
            DynamicGpuBuffer::new_with_usage(&device, "test", 4, wgpu::BufferUsages::VERTEX);
        assert!(buf.buffer().usage().contains(wgpu::BufferUsages::VERTEX));
        assert!(!buf.buffer().usage().contains(wgpu::BufferUsages::STORAGE));
        // COPY_DST/COPY_SRC are always implicitly present -- write/grow rely on them.
        assert!(buf.buffer().usage().contains(wgpu::BufferUsages::COPY_DST));
        assert!(buf.buffer().usage().contains(wgpu::BufferUsages::COPY_SRC));

        // Reallocation (on growth) must carry the same requested usage
        // forward, not silently fall back to the STORAGE default.
        buf.ensure_capacity(&device, &queue, 100).expect("grow");
        assert!(buf.buffer().usage().contains(wgpu::BufferUsages::VERTEX));
        assert!(!buf.buffer().usage().contains(wgpu::BufferUsages::STORAGE));
    }
}
