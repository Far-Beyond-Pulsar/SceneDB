//! Growable, variable-length-per-row GPU pool for `#[gpu]` fields whose
//! Rust type is `Vec<T>` (`T: Pod`) — e.g. a component storing its own
//! vertex array directly, instead of an index into a separate asset
//! registry. Two pieces work together, both reusing already-proven
//! mechanics rather than adding a new one:
//!
//! - A per-entity, row-indexed (`entity.index()`-keyed) handle table
//!   (`{offset, count}` — [`VarLenHandle`], itself a plain fixed-size `Pod`
//!   value). This is nothing new: it's the exact same
//!   `GrowableSceneBuffer<T>` mechanism every scalar `#[gpu]` field already
//!   uses, with `T = VarLenHandle`.
//! - ONE shared, growable pool (this module's [`VarLenGpuPool<T>`], built on
//!   [`super::DynamicGpuBuffer`]'s existing grow-and-copy mechanics) that
//!   every entity's variable-length payload is suballocated from, tracked by
//!   [`super::freelist::RangeList`] — the same allocator
//!   [`super::assets::GeometryArena`] already proved, shared rather than
//!   reimplemented (see that module for the extraction).
//!
//! Each write frees the entity's PREVIOUS allocation (if any — the vec's
//! length may differ from last time) before allocating fresh space, so the
//! pool never accumulates orphaned space from a shrinking/growing field.
//! Despawn (or overwriting with an empty `Vec`) frees the same way — see
//! `world_mirror.rs`'s despawn hook, the only other caller of
//! [`VarLenGpuPool::free_handle`] besides this module's own re-write path.
//!
//! No new `#[gpu(...)]` attribute syntax: a `Vec<T>`-typed `#[gpu]` field is
//! detected by its Rust type alone (`pulsar_scenedb_derive`'s codegen) and
//! routed through this pool automatically. Every OTHER `#[gpu]` field
//! (scalars, fixed-size arrays, structs) is completely unaffected — same
//! generated code, same performance, as before this module existed.

use super::dynamic_buffer::{CapacityError, DynamicGpuBuffer};
use super::freelist::RangeList;
use crate::page::Pod;
use std::any::Any;
use std::sync::{Arc, RwLock};

/// A `Pod`, per-entity-row handle into a [`VarLenGpuPool`] — what actually
/// lands in the row-indexed growable column for a `Vec<T>`-typed `#[gpu]`
/// field. `count == 0` means "no allocation" (an empty or never-written
/// `Vec`) — `offset` is meaningless in that case, never dereferenced.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct VarLenHandle {
    pub offset: u32,
    pub count: u32,
}

// SAFETY: #[repr(C)], two plain u32s, no padding, every bit pattern valid.
unsafe impl Pod for VarLenHandle {}

struct Inner<T: Pod> {
    buf: DynamicGpuBuffer<T>,
    free: RangeList,
}

/// Shared growable pool backing every entity's `Vec<T>`-typed `#[gpu]`
/// field for one (struct, field) column (or one shared `#[gpu(buffer =
/// "key")]` group, same sharing rule scalar `#[gpu]` fields already have).
///
/// `RwLock`-guarded for the same reason [`super::GrowableSceneBuffer`] is —
/// growth needs `&mut`, but the derive-generated dispatch only ever has
/// `&self` to call through (see that type's doc for the full rationale;
/// identical here).
pub struct VarLenGpuPool<T: Pod> {
    device: Arc<wgpu::Device>,
    inner: RwLock<Inner<T>>,
}

impl<T: Pod + Send + Sync + 'static> VarLenGpuPool<T> {
    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32) -> Self {
        Self::new_with_usage(device, label, initial_capacity, wgpu::BufferUsages::STORAGE)
    }

    /// Same as [`Self::new`], but with an explicit base buffer usage instead
    /// of the default `STORAGE` — for pools bound a different way than a
    /// `#[gpu] Vec<T>` field's usual shader-storage-buffer read (e.g. a
    /// fixed-function `VERTEX`/`INDEX` buffer). Every derive-generated
    /// `#[gpu] Vec<T>` field still goes through [`Self::new`] (`STORAGE`,
    /// unchanged) — this is for direct, non-derive callers like Helio's mesh
    /// pool that want the SAME pool/freelist mechanics with a different
    /// underlying buffer usage.
    pub fn new_with_usage(
        device: Arc<wgpu::Device>,
        label: &str,
        initial_capacity: u32,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buf = DynamicGpuBuffer::new_with_usage(&device, label, initial_capacity, usage);
        let free = RangeList::new(initial_capacity as u64);
        Self { device, inner: RwLock::new(Inner { buf, free }) }
    }

    /// Frees `prev`'s allocation (if any — `count == 0` is a no-op, so this
    /// is safe to call with a fresh entity's default handle) and, if `data`
    /// is non-empty, allocates fresh space and writes it, growing the
    /// backing buffer first if the freelist can't satisfy the request.
    /// Returns the new handle to store in the row-indexed handle table —
    /// `VarLenHandle::default()` (count 0) if `data` was empty.
    pub fn write_var_row(
        &self,
        queue: &wgpu::Queue,
        prev: VarLenHandle,
        data: &[T],
    ) -> Result<VarLenHandle, CapacityError> {
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        if prev.count > 0 {
            guard.free.free(prev.offset as u64, prev.count as u64);
        }
        if data.is_empty() {
            return Ok(VarLenHandle::default());
        }

        let len = data.len() as u64;
        let offset = match guard.free.alloc(len, 1) {
            Some(offset) => offset,
            None => {
                // Exhausted -- grow the backing buffer (existing
                // DynamicGpuBuffer grow-and-copy mechanics, not
                // reimplemented) then extend the freelist's tracked total
                // to match the ACTUAL new capacity (which may be larger
                // than the bare minimum requested -- `ensure_capacity`
                // doubles), and retry. Must succeed: the new capacity is
                // guaranteed >= old_total + len by construction.
                let old_total = guard.buf.capacity() as u64;
                let min_capacity = old_total.saturating_add(len).min(u32::MAX as u64) as u32;
                guard.buf.ensure_capacity(&self.device, queue, min_capacity)?;
                let new_total = guard.buf.capacity() as u64;
                guard.free.extend_total(old_total, new_total);
                guard
                    .free
                    .alloc(len, 1)
                    .expect("freelist must satisfy an allocation immediately after extending past it")
            }
        };

        guard.buf.write(queue, offset as u32, data);
        Ok(VarLenHandle { offset: offset as u32, count: data.len() as u32 })
    }

    /// Frees `handle`'s allocation without writing a replacement — the
    /// despawn/removal path (an entity going away, or its `Vec` field being
    /// removed outright, has nothing new to write, only old space to give
    /// back). No-op if `handle.count == 0`.
    pub fn free_handle(&self, handle: VarLenHandle) {
        if handle.count == 0 {
            return;
        }
        let mut guard = self.inner.write().expect("VarLenGpuPool lock poisoned");
        guard.free.free(handle.offset as u64, handle.count as u64);
    }

    pub fn epoch(&self) -> u64 {
        self.inner.read().expect("VarLenGpuPool lock poisoned").buf.epoch()
    }

    pub fn capacity(&self) -> u32 {
        self.inner.read().expect("VarLenGpuPool lock poisoned").buf.capacity()
    }

    /// Runs `f` against the pool's current `wgpu::Buffer` — the lock-safe
    /// way to reach it for bind-group construction. Do not stash the
    /// `&wgpu::Buffer` past this call (same contract as
    /// [`super::GrowableGpuBufferDispatch::with_buffer`]).
    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let guard = self.inner.read().expect("VarLenGpuPool lock poisoned");
        f(guard.buf.buffer());
    }

    /// Same buffer access as [`Self::with_buffer`], but as a held guard
    /// instead of a closure — for callers that need a real `&wgpu::Buffer`-
    /// shaped struct field (e.g. a per-frame bind-group descriptor threaded
    /// through several passes) rather than a callback. Holds the SAME read
    /// lock `with_buffer` takes for the guard's lifetime; a concurrent
    /// `write_var_row`/`free_handle` blocks until it's dropped, exactly like
    /// any other `RwLock` reader — callers should drop it once the frame's
    /// draw calls are recorded, not hold it indefinitely.
    pub fn read_buffer(&self) -> VarLenBufferRef<'_, T> {
        VarLenBufferRef(self.inner.read().expect("VarLenGpuPool lock poisoned"))
    }

    pub fn as_any(&self) -> &dyn Any
    where
        Self: 'static,
    {
        self
    }
}

/// A read-locked handle to a [`VarLenGpuPool`]'s current `wgpu::Buffer`.
/// `Deref`s straight to it, so `pool.read_buffer().slice(..)` etc. work the
/// same as if a bare `&wgpu::Buffer` had been borrowed.
pub struct VarLenBufferRef<'a, T: Pod>(std::sync::RwLockReadGuard<'a, Inner<T>>);

impl<'a, T: Pod> std::ops::Deref for VarLenBufferRef<'a, T> {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.0.buf.buffer()
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
            label: Some("var-len-gpu-pool-test"),
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
    fn write_then_readback_round_trips() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let handle = pool
            .write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3])
            .expect("fits initial capacity");
        assert_eq!(handle, VarLenHandle { offset: 0, count: 3 });

        let mut bytes = Vec::new();
        pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 3 * 4));
        let got: Vec<u32> = bytes.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![1, 2, 3]);
    }

    #[test]
    fn re_write_with_a_different_length_frees_the_old_slot_and_reuses_it() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 16);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3, 4]).unwrap();
        assert_eq!(h1.offset, 0);

        // Shrink: re-write with fewer elements -- must free [0,4) and
        // re-allocate a smaller [0,2) slot, not leak the tail.
        let h2 = pool.write_var_row(&queue, h1, &[9, 9]).unwrap();
        assert_eq!(h2, VarLenHandle { offset: 0, count: 2 }, "reuses the freed slot from the same offset");

        // A second entity's write must now be able to reuse [2,4) (freed by
        // the shrink above), proving the old tail wasn't orphaned.
        let h3 = pool.write_var_row(&queue, VarLenHandle::default(), &[7, 7]).unwrap();
        assert_eq!(h3, VarLenHandle { offset: 2, count: 2 });
    }

    #[test]
    fn exhaustion_grows_the_backing_buffer_transparently() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 2);
        assert_eq!(pool.capacity(), 2);

        // 10 elements doesn't fit the initial capacity of 2 -- must grow,
        // not fail.
        let handle = pool.write_var_row(&queue, VarLenHandle::default(), &(0..10).collect::<Vec<u32>>()).unwrap();
        assert_eq!(handle, VarLenHandle { offset: 0, count: 10 });
        assert!(pool.capacity() >= 10);

        let mut bytes = Vec::new();
        pool.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 10 * 4));
        let got: Vec<u32> = bytes.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn writing_an_empty_slice_frees_the_previous_allocation_and_allocates_nothing() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3]).unwrap();
        let h2 = pool.write_var_row(&queue, h1, &[]).unwrap();
        assert_eq!(h2, VarLenHandle::default(), "empty write yields the no-allocation sentinel");

        // The freed [0,3) must be fully reusable by a fresh write.
        let h3 = pool.write_var_row(&queue, VarLenHandle::default(), &[5, 5, 5]).unwrap();
        assert_eq!(h3.offset, 0);
    }

    #[test]
    fn free_handle_reclaims_space_with_no_replacement_write() {
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);

        let h1 = pool.write_var_row(&queue, VarLenHandle::default(), &[1, 2, 3, 4]).unwrap();
        pool.free_handle(h1);

        let h2 = pool.write_var_row(&queue, VarLenHandle::default(), &[9, 9, 9, 9]).unwrap();
        assert_eq!(h2.offset, 0, "the space freed by free_handle must be reusable");
    }

    #[test]
    fn read_buffer_derefs_to_the_same_buffer_with_buffer_reaches() {
        // Proves `read_buffer()` isn't a second, divergent path -- it must
        // observe the exact same bytes `with_buffer` (the pre-existing,
        // already-proven-correct accessor) sees, since both just borrow the
        // same lock-guarded `wgpu::Buffer`. This is the accessor a caller
        // like Helio's `MeshBuffers<'a>` (which needs a real `&wgpu::Buffer`-
        // shaped struct field, not a callback) will hold instead.
        let (device, queue) = test_device();
        let pool: VarLenGpuPool<u32> = VarLenGpuPool::new(Arc::clone(&device), "test", 8);
        pool.write_var_row(&queue, VarLenHandle::default(), &[11, 22, 33]).unwrap();

        let via_guard = readback(&device, &queue, &pool.read_buffer(), 3 * 4);
        let mut via_closure = Vec::new();
        pool.with_buffer(&mut |b| via_closure = readback(&device, &queue, b, 3 * 4));
        assert_eq!(via_guard, via_closure);

        let got: Vec<u32> =
            via_guard.chunks(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![11, 22, 33]);
    }
}
