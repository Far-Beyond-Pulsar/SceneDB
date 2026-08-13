//! Explicit, decoupled diagnostic VRAM readback (design "The `Once` path:
//! CPU asset handles, not VRAM readbacks", operational detail "Explicit VRAM
//! readback (decoupled, diagnostic-only)"):
//!
//! > Standard `get`, `get_mut`, and `Query` iteration operate **strictly on
//! > CPU storage**... They never read from VRAM. VRAM data readback is
//! > decoupled from `get()` and reserved for explicit, low-level GPU
//! > diagnostics or pipeline verification if ever required... where
//! > `device.poll(wgpu::Maintain::Wait)` is the caller's responsibility. It
//! > is never on the hot path and never invoked by the unified surface.
//!
//! Before this module, every real-device test in this crate hand-rolled the
//! same `copy_buffer_to_buffer` → `map_async` → `poll` → `get_mapped_range`
//! sequence in its own private `#[cfg(test)]` helper (over a dozen near-
//! identical copies across `dirty_tracked_scene_buffer.rs`,
//! `dynamic_buffer.rs`, `growable_scene_buffer.rs`, and the `tests/gpu_*.rs`
//! integration suite) — proof the pattern is real and correct, but with no
//! single production entry point a caller outside `#[cfg(test)]` could use.
//! This module is that one entry point: the ONLY sanctioned way to pull
//! bytes back from VRAM in this crate, named and documented as the
//! diagnostic-only operation it is. Nothing in the `get`/`get_mut`/query path
//! calls this, and it never will — see the module doc above.
//!
//! Blocking, by design: [`readback_bytes`] submits the copy and blocks the
//! calling thread on `device.poll(wgpu::PollType::wait_indefinitely())`
//! before returning. That is exactly the cost profile a *diagnostic* read
//! should have (correctness and simplicity over throughput) and exactly why
//! it must never be reachable from `get()`/`get_mut()`/query iteration, which
//! promise a synchronous, VRAM-free, system-RAM-only cost (see `scene_store`
//! module docs, `MirrorMode::Once`).

use std::ops::Range;

/// Copies `byte_range` out of `src` (a GPU-resident buffer) into a freshly
/// allocated `MAP_READ` staging buffer, submits the copy, blocks on
/// `device.poll(wgpu::PollType::wait_indefinitely())` until it completes, and
/// returns the bytes.
///
/// This is the crate's single sanctioned VRAM readback primitive — see the
/// module doc for why it exists and why it is never called from the
/// `get`/`get_mut`/query path. Every real-device test that previously
/// hand-rolled this sequence now calls this function instead.
///
/// # Panics
///
/// Panics if `byte_range` is empty, out of bounds of `src`, or the map/poll
/// fails (a diagnostic call is expected to succeed on a live device; a
/// caller wrapping this for production telemetry should catch panics or
/// pre-validate the range itself — this function does not soften failures
/// into a `Result`, matching every hand-rolled helper it replaces).
pub fn readback_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    byte_range: Range<u64>,
) -> Vec<u8> {
    let len = byte_range
        .end
        .checked_sub(byte_range.start)
        .expect("readback_bytes: range end must not precede start");
    assert!(len > 0, "readback_bytes: empty range");
    assert!(
        byte_range.end <= src.size(),
        "readback_bytes: range {byte_range:?} exceeds buffer size {}",
        src.size(),
    );

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scenedb-diagnostic-readback-staging"),
        size: len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("scenedb-diagnostic-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(src, byte_range.start, &staging, 0, len);
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("readback_bytes: map_async failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("readback_bytes: device.poll failed");
    let data = slice
        .get_mapped_range()
        .expect("readback_bytes: get_mapped_range failed")
        .to_vec();
    staging.unmap();
    data
}

/// Convenience over [`readback_bytes`] for a `T: Pod` row buffer: reads back
/// exactly `std::mem::size_of::<T>()` bytes starting at `row *
/// size_of::<T>()` and reinterprets them as `T`.
///
/// Still fully decoupled/diagnostic-only (see module doc) — this is
/// `readback_bytes` plus the row-arithmetic every hand-rolled per-file
/// `readback_u32`/`readback_row` helper duplicated, not a new code path into
/// VRAM.
///
/// # Panics
///
/// Same as [`readback_bytes`], plus if the returned byte count doesn't
/// exactly match `size_of::<T>()` (defensive; cannot happen given the
/// `byte_range` this constructs, but guards against a future
/// `readback_bytes` change silently under- or over-reading).
pub fn readback_row<T: crate::page::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    row: u32,
) -> T {
    let stride = std::mem::size_of::<T>() as u64;
    let start = row as u64 * stride;
    let bytes = readback_bytes(device, queue, src, start..start + stride);
    assert_eq!(bytes.len() as u64, stride, "readback_row: unexpected byte count");
    // SAFETY: `T: Pod` guarantees every bit pattern is valid, `bytes.len()`
    // was just asserted to equal `size_of::<T>()`, and `Vec<u8>`'s allocation
    // has no alignment guarantee stronger than `T`'s own — so read via
    // `ptr::read_unaligned` rather than a reference cast.
    unsafe { (bytes.as_ptr() as *const T).read_unaligned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device_queue() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter — GPU tests need a local GPU");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("scenedb-readback-test"),
            ..Default::default()
        }))
        .expect("device")
    }

    #[test]
    fn readback_bytes_round_trips_a_written_range() {
        let (device, queue) = test_device_queue();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let payload: [u8; 16] = [7; 16];
        queue.write_buffer(&buf, 16, &payload);
        let got = readback_bytes(&device, &queue, &buf, 16..32);
        assert_eq!(got, payload.to_vec());
    }

    #[test]
    fn readback_bytes_reads_only_the_requested_range() {
        let (device, queue) = test_device_queue();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &[1u8; 8]);
        queue.write_buffer(&buf, 8, &[2u8; 8]);
        let got = readback_bytes(&device, &queue, &buf, 8..16);
        assert_eq!(got, vec![2u8; 8]);
    }

    #[test]
    #[should_panic(expected = "exceeds buffer size")]
    fn readback_bytes_rejects_an_out_of_bounds_range() {
        let (device, queue) = test_device_queue();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        readback_bytes(&device, &queue, &buf, 0..32);
    }

    #[test]
    #[should_panic(expected = "empty range")]
    fn readback_bytes_rejects_an_empty_range() {
        let (device, queue) = test_device_queue();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        readback_bytes(&device, &queue, &buf, 4..4);
    }

    #[test]
    fn readback_row_reinterprets_the_row_bytes_as_t() {
        let (device, queue) = test_device_queue();
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("src"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &11u32.to_ne_bytes());
        queue.write_buffer(&buf, 4, &22u32.to_ne_bytes());
        queue.write_buffer(&buf, 8, &33u32.to_ne_bytes());
        assert_eq!(readback_row::<u32>(&device, &queue, &buf, 0), 11);
        assert_eq!(readback_row::<u32>(&device, &queue, &buf, 1), 22);
        assert_eq!(readback_row::<u32>(&device, &queue, &buf, 2), 33);
    }
}
