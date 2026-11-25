//! `TextureStore::register_mips_staged`: a texture written a few rows at a time
//! under a budget reads back exactly what `register_mips` would have written.

use pulsar_scenedb::gpu::{MipSource, TextureStore, UploadBudget};
use std::sync::Arc;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let has_bc = adapter
        .features()
        .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    let required_features = if has_bc {
        wgpu::Features::TEXTURE_COMPRESSION_BC
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("texture-mips-test"),
        required_features,
        ..Default::default()
    }))
    .ok()?;
    Some(Gpu { device, queue })
}

fn descriptor(
    width: u32,
    height: u32,
    mips: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("mips-test"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

/// Level `level` of `texture` as tightly packed rows of blocks.
fn read_level(gpu: &Gpu, texture: &wgpu::Texture, level: u32) -> Vec<u8> {
    let format = texture.format();
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap();
    let logical = wgpu::Extent3d {
        width: (texture.width() >> level).max(1),
        height: (texture.height() >> level).max(1),
        depth_or_array_layers: 1,
    };
    let physical = logical.physical_size(format);
    let blocks_wide = physical.width / block_width;
    let rows = physical.height / block_height;
    let unpadded = blocks_wide * block_size;
    let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mips-readback"),
        size: u64::from(padded * rows),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(rows),
            },
        },
        physical,
    );
    gpu.queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let mapped = staging.slice(..).get_mapped_range().expect("mapped range").to_vec();
    let mut packed = Vec::with_capacity((unpadded * rows) as usize);
    for row in 0..rows {
        let start = (row * padded) as usize;
        packed.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    packed
}

/// A level's worth of recognisable bytes.
fn pattern(level: u32, length: usize) -> Vec<u8> {
    (0..length).map(|i| (i as u32 * 7 + level * 41 + 3) as u8).collect()
}

struct Levels(Vec<Vec<u8>>);

impl MipSource for Levels {
    fn level_count(&self) -> u32 {
        self.0.len() as u32
    }
    fn level(&self, level: u32) -> &[u8] {
        &self.0[level as usize]
    }
}

fn rgba_chain(width: u32, height: u32) -> (wgpu::TextureDescriptor<'static>, Arc<Levels>) {
    let mips = 32 - width.max(height).leading_zeros();
    let desc = descriptor(width, height, mips, wgpu::TextureFormat::Rgba8Unorm);
    let levels = (0..mips)
        .map(|level| {
            let (w, h) = ((width >> level).max(1), (height >> level).max(1));
            pattern(level, (w * h * 4) as usize)
        })
        .collect();
    (desc, Arc::new(Levels(levels)))
}

#[test]
fn a_staged_texture_arrives_in_bands_and_reads_back_whole() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // 64x64: level 0 alone is 16 KB, 256 bytes a row.
    let (desc, source) = rgba_chain(64, 64);
    let mut store = TextureStore::new(4);
    let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = completed.clone();
    let slot = store
        .register_mips_staged(
            &gpu.device,
            &desc,
            source.clone(),
            None,
            7,
            Some(Box::new(move |_| flag.store(true, std::sync::atomic::Ordering::SeqCst))),
        )
        .unwrap();
    assert!(store.texture(slot).is_none(), "vacant until whole");
    assert!(store.has_staged_uploads());

    let mut frames = 0;
    while store.has_staged_uploads() {
        let report = store.advance_uploads(&gpu.queue, UploadBudget { bytes: 4096, time: None }, None);
        assert!(report.bytes > 0, "every call makes progress");
        // A band is whole rows: never much past the budget.
        assert!(report.bytes <= 4096 + 256, "{} bytes in one frame", report.bytes);
        frames += 1;
        assert!(frames < 200, "the upload never finished");
    }
    assert!(frames >= 4, "16 KB at 4 KB a frame took {frames} frames");
    assert!(completed.load(std::sync::atomic::Ordering::SeqCst), "the completion ran");
    let texture = store.texture(slot).expect("whole now");
    for (level, expected) in source.0.iter().enumerate() {
        assert_eq!(&read_level(&gpu, texture, level as u32), expected, "level {level}");
    }
    assert_eq!(store.upload_count(), 1);
}

#[test]
fn a_replacing_upload_keeps_the_old_texture_until_it_is_whole() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (small_desc, small) = rgba_chain(8, 8);
    let mut store = TextureStore::new(4);
    let slot = store
        .register_mips_staged(&gpu.device, &small_desc, small.clone(), None, 1, None)
        .unwrap();
    store.advance_uploads(&gpu.queue, UploadBudget { bytes: usize::MAX, time: None }, None);
    let coarse = store.texture(slot).unwrap().clone();
    assert_eq!(coarse.width(), 8);

    let (big_desc, big) = rgba_chain(64, 64);
    let again = store
        .register_mips_staged(&gpu.device, &big_desc, big.clone(), Some(slot), 1, None)
        .unwrap();
    assert_eq!(again, slot, "the same slot");
    store.advance_uploads(&gpu.queue, UploadBudget { bytes: 2048, time: None }, None);
    assert!(store.has_staged_uploads());
    assert_eq!(store.texture(slot).unwrap().width(), 8, "still the coarse one");
    while store.has_staged_uploads() {
        store.advance_uploads(&gpu.queue, UploadBudget { bytes: 2048, time: None }, None);
    }
    assert_eq!(store.texture(slot).unwrap().width(), 64, "swapped in whole");
    assert_eq!(&read_level(&gpu, store.texture(slot).unwrap(), 0), &big.0[0]);
}

#[test]
fn the_nearest_upload_is_written_first_and_leaving_cancels() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (desc, source) = rgba_chain(16, 16);
    let mut store = TextureStore::new(4);
    let far = store.register_mips_staged(&gpu.device, &desc, source.clone(), None, 10, None).unwrap();
    let near = store.register_mips_staged(&gpu.device, &desc, source.clone(), None, 20, None).unwrap();
    // Key 20 is nearer than key 10.
    let rank = |key: u128| if key == 20 { 1.0 } else { 9.0 };
    let report = store.advance_uploads(
        &gpu.queue,
        UploadBudget { bytes: 1, time: None },
        Some(&rank),
    );
    assert_eq!(report.finished, 0);
    // Finish only what the nearest one needs: it is the one that completes first.
    while store.texture(near).is_none() {
        store.advance_uploads(&gpu.queue, UploadBudget { bytes: 256, time: None }, Some(&rank));
    }
    assert!(store.texture(far).is_none(), "the far one waits its turn");

    store.unregister(far).expect("an unfinished upload can be cancelled");
    assert!(!store.has_staged_uploads());
}

#[test]
fn a_bad_source_is_refused_before_a_slot_is_taken() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (desc, source) = rgba_chain(8, 8);
    let mut short = source.0.clone();
    short[0].pop();
    let mut store = TextureStore::new(1);
    assert!(store
        .register_mips_staged(&gpu.device, &desc, Arc::new(Levels(short)), None, 1, None)
        .is_err());
    // The one slot is still free.
    assert!(store.register_mips_staged(&gpu.device, &desc, source, None, 1, None).is_ok());
}
