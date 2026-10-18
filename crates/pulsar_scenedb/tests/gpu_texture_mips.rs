//! `TextureStore::register_mips`: mip chains and block-compressed formats go to
//! the GPU in the layout wgpu expects, each level readable back unchanged.

use pulsar_scenedb::gpu::{TextureError, TextureStore};

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    has_bc: bool,
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
    Some(Gpu { device, queue, has_bc })
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

#[test]
fn an_uncompressed_mip_chain_uploads_every_level() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    // 8x4 RGBA8: levels 8x4, 4x2, 2x1, 1x1.
    let desc = descriptor(8, 4, 4, wgpu::TextureFormat::Rgba8Unorm);
    let levels: Vec<Vec<u8>> = [(8, 4), (4, 2), (2, 1), (1, 1)]
        .iter()
        .enumerate()
        .map(|(level, (w, h))| pattern(level as u32, w * h * 4))
        .collect();
    let slices: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();

    let mut store = TextureStore::new(4);
    let slot = store.register_mips(&gpu.device, &gpu.queue, &desc, &slices).unwrap();
    let texture = store.texture(slot).unwrap();
    assert_eq!(texture.mip_level_count(), 4);
    for (level, expected) in levels.iter().enumerate() {
        assert_eq!(&read_level(&gpu, texture, level as u32), expected, "level {level}");
    }
    assert_eq!(store.upload_count(), 1, "the whole chain is one upload");
}

#[test]
fn a_block_compressed_chain_uploads_in_blocks_and_small_levels_round_up() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    if !gpu.has_bc {
        eprintln!("adapter has no TEXTURE_COMPRESSION_BC; skipping");
        return;
    }
    // 8x8 BC7: levels 8x8 (2x2 blocks), 4x4 (1), 2x2 (1, padded to 4x4), 1x1 (1).
    let desc = descriptor(8, 8, 4, wgpu::TextureFormat::Bc7RgbaUnorm);
    let levels: Vec<Vec<u8>> = [4usize, 1, 1, 1]
        .iter()
        .enumerate()
        .map(|(level, blocks)| pattern(level as u32, blocks * 16))
        .collect();
    let slices: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();

    let mut store = TextureStore::new(4);
    let slot = store.register_mips(&gpu.device, &gpu.queue, &desc, &slices).unwrap();
    let texture = store.texture(slot).unwrap();
    for (level, expected) in levels.iter().enumerate() {
        assert_eq!(&read_level(&gpu, texture, level as u32), expected, "level {level}");
    }
}

#[test]
fn a_bad_chain_is_refused_before_it_takes_a_slot() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let desc = descriptor(4, 4, 3, wgpu::TextureFormat::Rgba8Unorm);
    let good: Vec<Vec<u8>> = vec![vec![0; 64], vec![0; 16], vec![0; 4]];
    let mut store = TextureStore::new(4);

    let two: Vec<&[u8]> = good.iter().take(2).map(Vec::as_slice).collect();
    assert_eq!(
        store.register_mips(&gpu.device, &gpu.queue, &desc, &two),
        Err(TextureError::MipCountMismatch)
    );

    let mut wrong = good.clone();
    wrong[1] = vec![0; 15];
    let wrong: Vec<&[u8]> = wrong.iter().map(Vec::as_slice).collect();
    assert_eq!(
        store.register_mips(&gpu.device, &gpu.queue, &desc, &wrong),
        Err(TextureError::MipSizeMismatch)
    );

    assert_eq!(store.slot_count(), 0, "neither failure took a slot");
    let ok: Vec<&[u8]> = good.iter().map(Vec::as_slice).collect();
    assert_eq!(store.register_mips(&gpu.device, &gpu.queue, &desc, &ok), Ok(0));
}

#[test]
fn a_mip_texture_can_be_shared_between_stores() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let desc = descriptor(2, 2, 2, wgpu::TextureFormat::Rgba8Unorm);
    let (level0, level1) = (pattern(0, 16), pattern(1, 4));
    let mut first = TextureStore::new(2);
    let mut second = TextureStore::new(2);
    let slot = first
        .register_mips(&gpu.device, &gpu.queue, &desc, &[&level0, &level1])
        .unwrap();
    let shared = second.register_shared(first.texture(slot).unwrap().clone()).unwrap();
    assert_eq!(second.texture(shared).unwrap().mip_level_count(), 2);
    assert_eq!(read_level(&gpu, second.texture(shared).unwrap(), 1), level1);
    assert_eq!(second.upload_count(), 0);
}
