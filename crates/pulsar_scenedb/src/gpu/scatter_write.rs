//! GPU-side scatter-write compute pass (SceneDB#39).
//!
//! [`super::DirtyTrackedSceneBuffer::flush`]'s run-coalescing only reduces
//! `queue.write_buffer` call count when dirty rows are *adjacent*. Real
//! churn workloads (entities despawning/respawning keyed by entity index,
//! not spatial or temporal locality) routinely dirty rows scattered roughly
//! uniformly across the whole capacity — for which coalescing produces
//! close to one run per dirty row, i.e. one `queue.write_buffer` call per
//! row, no better than writing immediately. Measured, not assumed: isolating
//! despawn/insert cost from flush cost at 100k entities/10% churn found
//! despawn+insert together cost under 1% of the per-frame total; the other
//! 99%+ was `flush_gpu_mirror` itself, at ~28ms/frame for exactly this
//! reason (roughly 20,000 individual `write_buffer` calls across two
//! deferred columns, each costing the ~1.4µs/call `write_buffer` itself
//! already measured to cost — see `world_mirror_scale.rs`'s Scenario 4).
//!
//! This moves the scatter itself onto the GPU: every dirty (row, value) pair
//! is packed into two flat, contiguous host-side arrays — `indices` (one
//! `u32` per dirty row) and `values` (that row's element, as `u32` words) —
//! uploaded via exactly two `queue.write_buffer` calls regardless of how
//! many rows are dirty or how scattered they are, then one compute dispatch
//! writes each value to `destination[indices[i]]`. Host-side cost becomes
//! O(dirty count) of cheap CPU packing plus a small, *constant* number of
//! GPU calls, instead of O(dirty count) GPU calls.
//!
//! # One pipeline, every element type
//!
//! The shader has no notion of `T` — it treats every buffer as `array<u32>`
//! and copies a runtime `words_per_element` count per invocation. This
//! requires `size_of::<T>()` to be a multiple of 4 bytes, true for every
//! `Pod` type actually used through the World-mirror path in this crate
//! (`u32`, `[f32; 16]`, and the derive's packed structs — all-`#[gpu]`-field
//! aggregates of such types). [`ScatterWritePipeline::new`] doesn't assert
//! this itself (it has no `T` in scope); [`super::DirtyTrackedSceneBuffer::new`]
//! does, at construction, so a misuse fails immediately and loudly rather
//! than corrupting unrelated words at flush time.
const SHADER_SRC: &str = r#"
struct Params {
    words_per_element: u32,
    dirty_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> destination: array<u32>;

@compute @workgroup_size(64)
fn scatter_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.dirty_count) {
        return;
    }
    let dest_row = indices[i];
    let src_base = i * params.words_per_element;
    let dst_base = dest_row * params.words_per_element;
    for (var w: u32 = 0u; w < params.words_per_element; w = w + 1u) {
        destination[dst_base + w] = values[src_base + w];
    }
}
"#;

/// One compute pipeline + its small persistent uniform buffer. Cheap to own
/// per [`super::DirtyTrackedSceneBuffer`] (created once at column
/// registration, not per flush) — not shared globally across columns, to
/// avoid cross-device/cross-column lifetime plumbing for a cost that's
/// already amortized over the column's entire lifetime.
pub struct ScatterWritePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// 8 bytes (`words_per_element: u32, dirty_count: u32`), rewritten via
    /// `queue.write_buffer` before every dispatch — one persistent buffer
    /// reused forever, not recreated per call.
    params_buf: wgpu::Buffer,
}

impl ScatterWritePipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scenedb-scatter-write-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scenedb-scatter-write-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scenedb-scatter-write-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scenedb-scatter-write-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_write"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scenedb-scatter-write-params"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, bind_group_layout, params_buf }
    }

    /// Writes `indices[0..dirty_count]` (one `u32` per dirty row) to
    /// `destination`, `words_per_element` words at a time, reading each
    /// row's data from `values[0..dirty_count * words_per_element]` in the
    /// same order as `indices`. Caller owns growing `indices`/`values` to
    /// fit `dirty_count`, and `destination` to cover every row named in
    /// `indices` — this type only knows about buffers and counts, not row/
    /// capacity semantics.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        indices: &wgpu::Buffer,
        values: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        words_per_element: u32,
        dirty_count: u32,
    ) {
        if dirty_count == 0 {
            return;
        }
        let mut params = [0u8; 8];
        params[0..4].copy_from_slice(&words_per_element.to_ne_bytes());
        params[4..8].copy_from_slice(&dirty_count.to_ne_bytes());
        queue.write_buffer(&self.params_buf, 0, &params);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scenedb-scatter-write-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: destination.as_entire_binding() },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scenedb-scatter-write-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scenedb-scatter-write-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = dirty_count.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit([encoder.finish()]);
    }
}
