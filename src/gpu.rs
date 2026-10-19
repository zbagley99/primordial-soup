//! GPU-resident Fluoddity studio simulation. Reference equations are MIT licensed.
use crate::{
    config::{Project, WorldSettings},
    coords::Sizing,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    num::NonZeroU64,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
#[cfg(not(target_arch = "wasm32"))]
pub mod audit;
pub const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const VECTOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
pub const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct RgbaReadback {
    pub buffer: wgpu::Buffer,
    pub size: [u32; 2],
    stride: u32,
}
impl RgbaReadback {
    /// Depad the mapped region directly into its one PNG pixel buffer.
    pub fn rgba(&self, mapped: &[u8]) -> Result<Vec<u8>> {
        ensure!(
            mapped.len() as u64 == self.buffer.size(),
            "Incomplete screenshot mapping"
        );
        let row = self.size[0] as usize * 4;
        let mut pixels = Vec::with_capacity(row * self.size[1] as usize);
        for source in mapped.chunks_exact(self.stride as usize) {
            pixels.extend_from_slice(&source[..row]);
        }
        Ok(pixels)
    }
}

/// Read back only the selected physical-pixel region, without a full-frame CPU copy.
pub fn readback_rgba(
    d: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    origin: [u32; 2],
    size: [u32; 2],
) -> Result<RgbaReadback> {
    ensure!(
        matches!(
            texture.format(),
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        ),
        "Screenshot requires an RGBA8 texture"
    );
    ensure!(
        texture.usage().contains(wgpu::TextureUsages::COPY_SRC) && texture.sample_count() == 1,
        "Screenshot texture cannot be copied"
    );
    ensure!(
        size[0] > 0 && size[1] > 0,
        "Screenshot dimensions must be positive"
    );
    let x = origin[0]
        .checked_add(size[0])
        .context("Screenshot crop overflow")?;
    let y = origin[1]
        .checked_add(size[1])
        .context("Screenshot crop overflow")?;
    ensure!(
        x <= texture.width() && y <= texture.height(),
        "Screenshot crop exceeds texture"
    );
    let row = size[0].checked_mul(4).context("Screenshot row overflow")?;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let stride = row
        .checked_add(alignment - 1)
        .context("Screenshot stride overflow")?
        / alignment
        * alignment;
    let bytes = u64::from(stride)
        .checked_mul(u64::from(size[1]))
        .context("Screenshot buffer overflow")?;
    ensure!(
        u64::from(row) * u64::from(size[1]) <= 256 * 1024 * 1024,
        "Screenshot exceeds 256 MiB pixel limit"
    );
    ensure!(
        bytes <= d.limits().max_buffer_size,
        "Screenshot exceeds GPU readback buffer limit"
    );
    let buffer = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cropped screenshot readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            origin: wgpu::Origin3d {
                x: origin[0],
                y: origin[1],
                z: 0,
            },
            ..texture.as_image_copy()
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
    Ok(RgbaReadback {
        buffer,
        size,
        stride,
    })
}
pub const MAX_STEPS: u32 = 1024;
/// Preserve the measured 256-lane default when supported; fall back by device limits.
pub fn default_workgroup(limits: &wgpu::Limits) -> u32 {
    [256, 128, 64]
        .into_iter()
        .find(|n| {
            *n <= limits.max_compute_workgroup_size_x
                && *n <= limits.max_compute_invocations_per_workgroup
        })
        .unwrap_or(64)
}
#[derive(Clone, Copy, Debug)]
pub struct Shove {
    pub center: [f32; 2],
    pub strength: f32,
    pub size: f32,
}
#[derive(Clone, Copy, Debug)]
pub struct Paint {
    pub uv: [f32; 2],
    pub previous_uv: [f32; 2],
    pub size: f32,
    pub power: f32,
    pub erase: bool,
    pub mode: u32,
    pub layer: u32,
    pub angle: f32,
    pub line_gain: f32,
}
#[derive(Clone, Debug)]
pub struct Pick {
    pub index: u32,
    pub position: [f32; 2],
    pub distance: f32,
    pub cohort: f32,
    pub rule: Vec<f64>,
    pub mode: crate::network::Mode,
}
pub struct PreparedProject {
    configs: wgpu::Buffer,
    count: u32,
    weights: wgpu::Buffer,
    cohorts: Vec<Vec<f32>>,
    key: String,
    memory: Option<wgpu::Buffer>,
    pipeline: Option<wgpu::ComputePipeline>,
    state: wgpu::BindGroup,
    bake: wgpu::BindGroup,
    pick: wgpu::BindGroup,
    mode: crate::network::Mode,
    revision: u32,
}
pub struct Engine {
    interaction_layout: wgpu::BindGroupLayout,
    interaction_group: wgpu::BindGroup,
    interaction_uniform: wgpu::Buffer,
    interaction_dummy: wgpu::TextureView,
    /// Uniform bits and map view behind `interaction_group`; unchanged input skips rebinding.
    interaction_key: Option<([u32; 16], wgpu::TextureView)>,
    empty_group: wgpu::BindGroup,

    pub entities: wgpu::Buffer,
    pub configs: wgpu::Buffer,
    pub canvas: [wgpu::Texture; 2],
    pub canvas_views: [wgpu::TextureView; 2],
    pub current_canvas: usize,
    field: wgpu::Texture,
    pub(crate) field_view: wgpu::TextureView,
    // Monotonic: clear/reset never make an exposed or painted field pristine again.
    field_may_have_content: AtomicBool,
    pub field_size: [u32; 2],
    pub sizing: Sizing,
    pub frame_count: u32,
    pub config_count: u32,
    workgroup: u32,
    uniform_stride: u32,
    uniforms: wgpu::Buffer,
    canvas_uniforms: wgpu::Buffer,
    splat: wgpu::Buffer,
    freelist: wgpu::Buffer,
    entity_pipeline: wgpu::ComputePipeline,
    diagnostic_group: Option<wgpu::BindGroup>,
    mode: crate::network::Mode,
    revision: u32,
    brain_weights: wgpu::Buffer,
    pub brain_memory: wgpu::Buffer,
    cohort_weights: Vec<Vec<f32>>,
    brain_key: String,
    canvas_pipeline: wgpu::RenderPipeline,
    bake_pipeline: wgpu::ComputePipeline,
    entity_state: wgpu::BindGroup,
    canvas_state: wgpu::BindGroup,
    bake_group: wgpu::BindGroup,
    entity_textures: [[wgpu::BindGroup; 2]; 2],
    samplers: [wgpu::Sampler; 2],
    camera_strengths: [f32; 2],
    canvas_textures: [[wgpu::BindGroup; 2]; 2],
    rules_dirty: bool,
    wrap: usize,
    prepared_steps: u32,
    // Previously encoded steps may still reference slots after a smaller prepare.
    initialized_uniform_steps: u32,
    paint_uniform: wgpu::Buffer,
    paint_group: wgpu::BindGroup,
    paint_pipelines: [wgpu::RenderPipeline; 4],
    pick_reduce: wgpu::ComputePipeline,
    pick_derive: wgpu::ComputePipeline,
    pick_group: wgpu::BindGroup,
    pick_result: wgpu::Buffer,
    pick_staging: wgpu::Buffer,
    pick_uniform: wgpu::Buffer,
    pick_pending: bool,
    pick_mapping: bool,
    pick_radius: f32,
    pick_status: Arc<Mutex<Option<Result<(), String>>>>,
}
fn buffer(d: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}
pub fn module(d: &wgpu::Device, name: &str) -> wgpu::ShaderModule {
    d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(shader(name).into()),
    })
}
fn entry(binding: u32, buffer: &wgpu::Buffer, size: Option<u64>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: size.and_then(NonZeroU64::new),
        }),
    }
}
fn group(
    d: &wgpu::Device,
    l: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: l,
        entries,
    })
}
fn buf_layout(
    binding: u32,
    ty: wgpu::BufferBindingType,
    dynamic: bool,
    size: u64,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}
fn tex_layout(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn sampler_layout(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
fn layout(d: &wgpu::Device, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries,
    })
}
fn compute(
    d: &wgpu::Device,
    m: &wgpu::ShaderModule,
    entry: &str,
    l: Option<&wgpu::PipelineLayout>,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: l,
        module: m,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}
pub fn render_pipeline(
    d: &wgpu::Device,
    m: &wgpu::ShaderModule,
    l: Option<&wgpu::PipelineLayout>,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    mask: wgpu::ColorWrites,
) -> wgpu::RenderPipeline {
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: l,
        vertex: wgpu::VertexState {
            module: m,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: m,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: mask,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}
pub fn texture(
    d: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
impl Engine {
    pub fn preflight(d: &wgpu::Device, s: Sizing, workgroup: u32) -> Result<()> {
        let l = d.limits();
        ensure!(
            matches!(workgroup, 64 | 128 | 256),
            "Workgroup size must be 64, 128, or 256"
        );
        ensure!(
            workgroup <= l.max_compute_workgroup_size_x
                && workgroup <= l.max_compute_invocations_per_workgroup,
            "Device cannot support workgroup size {workgroup}"
        );
        ensure!(
            s.entity_count > 0 && s.entity_count < 1 << 24,
            "Particle count must fit the GPU picker"
        );
        ensure!(
            s.canvas
                .iter()
                .all(|v| *v > 0 && *v <= l.max_texture_dimension_2d),
            "Canvas exceeds device texture limits"
        );
        for bytes in [
            s.entity_count as u64 * 32,
            s.canvas[0] as u64 * s.canvas[1] as u64 * 12,
        ] {
            ensure!(
                bytes <= l.max_storage_buffer_binding_size as u64 && bytes <= l.max_buffer_size,
                "Simulation storage exceeds device limits ({bytes} bytes)"
            );
        }
        ensure!(
            s.entity_count.div_ceil(workgroup) <= l.max_compute_workgroups_per_dimension,
            "Particle dispatch exceeds device limits"
        );
        ensure!(
            s.sqrt_world_size.is_finite() && s.sqrt_world_size > 0.0,
            "Invalid world scale"
        );
        Ok(())
    }
    pub fn new(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        s: Sizing,
        project: &Project,
        workgroup: u32,
    ) -> Result<Self> {
        project.validate()?;
        Self::preflight(d, s, workgroup)?;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let entities = buffer(d, "entities", s.entity_count as u64 * 32, storage);
        let (configs, config_count) = Self::make_configs(d, q, project)?;
        let mode = project.config().mode();
        let revision = project.config().network_revision();
        let (brain_weights, cohort_weights) = Self::make_brain_weights(d, q, project)?;
        let brain_memory = Self::make_brain_memory(d, s.entity_count, mode, revision)?;
        let brain_key = Self::brain_key(project);
        let splat = buffer(
            d,
            "splat",
            s.canvas[0] as u64 * s.canvas[1] as u64 * 12,
            storage,
        );
        let freelist = buffer(d, "studio dummy free list", 8, storage);
        let stride = d
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(96)
            .next_multiple_of(d.limits().min_uniform_buffer_offset_alignment);
        let uniforms = buffer(
            d,
            "entity uniform ring",
            stride as u64 * MAX_STEPS as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let canvas_uniforms = buffer(
            d,
            "canvas uniform ring",
            stride as u64 * MAX_STEPS as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let canvas = std::array::from_fn(|_| texture(d, "canvas", s.canvas, CANVAS_FORMAT));
        let canvas_views = std::array::from_fn(|i| canvas[i].create_view(&Default::default()));
        let ratio = s.canvas[0] as f64 / s.canvas[1] as f64;
        let field_size = if s.canvas[0] as u64 * s.canvas[1] as u64 <= 512 * 512 {
            s.canvas
        } else {
            [
                (512.0 * ratio.sqrt()).round() as u32,
                (512.0 / ratio.sqrt()).round() as u32,
            ]
        };
        let field = texture(d, "walls and trails", field_size, FIELD_FORMAT);
        let field_view = field.create_view(&Default::default());
        let dummy = texture(d, "inactive camera field", [1, 1], FIELD_FORMAT);
        let dummy_view = dummy.create_view(&Default::default());
        let cs = wgpu::ShaderStages::COMPUTE;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let uniform = wgpu::BufferBindingType::Uniform;
        let interaction_layout = layout(
            d,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cs | fs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buf_layout(1, uniform, false, 64, cs | fs),
            ],
        );
        let interaction_uniform = buffer(
            d,
            "displayed Cells map",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let interaction_dummy = texture(
            d,
            "identity input map",
            [1, 1],
            wgpu::TextureFormat::Rgba32Float,
        )
        .create_view(&Default::default());
        let interaction_group = group(
            d,
            &interaction_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&interaction_dummy),
                },
                entry(1, &interaction_uniform, None),
            ],
        );
        let empty_layout = layout(d, &[]);
        let empty_group = group(d, &empty_layout, &[]);
        let entity_layout = layout(
            d,
            &[
                buf_layout(0, rw, false, 32, cs),
                buf_layout(1, ro, false, 416, cs),
                buf_layout(2, uniform, true, 96, cs),
                buf_layout(3, rw, false, 8, cs),
                buf_layout(4, rw, false, 8, cs),
                buf_layout(5, ro, false, 4, cs),
                buf_layout(6, rw, false, 32, cs),
            ],
        );
        let entity_tex_layout = layout(
            d,
            &[
                tex_layout(0, cs),
                sampler_layout(1, cs),
                tex_layout(2, cs),
                sampler_layout(3, cs),
                tex_layout(4, cs),
            ],
        );
        let canvas_layout = layout(
            d,
            &[
                buf_layout(0, uniform, true, 48, fs),
                buf_layout(1, ro, false, 8, fs),
            ],
        );
        let canvas_tex_layout = layout(d, &[tex_layout(0, fs), sampler_layout(1, fs)]);
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&entity_layout),
                Some(&entity_tex_layout),
                Some(&interaction_layout),
            ],
            immediate_size: 0,
        });
        let entity_pipeline = compute(
            d,
            &module(d, "entityUpdate.wgsl"),
            "main",
            Some(&pl),
            &[
                ("CONFIG_PER_COHORT", 1.0),
                ("WORKGROUP_SIZE", workgroup as f64),
                ("NETWORK_MODE", mode.id() as f64),
                ("BRAIN_REVISION", revision as f64),
            ],
        );
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&canvas_layout), Some(&canvas_tex_layout)],
            immediate_size: 0,
        });
        let canvas_pipeline = render_pipeline(
            d,
            &module(d, "canvas.wgsl"),
            Some(&pl),
            CANVAS_FORMAT,
            None,
            wgpu::ColorWrites::ALL,
        );
        let bake_pipeline = compute(
            d,
            &module(d, "cohortRules.wgsl"),
            "main",
            None,
            &[("CONFIG_PER_COHORT", 1.0)],
        );
        let bake_group = group(
            d,
            &bake_pipeline.get_bind_group_layout(0),
            &[entry(0, &configs, None)],
        );
        let entity_state = group(
            d,
            &entity_layout,
            &[
                entry(0, &entities, None),
                entry(1, &configs, None),
                entry(2, &uniforms, Some(96)),
                entry(3, &freelist, None),
                entry(4, &splat, None),
                entry(5, &brain_weights, None),
                entry(6, &brain_memory, None),
            ],
        );
        let canvas_state = group(
            d,
            &canvas_layout,
            &[entry(0, &canvas_uniforms, Some(48)), entry(1, &splat, None)],
        );
        let samplers: [wgpu::Sampler; 2] = std::array::from_fn(|i| {
            d.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: if i == 1 {
                    wgpu::AddressMode::Repeat
                } else {
                    wgpu::AddressMode::ClampToEdge
                },
                address_mode_v: if i == 1 {
                    wgpu::AddressMode::Repeat
                } else {
                    wgpu::AddressMode::ClampToEdge
                },
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        });
        let entity_textures = std::array::from_fn(|wrap| {
            std::array::from_fn(|i| {
                group(
                    d,
                    &entity_tex_layout,
                    &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&canvas_views[i]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&samplers[wrap]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&field_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&samplers[wrap]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&dummy_view),
                        },
                    ],
                )
            })
        });
        let canvas_textures = std::array::from_fn(|wrap| {
            std::array::from_fn(|i| {
                group(
                    d,
                    &canvas_tex_layout,
                    &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&canvas_views[i]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&samplers[wrap]),
                        },
                    ],
                )
            })
        });
        let paint_uniform = buffer(
            d,
            "field brush",
            64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let paint_layout = layout(
            d,
            &[buf_layout(
                0,
                uniform,
                false,
                64,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
            )],
        );
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&paint_layout),
                Some(&empty_layout),
                Some(&interaction_layout),
            ],
            immediate_size: 0,
        });
        let pm = module(d, "strafeDraw.wgsl");
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let paint_pipelines = std::array::from_fn(|i| {
            render_pipeline(
                d,
                &pm,
                Some(&pl),
                FIELD_FORMAT,
                if i % 2 == 0 { Some(additive) } else { None },
                if i < 2 {
                    wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN
                } else {
                    wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA
                },
            )
        });
        let paint_group = group(d, &paint_layout, &[entry(0, &paint_uniform, None)]);
        let pick_result = buffer(d, "pick result", 336, storage);
        let pick_staging = buffer(
            d,
            "pick staging",
            336,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let pick_uniform = buffer(
            d,
            "pick params",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let pick_layout = layout(
            d,
            &[
                buf_layout(0, ro, false, 32, cs),
                buf_layout(1, ro, false, 416, cs),
                buf_layout(2, rw, false, 336, cs),
                buf_layout(3, uniform, false, 48, cs),
            ],
        );
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&pick_layout),
                Some(&empty_layout),
                Some(&interaction_layout),
            ],
            immediate_size: 0,
        });
        let pm = module(d, "entityPick.wgsl");
        let pick_reduce = compute(d, &pm, "reduce", Some(&pl), &[("CONFIG_PER_COHORT", 1.0)]);
        let pick_derive = compute(d, &pm, "derive", Some(&pl), &[("CONFIG_PER_COHORT", 1.0)]);
        let pick_group = group(
            d,
            &pick_layout,
            &[
                entry(0, &entities, None),
                entry(1, &configs, None),
                entry(2, &pick_result, None),
                entry(3, &pick_uniform, None),
            ],
        );
        Ok(Self {
            entities,
            configs,
            canvas,
            canvas_views,
            current_canvas: 0,
            field,
            field_view,
            field_may_have_content: AtomicBool::new(false),
            field_size,
            interaction_layout,
            interaction_group,
            interaction_uniform,
            interaction_dummy,
            interaction_key: None,
            empty_group,
            sizing: s,
            frame_count: 0,
            config_count,
            workgroup,
            uniform_stride: stride,
            uniforms,
            canvas_uniforms,
            splat,
            freelist,
            entity_pipeline,
            diagnostic_group: None,
            mode,
            revision,
            brain_weights,
            brain_memory,
            cohort_weights,
            brain_key,
            canvas_pipeline,
            bake_pipeline,
            entity_state,
            canvas_state,
            bake_group,
            entity_textures,
            samplers,
            camera_strengths: [0.0; 2],
            canvas_textures,
            rules_dirty: true,
            wrap: 0,
            prepared_steps: 0,
            initialized_uniform_steps: 0,
            paint_uniform,
            paint_group,
            paint_pipelines,
            pick_reduce,
            pick_derive,
            pick_group,
            pick_result,
            pick_staging,
            pick_uniform,
            pick_pending: false,
            pick_mapping: false,
            pick_radius: 0.0,
            pick_status: Arc::new(Mutex::new(None)),
        })
    }
    fn make_configs(d: &wgpu::Device, q: &wgpu::Queue, p: &Project) -> Result<(wgpu::Buffer, u32)> {
        let mut c = p
            .configs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Project needs a config"))?
            .clone();
        c.validate()?;
        let n = c.cohorts.clamp(1, 4096);
        c.trail_weight = 1.0;
        let packed = c.pack();
        let b = buffer(
            d,
            "cohort configs",
            n as u64 * 416,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let data: Vec<u32> = (0..n).flat_map(|_| packed).collect();
        q.write_buffer(&b, 0, bytemuck::cast_slice(&data));
        Ok((b, n))
    }
    fn brain_key(p: &Project) -> String {
        let c = p.config();
        if c.mode() == crate::network::Mode::Fourier {
            return String::new();
        }
        serde_json::to_string(&(c.cohorts, c.bank())).unwrap()
    }
    fn make_brain_memory(
        d: &wgpu::Device,
        count: u32,
        mode: crate::network::Mode,
        revision: u32,
    ) -> Result<wgpu::Buffer> {
        let size = if revision >= 3 && mode != crate::network::Mode::Fourier {
            u64::from(count)
                * if mode == crate::network::Mode::Gru {
                    48
                } else {
                    16
                }
        } else if mode == crate::network::Mode::Gru {
            u64::from(count) * 32
        } else {
            32
        };
        ensure!(
            size <= d.limits().max_buffer_size
                && size <= u64::from(d.limits().max_storage_buffer_binding_size),
            "GRU memory exceeds GPU capacity"
        );
        ensure!(
            d.limits().max_storage_buffers_per_shader_stage >= 6,
            "Network modes need six storage bindings"
        );
        Ok(buffer(
            d,
            "particle brain memory",
            size.max(32),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        ))
    }
    fn make_brain_weights(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        p: &Project,
    ) -> Result<(wgpu::Buffer, Vec<Vec<f32>>)> {
        let c = p.config();
        let weights = if c.mode() == crate::network::Mode::Fourier {
            vec![]
        } else {
            let bank = c.bank().context("missing brain bank")?;
            (0..c.cohorts.clamp(1, 4096))
                .map(|i| crate::network::derive(&bank, i))
                .collect::<Vec<_>>()
        };
        let mut packed = Vec::new();
        for rule in &weights {
            packed.extend(rule.iter().copied());
            packed.resize(
                packed.len() + crate::network::WEIGHT_STRIDE - rule.len(),
                0.0,
            );
        }
        if packed.is_empty() {
            packed.push(0.0);
        }
        let b = buffer(
            d,
            "cohort brain weights",
            (packed.len() * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        q.write_buffer(&b, 0, bytemuck::cast_slice(&packed));
        Ok((b, weights))
    }
    pub fn prepare_project(
        &self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        p: &Project,
    ) -> Result<PreparedProject> {
        p.validate()?;
        let mode = p.config().mode();
        let revision = p.config().network_revision();
        let key = Self::brain_key(p);
        // Prepare every fallible resource before committing any active state.
        let memory = if mode != self.mode || key != self.brain_key {
            Some(Self::make_brain_memory(
                d,
                self.sizing.entity_count,
                mode,
                revision,
            )?)
        } else {
            None
        };
        let (b, n) = Self::make_configs(d, q, p)?;
        let (weights, cohorts) = Self::make_brain_weights(d, q, p)?;
        let entity_layout = self.entity_pipeline.get_bind_group_layout(0);
        let texture_layout = self.entity_pipeline.get_bind_group_layout(1);
        let pipeline = if mode != self.mode || revision != self.revision {
            let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("network update"),
                bind_group_layouts: &[
                    Some(&entity_layout),
                    Some(&texture_layout),
                    Some(&self.interaction_layout),
                ],
                immediate_size: 0,
            });
            Some(compute(
                d,
                &module(d, "entityUpdate.wgsl"),
                "main",
                Some(&pl),
                &[
                    ("CONFIG_PER_COHORT", 1.0),
                    ("WORKGROUP_SIZE", self.workgroup as f64),
                    ("NETWORK_MODE", mode.id() as f64),
                    ("BRAIN_REVISION", revision as f64),
                ],
            ))
        } else {
            None
        };
        let state = group(
            d,
            &entity_layout,
            &[
                entry(0, &self.entities, None),
                entry(1, &b, None),
                entry(2, &self.uniforms, Some(96)),
                entry(3, &self.freelist, None),
                entry(4, &self.splat, None),
                entry(5, &weights, None),
                entry(6, memory.as_ref().unwrap_or(&self.brain_memory), None),
            ],
        );
        let bake = group(
            d,
            &self.bake_pipeline.get_bind_group_layout(0),
            &[entry(0, &b, None)],
        );
        let pick = group(
            d,
            &self.pick_reduce.get_bind_group_layout(0),
            &[
                entry(0, &self.entities, None),
                entry(1, &b, None),
                entry(2, &self.pick_result, None),
                entry(3, &self.pick_uniform, None),
            ],
        );
        Ok(PreparedProject {
            configs: b,
            count: n,
            weights,
            cohorts,
            key,
            memory,
            pipeline,
            state,
            bake,
            pick,
            mode,
            revision,
        })
    }
    pub fn commit_project(&mut self, prepared: PreparedProject) {
        let PreparedProject {
            configs: b,
            count: n,
            weights,
            cohorts,
            key,
            memory,
            pipeline,
            state,
            bake,
            pick,
            mode,
            revision,
        } = prepared;
        self.cancel_pick();
        self.configs = b;
        self.config_count = n;
        self.brain_weights = weights;
        self.cohort_weights = cohorts;
        self.brain_key = key;
        if let Some(memory) = memory {
            self.brain_memory = memory;
        }
        if let Some(pipeline) = pipeline {
            self.entity_pipeline = pipeline;
        }
        self.entity_state = state;
        self.bake_group = bake;
        self.pick_group = pick;
        self.rules_dirty = mode == crate::network::Mode::Fourier;
        if mode != self.mode || revision != self.revision {
            self.reset();
        }
        self.mode = mode;
        self.revision = revision;
    }
    /// Synchronous adapter for native validation and legacy preview uploads.
    /// Interactive changes use scoped asynchronous preparation in SoupApp.
    pub fn upload_project(&mut self, d: &wgpu::Device, q: &wgpu::Queue, p: &Project) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let oom = d.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let validation = d.push_error_scope(wgpu::ErrorFilter::Validation);
            let prepared = self.prepare_project(d, q, p);
            let validation = validation.pop();
            let oom = oom.pop();
            let error = pollster::block_on(async { validation.await.or(oom.await) });
            ensure!(error.is_none(), "Could not prepare GPU project: {error:?}");
            self.commit_project(prepared?);
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            ensure!(
                p.config().mode() == self.mode && self.mode == crate::network::Mode::Fourier,
                "Use asynchronous project preparation for network changes"
            );
            let prepared = self.prepare_project(d, q, p)?;
            self.commit_project(prepared);
            Ok(())
        }
    }
    /// Snapshot the finished cohort rules, independent of subsequent switches.
    pub fn snapshot_rules(&self, d: &wgpu::Device, q: &wgpu::Queue) -> wgpu::Buffer {
        let source = if self.mode == crate::network::Mode::Fourier {
            &self.configs
        } else {
            &self.brain_weights
        };
        let snapshot = buffer(
            d,
            "comparison rule snapshot",
            source.size(),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = d.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &snapshot, 0, source.size());
        q.submit([encoder.finish()]);
        snapshot
    }
    pub fn reset(&mut self) {
        self.cancel_pick();
        self.frame_count = 0;
    }
    pub fn cancel_pick(&mut self) {
        self.pick_pending = false;
        // Unmap only an active mapping. A failed map has already returned the
        // buffer to the unmapped state; cancelling a pending map is permitted.
        if self.pick_mapping
            && !self
                .pick_status
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(Result::is_err)
        {
            self.pick_staging.unmap();
        }
        self.pick_mapping = false;
        self.pick_status = Arc::new(Mutex::new(None));
    }
    pub fn set_camera_field(&mut self, d: &wgpu::Device, view: &wgpu::TextureView) {
        let layout = self.entity_pipeline.get_bind_group_layout(1);
        self.entity_textures = std::array::from_fn(|wrap| {
            std::array::from_fn(|i| {
                group(
                    d,
                    &layout,
                    &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&self.canvas_views[i]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.samplers[wrap]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.field_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.samplers[wrap]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                    ],
                )
            })
        });
    }
    pub fn set_camera_strengths(&mut self, strengths: [f32; 2]) {
        self.camera_strengths = strengths.map(|v| if v.is_finite() { v } else { 0.0 });
    }
    /// Clear just one painted layer using the source whole-field eraser and its write mask.
    /// Shares the brush uniform: encode no additional paint pass before this submission.
    pub fn clear_layer(
        &self,
        q: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layer: u32,
    ) -> Result<()> {
        self.paint(
            q,
            encoder,
            Paint {
                uv: [0.5, 0.5],
                previous_uv: [0.5, 0.5],
                size: 4.0,
                power: 0.0,
                erase: true,
                mode: 0,
                layer,
                angle: 0.0,
                line_gain: 1.0,
            },
        )
    }
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup
    }
    /// Resource sizes, excluding driver allocation granularity and external renderer textures.
    pub fn allocated_bytes(&self) -> u64 {
        [
            &self.entities,
            &self.configs,
            &self.uniforms,
            &self.canvas_uniforms,
            &self.splat,
            &self.freelist,
            &self.paint_uniform,
            &self.pick_result,
            &self.pick_staging,
            &self.pick_uniform,
            &self.brain_weights,
            &self.brain_memory,
        ]
        .iter()
        .map(|b| b.size())
        .sum::<u64>()
            + self.sizing.canvas[0] as u64 * self.sizing.canvas[1] as u64 * 16
            + self.field_size[0] as u64 * self.field_size[1] as u64 * 8
            + 8
    }
    /// Fixed-point deposit storage, exposed for GPU validation readback.
    pub fn splat_buffer(&self) -> &wgpu::Buffer {
        &self.splat
    }
    pub fn current_canvas_view(&self) -> &wgpu::TextureView {
        &self.canvas_views[self.current_canvas]
    }
    pub fn prepare_frame(
        &mut self,
        q: &wgpu::Queue,
        w: &WorldSettings,
        steps: u32,
        shove: Option<Shove>,
        strengths: [f32; 2],
    ) -> Result<()> {
        w.validate()?;
        ensure!(
            strengths.iter().all(|v| v.is_finite()),
            "Invalid field strength"
        );
        if let Some(s) = shove {
            ensure!(
                s.center.iter().all(|v| v.is_finite())
                    && s.strength.is_finite()
                    && s.size.is_finite()
                    && (s.strength == 0.0 || s.size > 0.0),
                "Invalid shove parameters"
            );
        }
        ensure!(steps <= MAX_STEPS, "Physics steps exceed {MAX_STEPS}");
        self.prepared_steps = steps;
        self.initialized_uniform_steps = self.initialized_uniform_steps.max(steps);
        self.wrap = usize::from(w.boundary_conditions == 1);
        let world = w.pack(self.sizing.sqrt_world_size, self.config_count);
        let field_active = u32::from(self.field_sampling_active());
        let mut entity = vec![0u32; steps as usize * self.uniform_stride as usize / 4];
        let mut canvas = entity.clone();
        for i in 0..steps {
            let offset = i as usize * self.uniform_stride as usize / 4;
            let e = &mut entity[offset..offset + 24];
            e[..8].copy_from_slice(&world);
            e[8] = (self.sizing.canvas[0] as f32).to_bits();
            e[9] = (self.sizing.canvas[1] as f32).to_bits();
            e[10] = (self.field_size[0] as f32).to_bits();
            e[11] = (self.field_size[1] as f32).to_bits();
            if let Some(s) = shove {
                e[12] = s.center[0].to_bits();
                e[13] = s.center[1].to_bits();
                e[14] = s.strength.to_bits();
                e[15] = s.size.to_bits();
            }
            e[16] = self.frame_count.wrapping_add(i);
            e[17] = field_active;
            e[18] = strengths[0].to_bits();
            e[19] = strengths[1].to_bits();
            e[20] = self.camera_strengths[0].to_bits();
            e[21] = self.camera_strengths[1].to_bits();
            let c = &mut canvas[offset..offset + 12];
            c[..8].copy_from_slice(&world);
            c[8] = e[16];
        }
        if steps > 0 {
            q.write_buffer(&self.uniforms, 0, bytemuck::cast_slice(&entity));
            q.write_buffer(&self.canvas_uniforms, 0, bytemuck::cast_slice(&canvas));
        }
        Ok(())
    }
    pub fn encode_bake(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.rules_dirty && self.mode == crate::network::Mode::Fourier {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cohort rule bake"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.bake_pipeline);
            p.set_bind_group(0, &self.bake_group, &[]);
            p.dispatch_workgroups(self.config_count.div_ceil(64), 1, 1);
            drop(p);
            self.rules_dirty = false;
        }
    }
    pub fn encode_step(&mut self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        self.encode_step_profiled(encoder, index, None);
    }
    pub fn encode_step_profiled(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        index: u32,
        mut profile: Option<&mut crate::profiling::ProfileFrame>,
    ) {
        use crate::profiling::Stage;
        assert!(index < self.prepared_steps);
        let entity = profile.as_mut().and_then(|p| p.begin(encoder));
        self.encode_bake(encoder);
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entity update"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.entity_pipeline);
            p.set_bind_group(0, &self.entity_state, &[index * self.uniform_stride]);
            p.set_bind_group(
                1,
                &self.entity_textures[self.wrap][self.current_canvas],
                &[],
            );
            p.set_bind_group(2, &self.interaction_group, &[]);
            if let Some(group) = &self.diagnostic_group {
                p.set_bind_group(3, group, &[]);
            }
            p.dispatch_workgroups(self.sizing.entity_count.div_ceil(self.workgroup), 1, 1);
        }
        if let Some(p) = profile.as_mut() {
            p.end(encoder, Stage::Entity, entity);
        }
        let trails = profile.as_mut().and_then(|p| p.begin(encoder));
        let back = 1 - self.current_canvas;
        draw(
            encoder,
            &self.canvas_views[back],
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            &self.canvas_pipeline,
            &[
                (&self.canvas_state, &[index * self.uniform_stride]),
                (&self.canvas_textures[self.wrap][self.current_canvas], &[]),
            ],
        );
        // Drain the deposits for the next step. A transfer clear is far cheaper
        // than zeroing from the canvas fragment shader (~0.23 ms per 1024² step
        // measured on an RTX 3090), and it is part of the trail stage.
        encoder.clear_buffer(&self.splat, 0, None);
        if let Some(p) = profile.as_mut() {
            p.end(encoder, Stage::Trails, trails);
        }
        self.current_canvas = back;
        self.frame_count = self.frame_count.wrapping_add(1);
    }
    /// Whether physics must conservatively sample the painted field. Once active,
    /// this stays active for the engine's lifetime, including after clear/reset.
    pub fn field_sampling_active(&self) -> bool {
        self.field_may_have_content.load(Ordering::Acquire)
    }
    fn activate_field_sampling(&self, q: &wgpu::Queue) {
        if !self.field_sampling_active() {
            // Already encoded steps can outlive a later smaller/zero-step prepare.
            // Patch every initialized slot before publishing the active flag.
            // Concurrent first callers can safely queue duplicate patches; none
            // may return a writable texture handle before patches are queued.
            for i in 0..self.initialized_uniform_steps {
                q.write_buffer(
                    &self.uniforms,
                    u64::from(i) * u64::from(self.uniform_stride) + 68,
                    bytemuck::bytes_of(&1u32),
                );
            }
            self.field_may_have_content.store(true, Ordering::Release);
        }
    }
    /// Obtain the field for raw access. Call this BEFORE queue writes, copies,
    /// or submissions touching it. Even read-only access permanently enables
    /// sampling, because a returned/cloned texture handle can later be written.
    pub fn field_texture(&self, q: &wgpu::Queue) -> &wgpu::Texture {
        self.activate_field_sampling(q);
        &self.field
    }
    pub fn paint(
        &self,
        q: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        p: Paint,
    ) -> Result<()> {
        ensure!(
            p.uv.iter()
                .chain(p.previous_uv.iter())
                .chain([p.size, p.power, p.angle, p.line_gain].iter())
                .all(|v| v.is_finite())
                && p.size > 0.0,
            "Invalid brush parameters"
        );
        ensure!(p.layer < 2 && p.mode < 4, "Invalid brush layer/mode");
        self.activate_field_sampling(q);
        let mut u = [0u32; 16];
        for (i, f) in [
            (0, self.field_size[0] as f32),
            (1, self.field_size[1] as f32),
            (4, p.uv[0]),
            (5, p.uv[1]),
            (6, p.previous_uv[0]),
            (7, p.previous_uv[1]),
            (8, p.size),
            (9, p.power),
            (13, p.angle),
            (14, p.line_gain),
        ] {
            u[i] = f.to_bits();
        }
        u[10] = u32::from(p.erase);
        u[11] = p.mode;
        u[12] = p.layer;
        q.write_buffer(&self.paint_uniform, 0, bytemuck::cast_slice(&u));
        draw(
            encoder,
            &self.field_view,
            wgpu::LoadOp::Load,
            &self.paint_pipelines[(p.layer * 2 + u32::from(p.erase)) as usize],
            &[
                (&self.paint_group, &[]),
                (&self.empty_group, &[]),
                (&self.interaction_group, &[]),
            ],
        );
        Ok(())
    }
    /// Capture input against the image currently displayed. Call before encoding
    /// input/ticks and before the next rendering pass overwrites any map target.
    pub fn set_interaction_map(
        &mut self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        cells: Option<&crate::cells::Cells>,
    ) {
        let mut u = [0.0f32; 16];
        let view = if let Some(c) = cells.filter(|c| c.samples > 0) {
            u[0] = self.sizing.canvas[0] as f32;
            u[1] = self.sizing.canvas[1] as f32;
            u[4] = c.displayed_camera.pan[0];
            u[5] = c.displayed_camera.pan[1];
            u[6] = c.displayed_camera.zoom;
            u[8] = c.size[0] as f32;
            u[9] = c.size[1] as f32;
            u[10] = (c.padded[0] - c.size[0]) as f32 * 0.5;
            u[11] = (c.padded[1] - c.size[1]) as f32 * 0.5;
            u[12] = c.padded[0] as f32;
            u[13] = c.padded[1] as f32;
            u[14] = c.size[0].min(c.size[1]) as f32 / 1080.0;
            u[15] = 1.0;
            &c.map_views[c.displayed_map]
        } else {
            &self.interaction_dummy
        };
        let bits = u.map(f32::to_bits);
        if self
            .interaction_key
            .as_ref()
            .is_some_and(|(b, v)| *b == bits && v == view)
        {
            return;
        }
        self.interaction_key = Some((bits, view.clone()));
        q.write_buffer(&self.interaction_uniform, 0, bytemuck::cast_slice(&u));
        self.interaction_group = group(
            d,
            &self.interaction_layout,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                entry(1, &self.interaction_uniform, None),
            ],
        );
    }
    pub fn clear_field(&self, encoder: &mut wgpu::CommandEncoder) {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &self.field_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })];
        let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear walls/trails"),
            color_attachments: &attachments,
            ..Default::default()
        });
    }
    pub fn request_pick(
        &mut self,
        q: &wgpu::Queue,
        w: &WorldSettings,
        target: [f32; 2],
        radius: f32,
        highlight: f32,
    ) -> bool {
        if self.pick_mapping
            || self.pick_pending
            || !target.iter().all(|v| v.is_finite())
            || !radius.is_finite()
            || radius == 0.0
            || !highlight.is_finite()
        {
            return false;
        }
        let mut u = [0u32; 12];
        u[..8].copy_from_slice(&w.pack(self.sizing.sqrt_world_size, self.config_count));
        u[8] = target[0].to_bits();
        u[9] = target[1].to_bits();
        u[10] = radius.to_bits();
        u[11] = highlight.to_bits();
        q.write_buffer(&self.pick_uniform, 0, bytemuck::cast_slice(&u));
        q.write_buffer(&self.pick_result, 0, &u32::MAX.to_le_bytes());
        self.pick_radius = radius.abs();
        self.pick_pending = true;
        true
    }
    pub fn encode_pick(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if !self.pick_pending {
            return false;
        }
        self.encode_bake(encoder);
        for (pipeline, n) in [
            (&self.pick_reduce, self.sizing.entity_count.div_ceil(256)),
            (&self.pick_derive, 1),
        ] {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU pick"),
                timestamp_writes: None,
            });
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &self.pick_group, &[]);
            p.set_bind_group(1, &self.empty_group, &[]);
            p.set_bind_group(2, &self.interaction_group, &[]);
            p.dispatch_workgroups(n, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.pick_result, 0, &self.pick_staging, 0, 336);
        self.pick_pending = false;
        true
    }
    /// Call after submitting the encoder containing encode_pick. Poll the device each frame.
    /// True only after an encoded pick starts asynchronous CPU readback.
    pub fn pick_readback_pending(&self) -> bool {
        self.pick_mapping
    }
    pub fn begin_pick_readback(&mut self) {
        if self.pick_mapping {
            return;
        }
        self.pick_mapping = true;
        let status = self.pick_status.clone();
        self.pick_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                *status.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
            });
    }
    pub fn take_pick(&mut self) -> Result<Option<Option<Pick>>> {
        let status = self.pick_status.lock().unwrap().take();
        let Some(status) = status else {
            return Ok(None);
        };
        self.pick_mapping = false;
        if let Err(e) = status {
            bail!("GPU picker readback: {e}")
        }
        let bytes = match self.pick_staging.slice(..).get_mapped_range() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.pick_staging.unmap();
                bail!("GPU picker mapped range: {error}")
            }
        };
        let mut result = decode_pick(&bytes, self.pick_radius);
        if let Some(pick) = result.as_mut() {
            pick.mode = self.mode;
            if self.mode != crate::network::Mode::Fourier {
                pick.rule = self
                    .cohort_weights
                    .get(pick.cohort as usize)
                    .context("picked unknown cohort")?
                    .iter()
                    .map(|v| *v as f64)
                    .collect();
            }
        }
        drop(bytes);
        self.pick_staging.unmap();
        Ok(Some(result))
    }
}
pub fn draw(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipeline: &wgpu::RenderPipeline,
    groups: &[(&wgpu::BindGroup, &[u32])],
) {
    let attachments = [Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
    })];
    let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("GPU fullscreen pass"),
        color_attachments: &attachments,
        ..Default::default()
    });
    p.set_pipeline(pipeline);
    for (i, (g, offset)) in groups.iter().enumerate() {
        p.set_bind_group(i as u32, *g, offset);
    }
    p.draw(0..4, 0..1);
}

/// Decode a completed source-format result; a miss must never expose stale payload lanes.
fn decode_pick(bytes: &[u8], radius: f32) -> Option<Pick> {
    let key = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    if key == u32::MAX {
        return None;
    }
    let f = |i: usize| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    Some(Pick {
        mode: crate::network::Mode::Fourier,
        index: key & 0x00ff_ffff,
        position: [f(1), f(2)],
        cohort: f(3),
        distance: (key >> 24) as f32 / 255.0 * radius,
        rule: (4..84).map(|i| f(i) as f64).collect(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picker_retains_large_indices_and_unsigned_distance() {
        let mut lanes = [0u32; 84];
        lanes[0] = (200u32 << 24) | 1_200_001;
        lanes[1] = 0.25f32.to_bits();
        lanes[2] = (-0.75f32).to_bits();
        lanes[3] = 7f32.to_bits();
        lanes[83] = (-2.5f32).to_bits();
        let picked = decode_pick(bytemuck::cast_slice(&lanes), 0.2).unwrap();
        assert_eq!(picked.index, 1_200_001);
        assert_eq!(picked.position, [0.25, -0.75]);
        assert_eq!(picked.cohort, 7.0);
        assert_eq!(picked.rule.len(), 80);
        assert_eq!(picked.rule[79], -2.5);
        assert!((picked.distance - 200.0 / 255.0 * 0.2).abs() < 1e-7);
        lanes[0] = u32::MAX;
        assert!(decode_pick(bytemuck::cast_slice(&lanes), 0.2).is_none());
    }
}

#[cfg(test)]
mod workgroup_tests {
    #[test]
    fn default_tracks_supported_limits() {
        let mut limits = wgpu::Limits::default();
        assert_eq!(super::default_workgroup(&limits), 256);
        limits.max_compute_invocations_per_workgroup = 128;
        assert_eq!(super::default_workgroup(&limits), 128);
        limits.max_compute_workgroup_size_x = 64;
        assert_eq!(super::default_workgroup(&limits), 64);
    }
}
