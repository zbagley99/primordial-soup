//! Fluoddity camera, temporal accumulation, bloom and tone curve.
//! Shader equations retain Jesse Gelders' MIT-licensed implementation.
use crate::{
    config::{Preferences, Project},
    coords::Camera,
    gpu::{self, Engine},
};
use anyhow::{Result, ensure};

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const OUTPUT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const LEVELS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlurSchedule {
    pub samples: u32,
    pub stride: u32,
    pub sample_at: u32,
}
pub fn blur_schedule(steps: u32, requested: u32) -> BlurSchedule {
    let steps = steps.max(1);
    if requested <= 1 {
        return BlurSchedule {
            samples: 1,
            stride: steps,
            sample_at: steps - 1,
        };
    }
    let stride = (steps / requested).max(1);
    BlurSchedule {
        samples: steps.div_ceil(stride),
        stride,
        sample_at: 0,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overlays {
    pub walls: bool,
    pub trails: bool,
    pub reticle: [f32; 2],
    pub radius: f32,
    pub style: u32,
    pub angle: f32,
    pub line: Option<([f32; 2], [f32; 2])>,
    pub crop: Option<[f32; 2]>,
}
impl Overlays {
    /// Assembly reads reticle position, style and angle only while a reticle is
    /// drawn, so a hidden reticle compares equal wherever the pointer is.
    pub fn normalized(mut self) -> Self {
        if self.radius <= 0.0 {
            self.radius = 0.0;
            self.reticle = [0.0; 2];
            self.style = 0;
            self.angle = 0.0;
        }
        self
    }
}

fn uniform(d: &wgpu::Device, name: &str, size: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: b.as_entire_binding(),
    }
}
fn tex(binding: u32, t: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(t),
    }
}
fn samp(binding: u32, s: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(s),
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
fn additive() -> wgpu::BlendState {
    let channel = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: channel,
        alpha: channel,
    }
}
fn pipeline(
    d: &wgpu::Device,
    name: &str,
    particle: bool,
    format: wgpu::TextureFormat,
    blend: bool,
) -> wgpu::RenderPipeline {
    let module = gpu::module(d, name);
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(if particle { "vs_main" } else { "fullscreen_vs" }),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: blend.then(additive),
                write_mask: wgpu::ColorWrites::ALL,
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

struct Targets {
    hdr: wgpu::TextureView,
    accum: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    clean: Option<CleanTarget>,
    mips: Vec<wgpu::TextureView>,
    mip_sizes: Vec<[u32; 2]>,
    accum_group: wgpu::BindGroup,
    down_groups: [Vec<wgpu::BindGroup>; 2],
    up_groups: Vec<wgpu::BindGroup>,
    assembly_groups: [[wgpu::BindGroup; 2]; 2],
}

struct CleanTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct PreparedCells {
    cells: crate::cells::Cells,
    assembly: [[wgpu::BindGroup; 2]; 2],
    bloom: wgpu::BindGroup,
}
pub struct Renderer {
    pub cells: Option<crate::cells::Cells>,
    cells_assembly: Option<[[wgpu::BindGroup; 2]; 2]>,
    cells_bloom: Option<wgpu::BindGroup>,
    cell_active: bool,
    cell_dirty: bool,
    frame_prefs: Preferences,
    pub size: [u32; 2],
    targets: Targets,
    trail: wgpu::RenderPipeline,
    particles: wgpu::RenderPipeline,
    accumulate: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    assembly: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    view_uniform: wgpu::Buffer,
    particle_uniform: wgpu::Buffer,
    accum_uniform: wgpu::Buffer,
    assembly_uniform: wgpu::Buffer,
    clean_uniform: wgpu::Buffer,
    down_uniforms: Vec<wgpu::Buffer>,
    up_uniforms: Vec<wgpu::Buffer>,
    trail_group: wgpu::BindGroup,
    trail_textures: [wgpu::BindGroup; 2],
    particle_group: wgpu::BindGroup,
    assembly_group: wgpu::BindGroup,
    clean_group: wgpu::BindGroup,
    pub direct: bool,
    pub samples_taken: u32,
    /// Bloom inputs whose result the mip chain currently holds. Sampling, Cells
    /// resolve or new targets clear it; held overlay-only frames reuse the chain.
    bloom_valid: Option<[u32; 4]>,
}

impl Renderer {
    pub(crate) fn prepare_trail_bindings(
        &self,
        d: &wgpu::Device,
        views: &[wgpu::TextureView; 2],
    ) -> [wgpu::BindGroup; 2] {
        std::array::from_fn(|i| {
            group(
                d,
                &self.trail.get_bind_group_layout(1),
                &[tex(0, &views[i]), samp(1, &self.sampler)],
            )
        })
    }
    pub(crate) fn commit_trail_bindings(&mut self, bindings: [wgpu::BindGroup; 2]) {
        self.trail_textures = bindings;
        self.samples_taken = 0;
        self.bloom_valid = None;
    }
    /// Rebind after committing a project that replaces the trail texture pair.
    pub fn rebind_trails(&mut self, d: &wgpu::Device, views: &[wgpu::TextureView; 2]) {
        self.commit_trail_bindings(self.prepare_trail_bindings(d, views));
    }
    pub fn new(d: &wgpu::Device, e: &Engine, size: [u32; 2]) -> Result<Self> {
        Self::new_with_export_target_policy(d, e, size, false)
    }

    /// Reconstruct the previous idle allocation policy for a native benchmark.
    /// Normal application startup always uses lazy export allocation.
    #[cfg(not(target_arch = "wasm32"))]
    #[doc(hidden)]
    pub fn new_eager_for_benchmark(d: &wgpu::Device, e: &Engine, size: [u32; 2]) -> Result<Self> {
        Self::new_with_export_target_policy(d, e, size, true)
    }

    fn new_with_export_target_policy(
        d: &wgpu::Device,
        e: &Engine,
        size: [u32; 2],
        eager_export: bool,
    ) -> Result<Self> {
        Self::preflight(d, size)?;
        let trail = pipeline(d, "camera.wgsl", false, HDR, false);
        let particles = pipeline(d, "camBrush.wgsl", true, HDR, true);
        let accumulate = pipeline(d, "accumulate.wgsl", false, HDR, true);
        let down = pipeline(d, "bloomDownsample.wgsl", false, HDR, false);
        let up = pipeline(d, "bloomUpsample.wgsl", false, HDR, true);
        let assembly = pipeline(d, "frameAssembly.wgsl", false, OUTPUT, false);
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("render sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view_uniform = uniform(d, "camera view", 48);
        let particle_uniform = uniform(d, "particle view", 704);
        let accum_uniform = uniform(d, "temporal weight", 16);
        let assembly_uniform = uniform(d, "tone and overlays", 144);
        let clean_uniform = uniform(d, "export tone", 144);
        let down_uniforms = (0..LEVELS)
            .map(|_| uniform(d, "bloom down", 16))
            .collect::<Vec<_>>();
        let up_uniforms = (0..LEVELS)
            .map(|_| uniform(d, "bloom up", 16))
            .collect::<Vec<_>>();
        let trail_group = group(
            d,
            &trail.get_bind_group_layout(0),
            &[entry(0, &view_uniform)],
        );
        let trail_textures = std::array::from_fn(|i| {
            group(
                d,
                &trail.get_bind_group_layout(1),
                &[tex(0, &e.canvas_views[i]), samp(1, &sampler)],
            )
        });
        let particle_group = group(
            d,
            &particles.get_bind_group_layout(0),
            &[entry(0, &particle_uniform), entry(1, &e.entities)],
        );
        let assembly_group = group(
            d,
            &assembly.get_bind_group_layout(0),
            &[entry(0, &assembly_uniform)],
        );
        let clean_group = group(
            d,
            &assembly.get_bind_group_layout(0),
            &[entry(0, &clean_uniform)],
        );
        let targets = Self::targets(
            d,
            e,
            size,
            &sampler,
            &accumulate,
            &down,
            &up,
            &assembly,
            &accum_uniform,
            &down_uniforms,
            &up_uniforms,
            eager_export,
        );
        Ok(Self {
            cells: None,
            cells_assembly: None,
            cells_bloom: None,
            cell_active: false,
            cell_dirty: false,
            frame_prefs: Preferences::default(),
            size,
            targets,
            trail,
            particles,
            accumulate,
            down,
            up,
            assembly,
            sampler,
            view_uniform,
            particle_uniform,
            accum_uniform,
            assembly_uniform,
            clean_uniform,
            down_uniforms,
            up_uniforms,
            trail_group,
            trail_textures,
            particle_group,
            assembly_group,
            clean_group,
            direct: true,
            samples_taken: 0,
            bloom_valid: None,
        })
    }

    pub fn prepare_cells(
        &self,
        d: &wgpu::Device,
        e: &Engine,
        settings: &crate::cells::Settings,
    ) -> Result<PreparedCells> {
        let cells = crate::cells::Cells::new(d, e, self.size, settings)?;
        let dummy = gpu::texture(d, "Cells no bloom", [1, 1], HDR).create_view(&Default::default());
        let assembly = std::array::from_fn(|map| {
            std::array::from_fn(|i| {
                group(
                    d,
                    &self.assembly.get_bind_group_layout(1),
                    &[
                        tex(0, &cells.scene),
                        tex(
                            1,
                            if i == 0 {
                                &dummy
                            } else {
                                &self.targets.mips[0]
                            },
                        ),
                        tex(2, &e.field_view),
                        samp(3, &self.sampler),
                        tex(4, &cells.map_views[map]),
                    ],
                )
            })
        });
        let bloom = group(
            d,
            &self.down.get_bind_group_layout(0),
            &[
                entry(0, &self.down_uniforms[0]),
                tex(1, &cells.marks),
                samp(2, &self.sampler),
            ],
        );
        Ok(PreparedCells {
            cells,
            assembly,
            bloom,
        })
    }
    pub fn commit_cells(&mut self, prepared: PreparedCells) {
        self.cells = Some(prepared.cells);
        self.cells_assembly = Some(prepared.assembly);
        self.cells_bloom = Some(prepared.bloom);
        self.samples_taken = 0;
        self.bloom_valid = None;
    }
    pub fn set_preferences(&mut self, prefs: &Preferences) {
        self.frame_prefs = prefs.clone();
    }
    pub fn cells_active(&self) -> bool {
        self.cell_active
    }
    pub fn appearance(&self) -> &crate::cells::Settings {
        &self.frame_prefs.cells
    }
    pub fn preflight(d: &wgpu::Device, size: [u32; 2]) -> Result<()> {
        ensure!(
            size.iter()
                .all(|v| *v > 0 && *v <= d.limits().max_texture_dimension_2d),
            "Render dimensions exceed GPU limits"
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn targets(
        d: &wgpu::Device,
        e: &Engine,
        size: [u32; 2],
        s: &wgpu::Sampler,
        accum: &wgpu::RenderPipeline,
        down: &wgpu::RenderPipeline,
        up: &wgpu::RenderPipeline,
        assembly: &wgpu::RenderPipeline,
        weight: &wgpu::Buffer,
        du: &[wgpu::Buffer],
        uu: &[wgpu::Buffer],
        eager_export: bool,
    ) -> Targets {
        let hdr = gpu::texture(d, "camera HDR", size, HDR).create_view(&Default::default());
        let accum_view =
            gpu::texture(d, "temporal HDR", size, HDR).create_view(&Default::default());
        let output = gpu::texture(d, "soup output", size, OUTPUT);
        let output_view = output.create_view(&Default::default());
        // The former clean target was allocated here, before bloom textures.
        // The control allocates and retains it without encoding any export work.
        let clean = eager_export.then(|| {
            let texture = gpu::texture(d, "clean export output", size, OUTPUT);
            let view = texture.create_view(&Default::default());
            CleanTarget { texture, view }
        });
        let mut mip_size = size;
        let mut mip_sizes = Vec::new();
        let mips = (0..LEVELS)
            .map(|_| {
                mip_size = mip_size.map(|v| (v / 2).max(1));
                mip_sizes.push(mip_size);
                gpu::texture(d, "bloom mip", mip_size, HDR).create_view(&Default::default())
            })
            .collect::<Vec<_>>();
        let accum_group = group(
            d,
            &accum.get_bind_group_layout(0),
            &[entry(0, weight), tex(1, &hdr), samp(2, s)],
        );
        let down_groups = std::array::from_fn(|source| {
            (0..LEVELS)
                .map(|i| {
                    group(
                        d,
                        &down.get_bind_group_layout(0),
                        &[
                            entry(0, &du[i]),
                            tex(
                                1,
                                if i > 0 {
                                    &mips[i - 1]
                                } else if source == 0 {
                                    &hdr
                                } else {
                                    &accum_view
                                },
                            ),
                            samp(2, s),
                        ],
                    )
                })
                .collect()
        });
        let up_groups = (0..LEVELS)
            .map(|i| {
                group(
                    d,
                    &up.get_bind_group_layout(0),
                    &[entry(0, &uu[i]), tex(1, &mips[i]), samp(2, s)],
                )
            })
            .collect();
        let dummy = gpu::texture(d, "bloom disabled", [1, 1], HDR).create_view(&Default::default());
        let assembly_groups = std::array::from_fn(|source| {
            std::array::from_fn(|bloom| {
                group(
                    d,
                    &assembly.get_bind_group_layout(1),
                    &[
                        tex(0, if source == 0 { &hdr } else { &accum_view }),
                        tex(1, if bloom == 0 { &dummy } else { &mips[0] }),
                        tex(2, &e.field_view),
                        samp(3, s),
                        tex(4, &dummy),
                    ],
                )
            })
        });
        Targets {
            hdr,
            accum: accum_view,
            output,
            output_view,
            clean,
            mips,
            mip_sizes,
            accum_group,
            down_groups,
            up_groups,
            assembly_groups,
        }
    }
    pub fn resize(&mut self, d: &wgpu::Device, e: &Engine, size: [u32; 2]) -> Result<bool> {
        if size == self.size {
            return Ok(false);
        }
        Self::preflight(d, size)?;
        self.targets = Self::targets(
            d,
            e,
            size,
            &self.sampler,
            &self.accumulate,
            &self.down,
            &self.up,
            &self.assembly,
            &self.accum_uniform,
            &self.down_uniforms,
            &self.up_uniforms,
            false,
        );
        self.cells = None;
        self.cells_assembly = None;
        self.cells_bloom = None;
        self.cell_active = false;
        self.size = size;
        self.samples_taken = 0;
        self.bloom_valid = None;
        Ok(true)
    }
    /// Highlighting lives only in assembly; removing it preserves held temporal color.
    pub fn clear_cells_highlight(&mut self) -> bool {
        if !self.cell_active {
            return false;
        }
        self.cells.as_mut().unwrap().highlight = -1.0;
        true
    }
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.targets.output_view
    }
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.targets.output
    }
    pub fn clean_texture(&self) -> Result<&wgpu::Texture> {
        self.targets
            .clean
            .as_ref()
            .map(|target| &target.texture)
            .ok_or_else(|| anyhow::anyhow!("Export render target has not been prepared"))
    }
    /// Logical texture storage, excluding driver allocation/alignment overhead.
    pub fn clean_output_bytes(&self) -> u64 {
        if self.targets.clean.is_some() {
            self.size[0] as u64 * self.size[1] as u64 * 4
        } else {
            0
        }
    }
    /// Logical render storage (textures and Cells buffers); excludes driver heap overhead.
    pub fn texture_bytes(&self) -> u64 {
        let mut total =
            self.size[0] as u64 * self.size[1] as u64 * 20 + 8 + self.clean_output_bytes();
        for size in &self.targets.mip_sizes {
            total += size[0] as u64 * size[1] as u64 * 8;
        }
        total + self.cells.as_ref().map_or(0, |c| c.total_bytes)
    }
    /// Submitted work retains its resource references; no synchronous GPU wait is needed.
    pub fn release_clean(&mut self) {
        self.targets.clean = None;
    }

    /// Assemble an export image without editor overlays. Its uniforms are a
    /// separate buffer: both display and export passes share one submission.
    pub fn present_clean(
        &mut self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        e: &Engine,
        camera: &Camera,
        prefs: &Preferences,
    ) {
        let source = usize::from(!self.direct);
        let bloom = prefs.bloom_enabled && prefs.bloom_intensity > 0.0;
        let mut u = [0u32; 36];
        u[32..35].copy_from_slice(&self.frame_prefs.cells.theme.background().map(f32::to_bits));
        if self.cell_active {
            let cells = self.cells.as_ref().unwrap();
            u[35] = ((cells.padded[0] - cells.size[0]) as f32 * 0.5).to_bits();
        }
        u[..8].copy_from_slice(&self.view(e, camera)[..8]);
        for (i, value) in [
            (8, if bloom { prefs.bloom_intensity } else { 0.0 }),
            (9, prefs.brightness),
            (10, prefs.tonemap_softness.max(0.1)),
            (
                19,
                if self.cell_active || self.direct {
                    1.0
                } else {
                    0.0
                },
            ),
            (24, 1.0),
            (25, 1.0),
        ] {
            u[i] = value.to_bits();
        }
        u[7] = if self.cell_active {
            1.0f32.to_bits()
        } else {
            0
        };
        q.write_buffer(&self.clean_uniform, 0, bytemuck::cast_slice(&u));
        let target = self.targets.clean.get_or_insert_with(|| {
            let texture = gpu::texture(d, "clean export output", self.size, OUTPUT);
            let view = texture.create_view(&Default::default());
            CleanTarget { texture, view }
        });
        gpu::draw(
            encoder,
            &target.view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.assembly,
            &[
                (&self.clean_group, &[]),
                (
                    if self.cell_active {
                        &self.cells_assembly.as_ref().unwrap()
                            [self.cells.as_ref().unwrap().displayed_map][usize::from(bloom)]
                    } else {
                        &self.targets.assembly_groups[source][usize::from(bloom)]
                    },
                    &[],
                ),
            ],
        );
    }

    pub fn begin_frame(
        &mut self,
        q: &wgpu::Queue,
        e: &Engine,
        camera: &Camera,
        project: &Project,
        samples: u32,
        highlight: f32,
    ) {
        self.cell_active = self.cells.is_some()
            && self.frame_prefs.cells.style == crate::cells::Style::Cells
            && !camera.trail_mode;
        if self.cell_active {
            self.cells.as_mut().unwrap().begin(
                q,
                e,
                camera,
                project,
                &self.frame_prefs,
                samples,
                highlight,
            );
            self.cell_dirty = true;
        }
        self.samples_taken = 0;
        self.bloom_valid = None;
        self.direct = samples == 1;
        let view = self.view(e, camera);
        q.write_buffer(&self.view_uniform, 0, bytemuck::cast_slice(&view));
        let mut particle = [0u32; 176];
        particle[..8].copy_from_slice(&view[..8]);
        particle[8] = 1.5f32.to_bits();
        particle[9] = 0.045f32.to_bits();
        let config = &project.configs[0];
        particle[12] = u32::from(config.color_by_cohort);
        particle[13] = highlight.to_bits();
        for i in 0..40 {
            particle[16 + i * 4 + 2] = (config.color_sensitivity as f32).to_bits();
            particle[16 + i * 4 + 3] = (config.color_offset as f32).to_bits();
        }
        q.write_buffer(&self.particle_uniform, 0, bytemuck::cast_slice(&particle));
        q.write_buffer(
            &self.accum_uniform,
            0,
            bytemuck::cast_slice(&[1.0 / samples.max(1) as f32, 0.0, 0.0, 0.0]),
        );
    }
    fn view(&self, e: &Engine, camera: &Camera) -> [u32; 12] {
        let mut u = [0; 12];
        for (i, v) in [
            e.sizing.canvas[0] as f32,
            e.sizing.canvas[1] as f32,
            self.size[0] as f32,
            self.size[1] as f32,
            camera.pan[0],
            camera.pan[1],
            camera.zoom,
        ]
        .iter()
        .enumerate()
        {
            u[i] = v.to_bits();
        }
        u
    }
    pub fn clear_accumulator(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.cell_active {
            self.cells.as_ref().unwrap().clear(encoder);
            return;
        }
        if !self.direct {
            let a = [Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.accum,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear temporal accumulation"),
                color_attachments: &a,
                ..Default::default()
            });
        }
    }
    pub fn sample(&mut self, encoder: &mut wgpu::CommandEncoder, e: &Engine, camera: &Camera) {
        self.sample_profiled(encoder, e, camera, None);
    }
    pub fn sample_profiled(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        e: &Engine,
        camera: &Camera,
        mut profile: Option<&mut crate::profiling::ProfileFrame>,
    ) {
        let start = if self.cell_active {
            None
        } else {
            profile.as_mut().and_then(|p| p.begin(encoder))
        };
        if self.cell_active {
            self.cells
                .as_mut()
                .unwrap()
                .sample(encoder, e, profile.as_deref_mut());
        } else if camera.trail_mode {
            gpu::draw(
                encoder,
                &self.targets.hdr,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                &self.trail,
                &[
                    (&self.trail_group, &[]),
                    (&self.trail_textures[e.current_canvas], &[]),
                ],
            );
        } else {
            let a = [Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particle camera"),
                color_attachments: &a,
                ..Default::default()
            });
            pass.set_pipeline(&self.particles);
            pass.set_bind_group(0, &self.particle_group, &[]);
            pass.draw(0..4, 0..e.sizing.entity_count);
        }
        if !self.direct && !self.cell_active {
            gpu::draw(
                encoder,
                &self.targets.accum,
                wgpu::LoadOp::Load,
                &self.accumulate,
                &[(&self.targets.accum_group, &[])],
            );
        }
        self.samples_taken += 1;
        self.bloom_valid = None;
        if let Some(p) = profile.as_mut() {
            p.end(encoder, crate::profiling::Stage::Drawing, start);
        }
    }

    pub fn present(
        &mut self,
        q: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        e: &Engine,
        camera: &Camera,
        prefs: &Preferences,
        overlay: Overlays,
    ) {
        self.present_profiled(q, encoder, e, camera, prefs, overlay, None);
    }
    pub fn present_profiled(
        &mut self,
        q: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        e: &Engine,
        camera: &Camera,
        prefs: &Preferences,
        overlay: Overlays,
        mut profile: Option<&mut crate::profiling::ProfileFrame>,
    ) {
        if self.cell_active && self.cell_dirty {
            let cell_start = profile.as_mut().and_then(|p| p.begin(encoder));
            let sharpen = self.frame_prefs.cells.sharpen > 0.0;
            self.cells.as_mut().unwrap().resolve(encoder, sharpen);
            self.cell_dirty = false;
            self.bloom_valid = None;
            if let Some(p) = profile.as_mut() {
                p.end(encoder, crate::profiling::Stage::CellsResolve, cell_start);
            }
        }
        let start = profile.as_mut().and_then(|p| p.begin(encoder));
        let source = usize::from(!self.direct);
        let bloom = prefs.bloom_enabled && prefs.bloom_intensity > 0.0;
        let bloom_key = [
            prefs.bloom_threshold.to_bits(),
            prefs.bloom_radius.to_bits(),
            source as u32,
            u32::from(self.cell_active),
        ];
        if bloom && self.bloom_valid != Some(bloom_key) {
            self.bloom_valid = Some(bloom_key);
            for i in 0..LEVELS {
                let size = if i == 0 {
                    self.size
                } else {
                    self.targets.mip_sizes[i - 1]
                };
                let u = [
                    (1.0 / size[0] as f32).to_bits(),
                    (1.0 / size[1] as f32).to_bits(),
                    prefs.bloom_threshold.to_bits(),
                    u32::from(i == 0),
                ];
                q.write_buffer(&self.down_uniforms[i], 0, bytemuck::cast_slice(&u));
                gpu::draw(
                    encoder,
                    &self.targets.mips[i],
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    &self.down,
                    &[(
                        if self.cell_active && i == 0 {
                            self.cells_bloom.as_ref().unwrap()
                        } else {
                            &self.targets.down_groups[source][i]
                        },
                        &[],
                    )],
                );
            }
            for i in (1..LEVELS).rev() {
                let size = self.targets.mip_sizes[i];
                q.write_buffer(
                    &self.up_uniforms[i],
                    0,
                    bytemuck::cast_slice(&[
                        1.0 / size[0] as f32,
                        1.0 / size[1] as f32,
                        prefs.bloom_radius,
                        0.0,
                    ]),
                );
                gpu::draw(
                    encoder,
                    &self.targets.mips[i - 1],
                    wgpu::LoadOp::Load,
                    &self.up,
                    &[(&self.targets.up_groups[i], &[])],
                );
            }
        }
        if let Some(p) = profile.as_mut() {
            p.end(encoder, crate::profiling::Stage::Bloom, start);
        }
        let assembly = profile.as_mut().and_then(|p| p.begin(encoder));
        let mut u = [0u32; 36];
        u[32..35].copy_from_slice(&self.frame_prefs.cells.theme.background().map(f32::to_bits));
        if self.cell_active {
            let cells = self.cells.as_ref().unwrap();
            u[35] = ((cells.padded[0] - cells.size[0]) as f32 * 0.5).to_bits();
        }
        u[..8].copy_from_slice(&self.view(e, camera)[..8]);
        for (i, value) in [
            (8, if bloom { prefs.bloom_intensity } else { 0.0 }),
            (9, prefs.brightness),
            (10, prefs.tonemap_softness.max(0.1)),
            (
                11,
                if overlay.walls {
                    prefs.field_opacity
                } else {
                    0.0
                },
            ),
            (12, overlay.reticle[0]),
            (13, overlay.reticle[1]),
            (14, overlay.radius),
            (15, overlay.angle),
            (
                17,
                if overlay.trails {
                    prefs.field_opacity
                } else {
                    0.0
                },
            ),
            (
                19,
                if self.cell_active || self.direct {
                    1.0
                } else {
                    0.0
                },
            ),
            (24, 1.0),
            (25, 1.0),
        ] {
            u[i] = value.to_bits();
        }
        u[7] = if self.cell_active {
            if self.cells.as_ref().unwrap().highlight >= 0.0 {
                2.0f32.to_bits()
            } else {
                1.0f32.to_bits()
            }
        } else {
            0
        };
        u[16] = overlay.style;
        if let Some((a, b)) = overlay.line {
            u[18] = 1.0f32.to_bits();
            u[28] = a[0].to_bits();
            u[29] = a[1].to_bits();
            u[30] = b[0].to_bits();
            u[31] = b[1].to_bits();
        }
        if let Some(crop) = overlay.crop {
            u[20] = crop[0].to_bits();
            u[21] = crop[1].to_bits();
            u[22] = 1.0f32.to_bits();
            u[23] = 0.65f32.to_bits();
        }
        q.write_buffer(&self.assembly_uniform, 0, bytemuck::cast_slice(&u));
        gpu::draw(
            encoder,
            &self.targets.output_view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.assembly,
            &[
                (&self.assembly_group, &[]),
                (
                    if self.cell_active {
                        &self.cells_assembly.as_ref().unwrap()
                            [self.cells.as_ref().unwrap().displayed_map][usize::from(bloom)]
                    } else {
                        &self.targets.assembly_groups[source][usize::from(bloom)]
                    },
                    &[],
                ),
            ],
        );
        if let Some(p) = profile.as_mut() {
            p.end(encoder, crate::profiling::Stage::Assembly, assembly);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_schedule_weights_actual_sample_count() {
        for steps in 1..=120 {
            for requested in 1..=32 {
                let s = blur_schedule(steps, requested);
                let actual = (0..steps)
                    .filter(|step| step % s.stride == s.sample_at)
                    .count() as u32;
                assert_eq!(actual, s.samples, "steps={steps}, requested={requested}");
            }
        }
        assert_eq!(blur_schedule(100, 8).samples, 9);
    }
    #[test]
    fn hidden_reticle_ignores_pointer_position() {
        let at = |reticle, radius| {
            Overlays {
                reticle,
                radius,
                style: 3,
                angle: 0.4,
                ..Default::default()
            }
            .normalized()
        };
        assert_eq!(at([0.1, 0.2], 0.0), at([0.7, 0.9], 0.0));
        assert_eq!(at([0.1, 0.2], 0.0), Overlays::default());
        assert_ne!(at([0.1, 0.2], 0.01), at([0.7, 0.9], 0.01));
        // A line preview follows the pointer even without a reticle.
        let line = |end| {
            Overlays {
                line: Some(([0.5; 2], end)),
                ..Default::default()
            }
            .normalized()
        };
        assert_ne!(line([0.1, 0.2]), line([0.3, 0.4]));
    }
}
