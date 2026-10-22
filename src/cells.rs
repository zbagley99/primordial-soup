//! GPU-only cellular appearance. Render resources never write simulation buffers.
use crate::{
    config::{Preferences, Project},
    coords::Camera,
    gpu::{self, Engine},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    #[default]
    Classic,
    Cells,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    #[default]
    Dark,
    Light,
}
impl Theme {
    /// Linear RGB, shared by pigment, selection dimming and final world masking.
    pub fn background(self) -> [f32; 3] {
        match self {
            Self::Dark => [0.008, 0.012, 0.02],
            Self::Light => [0.94, 0.927, 0.89],
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "StoredSettings")]
pub struct Settings {
    pub style: Style,
    pub theme: Theme,
    pub scale: f32,
    pub density: f32,
    pub relief: f32,
    pub wash: f32,
    pub structure_threshold: f32,
    pub membranes: f32,
    pub reference_palette: bool,
    pub contrast: f32,
    pub chromatic: f32,
    pub sharpen: f32,
    pub zoom_response: f32,
    pub detailed: bool,
    pub fast: bool,
}
/// Typed migration supports JSON and eframe's RON enums equally. Presence of
/// theme distinguishes the prototype's automatic palette from explicit choices.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct StoredSettings {
    style: Style,
    #[serde(deserialize_with = "present_theme")]
    theme: Option<Theme>,
    scale: f32,
    density: f32,
    relief: f32,
    wash: f32,
    structure_threshold: f32,
    membranes: f32,
    reference_palette: bool,
    contrast: f32,
    chromatic: f32,
    sharpen: f32,
    zoom_response: f32,
    detailed: bool,
    fast: bool,
    // Accept preferences written by the reverted microscope experiment.
    #[serde(rename = "microscope")]
    _retired_microscope: bool,
}
fn present_theme<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Theme>, D::Error> {
    Theme::deserialize(d).map(Some)
}
impl Default for StoredSettings {
    fn default() -> Self {
        let s = Settings::default();
        Self {
            style: s.style,
            theme: None,
            scale: s.scale,
            density: s.density,
            relief: s.relief,
            wash: s.wash,
            structure_threshold: s.structure_threshold,
            membranes: s.membranes,
            reference_palette: s.reference_palette,
            contrast: s.contrast,
            chromatic: s.chromatic,
            sharpen: s.sharpen,
            zoom_response: s.zoom_response,
            detailed: s.detailed,
            fast: s.fast,
            _retired_microscope: false,
        }
    }
}
impl From<StoredSettings> for Settings {
    fn from(s: StoredSettings) -> Self {
        Self {
            style: s.style,
            theme: s.theme.unwrap_or_default(),
            scale: s.scale,
            density: s.density,
            relief: s.relief,
            wash: s.wash,
            structure_threshold: s.structure_threshold,
            membranes: s.membranes,
            reference_palette: s.theme.is_some() && s.reference_palette,
            contrast: s.contrast,
            chromatic: s.chromatic,
            sharpen: s.sharpen,
            zoom_response: s.zoom_response,
            detailed: s.detailed,
            fast: s.fast,
        }
    }
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            style: Style::Classic,
            theme: Theme::Dark,
            scale: 0.85,
            density: 0.205,
            relief: 0.351,
            wash: 0.25,
            structure_threshold: 1.02,
            membranes: 0.77,
            reference_palette: false,
            contrast: 0.92,
            chromatic: 0.06,
            sharpen: 0.12,
            zoom_response: 1.0,
            detailed: false,
            fast: true,
        }
    }
}
impl Settings {
    pub fn sanitize(&mut self) {
        let d = Self::default();
        for (v, f, lo, hi) in [
            (&mut self.scale, d.scale, 0.5, 4.0),
            (&mut self.density, d.density, 0.0, 1.0),
            (&mut self.relief, d.relief, 0.0, 0.9),
            (&mut self.wash, d.wash, 0.0, 1.0),
            (
                &mut self.structure_threshold,
                d.structure_threshold,
                0.0,
                4.0,
            ),
            (&mut self.membranes, d.membranes, 0.0, 1.0),
            (&mut self.contrast, d.contrast, 0.0, 2.0),
            (&mut self.chromatic, d.chromatic, 0.0, 0.5),
            (&mut self.sharpen, d.sharpen, 0.0, 0.5),
            (&mut self.zoom_response, d.zoom_response, 0.2, 1.0),
        ] {
            *v = if v.is_finite() { v.clamp(lo, hi) } else { f };
        }
    }
}
/// Bound each integer sum independently of total population, then add batches in f32.
/// This keeps tails visible even at maximum radius and the largest supported worlds.
pub const PARTICLES_PER_BATCH: u32 = 262_144;
/// Batch size for sprites wide enough that faint tails need a larger quantum.
const BATCH_FLOOR: u32 = 65_536;
/// Upper bound on the compact sprite list; past it the particle batches remain.
const SPRITE_LIST_BYTES: u64 = 256 << 20;
/// Cover the widest sprite, Gaussian pyramid, cubic reconstruction, broad
/// displacement and finishing. The physical guard covers odd mip ratios and
/// tiny outputs, where a reference-pixel margin alone is insufficient.
pub fn padding(size: [u32; 2]) -> u32 {
    let scale = size[0].min(size[1]) as f32 / 1080.0;
    let highest_mip = ((24.0 * scale).max(1.0).log2() - 2.0)
        .ceil()
        .clamp(0.0, 5.0) as i32;
    let blur_support = 5.5 * 2.0f32.powi(highest_mip + 1) - 3.5;
    (blur_support + (24.0 * scale).max(0.75) + scale + scale.max(1.0) + 8.0).ceil() as u32
}
/// Bytes of the largest Cells storage binding, the interleaved color and mass
/// accumulator, for a working size.
pub fn largest_binding_bytes(size: [u32; 2]) -> u64 {
    let pad = padding(size);
    size.iter().map(|n| u64::from(n + 2 * pad)).product::<u64>() * 16
}
/// The largest working size at the same aspect whose Cells bindings fit
/// `limit` bytes; unchanged when it already fits.
pub fn fit_working_size(size: [u32; 2], limit: u64) -> [u32; 2] {
    let mut fitted = size;
    while largest_binding_bytes(fitted) > limit && fitted.iter().all(|v| *v > 1) {
        let shrink = (limit as f64 / largest_binding_bytes(fitted) as f64).sqrt() * 0.99;
        fitted = fitted.map(|v| ((v as f64 * shrink).floor() as u32).max(1));
    }
    fitted
}
/// Guaranteed sum bound includes nearest-integer rounding and f32 conversion margin.
pub fn quantum(n: u32, batch: u32) -> Result<f32> {
    let cap = u32::MAX / n.clamp(1, batch.min(PARTICLES_PER_BATCH).max(1));
    let q = (((cap.saturating_sub(2)) as f64 / 0.62).floor() as u64).min(1 << 24);
    ensure!(q >= 32, "Cells precision budget exceeded");
    Ok(q as f32)
}
fn tex(
    d: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    mips: u32,
) -> wgpu::Texture {
    d.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | if format == wgpu::TextureFormat::R8Unorm {
                wgpu::TextureUsages::empty()
            } else {
                wgpu::TextureUsages::STORAGE_BINDING
            },
        view_formats: &[],
    })
}
fn view(t: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..Default::default()
    })
}
fn b(i: u32, v: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: i,
        resource: v.as_entire_binding(),
    }
}
fn t(i: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: i,
        resource: wgpu::BindingResource::TextureView(v),
    }
}
fn group(
    d: &wgpu::Device,
    l: &wgpu::BindGroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Cells"),
        layout: l,
        entries,
    })
}
fn compute(d: &wgpu::Device, name: &str, entry: &str) -> wgpu::ComputePipeline {
    d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &gpu::module(d, name),
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn dispatch(
    e: &mut wgpu::CommandEncoder,
    p: &wgpu::ComputePipeline,
    g: &wgpu::BindGroup,
    n: [u32; 2],
) {
    let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Cells"),
        ..Default::default()
    });
    pass.set_pipeline(p);
    pass.set_bind_group(0, g, &[]);
    pass.dispatch_workgroups(n[0], n[1], 1);
}
fn raster_with(
    d: &wgpu::Device,
    name: &str,
    formats: &[wgpu::TextureFormat],
    write_mask: wgpu::ColorWrites,
) -> wgpu::RenderPipeline {
    let m = gpu::module(d, name);
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: None,
        vertex: wgpu::VertexState {
            module: &m,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &m,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &formats
                .iter()
                .map(|f| {
                    Some(wgpu::ColorTargetState {
                        format: *f,
                        blend: None,
                        write_mask,
                    })
                })
                .collect::<Vec<_>>(),
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
fn clear(e: &mut wgpu::CommandEncoder, views: &[&wgpu::TextureView]) {
    let attachments: Vec<_> = views
        .iter()
        .map(|v| {
            Some(wgpu::RenderPassColorAttachment {
                view: v,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let _pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Cells clear"),
        color_attachments: &attachments,
        ..Default::default()
    });
}
pub struct Cells {
    pub size: [u32; 2],
    pub padded: [u32; 2],
    pub grid: [u32; 2],
    pub map_size: [u32; 2],
    pub detailed: bool,
    pub fast: bool,
    uniform: wgpu::Buffer,
    pub counts: wgpu::Buffer,
    _accum: wgpu::Buffer,
    _selected: wgpu::Buffer,
    sites: wgpu::Buffer,
    pub membrane_history: wgpu::Buffer,
    membrane_accumulate: wgpu::ComputePipeline,
    membrane_group: wgpu::BindGroup,
    membranes_active: bool,
    single_sample: bool,
    pub density: wgpu::Texture,
    _density_temp: wgpu::Texture,
    _density_views: Vec<wgpu::TextureView>,
    _dummy: wgpu::Texture,
    dummy: wgpu::TextureView,
    pub images: [wgpu::Texture; 2],
    image_views: [wgpu::TextureView; 2],
    masks: [wgpu::Texture; 2],
    mask_views: [wgpu::TextureView; 2],
    _wash: wgpu::Texture,
    _wash_views: Vec<wgpu::TextureView>,
    pub maps: [wgpu::Texture; 2],
    pub map_views: [wgpu::TextureView; 2],
    pub displayed_map: usize,
    _scene: wgpu::Texture,
    pub scene: wgpu::TextureView,
    _marks: wgpu::Texture,
    pub marks: wgpu::TextureView,
    count: wgpu::ComputePipeline,
    horizontal: wgpu::ComputePipeline,
    vertical: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    count_group: wgpu::BindGroup,
    horizontal_group: wgpu::BindGroup,
    vertical_group: wgpu::BindGroup,
    density_down: Vec<wgpu::BindGroup>,
    sprites: wgpu::RenderPipeline,
    sprite_group: wgpu::BindGroup,
    average: wgpu::ComputePipeline,
    average_groups: [wgpu::BindGroup; 2],
    blur: wgpu::ComputePipeline,
    blur_first_pipeline: wgpu::ComputePipeline,
    blur_first: [wgpu::BindGroup; 2],
    blur_groups: Vec<wgpu::BindGroup>,
    make_map: wgpu::ComputePipeline,
    map_groups: [wgpu::BindGroup; 2],
    _mass: wgpu::Texture,
    _shaded: wgpu::Texture,
    shaded_size: [u32; 2],
    shade: wgpu::ComputePipeline,
    shade_groups: [[wgpu::BindGroup; 2]; 2],
    finish: wgpu::ComputePipeline,
    finish_groups: [[wgpu::BindGroup; 2]; 2],
    pub active: usize,
    pub samples: u32,
    pub displayed_camera: Camera,
    pub highlight: f32,
    pub total_bytes: u64,
    frame_camera: Camera,
    wrapping: bool,
    batch_limit: u32,
    batch_settings: wgpu::Buffer,
    plan: wgpu::Buffer,
    indirect: wgpu::Buffer,
    box_rows: wgpu::ComputePipeline,
    box_columns: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    plan_batches: wgpu::ComputePipeline,
    box_rows_group: wgpu::BindGroup,
    box_columns_group: wgpu::BindGroup,
    prepare_group: wgpu::BindGroup,
    plan_group: wgpu::BindGroup,
    slot_groups: Vec<wgpu::BindGroup>,
    /// Listed sprite copies that fit the compact batch.
    capacity: u32,
    sprite_list: wgpu::Buffer,
    radii: wgpu::Buffer,
    tiled: Option<crate::cells_tiled::Tiled>,
    analytic_distances: bool,
    /// Particle batches encoded per sample this frame (odd).
    batches: u32,
}
impl Cells {
    pub fn new(d: &wgpu::Device, e: &Engine, size: [u32; 2], settings: &Settings) -> Result<Self> {
        let scale = size[0].min(size[1]) as f32 / 1080.0;
        let pad = padding(size);
        let padded = size.map(|n| n + 2 * pad);
        let aspect = e.sizing.canvas[0] as f32 / e.sizing.canvas[1] as f32;
        let base = if settings.detailed { 512.0 } else { 256.0 };
        let grid = [
            (base * aspect.sqrt()).round() as u32,
            (base / aspect.sqrt()).round() as u32,
        ];
        let spacing = (8.0 * scale).max(1.0);
        let map_size = padded.map(|n| (n as f32 / spacing).floor() as u32 + 1);
        ensure!(
            padded
                .iter()
                .chain(grid.iter())
                .all(|n| *n > 0 && *n <= d.limits().max_texture_dimension_2d),
            "Cells texture dimensions exceed GPU limits"
        );
        let bytes = u64::from(padded[0]) * u64::from(padded[1]) * 4;
        let buffer_limit =
            u64::from(d.limits().max_storage_buffer_binding_size).min(d.limits().max_buffer_size);
        ensure!(
            bytes * 4 <= buffer_limit,
            "Cells framebuffer exceeds GPU buffer limits"
        );
        ensure!(
            d.limits().max_storage_buffers_per_shader_stage >= 4,
            "Cells needs four storage bindings per shader stage"
        );
        quantum(e.sizing.entity_count, PARTICLES_PER_BATCH)?;
        let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cells frame"),
            contents: &[0; 176],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let make_buffer = |label, size| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let counts = make_buffer(
            "Cells occupancy",
            u64::from(grid[0]) * u64::from(grid[1]) * 4,
        );
        let accum = make_buffer("Cells fixed point color and mass", bytes * 4);
        let selected = make_buffer("Cells fixed point selected mass", bytes);
        let sites = make_buffer("Cells nearest two distances", bytes * 2);
        let membrane_history = make_buffer("Cells temporal membrane appearance", bytes * 2);
        let levels = grid[0].min(grid[1]).ilog2() + 1;
        let density = tex(
            d,
            "Cells scalar density",
            grid,
            wgpu::TextureFormat::R32Float,
            levels,
        );
        let density_views = (0..levels).map(|l| view(&density, l)).collect::<Vec<_>>();
        let density_all = density.create_view(&Default::default());
        let temp = tex(
            d,
            "Cells density horizontal",
            grid,
            wgpu::TextureFormat::R32Float,
            1,
        );
        let temp_view = view(&temp, 0);
        let dummy_texture = tex(
            d,
            "Cells raster attachment",
            padded,
            wgpu::TextureFormat::R8Unorm,
            1,
        );
        let dummy = view(&dummy_texture, 0);
        let images = std::array::from_fn(|_| {
            tex(
                d,
                "Cells temporal color and mass",
                padded,
                wgpu::TextureFormat::Rgba32Float,
                1,
            )
        });
        let image_views = std::array::from_fn(|i| view(&images[i], 0));
        let masks = std::array::from_fn(|_| {
            tex(
                d,
                "Cells temporal selected mass",
                padded,
                wgpu::TextureFormat::R32Float,
                1,
            )
        });
        let mask_views = std::array::from_fn(|i| view(&masks[i], 0));
        let wash_size = padded.map(|n| (n / 2).max(1));
        let wash_levels = 6.min(wash_size[0].min(wash_size[1]).ilog2() + 1);
        let wash = tex(
            d,
            "Cells unthresholded wash",
            wash_size,
            // Filterable, so the wash reconstruction can use bilinear taps.
            wgpu::TextureFormat::Rgba16Float,
            wash_levels,
        );
        let wash_views = (0..wash_levels).map(|l| view(&wash, l)).collect::<Vec<_>>();
        let wash_all = wash.create_view(&Default::default());
        let maps = std::array::from_fn(|_| {
            tex(
                d,
                "Cells displayed displacement",
                map_size,
                wgpu::TextureFormat::Rgba32Float,
                1,
            )
        });
        let map_views = std::array::from_fn(|i| view(&maps[i], 0));
        let scene_tex = tex(
            d,
            "Cells clean pigment",
            size,
            wgpu::TextureFormat::Rgba16Float,
            1,
        );
        let scene = view(&scene_tex, 0);
        let marks_tex = tex(
            d,
            "Cells bloom marks",
            size,
            wgpu::TextureFormat::Rgba16Float,
            1,
        );
        let marks = view(&marks_tex, 0);
        let count = compute(d, "cellsDensity.wgsl", "count");
        let horizontal = compute(d, "cellsDensity.wgsl", "horizontal");
        let vertical = compute(d, "cellsDensity.wgsl", "vertical");
        let down = compute(d, "cellsDensity.wgsl", "down");
        let count_group = group(
            d,
            &count.get_bind_group_layout(0),
            &[b(0, &uniform), b(1, &e.entities), b(2, &counts)],
        );
        let horizontal_group = group(
            d,
            &horizontal.get_bind_group_layout(0),
            &[b(0, &uniform), b(2, &counts), t(4, &temp_view)],
        );
        let vertical_group = group(
            d,
            &vertical.get_bind_group_layout(0),
            &[b(0, &uniform), t(3, &temp_view), t(4, &density_views[0])],
        );
        let density_down = (1..levels)
            .map(|i| {
                group(
                    d,
                    &down.get_bind_group_layout(0),
                    &[
                        t(3, &density_views[i as usize - 1]),
                        t(4, &density_views[i as usize]),
                    ],
                )
            })
            .collect();
        // Sprite planning (cellsPlan.wgsl). Wrapping can change without a
        // rebuild, so the list holds all nine copies when the buffer limit allows.
        let population = e.sizing.entity_count;
        let entry_limit = (buffer_limit.min(SPRITE_LIST_BYTES) / 8).min(1 << 28);
        let capacity = (u64::from(population) * 9).min(entry_limit).max(1) as u32;
        let slots = population.div_ceil(BATCH_FLOOR) + 1;
        let plan = make_buffer("Cells sprite plan", 16);
        let indirect = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cells sprite batches"),
            size: u64::from(slots) * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let batch_settings = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cells sprite batch settings"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | if cfg!(not(target_arch = "wasm32")) {
                    wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        });
        let sprite_list = make_buffer("Cells listed sprite copies", u64::from(capacity) * 8);
        let radii = make_buffer("Cells sprite radii", u64::from(population) * 4);
        let rows = make_buffer(
            "Cells occupancy row windows",
            u64::from(grid[0]) * u64::from(grid[1]) * 4,
        );
        let box_rows = compute(d, "cellsPlan.wgsl", "box_rows");
        let box_columns = compute(d, "cellsPlan.wgsl", "box_columns");
        let prepare = compute(d, "cellsPlan.wgsl", "prepare");
        let plan_batches = compute(d, "cellsPlan.wgsl", "plan_batches");
        let box_rows_group = group(
            d,
            &box_rows.get_bind_group_layout(0),
            &[b(0, &uniform), b(2, &counts), b(3, &rows)],
        );
        let box_columns_group = group(
            d,
            &box_columns.get_bind_group_layout(0),
            &[
                b(0, &uniform),
                b(3, &rows),
                b(4, &plan),
                b(5, &batch_settings),
            ],
        );
        let prepare_group = group(
            d,
            &prepare.get_bind_group_layout(0),
            &[
                b(0, &uniform),
                b(1, &e.entities),
                b(4, &plan),
                b(5, &batch_settings),
                t(6, &density_all),
                b(7, &sprite_list),
                b(8, &radii),
            ],
        );
        let plan_group = group(
            d,
            &plan_batches.get_bind_group_layout(0),
            &[
                b(0, &uniform),
                b(4, &plan),
                b(5, &batch_settings),
                b(9, &indirect),
            ],
        );
        // The attachment only satisfies WebGPU; sprites write storage buffers.
        let sprites = raster_with(
            d,
            "cellsSprites.wgsl",
            &[wgpu::TextureFormat::R8Unorm],
            wgpu::ColorWrites::empty(),
        );
        let sprite_group = group(
            d,
            &sprites.get_bind_group_layout(0),
            &[
                b(0, &uniform),
                b(1, &e.entities),
                t(2, &density_all),
                b(3, &accum),
                b(4, &selected),
                b(5, &plan),
                b(6, &sprite_list),
                b(7, &radii),
                b(8, &sites),
                b(9, &batch_settings),
            ],
        );
        // One small uniform per encoded batch: its index among the batches.
        let stride = u64::from(d.limits().min_uniform_buffer_offset_alignment).max(16);
        let slot_buffer = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cells sprite batch indices"),
            contents: &(0..slots)
                .flat_map(|k| {
                    let mut v = vec![0u8; stride as usize];
                    v[..4].copy_from_slice(&k.to_le_bytes());
                    v
                })
                .collect::<Vec<_>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let slot_groups = (0..u64::from(slots))
            .map(|k| {
                group(
                    d,
                    &sprites.get_bind_group_layout(1),
                    &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &slot_buffer,
                            offset: k * stride,
                            size: wgpu::BufferSize::new(16),
                        }),
                    }],
                )
            })
            .collect();
        let membrane_accumulate = compute(d, "cellsMembranes.wgsl", "accumulate");
        let membrane_group = group(
            d,
            &membrane_accumulate.get_bind_group_layout(0),
            &[
                b(0, &uniform),
                b(1, &sites),
                b(2, &membrane_history),
                t(3, &density_all),
            ],
        );
        let average = compute(d, "cellsAverage.wgsl", "main");
        let average_groups = std::array::from_fn(|i| {
            group(
                d,
                &average.get_bind_group_layout(0),
                &[
                    b(0, &uniform),
                    b(1, &accum),
                    b(2, &selected),
                    b(5, &plan),
                    t(6, &image_views[i]),
                    t(7, &mask_views[i]),
                    t(8, &image_views[1 - i]),
                    t(9, &mask_views[1 - i]),
                ],
            )
        });
        // Mass for relief heights, extracted by the first wash reduction.
        let mass_tex = tex(
            d,
            "Cells relief mass",
            padded,
            wgpu::TextureFormat::R32Float,
            1,
        );
        let mass = view(&mass_tex, 0);
        let blur = compute(d, "cellsBlur.wgsl", "main");
        let blur_first_pipeline = compute(d, "cellsBlurFirst.wgsl", "first");
        let blur_first = std::array::from_fn(|i| {
            group(
                d,
                &blur_first_pipeline.get_bind_group_layout(0),
                &[t(0, &image_views[i]), t(1, &wash_views[0]), t(2, &mass)],
            )
        });
        let blur_groups = (1..wash_levels)
            .map(|l| {
                group(
                    d,
                    &blur.get_bind_group_layout(0),
                    &[
                        t(0, &wash_views[l as usize - 1]),
                        t(1, &wash_views[l as usize]),
                    ],
                )
            })
            .collect();
        let make_map = compute(d, "cellsPost.wgsl", "build_map");
        let map_groups = std::array::from_fn(|j| {
            group(
                d,
                &make_map.get_bind_group_layout(0),
                &[b(0, &uniform), t(5, &map_views[j]), t(9, &mass)],
            )
        });
        // Sharpening samples neighbouring shaded pixels; the margin covers its
        // widest offset plus the bilinear footprint.
        let margin = scale.max(1.0).ceil() as u32 + 1;
        let shaded_size = size.map(|n| n + 2 * margin);
        let shaded_tex = tex(
            d,
            "Cells shaded before finishing",
            shaded_size,
            wgpu::TextureFormat::Rgba16Float,
            1,
        );
        let shaded = view(&shaded_tex, 0);
        let wash_sampler = d.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cells wash"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler = wgpu::BindGroupEntry {
            binding: 14,
            resource: wgpu::BindingResource::Sampler(&wash_sampler),
        };
        let shade = compute(d, "cellsPost.wgsl", "shade_pass");
        let shade_groups = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                group(
                    d,
                    &shade.get_bind_group_layout(0),
                    &[
                        b(0, &uniform),
                        t(1, &image_views[i]),
                        t(2, &mask_views[i]),
                        t(3, &wash_all),
                        t(4, &map_views[j]),
                        t(6, &density_all),
                        b(7, &membrane_history),
                        t(9, &mass),
                        t(10, &scene),
                        t(11, &marks),
                        t(12, &shaded),
                        sampler.clone(),
                    ],
                )
            })
        });
        let finish = compute(d, "cellsPost.wgsl", "finish");
        let finish_groups = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                group(
                    d,
                    &finish.get_bind_group_layout(0),
                    &[
                        b(0, &uniform),
                        t(1, &image_views[i]),
                        t(2, &mask_views[i]),
                        t(3, &wash_all),
                        t(4, &map_views[j]),
                        t(6, &density_all),
                        b(7, &membrane_history),
                        t(9, &mass),
                        t(10, &scene),
                        t(13, &shaded),
                        sampler.clone(),
                    ],
                )
            })
        });
        let pixels = u64::from(padded[0]) * u64::from(padded[1]);
        let mut total_bytes = pixels * (20 + 32 + 8 + 1 + 16)
            + u64::from(size[0]) * u64::from(size[1]) * 16
            + u64::from(grid[0]) * u64::from(grid[1]) * 12
            + u64::from(capacity) * 8
            + u64::from(population) * 4
            + u64::from(slots) * (32 + stride)
            + 64;
        for i in 0..levels {
            total_bytes += u64::from((grid[0] >> i).max(1)) * u64::from((grid[1] >> i).max(1)) * 4;
        }
        for i in 0..wash_levels {
            total_bytes +=
                u64::from((wash_size[0] >> i).max(1)) * u64::from((wash_size[1] >> i).max(1)) * 8;
        }
        total_bytes += u64::from(map_size[0]) * u64::from(map_size[1]) * 32 + 176;
        total_bytes += u64::from(shaded_size[0]) * u64::from(shaded_size[1]) * 8 + pixels * 4;
        let mut cells = Self {
            size,
            padded,
            grid,
            map_size,
            detailed: settings.detailed,
            fast: settings.fast,
            uniform,
            counts,
            _accum: accum,
            _selected: selected,
            sites,
            membrane_history,
            membrane_accumulate,
            membrane_group,
            membranes_active: false,
            single_sample: false,
            density,
            _density_temp: temp,
            _density_views: density_views,
            _dummy: dummy_texture,
            dummy,
            images,
            image_views,
            masks,
            mask_views,
            _wash: wash,
            _wash_views: wash_views,
            maps,
            map_views,
            displayed_map: 0,
            _scene: scene_tex,
            scene,
            _marks: marks_tex,
            marks,
            count,
            horizontal,
            vertical,
            down,
            count_group,
            horizontal_group,
            vertical_group,
            density_down,
            sprites,
            sprite_group,
            average,
            average_groups,
            blur,
            blur_first_pipeline,
            blur_first,
            blur_groups,
            make_map,
            map_groups,
            _mass: mass_tex,
            _shaded: shaded_tex,
            shaded_size,
            shade,
            shade_groups,
            finish,
            finish_groups,
            active: 0,
            samples: 0,
            displayed_camera: Camera::default(),
            highlight: -1.0,
            total_bytes,
            frame_camera: Camera::default(),
            wrapping: false,
            batch_limit: PARTICLES_PER_BATCH,
            batch_settings,
            plan,
            indirect,
            box_rows,
            box_columns,
            prepare,
            plan_batches,
            box_rows_group,
            box_columns_group,
            prepare_group,
            plan_group,
            slot_groups,
            capacity,
            sprite_list,
            radii,
            tiled: None,
            analytic_distances: false,
            batches: 1,
        };
        if settings.fast {
            cells.enable_tiled(d, e, None);
        }
        Ok(cells)
    }
    pub fn begin(
        &mut self,
        q: &wgpu::Queue,
        e: &Engine,
        camera: &Camera,
        p: &Project,
        prefs: &Preferences,
        samples: u32,
        highlight: f32,
    ) {
        self.samples = 0;
        self.single_sample = samples.max(1) == 1;
        self.active = 0;
        self.frame_camera = *camera;
        self.wrapping = p.world.boundary_conditions == 1;
        self.highlight = highlight;
        let s = &prefs.cells;
        let scale = self.size[0].min(self.size[1]) as f32 / 1080.0;
        let transition = ((camera.zoom - 2.0) / 6.0).clamp(0.0, 1.0);
        let membrane_blend = s.membranes * transition * transition * (3.0 - 2.0 * transition);
        self.membranes_active = membrane_blend > 0.0;
        let maximum_radius = (1.5 * s.scale * camera.zoom.powf(s.zoom_response) * 2.0)
            .max(0.75 / scale)
            .min(24.0);
        self.batch_limit = if maximum_radius > 6.0 {
            BATCH_FLOOR
        } else {
            PARTICLES_PER_BATCH
        };
        let quantum = quantum(e.sizing.entity_count, self.batch_limit).unwrap();
        let config = p.config();
        let background = s.theme.background();
        let values = [
            e.sizing.canvas[0] as f32,
            e.sizing.canvas[1] as f32,
            self.size[0] as f32,
            self.size[1] as f32,
            camera.pan[0],
            camera.pan[1],
            camera.zoom,
            p.world.boundary_conditions as f32,
            self.padded[0] as f32,
            self.padded[1] as f32,
            (self.padded[0] - self.size[0]) as f32 * 0.5,
            (self.padded[1] - self.size[1]) as f32 * 0.5,
            self.grid[0] as f32,
            self.grid[1] as f32,
            self.map_size[0] as f32,
            self.map_size[1] as f32,
            s.scale,
            s.density,
            s.relief,
            s.wash,
            u32::from(s.reference_palette) as f32,
            s.contrast,
            s.chromatic,
            s.sharpen,
            config.color_sensitivity as f32,
            config.color_offset as f32,
            u32::from(config.color_by_cohort) as f32,
            highlight,
            1.0 / quantum,
            1.0 / samples.max(1) as f32,
            scale,
            quantum,
            prefs.brightness,
            prefs.tonemap_softness.max(0.1),
            s.zoom_response,
            e.sizing.entity_count as f32,
            background[0],
            background[1],
            background[2],
            u32::from(s.theme == Theme::Light) as f32,
            s.structure_threshold,
            membrane_blend,
            u32::from(self.analytic_distances) as f32,
            0.0,
        ];
        q.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&values));
        // An odd count of particle batches ends on the same image as the single
        // compact batch, so the CPU knows the final image either way.
        let population = e.sizing.entity_count;
        self.batches = population.div_ceil(self.batch_limit) | 1;
        let settings = [
            self.batch_limit,
            self.batches,
            self.capacity,
            if self.wrapping { 9 } else { 1 },
            population,
            population.div_ceil(self.batches),
            self.slot_groups.len() as u32,
            0,
        ];
        q.write_buffer(&self.batch_settings, 0, bytemuck::cast_slice(&settings));
    }
    pub fn clear(&self, e: &mut wgpu::CommandEncoder) {
        clear(e, &[&self.image_views[0], &self.mask_views[0]]);
        if self.membranes_active && !self.single_sample {
            e.clear_buffer(&self.membrane_history, 0, None);
        }
    }
    pub fn sample(
        &mut self,
        e: &mut wgpu::CommandEncoder,
        engine: &Engine,
        mut profile: Option<&mut crate::profiling::ProfileFrame>,
    ) {
        let start = profile.as_mut().and_then(|p| p.begin(e));
        e.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cells density"),
                ..Default::default()
            });
            let mut dispatch = |p: &wgpu::ComputePipeline, g: &wgpu::BindGroup, n: [u32; 2]| {
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                pass.dispatch_workgroups(n[0], n[1], 1);
            };
            dispatch(
                &self.count,
                &self.count_group,
                [engine.sizing.entity_count.div_ceil(256), 1],
            );
            dispatch(
                &self.horizontal,
                &self.horizontal_group,
                self.grid.map(|v| v.div_ceil(8)),
            );
            dispatch(
                &self.vertical,
                &self.vertical_group,
                self.grid.map(|v| v.div_ceil(8)),
            );
            for (i, g) in self.density_down.iter().enumerate() {
                dispatch(
                    &self.down,
                    g,
                    self.grid.map(|v| (v >> (i + 1)).max(1).div_ceil(8)),
                );
            }
        }
        if let Some(p) = profile.as_mut() {
            p.end(e, crate::profiling::Stage::CellsDensity, start);
        }
        let start = profile.as_mut().and_then(|p| p.begin(e));
        let planning = profile.as_mut().and_then(|p| p.begin(e));
        // Plan the batches on the GPU: overlap bound, reachable copies, batch mode.
        e.clear_buffer(&self.plan, 0, None);
        let cells = self.grid.map(|v| v.div_ceil(8));
        {
            let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cells planning"),
                ..Default::default()
            });
            let mut dispatch = |p: &wgpu::ComputePipeline, g: &wgpu::BindGroup, n: [u32; 2]| {
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                pass.dispatch_workgroups(n[0], n[1], 1);
            };
            dispatch(&self.box_rows, &self.box_rows_group, cells);
            dispatch(&self.box_columns, &self.box_columns_group, cells);
            dispatch(
                &self.prepare,
                &self.prepare_group,
                [engine.sizing.entity_count.div_ceil(256), 1],
            );
            dispatch(&self.plan_batches, &self.plan_group, [1, 1]);
        }
        // Reversed positive-float distance keys: zero means absent, larger is
        // nearer. Preserve the two winners across every particle batch.
        if self.membranes_active && self.tiled.is_none() {
            e.clear_buffer(&self.sites, 0, None);
        }
        if let Some(p) = profile.as_mut() {
            p.end(e, crate::profiling::Stage::CellsPlanning, planning);
        }
        if let Some(tiled) = &self.tiled {
            let start = profile.as_mut().and_then(|p| p.begin(e));
            tiled.encode(e, self.active);
            if let Some(p) = profile.as_mut() {
                p.end(e, crate::profiling::Stage::CellsTiled, start);
            }
        }
        // The average pass clears the integer sums it reads, so they start at
        // zero here. Passes the plan leaves empty draw and dispatch nothing.
        let first_image = self.active;
        for k in 0..self.batches {
            let raster = profile.as_mut().and_then(|p| p.begin(e));
            {
                let a = [Some(wgpu::RenderPassColorAttachment {
                    view: &self.dummy,
                    depth_slice: None,
                    resolve_target: None,
                    // Never written, so loading and storing it costs no clear.
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Cells additive integer sprites"),
                    color_attachments: &a,
                    ..Default::default()
                });
                pass.set_pipeline(&self.sprites);
                pass.set_bind_group(0, &self.sprite_group, &[]);
                pass.set_bind_group(1, &self.slot_groups[k as usize], &[]);
                pass.draw_indirect(&self.indirect, u64::from(k) * 16);
            }
            if let Some(p) = profile.as_mut() {
                p.end(e, crate::profiling::Stage::CellsRaster, raster);
            }
            let average = profile.as_mut().and_then(|p| p.begin(e));
            let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cells"),
                ..Default::default()
            });
            pass.set_pipeline(&self.average);
            pass.set_bind_group(0, &self.average_groups[first_image ^ (k as usize & 1)], &[]);
            let slots = self.slot_groups.len() as u64;
            pass.dispatch_workgroups_indirect(&self.indirect, (slots + u64::from(k)) * 16);
            drop(pass);
            if let Some(p) = profile.as_mut() {
                p.end(e, crate::profiling::Stage::CellsAverage, average);
            }
        }
        // Batches are odd in number, and a compact plan runs only the first.
        self.active = first_image ^ 1;
        if self.membranes_active {
            let membranes = profile.as_mut().and_then(|p| p.begin(e));
            dispatch(
                e,
                &self.membrane_accumulate,
                &self.membrane_group,
                self.padded.map(|v| v.div_ceil(8)),
            );
            if let Some(p) = profile.as_mut() {
                p.end(e, crate::profiling::Stage::CellsMembranes, membranes);
            }
        }
        self.samples += 1;
        if let Some(p) = profile.as_mut() {
            p.end(e, crate::profiling::Stage::CellsSprites, start);
        }
    }
    /// `sharpen` must match the frame's Cells sharpening (nonzero), which needs
    /// the separate finishing pass.
    pub fn resolve(&mut self, e: &mut wgpu::CommandEncoder, sharpen: bool) {
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cells resolve"),
            ..Default::default()
        });
        let mut dispatch = |p: &wgpu::ComputePipeline, g: &wgpu::BindGroup, n: [u32; 2]| {
            pass.set_pipeline(p);
            pass.set_bind_group(0, g, &[]);
            pass.dispatch_workgroups(n[0], n[1], 1);
        };
        let size = self.padded.map(|n| (n / 2).max(1));
        dispatch(
            &self.blur_first_pipeline,
            &self.blur_first[self.active],
            size.map(|v| v.div_ceil(8)),
        );
        for (i, g) in self.blur_groups.iter().enumerate() {
            dispatch(
                &self.blur,
                g,
                size.map(|v| (v >> (i + 1)).max(1).div_ceil(8)),
            );
        }
        let next = 1 - self.displayed_map;
        dispatch(
            &self.make_map,
            &self.map_groups[next],
            self.map_size.map(|v| v.div_ceil(8)),
        );
        dispatch(
            &self.shade,
            &self.shade_groups[self.active][next],
            self.shaded_size.map(|v| v.div_ceil(8)),
        );
        if sharpen {
            dispatch(
                &self.finish,
                &self.finish_groups[self.active][next],
                self.size.map(|v| v.div_ceil(8)),
            );
        }
        self.displayed_map = next;
        self.displayed_camera = self.frame_camera;
    }
    pub fn selected_texture(&self) -> &wgpu::Texture {
        &self.masks[self.active]
    }
    /// Particle batches encoded per sample. The GPU runs a single compact pass
    /// whenever its overlap bound allows, so this is an upper bound, not a count
    /// of executed passes.
    pub fn max_sprite_batches(&self) -> u32 {
        self.batches
    }
    /// Completed GPU plan for diagnostic readback outside a timed scene:
    /// overlap bound, listed copies, compact/fallback mode, and reserved word.
    pub fn sprite_plan_buffer(&self) -> &wgpu::Buffer {
        &self.plan
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sprite_batch_settings_buffer(&self) -> &wgpu::Buffer {
        &self.batch_settings
    }
    /// Enable Fast Cells gathering; unsupported adapters retain the raster path.
    pub fn enable_tiled(
        &mut self,
        d: &wgpu::Device,
        engine: &Engine,
        index_limit: Option<u32>,
    ) -> bool {
        if self.tiled.is_some() {
            self.analytic_distances = true;
            return true;
        }
        self.tiled = crate::cells_tiled::Tiled::new(
            d,
            &self.uniform,
            &engine.entities,
            &self.plan,
            &self.sprite_list,
            &self.radii,
            &self.indirect,
            &self._accum,
            &self._selected,
            &self.sites,
            self.padded,
            self.capacity,
            index_limit,
            [&self.image_views[0], &self.image_views[1]],
            [&self.mask_views[0], &self.mask_views[1]],
            self.slot_groups.len() as u32,
        );
        if let Some(tiled) = &self.tiled {
            self.analytic_distances = true;
            self.total_bytes += tiled.allocated_bytes();
        }
        self.tiled.is_some()
    }
    pub fn tiled_state_buffer(&self) -> Option<&wgpu::Buffer> {
        self.tiled.as_ref().map(|tiled| tiled.state_buffer())
    }
    pub fn membrane_sites_buffer(&self) -> &wgpu::Buffer {
        &self.sites
    }
    /// Match Fast Cells pixel distances in explicit raster comparison fixtures.
    /// Cells with Fast Cells disabled retain hardware-interpolated coordinates.
    pub fn use_analytic_distances(&mut self) {
        self.analytic_distances = true;
    }
}
