//! Source-faithful Fluoddity camera ingest/blur/map, with opt-in local capture.
use crate::gpu;
use eframe::egui;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub mapping: u32,
    pub destination: u32,
    pub gain: f32,
    pub blur: f32,
    pub direction: u32,
    pub facing: u32,
    pub mirror: bool,
    pub preview: bool,
    pub device_id: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            mapping: 1,
            destination: 1,
            gain: 1.0,
            blur: 0.3,
            direction: 0,
            facing: 0,
            mirror: true,
            preview: true,
            device_id: String::new(),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct InputDevice {
    id: String,
    label: String,
}
impl Settings {
    pub fn sanitize(&mut self) {
        if self.mapping > 2 {
            self.mapping = 1;
        }
        if self.destination > 1 {
            self.destination = 1;
        }
        if self.direction > 1 {
            self.direction = 0;
        }
        if self.facing > 1 {
            self.facing = 0;
        }
        self.gain = if self.gain.is_finite() {
            self.gain.clamp(0.0, 8.0)
        } else {
            1.0
        };
        self.blur = if self.blur.is_finite() {
            self.blur.clamp(0.0, 1.0)
        } else {
            0.3
        };
    }
}

fn pipeline(
    d: &wgpu::Device,
    name: &str,
    fragment: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let m = gpu::module(d, name);
    d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: None,
        vertex: wgpu::VertexState {
            module: &m,
            entry_point: Some("fullscreen_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &m,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
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
fn uniform(d: &wgpu::Device) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera uniform"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn bind<'a>(binding: u32, resource: wgpu::BindingResource<'a>) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry { binding, resource }
}
fn cover(size: [u32; 2], aspect: f32) -> [f32; 2] {
    let c = size[0] as f32 / size[1].max(1) as f32;
    if c > aspect {
        [aspect / c, 1.0]
    } else {
        [1.0, c / aspect]
    }
}
pub struct Webcam {
    pub settings: Settings,
    pub device_path: String,
    pub error: String,
    pub running: bool,
    pub starting: bool,
    devices: Vec<InputDevice>,
    devices_error: String,
    discovered: bool,
    #[cfg(target_arch = "wasm32")]
    devices_revision: u32,
    #[cfg(target_arch = "wasm32")]
    devices_refresh_pending: bool,
    field: wgpu::Texture,
    field_view: wgpu::TextureView,
    preview: wgpu::Texture,
    preview_view: wgpu::TextureView,
    scalars: [wgpu::Texture; 2],
    scalar_views: [wgpu::TextureView; 2],
    temp: wgpu::Texture,
    temp_view: wgpu::TextureView,
    blurred: wgpu::Texture,
    blurred_view: wgpu::TextureView,
    image: wgpu::Texture,
    image_size: [u32; 2],
    sampler: wgpu::Sampler,
    uniforms: [wgpu::Buffer; 4],
    ingest: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    map: wgpu::RenderPipeline,
    preview_pipeline: wgpu::RenderPipeline,
    current: usize,
    primed: bool,
    clear_pending: bool,
    last_settings: Settings,
    last_aspect: f32,
    #[cfg(not(target_arch = "wasm32"))]
    source: Option<native::Source>,
    #[cfg(target_arch = "wasm32")]
    last_frame: f64,
}
impl Webcam {
    pub fn new(d: &wgpu::Device) -> Self {
        let field = gpu::texture(d, "camera vector field", [256, 256], gpu::VECTOR_FORMAT);
        let field_view = field.create_view(&Default::default());
        let preview = gpu::texture(
            d,
            "camera preview",
            [256, 256],
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let preview_view = preview.create_view(&Default::default());
        let scalars = std::array::from_fn(|_| {
            gpu::texture(d, "camera luminance/motion", [256, 256], gpu::VECTOR_FORMAT)
        });
        let scalar_views = std::array::from_fn(|i| scalars[i].create_view(&Default::default()));
        let temp = gpu::texture(d, "camera blurx", [256, 256], gpu::VECTOR_FORMAT);
        let temp_view = temp.create_view(&Default::default());
        let blurred = gpu::texture(d, "camera blury", [256, 256], gpu::VECTOR_FORMAT);
        let blurred_view = blurred.create_view(&Default::default());
        let settings = Settings::default();
        Self {
            settings: settings.clone(),
            last_settings: settings,
            device_path: String::new(),
            error: String::new(),
            running: false,
            starting: false,
            devices: Vec::new(),
            devices_error: String::new(),
            discovered: false,
            #[cfg(target_arch = "wasm32")]
            devices_revision: 0,
            #[cfg(target_arch = "wasm32")]
            devices_refresh_pending: false,
            field,
            field_view,
            preview,
            preview_view,
            scalars,
            scalar_views,
            temp,
            temp_view,
            blurred,
            blurred_view,
            image: gpu::texture(d, "camera image", [1, 1], wgpu::TextureFormat::Rgba8Unorm),
            image_size: [1, 1],
            sampler: d.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            uniforms: std::array::from_fn(|_| uniform(d)),
            ingest: pipeline(d, "cameraIngest.wgsl", "ingest_fs", gpu::VECTOR_FORMAT),
            blur_pipeline: pipeline(d, "cameraBlur.wgsl", "blur_fs", gpu::VECTOR_FORMAT),
            map: pipeline(d, "cameraMap.wgsl", "map_fs", gpu::VECTOR_FORMAT),
            preview_pipeline: pipeline(
                d,
                "cameraPreview.wgsl",
                "preview_fs",
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            current: 0,
            primed: false,
            clear_pending: true,
            last_aspect: 1.0,
            #[cfg(not(target_arch = "wasm32"))]
            source: None,
            #[cfg(target_arch = "wasm32")]
            last_frame: -1.0,
        }
    }
    pub fn field_view(&self) -> &wgpu::TextureView {
        &self.field_view
    }
    pub fn preview_view(&self) -> &wgpu::TextureView {
        &self.preview_view
    }
    /// Poll only while capture or asynchronous discovery can change camera state.
    /// Merely leaving the camera settings window open does not require a timer.
    pub fn needs_poll(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        let discovery_pending = self.devices_refresh_pending;
        #[cfg(not(target_arch = "wasm32"))]
        let discovery_pending = false;
        self.running || self.starting || discovery_pending
    }
    pub fn field_texture(&self) -> &wgpu::Texture {
        &self.field
    }
    pub fn allocated_bytes(&self) -> u64 {
        [
            &self.field,
            &self.preview,
            &self.scalars[0],
            &self.scalars[1],
            &self.temp,
            &self.blurred,
            &self.image,
        ]
        .iter()
        .map(|t| t.width() as u64 * t.height() as u64 * 4)
        .sum::<u64>()
            + 64
    }
    pub fn strengths(&self) -> [f32; 2] {
        if !self.running {
            return [0.0; 2];
        }
        let gain = self.settings.gain.clamp(0.0, 8.0);
        if self.settings.destination == 0 {
            [gain * 0.01, 0.0]
        } else {
            [0.0, gain * 0.05]
        }
    }
    pub fn start(&mut self) {
        self.stop();
        self.error.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.refresh_devices();
            let path = if self.device_path.is_empty() {
                self.devices.first().map(|d| d.id.as_str()).unwrap_or("")
            } else {
                &self.device_path
            };
            match native::Source::start(path) {
                Ok(source) => {
                    self.source = Some(source);
                    self.running = true
                }
                Err(e) => self.error = e.to_string(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.starting = true;
            browser::camera_start(
                if self.settings.facing == 0 {
                    "user"
                } else {
                    "environment"
                },
                &self.settings.device_id,
            );
        }
    }
    fn refresh_devices(&mut self) {
        self.discovered = true;
        #[cfg(target_arch = "wasm32")]
        {
            // Snapshot before starting: even an immediately rejected request settles
            // by advancing the revision, so its pending poll can then stop.
            self.devices_revision = browser::camera_devices_revision();
            self.devices_refresh_pending = true;
            browser::camera_refresh_devices();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            (self.devices, self.devices_error) = native::discover();
        }
    }
    pub fn stop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.source = None;
        }
        #[cfg(target_arch = "wasm32")]
        {
            browser::camera_stop();
            self.last_frame = -1.0;
        }
        self.running = false;
        self.starting = false;
        self.clear_pending = true;
        self.primed = false;
    }
    fn ensure_image(&mut self, d: &wgpu::Device, size: [u32; 2]) {
        if size != self.image_size {
            self.image = gpu::texture(d, "camera image", size, wgpu::TextureFormat::Rgba8Unorm);
            self.image_size = size;
        }
    }
    /// GPU-only ingest/processing of the newest frame. Call once per application frame.
    pub fn encode(
        &mut self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        c: &mut wgpu::CommandEncoder,
        aspect: f32,
    ) {
        self.settings.sanitize();
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        if self.clear_pending {
            for view in [
                &self.field_view,
                &self.scalar_views[0],
                &self.scalar_views[1],
                &self.preview_view,
            ] {
                clear(c, view);
            }
            self.clear_pending = false;
            self.primed = false;
        }
        let mut new_frame = false;
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(source) = &self.source {
                if let Some(error) = source.error() {
                    self.error = error;
                    self.stop();
                } else if let Some(frame) = source.take() {
                    self.ensure_image(d, [640, 480]);
                    q.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.image,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &frame,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(640 * 4),
                            rows_per_image: Some(480),
                        },
                        wgpu::Extent3d {
                            width: 640,
                            height: 480,
                            depth_or_array_layers: 1,
                        },
                    );
                    new_frame = true;
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            let revision = browser::camera_devices_revision();
            if revision != self.devices_revision {
                self.devices_revision = revision;
                self.devices_refresh_pending = false;
                self.devices = serde_json::from_str(&browser::camera_devices()).unwrap_or_default();
                self.devices_error = browser::camera_devices_error();
            }
            let was_running = self.running;
            self.running = browser::camera_running();
            self.starting = browser::camera_starting();
            if was_running && !self.running {
                for view in [
                    &self.field_view,
                    &self.scalar_views[0],
                    &self.scalar_views[1],
                    &self.preview_view,
                ] {
                    clear(c, view);
                }
                self.primed = false;
            }
            self.error = browser::camera_error();
            let id = browser::camera_frame();
            if self.running && id != self.last_frame {
                let video = browser::camera_video();
                if !video.is_null() {
                    if let Ok(video) = video.dyn_into::<web_sys::HtmlVideoElement>() {
                        let size = [browser::camera_width(), browser::camera_height()];
                        if size
                            .iter()
                            .all(|v| *v > 0 && *v <= d.limits().max_texture_dimension_2d)
                        {
                            self.ensure_image(d, size);
                            if let (Some(queue), Some(texture)) =
                                (q.as_webgpu(), self.image.as_webgpu())
                            {
                                if browser::camera_copy(
                                    queue.as_ref(),
                                    texture.as_ref(),
                                    video.as_ref(),
                                    size[0],
                                    size[1],
                                ) {
                                    new_frame = true;
                                    self.last_frame = id;
                                }
                            } else {
                                self.error =
                                    "Camera video upload requires the browser WebGPU backend."
                                        .to_owned();
                            }
                        }
                    }
                }
            }
        }
        if new_frame {
            let scale = cover(self.image_size, aspect);
            q.write_buffer(
                &self.uniforms[0],
                0,
                bytemuck::cast_slice(&[
                    scale[0],
                    scale[1],
                    u8::from(self.settings.mirror) as f32,
                    u8::from(self.primed) as f32,
                ]),
            );
            let image = self.image.create_view(&Default::default());
            let next = 1 - self.current;
            let entries = [
                bind(0, self.uniforms[0].as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(&image)),
                bind(2, wgpu::BindingResource::Sampler(&self.sampler)),
                bind(
                    3,
                    wgpu::BindingResource::TextureView(&self.scalar_views[self.current]),
                ),
            ];
            pass(d, c, &self.ingest, &self.scalar_views[next], &entries);
            self.current = next;
            self.primed = true;
        }
        if self.primed
            && (new_frame || self.settings != self.last_settings || aspect != self.last_aspect)
        {
            self.remap(d, q, c, aspect);
        }
        if self.settings.preview && self.primed {
            pass(
                d,
                c,
                &self.preview_pipeline,
                &self.preview_view,
                &[
                    bind(0, wgpu::BindingResource::TextureView(&self.field_view)),
                    bind(1, wgpu::BindingResource::Sampler(&self.sampler)),
                ],
            );
        }
        self.last_settings = self.settings.clone();
        self.last_aspect = aspect;
    }
    /// Synthetic input for deterministic GPU checks; never opens a camera device.
    pub fn ingest_rgba(
        &mut self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        c: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        bytes: &[u8],
        aspect: f32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            size.iter()
                .all(|v| *v > 0 && *v <= d.limits().max_texture_dimension_2d)
                && bytes.len() == size[0] as usize * size[1] as usize * 4,
            "Invalid camera frame dimensions/data"
        );
        anyhow::ensure!(aspect.is_finite() && aspect > 0.0, "Invalid world aspect");
        self.settings.sanitize();
        self.clear_pending = false;
        self.ensure_image(d, size);
        q.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.image,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size[0] * 4),
                rows_per_image: Some(size[1]),
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        let image = self.image.create_view(&Default::default());
        let scale = cover(size, aspect);
        q.write_buffer(
            &self.uniforms[0],
            0,
            bytemuck::cast_slice(&[
                scale[0],
                scale[1],
                u8::from(self.settings.mirror) as f32,
                u8::from(self.primed) as f32,
            ]),
        );
        let next = 1 - self.current;
        pass(
            d,
            c,
            &self.ingest,
            &self.scalar_views[next],
            &[
                bind(0, self.uniforms[0].as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(&image)),
                bind(2, wgpu::BindingResource::Sampler(&self.sampler)),
                bind(
                    3,
                    wgpu::BindingResource::TextureView(&self.scalar_views[self.current]),
                ),
            ],
        );
        self.current = next;
        self.primed = true;
        self.remap(d, q, c, aspect);
        Ok(())
    }
    fn remap(&self, d: &wgpu::Device, q: &wgpu::Queue, c: &mut wgpu::CommandEncoder, aspect: f32) {
        let sigma = self.settings.blur.clamp(0.0, 1.0).powi(2) * 16.0;
        let v = sigma / 256.0 / 4.0;
        let on = u8::from(sigma >= 0.5) as f32;
        q.write_buffer(
            &self.uniforms[1],
            0,
            bytemuck::cast_slice(&[v / aspect, 0.0, on, 0.0]),
        );
        q.write_buffer(
            &self.uniforms[2],
            0,
            bytemuck::cast_slice(&[0.0, v, on, 0.0]),
        );
        q.write_buffer(
            &self.uniforms[3],
            0,
            bytemuck::cast_slice(&[
                self.settings.mapping.min(2) as f32,
                if self.settings.direction == 1 {
                    -1.0
                } else {
                    1.0
                },
                aspect,
                (aspect.max(1.0) / 256.0).max(sigma / 256.0),
            ]),
        );
        for (u, source, target) in [
            (1, &self.scalar_views[self.current], &self.temp_view),
            (2, &self.temp_view, &self.blurred_view),
        ] {
            pass(
                d,
                c,
                &self.blur_pipeline,
                target,
                &[
                    bind(0, self.uniforms[u].as_entire_binding()),
                    bind(1, wgpu::BindingResource::TextureView(source)),
                    bind(2, wgpu::BindingResource::Sampler(&self.sampler)),
                ],
            );
        }
        pass(
            d,
            c,
            &self.map,
            &self.field_view,
            &[
                bind(0, self.uniforms[3].as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(&self.blurred_view)),
                bind(2, wgpu::BindingResource::Sampler(&self.sampler)),
            ],
        );
    }
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        if !self.discovered {
            self.refresh_devices();
        }
        ui.label("Use a webcam image to influence the particles. Edges or motion become wall forces or trail signals.");
        ui.small("To save an animation of the world, use Share → Record World instead. No webcam is needed.");
        ui.separator();
        #[cfg(target_arch = "wasm32")]
        let selection = &mut self.settings.device_id;
        #[cfg(not(target_arch = "wasm32"))]
        let selection = &mut self.device_path;
        let old_device = selection.clone();
        let name = if selection.is_empty() {
            "Default webcam".to_owned()
        } else {
            self.devices
                .iter()
                .find(|d| &d.id == selection)
                .map_or_else(
                    || "Selected webcam unavailable".to_owned(),
                    |d| d.label.clone(),
                )
        };
        egui::ComboBox::from_id_salt("webcam input device")
            .selected_text(name)
            .show_ui(ui, |ui| {
                ui.selectable_value(selection, String::new(), "Default webcam");
                for device in &self.devices {
                    ui.selectable_value(selection, device.id.clone(), &device.label);
                }
            });
        let device_changed = old_device != *selection;
        if ui.button("Refresh webcams").clicked() {
            self.refresh_devices();
        }
        if !self.devices_error.is_empty() {
            ui.add(egui::Label::new(&self.devices_error).wrap());
        }
        if self.devices.is_empty() {
            ui.small("No inputs listed. Connect a webcam or start a virtual camera feed. Browsers may hide the device list until you allow camera access; Default can still be tried.");
        }
        if device_changed && (self.running || self.starting) {
            self.stop();
        }
        let before = self.settings.clone();
        choice(
            ui,
            "Mapping",
            &mut self.settings.mapping,
            &["Edges (across)", "Edges (along)", "Motion"],
        );
        choice(
            ui,
            "Destination",
            &mut self.settings.destination,
            &["Walls", "Trails"],
        );
        ui.add(egui::Slider::new(&mut self.settings.gain, 0.0..=8.0).text("Gain"));
        ui.add(egui::Slider::new(&mut self.settings.blur, 0.0..=1.0).text("Blur"));
        choice(
            ui,
            "Direction",
            &mut self.settings.direction,
            &["Toward", "Away"],
        );
        let facing = self.settings.facing;
        #[cfg(target_arch = "wasm32")]
        choice(
            ui,
            "Preferred lens",
            &mut self.settings.facing,
            &["Front", "Back"],
        );
        if facing != self.settings.facing {
            self.settings.mirror = self.settings.facing == 0;
            if self.running || self.starting {
                self.stop();
            }
        }
        ui.checkbox(&mut self.settings.mirror, "Mirror");
        ui.checkbox(&mut self.settings.preview, "Preview");
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.collapsing("Manual V4L2 device", |ui| {
                ui.text_edit_singleline(&mut self.device_path);
            });
        }
        let mut action = false;
        if ui
            .button(if self.running || self.starting {
                "Stop webcam"
            } else {
                "Start webcam"
            })
            .clicked()
        {
            if self.running || self.starting {
                self.stop()
            } else {
                self.start()
            }
            action = true;
        }
        if !self.error.is_empty() {
            ui.add(
                egui::Label::new(egui::RichText::new(&self.error).color(egui::Color32::LIGHT_RED))
                    .wrap(),
            );
        }
        action || before != self.settings
    }
}
impl Drop for Webcam {
    fn drop(&mut self) {
        self.stop();
    }
}
fn choice(ui: &mut egui::Ui, label: &str, value: &mut u32, items: &[&str]) {
    egui::ComboBox::from_label(label)
        .selected_text(*items.get(*value as usize).unwrap_or(&items[0]))
        .show_ui(ui, |ui| {
            for (i, item) in items.iter().enumerate() {
                ui.selectable_value(value, i as u32, *item);
            }
        });
}
fn pass(
    d: &wgpu::Device,
    c: &mut wgpu::CommandEncoder,
    p: &wgpu::RenderPipeline,
    target: &wgpu::TextureView,
    entries: &[wgpu::BindGroupEntry<'_>],
) {
    let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera pass"),
        layout: &p.get_bind_group_layout(0),
        entries,
    });
    gpu::draw(
        c,
        target,
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        p,
        &[(&group, &[])],
    );
}
fn clear(c: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
    let attachments = [Some(wgpu::RenderPassColorAttachment {
        view: target,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    })];
    let _p = c.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("camera clear"),
        color_attachments: &attachments,
        ..Default::default()
    });
}
#[cfg(target_arch = "wasm32")]
mod browser {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(module = "/web/camera.js")]
    unsafe extern "C" {
        #[wasm_bindgen(js_name=cameraStart)]
        pub fn camera_start(facing: &str, device_id: &str);
        #[wasm_bindgen(js_name=cameraRefreshDevices)]
        pub fn camera_refresh_devices();
        #[wasm_bindgen(js_name=cameraDevicesRevision)]
        pub fn camera_devices_revision() -> u32;
        #[wasm_bindgen(js_name=cameraDevices)]
        pub fn camera_devices() -> String;
        #[wasm_bindgen(js_name=cameraDevicesError)]
        pub fn camera_devices_error() -> String;
        #[wasm_bindgen(js_name=cameraStop)]
        pub fn camera_stop();
        #[wasm_bindgen(js_name=cameraRunning)]
        pub fn camera_running() -> bool;
        #[wasm_bindgen(js_name=cameraStarting)]
        pub fn camera_starting() -> bool;
        #[wasm_bindgen(js_name=cameraError)]
        pub fn camera_error() -> String;
        #[wasm_bindgen(js_name=cameraVideo)]
        pub fn camera_video() -> JsValue;
        #[wasm_bindgen(js_name=cameraWidth)]
        pub fn camera_width() -> u32;
        #[wasm_bindgen(js_name=cameraHeight)]
        pub fn camera_height() -> u32;
        #[wasm_bindgen(js_name=cameraCopy)]
        pub fn camera_copy(
            queue: &JsValue,
            texture: &JsValue,
            video: &JsValue,
            width: u32,
            height: u32,
        ) -> bool;
        #[wasm_bindgen(js_name=cameraFrame)]
        pub fn camera_frame() -> f64;
    }
}
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::InputDevice;
    use std::{
        io::Read,
        process::{Child, Command, Stdio},
        sync::{Arc, Mutex},
    };
    pub(super) fn discover() -> (Vec<InputDevice>, String) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            #[repr(C)]
            #[derive(Default)]
            struct Capability {
                driver: [u8; 16],
                card: [u8; 32],
                bus: [u8; 32],
                version: u32,
                caps: u32,
                device_caps: u32,
                reserved: [u32; 3],
            }
            unsafe extern "C" {
                fn ioctl(fd: std::ffi::c_int, request: std::ffi::c_ulong, ...) -> std::ffi::c_int;
            }
            let mut devices = Vec::new();
            let mut denied = false;
            if let Ok(entries) = std::fs::read_dir("/sys/class/video4linux") {
                for entry in entries.flatten() {
                    let path = format!("/dev/{}", entry.file_name().to_string_lossy());
                    let Ok(file) = std::fs::File::open(&path) else {
                        denied = true;
                        continue;
                    };
                    let mut capability = Capability::default();
                    // Linux VIDIOC_QUERYCAP: read-only capability query; no capture starts.
                    if unsafe { ioctl(file.as_raw_fd(), 0x80685600, &mut capability) } != 0 {
                        continue;
                    }
                    let flags = if capability.caps & 0x80000000 != 0 {
                        capability.device_caps
                    } else {
                        capability.caps
                    };
                    if flags & (1 | 0x1000) == 0 {
                        continue;
                    }
                    let label = std::fs::read_to_string(entry.path().join("name"))
                        .unwrap_or_else(|_| path.clone())
                        .trim()
                        .to_owned();
                    devices.push(InputDevice {
                        id: path.clone(),
                        label: format!("{label} ({path})"),
                    });
                }
            }
            devices.sort_by(|a, b| a.id.cmp(&b.id));
            let error = if denied {
                "Some camera devices cannot be opened. Check your Linux video-device permissions."
                    .to_owned()
            } else {
                String::new()
            };
            (devices, error)
        }
        #[cfg(not(target_os = "linux"))]
        {
            (
                Vec::new(),
                "Native webcam input currently requires Linux V4L2.".to_owned(),
            )
        }
    }
    fn lower_priority() {
        #[cfg(target_os = "linux")]
        {
            unsafe extern "C" {
                fn nice(increment: std::ffi::c_int) -> std::ffi::c_int;
            }
            // Linux nice is per-thread; lower just this bounded frame reader, preserving UI priority.
            unsafe {
                nice(19);
            }
        }
    }
    pub struct Source {
        child: Option<Child>,
        last_frame: Arc<Mutex<std::time::Instant>>,
        latest: Arc<Mutex<Option<Vec<u8>>>>,
        error: Arc<Mutex<Option<String>>>,
        diagnostic: Arc<Mutex<String>>,
    }
    impl Source {
        pub fn start(device: &str) -> anyhow::Result<Self> {
            anyhow::ensure!(
                !device.is_empty(),
                "No usable webcam is available. Connect a webcam or start a virtual camera feed, then refresh the device list. A webcam is not needed to record the world."
            );
            anyhow::ensure!(
                device.starts_with("/dev/video") && std::path::Path::new(device).exists(),
                "Camera device does not exist: {device}"
            );
            let mut child = Command::new("nice")
                .args([
                    "-n",
                    "19",
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-threads",
                    "1",
                    "-filter_threads",
                    "1",
                    "-f",
                    "v4l2",
                    "-framerate",
                    "15",
                    "-video_size",
                    "640x480",
                    "-i",
                    device,
                    "-vf",
                    "format=rgba",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null())
                .spawn()
                .map_err(|e| anyhow::anyhow!("Camera requires ffmpeg and nice on PATH: {e}"))?;
            let mut stdout = child.stdout.take().unwrap();
            let mut stderr = child.stderr.take().unwrap();
            let latest = Arc::new(Mutex::new(None));
            let frames = latest.clone();
            let error = Arc::new(Mutex::new(None));
            let errors = error.clone();
            let last_frame = Arc::new(Mutex::new(std::time::Instant::now()));
            let last_seen = last_frame.clone();
            let diagnostic = Arc::new(Mutex::new(String::new()));
            let log = diagnostic.clone();
            let frame_diagnostic = diagnostic.clone();
            std::thread::spawn(move || {
                lower_priority();
                let mut chunk = [0u8; 1024];
                while let Ok(n) = stderr.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    let mut text = log.lock().unwrap();
                    text.push_str(&String::from_utf8_lossy(&chunk[..n]));
                    if text.len() > 8192 {
                        let mut start = text.len() - 8192;
                        while !text.is_char_boundary(start) {
                            start += 1;
                        }
                        text.drain(..start);
                    }
                }
            });
            std::thread::spawn(move || {
                lower_priority();
                loop {
                    let mut frame = vec![0u8; 640 * 480 * 4];
                    if let Err(e) = stdout.read_exact(&mut frame) {
                        let detail = frame_diagnostic.lock().unwrap().clone();
                        *errors.lock().unwrap() = Some(if detail.trim().is_empty() {
                            format!("Camera frame capture ended: {e}")
                        } else {
                            format!("Camera capture failed: {}", detail.trim())
                        });
                        break;
                    }
                    *last_seen.lock().unwrap() = std::time::Instant::now();
                    *frames.lock().unwrap() = Some(frame);
                }
            });
            Ok(Self {
                child: Some(child),
                last_frame,
                latest,
                error,
                diagnostic,
            })
        }
        pub fn take(&self) -> Option<Vec<u8>> {
            self.latest.lock().unwrap().take()
        }
        pub fn error(&self) -> Option<String> {
            self.error.lock().unwrap().clone().map(|mut error| {
                let detail=self.diagnostic.lock().unwrap().clone();
                if !detail.trim().is_empty() && !error.contains(detail.trim()) {
                    error.push_str(": ");error.push_str(detail.trim());
                }
                error
            }).or_else(|| {
                if self.last_frame.lock().unwrap().elapsed()>std::time::Duration::from_secs(5) {
                    Some("Camera produced no frames for 5 seconds. Check device permissions and support for 640x480 at 15 fps.".to_owned())
                } else {None}
            })
        }
    }
    impl Drop for Source {
        fn drop(&mut self) {
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                // Reap away from the UI thread even if a camera driver stalls process termination.
                std::thread::spawn(move || {
                    lower_priority();
                    let _ = child.wait();
                });
            }
        }
    }
}
