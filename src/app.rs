use crate::{
    config::{self, Preferences, Project},
    controls,
    coords::{self, Camera},
    gpu::{Engine, Paint, Shove},
    history::History,
    render::{self, Overlays, Renderer},
    share,
    studio_ui::{self, Icon, InspectorTab},
};
use anyhow::{Result, ensure};
use eframe::{
    egui::{self, Color32, Key, Pos2, Rect, Sense},
    egui_wgpu,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use web_time::Instant;

#[derive(Default, Clone)]
pub struct StartOptions {
    pub preset: Option<String>,
    pub world_size: Option<f64>,
    pub physics_steps: Option<u32>,
    pub ticks_per_second: Option<u32>,
    pub presentation_fps: Option<u32>,
    pub paused: bool,
    pub debug: bool,
    pub trail_mode: bool,
    pub no_panels: bool,
    pub shared: Option<String>,
    pub name: Option<String>,
    pub proposed: ProposedSettings,
    pub splash: Option<String>,
    pub persisted: Option<String>,
    pub storage_error: Option<String>,
    pub no_calibrate: bool,
}
#[derive(Default, Clone)]
pub struct ProposedSettings {
    pub world_size: Option<f64>,
    pub canvas_aspect: Option<f64>,
    pub brightness: Option<f32>,
    pub tonemap_softness: Option<f32>,
    pub trail_mode: Option<bool>,
}
impl ProposedSettings {
    fn is_empty(&self) -> bool {
        self.world_size.is_none()
            && self.canvas_aspect.is_none()
            && self.brightness.is_none()
            && self.tonemap_softness.is_none()
            && self.trail_mode.is_none()
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct SavedState {
    project: Project,
    prefs: Preferences,
    saves: BTreeMap<String, Project>,
    checkpoints: Vec<Project>,
    camera_settings: crate::webcam::Settings,
    recording_settings: crate::record::RecordingSettings,
    simulation: crate::simulation::SimulationSettings,
    auto_calibrate: bool,
}
impl SavedState {
    fn for_persistence(&self, preview: Option<&Project>) -> Self {
        let mut state = self.clone();
        if let Some(original) = preview {
            state.project = original.clone();
        }
        state
    }
}
impl Default for SavedState {
    fn default() -> Self {
        Self {
            project: Project::from_json(include_str!("../assets/default-project.json"))
                .expect("shipped startup project must be valid"),
            prefs: Preferences {
                cells: crate::cells::Settings {
                    style: crate::cells::Style::Cells,
                    ..Default::default()
                },
                world_size: 1.0,
                ..Default::default()
            },
            saves: BTreeMap::new(),
            checkpoints: Vec::new(),
            camera_settings: Default::default(),
            recording_settings: Default::default(),
            simulation: Default::default(),
            auto_calibrate: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Shove,
    Walls,
    Trails,
}
impl Tool {
    fn label(self) -> &'static str {
        match self {
            Self::Select => "Select (1)",
            Self::Shove => "Shove (2)",
            Self::Walls => "Walls (3)",
            Self::Trails => "Trails (4)",
        }
    }
}

struct ComparisonReadback {
    started: Instant,
    buffer: wgpu::Buffer,
    record: serde_json::Value,
    status: Arc<Mutex<Option<Result<(), String>>>>,
    mode: crate::network::Mode,
}
struct PendingProject {
    prepared: crate::gpu::PreparedProject,
    next: Project,
    history: History,
    label: String,
    reset: bool,
    result: Arc<Mutex<Option<Option<String>>>>,
    purpose: ProjectPurpose,
}
enum ProjectPurpose {
    Apply,
    Preview { original: Project },
    RestorePreview,
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct AppearanceKey {
    size: [u32; 2],
    style: crate::cells::Style,
    detailed: bool,
    fast: bool,
}
struct PendingExport {
    session: crate::record::ExportSession,
    result: Arc<Mutex<Option<Option<String>>>>,
}
struct PendingRenderer {
    key: AppearanceKey,
    render: Renderer,
    settings: crate::cells::Settings,
    previous: crate::cells::Settings,
    result: Arc<Mutex<Option<Option<String>>>>,
}
enum InspectorAction {
    Mode(crate::network::Mode),
    Layout(u32),
    Mutation(f64, bool),
    Adopt,
    Undo,
}

#[derive(Clone, Copy)]
enum HistoryAction {
    Undo,
    Redo,
    PreviousGeneration,
}

pub struct SoupApp {
    context: egui::Context,
    pending_project: Option<PendingProject>,
    pending_renderer: Option<PendingRenderer>,
    failed_appearance: Option<AppearanceKey>,
    pending_export: Option<PendingExport>,
    #[cfg(target_arch = "wasm32")]
    _visibility_wake: crate::web::VisibilityWake,
    state: SavedState,
    presets: Vec<config::Preset>,
    history: History,
    gpu: Engine,
    render: Renderer,
    rs: egui_wgpu::RenderState,
    texture: egui::TextureId,
    camera: Camera,
    tool: Tool,
    show_panels: bool,
    inspector_tab: InspectorTab,
    panels_hidden: bool,
    mutation_draft: Option<f64>,
    paused: bool,
    clock: crate::simulation::SimulationClock,
    last_tick_plan: Instant,
    presentation_clock: PresentationClock,
    presentations: u64,
    visible: bool,
    show_timing: bool,
    show_network_settings: bool,
    advance_steps: u64,
    comparison: Option<(u64, Instant, serde_json::Value)>,
    comparison_result: Option<serde_json::Value>,
    comparison_readback: Option<ComparisonReadback>,
    comparison_stale: bool,
    comparison_clear: bool,
    profiling_enabled: bool,
    profiler: Option<crate::profiling::GpuProfiler>,
    last_gpu_sample: Option<crate::profiling::GpuSample>,
    /// Always-on frame total; stage profiling supplies the total instead while enabled.
    frame_timer: Option<crate::profiling::FrameTimer>,
    gpu_window: crate::performance::GpuWindow,
    tick_rate: crate::performance::RateWindow,
    /// Logical ticks executed and requested in the most recent scene submission.
    last_scene_steps: [u32; 2],
    /// When the frame-rate meter last began receiving live frames; None while it is frozen.
    meter_running_since: Option<f64>,
    frame_interval_ms: f64,
    cpu_encode_ms: f64,
    cpu_submit_ms: f64,
    requested_delay: Option<Duration>,
    submissions: u64,
    held_view: Option<Camera>,
    /// Overlay, highlight and tone inputs of the image last presented.
    presented: Option<PresentationKey>,
    /// Last `#status` text; `None` unless the page was opened with `?debug`.
    #[cfg(target_arch = "wasm32")]
    status_text: Option<String>,
    highlighted: Option<crate::gpu::Pick>,
    pick_intent: Option<PickIntent>,
    queued_pick: Option<QueuedPick>,
    origin: Instant,
    previous_frame: Instant,
    fps: crate::performance::Meter,
    frames: u64,
    debug: bool,
    error: Arc<Mutex<Option<String>>>,
    notice: String,
    help: bool,
    transfer: bool,
    transfer_text: String,
    save_name: String,
    preview: Option<Project>,
    preview_hovered: bool,
    stroke: crate::interaction::StrokeState,
    clear_layer: Option<u32>,
    capture_requested: bool,
    capture: Option<Capture>,
    workgroup: u32,
    incoming: Arc<Mutex<Vec<std::result::Result<String, String>>>>,
    webcam: crate::webcam::Webcam,
    webcam_texture: egui::TextureId,
    show_camera: bool,
    show_recording: bool,
    recorder: Option<crate::record::ExportSession>,
    show_capture: bool,
    capture_crop: [u32; 2],
    capture_origin: [u32; 2],
    crop_anchor: Option<[u32; 2]>,
    capture_qr: bool,
    capture_clipboard: bool,
    show_link: bool,
    link_name: String,
    link_include: [bool; 5],
    link_splash: u32,
    proposed: ProposedSettings,
    can_save: bool,
    save_now: bool,
    touch_seen: bool,
    canvas_buttons: [bool; 2],
    notices: Arc<Mutex<Vec<String>>>,
    archive_session: String,
    archive_cursor: Option<String>,
    archive_count: u64,
    #[cfg(not(target_arch = "wasm32"))]
    archive_writer: Option<crate::archive::Writer>,
    state_dirty: bool,
    last_save: Instant,
    calibration: Option<crate::calibration::Calibration>,
    rate_calibration: Option<RateCalibration>,
}

struct RateCalibration {
    search: crate::rate_calibration::AutoCalibration,
    restore_pause: bool,
    started: Instant,
}

struct Capture {
    readback: crate::gpu::RgbaReadback,
    status: Arc<Mutex<Option<std::result::Result<(), String>>>>,
    clipboard: bool,
    qr: Option<String>,
}

/// What frame assembly shows besides resampled scene content. A change
/// re-presents the held image; pointer motion alone does not.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PresentationKey {
    overlays: Overlays,
    highlight: f32,
    tone: [f32; 7],
}
impl PresentationKey {
    fn new(overlays: Overlays, highlight: f32, prefs: &Preferences) -> Self {
        Self {
            overlays: overlays.normalized(),
            highlight,
            tone: [
                f32::from(u8::from(prefs.bloom_enabled)),
                prefs.bloom_threshold,
                prefs.bloom_intensity,
                prefs.bloom_radius,
                prefs.brightness,
                prefs.tonemap_softness,
                prefs.field_opacity,
            ],
        }
    }
}

/// Display frames land with jitter around the presentation interval. Requiring
/// the full interval skips alternate vsyncs whenever the display period is
/// marginally shorter, so deadlines within this margin count as reached. It
/// stays below the 1.5 ms between a 60 Hz vsync and a 55 fps interval, so
/// presentation rates below about 57 fps do not round up to the display rate.
const PACING_SLACK: Duration = Duration::from_millis(1);
fn deadline_reached(elapsed: Duration, interval: Duration) -> bool {
    elapsed + PACING_SLACK >= interval
}

/// Preserve the scheduled phase across ordinary callback jitter and encoding work.
struct PresentationClock {
    last: Instant,
    interval: Option<Duration>,
    active: bool,
}
impl PresentationClock {
    fn new(start: Instant) -> Self {
        Self {
            last: start,
            interval: None,
            active: false,
        }
    }
    fn configure(&mut self, frame_start: Instant, interval: Duration, active: bool) {
        if active && (!self.active || self.interval != Some(interval)) {
            // Resuming or changing the cap starts a fresh cadence, without debt.
            self.last = self.last.max(frame_start);
        }
        self.interval = Some(interval);
        self.active = active;
    }
    fn suspend(&mut self) {
        self.active = false;
    }
    fn due(&self, frame_start: Instant, interval: Duration) -> bool {
        deadline_reached(frame_start.saturating_duration_since(self.last), interval)
    }
    fn consume(&mut self, frame_start: Instant, interval: Duration) {
        if !self.due(frame_start, interval) {
            return;
        }
        let deadline = self.last + interval;
        // A long stall earns one update, never a sequence of catch-up frames.
        self.last = if frame_start.saturating_duration_since(deadline) >= interval {
            frame_start
        } else {
            deadline
        };
    }
    fn remaining(&self, now: Instant, interval: Duration) -> Duration {
        (self.last + interval).saturating_duration_since(now)
    }
}

/// Input that can change a control and so earns one settling pass. Plain
/// pointer motion is drawn in the frame it arrives and needs no follow-up.
fn settles_controls(event: &egui::Event) -> bool {
    !matches!(
        event,
        egui::Event::PointerMoved(_) | egui::Event::MouseMoved(_)
    )
}

#[derive(Clone, Copy)]
struct PickIntent {
    highlight: Option<f32>,
    force: bool,
    highlight_only: bool,
    one_click: bool,
}
#[derive(Clone, Copy)]
struct QueuedPick {
    target: [f32; 2],
    radius: f32,
    cohort: f32,
    intent: PickIntent,
}

pub fn wgpu_options() -> egui_wgpu::WgpuConfiguration {
    let mut create = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    create.instance_descriptor.backends = if cfg!(target_arch = "wasm32") {
        wgpu::Backends::BROWSER_WEBGPU
    } else {
        wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12
    };
    create.power_preference = wgpu::PowerPreference::HighPerformance;
    create.device_descriptor = Arc::new(|adapter| {
        let a = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: a.max_storage_buffer_binding_size,
            max_buffer_size: a.max_buffer_size,
            max_texture_dimension_2d: a.max_texture_dimension_2d,
            ..wgpu::Limits::default()
        };
        wgpu::DeviceDescriptor {
            label: Some("Primordial Soup GPU"),
            required_limits: limits,
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }
    });
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(create),
        ..Default::default()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> eframe::Result {
    eframe::run_native(
        "Primordial Soup",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: wgpu_options(),
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 800.0])
                .with_min_inner_size([640.0, 400.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(SoupApp::new(cc, StartOptions::default())?))),
    )
}

impl SoupApp {
    pub fn new(cc: &eframe::CreationContext<'_>, opts: StartOptions) -> Result<Self> {
        let rs = cc
            .wgpu_render_state
            .clone()
            .ok_or_else(|| anyhow::anyhow!("A hardware WebGPU/Vulkan device is required"))?;
        ensure!(
            rs.adapter.get_info().device_type != wgpu::DeviceType::Cpu,
            "No hardware GPU adapter available"
        );
        let error = Arc::new(Mutex::new(None));
        let errors = error.clone();
        let ctx = cc.egui_ctx.clone();
        rs.device
            .on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                *errors.lock().unwrap() = Some(e.to_string());
                ctx.request_repaint();
            }));
        let errors = error.clone();
        let ctx = cc.egui_ctx.clone();
        rs.device.set_device_lost_callback(move |reason, detail| {
            if reason != wgpu::DeviceLostReason::Destroyed {
                *errors.lock().unwrap() = Some(format!(
                    "GPU device lost: {detail}. Reload to create a new device."
                ));
                ctx.request_repaint();
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        let mut state: SavedState = cc
            .storage
            .and_then(|s| eframe::get_value(s, "primordial-soup.v1"))
            .unwrap_or_default();
        #[cfg(target_arch = "wasm32")]
        let mut state: SavedState = opts
            .persisted
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        if state.project.validate().is_err() {
            state.project = SavedState::default().project;
        }
        state.checkpoints.retain(|p| p.validate().is_ok());
        state.saves.retain(|_, p| p.validate().is_ok());
        sanitize_preferences(&mut state.prefs);
        let presets = config::preset_catalog();
        if let Some(preset) = opts.preset {
            if let Some(p) = presets.iter().find(|p| {
                p.name.eq_ignore_ascii_case(&preset)
                    || p.name
                        .rsplit('/')
                        .next()
                        .is_some_and(|stem| stem.eq_ignore_ascii_case(&preset))
            }) {
                state.project = p.project.clone();
            }
        }
        let mut notice = opts.storage_error.unwrap_or_default();
        if let Some(shared) = opts.shared {
            match share::decode_text(&shared) {
                Ok(Some(p)) => state.project = p,
                Err(e) => notice = format!("Could not open share link: {e}"),
                _ => {}
            }
        }
        if let Some(name) = opts.name {
            state.project.name = name.chars().take(64).collect();
        }
        if let Some(v) = opts.world_size {
            state.prefs.world_size = v;
        }
        if let Some(v) = opts.physics_steps {
            state.prefs.physics_steps = v.clamp(1, 120);
        }
        if let Some(v) = opts.ticks_per_second {
            state.simulation.mode = crate::simulation::SimulationMode::FixedRate;
            state.simulation.ticks_per_second = v;
        }
        if let Some(v) = opts.presentation_fps {
            state.simulation.presentation_fps = v;
        }
        state.simulation.sanitize();
        if opts.no_calibrate {
            state.auto_calibrate = false;
        }
        let size = coords::sizing_for(state.prefs.world_size, state.prefs.canvas_aspect)?;
        let workgroup = crate::gpu::default_workgroup(&rs.device.limits());
        let mut gpu = Engine::new(&rs.device, &rs.queue, size, &state.project, workgroup)?;
        let mut webcam = crate::webcam::Webcam::new(&rs.device);
        webcam.settings = state.camera_settings.clone();
        webcam.settings.sanitize();
        gpu.set_camera_field(&rs.device, webcam.field_view());
        let render = Renderer::new(&rs.device, &gpu, [1280, 800])?;
        let texture = rs.renderer.write().register_native_texture(
            &rs.device,
            render.output_view(),
            wgpu::FilterMode::Linear,
        );
        let webcam_texture = rs.renderer.write().register_native_texture(
            &rs.device,
            webcam.preview_view(),
            wgpu::FilterMode::Linear,
        );
        let mut history = History::default();
        history.seed(state.project.clone());
        studio_ui::install(&cc.egui_ctx);
        let now = Instant::now();
        let frame_timer = crate::profiling::FrameTimer::new(&rs.device, &rs.queue);
        let mut app = Self {
            pending_project: None,
            pending_renderer: None,
            failed_appearance: None,
            pending_export: None,
            context: cc.egui_ctx.clone(),
            #[cfg(target_arch = "wasm32")]
            _visibility_wake: crate::web::VisibilityWake::new(cc.egui_ctx.clone())
                .map_err(|e| anyhow::anyhow!("Visibility listener: {e:?}"))?,
            state,
            presets,
            history,
            gpu,
            render,
            rs,
            texture,
            camera: Camera {
                trail_mode: opts.trail_mode,
                ..Default::default()
            },
            tool: Tool::Select,
            show_panels: false,
            inspector_tab: InspectorTab::World,
            panels_hidden: opts.no_panels,
            mutation_draft: None,
            paused: opts.paused,
            clock: Default::default(),
            last_tick_plan: now,
            presentation_clock: PresentationClock::new(now),
            presentations: 0,
            visible: true,
            show_timing: false,
            show_network_settings: false,
            comparison: None,
            comparison_result: None,
            comparison_readback: None,
            comparison_stale: false,
            comparison_clear: false,
            advance_steps: 100,
            profiling_enabled: false,
            profiler: None,
            last_gpu_sample: None,
            frame_timer,
            gpu_window: Default::default(),
            tick_rate: Default::default(),
            last_scene_steps: [0; 2],
            meter_running_since: None,
            frame_interval_ms: 0.0,
            cpu_encode_ms: 0.0,
            cpu_submit_ms: 0.0,
            requested_delay: None,
            submissions: 0,
            held_view: None,
            presented: None,
            #[cfg(target_arch = "wasm32")]
            status_text: opts.debug.then(String::new),
            highlighted: None,
            pick_intent: None,
            queued_pick: None,
            origin: now,
            previous_frame: now,
            fps: Default::default(),
            frames: 0,
            debug: opts.debug,
            error,
            notice,
            help: opts.splash.is_some(),
            transfer: false,
            transfer_text: String::new(),
            save_name: String::new(),
            preview: None,
            preview_hovered: false,
            stroke: Default::default(),
            clear_layer: None,
            capture_requested: false,
            capture: None,
            workgroup,
            incoming: Arc::new(Mutex::new(Vec::new())),
            webcam,
            webcam_texture,
            show_camera: false,
            show_recording: false,
            recorder: None,
            show_capture: false,
            capture_crop: [1280, 800],
            capture_origin: [0, 0],
            crop_anchor: None,
            capture_qr: false,
            capture_clipboard: false,
            show_link: false,
            link_name: String::new(),
            link_include: [false; 5],
            link_splash: 0,
            proposed: opts.proposed,
            can_save: cfg!(target_arch = "wasm32") || cc.storage.is_some(),
            save_now: false,
            touch_seen: false,
            canvas_buttons: [false; 2],
            notices: Arc::new(Mutex::new(Vec::new())),
            archive_session: format!(
                "{:x}-{:x}",
                crate::archive::epoch_ms() as u64,
                (random_seed() * u32::MAX as f64) as u32
            ),
            archive_cursor: None,
            archive_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            archive_writer: None,
            state_dirty: false,
            last_save: now,
            calibration: None,
            rate_calibration: None,
        };
        #[cfg(target_arch = "wasm32")]
        crate::web::set_panels_hidden(app.panels_hidden);
        app.log_state("Session Start", 0.0);
        if app.state.auto_calibrate
            && app.state.simulation.mode == crate::simulation::SimulationMode::PerFrame
            && !opts.paused
            && opts.world_size.is_none()
            && opts.physics_steps.is_none()
            && app.state.prefs.calibration_version < crate::calibration::VERSION
        {
            app.start_calibration();
        }
        Ok(app)
    }
    fn start_calibration(&mut self) {
        if self.comparison_busy() || self.pending_project.is_some() {
            return;
        }
        if self.state.simulation.mode != crate::simulation::SimulationMode::PerFrame
            || self.profiling_enabled
            || self.clock.pending_exact_steps() > 0
        {
            self.notice =
                "Calibration requires frame-based mode with profiling and exact stepping off"
                    .into();
            return;
        }
        if self.recorder.is_some() || self.pending_export.is_some() {
            self.notice = "Finish recording before calibration".into();
            return;
        }
        self.cancel_rate_calibration();
        match crate::calibration::Calibration::new(
            &self.rs.device,
            &self.rs.queue,
            &self.state.project,
            self.workgroup,
        ) {
            Ok(c) => {
                self.calibration = Some(c);
                self.notice = "Calibrating GPU…".into();
            }
            Err(e) => self.notice = format!("Calibration unavailable: {e}"),
        }
    }

    fn cancel_rate_calibration(&mut self) {
        if let Some(mut run) = self.rate_calibration.take() {
            self.state.prefs.physics_steps = run.search.cancel();
            if run.restore_pause {
                self.paused = true;
            }
            self.held_view = None;
            self.clock.clear_wall_debt();
            self.state_dirty = true;
        }
    }
    fn start_rate_calibration(&mut self) {
        if self.comparison_busy() || self.pending_project.is_some() {
            return;
        }
        if self.state.simulation.mode != crate::simulation::SimulationMode::PerFrame
            || self.profiling_enabled
            || self.clock.pending_exact_steps() > 0
        {
            self.notice =
                "Rate calibration requires frame-based mode with profiling and exact stepping off"
                    .into();
            return;
        }
        if self.rate_calibration.is_some() {
            self.cancel_rate_calibration();
            self.notice = "Physics-rate calibration cancelled; original rate restored".into();
            return;
        }
        if (self.recorder.is_some() || self.pending_export.is_some()) || self.calibration.is_some()
        {
            self.notice = "Finish recording or GPU calibration before tuning Physics Rate".into();
            return;
        }
        let search =
            crate::rate_calibration::AutoCalibration::new(self.state.prefs.physics_steps, 1, 60);
        self.state.prefs.physics_steps = search.rate();
        self.rate_calibration = Some(RateCalibration {
            search,
            restore_pause: self.paused,
            started: Instant::now(),
        });
        self.paused = false;
        self.held_view = None;
        self.previous_frame = Instant::now();
        self.notice = "Calibrating Physics Rate against the current rendered project…".into();
    }
    fn feed_rate_calibration(&mut self, frame_ms: f64, visible: bool) {
        let Some(run) = self.rate_calibration.as_mut() else {
            return;
        };
        if visible && !self.paused {
            run.search.on_frame(frame_ms);
        }
        let rate = run.search.rate();
        if self.state.prefs.physics_steps != rate {
            self.state.prefs.physics_steps = rate;
            self.held_view = None;
            self.state_dirty = true;
        }
        if run.search.finished() {
            let restore_pause = run.restore_pause;
            self.rate_calibration = None;
            if restore_pause {
                self.paused = true;
            }
            self.clock.clear_wall_debt();
            self.save_now = true;
            self.notice =
                format!("Physics Rate calibrated: {rate} steps/frame · simulation preserved");
        } else {
            self.notice = format!(
                "Calibrating Physics Rate: probe {} / 8 · {rate} steps/frame",
                run.search.probe_count() + 1
            );
        }
    }

    fn reset(&mut self) {
        if self.pending_project.is_some() {
            return;
        }
        if self.comparison_busy() {
            self.notice = "Cancel comparison before resetting".into();
            return;
        }
        self.comparison_stale = true;
        if self.recorder.is_some() || self.pending_export.is_some() {
            self.notice = "Finish or cancel recording before resetting the world".into();
            return;
        }
        self.archive_command("resetSimulation");
        self.gpu.reset();
        self.clock.reset();
        self.last_tick_plan = Instant::now();
        self.highlighted = None;
        self.held_view = None;
        self.pick_intent = None;
        self.queued_pick = None;
    }
    fn toggle_pause(&mut self) {
        if self.comparison_busy() || self.pending_project.is_some() || self.pending_export.is_some()
        {
            return;
        }
        if self.clock.pending_exact_steps() > 0 {
            self.notice = "Exact advancement is still running".into();
            return;
        }
        self.archive_command("togglePause");
        self.paused = !self.paused;
        if !self.paused {
            self.comparison_stale = true;
        }
        self.clock.clear_wall_debt();
        self.last_tick_plan = Instant::now();
        self.held_view = None;
    }
    fn save_pending(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = ctx;
        #[cfg(target_arch = "wasm32")]
        let _ = frame;
        if self.save_now
            || (self.state_dirty && self.last_save.elapsed() >= Duration::from_secs(10))
        {
            let state = self.persistent_state();
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(storage) = frame.storage_mut() {
                eframe::set_value(storage, "primordial-soup.v1", &state);
                storage.flush();
                if self.save_now {
                    self.notice = "Saved projects and preferences".into();
                }
            }
            #[cfg(target_arch = "wasm32")]
            match serde_json::to_string(&state) {
                Ok(text) => {
                    crate::web::save_state(&text, self.notices.clone(), ctx.clone(), self.save_now)
                }
                Err(e) => self.notice = format!("Project save failed: {e}"),
            }
            self.save_now = false;
            self.state_dirty = false;
            self.last_save = Instant::now();
        }
    }
    fn service_pending(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.finish_project_apply();
        self.finish_renderer();
        self.finish_export();
        self.finish_comparison_readback();
        let now = self.origin.elapsed().as_secs_f64() * 1000.0;
        for notice in std::mem::take(&mut *self.notices.lock().unwrap()) {
            if notice.starts_with("Archive storage failed") {
                self.state.prefs.strong_logging = false;
            }
            self.notice = notice;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = self.rs.device.poll(wgpu::PollType::Poll) {
            *self.error.lock().unwrap() = Some(e.to_string());
        }
        if let Err(e) = self.resolve_pick(now) {
            self.notice = e.to_string();
        }
        if let Some(profiler) = self.profiler.as_mut() {
            for sample in profiler.poll(&self.rs.device, &self.rs.queue) {
                self.gpu_window.push(now, sample.total_ms);
                self.last_gpu_sample = Some(sample);
            }
        }
        if let Some(timer) = self.frame_timer.as_mut() {
            for total in timer.poll() {
                self.gpu_window.push(now, total);
            }
        }
        if let Some(r) = self.recorder.as_mut() {
            match r.pump(&self.rs.device) {
                Ok(progress) => {
                    if progress.finished {
                        self.notice = if let Some(path) = progress.output {
                            if path.is_empty() {
                                "Recording cancelled".into()
                            } else {
                                format!("Recording saved: {path}")
                            }
                        } else {
                            "Recording complete".into()
                        };
                        self.paused = r.was_paused;
                        self.held_view = None;
                        self.recorder = None;
                    }
                }
                Err(e) => {
                    self.notice = format!("Video export: {e}");
                    self.paused = r.was_paused;
                    self.held_view = None;
                    self.recorder = None;
                }
            }
        }
        self.capture_finish(ctx);
        if self.capture.is_none() && !self.capture_requested {
            self.render.release_clean();
        }
        self.save_pending(ctx, frame);
        let pending = self.pending_export.is_some()
            || self.pending_renderer.is_some()
            || self.comparison_readback.is_some()
            || self.pending_project.is_some()
            || self.capture.is_some()
            || self.gpu.pick_readback_pending()
            || self.profiler.as_ref().is_some_and(|p| p.has_pending())
            || self.frame_timer.as_ref().is_some_and(|t| t.has_pending())
            || self.recorder.as_ref().is_some_and(|r| !r.ready());
        let service = pending.then_some(Duration::from_millis(if self.visible { 16 } else { 100 }));
        let save = self
            .state_dirty
            .then(|| Duration::from_secs(10).saturating_sub(self.last_save.elapsed()));
        if let Some(delay) = [service, save].into_iter().flatten().min() {
            ctx.request_repaint_after(delay);
        }
    }
    fn advance_exactly(&mut self, steps: u64) {
        if self.comparison_busy() || self.pending_project.is_some() {
            return;
        }
        if (self.recorder.is_some() || self.pending_export.is_some())
            || self.calibration.is_some()
            || self.rate_calibration.is_some()
        {
            self.notice = "Finish recording or calibration before exact stepping".into();
            return;
        }
        self.paused = true;
        self.clock.clear_wall_debt();
        self.clock.queue_exact_steps(steps);
        self.held_view = None;
        self.notice = format!("Queued {steps} ticks; remains paused when complete");
    }
    fn painting_input(&self, ctx: &egui::Context) -> bool {
        matches!(self.tool, Tool::Walls | Tool::Trails)
            && (!self.show_capture || self.panels_hidden)
            && ctx.input(|i| {
                (self.canvas_buttons[0] && i.pointer.primary_down())
                    || (self.canvas_buttons[1] && i.pointer.secondary_down())
            })
    }
    fn comparison_busy(&self) -> bool {
        self.comparison.is_some() || self.comparison_readback.is_some()
    }
    fn finish_comparison_readback(&mut self) {
        let status = self
            .comparison_readback
            .as_ref()
            .and_then(|r| r.status.lock().unwrap().take());
        let Some(status) = status else {
            return;
        };
        let mut pending = self.comparison_readback.take().unwrap();
        let result = status.and_then(|_| {
            pending
                .buffer
                .slice(..)
                .get_mapped_range()
                .map(|bytes| {
                    let stride = if pending.mode == crate::network::Mode::Fourier {
                        416
                    } else {
                        crate::network::WEIGHT_STRIDE * 4
                    };
                    let length = pending
                        .mode
                        .parameters_at(pending.record["revision"].as_u64().unwrap_or(1) as u32)
                        * 4;
                    let mut hash = 0xcbf29ce484222325u64;
                    for chunk in bytes.chunks_exact(stride) {
                        for byte in &chunk[..length] {
                            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
                        }
                    }
                    format!(
                        "{}-v{}-{hash:016x}",
                        pending.mode.label(),
                        pending.record["revision"].as_u64().unwrap_or(1)
                    )
                })
                .map_err(|e| e.to_string())
        });
        pending.buffer.unmap();
        match result {
            Ok(hash) => pending.record["effective_rule_hash"] = serde_json::json!(hash),
            Err(error) => {
                pending.record["status"] = serde_json::json!("error");
                pending.record["error"] = serde_json::json!(error);
            }
        }
        pending.record["elapsed_wall_ms"] =
            serde_json::json!(pending.started.elapsed().as_secs_f64() * 1000.0);
        self.comparison_result = Some(pending.record);
    }
    fn restart_comparison(&mut self, ticks: u64) {
        if self.pending_project.is_some()
            || (self.recorder.is_some() || self.pending_export.is_some())
            || self.calibration.is_some()
            || self.rate_calibration.is_some()
        {
            self.notice =
                "Finish recording, calibration or project preparation before comparison".into();
            return;
        }
        self.cancel_comparison();
        self.restore_preset_preview();
        if self.preview.is_some() {
            self.notice = "Could not restore preset preview before comparison".into();
            return;
        }
        self.paused = true;
        self.reset();
        self.clear_layer = None;
        self.comparison_clear = true;
        self.stroke = Default::default();
        self.state.auto_calibrate = false;
        self.webcam.settings.gain = 0.0;
        self.state_dirty = true;
        self.comparison_stale = false;
        self.last_gpu_sample = None;
        // Start a fresh profiler generation; old asynchronous samples cannot leak.
        self.profiler = if self.profiling_enabled {
            crate::profiling::GpuProfiler::new(&self.rs.device)
        } else {
            None
        };
        let info = self.rs.adapter.get_info();
        let c = self.state.project.config();
        let record = serde_json::json!({"version":1,"application":"primordial-soup","status":"running","mode":c.mode(),"revision":c.bank().map_or(1, |b| b.revision),"parameters":c.network_parameters(),"memory_bytes_per_particle":if c.mode()==crate::network::Mode::Fourier{0}else if c.network_revision()>=3{if c.mode()==crate::network::Mode::Gru{48}else{16}}else if c.mode()==crate::network::Mode::Gru{32}else{0},"project":serde_json::from_str::<serde_json::Value>(&self.state.project.to_json().unwrap()).unwrap(),"project_hash":crate::archive::hash_state(&self.state.project),"preferences":self.state.prefs,"simulation":self.state.simulation,"canvas":self.gpu.sizing.canvas,"particles":self.gpu.sizing.entity_count,"effective_cohorts":self.gpu.config_count,"target_ticks":ticks,"adapter":info.name,"backend":format!("{:?}",info.backend),"workgroup":self.workgroup,"profiling_enabled":self.profiling_enabled});
        self.comparison_result = None;
        self.clock.queue_exact_steps(ticks);
        self.comparison = Some((ticks, Instant::now(), record));
        self.show_timing = true;
        self.panels_hidden = false;
    }
    fn cancel_comparison(&mut self) {
        if let Some(mut pending) = self.comparison_readback.take() {
            pending.buffer.unmap();
            pending.record["status"] = serde_json::json!("incomplete");
            self.comparison_result = Some(pending.record);
        }
        if let Some((_, _, mut record)) = self.comparison.take() {
            self.clock.cancel_exact_steps();
            self.paused = true;
            record["status"] = serde_json::json!("incomplete");
            record["completed_ticks"] = serde_json::json!(self.clock.completed_ticks());
            self.comparison_result = Some(record);
            self.comparison_stale = false;
        }
    }
    fn finish_comparison(&mut self) {
        if let Some((_, started, mut record)) = self.comparison.take() {
            record["status"] = serde_json::json!("complete");
            record["completed_ticks"] = serde_json::json!(self.clock.completed_ticks());
            record["elapsed_wall_ms"] = serde_json::json!(started.elapsed().as_secs_f64() * 1000.0);
            record["last_batch_cpu_encode_ms"] = serde_json::json!(self.cpu_encode_ms);
            record["last_batch_submit_ms"] = serde_json::json!(self.cpu_submit_ms);
            record["gpu_sample"] = serde_json::json!(self.last_gpu_sample);
            record["gpu_sample_note"] = serde_json::json!(
                "Latest available asynchronous sample from this run; may precede the final batch. Wall time includes scheduling and suspension."
            );
            let buffer = self.gpu.snapshot_rules(&self.rs.device, &self.rs.queue);
            let status = Arc::new(Mutex::new(None));
            let result = status.clone();
            let context = self.context.clone();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                *result.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
                context.request_repaint();
            });
            self.comparison_readback = Some(ComparisonReadback {
                started,
                buffer,
                record,
                status,
                mode: self.state.project.config().mode(),
            });
            self.paused = true;
        }
    }
    /// The frame-rate meter only judges live frames; paused, recorded and exported frames don't count.
    fn meter_live(&self) -> bool {
        !self.paused && self.recorder.is_none() && self.pending_export.is_none()
    }
    fn meter_idle_label(&self) -> &'static str {
        if self.recorder.is_some() {
            "Recording"
        } else if self.pending_export.is_some() {
            "Exporting"
        } else {
            "Paused"
        }
    }
    fn dot_state(&self, now: f64) -> crate::performance::DotState {
        crate::performance::dot_state(self.fps.band, self.meter_running_since, now)
    }
    fn rate_label(&self) -> String {
        match self.state.simulation.mode {
            crate::simulation::SimulationMode::PerFrame => {
                format!("×{}", self.state.prefs.physics_steps)
            }
            crate::simulation::SimulationMode::FixedRate => {
                format!("{} ticks/s", self.state.simulation.ticks_per_second)
            }
        }
    }
    fn badge_text(&self, dot: crate::performance::DotState) -> String {
        let head = match dot {
            crate::performance::DotState::Idle => self.meter_idle_label().to_owned(),
            _ => format!("{} fps", self.fps.readout),
        };
        crate::performance::badge_text(
            &head,
            self.gpu_window.median(),
            u64::from(self.gpu.sizing.entity_count),
            &self.rate_label(),
        )
    }
    /// The latest stage sample, only while stage profiling is on.
    fn stage_sample(&self) -> Option<&crate::profiling::GpuSample> {
        self.last_gpu_sample
            .as_ref()
            .filter(|_| self.profiling_enabled)
    }
    fn performance_tooltip(&self, dot: crate::performance::DotState) -> String {
        use crate::performance::DotState;
        let band = self.fps.band;
        let mut text = match dot {
            DotState::Live(_) => format!("{}\n{}", band.description(), band.help()),
            DotState::Measuring => format!(
                "Measuring… the frame rate needs a moment after resuming.\nLast verdict: {}",
                band.description()
            ),
            DotState::Idle => format!(
                "{}: the frame-rate meter is not updating.\nLast verdict: {}",
                self.meter_idle_label(),
                band.description()
            ),
        };
        if let (Some(median), Some(p95)) = (self.gpu_window.median(), self.gpu_window.p95()) {
            text += &format!(
                "\n\nGPU {median:.2} ms median · {p95:.2} ms p95 over the last second of scene work (budget {:.0} ms)",
                crate::performance::TARGET_GPU_MS
            );
            if let Some(sample) = self.stage_sample() {
                text += "\nLast profiled frame:";
                for (name, ms) in crate::profiling::Stage::NAMES
                    .iter()
                    .zip(sample.stages_ms)
                    .filter(|(_, ms)| *ms > 0.0)
                {
                    text += &format!("\n  {name} {ms:.2} ms");
                }
            } else {
                text += "\nProfile GPU stages in Simulation Timing for a per-stage breakdown.";
            }
        }
        let s = &self.gpu.sizing;
        text += &format!(
            "\n\n{} particles · {} × {} trails",
            crate::performance::grouped(u64::from(s.entity_count)),
            s.canvas[0],
            s.canvas[1]
        );
        text
    }
    fn active_cells(&self) -> Option<&crate::cells::Cells> {
        self.render
            .cells
            .as_ref()
            .filter(|_| self.render.cells_active())
    }
    fn style_label(&self) -> &'static str {
        if self.active_cells().is_some() {
            "Cells"
        } else if self.state.prefs.cells.style == crate::cells::Style::Cells
            && self.camera.trail_mode
        {
            "Classic (trail camera)"
        } else {
            "Classic"
        }
    }
    /// Achieved logical ticks per second; zero while paused.
    fn achieved_tick_rate(&self) -> Option<f64> {
        if self.paused && self.clock.pending_exact_steps() == 0 {
            Some(0.0)
        } else {
            self.tick_rate.rate()
        }
    }
    fn workload_lines(&self) -> Vec<String> {
        use crate::performance::{compact_count, grouped};
        let s = &self.gpu.sizing;
        let particles = u64::from(s.entity_count);
        let size = self.render.size;
        let cells = self.active_cells();
        let mut lines = vec![
            format!("Particles      {}", grouped(particles)),
            format!("Trail canvas   {} × {}", s.canvas[0], s.canvas[1]),
            match cells {
                Some(c) => format!(
                    "Render         {} × {} px · Cells work {} × {}",
                    size[0], size[1], c.padded[0], c.padded[1]
                ),
                None => format!("Render         {} × {} px", size[0], size[1]),
            },
            match cells {
                Some(c) if c.tiled_state_buffer().is_some() => {
                    "Style          Cells · Fast Cells".to_owned()
                }
                Some(c) => match c.max_sprite_batches() {
                    1 => "Style          Cells · up to 1 sprite pass".to_owned(),
                    n => format!("Style          Cells · up to {n} sprite passes"),
                },
                None => format!("Style          {}", self.style_label()),
            },
            format!(
                "Motion blur    {} requested · {} samples last frame",
                self.state.prefs.motion_blur_samples, self.render.samples_taken
            ),
            format!(
                "Ticks          {} of {} requested in the last scene",
                self.last_scene_steps[0], self.last_scene_steps[1]
            ),
        ];
        let rate = self.achieved_tick_rate();
        lines.push(format!(
            "Ticks/s        {}",
            rate.map_or("measuring".into(), |r| format!("{r:.0}"))
        ));
        lines.push(format!(
            "Updates/s      {}",
            rate.map_or("measuring".into(), |r| format!(
                "{} particle updates",
                compact_count((r * particles as f64) as u64)
            ))
        ));
        lines
    }
    /// GPU frame time against the budget, stacked by stage while stage profiling is on.
    fn gpu_budget(&self, ui: &mut egui::Ui) {
        use crate::performance::{Band, TARGET_GPU_MS};
        let (Some(median), Some(p95)) = (self.gpu_window.median(), self.gpu_window.p95()) else {
            ui.small(if self.frame_timer.is_none() && self.profiler.is_none() {
                "GPU timestamps are unavailable on this device; CPU timings remain available."
            } else {
                "GPU time appears after the next scene submission."
            });
            return;
        };
        ui.monospace(format!(
            "GPU frame      {median:.2} ms median · {p95:.2} ms p95 · {} frames",
            self.gpu_window.len()
        ));
        let width = ui.available_width().clamp(120.0, 340.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 12.0), Sense::hover());
        let full_scale = (TARGET_GPU_MS * 2.0).max(p95 * 1.1);
        let x = |ms: f64| rect.left() + (ms / full_scale).clamp(0.0, 1.0) as f32 * rect.width();
        let painter = ui.painter();
        painter.rect_filled(rect, 2.0, Color32::from_gray(45));
        let rgb = |c: [u8; 3]| Color32::from_rgb(c[0], c[1], c[2]);
        let stages = self.stage_sample();
        if let Some(sample) = stages {
            let mut start = 0.0;
            for (i, ms) in sample.stages_ms.iter().enumerate() {
                if *ms <= 0.0 {
                    continue;
                }
                painter.rect_filled(
                    Rect::from_min_max(
                        egui::pos2(x(start), rect.top()),
                        egui::pos2(x(start + ms), rect.bottom()),
                    ),
                    0.0,
                    STAGE_COLORS[i],
                );
                start += ms;
            }
            if sample.total_ms > start {
                painter.rect_filled(
                    Rect::from_min_max(
                        egui::pos2(x(start), rect.top()),
                        egui::pos2(x(sample.total_ms), rect.bottom()),
                    ),
                    0.0,
                    Color32::from_gray(95),
                );
            }
        } else {
            let band = if median <= TARGET_GPU_MS {
                Band::Green
            } else if median <= TARGET_GPU_MS * 1.5 {
                Band::Yellow
            } else {
                Band::Red
            };
            painter.rect_filled(
                Rect::from_min_max(rect.min, egui::pos2(x(median), rect.bottom())),
                2.0,
                rgb(band.label_color()),
            );
        }
        painter.vline(
            x(p95),
            rect.y_range(),
            egui::Stroke::new(1.0, Color32::from_gray(200)),
        );
        painter.vline(
            x(TARGET_GPU_MS),
            rect.y_range().expand(2.0),
            egui::Stroke::new(2.0, Color32::WHITE),
        );
        let headroom = TARGET_GPU_MS - median;
        ui.small(format!(
            "Budget {TARGET_GPU_MS:.0} ms (world 1 · physics rate 15): {} White line: budget; grey tick: p95.",
            if headroom >= 0.0 {
                format!("{headroom:.2} ms headroom.")
            } else {
                format!("{:.2} ms over.", -headroom)
            }
        ));
        if let Some(sample) = stages {
            ui.horizontal_wrapped(|ui| {
                for (i, (name, ms)) in crate::profiling::Stage::NAMES
                    .iter()
                    .zip(sample.stages_ms)
                    .enumerate()
                    .filter(|(_, (_, ms))| *ms > 0.0)
                {
                    let (swatch, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
                    ui.painter().rect_filled(swatch, 1.0, STAGE_COLORS[i]);
                    ui.small(format!("{name} {ms:.2}"));
                }
            });
            if let Some(per_step) = sample.physics_ms_per_step() {
                ui.small(format!(
                    "Physics {per_step:.3} ms per tick across {} ticks (entity + trails).",
                    sample.steps
                ));
            }
        }
        if self.gpu_window.quantized() {
            ui.small("Timestamps are rounded to 0.1 ms (Chrome's default). Enable chrome://flags/#enable-webgpu-developer-features for finer timing.");
        }
    }
    /// Snapshot of measured workload and timing for the UI and native cadence probe.
    pub fn performance_report(&self) -> serde_json::Value {
        let info = self.rs.adapter.get_info();
        let s = &self.gpu.sizing;
        let cells = self.active_cells();
        let rate = self.achieved_tick_rate();
        let stages = self.stage_sample().map(|sample| {
            serde_json::json!({
                "total_ms": sample.total_ms,
                "steps": sample.steps,
                "physics_ms_per_step": sample.physics_ms_per_step(),
                "stages_ms": crate::profiling::Stage::NAMES
                    .iter()
                    .zip(sample.stages_ms)
                    .map(|(name, ms)| ((*name).to_owned(), serde_json::json!(ms)))
                    .collect::<serde_json::Map<_, _>>(),
            })
        });
        serde_json::json!({
            "version": 1,
            "application": "primordial-soup",
            "kind": "performance-report",
            "adapter": info.name,
            "backend": format!("{:?}", info.backend),
            "driver": info.driver,
            "driver_info": info.driver_info,
            "workgroup": self.workgroup,
            "workload": {
                "particles": s.entity_count,
                "trail_canvas": s.canvas,
                "world_size": self.state.prefs.world_size,
                "physics_steps": self.state.prefs.physics_steps,
                "simulation": self.state.simulation,
                "render_size": self.render.size,
                "style": self.style_label(),
                "cells_padded_size": cells.map(|c| c.padded),
                "cells_sprite_batches_max": cells.map(|c| c.max_sprite_batches()),
                "motion_blur_samples": self.state.prefs.motion_blur_samples,
                "bloom": self.state.prefs.bloom_enabled,
                "last_scene_ticks": self.last_scene_steps[0],
                "last_scene_requested_ticks": self.last_scene_steps[1],
                "completed_ticks": self.clock.completed_ticks(),
                "presentations": self.presentations,
                "submissions": self.submissions,
                "camera": {"zoom":self.camera.zoom,"pan":self.camera.pan,"trail_mode":self.camera.trail_mode},
                "cells_fast_requested": self.state.prefs.cells.fast,
                "cells_fast_available": cells.is_some_and(|c| c.tiled_state_buffer().is_some()),
                "actual_motion_blur_samples": self.render.samples_taken,
                "achieved_ticks_per_second": rate,
                "particle_updates_per_second": rate.map(|r| r * f64::from(s.entity_count)),
                "paused": self.paused,
            },
            "gpu": {
                "median_ms": self.gpu_window.median(),
                "p95_ms": self.gpu_window.p95(),
                "samples": self.gpu_window.len(),
                "window_ms": crate::performance::GPU_WINDOW_MS,
                "target_ms": crate::performance::TARGET_GPU_MS,
                "timestamps_rounded_to_0_1_ms": self.gpu_window.quantized(),
                "method": if self.profiling_enabled { "stage profiler" } else { "frame timer" },
                "stages": stages,
            },
            "fps": {
                "readout": self.fps.readout,
                "band": format!("{:?}", self.fps.band),
                "live": self.meter_live(),
            },
            "cpu": {
                "frame_interval_ms": self.frame_interval_ms,
                "encode_ms": self.cpu_encode_ms,
                "submit_ms": self.cpu_submit_ms,
            },
        })
    }
    /// Set a finite live workload for the native surface-cadence measurement tool.
    #[doc(hidden)]
    pub fn set_measurement_view(&mut self, physics_steps: u32, camera: Camera) {
        self.cancel_rate_calibration();
        self.state.prefs.physics_steps = physics_steps.clamp(1, 120);
        self.state.prefs.cells.style = crate::cells::Style::Cells;
        self.state.prefs.cells.fast = true;
        self.state.prefs.motion_blur_samples = 1;
        self.state.auto_calibrate = false;
        self.calibration = None;
        self.state.simulation = Default::default();
        self.clock.clear_wall_debt();
        self.camera = Camera {
            pan: camera.pan.map(|v| if v.is_finite() { v } else { 0.0 }),
            trail_mode: camera.trail_mode,
            ..Default::default()
        };
        self.camera.set_zoom(camera.zoom);
        self.paused = false;
        self.held_view = None;
        self.last_tick_plan = Instant::now();
        self.presentation_clock = PresentationClock::new(self.last_tick_plan);
        self.context.request_repaint();
    }
    fn timing_window(&mut self, ctx: &egui::Context) {
        if !self.show_timing || self.panels_hidden {
            return;
        }
        let old_settings = self.state.simulation;
        let old_auto = self.state.auto_calibrate;
        let mut open = self.show_timing;
        studio_ui::window("Simulation Timing", self.ui_layout(ctx).dialogs)
            .open(&mut open)
            .default_width(360.0)
            .show(ctx, |ui| {
                studio_ui::dialog_controls(ui);
                ui.label(format!("Completed tick {}", self.clock.completed_ticks()));
                ui.small("Reset initializes the world at tick 0. Each tick keeps the original numerical update.");
                ui.add_enabled_ui(!self.comparison_busy() && (self.recorder.is_none() && self.pending_export.is_none()) && self.calibration.is_none() && self.rate_calibration.is_none(), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Step 1").clicked() { self.advance_exactly(1); }
                        ui.add(egui::DragValue::new(&mut self.advance_steps).range(0..=1_000_000));
                        if ui.button("Advance N").clicked() { self.advance_exactly(self.advance_steps); }
                    });
                    ui.small(format!("{} exact ticks remaining; executes at most 120 per batch", self.clock.pending_exact_steps()));
                    ui.separator();
                    ui.selectable_value(&mut self.state.simulation.mode, crate::simulation::SimulationMode::PerFrame, "Frame based: Physics Rate steps per frame");
                    ui.selectable_value(&mut self.state.simulation.mode, crate::simulation::SimulationMode::FixedRate, "Fixed simulation ticks per second");
                    ui.add_enabled(self.state.simulation.mode == crate::simulation::SimulationMode::FixedRate,
                        egui::Slider::new(&mut self.state.simulation.ticks_per_second, 1..=7200).logarithmic(true).text("Ticks / second"));
                    ui.add(egui::Slider::new(&mut self.state.simulation.presentation_fps, 1..=60).text("Presentation FPS"));
                    ui.small("Fixed ticks preserve elapsed simulation time across display rates. If overloaded, owed ticks remain queued.");
                    ui.label(format!("Automatic clock lag: {} ticks", self.clock.wall_debt_ticks()));
                    ui.checkbox(&mut self.state.auto_calibrate, "Automatic calibration on startup");
                });
                ui.separator();
                ui.strong("Network comparison");
                ui.small("Restart clears particles, all trails, painted Walls/Trails and network memory. Camera influence and automatic calibration are disabled.");
                if self.comparison_busy() {
                    ui.label("Comparison running; simulation settings are locked.");
                    if ui.button("Cancel comparison").clicked() {self.cancel_comparison();}
                } else {
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Restart comparison").clicked() {self.restart_comparison(0);}
                        if ui.button("Run N ticks").clicked() {self.restart_comparison(self.advance_steps);}
                    });
                }
                if self.comparison_result.is_some() {
                    let incomplete=self.comparison_result.as_ref().is_some_and(|r|r["status"]!="complete");
                    ui.label(if incomplete {"Comparison incomplete (cancelled)"} else if self.comparison_stale {"Previous comparison (settings changed)"} else {"Comparison complete"});
                    if ui.button("Save comparison JSON").clicked() {
                        let record=serde_json::to_vec_pretty(self.comparison_result.as_ref().unwrap()).unwrap();
                        self.notice=download("network-comparison.json",&record,"application/json").unwrap_or_else(|e|e.to_string());
                    }
                }
                ui.small("Hidden tabs and minimized windows suspend ticks and discard elapsed wall time. Queued exact work resumes on return. Recordings wait too; in-flight encoding can finish.");
                ui.small("Idle pause has no app timer. Active capture, camera, calibration and deferred saves wake only as needed.");
                ui.separator();
                let dot = self.dot_state(self.origin.elapsed().as_secs_f64() * 1000.0);
                let tooltip = self.performance_tooltip(dot);
                ui.horizontal(|ui| {
                    status_dot(ui, dot);
                    ui.strong("Workload");
                })
                .response
                .on_hover_text(tooltip);
                ui.monospace(self.workload_lines().join("\n"));
                self.gpu_budget(ui);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Copy performance report").on_hover_text("JSON with adapter, workload, GPU median/p95 and stage timings, for comparing devices or builds").clicked() {
                        ctx.copy_text(serde_json::to_string_pretty(&self.performance_report()).unwrap_or_default());
                        self.notice = "Copied performance report (JSON)".into();
                    }
                    if ui.button("Save").clicked() {
                        let report = serde_json::to_vec_pretty(&self.performance_report()).unwrap_or_default();
                        self.notice = download("performance-report.json", &report, "application/json").unwrap_or_else(|e| e.to_string());
                    }
                });
                ui.separator();
                let previous = self.profiling_enabled;
                ui.add_enabled(!self.comparison_busy(),egui::Checkbox::new(&mut self.profiling_enabled,"Profile GPU stages (locks calibration)"));
                if previous != self.profiling_enabled {
                    if self.profiling_enabled {
                        self.cancel_rate_calibration();
                        self.calibration = None;
                        self.state.auto_calibrate = false;
                        self.profiler = crate::profiling::GpuProfiler::new(&self.rs.device);
                    } else {
                        self.profiler = None;
                        self.last_gpu_sample = None;
                    }
                }
                if self.profiling_enabled && self.profiler.is_none() {
                    ui.label("GPU timestamps unavailable on this device; CPU timings remain available.");
                }
                ui.monospace(format!("Frame interval {:.3} ms{}\nCPU prepare/encode {:.3} ms\nQueue submission call {:.3} ms\nScene submissions {}", self.frame_interval_ms, if self.paused {" (input-driven while paused)"} else {""}, self.cpu_encode_ms, self.cpu_submit_ms, self.submissions));
                if let Some(sample) = self.last_gpu_sample.as_ref() {
                    ui.monospace(format!("GPU total {:.3} ms (last profiled frame)", sample.total_ms));
                    for (name, ms) in crate::profiling::Stage::NAMES.iter().zip(sample.stages_ms) {
                        ui.monospace(format!("{name}: {ms:.3} ms"));
                    }
                }
                ui.monospace(format!("Requested next repaint: {}", self.requested_delay.map_or("on change".into(), |d| format!("{:.2} ms", d.as_secs_f64()*1000.0))));
                ui.small("Optional timestamps add overhead. They exclude egui composition and display presentation. Compare fixed workloads with calibration off.");
            });
        self.show_timing = open;
        if self.state.simulation != old_settings {
            self.state.simulation.sanitize();
            self.cancel_rate_calibration();
            self.calibration = None;
            if self.state.simulation.mode != old_settings.mode
                || self.state.simulation.ticks_per_second != old_settings.ticks_per_second
            {
                self.clock.clear_wall_debt();
                self.last_tick_plan = Instant::now();
            }
        }
        if self.state.simulation != old_settings || self.state.auto_calibrate != old_auto {
            self.comparison_stale = true;
            self.state_dirty = true;
        }
    }
    fn set_project(
        &mut self,
        next: Project,
        label: &str,
        reset: bool,
        now: f64,
        coalesce: Option<&str>,
    ) {
        if self.pending_project.is_some() {
            return;
        }
        if self.comparison_busy() {
            self.notice = "Cancel comparison before editing the simulation".into();
            return;
        }
        if (next.config().mode() != self.state.project.config().mode()
            || next.config().network_revision() != self.state.project.config().network_revision())
            && (self.calibration.is_some() || self.rate_calibration.is_some())
        {
            self.notice = "Finish calibration before changing network".into();
            return;
        }
        let mode_changed = next.config().mode() != self.state.project.config().mode()
            || next.config().network_revision() != self.state.project.config().network_revision();
        if next == self.state.project {
            if reset {
                self.reset();
            }
            return;
        }
        if self.recorder.is_some() || self.pending_export.is_some() {
            self.notice = "Finish or cancel recording before changing the project".into();
            return;
        }
        let mut history = self.history.clone();
        history.record(
            self.preview
                .clone()
                .unwrap_or_else(|| self.state.project.clone()),
            next.clone(),
            label,
            coalesce,
            now,
        );
        self.queue_project(next, history, label, reset || mode_changed);
    }
    fn queue_project(&mut self, next: Project, history: History, label: &str, reset: bool) {
        self.queue_project_for(next, history, label, reset, ProjectPurpose::Apply);
    }
    fn queue_project_for(
        &mut self,
        next: Project,
        history: History,
        label: &str,
        reset: bool,
        purpose: ProjectPurpose,
    ) {
        if (self.recorder.is_some() || self.pending_export.is_some())
            || self.calibration.is_some()
            || self.rate_calibration.is_some()
            || self.comparison_busy()
        {
            self.notice =
                "Finish recording, calibration or comparison before changing the project".into();
            return;
        }
        if self.pending_project.is_some() {
            self.notice = "A project change is still being prepared".into();
            return;
        }
        let oom = self
            .rs
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self
            .rs
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let prepared = self
            .gpu
            .prepare_project(&self.rs.device, &self.rs.queue, &next);
        let validation = validation.pop();
        let oom = oom.pop();
        let status = Arc::new(Mutex::new(None));
        let result = status.clone();
        let context = self.context.clone();
        let future = async move {
            let error = validation.await.or(oom.await).map(|e| e.to_string());
            *result.lock().unwrap() = Some(error);
            context.request_repaint();
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(future);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(future);
        match prepared {
            Ok(prepared) => {
                self.notice = format!("Preparing {} network…", next.config().mode().label());
                self.pending_project = Some(PendingProject {
                    prepared,
                    next,
                    history,
                    label: label.into(),
                    reset,
                    result: status,
                    purpose,
                });
                self.context.request_repaint();
            }
            Err(e) => self.notice = format!("Could not apply project: {e}"),
        }
    }
    fn finish_project_apply(&mut self) {
        let result = self
            .pending_project
            .as_ref()
            .and_then(|p| p.result.lock().unwrap().take());
        let Some(error) = result else {
            return;
        };
        let pending = self.pending_project.take().unwrap();
        if let Some(error) = error {
            self.notice = format!("Could not apply project: {error}");
            return;
        }
        self.gpu.commit_project(pending.prepared);
        self.notice.clear();
        self.state.project = pending.next;
        match pending.purpose {
            ProjectPurpose::Preview { original } => self.preview = Some(original),
            ProjectPurpose::RestorePreview => self.preview = None,
            ProjectPurpose::Apply => {
                self.preview = None;
                self.history = pending.history;
                self.state_dirty = true;
                self.log_state(&pending.label, self.origin.elapsed().as_secs_f64() * 1000.0);
            }
        }
        self.pick_intent = None;
        self.queued_pick = None;
        self.highlighted = None;
        self.held_view = None;
        self.comparison_stale = true;
        if pending.reset {
            self.reset();
        }
    }
    fn load(&mut self, project: Project, label: &str, now: f64) {
        self.cancel_pending_preview();
        if self.pending_project.is_some()
            || self.comparison_busy()
            || (self.recorder.is_some() || self.pending_export.is_some())
        {
            return;
        }
        if let Some(before) = self.preview.clone() {
            let mut history = self.history.clone();
            history.record(before, project.clone(), label, None, now);
            self.queue_project(project, history, label, true);
        } else {
            self.set_project(project, label, true, now, None);
        }
    }
    fn navigate_history(&mut self, action: HistoryAction) {
        if self.pending_project.is_some()
            || self.comparison_busy()
            || self.calibration.is_some()
            || self.rate_calibration.is_some()
            || (self.recorder.is_some() || self.pending_export.is_some())
        {
            self.notice = "Finish the current operation before undo/redo".into();
            return;
        }
        let redo = matches!(action, HistoryAction::Redo);
        let mut history = self.history.clone();
        let next = if redo { history.redo() } else { history.undo() };
        if let Some(next) = next {
            let reset = matches!(action, HistoryAction::PreviousGeneration)
                || next.config().mode() != self.state.project.config().mode()
                || next.config().network_revision()
                    != self.state.project.config().network_revision();
            self.queue_project(next, history, if redo { "Redo" } else { "Undo" }, reset);
        }
    }
    fn reroll(&mut self, random_rule: bool, now: f64) {
        if !random_rule
            && (rule_generated(&self.state.project)
                || self.state.project.config().mutation_scale == 0.0)
        {
            return;
        }
        let mut p = self.state.project.clone();
        p.config_mut().mutation_seed = random_seed();
        if random_rule {
            p.config_mut().randomize_network();
        }
        self.set_project(
            p,
            if random_rule {
                "Randomize Behavior"
            } else {
                "Reroll Mutations"
            },
            self.state.prefs.reset_on_behavior_change,
            now,
            None,
        );
    }
    fn choose_layout(&mut self, cohorts: u32, now: f64) {
        let mut p = self.state.project.clone();
        p.config_mut().cohorts = cohorts;
        p.config_mut().initial_conditions = 0;
        self.set_project(p, "Cohort Layout", true, now, None);
    }
    fn adopt(&mut self, now: f64) {
        if !rule_generated(&self.state.project) && self.state.project.config().mutation_scale == 0.0
        {
            return;
        }
        let Some(pick) = self.highlighted.take() else {
            return;
        };
        let mut p = self.state.project.clone();
        if pick.mode != p.config().mode() {
            return;
        }
        p.config_mut()
            .adopt_parent(pick.rule.iter().map(|v| *v as f32).collect(), random_seed());
        // Generate children from the rule the selected cohort actually obeyed.
        self.set_project(
            p,
            &format!("Adopt Cohort {}", pick.cohort as u32 + 1),
            self.state.prefs.reset_on_behavior_change,
            now,
            None,
        );
    }
    fn checkpoint(&mut self) {
        self.state.checkpoints.push(self.state.project.clone());
        if self.state.checkpoints.len() > 100 {
            self.state.checkpoints.remove(0);
        }
        self.state_dirty = true;
        self.notice = "Checkpoint saved".into();
    }
    fn log_state(&mut self, label: &str, now: f64) {
        let _ = now;
        if self.state.prefs.strong_logging {
            let hash = crate::archive::hash_state(&self.state.project);
            if self.archive_cursor.as_ref() == Some(&hash) {
                return;
            }
            let kind = if label == "undo" || label == "redo" {
                label
            } else if self.archive_cursor.is_none() || label.contains("Load") || label == "Import" {
                "enter"
            } else {
                "act"
            };
            let event = crate::archive::Event {
                session: self.archive_session.clone(),
                at: crate::archive::epoch_ms(),
                kind: kind.to_owned(),
                label: label.to_owned(),
                hash: Some(hash.clone()),
                from: self.archive_cursor.clone(),
                state: Some(crate::archive::state(&self.state.project)),
                field: None,
                value: None,
                previous: None,
            };
            self.write_archive(&event);
            self.archive_cursor = Some(hash);
        }
    }
    fn write_archive(&mut self, event: &crate::archive::Event) {
        #[cfg(target_arch = "wasm32")]
        match serde_json::to_string(event) {
            Ok(text) => {
                crate::web::append_archive(&text, self.notices.clone(), self.context.clone())
            }
            Err(e) => self.notice = format!("Archive storage failed: {e}"),
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let result = (|| -> Result<()> {
                if self.archive_writer.is_none() {
                    self.archive_writer = Some(crate::archive::Writer::start_with_wakeup(
                        self.notices.clone(),
                        {
                            let context = self.context.clone();
                            Arc::new(move || context.request_repaint())
                        },
                    )?);
                }
                self.archive_writer.as_ref().unwrap().append(event)
            })();
            if let Err(e) = result {
                self.notice = format!("Archive storage failed: {e}");
                self.state.prefs.strong_logging = false;
            }
        }
        self.archive_count += 1;
    }
    fn archive_command(&mut self, command: &str) {
        if !self.state.prefs.strong_logging {
            return;
        }
        self.write_archive(&crate::archive::Event {
            session: self.archive_session.clone(),
            at: crate::archive::epoch_ms(),
            kind: command.to_owned(),
            label: command.to_owned(),
            hash: None,
            from: None,
            state: None,
            field: None,
            value: None,
            previous: None,
        });
    }
    fn mobile(&self) -> bool {
        self.state.prefs.mobile_mode == 1 || (self.state.prefs.mobile_mode == 0 && self.touch_seen)
    }
    fn share_url(&self) -> Result<String> {
        let name = if self.link_name.trim().is_empty() {
            format!(
                "Shared-{}",
                self.state.project.name.trim_start_matches("Shared-")
            )
        } else {
            self.link_name.clone()
        };
        let mut pairs = vec![(
            "name",
            url_encode(&name.chars().take(64).collect::<String>()),
        )];
        if self.link_splash > 0 {
            pairs.push((
                "splash",
                ["", "welcome", "guide", "controls"][self.link_splash.min(3) as usize].to_owned(),
            ));
        }
        for (enabled, key, value) in [
            (
                self.link_include[0],
                "worldSize",
                self.state.prefs.world_size.to_string(),
            ),
            (
                self.link_include[1],
                "canvasAspect",
                self.state.prefs.canvas_aspect.to_string(),
            ),
            (
                self.link_include[2],
                "brightness",
                self.state.prefs.brightness.to_string(),
            ),
            (
                self.link_include[3],
                "tonemapSoftness",
                self.state.prefs.tonemap_softness.to_string(),
            ),
            (
                self.link_include[4],
                "trailmap",
                u8::from(self.camera.trail_mode).to_string(),
            ),
        ] {
            if enabled {
                pairs.push((key, value));
            }
        }
        Ok(format!(
            "{}?{}{}",
            if self.state.project.extended() {
                app_share_base_url()
            } else {
                share_base_url()
            },
            pairs
                .into_iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("&"),
            share::encode_link(&self.state.project)?
        ))
    }
    fn share(&mut self, ctx: &egui::Context) {
        match self.share_url() {
            Ok(url) => {
                ctx.copy_text(url.clone());
                self.transfer_text = url;
                self.notice = "Share link copied".into();
            }
            Err(e) => self.notice = e.to_string(),
        }
    }

    fn import_text(&mut self, text: &str, now: f64) -> Result<()> {
        anyhow::ensure!(
            (self.recorder.is_none() && self.pending_export.is_none()),
            "Finish or cancel recording before importing a project"
        );
        if serde_json::from_str::<serde_json::Value>(text).is_ok_and(|v| v.get("files").is_some()) {
            let saves = crate::export::import_saves(text)?;
            let mut count = 0;
            let total = saves.len();
            for (name, project) in saves {
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    self.state.saves.entry(name)
                {
                    entry.insert(project);
                    count += 1;
                }
            }
            self.save_now = true;
            self.notice = format!(
                "Imported {count} custom saves; {} already saved",
                total - count
            );
        } else {
            let project = share::decode_text(text)?
                .ok_or_else(|| anyhow::anyhow!("No preset found in the text"))?;
            self.load(project, "Import", now);
            self.notice = "Imported preset".into();
        }
        Ok(())
    }

    fn toggle_panels(&mut self, ctx: &egui::Context) {
        self.panels_hidden = !self.panels_hidden;
        self.mutation_draft = None;
        egui::Popup::close_all(ctx);
        ctx.memory_mut(|memory| {
            if let Some(focused) = memory.focused() {
                memory.surrender_focus(focused);
            }
        });
        self.crop_anchor = None;
        self.stroke = Default::default();
        self.canvas_buttons = [false; 2];
        #[cfg(target_arch = "wasm32")]
        crate::web::set_panels_hidden(self.panels_hidden);
    }

    fn select_tool(&mut self, tool: Tool) {
        self.tool = tool;
        self.stroke = Default::default();
        self.highlighted = None;
        self.render.clear_cells_highlight();
        if tool == Tool::Select && self.inspector_tab == InspectorTab::Brush {
            self.inspector_tab = InspectorTab::World;
        }
    }

    fn hotkeys(&mut self, ctx: &egui::Context, now: f64) {
        if ctx.text_edit_focused() {
            return;
        }
        let commands: Vec<_> = crate::interaction::command_keys(ctx)
            .into_iter()
            .filter(|(key, _)| {
                (!self.comparison_busy() && self.pending_project.is_none())
                    || matches!(
                        key,
                        Key::M | Key::Home | Key::X | Key::H | Key::Slash | Key::P
                    )
            })
            .collect();
        let pressed = |key| commands.iter().any(|(k, _)| *k == key);
        let shifted = |key| {
            commands
                .iter()
                .rev()
                .find(|(k, _)| *k == key)
                .is_some_and(|(_, m)| m.shift)
        };
        if pressed(Key::Space) {
            self.toggle_pause();
        }
        if pressed(Key::R) {
            self.reset();
        }
        if pressed(Key::F) {
            self.reroll(false, now);
        }
        if pressed(Key::B) {
            self.reroll(true, now);
        }
        if pressed(Key::Z) {
            self.navigate_history(if shifted(Key::Z) {
                HistoryAction::Redo
            } else {
                HistoryAction::Undo
            });
        }
        if pressed(Key::C) {
            if shifted(Key::C) {
                self.share(ctx);
            } else {
                self.checkpoint();
            }
        }
        if pressed(Key::V) {
            if shifted(Key::V) {
                self.transfer = true;
            } else if let Some(p) = self.state.checkpoints.last() {
                self.load(p.clone(), "Load Checkpoint", now);
            }
        }
        if pressed(Key::M) {
            self.camera.toggle_mode();
            self.held_view = None;
        }
        if pressed(Key::G) {
            let mut next = self.state.project.clone();
            let config = next.config_mut();
            config.color_by_cohort = !config.color_by_cohort;
            self.set_project(next, "Toggle Color by Cohort", false, now, None);
        }
        for (key, _) in &commands {
            let change = match key {
                Key::OpenBracket => -0.01,
                Key::CloseBracket => 0.01,
                _ => continue,
            };
            let mut next = self.state.project.clone();
            let scale = ((next.config().mutation_scale + change) * 100.0).round() / 100.0;
            let scale = scale.clamp(0.0, 0.6);
            next.config_mut().mutation_scale = scale;
            self.set_project(next, "Mutation Scale", false, now, Some("mutation"));
            if self.state.project.config().mutation_scale == scale {
                self.notice = format!("Mutation Scale: {scale:.2}");
            }
        }
        for (key, _) in &commands {
            let increase = match key {
                Key::Comma => false,
                Key::Period => true,
                _ => continue,
            };
            self.cancel_rate_calibration();
            let rate = self.state.prefs.physics_steps.clamp(1, 60);
            self.state.prefs.physics_steps = if increase {
                ((rate / 5 + 1) * 5).min(60)
            } else {
                ((rate - 1) / 5 * 5).max(1)
            };
            self.state_dirty = true;
            self.held_view = None;
            self.notice = format!(
                "Physics Rate: {} steps/frame",
                self.state.prefs.physics_steps
            );
        }
        if pressed(Key::Home) {
            self.camera.reset();
            self.held_view = None;
        }
        if pressed(Key::X) {
            self.toggle_panels(ctx);
        }
        if pressed(Key::Enter) {
            self.confirm_selection();
        }
        if pressed(Key::H) || pressed(Key::Slash) {
            self.help = !self.help;
        }
        if pressed(Key::P) {
            self.show_capture = true;
            self.capture_qr = shifted(Key::P);
            self.capture_clipboard = true;
        }
        for (key, tool) in [
            (Key::Num1, Tool::Select),
            (Key::Num2, Tool::Shove),
            (Key::Num3, Tool::Walls),
            (Key::Num4, Tool::Trails),
        ] {
            if pressed(key) {
                self.select_tool(tool);
            }
        }
        if self.tool == Tool::Select && (pressed(Key::ArrowRight) || pressed(Key::ArrowLeft)) {
            let n = self.state.project.config().cohorts as i32;
            let cohort = self.highlighted.as_ref().map_or(0, |p| {
                (p.cohort as i32 + if pressed(Key::ArrowRight) { 1 } else { -1 }).rem_euclid(n)
            });
            self.queue_pick(
                self.camera.pan,
                -1.0,
                cohort as f32,
                PickIntent {
                    highlight: None,
                    force: false,
                    highlight_only: true,
                    one_click: false,
                },
            );
        }
    }

    fn queue_pick(&mut self, target: [f32; 2], radius: f32, cohort: f32, intent: PickIntent) {
        if self.gpu.request_pick(
            &self.rs.queue,
            &self.state.project.world,
            target,
            radius,
            cohort,
        ) {
            self.pick_intent = Some(intent);
        } else {
            self.queued_pick = Some(QueuedPick {
                target,
                radius,
                cohort,
                intent,
            });
        }
    }
    fn confirm_selection(&mut self) {
        if self.tool != Tool::Select {
            return;
        }
        let one = self.state.prefs.one_click_selection || self.state.project.config().cohorts <= 1;
        if !one && self.highlighted.is_none() {
            return;
        }
        let highlight = self.highlighted.as_ref().map(|p| p.cohort);
        self.queue_pick(
            self.camera.pan,
            -1.0,
            highlight.unwrap_or(-1.0),
            PickIntent {
                highlight,
                force: true,
                highlight_only: false,
                one_click: one,
            },
        );
    }
    fn resolve_pick(&mut self, now: f64) -> Result<()> {
        let picked = match self.gpu.take_pick() {
            Ok(picked) => picked,
            Err(e) => {
                // A failed readback ends the pick. Leaving its intent would keep
                // encoding scenes for a result that never arrives.
                self.pick_intent = None;
                self.queued_pick = None;
                return Err(e);
            }
        };
        if let Some(result) = picked {
            let intent = self.pick_intent.take();
            if let Some(next) = self.queued_pick.take() {
                self.queue_pick(next.target, next.radius, next.cohort, next.intent);
                return Ok(());
            }
            let confirm = intent.is_some_and(|i| {
                !i.highlight_only
                    && result.as_ref().is_some_and(|p| {
                        i.force
                            || ((i.one_click || !self.mobile())
                                && crate::interaction::should_confirm(
                                    i.highlight,
                                    p.cohort,
                                    i.one_click,
                                ))
                    })
            });
            let clearing = result.is_none();
            self.highlighted = result;
            if !(clearing && self.render.clear_cells_highlight()) {
                self.held_view = None;
            }
            if confirm {
                self.adopt(now);
            }
        }
        Ok(())
    }

    fn ui_layout(&self, ctx: &egui::Context) -> studio_ui::Layout {
        studio_ui::Layout::new(ctx.content_rect(), self.show_panels, true)
    }

    fn status_controls(&mut self, ui: &mut egui::Ui, now: f64) {
        ui.scope(|ui| {
            ui.spacing_mut().interact_size.y = 18.0;
            ui.spacing_mut().button_padding = egui::vec2(2.0, 0.0);
            ui.horizontal(|ui| {
                if self.state.prefs.show_fps_counter {
                    let dot = self.dot_state(now);
                    status_dot(ui, dot);
                    let fps = if dot == crate::performance::DotState::Idle {
                        format!("{} fps · {}", self.fps.readout, self.meter_idle_label().to_lowercase())
                    } else { format!("{} fps", self.fps.readout) };
                    if ui.add(egui::Button::new(egui::RichText::new(fps).size(11.0).color(studio_ui::MUTED)).frame(false))
                        .on_hover_text(format!("{}\n\n{}", self.badge_text(dot), self.performance_tooltip(dot))).clicked()
                    {
                        self.inspector_tab = InspectorTab::Settings;
                        self.show_panels = true;
                    }
                }
                if self.state.prefs.show_physics_slider {
                    if self.state.prefs.show_fps_counter { ui.label(egui::RichText::new("·").size(11.0).color(studio_ui::MUTED)); }
                    let rate = ui.add(egui::Button::new(egui::RichText::new(format!("Physics {}", self.rate_label())).size(11.0).color(studio_ui::MUTED)).frame(false))
                        .on_hover_text("Configured physics rate. Click to adjust; right-click to calibrate. FPS measurement pauses with the simulation.");
                    if rate.clicked() {
                        self.inspector_tab = InspectorTab::Settings;
                        self.show_panels = true;
                    }
                    if rate.secondary_clicked() { self.start_rate_calibration(); }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("X to hide").size(11.0).color(studio_ui::MUTED));
                });
            });
        });
    }

    fn speed_controls(&mut self, ui: &mut egui::Ui) {
        let fixed = self.state.simulation.mode == crate::simulation::SimulationMode::FixedRate;
        let previous = if fixed {
            self.state.simulation.ticks_per_second
        } else {
            self.state.prefs.physics_steps
        };
        let mut speed = previous;
        let locked = self.comparison_busy()
            || self.pending_project.is_some()
            || self.recorder.is_some()
            || self.pending_export.is_some()
            || self.clock.pending_exact_steps() > 0;
        let changed = ui
            .scope(|ui| {
                ui.spacing_mut().interact_size.y = studio_ui::SPEED_ROW_HEIGHT - 4.0;
                ui.add_enabled_ui(!locked, |ui| {
                    controls::physics_speed_bar(ui, &mut speed, fixed).changed()
                })
                .inner
            })
            .inner;
        if changed {
            self.cancel_rate_calibration();
            self.calibration = None;
            if fixed {
                self.state.simulation.ticks_per_second = speed;
                self.clock.clear_wall_debt();
                self.last_tick_plan = Instant::now();
            } else {
                self.state.prefs.physics_steps = speed;
                if self.state.prefs.strong_logging {
                    self.write_archive(&crate::archive::Event {
                        session: self.archive_session.clone(),
                        at: crate::archive::epoch_ms(),
                        kind: "preference".into(),
                        field: Some("physics_steps".into()),
                        value: Some(serde_json::json!(speed)),
                        previous: Some(serde_json::json!(previous)),
                        label: "Physics Rate".into(),
                        hash: None,
                        from: None,
                        state: None,
                    });
                }
            }
            self.held_view = None;
            self.comparison_stale = true;
            self.state_dirty = true;
        }
    }

    fn cohort_controls(&mut self, ui: &mut egui::Ui, now: f64) {
        let locked = self.pending_project.is_some()
            || self.preview.is_some()
            || self.comparison_busy()
            || self.calibration.is_some()
            || self.rate_calibration.is_some()
            || self.recorder.is_some()
            || self.pending_export.is_some()
            || self.clock.pending_exact_steps() > 0;
        if locked {
            self.mutation_draft = None;
        }
        let config = self.state.project.config();
        let mut mutation = self.mutation_draft.unwrap_or(config.mutation_scale);
        let mut layout = None;
        let response = ui
            .add_enabled_ui(!locked, |ui| {
                ui.horizontal(|ui| {
                    for count in [1, 4, 16] {
                        if studio_ui::cohort_button(
                            ui,
                            count,
                            config.cohorts == count && config.initial_conditions == 0,
                        )
                        .clicked()
                        {
                            layout = Some(count);
                        }
                    }
                    ui.add_space(4.0);
                    ui.vertical(|ui| controls::mutation_bar(ui, &mut mutation))
                        .inner
                })
                .inner
            })
            .inner;
        if response.dragged() || response.is_pointer_button_down_on() {
            self.mutation_draft = Some(mutation);
        } else if self.mutation_draft.take().is_some() || response.changed() {
            let mut project = self.state.project.clone();
            project.config_mut().mutation_scale = mutation;
            self.set_project(project, "Mutation Scale", false, now, Some("mutation"));
        }
        if let Some(count) = layout {
            self.mutation_draft = None;
            self.choose_layout(count, now);
        }
    }

    fn menu(&mut self, ctx: &egui::Context, now: f64) {
        self.preview_hovered = false;
        if self.panels_hidden {
            self.restore_preset_preview();
            return;
        }
        let layout = self.ui_layout(ctx);
        let popup_top = layout.dock.top().min(layout.cohorts.top());
        let locked = self.comparison_busy()
            || self
                .pending_project
                .as_ref()
                .is_some_and(|p| !matches!(p.purpose, ProjectPurpose::Preview { .. }));
        egui::Area::new("cohort controls".into())
            .order(egui::Order::Middle)
            .fixed_pos(layout.cohorts.min)
            .movable(false)
            .show(ctx, |ui| {
                studio_ui::surface().inner_margin(6).show(ui, |ui| {
                    ui.set_width((layout.cohorts.width() - 12.0).max(0.0));
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                    self.cohort_controls(ui, now);
                });
            });
        egui::Area::new("command dock".into())
            .order(egui::Order::Middle)
            .fixed_pos(layout.dock.min)
            .movable(false)
            .show(ctx, |ui| {
                studio_ui::surface().inner_margin(6).show(ui, |ui| {
                    ui.set_width((layout.dock.width() - 12.0).max(0.0));
                    ui.spacing_mut().item_spacing = egui::vec2(4.0, 0.0);
                    self.status_controls(ui, now);
                    ui.add_space(4.0);
                    self.speed_controls(ui);
                    ui.add_space(4.0);
                    ui.spacing_mut().interact_size.y = if layout.narrow { 44.0 } else { 36.0 };
                    ui.add_enabled_ui(!locked, |ui| {
                        ui.horizontal(|ui| {
                            self.playback_controls(ui, layout.narrow);
                            let height = ui.spacing().interact_size.y;
                            let (divider, _) =
                                ui.allocate_exact_size(egui::vec2(8.0, height), Sense::hover());
                            ui.painter().vline(
                                divider.center().x,
                                divider.y_range().shrink(9.0),
                                egui::Stroke::new(1.0, studio_ui::LINE),
                            );
                            studio_ui::menu(ui, "Presets", popup_top, |ui| {
                                self.preset_picker(ui, ctx, now)
                            })
                            .on_hover_text("Fluoddity presets and your saved projects");
                            let icon = match self.tool {
                                Tool::Select => Icon::Select,
                                Tool::Shove => Icon::Shove,
                                Tool::Walls => Icon::Walls,
                                Tool::Trails => Icon::Trails,
                            };
                            let width = if layout.narrow { 44.0 } else { 36.0 };
                            studio_ui::icon_menu(
                                ui,
                                icon,
                                "Interaction tools",
                                width,
                                popup_top,
                                |ui| self.tool_menu(ui),
                            )
                            .on_hover_text(format!("{} · interaction tools", self.tool.label()));
                            if studio_ui::action(
                                ui,
                                Icon::Settings,
                                "Inspect",
                                if layout.narrow { 44.0 } else { 84.0 },
                                false,
                            )
                            .on_hover_text("World, appearance, and settings")
                            .clicked()
                            {
                                self.show_panels = !self.show_panels;
                            }
                            studio_ui::icon_menu(
                                ui,
                                Icon::More,
                                "More commands",
                                width,
                                popup_top,
                                |ui| {
                                    if ui.button("Reset world · R").clicked() {
                                        self.reset();
                                        ui.close();
                                    }
                                    if layout.narrow
                                        && ui
                                            .add_enabled(
                                                self.paused,
                                                egui::Button::new("Step one tick"),
                                            )
                                            .clicked()
                                    {
                                        self.advance_exactly(1);
                                        ui.close();
                                    }
                                    if ui.button("New variation · F").clicked() {
                                        self.reroll(false, now);
                                        ui.close();
                                    }
                                    ui.separator();
                                    self.command_menu(ui, ctx, now);
                                },
                            )
                            .on_hover_text("More commands");
                        });
                    });
                });
            });
        if !self.preview_hovered {
            self.restore_preset_preview();
        }
    }

    fn playback_controls(&mut self, ui: &mut egui::Ui, narrow: bool) {
        let width = if narrow { 44.0 } else { 36.0 };
        if studio_ui::action(
            ui,
            if self.paused { Icon::Play } else { Icon::Pause },
            if self.paused { "Resume" } else { "Pause" },
            width,
            true,
        )
        .on_hover_text("Pause or resume · Space")
        .clicked()
        {
            self.toggle_pause();
        }
        if !narrow {
            ui.add_enabled_ui(self.paused, |ui| {
                if studio_ui::action(ui, Icon::Step, "Step one tick", width, false)
                    .on_hover_text("Advance one tick while paused")
                    .clicked()
                {
                    self.advance_exactly(1);
                }
            });
        }
    }

    fn tool_menu(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("INTERACT")
                .size(10.0)
                .color(studio_ui::MUTED),
        );
        for (tool, label) in [
            (Tool::Select, "Select · 1"),
            (Tool::Shove, "Shove · 2"),
            (Tool::Walls, "Walls · 3"),
            (Tool::Trails, "Trails · 4"),
        ] {
            if ui.selectable_label(self.tool == tool, label).clicked() {
                self.select_tool(tool);
                ui.close();
            }
        }
        ui.separator();
        if ui.button("Fit view · Home").clicked() {
            self.camera.reset();
            self.held_view = None;
            ui.close();
        }
        if self.tool != Tool::Select && ui.button("Brush settings…").clicked() {
            self.inspector_tab = InspectorTab::Brush;
            self.show_panels = true;
            ui.close();
        }
    }

    fn preset_picker(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, now: f64) {
        let mut candidate = None;
        let mut committed = false;
        // A popup remembers its compact height. Let expanding a
        // category grow the viewport instead of clipping its body.
        let height =
            (self.ui_layout(ctx).dock.top() - ctx.content_rect().top() - 24.0).clamp(100.0, 650.0);
        ui.set_max_height(height);
        egui::ScrollArea::vertical()
            .max_height(height)
            .show(ui, |ui| {
                ui.add(egui::Hyperlink::from_label_and_url(
                    "Fluoddity presets · Jesse Gelders",
                    "https://fluoddity.com/",
                )).on_hover_text("Core and Archive are original Fluoddity presets. Your saved presets are stored locally.");
                ui.separator();
                for category in ["Core", "Custom", "Archive"] {
                    let count = if category == "Custom" {
                        self.state.saves.len()
                    } else {
                        self.presets
                            .iter()
                            .filter(|p| p.category == category)
                            .count()
                    };
                    if count == 0 {
                        continue;
                    }
                    let title = if category == "Custom" {
                        format!("Your saved presets ({count})")
                    } else {
                        format!("{category} · Fluoddity ({count})")
                    };
                    egui::CollapsingHeader::new(title)
                        .id_salt(("preset category", category))
                        .default_open(category != "Archive")
                        .show(ui, |ui| {
                            if category == "Custom" {
                                let mut delete = None;
                                for (name, p) in &self.state.saves {
                                    ui.horizontal(|ui| {
                                        let response = ui.button(name);
                                        if response.hovered() {
                                            self.preview_hovered = true;
                                            candidate = Some(p.clone());
                                        }
                                        if response.clicked() {
                                            candidate = Some(p.clone());
                                            committed = true;
                                        }
                                        if ui
                                            .small_button("×")
                                            .on_hover_text("Delete this save")
                                            .clicked()
                                        {
                                            delete = Some(name.clone());
                                        }
                                    });
                                }
                                if let Some(name) = delete {
                                    self.state.saves.remove(&name);
                                    self.save_now = true;
                                }
                            } else {
                                for preset in &self.presets {
                                    if preset.category != category {
                                        continue;
                                    }
                                    let response = ui.button(preset.name);
                                    if response.hovered() {
                                        self.preview_hovered = true;
                                        candidate = Some(preset.project.clone());
                                    }
                                    if response.clicked() {
                                        candidate = Some(preset.project.clone());
                                        committed = true;
                                    }
                                }
                            }
                        });
                }
            });
        if let Some(p) =
            candidate.filter(|_| self.recorder.is_none() && self.pending_export.is_none())
        {
            if committed {
                self.load(p, "Load Preset", now);
                ui.close();
            } else if p != self.state.project
                && p.config().mode() == self.state.project.config().mode()
            {
                self.preview_project(p);
            } else if p.config().mode() != self.state.project.config().mode() {
                self.restore_preset_preview();
            } else {
                self.cancel_pending_preview();
            }
        }
    }

    fn command_menu(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, now: f64) {
        ui.menu_button("File", |ui| {
            ui.menu_button("Presets", |ui| self.preset_picker(ui, ctx, now));
            ui.menu_button("Save", |ui| {
                ui.text_edit_singleline(&mut self.save_name);
                if ui
                    .add_enabled(self.can_save, egui::Button::new("Save as…"))
                    .clicked()
                {
                    let name = crate::export::sanitize_name(&self.save_name);
                    if !name.is_empty() {
                        let mut p = self.state.project.clone();
                        p.name = name.clone();
                        self.state.saves.insert(name.clone(), p);
                        self.notice = format!("Saving {name}…");
                        self.save_now = true;
                        ui.close();
                    }
                }
            });
            if ui.button("Import / Export JSON…").clicked() {
                self.transfer = true;
                self.transfer_text = self.state.project.to_json().unwrap_or_default();
                ui.close();
            }
            if ui.button("Download JSON").clicked() {
                if let Ok(text) = self.state.project.to_json() {
                    self.notice = download(
                        &format!("{}.json", safe_name(&self.state.project.name)),
                        text.as_bytes(),
                        "application/json",
                    )
                    .unwrap_or_else(|e| e.to_string());
                }
                ui.close();
            }
            if ui
                .add_enabled(
                    !self.state.saves.is_empty(),
                    egui::Button::new("Export Custom Saves (ZIP)"),
                )
                .clicked()
            {
                self.notice = crate::export::export_zip(&self.state.saves)
                    .and_then(|bytes| {
                        download(
                            "Primordial-Soup-Custom-Saves.zip",
                            &bytes,
                            "application/zip",
                        )
                    })
                    .unwrap_or_else(|e| e.to_string());
                ui.close();
            }
            if ui.button("Screenshot (P)").clicked() {
                self.show_capture = true;
                self.capture_qr = false;
                self.capture_clipboard = false;
                ui.close();
            }
        });
        ui.menu_button("Share", |ui| {
            if ui.button("Copy Share Link (Shift+C)").clicked() {
                self.share(ctx);
                ui.close();
            }
            if ui.button("Open Share Link (Shift+V)…").clicked() {
                self.transfer = true;
                self.transfer_text.clear();
                ui.close();
            }
            if ui.button("Save Image (P)").clicked() {
                self.show_capture = true;
                self.capture_qr = false;
                self.capture_clipboard = false;
                ui.close();
            }
            if ui.button("Copy Image with QR (Shift+P)…").clicked() {
                self.show_capture = true;
                self.capture_qr = true;
                self.capture_clipboard = true;
                ui.close();
            }
            if ui.button("Project Link Settings…").clicked() {
                self.show_link = true;
                ui.close();
            }
            #[cfg(target_arch = "wasm32")]
            if ui.button("Import Image with QR…").clicked() {
                crate::web::import_image(self.incoming.clone(), ctx.clone());
                ui.close();
            }
            if ui.button("Record World (Video)…").clicked() {
                if self.recorder.is_none() && self.pending_export.is_none() && !self.show_recording
                {
                    self.state.recording_settings.physics_steps = self.state.prefs.physics_steps;
                    self.state.recording_settings.motion_blur_samples =
                        self.state.prefs.motion_blur_samples;
                }
                self.show_recording = true;
                ui.close();
            }
        });
        ui.menu_button("History", |ui| {
            if ui
                .add_enabled(
                    self.history.can_undo(),
                    egui::Button::new(format!("Undo {} (Z)", self.history.undo_label())),
                )
                .clicked()
            {
                self.navigate_history(HistoryAction::Undo);
                ui.close();
            }
            if ui
                .add_enabled(
                    self.history.can_redo(),
                    egui::Button::new(format!("Redo {} (Shift+Z)", self.history.redo_label())),
                )
                .clicked()
            {
                self.navigate_history(HistoryAction::Redo);
                ui.close();
            }
            if ui.button("Set Checkpoint (C)").clicked() {
                self.checkpoint();
            }
            let checkpoints = self.state.checkpoints.clone();
            for (i, p) in checkpoints.into_iter().enumerate().rev() {
                if ui.button(format!("{} · {}", i + 1, p.name)).clicked() {
                    self.load(p, "Load Checkpoint", now);
                    ui.close();
                }
            }
        });
        ui.menu_button("Editor", |ui| {
            if ui.checkbox(&mut self.show_panels, "Inspector").changed() {
                ui.close();
            }
            ui.checkbox(&mut self.debug, "Diagnostic overlay");
            if ui.button("Reset View (Home)").clicked() {
                self.camera.reset();
            }
            if ui.button("Reset Preferences").clicked() {
                if let Err(e) = self.rebuild(Preferences::default()) {
                    self.notice = e.to_string();
                }
                self.save_now = true;
            }
            if self.state.prefs.strong_logging {
                ui.separator();
                ui.label(format!(
                    "{} archive events this session",
                    self.archive_count
                ));
                if ui.button("Export State Archive (JSON)").clicked() {
                    #[cfg(target_arch = "wasm32")]
                    crate::web::export_archive(self.notices.clone(), ctx.clone());
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(writer) = &self.archive_writer {
                        if let Err(e) = writer.export() {
                            self.notice = e.to_string();
                        }
                    }
                    ui.close();
                }
                if ui.button("Clear State Archive").clicked() {
                    #[cfg(target_arch = "wasm32")]
                    crate::web::clear_archive(self.notices.clone(), ctx.clone());
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(writer) = &self.archive_writer {
                        if let Err(e) = writer.clear() {
                            self.notice = e.to_string();
                        }
                    }
                    self.archive_cursor = None;
                    self.archive_count = 0;
                    ui.close();
                }
            }
        });
        ui.menu_button("Simulation", |ui| {
            if ui.button("Timing & Experiments…").clicked() {
                self.show_timing = true;
                ui.close();
            }
            if ui.button("Step one tick").clicked() {
                self.advance_exactly(1);
                ui.close();
            }
            if ui
                .button(if self.paused {
                    "Resume (Space)"
                } else {
                    "Pause (Space)"
                })
                .clicked()
            {
                self.toggle_pause();
                ui.close();
            }
            if ui.button("Reset (R)").clicked() {
                self.reset();
                ui.close();
            }
            if ui.button("Randomize Behavior (B)").clicked() {
                self.reroll(true, now);
                ui.close();
            }
            if ui.button("View: Particles / Trails (M)").clicked() {
                self.camera.toggle_mode();
                self.held_view = None;
            }
            if ui.button("Webcam Input…").clicked() {
                self.show_camera = true;
                ui.close();
            }
            if ui.button("Auto-calibrate GPU").clicked() {
                self.start_calibration();
                ui.close();
            }
            if ui
                .button(if self.rate_calibration.is_some() {
                    "Cancel Physics Rate Calibration"
                } else {
                    "Auto-calibrate Physics Rate"
                })
                .clicked()
            {
                self.start_rate_calibration();
                ui.close();
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("Guide and Controls (?)").clicked() {
                self.help = true;
                ui.close();
            }
            ui.hyperlink_to("Fluoddity by Jesse Gelders", "https://fluoddity.com/");
        });
    }

    fn restore_preset_preview(&mut self) {
        self.cancel_pending_preview();
        if self.pending_project.is_some() {
            return;
        }
        if let Some(before) = self.preview.clone() {
            self.queue_project_for(
                before,
                self.history.clone(),
                "Restore preview",
                true,
                ProjectPurpose::RestorePreview,
            );
        }
    }
    fn cancel_pending_preview(&mut self) {
        if self
            .pending_project
            .as_ref()
            .is_some_and(|p| matches!(p.purpose, ProjectPurpose::Preview { .. }))
        {
            self.pending_project = None;
        }
    }
    fn preview_project(&mut self, project: Project) {
        if self
            .pending_project
            .as_ref()
            .is_some_and(|p| p.next == project)
        {
            return;
        }
        self.cancel_pending_preview();
        let original = self
            .preview
            .clone()
            .unwrap_or_else(|| self.state.project.clone());
        self.queue_project_for(
            project,
            self.history.clone(),
            "Preview",
            true,
            ProjectPurpose::Preview { original },
        );
    }
    fn persistent_state(&self) -> SavedState {
        let mut state = self.state.for_persistence(self.preview.as_ref());
        if self.pending_renderer.is_some() {
            state.prefs.cells = self.render.appearance().clone();
        }
        state
    }

    fn network_settings_window(&mut self, ctx: &egui::Context, now: f64) {
        if !self.show_network_settings || self.panels_hidden {
            return;
        }
        let mut open = self.show_network_settings;
        let mut next = self.state.project.clone();
        let locked = self.comparison_busy()
            || self.pending_project.is_some()
            || self.recorder.is_some()
            || self.pending_export.is_some()
            || self.calibration.is_some()
            || self.rate_calibration.is_some();
        studio_ui::window("Network Settings", self.ui_layout(ctx).dialogs)
            .open(&mut open).collapsible(false).resizable(false)
            .default_pos([20.0, 140.0]).default_width(320.0)
            .show(ctx, |ui| {
                studio_ui::dialog_controls(ui);
                ui.add_enabled_ui(!locked, |ui| {
                    let c = next.config_mut();
                    let modern = c.mode()!=crate::network::Mode::Fourier && c.network_revision()>=3;
                    let help = if modern { "Controls sensitivity to vector flow. Occupancy and own speed are separate inputs." }
                        else { "Controls sensitivity to vector trails." };
                    crate::controls::slider(ui,"network_sensor_gain","Sensor Gain",&mut c.sensor_gain,0.0,8.0,1.0,false,false,help);
                    ui.add(egui::Slider::new(&mut c.mutation_scale,0.0..=0.6).text("Mutation Scale"))
                        .on_hover_text("Variation around an adopted parent. Random search generates independent rules.");
                    ui.small(if modern {"Two local flow vectors, two occupancy trails, and own speed. Signed acceleration and strafe outputs."}
                        else {"Two local vector sensors. Saved legacy models keep their original equations."});
                });
            });
        self.show_network_settings = open;
        if next != self.state.project {
            let reset = next.config().sensor_gain != self.state.project.config().sensor_gain
                && self.state.prefs.reset_on_behavior_change;
            self.set_project(
                next,
                "Network Settings",
                reset,
                now,
                Some("network-settings"),
            );
        }
    }

    fn toolbar(&mut self, ctx: &egui::Context, _now: f64) {
        if self.panels_hidden || self.comparison_busy() || self.pending_project.is_some() {
            return;
        }
        let layout = self.ui_layout(ctx);
        if layout.narrow && self.show_panels {
            return;
        }
        egui::Area::new("scene identity".into())
            .order(egui::Order::Middle)
            .fixed_pos(layout.identity.min)
            .movable(false)
            .show(ctx, |ui| {
                ui.set_width(layout.identity.width());
                ui.spacing_mut().item_spacing = egui::vec2(6.0, 2.0);
                if let Some(pick) = self.highlighted.as_ref().filter(|_| !self.show_panels) {
                    let cohort = pick.cohort as u32 + 1;
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("Cohort {cohort}")).size(13.0));
                        if ui
                            .button("Adopt")
                            .on_hover_text("Create the next generation · Enter")
                            .clicked()
                        {
                            self.confirm_selection();
                        }
                        if ui
                            .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                            .clicked()
                        {
                            self.navigate_history(HistoryAction::Undo);
                        }
                    });
                    return;
                }
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(&self.state.project.name)
                            .size(14.0)
                            .color(studio_ui::TEXT),
                    )
                    .truncate(),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "{} · {} particles",
                        self.state.project.config().mode().label(),
                        crate::performance::grouped(u64::from(self.gpu.sizing.entity_count))
                    ))
                    .size(11.0)
                    .color(studio_ui::MUTED),
                );
            });
    }

    fn world_controls(&mut self, ui: &mut egui::Ui) -> Option<InspectorAction> {
        let mut action = None;
        ui.label(
            egui::RichText::new("BEHAVIOR")
                .size(10.0)
                .color(studio_ui::MUTED),
        );
        let config = self.state.project.config();
        let mut mode = config.mode();
        ui.horizontal(|ui| {
            for candidate in crate::network::Mode::ALL {
                ui.selectable_value(&mut mode, candidate, candidate.label())
                    .on_hover_text(
                        "Restores this model's rule and restarts particles, trails, and memory.",
                    );
            }
            if ui.small_button("Tune…").clicked() {
                self.show_network_settings = true;
            }
        });
        if mode != config.mode() {
            action = Some(InspectorAction::Mode(mode));
        }
        ui.small(format!(
            "{} parameters · {}",
            config.network_parameters(),
            if config.mode() == crate::network::Mode::Fourier {
                "vector sensing"
            } else if config.network_revision() >= 3 {
                "flow, occupancy & speed"
            } else {
                "legacy sensors"
            }
        ));
        ui.add_space(5.0);
        ui.label("Cohort layout · restarts the world");
        ui.horizontal(|ui| {
            for (count, label) in [(1, "Single"), (4, "2 × 2"), (16, "4 × 4")] {
                if ui
                    .selectable_label(config.cohorts == count, label)
                    .on_hover_text(format!("Restart with {count} cohorts"))
                    .clicked()
                {
                    action = Some(InspectorAction::Layout(count));
                }
            }
        });
        let mut mutation = config.mutation_scale;
        let fences = config.cohort_fences;
        controls::slider(
            ui,
            "toolbar_mutation",
            "Mutation scale",
            &mut mutation,
            0.0,
            0.6,
            1.0,
            false,
            false,
            "Variation around the parent behavior. 0 keeps every cohort identical.",
        );
        if mutation != config.mutation_scale || fences != config.cohort_fences {
            action = Some(InspectorAction::Mutation(mutation, fences));
        }
        ui.horizontal(|ui| {
            let mut trails = self.camera.trail_mode;
            ui.label("View");
            ui.selectable_value(&mut trails, false, "Particles");
            ui.selectable_value(&mut trails, true, "Trails");
            if trails != self.camera.trail_mode {
                self.camera.toggle_mode();
                self.held_view = None;
            }
        });
        if let Some(pick) = self.highlighted.as_ref() {
            ui.separator();
            ui.label(format!("Cohort {} selected", pick.cohort as u32 + 1));
            ui.horizontal(|ui| {
                if ui
                    .button("Create next generation")
                    .on_hover_text("Adopt this cohort as the parent behavior")
                    .clicked()
                {
                    action = Some(InspectorAction::Adopt);
                }
                if ui
                    .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                    .clicked()
                {
                    action = Some(InspectorAction::Undo);
                }
            });
        } else {
            ui.small("Select a particle to inspect its cohort.");
        }
        action
    }

    fn panels(&mut self, ctx: &egui::Context, now: f64) {
        if self.comparison_busy() || self.pending_project.is_some() {
            return;
        }
        if self.panels_hidden || !self.show_panels {
            return;
        }
        let before = self.state.project.clone();
        let old_prefs = self.state.prefs.clone();
        let hints = controls::PerformanceHints {
            band: self.fps.band,
            physics_ms_per_step: self.stage_sample().and_then(|s| s.physics_ms_per_step()),
            presentation_fps: self.state.simulation.presentation_fps,
            fixed_rate: (self.state.simulation.mode
                == crate::simulation::SimulationMode::FixedRate)
                .then_some(self.state.simulation.ticks_per_second),
        };
        let mut layout = self.ui_layout(ctx);
        if layout.narrow && !self.notice.is_empty() {
            layout.inspector.max.y -= if self.calibration.is_some() {
                72.0
            } else {
                36.0
            };
        }
        let mut command = None;
        egui::Area::new("world inspector".into())
            .order(egui::Order::Middle)
            .fixed_pos(layout.inspector.min)
            .movable(false)
            .show(ctx, |ui| {
                studio_ui::surface().show(ui, |ui| {
                    ui.set_width((layout.inspector.width() - 24.0).max(0.0));
                    ui.set_max_height((layout.inspector.height() - 24.0).max(0.0));
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Inspector")
                                .size(14.0)
                                .color(studio_ui::TEXT)
                                .strong(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if studio_ui::action(
                                ui,
                                Icon::Close,
                                "Close inspector",
                                if layout.narrow { 44.0 } else { 32.0 },
                                false,
                            )
                            .clicked()
                            {
                                self.show_panels = false;
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        for tab in [
                            InspectorTab::World,
                            InspectorTab::Appearance,
                            InspectorTab::Settings,
                            InspectorTab::Brush,
                        ] {
                            if tab == InspectorTab::Brush && self.tool == Tool::Select {
                                continue;
                            }
                            let selected = self.inspector_tab == tab;
                            let response = ui
                                .scope(|ui| {
                                    ui.visuals_mut().selection.bg_fill = Color32::TRANSPARENT;
                                    ui.selectable_label(
                                        selected,
                                        egui::RichText::new(tab.label()).color(if selected {
                                            studio_ui::TEXT
                                        } else {
                                            studio_ui::MUTED
                                        }),
                                    )
                                })
                                .inner;
                            if response.clicked() {
                                self.inspector_tab = tab;
                            }
                            if self.inspector_tab == tab {
                                ui.painter().hline(
                                    response.rect.x_range().shrink(7.0),
                                    response.rect.bottom(),
                                    egui::Stroke::new(1.5, studio_ui::ACCENT),
                                );
                            }
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .id_salt(self.inspector_tab.label())
                        .max_height(
                            (layout.inspector.bottom() - ui.cursor().top() - 12.0).max(24.0),
                        )
                        .auto_shrink([false, false])
                        .show(ui, |ui| match self.inspector_tab {
                            InspectorTab::World => {
                                command = self.world_controls(ui);
                                ui.separator();
                                controls::project_panel(
                                    ui,
                                    &mut self.state.project,
                                    &mut self.state.prefs,
                                );
                            }
                            InspectorTab::Appearance => {
                                let ready = self.pending_renderer.is_none()
                                    && self.recorder.is_none()
                                    && self.pending_export.is_none();
                                controls::appearance_panel(
                                    ui,
                                    &mut self.state.prefs,
                                    &hints,
                                    ready,
                                );
                                if self.pending_renderer.is_some() {
                                    ui.label("Preparing appearance…");
                                }
                                if self.failed_appearance.is_some()
                                    && ui.button("Retry appearance").clicked()
                                {
                                    if let Some(key) = self.failed_appearance.take() {
                                        self.state.prefs.cells.style = key.style;
                                        self.state.prefs.cells.detailed = key.detailed;
                                        self.state.prefs.cells.fast = key.fast;
                                        self.held_view = None;
                                        self.state_dirty = true;
                                    }
                                }
                                if self.camera.trail_mode {
                                    ui.small(
                                        "Switch to Particles view to see the cell appearance.",
                                    );
                                }
                            }
                            InspectorTab::Settings => {
                                controls::preferences_panel(ui, &mut self.state.prefs, &hints);
                                if ui.button("Timing & experiments…").clicked() {
                                    self.show_timing = true;
                                }
                                if ui.button("Auto-calibrate physics rate").clicked() {
                                    self.start_rate_calibration();
                                }
                                egui::CollapsingHeader::new("Device").show(ui, |ui| {
                                    ui.small(self.rs.adapter.get_info().name);
                                    ui.small(format!(
                                        "{} particles",
                                        crate::performance::grouped(u64::from(
                                            self.gpu.sizing.entity_count
                                        ))
                                    ));
                                    egui::ComboBox::from_id_salt("workgroup")
                                        .selected_text(format!("Workgroup {}", self.workgroup))
                                        .show_ui(ui, |ui| {
                                            for n in [64, 128, 256].into_iter().filter(|n| {
                                                *n <= crate::gpu::default_workgroup(
                                                    &self.rs.device.limits(),
                                                )
                                            }) {
                                                ui.selectable_value(
                                                    &mut self.workgroup,
                                                    n,
                                                    n.to_string(),
                                                );
                                            }
                                        });
                                });
                            }
                            InspectorTab::Brush => {
                                ui.label(egui::RichText::new(self.tool.label()).strong());
                                ui.small("Drag to paint · right-drag to erase");
                                controls::drawing_panel(ui, &mut self.state.prefs);
                                if matches!(self.tool, Tool::Walls | Tool::Trails)
                                    && ui
                                        .button(if self.tool == Tool::Walls {
                                            "Clear walls"
                                        } else {
                                            "Clear trails"
                                        })
                                        .clicked()
                                {
                                    self.clear_layer = Some(u32::from(self.tool == Tool::Trails));
                                }
                            }
                        });
                });
            });
        if self.state.project != before {
            let next = self.state.project.clone();
            let layout = next.config().cohorts != before.config().cohorts
                || next.config().initial_conditions != before.config().initial_conditions;
            self.state.project = before;
            self.set_project(next, "Edit Physics", layout, now, Some("physics"));
        }
        let sizing_changed = self.state.prefs.world_size != old_prefs.world_size
            || self.state.prefs.canvas_aspect != old_prefs.canvas_aspect;
        if sizing_changed || self.gpu.workgroup_size() != self.workgroup {
            self.comparison_stale = true;
            let rebuilt = if self.recorder.is_some() || self.pending_export.is_some() {
                Err(anyhow::anyhow!(
                    "Finish or cancel recording before changing world size or workgroup"
                ))
            } else {
                coords::sizing_for(self.state.prefs.world_size, self.state.prefs.canvas_aspect)
                    .and_then(|s| {
                        Engine::new(
                            &self.rs.device,
                            &self.rs.queue,
                            s,
                            &self.state.project,
                            self.workgroup,
                        )
                    })
                    .and_then(|e| {
                        let r = Renderer::new(&self.rs.device, &e, self.render.size)?;
                        Ok((e, r))
                    })
            };
            match rebuilt {
                Ok((e, r)) => {
                    self.gpu = e;
                    self.clock.reset();
                    self.gpu
                        .set_camera_field(&self.rs.device, self.webcam.field_view());
                    self.render = r;
                    self.refresh_texture();
                    self.highlighted = None;
                    self.held_view = None;
                    self.pick_intent = None;
                    self.queued_pick = None;
                }
                Err(e) => {
                    self.state.prefs = old_prefs.clone();
                    self.workgroup = self.gpu.workgroup_size();
                    self.notice = format!("Resize rejected: {e}");
                }
            }
        }
        if self.state.prefs != old_prefs {
            self.held_view = None;
            self.state_dirty = true;
            if self.state.prefs.strong_logging && !old_prefs.strong_logging {
                self.archive_cursor = None;
                self.log_state("Enable Strong Logging", now);
            }
            if self.state.prefs.strong_logging {
                let before = serde_json::to_value(&old_prefs).unwrap_or_default();
                let after = serde_json::to_value(&self.state.prefs).unwrap_or_default();
                if let Some(fields) = after.as_object() {
                    for (field, value) in fields {
                        if field == "strong_logging" || before[field] == *value {
                            continue;
                        }
                        self.write_archive(&crate::archive::Event {
                            session: self.archive_session.clone(),
                            at: crate::archive::epoch_ms(),
                            kind: "preference".into(),
                            label: "Edit Preferences".into(),
                            hash: None,
                            from: None,
                            state: None,
                            field: Some(field.clone()),
                            value: Some(value.clone()),
                            previous: Some(before[field].clone()),
                        });
                    }
                }
            }
        }
        if let Some(command) = command {
            match command {
                InspectorAction::Mode(mode) => {
                    let mut project = self.state.project.clone();
                    project.config_mut().switch_mode(mode, random_seed());
                    self.set_project(project, "Switch Network", true, now, None);
                }
                InspectorAction::Layout(count) => self.choose_layout(count, now),
                InspectorAction::Mutation(scale, fences) => {
                    let mut project = self.state.project.clone();
                    project.config_mut().mutation_scale = scale;
                    project.config_mut().cohort_fences = fences;
                    self.set_project(
                        project,
                        "Mutation Scale / Fences",
                        false,
                        now,
                        Some("mutation"),
                    );
                }
                InspectorAction::Adopt => self.confirm_selection(),
                InspectorAction::Undo => self.navigate_history(HistoryAction::Undo),
            }
        }
    }
    fn refresh_texture(&mut self) {
        self.pending_renderer = None;
        self.failed_appearance = None;
        self.rs
            .renderer
            .write()
            .update_egui_texture_from_wgpu_texture(
                &self.rs.device,
                self.render.output_view(),
                wgpu::FilterMode::Linear,
                self.texture,
            );
    }

    fn dialogs(&mut self, ctx: &egui::Context, now: f64) {
        if self.panels_hidden {
            return;
        }
        self.feature_windows(ctx, now);
        if self.help {
            studio_ui::window("Guide and Controls", self.ui_layout(ctx).dialogs).open(&mut self.help).default_width(480.0)
.show(ctx,|ui| {
            studio_ui::dialog_controls(ui);
            ui.label("Particles leave vector trails and sense them with two sensors. Fourier, MLP, or GRU rules turn those readings into motion and color. Select interesting cohorts and generate children to explore emergent behavior.");
            ui.separator();
            for text in ["Space: pause/resume · R: reset · F: reroll · B: randomize behavior", "1: Select · 2: Shove · 3: Walls · 4: Trails", "Click: highlight cohort · click again / Enter: adopt its rule", "Z / Shift+Z: undo/redo · C / V: checkpoint/load", "M: particles/trails · G: color by cohort · Home: reset view", "Physics Rate: , / . step through 1, 5, 10, …, 60", "[ / ]: lower/raise Mutation Scale by 0.01 (0–0.6)", "X: hide/show all panels", "Wheel: zoom at cursor · middle drag / WASD: pan · Q/E: zoom", "Right drag: erase field · Shift+drag: line · P: screenshot", "Shift+C: copy share link · Shift+V: open share link", "Drop a JSON preset on the canvas to import it"] { ui.label(text); }
            ui.separator(); ui.hyperlink_to("Original Fluoddity source and MIT license","https://github.com/aphid91/Fluoddity-Web");
        });
        }
        if self.transfer {
            let mut open = true;
            let mut import = false;
            studio_ui::window("Import / Export", self.ui_layout(ctx).dialogs).open(&mut open).default_width(630.0).show(ctx,|ui| {
                studio_ui::dialog_controls(ui);
                ui.label("Paste a JSON preset, share link or custom-save backup. Drop a JSON, ZIP or QR image onto the canvas to import a file.");
                ui.add(egui::TextEdit::multiline(&mut self.transfer_text).desired_rows(14).desired_width(f32::INFINITY).code_editor());
                ui.horizontal_wrapped(|ui| { if ui.button("Import").clicked() { import=true; }
                    if ui.button("Copy").clicked() { ctx.copy_text(self.transfer_text.clone()); }
                    if ui.button("Current JSON").clicked() { self.transfer_text=self.state.project.to_json().unwrap_or_default(); }
                    if self.state.project.extended() && self.state.project.config().mode()==crate::network::Mode::Fourier && ui.button("Fourier v9 JSON (drops other modes)").clicked() {
                        let mut p=self.state.project.clone();p.configs.truncate(1);p.selected=0;p.config_mut().networks=None;
                        self.transfer_text=p.to_json().unwrap_or_default();
                    }
                });
            });
            self.transfer = open;
            if import {
                match self.import_text(&self.transfer_text.clone(), now) {
                    Ok(()) => {
                        self.transfer = false;
                    }
                    Err(e) => self.notice = format!("Import rejected: {e}"),
                }
            }
        }
    }

    fn feature_windows(&mut self, ctx: &egui::Context, now: f64) {
        if self.show_camera {
            let mut open = true;
            let mut setup = false;
            studio_ui::window("Webcam Input", self.ui_layout(ctx).dialogs)
                .open(&mut open)
                .default_width(360.0)
                .vscroll(true)
                .show(ctx, |ui| {
                    studio_ui::dialog_controls(ui);
                    self.webcam.ui(ui);
                    ui.separator();
                    ui.label("Works best with Random initial conditions and a high Hazard Rate.");
                    if ui.button("Use recommended particle setup").clicked() {
                        setup = true;
                    }
                    if self.webcam.settings.preview && self.webcam.running {
                        ui.image((self.webcam_texture, egui::Vec2::splat(256.0)));
                    }
                });
            self.show_camera = open;
            self.state.camera_settings = self.webcam.settings.clone();
            if setup {
                let mut p = self.state.project.clone();
                p.config_mut().initial_conditions = 1;
                p.config_mut().hazard_rate = 0.003;
                self.set_project(p, "Camera Setup", false, now, None);
            }
        }
        if self.show_recording {
            let mut open = true;
            let mut start = false;
            let mut discard = false;
            studio_ui::window("Record World", self.ui_layout(ctx).dialogs)
                .open(&mut open)
                .default_width(360.0)
                .vscroll(true)
                .show(ctx, |ui| {
                    studio_ui::dialog_controls(ui);
                    if let Some(r) = self.recorder.as_mut() {
                        let progress = r.progress();
                        ui.label(format!(
                            "{} / {} frames",
                            progress.frames_done, progress.frames_total
                        ));
                        if self.paused {
                            ui.label("Paused — press Space to resume");
                        }
                        if ui.button("Finish early and save").clicked() {
                            r.cancel();
                        }
                        if ui.button("Cancel and discard").clicked() {
                            discard = true;
                        }
                    } else if self.pending_export.is_some() {
                        ui.label("Preparing recording…");
                    } else {
                        start = crate::record::settings(
                            &mut self.state.recording_settings,
                            ui,
                            self.gpu.sizing.canvas,
                        );
                    }
                });
            self.show_recording = open;
            if discard {
                if let Some(mut session) = self.recorder.take() {
                    self.paused = session.was_paused;
                    session.discard();
                    self.held_view = None;
                }
                self.notice = "Recording discarded".into();
            }
            if start
                && (self.pending_project.is_some()
                    || self.pending_renderer.is_some()
                    || self.comparison_busy()
                    || self.clock.pending_exact_steps() > 0)
            {
                self.notice = "Finish exact advancement before starting video export".into();
            } else if start && self.calibration.is_some() {
                self.notice = "Finish GPU calibration before starting video export".into();
            } else if start {
                self.cancel_rate_calibration();
                self.prepare_export();
            }
        }
        if self.show_capture {
            self.stroke = Default::default();
            let mut open = true;
            let mut capture = None;
            studio_ui::window("Screenshot", self.ui_layout(ctx).dialogs).open(&mut open).default_width(350.0)
                .default_pos([ctx.content_rect().center().x-175.0,(ctx.content_rect().height()-400.0).max(32.0)*0.5])
.show(ctx,|ui| {
                studio_ui::dialog_controls(ui);
                ui.label("Drag on the canvas to select a crop. Editor controls and brush overlays are excluded.");
                if let Some([w, h]) = ctx.data_mut(|d| d.get_temp::<[u32; 2]>(egui::Id::new(controls::PHYSICAL_CANVAS_ID))).filter(|p| *p != self.render.size) {
                    ui.small(format!("Render Scale: images are saved at the working resolution, {} × {} of the {w} × {h} screen. Choose 1× in Inspector → Settings → Performance for full resolution.", self.render.size[0], self.render.size[1]));
                }
                ui.add(egui::Slider::new(&mut self.capture_crop[0],1..=self.render.size[0]).text("Width"));
                ui.add(egui::Slider::new(&mut self.capture_crop[1],1..=self.render.size[1]).text("Height"));
                self.capture_crop = [self.capture_crop[0].min(self.render.size[0]),self.capture_crop[1].min(self.render.size[1])];
                for i in 0..2 { self.capture_origin[i] = self.capture_origin[i].min(self.render.size[i]-self.capture_crop[i]); }
                ui.add(egui::Slider::new(&mut self.capture_origin[0],0..=self.render.size[0]-self.capture_crop[0]).text("Left"));
                ui.add(egui::Slider::new(&mut self.capture_origin[1],0..=self.render.size[1]-self.capture_crop[1]).text("Top"));
                if ui.button("Full Frame").clicked(){self.capture_crop=self.render.size; self.capture_origin=[0,0];}
                ui.checkbox(&mut self.capture_qr,"Include QR code linking to this project");
                ui.horizontal(|ui|{if ui.button("Copy Image").clicked(){capture=Some(true);}if ui.button("Save PNG").clicked(){capture=Some(false);}});
            });
            self.show_capture = open;
            if let Some(copy) = capture {
                self.capture_clipboard = copy;
                self.capture_requested = true;
                self.show_capture = false;
            }
        }
        if self.show_link {
            let mut open = true;
            studio_ui::window("Project Link Settings", self.ui_layout(ctx).dialogs)
                .open(&mut open)
                .default_width(340.0)
                .show(ctx, |ui| {
                    studio_ui::dialog_controls(ui);
                    ui.label("Project Name");
                    ui.text_edit_singleline(&mut self.link_name);
                    egui::ComboBox::from_id_salt("link splash")
                        .selected_text(
                            ["Default", "Welcome", "Guide", "Controls"]
                                [self.link_splash.min(3) as usize],
                        )
                        .show_ui(ui, |ui| {
                            for (i, label) in ["Default", "Welcome", "Guide", "Controls"]
                                .iter()
                                .enumerate()
                            {
                                ui.selectable_value(&mut self.link_splash, i as u32, *label);
                            }
                        });
                    if self.state.project.extended() {ui.small("New network projects require Primordial Soup. Native links default to localhost:8080; set PRIMORDIAL_SOUP_URL to your hosted app address.");}
                    ui.label("Include suggested editor settings in copied links:");
                    for (i, label) in [
                        "World Size",
                        "Canvas Aspect",
                        "Brightness",
                        "Tonemap Softness",
                        "Trail Map View",
                    ]
                    .iter()
                    .enumerate()
                    {
                        ui.checkbox(&mut self.link_include[i], *label);
                    }
                    ui.small("The recipient chooses whether to apply these settings.");
                    if ui.button("Copy Link").clicked() {
                        self.share(ctx);
                    }
                });
            self.show_link = open;
        }
        if !self.proposed.is_empty() {
            let proposed = self.proposed.clone();
            let mut apply = false;
            let mut ignore = false;
            studio_ui::window("Suggested Editor Settings", self.ui_layout(ctx).dialogs)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    studio_ui::dialog_controls(ui);
                    ui.label("This project link suggests these editor settings:");
                    for (label, value) in [
                        ("World Size", proposed.world_size.map(|v| v.to_string())),
                        (
                            "Canvas Aspect",
                            proposed.canvas_aspect.map(|v| v.to_string()),
                        ),
                        ("Brightness", proposed.brightness.map(|v| v.to_string())),
                        (
                            "Tonemap Softness",
                            proposed.tonemap_softness.map(|v| v.to_string()),
                        ),
                        ("Trail Map", proposed.trail_mode.map(|v| v.to_string())),
                    ] {
                        if let Some(v) = value {
                            ui.label(format!("{label}: {v}"));
                        }
                    }
                    ui.horizontal(|ui| {
                        apply = ui.button("Apply").clicked();
                        ignore = ui.button("Keep My Settings").clicked();
                    });
                });
            if apply {
                let mut prefs = self.state.prefs.clone();
                if let Some(v) = proposed.world_size {
                    prefs.world_size = v;
                }
                if let Some(v) = proposed.canvas_aspect {
                    prefs.canvas_aspect = v;
                }
                if let Some(v) = proposed.brightness {
                    prefs.brightness = v;
                }
                if let Some(v) = proposed.tonemap_softness {
                    prefs.tonemap_softness = v;
                }
                match self.rebuild(prefs) {
                    Ok(()) => {
                        if let Some(v) = proposed.trail_mode {
                            self.camera.trail_mode = v;
                            self.held_view = None;
                        }
                    }
                    Err(e) => self.notice = e.to_string(),
                }
            }
            if apply || ignore {
                self.proposed = Default::default();
            }
        }
    }
    fn rebuild(&mut self, prefs: Preferences) -> Result<()> {
        anyhow::ensure!(
            (self.recorder.is_none() && self.pending_export.is_none()),
            "Finish or cancel recording before rebuilding the world"
        );
        let size = coords::sizing_for(prefs.world_size, prefs.canvas_aspect)?;
        let mut e = Engine::new(
            &self.rs.device,
            &self.rs.queue,
            size,
            &self.state.project,
            self.workgroup,
        )?;
        e.set_camera_field(&self.rs.device, self.webcam.field_view());
        let r = Renderer::new(&self.rs.device, &e, self.render.size)?;
        self.gpu = e;
        self.clock.reset();
        self.render = r;
        self.state.prefs = prefs;
        self.refresh_texture();
        self.highlighted = None;
        self.pick_intent = None;
        self.queued_pick = None;
        self.held_view = None;
        Ok(())
    }

    fn capture_start(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.capture_requested || self.capture.is_some() {
            return;
        }
        self.capture_requested = false;
        let size = self.render.size;
        let crop = [
            self.capture_crop[0].clamp(1, size[0]),
            self.capture_crop[1].clamp(1, size[1]),
        ];
        let qr = if self.capture_qr {
            match self.share_url() {
                Ok(s) => Some(s),
                Err(e) => {
                    self.notice = e.to_string();
                    return;
                }
            }
        } else {
            None
        };
        let texture = match self.render.clean_texture() {
            Ok(texture) => texture,
            Err(error) => {
                self.notice = error.to_string();
                return;
            }
        };
        let origin = [
            self.capture_origin[0].min(size[0] - crop[0]),
            self.capture_origin[1].min(size[1] - crop[1]),
        ];
        let readback =
            match crate::gpu::readback_rgba(&self.rs.device, encoder, texture, origin, crop) {
                Ok(readback) => readback,
                Err(error) => {
                    self.notice = format!("Screenshot: {error}");
                    return;
                }
            };
        self.capture = Some(Capture {
            readback,
            status: Arc::new(Mutex::new(None)),
            clipboard: self.capture_clipboard,
            qr,
        });
    }
    fn capture_finish(&mut self, ctx: &egui::Context) {
        let Some(c) = self.capture.as_ref() else {
            return;
        };
        let Some(result) = c.status.lock().unwrap().take() else {
            return;
        };
        let c = self.capture.take().unwrap();
        // Completion can be the final service wakeup of an otherwise idle pause.
        ctx.request_repaint();
        match result {
            Err(e) => self.notice = format!("Screenshot failed: {e}"),
            Ok(()) => {
                let map = match c.readback.buffer.slice(..).get_mapped_range() {
                    Ok(m) => m,
                    Err(e) => {
                        c.readback.buffer.unmap();
                        self.notice = format!("Screenshot failed: {e}");
                        return;
                    }
                };
                let pixels = c.readback.rgba(&map);
                drop(map);
                c.readback.buffer.unmap();
                let size = c.readback.size;
                let result = pixels.and_then(|mut pixels| -> Result<Vec<u8>> {
                    if let Some(url) = c.qr {
                        crate::export::stamp_qr(&mut pixels, size, &url)?;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if c.clipboard {
                        ctx.copy_image(egui::ColorImage::from_rgba_unmultiplied(
                            [size[0] as usize, size[1] as usize],
                            &pixels,
                        ));
                    }
                    crate::export::encode_png(&pixels, size)
                });
                self.notice = match result {
                    Ok(bytes) => {
                        if c.clipboard {
                            #[cfg(target_arch = "wasm32")]
                            {
                                crate::web::clipboard_png(
                                    &bytes,
                                    self.notices.clone(),
                                    ctx.clone(),
                                );
                                "Copying image…".to_owned()
                            }
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                "Image sent to clipboard".to_owned()
                            }
                        } else {
                            download(
                                &format!("{}.png", safe_name(&self.state.project.name)),
                                &bytes,
                                "image/png",
                            )
                            .unwrap_or_else(|e| e.to_string())
                        }
                    }
                    Err(e) => format!("Screenshot: {e}"),
                };
            }
        }
    }

    fn prepare_export(&mut self) {
        let d = &self.rs.device;
        let oom = d.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let prepared = crate::record::ExportSession::start(
            d,
            &self.gpu,
            &self.state.recording_settings,
            self.camera,
            &self.state.prefs,
            &self.state.project,
            self.paused,
        );
        let validation = validation.pop();
        let oom = oom.pop();
        let status = Arc::new(Mutex::new(None));
        let result = status.clone();
        let ctx = self.context.clone();
        let future = async move {
            *result.lock().unwrap() = Some(validation.await.or(oom.await).map(|e| e.to_string()));
            ctx.request_repaint();
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(future);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(future);
        match prepared {
            Ok(session) => {
                self.pending_export = Some(PendingExport {
                    session,
                    result: status,
                });
                self.paused = true;
                self.clock.clear_wall_debt();
                self.notice = "Preparing recording…".into();
            }
            Err(e) => self.notice = format!("Video export: {e}"),
        }
    }
    fn finish_export(&mut self) {
        let result = self
            .pending_export
            .as_ref()
            .and_then(|p| p.result.lock().unwrap().take());
        let Some(result) = result else {
            return;
        };
        let pending = self.pending_export.take().unwrap();
        if let Some(error) = result {
            self.paused = pending.session.was_paused;
            self.notice = format!("Video export could not be prepared: {error}");
        } else {
            self.recorder = Some(pending.session);
            self.paused = false;
            self.held_view = None;
            self.notice = "Recording started".into();
        }
        self.last_tick_plan = Instant::now();
    }
    fn prepare_renderer(&mut self, size: [u32; 2]) -> Result<()> {
        self.state.prefs.cells.sanitize();
        let desired = &self.state.prefs.cells;
        let needs = self.render.size != size
            || match self.render.cells.as_ref() {
                Some(c) => {
                    desired.style != crate::cells::Style::Cells
                        || c.detailed != desired.detailed
                        || c.fast != desired.fast
                }
                None => desired.style == crate::cells::Style::Cells,
            };
        if !needs {
            self.render.set_preferences(&self.state.prefs);
            return Ok(());
        }
        let key = AppearanceKey {
            size,
            style: desired.style,
            detailed: desired.detailed,
            fast: desired.fast,
        };
        if self.pending_renderer.is_some() || self.failed_appearance == Some(key) {
            return Ok(());
        }
        let d = &self.rs.device;
        let oom = d.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let prepared = (|| -> Result<Renderer> {
            let mut renderer = Renderer::new(d, &self.gpu, size)?;
            if desired.style == crate::cells::Style::Cells {
                renderer.commit_cells(renderer.prepare_cells(d, &self.gpu, desired)?);
            }
            renderer.set_preferences(&self.state.prefs);
            Ok(renderer)
        })();
        let validation = validation.pop();
        let oom = oom.pop();
        let status = Arc::new(Mutex::new(None));
        let result = status.clone();
        let ctx = self.context.clone();
        let future = async move {
            *result.lock().unwrap() = Some(validation.await.or(oom.await).map(|e| e.to_string()));
            ctx.request_repaint();
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(future);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(future);
        match prepared {
            Ok(render) => {
                self.pending_renderer = Some(PendingRenderer {
                    key,
                    render,
                    settings: desired.clone(),
                    previous: self.render.appearance().clone(),
                    result: status,
                });
                self.context.request_repaint();
            }
            Err(e) => {
                self.failed_appearance = Some(key);
                self.state.prefs.cells = self.render.appearance().clone();
                self.state_dirty = true;
                self.notice = format!("Appearance could not be prepared: {e}");
            }
        }
        Ok(())
    }
    fn finish_renderer(&mut self) {
        let result = self
            .pending_renderer
            .as_ref()
            .and_then(|r| r.result.lock().unwrap().take());
        let Some(result) = result else {
            return;
        };
        let pending = self.pending_renderer.take().unwrap();
        if let Some(error) = result {
            self.failed_appearance = Some(pending.key);
            self.state_dirty = true;
            if self.state.prefs.cells.style == pending.key.style
                && self.state.prefs.cells.detailed == pending.key.detailed
                && self.state.prefs.cells.fast == pending.key.fast
            {
                self.state.prefs.cells.style = pending.previous.style;
                self.state.prefs.cells.detailed = pending.previous.detailed;
                self.state.prefs.cells.fast = pending.previous.fast;
            }
            self.notice = format!("Appearance unchanged: {error}");
            return;
        }
        if self.state.prefs.cells.style != pending.settings.style
            || self.state.prefs.cells.detailed != pending.settings.detailed
            || self.state.prefs.cells.fast != pending.settings.fast
        {
            return;
        }
        self.render = pending.render;
        self.state_dirty = true;
        self.refresh_texture();
        self.held_view = None;
    }
    fn draw_frame(
        &mut self,
        ctx: &egui::Context,
        rect: Rect,
        response: egui::Response,
        _now: f64,
        frame_start: Instant,
    ) -> Result<()> {
        let d = self.rs.device.clone();
        let q = self.rs.queue.clone();
        let physical = [
            (rect.width() * ctx.pixels_per_point()).round().max(1.0) as u32,
            (rect.height() * ctx.pixels_per_point()).round().max(1.0) as u32,
        ];
        ctx.data_mut(|d| d.insert_temp(egui::Id::new(controls::PHYSICAL_CANVAS_ID), physical));
        let cells = self.state.prefs.cells.style == crate::cells::Style::Cells;
        let mut working = self.state.prefs.render_scale.size(physical, cells);
        if cells {
            // Keep Cells within one storage binding instead of failing back to Classic.
            let limits = d.limits();
            working = crate::cells::fit_working_size(
                working,
                u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size),
            );
        }
        ctx.data_mut(|d| d.insert_temp(egui::Id::new(controls::WORKING_CANVAS_ID), working));
        self.prepare_renderer(working)?;
        self.gpu.set_interaction_map(
            &d,
            &q,
            if self.render.cells_active() {
                self.render.cells.as_ref()
            } else {
                None
            },
        );
        let window = [rect.width() as f64, rect.height() as f64];
        let canvas = self.gpu.sizing.canvas.map(f64::from);
        let pointer = response
            .hover_pos()
            .or_else(|| ctx.input(|i| i.pointer.interact_pos()));
        let pixel = pointer.map(|p| [(p.x - rect.left()) as f64, (p.y - rect.top()) as f64]);
        let canvas_hovered = self.pending_renderer.is_none()
            && response.hovered()
            && pointer.is_some_and(|p| ctx.layer_id_at(p) == Some(response.layer_id));
        let capture_active = self.show_capture && !self.panels_hidden;
        if capture_active {
            // Crops are in the working resolution the screenshot reads, which
            // Render Scale can make smaller than the screen's pixels.
            let size = self.render.size;
            let px = pixel.map(|p| {
                [
                    (p[0] / window[0] * size[0] as f64)
                        .round()
                        .clamp(0.0, size[0] as f64) as u32,
                    (p[1] / window[1] * size[1] as f64)
                        .round()
                        .clamp(0.0, size[1] as f64) as u32,
                ]
            });
            if canvas_hovered && ctx.input(|i| i.pointer.primary_pressed()) {
                self.crop_anchor = px;
            }
            if ctx.input(|i| i.pointer.primary_down()) {
                if let Some((a, b)) = self.crop_anchor.zip(px) {
                    self.capture_origin = [
                        a[0].min(b[0]).min(size[0] - 1),
                        a[1].min(b[1]).min(size[1] - 1),
                    ];
                    self.capture_crop = [
                        a[0].abs_diff(b[0])
                            .max(1)
                            .min(size[0] - self.capture_origin[0]),
                        a[1].abs_diff(b[1])
                            .max(1)
                            .min(size[1] - self.capture_origin[1]),
                    ];
                }
            } else {
                self.crop_anchor = None;
            }
            let points = egui::vec2(
                window[0] as f32 / size[0] as f32,
                window[1] as f32 / size[1] as f32,
            );
            let from = rect.min
                + egui::vec2(self.capture_origin[0] as f32, self.capture_origin[1] as f32) * points;
            let to = from
                + egui::vec2(self.capture_crop[0] as f32, self.capture_crop[1] as f32) * points;
            ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                "crop rectangle".into(),
            ))
            .rect_stroke(
                Rect::from_min_max(from, to),
                0.0,
                egui::Stroke::new(1.0, Color32::WHITE),
                egui::StrokeKind::Inside,
            );
        }
        let multi = ctx
            .input(|i| i.multi_touch())
            .filter(|t| t.num_touches >= 2);
        if let Some(t) = multi {
            self.camera.pan_by_pixels(
                [t.translation_delta.x as f64, t.translation_delta.y as f64],
                window,
                canvas,
            );
            if let Some(p) = pixel {
                self.camera.zoom_at_pixel(
                    (t.zoom_delta as f64).ln() / 1.1f64.ln(),
                    p,
                    window,
                    canvas,
                );
            }
            self.held_view = None;
            self.stroke = Default::default();
        }
        for (i, button) in [egui::PointerButton::Primary, egui::PointerButton::Secondary]
            .into_iter()
            .enumerate()
        {
            if ctx.input(|s| s.pointer.button_pressed(button)) {
                self.canvas_buttons[i] = canvas_hovered && multi.is_none();
            }
            if !ctx.input(|s| s.pointer.button_down(button)) {
                self.canvas_buttons[i] = false;
            }
        }
        let mut uv = None;
        let mut world = None;
        if canvas_hovered {
            if let Some(p) = pixel {
                let wheel = ctx.input(|i| i.smooth_scroll_delta.y);
                if wheel != 0.0 {
                    self.camera
                        .zoom_at_pixel(wheel as f64 / 100.0, p, window, canvas);
                    self.held_view = None;
                }
                let input_camera = self
                    .render
                    .cells
                    .as_ref()
                    .filter(|_| self.render.cells_active())
                    .map_or(self.camera, |c| c.displayed_camera);
                let displayed_size = self
                    .render
                    .cells
                    .as_ref()
                    .filter(|_| self.render.cells_active())
                    .map_or(window, |c| c.size.map(|v| v as f64));
                let displayed_pixel = [
                    p[0] / window[0] * displayed_size[0],
                    p[1] / window[1] * displayed_size[1],
                ];
                let w = input_camera.screen_to_world(displayed_pixel, displayed_size, canvas);
                world = Some(w.map(|v| v as f32));
                uv = Some(coords::world_to_uv(w, canvas).map(|v| v as f32));
            }
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = ctx.input(|i| i.pointer.delta());
            self.camera
                .pan_by_pixels([delta.x as f64, delta.y as f64], window, canvas);
            self.held_view = None;
        }
        if !ctx.text_edit_focused() {
            let dt = ctx.input(|i| i.stable_dt.min(0.05)) as f64;
            let movement = ctx.input(|i| {
                [
                    (i.key_down(Key::D) as u8 as f64 - i.key_down(Key::A) as u8 as f64) * 0.9 * dt,
                    (i.key_down(Key::W) as u8 as f64 - i.key_down(Key::S) as u8 as f64) * 0.9 * dt,
                ]
            });
            if movement != [0.0; 2] {
                self.camera.pan_by_fraction(movement, canvas);
                self.held_view = None;
            }
            let zoom =
                ctx.input(|i| i.key_down(Key::E) as u8 as f64 - i.key_down(Key::Q) as u8 as f64);
            if zoom != 0.0 {
                self.camera.zoom_by_factor(2.2f64.powf(zoom * dt) as f32);
                self.held_view = None;
            }
        }
        if !capture_active
            && self.tool == Tool::Select
            && response.clicked()
            && canvas_hovered
            && multi.is_none()
        {
            if let Some(target) = world {
                let tol = 40.0 / ctx.pixels_per_point() as f64;
                let (camera, size) = self
                    .render
                    .cells
                    .as_ref()
                    .filter(|_| self.render.cells_active())
                    .map_or((self.camera, window), |c| {
                        (c.displayed_camera, c.size.map(|v| v as f64))
                    });
                let a = camera.screen_to_world([0.0, 0.0], size, canvas);
                let b = camera.screen_to_world([tol / window[0] * size[0], 0.0], size, canvas);
                let radius = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() as f32;
                let highlight = self.highlighted.as_ref().map(|p| p.cohort);
                self.queue_pick(
                    target,
                    radius,
                    highlight.unwrap_or(-1.0),
                    PickIntent {
                        highlight,
                        force: false,
                        highlight_only: false,
                        one_click: self.state.prefs.one_click_selection
                            || self.state.project.config().cohorts <= 1,
                    },
                );
            }
        }
        if !capture_active
            && self.tool == Tool::Select
            && response.secondary_clicked()
            && canvas_hovered
        {
            if self.highlighted.take().is_some() {
                if !self.render.clear_cells_highlight() {
                    self.held_view = None;
                }
            } else {
                self.navigate_history(HistoryAction::PreviousGeneration);
            }
        }
        let primary = !self.comparison_busy()
            && ctx.input(|i| i.pointer.primary_down())
            && self.canvas_buttons[0]
            && multi.is_none()
            && !capture_active;
        let secondary = !self.comparison_busy()
            && ctx.input(|i| i.pointer.secondary_down())
            && self.canvas_buttons[1]
            && multi.is_none()
            && !capture_active;
        let shift = ctx.input(|i| i.modifiers.shift);
        // Stroke state follows input on every frame; only a returned segment is GPU work.
        let mut stroke = None;
        if !capture_active
            && matches!(self.tool, Tool::Walls | Tool::Trails)
            && canvas_hovered
            && self.clear_layer.is_none()
        {
            if let Some(point) = uv {
                stroke = self.stroke.update(crate::interaction::StrokeInput {
                    uv: point,
                    left_dragging: primary,
                    right_dragging: secondary,
                    left_pressed: self.canvas_buttons[0]
                        && ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary)),
                    right_pressed: self.canvas_buttons[1]
                        && ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary)),
                    shift,
                });
            }
        } else {
            self.stroke = Default::default();
        }
        let highlight = self.highlighted.as_ref().map_or(-1.0, |p| p.cohort);
        let overlays = Overlays {
            walls: self.state.prefs.field_always_show || self.tool == Tool::Walls,
            trails: self.state.prefs.trails_always_show || self.tool == Tool::Trails,
            reticle: uv.unwrap_or([0.0; 2]),
            radius: if self.state.prefs.show_reticle && canvas_hovered && self.tool != Tool::Select
            {
                self.state.prefs.draw_size
            } else {
                0.0
            },
            style: if self.tool == Tool::Shove {
                1
            } else {
                [2, 3, 0, 4][self.state.prefs.brush_mode.min(3) as usize]
            },
            angle: self.state.prefs.draw_angle,
            line: self.stroke.anchor.zip(uv),
            crop: None,
        }
        .normalized();
        let presentation = PresentationKey::new(overlays, highlight, &self.state.prefs);
        if self.presented.is_some_and(|p| p.highlight != highlight) && !self.render.cells_active() {
            // Classic draws the highlight into sampled particles, not in assembly.
            self.held_view = None;
        }
        let normal_clock = self.calibration.is_none()
            && (self.recorder.is_none() && self.pending_export.is_none());
        let interval = self.state.simulation.frame_interval();
        self.presentation_clock.configure(
            frame_start,
            interval,
            self.visible && !self.paused && normal_clock,
        );
        let presentation_due = self.presentation_clock.due(frame_start, interval);
        let tick_due = match self.state.simulation.mode {
            // One automatic frame owns both the physics batch and its image.
            crate::simulation::SimulationMode::PerFrame => presentation_due,
            crate::simulation::SimulationMode::FixedRate => deadline_reached(
                frame_start.saturating_duration_since(self.last_tick_plan),
                self.state
                    .simulation
                    .service_interval(crate::simulation::DEFAULT_MAX_STEPS_PER_UPDATE),
            ),
        } || self.clock.wall_debt_ticks() > 0;
        // Raw input alone is not a reason to present: hovering a paused canvas
        // changes nothing on screen unless an overlay or the camera moves.
        let render_frame = self.held_view != Some(self.camera)
            || self.presented != Some(presentation)
            || stroke.is_some()
            || self.painting_input(ctx)
            || self.capture_requested
            || self.clear_layer.is_some()
            || self.queued_pick.is_some()
            || self.clock.pending_exact_steps() > 0
            || self.webcam.needs_poll()
            || (!self.paused && normal_clock && presentation_due)
            || self
                .recorder
                .as_ref()
                .is_some_and(|r| r.ready() && !self.paused);
        // A requested pick and a comparison's field clear are encoded with the scene.
        let pick_waiting = self.pick_intent.is_some() && !self.gpu.pick_readback_pending();
        let needs_scene = render_frame
            || self.gpu.frame_count == 0
            || (!self.paused && normal_clock && tick_due)
            || self.comparison_clear
            || pick_waiting;
        if !needs_scene {
            self.capture_finish(ctx);
            if self.capture.is_none() {
                self.render.release_clean();
            }
            return Ok(());
        }
        let encode_started = Instant::now();
        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Primordial Soup frame"),
        });
        let mut profile = if self.profiling_enabled && self.calibration.is_none() {
            self.profiler
                .as_mut()
                .and_then(|p| p.begin_frame(&mut encoder))
        } else {
            None
        };
        // Stage profiling reports its own frame total; never instrument a frame twice.
        let mut frame_timing = if !self.profiling_enabled && self.calibration.is_none() {
            self.frame_timer
                .as_mut()
                .and_then(|t| t.begin(&mut encoder))
        } else {
            None
        };
        let camera_polling = self.webcam.needs_poll();
        self.webcam.encode(
            &d,
            &q,
            &mut encoder,
            self.gpu.sizing.canvas[0] as f32 / self.gpu.sizing.canvas[1] as f32,
        );
        if camera_polling && !self.webcam.needs_poll() {
            ctx.request_repaint();
        }
        self.gpu.set_camera_strengths(if self.comparison_busy() {
            [0.0; 2]
        } else {
            self.webcam.strengths()
        });
        if self.comparison_clear {
            self.gpu.clear_field(&mut encoder);
            self.comparison_clear = false;
        }
        let cleared = self.clear_layer.take();
        if let Some(layer) = cleared {
            self.gpu.clear_layer(&q, &mut encoder, layer)?;
        }
        if !self.comparison_busy()
            && (self.painting_input(ctx) || self.webcam.strengths() != [0.0; 2])
        {
            self.comparison_stale = true;
        }
        // A clear queued this frame reset the stroke above, so it never paints too.
        if let Some(stroke) = stroke {
            self.gpu.paint(
                &q,
                &mut encoder,
                Paint {
                    uv: stroke.uv,
                    previous_uv: stroke.previous,
                    size: self.state.prefs.draw_size,
                    power: self.state.prefs.draw_power,
                    erase: stroke.erase,
                    mode: self.state.prefs.brush_mode,
                    layer: u32::from(self.tool == Tool::Trails),
                    angle: self.state.prefs.draw_angle,
                    line_gain: crate::interaction::LINE_STROKE_GAIN,
                },
            )?;
        }
        let recording_ready = self.recorder.as_ref().is_some_and(|r| r.ready());
        let recording_frame = recording_ready && !self.paused && self.visible;
        let elapsed = frame_start.saturating_duration_since(self.last_tick_plan);
        let due = tick_due;
        let logical_steps = if !self.visible || self.calibration.is_some() {
            self.clock.suspend();
            0
        } else if self.recorder.is_some() || self.pending_export.is_some() {
            self.clock.clear_wall_debt();
            if recording_frame {
                self.recorder.as_ref().unwrap().settings.physics_steps
            } else {
                0
            }
        } else if self.paused
            || due
            || self.clock.pending_exact_steps() > 0
            || self.state.simulation.mode == crate::simulation::SimulationMode::FixedRate
        {
            // A pause/reset/settings transition during this UI pass may be newer.
            self.last_tick_plan = self.last_tick_plan.max(frame_start);
            self.clock
                .plan(
                    elapsed,
                    !self.paused,
                    self.state.simulation,
                    self.state.prefs.physics_steps,
                    crate::simulation::DEFAULT_MAX_STEPS_PER_UPDATE,
                )
                .steps
        } else {
            0
        };
        // Frame zero initializes particles; it is not an experimental simulation tick.
        let initialization =
            u32::from(self.gpu.frame_count == 0 && self.visible && self.calibration.is_none());
        let steps = logical_steps + initialization;
        let schedule = render::blur_schedule(
            steps,
            self.recorder
                .as_ref()
                .map_or(self.state.prefs.motion_blur_samples, |r| {
                    r.settings.motion_blur_samples
                }),
        );
        let shove =
            if self.tool == Tool::Shove && canvas_hovered && !self.paused && (primary || secondary)
            {
                world.map(|center| Shove {
                    center,
                    strength: crate::interaction::shove_strength(
                        self.state.prefs.draw_power,
                        steps,
                        !primary && secondary,
                    ),
                    size: crate::interaction::shove_radius(self.state.prefs.draw_size),
                })
            } else {
                None
            };
        self.gpu.prepare_frame(
            &q,
            &self.state.project.world,
            steps,
            shove,
            [
                self.state.prefs.walls_strength * 0.01,
                self.state.prefs.trails_strength * 0.001,
            ],
        )?;
        self.gpu.encode_bake(&mut encoder);
        let hold = !render_frame
            || (steps == 0
                && self.paused
                && self.held_view == Some(self.camera)
                && self.render.samples_taken > 0);
        if recording_frame {
            if let Some(session) = self.recorder.as_mut() {
                session.render.begin_frame(
                    &q,
                    &self.gpu,
                    &session.camera,
                    &session.project,
                    schedule.samples,
                    -1.0,
                );
                session.render.clear_accumulator(&mut encoder);
            }
        }
        if !hold {
            self.render.begin_frame(
                &q,
                &self.gpu,
                &self.camera,
                &self.state.project,
                if steps == 0 { 1 } else { schedule.samples },
                highlight,
            );
            self.render.clear_accumulator(&mut encoder);
            if steps == 0 {
                self.render.sample_profiled(
                    &mut encoder,
                    &self.gpu,
                    &self.camera,
                    profile.as_mut(),
                );
            }
        }
        // Tick execution is independent of reuse of the last presentation image.
        for i in 0..steps {
            self.gpu
                .encode_step_profiled(&mut encoder, i, profile.as_mut());
            if render_frame && i % schedule.stride == schedule.sample_at {
                self.render.sample_profiled(
                    &mut encoder,
                    &self.gpu,
                    &self.camera,
                    profile.as_mut(),
                );
                if recording_frame {
                    if let Some(session) = self.recorder.as_mut() {
                        session.render.sample_profiled(
                            &mut encoder,
                            &self.gpu,
                            &session.camera,
                            profile.as_mut(),
                        );
                    }
                }
            }
        }
        if render_frame {
            self.held_view = Some(self.camera);
        }
        let pick_encoded = self.gpu.encode_pick(&mut encoder);
        if render_frame {
            self.render.present_profiled(
                &q,
                &mut encoder,
                &self.gpu,
                &self.camera,
                &self.state.prefs,
                overlays,
                profile.as_mut(),
            );
            if self.visible && !self.paused && normal_clock && presentation_due {
                self.presentation_clock.consume(frame_start, interval);
            }
            self.presentations += 1;
            self.presented = Some(presentation);
        }
        if recording_frame {
            if let Some(session) = self.recorder.as_mut() {
                session.render.present(
                    &q,
                    &mut encoder,
                    &self.gpu,
                    &session.camera,
                    &session.prefs,
                    Overlays::default(),
                );
            }
        }
        if self.capture_requested {
            self.render.present_clean(
                &d,
                &q,
                &mut encoder,
                &self.gpu,
                &self.camera,
                &self.state.prefs,
            );
        }
        let capture_was_empty = self.capture.is_none();
        self.capture_start(&mut encoder);
        if let Some(profile) = profile.as_mut() {
            profile.finish(&mut encoder);
        }
        if let Some(timing) = frame_timing.as_mut() {
            timing.finish(&mut encoder);
        }
        let commands = encoder.finish();
        self.cpu_encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;
        let submit_started = Instant::now();
        q.submit([commands]);
        self.cpu_submit_ms = submit_started.elapsed().as_secs_f64() * 1000.0;
        self.submissions += 1;
        if !self.comparison_busy() && logical_steps > 0 {
            self.comparison_stale = true;
        }
        self.clock.complete_steps(logical_steps);
        if self.comparison.as_ref().is_some_and(|(target, _, _)| {
            self.clock.completed_ticks() >= *target && self.clock.pending_exact_steps() == 0
        }) {
            self.finish_comparison();
        }
        if self.paused && logical_steps > 0 {
            ctx.request_repaint();
        }
        if let Some(profile) = profile {
            self.profiler.as_mut().unwrap().after_submit(profile);
        }
        if let Some(timing) = frame_timing {
            self.frame_timer.as_mut().unwrap().after_submit(timing);
        }
        let requested = if self.recorder.is_some() || self.pending_export.is_some() || self.paused {
            logical_steps
        } else {
            match self.state.simulation.mode {
                crate::simulation::SimulationMode::PerFrame => self.state.prefs.physics_steps,
                crate::simulation::SimulationMode::FixedRate => logical_steps
                    .saturating_add(self.clock.wall_debt_ticks().min(u64::from(u32::MAX)) as u32),
            }
        };
        self.last_scene_steps = [logical_steps, requested];
        if recording_frame {
            if let Some(r) = self.recorder.as_mut() {
                if let Err(e) = r.recorder.frame(
                    &d,
                    &q,
                    r.render.output_texture(),
                    r.render.size,
                    self.paused,
                ) {
                    r.discard();
                    self.notice = format!("Video export: {e}");
                    self.paused = r.was_paused;
                    self.recorder = None;
                }
            }
        }
        if pick_encoded {
            self.gpu.begin_pick_readback();
        }
        if capture_was_empty {
            if let Some(c) = self.capture.as_ref() {
                let status = c.status.clone();
                c.readback
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        *status.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
                    });
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        d.poll(wgpu::PollType::Poll)?;
        self.capture_finish(ctx);
        if !self.capture_requested && self.capture.is_none() {
            self.render.release_clean();
        }
        Ok(())
    }
}

impl eframe::App for SoupApp {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(target_arch = "wasm32")]
        if self._visibility_wake.take_transition() {
            self.clock.suspend();
            self.presentation_clock.suspend();
            self.last_tick_plan = Instant::now();
        }
        self.visible = app_visible(ctx);
        if !self.visible {
            self.clock.suspend();
            self.presentation_clock.suspend();
            self.last_tick_plan = Instant::now();
        }
        self.service_pending(ctx, frame);
        if self.rate_calibration.as_ref().is_some_and(|r| {
            r.started.elapsed() >= Duration::from_secs(crate::rate_calibration::CEILING_SECONDS)
        }) {
            self.cancel_rate_calibration();
            self.notice = "Physics-rate calibration timed out; original rate restored".into();
        }
        if let Some(c) = self.calibration.as_mut() {
            #[cfg(target_arch = "wasm32")]
            let visible = web_sys::window()
                .and_then(|w| w.document())
                .is_some_and(|d| !d.hidden());
            #[cfg(not(target_arch = "wasm32"))]
            let visible = ctx.input(|i| i.viewport().visible().unwrap_or(true));
            c.set_visible(visible);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
    }
    fn ui(&mut self, root: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx_owned = root.ctx().clone();
        let ctx = &ctx_owned;
        #[cfg(target_arch = "wasm32")]
        let _ = frame;
        let frame_start = Instant::now();
        let now = frame_start
            .saturating_duration_since(self.origin)
            .as_secs_f64()
            * 1000.0;
        let elapsed = frame_start
            .saturating_duration_since(self.previous_frame)
            .as_secs_f32();
        if ctx.current_pass_index() == 0 {
            self.visible = app_visible(ctx);
            self.previous_frame = frame_start;
            self.frame_interval_ms = elapsed as f64 * 1000.0;
            #[cfg(target_arch = "wasm32")]
            let visible = web_sys::window()
                .and_then(|w| w.document())
                .is_some_and(|d| !d.hidden());
            #[cfg(not(target_arch = "wasm32"))]
            let visible = ctx.input(|i| i.viewport().visible().unwrap_or(true));
            self.feed_rate_calibration(elapsed as f64 * 1000.0, visible);
        }
        if ctx.current_pass_index() == 0 {
            self.fps.sample(now, elapsed as f64 * 1000.0);
            controls::begin_performance_frame(ctx);
            // Mirrors the meter's update condition below: only live frames change its verdict.
            if self.meter_live() {
                self.meter_running_since.get_or_insert(now);
            } else {
                self.meter_running_since = None;
            }
            self.tick_rate.push(now, self.clock.completed_ticks());
        }
        if let Some(calibration) = self.calibration.as_mut() {
            match calibration.tick(&self.rs.device, &self.rs.queue) {
                Ok(Some((world, steps))) => {
                    self.calibration = None;
                    let mut prefs = self.state.prefs.clone();
                    prefs.world_size = world;
                    prefs.physics_steps = steps;
                    prefs.calibration_version = crate::calibration::VERSION;
                    if world < 0.5 {
                        prefs.bloom_enabled = false;
                    }
                    match self.rebuild(prefs) {
                        Ok(()) => {
                            self.save_now = true;
                            self.notice =
                                format!("Calibrated: world {world} · {steps} physics steps/frame");
                            self.start_rate_calibration();
                        }
                        Err(e) => self.notice = e.to_string(),
                    }
                }
                Ok(None) => {
                    self.notice = format!("Calibrating GPU: rung {} / 6", calibration.progress());
                    if self.visible {
                        ctx.request_repaint();
                    }
                }
                Err(e) => {
                    self.calibration = None;
                    self.state.prefs.calibration_version = crate::calibration::VERSION;
                    self.notice = format!("Calibration stopped: {e}");
                }
            }
        }
        if ctx.input(|i| {
            i.raw
                .events
                .iter()
                .any(|e| matches!(e, egui::Event::Touch { .. }))
        }) {
            self.touch_seen = true;
        }
        if ctx.current_pass_index() == 0 {
            self.hotkeys(ctx, now);
        }
        let fatal = self.error.lock().unwrap().clone();
        if let Some(detail) = fatal {
            self.cancel_rate_calibration();
            self.calibration = None;
            self.webcam.stop();
            if let Some(r) = self.recorder.as_mut() {
                r.discard();
            }
            self.recorder = None;
            egui::CentralPanel::default().show(root, |ui| {
                ui.heading("GPU unavailable");
                ui.label(detail);
            });
            return;
        }
        let mut canvas = None;
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(root, |ui| {
                let rect = ui.max_rect();
                let response = ui.allocate_rect(rect, Sense::click_and_drag());
                ui.painter().image(
                    self.texture,
                    rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                canvas = Some((rect, response));
            });
        self.menu(ctx, now);
        self.toolbar(ctx, now);
        self.panels(ctx, now);
        self.dialogs(ctx, now);
        self.timing_window(ctx);
        self.network_settings_window(ctx, now);
        let chrome = self.ui_layout(ctx);
        if self.panels_hidden {
            egui::Area::new("restore controls".into())
                .order(egui::Order::Middle)
                .anchor(egui::Align2::LEFT_BOTTOM, [chrome.margin, -chrome.margin])
                .show(ctx, |ui| {
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("Show controls · X").size(11.0))
                                .frame(false),
                        )
                        .clicked()
                    {
                        self.toggle_panels(ctx);
                    }
                });
        }
        if ctx.current_pass_index() == 0 && self.meter_live() {
            self.fps.update(now, controls::performance_dragging(ctx));
        }
        if !self.panels_hidden && !self.notice.is_empty() {
            let bottom =
                ctx.content_rect().bottom() - chrome.dock.top().min(chrome.cohorts.top()) + 10.0;
            let available = if self.show_panels && !chrome.narrow {
                chrome.inspector.left() - chrome.margin * 2.0
            } else {
                ctx.content_rect().width() - chrome.margin * 2.0
            };
            egui::Area::new("notice".into())
                .anchor(egui::Align2::LEFT_BOTTOM, [chrome.margin, -bottom])
                .order(egui::Order::Middle)
                .show(ctx, |ui| {
                    studio_ui::surface().inner_margin(6).show(ui, |ui| {
                        ui.set_max_width((available - 12.0).clamp(120.0, 400.0));
                        ui.spacing_mut().interact_size.y = 24.0;
                        ui.spacing_mut().button_padding = egui::vec2(4.0, 2.0);
                        ui.horizontal(|ui| {
                            let width = ui
                                .painter()
                                .layout_no_wrap(
                                    self.notice.clone(),
                                    egui::FontId::proportional(12.0),
                                    studio_ui::TEXT,
                                )
                                .size()
                                .x
                                .min((available - 50.0).clamp(60.0, 360.0));
                            ui.add_sized(
                                [width, 24.0],
                                egui::Label::new(egui::RichText::new(&self.notice).size(12.0))
                                    .truncate(),
                            )
                            .on_hover_text(&self.notice);
                            let dismiss = ui.button("×").on_hover_text("Dismiss message");
                            dismiss.widget_info(|| {
                                egui::WidgetInfo::labeled(
                                    egui::WidgetType::Button,
                                    ui.is_enabled(),
                                    "Dismiss message",
                                )
                            });
                            if dismiss.clicked() {
                                self.notice.clear();
                            }
                        });
                        if let Some(c) = self.calibration.as_mut() {
                            if ui.button("Cancel calibration").clicked() {
                                c.cancel();
                            }
                        }
                    });
                });
        }
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let image = file
                    .path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| {
                        matches!(
                            s.to_ascii_lowercase().as_str(),
                            "png" | "jpg" | "jpeg" | "webp" | "gif"
                        )
                    });
                let result = if file
                    .path()
                    .metadata()
                    .is_ok_and(|m| m.len() > if image { 64 } else { 16 } * 1024 * 1024)
                {
                    Err("Import file exceeds size limit".into())
                } else if image {
                    crate::export::decode_image_file(file.path())
                        .and_then(|p| p.to_json())
                        .map_err(|e| e.to_string())
                } else {
                    file.bytes()
                        .and_then(|b| decode_import_bytes(&b).map_err(|e| e.to_string()))
                };
                self.incoming.lock().unwrap().push(result);
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(web_file) = file.web_file().filter(|f| f.type_().starts_with("image/"))
                {
                    crate::web::import_image_file(
                        web_file.clone().into(),
                        self.incoming.clone(),
                        ctx.clone(),
                    );
                    continue;
                }
                let incoming = self.incoming.clone();
                let ctx = ctx.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = if file
                        .web_file()
                        .is_some_and(|f| f.size() > 16.0 * 1024.0 * 1024.0)
                    {
                        Err("Import file exceeds 16 MiB".into())
                    } else {
                        file.bytes_async()
                            .await
                            .and_then(|b| decode_import_bytes(&b).map_err(|e| e.to_string()))
                    };
                    incoming.lock().unwrap().push(result);
                    ctx.request_repaint();
                });
            }
        }
        let incoming = std::mem::take(&mut *self.incoming.lock().unwrap());
        for result in incoming {
            match result {
                Ok(text) => match self.import_text(&text, now) {
                    Ok(()) => {}
                    Err(e) => self.notice = e.to_string(),
                },
                Err(e) => self.notice = e,
            }
        }
        if ctx.current_pass_index() == 0 {
            if let Some((rect, response)) = canvas.filter(|_| self.visible) {
                if let Err(e) = self.draw_frame(ctx, rect, response, now, frame_start) {
                    *self.error.lock().unwrap() = Some(e.to_string());
                }
            } else {
                self.capture_finish(ctx);
                if self.capture.is_none() {
                    self.render.release_clean();
                }
            }
            self.frames += 1;
        }
        self.save_pending(ctx, frame);
        if !self.panels_hidden
            && self.debug
            && !self.show_panels
            && !chrome.compact
            && self.highlighted.is_none()
            && self.notice.is_empty()
        {
            egui::Area::new("debug".into()).order(egui::Order::Middle)
                .fixed_pos(egui::pos2(chrome.identity.left(), chrome.identity.bottom() + 8.0)).show(ctx, |ui| {
                    ui.set_max_width(360.0);
                    ui.label(egui::RichText::new(format!("{} · {:?}\n{} particles · {} × {} trails\nframe {} · substep {} · {} steps/frame\nblur {} samples · workgroup {}\nFrame interval {:.2} ms",self.rs.adapter.get_info().name,self.rs.adapter.get_info().backend,
                        self.gpu.sizing.entity_count,self.gpu.sizing.canvas[0],self.gpu.sizing.canvas[1],self.frames,self.gpu.frame_count,self.state.prefs.physics_steps,self.render.samples_taken,self.workgroup,elapsed*1000.0))
                        .monospace().size(11.0).color(studio_ui::MUTED));
                });
        }
        // The status line is shown and read only on `?debug` pages; skip the
        // formatting and DOM write everywhere else, and when nothing changed.
        #[cfg(target_arch = "wasm32")]
        if ctx.current_pass_index() == 0 && self.status_text.is_some() {
            let text = format!(
                "frame={} substep={} particles={} preset={} paused={} camera={} error={} export_gpu_bytes={} ticks={} pending={} debt={} mode={:?} presentation_fps={} visible={} submissions={} calibration={} encode_ms={:.4} submit_ms={:.4} presentations={} requested_ms={:?} ticks_per_second={} network={} network_pending={} comparison={}",
                self.frames,
                self.gpu.frame_count,
                self.gpu.sizing.entity_count,
                self.state.project.name,
                self.paused,
                if self.camera.trail_mode {
                    "trail"
                } else {
                    "particles"
                },
                self.error.lock().unwrap().is_some(),
                self.render.clean_output_bytes()
                    + self.recorder.as_ref().map_or(0, |r| r.allocated_bytes()),
                self.clock.completed_ticks(),
                self.clock.pending_exact_steps(),
                self.clock.wall_debt_ticks(),
                self.state.simulation.mode,
                self.state.simulation.presentation_fps,
                self.visible,
                self.submissions,
                self.calibration.is_some() || self.rate_calibration.is_some(),
                self.cpu_encode_ms,
                self.cpu_submit_ms,
                self.presentations,
                self.requested_delay.map(|d| d.as_secs_f64() * 1000.0),
                self.state.simulation.ticks_per_second,
                self.state.project.config().mode().label(),
                self.pending_project.is_some(),
                if self.comparison_busy() {
                    "running"
                } else {
                    self.comparison_result
                        .as_ref()
                        .and_then(|r| r["status"].as_str())
                        .unwrap_or("none")
                }
            );
            if self.status_text.as_ref() != Some(&text) {
                crate::web::status(&text);
                self.status_text = Some(text);
            }
        }
        let service = if self.pending_export.is_some()
            || self.pending_renderer.is_some()
            || self.capture.is_some()
            || self.pick_intent.is_some()
            || (self.visible && self.queued_pick.is_some())
            || self.profiler.as_ref().is_some_and(|p| p.has_pending())
            || self.frame_timer.as_ref().is_some_and(|t| t.has_pending())
            || self.recorder.as_ref().is_some_and(|r| !r.ready())
            || (self.visible && (self.calibration.is_some() || self.rate_calibration.is_some()))
        {
            Some(Duration::from_millis(16))
        } else if self.visible && self.webcam.needs_poll() {
            Some(Duration::from_millis(1000 / 15))
        } else {
            None
        };
        let save_deadline = self
            .state_dirty
            .then(|| Duration::from_secs(10).saturating_sub(self.last_save.elapsed()));
        let normal_running = self.visible
            && !self.paused
            && self.calibration.is_none()
            && (self.recorder.is_none() && self.pending_export.is_none());
        let navigating = self.visible
            && !ctx.text_edit_focused()
            && ctx.input(|i| {
                [Key::W, Key::A, Key::S, Key::D, Key::Q, Key::E]
                    .iter()
                    .any(|k| i.key_down(*k))
            });
        let service = if normal_running {
            let display_delay = self
                .presentation_clock
                .remaining(Instant::now(), self.state.simulation.frame_interval());
            let tick_delay = if self.clock.wall_debt_ticks() > 0 {
                Duration::ZERO
            } else if self.state.simulation.mode == crate::simulation::SimulationMode::PerFrame {
                display_delay
            } else {
                self.state
                    .simulation
                    .service_interval(crate::simulation::DEFAULT_MAX_STEPS_PER_UPDATE)
                    .saturating_sub(self.last_tick_plan.elapsed())
            };
            let next = tick_delay.min(display_delay);
            Some(service.map_or(next, |s| s.min(next)))
        } else if navigating || (self.visible && self.painting_input(ctx)) {
            let active_input = Duration::from_millis(if navigating { 16 } else { 50 });
            Some(service.map_or(active_input, |s| s.min(active_input)))
        } else {
            service
        };
        self.requested_delay = crate::simulation::repaint_after(
            normal_running,
            self.visible && self.clock.pending_exact_steps() > 0,
            self.state.simulation,
            service,
            save_deadline,
        );
        if let Some(delay) = self.requested_delay {
            ctx.request_repaint_after(delay);
        }
        if ctx.input(|i| i.raw.events.iter().any(settles_controls)) {
            // Reflect control changes after this UI pass, then return to idle.
            ctx.request_repaint();
        }
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let state = self.persistent_state();
        #[cfg(not(target_arch = "wasm32"))]
        eframe::set_value(storage, "primordial-soup.v1", &state);
        #[cfg(target_arch = "wasm32")]
        {
            let _ = storage;
            if let Ok(text) = serde_json::to_string(&state) {
                crate::web::save_state(&text, self.notices.clone(), self.context.clone(), false);
            }
        }
    }
    fn auto_save_interval(&self) -> Duration {
        Duration::from_secs(10)
    }
}

fn app_visible(ctx: &egui::Context) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ctx;
        web_sys::window()
            .and_then(|w| w.document())
            .is_some_and(|d| !d.hidden())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        ctx.input(|i| i.viewport().visible().unwrap_or(true))
    }
}

fn random_seed() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Math::random()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static STATE: AtomicU64 = AtomicU64::new(0);
        let _ = STATE.compare_exchange(
            0,
            web_time::SystemTime::now()
                .duration_since(web_time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        let mut z = STATE.fetch_add(0x9e3779b97f4a7c15, Ordering::Relaxed);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn decode_import_bytes(bytes: &[u8]) -> Result<String> {
    if bytes.starts_with(b"PK\x03\x04") {
        crate::export::export_saves(&crate::export::import_zip(bytes)?)
    } else {
        Ok(std::str::from_utf8(bytes)?.to_owned())
    }
}
fn rule_generated(project: &Project) -> bool {
    project.config().generated()
}
fn url_encode(text: &str) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
fn sanitize_preferences(p: &mut Preferences) {
    p.cells.sanitize();
    let d = Preferences::default();
    if !p.world_size.is_finite() || p.world_size < 0.05 || p.world_size > 27.0 {
        p.world_size = d.world_size;
    }
    if !p.canvas_aspect.is_finite() || p.canvas_aspect < 0.1 || p.canvas_aspect > 10.0 {
        p.canvas_aspect = 1.0;
    }
    p.physics_steps = p.physics_steps.clamp(1, 120);
    p.motion_blur_samples = p.motion_blur_samples.clamp(1, 32);
    p.brush_mode = p.brush_mode.min(3);
}
fn safe_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
        .take(100)
        .collect::<String>()
        .trim()
        .to_owned()
}
fn app_share_base_url() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        share_base_url()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("PRIMORDIAL_SOUP_URL").unwrap_or_else(|_| "http://localhost:8080/".into())
    }
}
fn share_base_url() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .map(|w| {
                format!(
                    "{}{}",
                    w.location().origin().unwrap_or_default(),
                    w.location().pathname().unwrap_or_default()
                )
            })
            .unwrap_or_default()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        "https://fluoddity.com/".to_owned()
    }
}
/// Stage colours for the GPU budget bar, in Stage::NAMES order.
const STAGE_COLORS: [Color32; 8] = [
    Color32::from_rgb(110, 160, 230),
    Color32::from_rgb(90, 200, 200),
    Color32::from_rgb(170, 130, 230),
    Color32::from_rgb(240, 200, 110),
    Color32::from_rgb(200, 200, 200),
    Color32::from_rgb(120, 210, 140),
    Color32::from_rgb(235, 140, 120),
    Color32::from_rgb(230, 130, 190),
];
/// About 8 px: the band colour while live, grey while measuring, a grey ring while frozen.
fn status_dot(ui: &mut egui::Ui, state: crate::performance::DotState) {
    use crate::performance::DotState;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
    let center = rect.center();
    let painter = ui.painter();
    match state {
        DotState::Live(band) => {
            let [r, g, b] = band.label_color();
            painter.circle_filled(center, 4.0, Color32::from_rgb(r, g, b));
        }
        DotState::Measuring => {
            painter.circle_filled(center, 4.0, Color32::from_gray(140));
        }
        DotState::Idle => {
            painter.circle_stroke(
                center,
                3.25,
                egui::Stroke::new(1.5, Color32::from_gray(150)),
            );
        }
    }
}
fn download(name: &str, bytes: &[u8], mime: &str) -> Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        crate::web::download(name, bytes, mime)?;
        Ok(format!("Downloaded {name}"))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = mime;
        let path = std::path::PathBuf::from("exports");
        std::fs::create_dir_all(&path)?;
        let path = path.join(format!(
            "{}-{}",
            web_time::SystemTime::now()
                .duration_since(web_time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            name
        ));
        std::fs::write(&path, bytes)?;
        Ok(format!("Saved {}", path.display()))
    }
}

#[cfg(test)]
mod preview_persistence_tests {
    use super::*;
    #[derive(Default)]
    struct MemoryStorage(BTreeMap<String, String>);
    impl eframe::Storage for MemoryStorage {
        fn get_string(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.0.insert(key.into(), value);
        }
        fn remove_string(&mut self, key: &str) {
            self.0.remove(key);
        }
        fn flush(&mut self) {}
    }
    #[test]
    fn fresh_state_uses_the_captured_startup_project_but_saved_projects_survive() {
        let fresh: SavedState = serde_json::from_str("{}").unwrap();
        let captured = Project::from_json(include_str!("../assets/default-project.json")).unwrap();
        assert_eq!(fresh.project, captured);
        assert_eq!(fresh.project.name, "Diversity");
        assert_eq!(fresh.prefs.cells.style, crate::cells::Style::Cells);
        assert_eq!(fresh.prefs.world_size, 1.0);
        assert_eq!(fresh.prefs.physics_steps, 5);
        assert_eq!(fresh.project.config().mutation_scale, 0.2);
        assert!(!fresh.auto_calibrate);

        let mut saved = SavedState {
            project: Project::default(),
            prefs: Preferences {
                brightness: 2.7,
                physics_steps: 17,
                ..Default::default()
            },
            auto_calibrate: true,
            ..Default::default()
        };
        saved.project.config_mut().mutation_scale = 0.45;
        let restored: SavedState =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(restored.project, saved.project);
        assert_eq!(restored.prefs, saved.prefs);
        assert!(restored.auto_calibrate);
    }
    #[test]
    fn complete_saved_state_roundtrips_eframe_ron_and_migrates_old_cells() {
        use crate::cells::{Style, Theme};
        let mut storage = MemoryStorage::default();
        for theme in [Theme::Dark, Theme::Light] {
            for palette in [false, true] {
                let mut state = SavedState::default();
                state.prefs.cells.style = Style::Cells;
                state.prefs.cells.theme = theme;
                state.prefs.cells.reference_palette = palette;
                state.prefs.cells.structure_threshold = 1.75;
                state.prefs.cells.membranes = 0.6;
                state
                    .project
                    .config_mut()
                    .switch_mode(crate::network::Mode::Gru, 0.25);
                state.saves.insert("Kept".into(), state.project.clone());
                eframe::set_value(&mut storage, "state", &state);
                let loaded: SavedState =
                    eframe::get_value(&storage, "state").expect("SavedState RON reload");
                assert_eq!(loaded.prefs, state.prefs);
                assert_eq!(loaded.project, state.project);
                assert_eq!(loaded.saves, state.saves);
                // Reverting the microscope option must not discard saved projects.
                let encoded = storage.0["state"].clone();
                let experimental = encoded.replacen("cells:(", "cells:(microscope:true,", 1);
                assert_ne!(experimental, encoded);
                storage.0.insert("state".into(), experimental);
                let restored: SavedState =
                    eframe::get_value(&storage, "state").expect("Reverted Cells RON reload");
                assert_eq!(restored.prefs, state.prefs);
                assert_eq!(restored.project, state.project);
                assert_eq!(restored.saves, state.saves);
                eframe::set_value(&mut storage, "state", &restored);
                assert!(!storage.0["state"].contains("microscope"));
            }
        }
        storage.0.insert(
            "state".into(),
            "(prefs:(cells:(style:cells,reference_palette:true,scale:2.0,wash:0.6)))".into(),
        );
        let loaded: SavedState = eframe::get_value(&storage, "state").expect("Legacy RON reload");
        assert_eq!(loaded.prefs.cells.theme, Theme::Dark);
        assert!(!loaded.prefs.cells.reference_palette);
        assert_eq!(loaded.prefs.cells.scale, 2.0);
        assert_eq!(loaded.prefs.cells.wash, 0.6);
        assert_eq!(loaded.prefs.cells.structure_threshold, 1.02);
        storage
            .0
            .insert("state".into(), "(prefs:(cells:(theme:dark,typo:1)))".into());
        assert!(eframe::get_value::<SavedState>(&storage, "state").is_none());
    }
    #[test]
    fn autosave_uses_original_until_preview_is_explicitly_loaded() {
        let original = Project::default();
        let mut hovered = original.clone();
        hovered
            .config_mut()
            .switch_mode(crate::network::Mode::Gru, 0.25);
        let mut state = SavedState::default();
        state.project = hovered.clone();
        state
            .saves
            .insert("Saved candidate".into(), hovered.clone());
        let saved = state.for_persistence(Some(&original));
        assert_eq!(saved.project, original);
        assert_eq!(saved.saves["Saved candidate"], hovered);
        assert_eq!(state.project, hovered);
        assert_eq!(state.for_persistence(None).project, hovered);
    }
}

#[cfg(test)]
mod pacing_tests {
    use super::*;

    fn key(reticle: [f32; 2], radius: f32, highlight: f32, prefs: &Preferences) -> PresentationKey {
        PresentationKey::new(
            Overlays {
                reticle,
                radius,
                style: 2,
                angle: prefs.draw_angle,
                ..Default::default()
            },
            highlight,
            prefs,
        )
    }

    #[test]
    fn select_hover_does_not_change_the_presented_image() {
        let prefs = Preferences::default();
        // Select hides the reticle, so pointer motion leaves the key unchanged.
        assert_eq!(
            key([0.1, 0.2], 0.0, -1.0, &prefs),
            key([0.8, 0.6], 0.0, -1.0, &prefs)
        );
        // A drawing tool's reticle follows the pointer.
        assert_ne!(
            key([0.1, 0.2], 0.01, -1.0, &prefs),
            key([0.8, 0.6], 0.01, -1.0, &prefs)
        );
    }

    #[test]
    fn highlight_and_tone_changes_present() {
        let prefs = Preferences::default();
        let base = key([0.0; 2], 0.0, -1.0, &prefs);
        assert_ne!(base, key([0.0; 2], 0.0, 2.0, &prefs));
        let mut brighter = prefs.clone();
        brighter.brightness += 0.5;
        assert_ne!(base, key([0.0; 2], 0.0, -1.0, &brighter));
        let mut no_bloom = prefs.clone();
        no_bloom.bloom_enabled = false;
        assert_ne!(base, key([0.0; 2], 0.0, -1.0, &no_bloom));
    }

    #[test]
    fn deadlines_tolerate_vsync_jitter() {
        let interval = Duration::from_secs_f64(1.0 / 60.0);
        assert!(deadline_reached(interval, interval));
        // A 60 Hz display period a fraction shorter than the interval still presents.
        assert!(deadline_reached(Duration::from_micros(16_500), interval));
        assert!(deadline_reached(Duration::from_micros(15_800), interval));
        // Half a 120 Hz frame early is not a frame.
        assert!(!deadline_reached(Duration::from_micros(8_333), interval));
        assert!(!deadline_reached(Duration::from_micros(14_000), interval));
        // A 55 fps setting waits past the first 60 Hz vsync instead of presenting at 60.
        let interval_55 = Duration::from_secs_f64(1.0 / 55.0);
        assert!(!deadline_reached(
            Duration::from_micros(16_667),
            interval_55
        ));
    }

    #[test]
    fn presentation_cadence_is_independent_of_encoding_cost() {
        let origin = Instant::now();
        let interval = Duration::from_secs_f64(1.0 / 60.0);
        let mut clock = PresentationClock::new(origin);
        let mut end_anchored = origin;
        let mut old_presentations = 0;
        for frame in 1..=18 {
            let start = origin + interval * frame;
            let cost = Duration::from_micros([500, 4_000, 8_000][frame as usize % 3]);
            assert!(clock.due(start, interval));
            clock.consume(start, interval);
            assert_eq!(clock.remaining(start + cost, interval), interval - cost);
            if deadline_reached(start.saturating_duration_since(end_anchored), interval) {
                old_presentations += 1;
                end_anchored = start + cost;
            }
        }
        // The former end-of-encoding anchor drops refreshes despite spare GPU time.
        assert!(old_presentations < 18);
    }

    #[test]
    fn presentation_clock_preserves_limits_and_newer_transition_epochs() {
        let origin = Instant::now();
        let interval = Duration::from_secs_f64(1.0 / 60.0);
        let mut clock = PresentationClock::new(origin);
        let mut presentations = 0;
        for callback in 1..=12 {
            let start = origin + Duration::from_nanos(callback * 1_000_000_000 / 120);
            if clock.due(start, interval) {
                presentations += 1;
                clock.consume(start, interval);
            }
        }
        assert_eq!(presentations, 6);
        let clock = PresentationClock::new(origin);
        assert!(!clock.due(origin + interval, Duration::from_secs_f64(1.0 / 55.0)));
        assert!(!clock.due(
            origin + Duration::from_millis(150),
            Duration::from_millis(200)
        ));
        let transition = origin + Duration::from_millis(20);
        let mut clock = PresentationClock::new(transition);
        clock.consume(origin, interval);
        assert!(!clock.due(origin, interval));
        assert_eq!(
            clock.remaining(origin, interval),
            interval + Duration::from_millis(20)
        );
        assert_eq!(clock.remaining(transition, interval), interval);
    }

    #[test]
    fn presentation_phase_survives_callback_jitter_and_early_input() {
        let origin = Instant::now();
        let interval = Duration::from_secs_f64(1.0 / 60.0);
        let mut clock = PresentationClock::new(origin);
        clock.configure(origin, interval, true);
        for frame in 1..=120 {
            // Alternating 18 and 15.33 ms callback gaps keep one update per refresh.
            let jitter = if frame % 2 == 1 {
                Duration::from_micros(1333)
            } else {
                Duration::ZERO
            };
            let start = origin + interval * frame + jitter;
            assert!(clock.due(start, interval));
            clock.consume(start, interval);
            let early_input = start + Duration::from_millis(2);
            let before = clock.last;
            clock.consume(early_input, interval);
            assert_eq!(clock.last, before);
        }
    }

    #[test]
    fn presentation_phase_obeys_caps_and_rebases_transitions() {
        let origin = Instant::now();
        let interval = Duration::from_secs_f64(1.0 / 55.0);
        let mut clock = PresentationClock::new(origin);
        clock.configure(origin, interval, true);
        let mut updates = 0;
        for frame in 1..=600 {
            let start = origin + Duration::from_nanos(frame * 1_000_000_000 / 60);
            if clock.due(start, interval) {
                clock.consume(start, interval);
                updates += 1;
            }
        }
        assert_eq!(updates, 550);
        let resumed = origin + Duration::from_secs(20);
        clock.consume(resumed, interval);
        assert!(!clock.due(resumed, interval));
        assert_eq!(clock.remaining(resumed, interval), interval);
        clock.suspend();
        clock.configure(resumed + Duration::from_secs(1), interval, true);
        assert!(!clock.due(resumed + Duration::from_secs(1), interval));
        let changed = resumed + Duration::from_secs(2);
        let slow = Duration::from_secs_f64(1.0 / 10.0);
        clock.configure(changed, slow, true);
        assert!(!clock.due(changed, slow));
        assert_eq!(clock.remaining(changed, slow), slow);
    }

    #[test]
    fn pointer_motion_needs_no_settling_pass() {
        assert!(!settles_controls(&egui::Event::PointerMoved(Pos2::new(
            3.0, 4.0
        ))));
        assert!(!settles_controls(&egui::Event::MouseMoved(egui::vec2(
            1.0, 0.0
        ))));
        assert!(settles_controls(&egui::Event::PointerButton {
            pos: Pos2::ZERO,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        }));
        assert!(settles_controls(&egui::Event::PointerGone));
        assert!(settles_controls(&egui::Event::Text("a".into())));
    }
}
