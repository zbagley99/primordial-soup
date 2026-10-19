//! Offline, backpressured video export. Recording arithmetic ports Fluoddity
//! (MIT, Jesse Gelders, 2026); browser muxing uses local MPL-2.0 mediabunny.
#[cfg(not(target_arch = "wasm32"))]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
use eframe::egui;
use std::sync::{Arc, Mutex};
pub const MIN_RECORDING_DIM: u32 = 128;
pub const FULL_FRAME: u32 = 1 << 20;
pub const RECORDING_FPS: u32 = 60;
pub const QUALITY_LABELS: [&str; 5] = [
    "Very Low (smallest file)",
    "Low",
    "Medium",
    "High (default)",
    "Very High (largest file)",
];
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RecordingSettings {
    pub resolution: [u32; 2],
    pub duration: u32,
    pub quality: u32,
    pub physics_steps: u32,
    pub motion_blur_samples: u32,
}
impl Default for RecordingSettings {
    fn default() -> Self {
        Self {
            resolution: [512; 2],
            duration: 5,
            quality: 3,
            physics_steps: 60,
            motion_blur_samples: 1,
        }
    }
}
impl RecordingSettings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=120).contains(&self.duration),
            "duration must be1..=120 seconds"
        );
        ensure!(self.quality < 5, "unknown recording quality");
        ensure!(
            (1..=480).contains(&self.physics_steps),
            "recording physics steps must be1..=480"
        );
        ensure!(
            (1..=self.physics_steps).contains(&self.motion_blur_samples),
            "recording blur must be1..=physics steps"
        );
        ensure!(
            self.resolution
                .iter()
                .all(|v| (MIN_RECORDING_DIM..=FULL_FRAME).contains(v)),
            "invalid recording dimensions"
        );
        Ok(())
    }
    pub fn fps(&self) -> u32 {
        RECORDING_FPS
    }
    pub fn output_size(&self, world: [u32; 2]) -> Result<[u32; 2]> {
        self.validate()?;
        let size = std::array::from_fn(|i| {
            even_dim(if self.resolution[i] == FULL_FRAME {
                world[i]
            } else {
                self.resolution[i]
            })
        });
        ensure!(
            size.iter().all(|&n| n <= 4096),
            "Recording dimensions must be 4096px or less"
        );
        Ok(size)
    }
    pub fn frame_count(&self) -> u32 {
        self.duration.saturating_mul(self.fps()).max(1)
    }
    pub fn physics_frame_count(&self) -> u32 {
        self.frame_count() * self.physics_steps
    }
    pub fn set_physics_steps(&mut self, next: u32) {
        let next = next.clamp(1, 480);
        self.motion_blur_samples =
            rescale_samples(self.motion_blur_samples, self.physics_steps, next);
        self.physics_steps = next;
    }
}
pub fn even_dim(value: u32) -> u32 {
    (value - value % 2).max(MIN_RECORDING_DIM)
}
pub fn clamp_resolution(requested: [u32; 2], window: [u32; 2]) -> [u32; 2] {
    [
        even_dim(requested[0].min(window[0].max(MIN_RECORDING_DIM))),
        even_dim(requested[1].min(window[1].max(MIN_RECORDING_DIM))),
    ]
}
pub fn rescale_samples(samples: u32, previous: u32, next: u32) -> u32 {
    let ceiling = next.clamp(1, 480);
    if ceiling <= 1 {
        return 1;
    }
    let old = previous.clamp(1, 480);
    if old <= 1 {
        return ceiling;
    }
    let value = samples.clamp(1, old);
    (1 + (((value - 1) as f64 / (old - 1) as f64) * (ceiling - 1) as f64).floor() as u32)
        .clamp(1, ceiling)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}
pub fn crop_rect(requested: [u32; 2], window: [u32; 2]) -> CropRect {
    let s = clamp_resolution(requested, window);
    CropRect {
        x: ((window[0] as f64 - s[0] as f64) / 2.0).floor() as i32,
        y: ((window[1] as f64 - s[1] as f64) / 2.0).floor() as i32,
        width: s[0],
        height: s[1],
    }
}
pub fn is_full_frame(requested: [u32; 2], window: [u32; 2]) -> bool {
    let s = clamp_resolution(requested, window);
    window[0].saturating_sub(s[0]) <= 1 && window[1].saturating_sub(s[1]) <= 1
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriverAction {
    Encode,
    Suspend,
    Finalize,
}
pub fn driver_action(finished: bool, paused: bool) -> DriverAction {
    if finished {
        DriverAction::Finalize
    } else if paused {
        DriverAction::Suspend
    } else {
        DriverAction::Encode
    }
}
/// Returns true only when Start Export is clicked. The app owns cancellation/progress.
pub fn settings(record: &mut RecordingSettings, ui: &mut egui::Ui, world: [u32; 2]) -> bool {
    ui.label("Record the entire world, without editor controls. Pan, zoom and window resizing do not change the recording.");
    ui.label(format!(
        "Video (MP4 / WebM) · {} frames/second",
        record.fps()
    ));
    let max = 4096;
    let mut size = std::array::from_fn(|i| {
        even_dim(if record.resolution[i] == FULL_FRAME {
            world[i]
        } else {
            record.resolution[i]
        })
    });
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut size[0], 128..=max).text("Width"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut size[1], 128..=max).text("Height"))
        .changed();
    if changed {
        record.resolution = size;
    }
    if ui.button("World resolution").clicked() {
        record.resolution = [FULL_FRAME; 2];
    }
    ui.add(egui::Slider::new(&mut record.duration, 1..=120).text("Duration (seconds)"));
    egui::ComboBox::from_id_salt("record quality")
        .selected_text(QUALITY_LABELS[record.quality.min(4) as usize])
        .show_ui(ui, |ui| {
            for (i, label) in QUALITY_LABELS.iter().enumerate() {
                ui.selectable_value(&mut record.quality, i as u32, *label);
            }
        });
    // Preserve the pre-gesture blur position once on release, rather than
    // compounding floor during every intermediate Physics Rate drag event.
    let key = ui.make_persistent_id("record physics gesture");
    let old = ui.ctx().data_mut(|d| d.get_temp::<(u32, u32)>(key));
    let mut steps = record.physics_steps;
    let r = ui.add(egui::Slider::new(&mut steps, 1..=480).text("Physics Rate"));
    if r.changed() {
        let baseline = old.unwrap_or((record.physics_steps, record.motion_blur_samples));
        ui.ctx().data_mut(|d| d.insert_temp(key, baseline));
        record.physics_steps = steps;
        record.motion_blur_samples = record.motion_blur_samples.min(steps).max(1);
    }
    if !r.dragged() && !r.is_pointer_button_down_on() {
        if let Some((from, samples)) = ui.ctx().data_mut(|d| d.remove_temp::<(u32, u32)>(key)) {
            record.motion_blur_samples = rescale_samples(samples, from, record.physics_steps);
        }
    }
    ui.add(
        egui::Slider::new(&mut record.motion_blur_samples, 1..=record.physics_steps)
            .text("Motion Blur"),
    );
    ui.label(format!(
        "{} video frames ·{} physics steps",
        record.frame_count(),
        record.physics_frame_count()
    ));
    ui.small(
        "Physics Rate sets simulation steps per video frame. Motion Blur increases rendering work.",
    );
    ui.small("Encoding runs locally. Browser clips are limited to 60 MiB of compressed video and 64 MiB including the container; temporary memory can be higher.");
    let validation = record.output_size(world);
    if let Err(error) = &validation {
        ui.colored_label(egui::Color32::LIGHT_RED, error.to_string());
    }
    ui.add_enabled(validation.is_ok(), egui::Button::new("Start recording"))
        .clicked()
}

/// Fixed export resources exist only for one recording. Display resizing cannot invalidate them.
pub struct ExportSession {
    pub recorder: Recorder,
    pub render: crate::render::Renderer,
    pub camera: crate::coords::Camera,
    pub prefs: crate::config::Preferences,
    pub project: crate::config::Project,
    pub was_paused: bool,
}
impl std::ops::Deref for ExportSession {
    type Target = Recorder;
    fn deref(&self) -> &Recorder {
        &self.recorder
    }
}
impl std::ops::DerefMut for ExportSession {
    fn deref_mut(&mut self) -> &mut Recorder {
        &mut self.recorder
    }
}
impl ExportSession {
    pub fn allocated_bytes(&self) -> u64 {
        self.render.texture_bytes() + self.recorder.staging.size()
    }
    pub fn start(
        device: &wgpu::Device,
        engine: &crate::gpu::Engine,
        settings: &RecordingSettings,
        view: crate::coords::Camera,
        prefs: &crate::config::Preferences,
        project: &crate::config::Project,
        was_paused: bool,
    ) -> Result<Self> {
        let size = settings.output_size(engine.sizing.canvas)?;
        let mut render = crate::render::Renderer::new(device, engine, size)?;
        if prefs.cells.style == crate::cells::Style::Cells {
            render.commit_cells(render.prepare_cells(device, engine, &prefs.cells)?);
        }
        render.set_preferences(prefs);
        let recorder = Recorder::start(device, settings, size)?;
        Ok(Self {
            recorder,
            render,
            camera: crate::coords::Camera {
                trail_mode: view.trail_mode,
                ..Default::default()
            },
            prefs: prefs.clone(),
            project: project.clone(),
            was_paused,
        })
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if !self.finished {
            self.discard();
        }
    }
}
#[derive(Clone, Debug)]
pub struct Progress {
    pub frames_done: u32,
    pub frames_total: u32,
    pub busy: bool,
    pub finished: bool,
    pub output: Option<String>,
}
#[derive(Debug)]
enum Ack {
    Started,
    Frame,
    Finished(String),
}
type Mailbox = Arc<Mutex<Option<std::result::Result<Ack, String>>>>;
pub struct Recorder {
    pub settings: RecordingSettings,
    pub size: [u32; 2],
    staging: wgpu::Buffer,
    stride: u32,
    map_status: Arc<Mutex<Option<std::result::Result<(), String>>>>,
    mapping: bool,
    inflight: bool,
    starting: bool,
    finishing: bool,
    cancelled: bool,
    finished: bool,
    frames: u32,
    output: Option<String>,
    mailbox: Mailbox,
    backend: Backend,
}
impl Recorder {
    pub fn start(
        device: &wgpu::Device,
        settings: &RecordingSettings,
        window: [u32; 2],
    ) -> Result<Self> {
        settings.validate()?;
        let size = clamp_resolution(settings.resolution, window);
        ensure!(
            size.iter().all(|&n| n <= 4096),
            "Recording exceeds the 4096px limit"
        );
        ensure!(
            size[0] <= window[0] && size[1] <= window[1],
            "recording requires a window of at least128px on each axis"
        );
        let stride = (size[0] * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        ensure!(
            stride as u64 * size[1] as u64 <= device.limits().max_buffer_size,
            "recording readback exceeds this GPU buffer limit"
        );
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offline video RGBA readback"),
            size: stride as u64 * size[1] as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mailbox = Arc::new(Mutex::new(None));
        let backend = Backend::start(size, settings.quality, mailbox.clone())?;
        Ok(Self {
            settings: settings.clone(),
            size,
            staging,
            stride,
            map_status: Arc::new(Mutex::new(None)),
            mapping: false,
            inflight: false,
            starting: true,
            finishing: false,
            cancelled: false,
            finished: false,
            frames: 0,
            output: None,
            mailbox,
            backend,
        })
    }
    pub fn ready(&self) -> bool {
        !self.starting
            && !self.mapping
            && !self.inflight
            && !self.finishing
            && !self.cancelled
            && !self.finished
            && self.frames < self.settings.frame_count()
    }
    pub fn frames_done(&self) -> u32 {
        self.frames
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
    pub fn discard(&mut self) {
        if self.mapping {
            self.staging.unmap();
        }
        self.mapping = false;
        self.cancelled = true;
        self.inflight = false;
        self.starting = false;
        self.finishing = false;
        self.finished = true;
        self.backend.cancel();
    }
    pub fn progress(&self) -> Progress {
        Progress {
            frames_done: self.frames,
            frames_total: self.settings.frame_count(),
            busy: !self.ready() && !self.finished,
            finished: self.finished,
            output: self.output.clone(),
        }
    }
    /// Call after the renderer's queue submission. The copy submission remains
    /// ordered behind it. Never advance simulation unless ready() returned true.
    pub fn frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        window: [u32; 2],
        paused: bool,
    ) -> Result<bool> {
        if paused || !self.ready() {
            return Ok(false);
        }
        ensure!(
            window[0] >= self.size[0] && window[1] >= self.size[1],
            "window shrank below the fixed recording size; cancel or restore its dimensions"
        );
        ensure!(
            texture.usage().contains(wgpu::TextureUsages::COPY_SRC),
            "recording output texture requires COPY_SRC usage"
        );
        let dimensions = texture.size();
        ensure!(
            dimensions.width >= window[0] && dimensions.height >= window[1],
            "recording window exceeds output texture dimensions"
        );
        ensure!(
            texture.format() == wgpu::TextureFormat::Rgba8Unorm,
            "recording expects renderer RGBA8 output"
        );
        let rect = crop_rect(self.size, window);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offline recording copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: rect.x as u32,
                    y: rect.y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.stride),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        self.mapping = true;
        let status = self.map_status.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                *status.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
            });
        Ok(true)
    }
    /// Every error terminates this recorder before returning to the app.
    /// Its isolated mailbox may receive late callbacks, but they cannot make
    /// this instance ready or affect any later recording.
    pub fn pump(&mut self, device: &wgpu::Device) -> Result<Progress> {
        if self.finished {
            return Ok(self.progress());
        }
        match self.pump_inner(device) {
            Ok(progress) => Ok(progress),
            Err(error) => {
                if self.mapping {
                    self.staging.unmap();
                }
                self.mapping = false;
                self.cancelled = true;
                self.finishing = false;
                self.inflight = false;
                self.starting = false;
                self.finished = true;
                self.backend.cancel();
                Err(error)
            }
        }
    }
    fn pump_inner(&mut self, device: &wgpu::Device) -> Result<Progress> {
        #[cfg(not(target_arch = "wasm32"))]
        device.poll(wgpu::PollType::Poll)?;
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let ack = self.mailbox.lock().unwrap().take();
        if let Some(ack) = ack {
            match ack.map_err(anyhow::Error::msg)? {
                Ack::Started => self.starting = false,
                Ack::Frame => {
                    self.inflight = false;
                    self.frames += 1;
                }
                Ack::Finished(path) => {
                    self.finishing = false;
                    self.finished = true;
                    self.output = Some(path);
                }
            }
        }
        let mapped = self.map_status.lock().unwrap().take();
        if let Some(result) = mapped {
            if let Err(error) = result {
                self.mapping = false; // A failed mapping is already unmapped.
                return Err(anyhow::Error::msg(error));
            }
            let range = self
                .staging
                .slice(..)
                .get_mapped_range()
                .context("recording mapped readback range")?;
            let width_bytes = self.size[0] as usize * 4;
            let mut rgba = Vec::with_capacity(width_bytes * self.size[1] as usize);
            for row in range.chunks_exact(self.stride as usize) {
                rgba.extend_from_slice(&row[..width_bytes]);
            }
            drop(range);
            self.staging.unmap();
            self.mapping = false;
            self.backend.add_frame(rgba)?;
            self.inflight = true;
        }
        if !self.starting
            && !self.mapping
            && !self.inflight
            && !self.finishing
            && !self.finished
            && (self.cancelled || self.frames >= self.settings.frame_count())
        {
            self.backend.finish()?;
            self.finishing = true;
        }
        Ok(self.progress())
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum EncodeCommand {
    Frame(Vec<u8>),
    Finish,
}
#[cfg(not(target_arch = "wasm32"))]
static ENCODER_WORKER: Mutex<()> = Mutex::new(());
#[cfg(not(target_arch = "wasm32"))]
struct Backend {
    sender: std::sync::mpsc::SyncSender<EncodeCommand>,
    discarded: Arc<std::sync::atomic::AtomicBool>,
    child: Arc<Mutex<Option<std::process::Child>>>,
}
#[cfg(not(target_arch = "wasm32"))]
impl Backend {
    fn start(size: [u32; 2], quality: u32, mailbox: Mailbox) -> Result<Self> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let discarded = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let child = Arc::new(Mutex::new(None::<std::process::Child>));
        let cancelled = discarded.clone();
        let process = child.clone();
        std::thread::Builder::new().name("local recording encoder".into()).spawn(move||{
            use std::{io::Write,process::{Command,Stdio},time::{SystemTime,UNIX_EPOCH,Instant,Duration},sync::atomic::Ordering};
            #[cfg(target_os="linux")]
            {unsafe extern "C" {fn nice(increment:std::ffi::c_int)->std::ffi::c_int;} unsafe {nice(19);}}
            let _worker=ENCODER_WORKER.lock().unwrap();
            let mut output=String::new();let mut log=String::new();
            let result=(||->Result<String>{
                if cancelled.load(Ordering::Acquire){return Ok(String::new());}
                std::fs::create_dir_all("exports")?;
                let stamp=SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
                output=format!("exports/primordial-soup-{stamp}.mp4");
                let mut frames=0u32;
                    log=format!("exports/primordial-soup-{stamp}.encoder.log");
                    let stderr=std::fs::File::create(&log)?;
                    let cq=[35,29,23,18,14][quality.min(4) as usize].to_string();
                    let mut encoder=Command::new("nice").args(["-n","19","ffmpeg","-hide_banner","-loglevel","error","-nostdin","-f","rawvideo","-pixel_format","rgba","-video_size",&format!("{}x{}",size[0],size[1]),"-framerate","60","-i","pipe:0","-an","-c:v","h264_nvenc","-preset","p5","-cq",&cq,"-b:v","0","-pix_fmt","yuv420p","-threads","1","-movflags","+faststart",&output]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::from(stderr)).spawn().context("Native video needs ffmpeg with NVIDIA h264_nvenc; no CPU video fallback is used")?;
                    let mut input=encoder.stdin.take().context("ffmpeg stdin unavailable")?;
                    *process.lock().unwrap()=Some(encoder);
                    *mailbox.lock().unwrap()=Some(Ok(Ack::Started));
                    while let Ok(command)=receiver.recv(){
                        if cancelled.load(Ordering::Acquire){break;}
                        match command {EncodeCommand::Frame(rgba)=>{input.write_all(&rgba).context("NVIDIA video frame write failed")?;frames+=1;*mailbox.lock().unwrap()=Some(Ok(Ack::Frame));},EncodeCommand::Finish=>break}
                    }
                    drop(input);
                    let deadline=Instant::now()+Duration::from_secs(30);
                    let status=loop {
                        let status=process.lock().unwrap().as_mut().context("Video encoder process unavailable")?.try_wait()?;
                        if let Some(status)=status {break status;}
                        if Instant::now()>deadline {bail!("Video encoder did not finish within30 seconds");}
                        std::thread::sleep(Duration::from_millis(10));
                    };
                    process.lock().unwrap().take();
                    if !status.success()&&!cancelled.load(Ordering::Acquire){bail!("NVIDIA encoding failed: {}",std::fs::read_to_string(&log).unwrap_or_default());}
                if frames==0||cancelled.load(Ordering::Acquire) {let _=std::fs::remove_file(&output);Ok(String::new())} else {Ok(output.clone())}
            })();
            // The worker alone reaps the child; cancel can kill it without waiting on a pipe or wait().
            let child_to_reap={process.lock().unwrap().take()};
            if let Some(mut child)=child_to_reap{let _=child.kill();let _=child.wait();}
            match result {
                Ok(path)=>{let _=std::fs::remove_file(&log);*mailbox.lock().unwrap()=Some(Ok(Ack::Finished(path)));}
                Err(error)=>{let _=std::fs::remove_file(&output);*mailbox.lock().unwrap()=Some(Err(format!("Recording: {error:#}")));}
            }
        }).context("Could not launch recording encoder")?;
        Ok(Self {
            sender,
            discarded,
            child,
        })
    }
    fn add_frame(&mut self, rgba: Vec<u8>) -> Result<()> {
        self.sender
            .try_send(EncodeCommand::Frame(rgba))
            .map_err(|e| anyhow::anyhow!("video writer unavailable or busy: {e}"))
    }
    fn cancel(&mut self) {
        self.discarded
            .store(true, std::sync::atomic::Ordering::Release);
        if let Some(child) = self.child.lock().unwrap().as_mut() {
            let _ = child.kill();
        }
        let _ = self.sender.try_send(EncodeCommand::Finish);
    }
    fn finish(&mut self) -> Result<()> {
        self.sender
            .try_send(EncodeCommand::Finish)
            .map_err(|e| anyhow::anyhow!("cannot finalize video: {e}"))
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for Backend {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(module = "/web/video.js")]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name=startVideo)]
    fn start_video(width: u32, height: u32, quality: u32) -> js_sys::Promise;
    #[wasm_bindgen::prelude::wasm_bindgen(js_name=addVideoFrame)]
    fn add_video_frame(session: u32, rgba: js_sys::Uint8Array) -> js_sys::Promise;
    #[wasm_bindgen::prelude::wasm_bindgen(js_name=finishVideo)]
    fn finish_video(session: u32) -> js_sys::Promise;
    #[wasm_bindgen::prelude::wasm_bindgen(js_name=cancelVideo)]
    fn cancel_video(session: u32) -> js_sys::Promise;
}
// Retain startup/finalization ownership even when the Recorder itself is kept
// alive after discard. Shared state is small and independently testable.
#[cfg(any(target_arch = "wasm32", test))]
#[derive(Default)]
struct BrowserSession {
    id: Option<u32>,
    cancelled: bool,
}
#[cfg(any(target_arch = "wasm32", test))]
impl BrowserSession {
    fn started(&mut self, id: u32) -> bool {
        if self.cancelled {
            return false;
        }
        self.id = Some(id);
        true
    }
    fn cancel(&mut self) -> Option<u32> {
        self.cancelled = true;
        self.id.take()
    }
    fn settled(&mut self, id: u32) {
        if self.id == Some(id) {
            self.id = None;
        }
    }
}
#[cfg(target_arch = "wasm32")]
struct Backend {
    session: Arc<Mutex<BrowserSession>>,
    mailbox: Mailbox,
}
#[cfg(target_arch = "wasm32")]
impl Backend {
    fn start(size: [u32; 2], quality: u32, mailbox: Mailbox) -> Result<Self> {
        let session = Arc::new(Mutex::new(BrowserSession::default()));
        let shared = session.clone();
        let done = mailbox.clone();
        let promise = start_video(size[0], size[1], quality);
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_bindgen_futures::JsFuture::from(promise).await;
            let ack = match result {
                Ok(id) => {
                    if let Some(id) = id.as_f64() {
                        let accepted = shared.lock().unwrap().started(id as u32);
                        if !accepted {
                            let _ =
                                wasm_bindgen_futures::JsFuture::from(cancel_video(id as u32)).await;
                        }
                        Ok(Ack::Started)
                    } else {
                        Err("browser video encoder returned an invalid session".into())
                    }
                }
                Err(e) => Err(format!("Browser video encoder: {e:?}")),
            };
            *done.lock().unwrap() = Some(ack);
        });
        Ok(Self { session, mailbox })
    }
    fn add_frame(&mut self, rgba: Vec<u8>) -> Result<()> {
        let state = self.session.lock().unwrap();
        ensure!(!state.cancelled, "Recording was discarded");
        let id = state.id.context("browser video encoder is not ready")?;
        drop(state);
        let bytes = js_sys::Uint8Array::from(rgba.as_slice());
        let promise = add_video_frame(id, bytes);
        let mailbox = self.mailbox.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_bindgen_futures::JsFuture::from(promise).await;
            *mailbox.lock().unwrap() = Some(
                result
                    .map(|_| Ack::Frame)
                    .map_err(|e| format!("Browser video frame: {e:?}")),
            );
        });
        Ok(())
    }
    fn cancel(&mut self) {
        if let Some(id) = self.session.lock().unwrap().cancel() {
            let promise = cancel_video(id);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
            });
        }
    }
    fn finish(&mut self) -> Result<()> {
        let id = self
            .session
            .lock()
            .unwrap()
            .id
            .context("browser video encoder is not ready")?;
        let promise = finish_video(id);
        let mailbox = self.mailbox.clone();
        let session = self.session.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let result = wasm_bindgen_futures::JsFuture::from(promise).await;
            let mut state = session.lock().unwrap();
            state.settled(id);
            if state.cancelled {
                return;
            }
            drop(state);
            *mailbox.lock().unwrap() = Some(
                result
                    .map(|v| Ack::Finished(v.as_string().unwrap_or_default()))
                    .map_err(|e| format!("Browser video finalization: {e:?}")),
            );
        });
        Ok(())
    }
}
#[cfg(target_arch = "wasm32")]
impl Drop for Backend {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::BrowserSession;
    #[test]
    fn delayed_start_after_discard_cannot_install_a_session_even_if_retained() {
        let mut session = BrowserSession::default();
        assert_eq!(session.cancel(), None);
        assert!(!session.started(42)); // Caller must release the late JS encoder.
        assert_eq!(session.id, None);
    }
    #[test]
    fn finalization_retains_a_cancellable_session_until_it_settles() {
        let mut session = BrowserSession::default();
        assert!(session.started(42));
        assert_eq!(session.id, Some(42));
        assert_eq!(session.cancel(), Some(42));
        session.settled(42);
        assert!(session.cancelled);
        assert_eq!(session.id, None);
        let mut replacement = BrowserSession::default();
        assert!(replacement.started(43));
        session.settled(42);
        assert_eq!(replacement.id, Some(43));
    }
}
