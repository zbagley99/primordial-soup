//! Bounded native-window probe of the production SoupApp event/render loop.
//! Run GPU probes serially under /tmp/gpu-timing.lock.
use anyhow::{Context, Result, bail, ensure};
use eframe::{App, egui};
use primordial_soup::{
    app::{SoupApp, StartOptions},
    coords::Camera,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

struct Options {
    output: PathBuf,
    rates: Vec<u32>,
    zooms: Vec<f32>,
    warmup: Duration,
    measure: Duration,
    warmup_frames: u64,
    frames: u64,
    wall_mode: bool,
    fullscreen: bool,
    size: [u32; 2],
    pan: [f32; 2],
}
impl Options {
    fn parse() -> Result<Self> {
        let mut opts = Self {
            output: "artifacts/local/native-cadence/report.json".into(),
            rates: vec![1, 15, 30, 60],
            zooms: vec![1.0, 8.0],
            warmup: Duration::from_secs(2),
            measure: Duration::from_secs(5),
            warmup_frames: 60,
            frames: 300,
            wall_mode: false,
            fullscreen: false,
            size: [1280, 800],
            pan: [0.0; 2],
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" {
                println!(
                    "cadence-check [--output FILE] [--rates 1,15,30,60] [--zooms 1,8] [--warmup-frames 60] [--frames 300] [--warmup-seconds 2 --seconds 5] [--width 1280] [--height 800] [--fullscreen] [--pan-x 0] [--pan-y 0]\nRuns a visible, isolated production Cells app with normal vsync and physical input filtered. World size 1; phases share an evolving simulation. UI intervals are not hardware presentation timestamps."
                );
                std::process::exit(0);
            }
            if arg == "--fullscreen" {
                opts.fullscreen = true;
                continue;
            }
            let value = args
                .next()
                .with_context(|| format!("missing value for {arg}"))?;
            match arg.as_str() {
                "--output" => opts.output = value.into(),
                "--rates" => {
                    opts.rates = value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--zooms" => {
                    opts.zooms = value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?
                }
                "--warmup-frames" => opts.warmup_frames = value.parse()?,
                "--frames" => opts.frames = value.parse()?,
                "--warmup-seconds" | "--seconds" => {
                    opts.wall_mode = true;
                    let seconds: f64 = value.parse()?;
                    ensure!(
                        seconds.is_finite() && seconds > 0.0 && seconds <= 600.0,
                        "duration must be in (0, 600] seconds"
                    );
                    if arg == "--seconds" {
                        opts.measure = Duration::from_secs_f64(seconds);
                    } else {
                        opts.warmup = Duration::from_secs_f64(seconds);
                    }
                }
                "--width" => opts.size[0] = value.parse()?,
                "--height" => opts.size[1] = value.parse()?,
                "--pan-x" => opts.pan[0] = value.parse()?,
                "--pan-y" => opts.pan[1] = value.parse()?,
                _ => bail!("unknown option {arg}"),
            }
        }
        ensure!(
            opts.frames > 0 && opts.frames <= 10000 && opts.warmup_frames <= 10000,
            "invalid frame counts"
        );
        ensure!(
            !opts.rates.is_empty()
                && opts.rates.len() <= 16
                && opts.rates.iter().all(|r| (1..=60).contains(r)),
            "rates must contain 1 to 16 values in 1..=60"
        );
        ensure!(
            !opts.zooms.is_empty()
                && opts.zooms.len() <= 16
                && opts
                    .zooms
                    .iter()
                    .all(|z| z.is_finite() && (0.1..=100.0).contains(z)),
            "zooms must contain 1 to 16 values in 0.1..=100"
        );
        ensure!(
            opts.size.iter().all(|s| (400..=8192).contains(s))
                && opts.pan.iter().all(|p| p.is_finite()),
            "invalid viewport or pan"
        );
        Ok(opts)
    }
}

fn distribution(values: impl Iterator<Item = f64>) -> Value {
    let mut values: Vec<_> = values.filter(|v| v.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Value::Null;
    }
    let percentile = |p: f64| {
        values[((values.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1)]
    };
    json!({"count":values.len(), "mean":values.iter().sum::<f64>() / values.len() as f64,
        "p50":percentile(0.5), "p95":percentile(0.95), "p99":percentile(0.99),
        "max":values.last(), "over_20ms":values.iter().filter(|v| **v > 20.0).count()})
}

struct Probe {
    app: SoupApp,
    opts: Options,
    cases: Vec<(u32, f32)>,
    phase: usize,
    phase_start: Instant,
    previous_ui: Option<Instant>,
    measurement_start: Option<Value>,
    samples: Vec<Value>,
    results: Vec<Value>,
    pending_phase: bool,
    phase_presentations: Option<u64>,
    measured_at: Option<Instant>,
    transition: Vec<Value>,
    failure: Option<String>,
    outcome: Arc<Mutex<Option<std::result::Result<(), String>>>>,
}
impl Probe {
    fn set_phase(&mut self) {
        let (rate, zoom) = self.cases[self.phase];
        self.app.set_measurement_view(
            rate,
            Camera {
                pan: self.opts.pan,
                zoom,
                trail_mode: false,
            },
        );
        self.phase_start = Instant::now();
        self.previous_ui = None;
        self.measurement_start = None;
        self.pending_phase = false;
        self.phase_presentations = None;
        self.measured_at = None;
        self.transition.clear();
        eprintln!(
            "cadence phase {}/{}: physics {rate}, zoom {zoom}",
            self.phase + 1,
            self.cases.len()
        );
    }
    fn finish_phase(&mut self, end: Value) {
        let (rate, zoom) = self.cases[self.phase];
        let field = |pointer: &str| {
            distribution(
                self.samples
                    .iter()
                    .filter_map(|s| s.pointer(pointer).and_then(Value::as_f64)),
            )
        };
        let intervals = field("/interval_ms");
        let delta = |key: &str| {
            end.pointer(key)
                .and_then(Value::as_u64)
                .zip(
                    self.measurement_start
                        .as_ref()
                        .and_then(|r| r.pointer(key))
                        .and_then(Value::as_u64),
                )
                .map(|(end, start)| end.saturating_sub(start))
        };
        let measured_seconds =
            self.previous_ui
                .zip(self.measured_at)
                .map(|(last_callback, start)| {
                    last_callback.saturating_duration_since(start).as_secs_f64()
                });
        let mut last_scene_count = self
            .measurement_start
            .as_ref()
            .and_then(|r| r.pointer("/workload/presentations"))
            .and_then(Value::as_u64);
        let mut last_scene_ms = if self.opts.wall_mode {
            None
        } else {
            self.measured_at
                .map(|start| start.duration_since(self.phase_start).as_secs_f64() * 1000.0)
        };
        let mut scene_intervals = Vec::new();
        for sample in &self.samples {
            let Some(count) = sample
                .pointer("/performance/workload/presentations")
                .and_then(Value::as_u64)
            else {
                continue;
            };
            let Some(elapsed) = sample["elapsed_ms"].as_f64() else {
                continue;
            };
            if last_scene_count.is_some_and(|previous| count > previous) {
                if let Some(previous) = last_scene_ms {
                    scene_intervals.push(elapsed - previous);
                }
                last_scene_ms = Some(elapsed);
            }
            last_scene_count = Some(count);
        }
        eprintln!(
            "cadence physics {rate}, zoom {zoom}: {} UI frames, interval p50={} p95={} ms",
            self.samples.len(),
            intervals["p50"],
            intervals["p95"]
        );
        self.results.push(json!({"physics_steps":rate, "zoom":zoom, "pan":self.opts.pan,
            "measured_seconds":measured_seconds,
            "ui_callbacks":self.samples.len(), "ui_interval_ms":intervals,
            "scene_presentations":delta("/workload/presentations"),
            "scene_update_interval_ms":distribution(scene_intervals.into_iter()),
            "window_app_presentations_per_second":delta("/workload/presentations").zip(measured_seconds).map(|(count, seconds)| count as f64 / seconds),
            "actual_working_extent_pixels":end.pointer("/workload/render_size"),
            "submissions":delta("/workload/submissions"), "completed_ticks":delta("/workload/completed_ticks"),
            "ui_callback_duration_ms":field("/callback_ms"),
            "cpu_encode_ms":field("/performance/cpu/encode_ms"), "cpu_submit_ms":field("/performance/cpu/submit_ms"),
            "performance_start":self.measurement_start, "performance_end":end,
            "transition_first_callbacks":self.transition, "samples":self.samples}));
        self.samples.clear();
    }
    fn write_report(&self) -> Result<()> {
        let report = json!({"kind":"native-cadence-check", "version":1,
            "provenance":{"package_version":env!("CARGO_PKG_VERSION"),
                "executable":std::env::current_exe().ok(), "arguments":std::env::args().collect::<Vec<_>>(),
                "unix_time_ms":SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
                "production_app":true, "present_mode":"production wgpu_options default (AutoVsync)",
                "interval_method":"Instant between first-pass SoupApp::ui callbacks; not hardware presentation timestamps",
                "scene_update_interval_method":"UI callback start intervals when workload.presentations increases; app scene updates, not physical surface presents",
                "measurement_boundary_note":"Elapsed measurement time ends at the last recorded UI callback start, excluding report work. Frame-count mode anchors the first scene interval at the warmup scene update; wall-time mode omits that interval because its initial callback may not update the scene.",
                "input_filter":"raw_input_hook clears events (including modifier changes) and hovered/dropped files; viewport metadata, visibility, resize and native close remain intact",
                "gpu_method":"production asynchronous timing window; raw snapshots may repeat, not independent GPU samples",
                "simulation":"continuous evolving world across phases; no reset between zooms/rates",
                "persistence":"fresh isolated directory; ordinary user state is not loaded"},
            "requested_viewport_points":self.opts.size, "fullscreen":self.opts.fullscreen, "world_size":1,
            "measurement_mode":if self.opts.wall_mode {"wall seconds"} else {"scene presentations"},
            "warmup_seconds":self.opts.warmup.as_secs_f64(), "measure_seconds":self.opts.measure.as_secs_f64(),
            "warmup_presentations":self.opts.warmup_frames, "measured_presentations":self.opts.frames,
            "failure":self.failure, "phases":self.results});
        std::fs::write(&self.opts.output, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("Wrote {}", self.opts.output.display());
        Ok(())
    }
}
impl App for Probe {
    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        raw_input.events.clear();
        raw_input.hovered_files.clear();
        raw_input.dropped_files.clear();
    }
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.pending_phase {
            self.set_phase();
        }
        self.app.logic(ctx, frame);
    }
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if self.phase >= self.cases.len() {
            return;
        }
        let now = Instant::now();
        let first_pass = ui.ctx().current_pass_index() == 0;
        self.app.ui(ui, frame);
        if !first_pass {
            return;
        }
        let callback_ms = now.elapsed().as_secs_f64() * 1000.0;
        let previous = self.previous_ui.replace(now);
        let elapsed = now.duration_since(self.phase_start);
        let report = self.app.performance_report();
        let count = report
            .pointer("/workload/presentations")
            .and_then(Value::as_u64);
        if self.phase_presentations.is_none() {
            self.phase_presentations = count;
        }
        let sample = json!({"elapsed_ms":elapsed.as_secs_f64() * 1000.0,
            "interval_ms":previous.map(|p| now.duration_since(p).as_secs_f64() * 1000.0),
            "callback_ms":callback_ms, "viewport_visible":ui.ctx().input(|i| i.viewport().visible()),
            "window_inner_size_points":ui.ctx().input(|i| i.viewport().inner_rect.map(|r| [r.width(), r.height()])),
            "pixels_per_point":ui.ctx().pixels_per_point(),
            "performance":report});
        if self.transition.len() < 12 {
            self.transition.push(sample.clone());
        }
        if elapsed >= Duration::from_secs(60) || (!self.opts.wall_mode && count.is_none()) {
            self.failure = Some(if count.is_none() {
                "missing workload.presentations counter".into()
            } else {
                "phase exceeded 60 second timeout".into()
            });
            self.finish_phase(report);
            let result = self
                .write_report()
                .and_then(|_| bail!(self.failure.clone().unwrap()))
                .map_err(|e| format!("{e:#}"));
            *self.outcome.lock().expect("outcome lock") = Some(result);
            self.phase = self.cases.len();
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        let warm = if self.opts.wall_mode {
            elapsed >= self.opts.warmup
        } else {
            count
                .unwrap()
                .saturating_sub(self.phase_presentations.unwrap())
                >= self.opts.warmup_frames
        };
        if !warm {
            return;
        }
        if self.measurement_start.is_none() {
            self.measurement_start = Some(report);
            self.measured_at = Some(now);
            return;
        }
        self.samples.push(sample);
        let complete = if self.opts.wall_mode {
            self.measured_at.unwrap().elapsed() >= self.opts.measure
        } else {
            count.unwrap().saturating_sub(
                self.measurement_start
                    .as_ref()
                    .unwrap()
                    .pointer("/workload/presentations")
                    .and_then(Value::as_u64)
                    .unwrap(),
            ) >= self.opts.frames
        };
        if complete {
            self.finish_phase(self.app.performance_report());
            self.phase += 1;
            if self.phase == self.cases.len() {
                let result = self.write_report().map_err(|e| format!("{e:#}"));
                *self.outcome.lock().expect("outcome lock") = Some(result);
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.pending_phase = true;
            }
        }
    }
}

fn run() -> Result<()> {
    let opts = Options::parse()?;
    let parent = opts
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let persistence = parent.join(format!(
        "isolated-state-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir(&persistence)?;
    let size = opts.size;
    let fullscreen = opts.fullscreen;
    let cases: Vec<_> = opts
        .rates
        .iter()
        .flat_map(|r| opts.zooms.iter().map(move |z| (*r, *z)))
        .collect();
    let outcome = Arc::new(Mutex::new(None));
    let result = outcome.clone();
    eframe::run_native(
        "Primordial Soup cadence probe",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: primordial_soup::app::wgpu_options(),
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([size[0] as f32, size[1] as f32])
                .with_fullscreen(fullscreen),
            persistence_path: Some(persistence),
            persist_window: false,
            ..Default::default()
        },
        Box::new(move |cc| {
            let app = SoupApp::new(
                cc,
                StartOptions {
                    world_size: Some(1.0),
                    physics_steps: Some(cases[0].0),
                    no_panels: true,
                    no_calibrate: true,
                    ..Default::default()
                },
            )?;
            let mut probe = Probe {
                app,
                opts,
                cases,
                phase: 0,
                phase_start: Instant::now(),
                previous_ui: None,
                measurement_start: None,
                samples: Vec::new(),
                results: Vec::new(),
                pending_phase: false,
                phase_presentations: None,
                measured_at: None,
                transition: Vec::new(),
                failure: None,
                outcome,
            };
            probe.set_phase();
            Ok(Box::new(probe))
        }),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    match result.lock().expect("outcome lock").take() {
        Some(Ok(())) => Ok(()),
        Some(Err(error)) => bail!(error),
        None => bail!("cadence window closed before all phases completed"),
    }
}
fn main() {
    if let Err(error) = run() {
        eprintln!("cadence-check: {error:#}");
        std::process::exit(1);
    }
}
