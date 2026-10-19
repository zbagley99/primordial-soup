//! Fluoddity's bounded calibration ladder, with asynchronous GPU readbacks.
use crate::{config::Project, coords::sizing_for, gpu::Engine};
use anyhow::{Result, bail};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use web_time::Instant;
pub const VERSION: u32 = 2;
pub const PROGRESSION: [(f64, u32); 7] = [
    (0.1, 1),
    (0.25, 1),
    (0.25, 10),
    (0.6, 10),
    (0.6, 15),
    (1.0, 15),
    (1.0, 20),
];
pub const BUDGET_MS: f64 = 16.7 * 0.7;
pub struct Calibration {
    engine: Engine,
    project: Project,
    rung: usize,
    best: usize,
    warmup: u32,
    samples: Vec<f64>,
    working: Duration,
    visibility_changed: Instant,
    visible: bool,
    pending: bool,
    status: Arc<Mutex<Option<std::result::Result<f64, String>>>>,
    timing: Option<Timing>,
    workgroup: u32,
    cancelled: bool,
}
struct Timing {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    mapped: bool,
}
impl Drop for Calibration {
    fn drop(&mut self) {
        if self
            .status
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(Result::is_err)
        {
            if let Some(t) = self.timing.as_mut() {
                t.mapped = false;
            }
        }
    }
}
impl Timing {
    fn new(d: &wgpu::Device) -> Option<Self> {
        d.features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Self {
                queries: d.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("calibration timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolve: d.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 256,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                staging: d.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                mapped: false,
            })
    }
    fn mark(&self, e: &mut wgpu::CommandEncoder, i: u32) {
        let _pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("calibration boundary"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: None,
            }),
        });
    }
}
impl Drop for Timing {
    fn drop(&mut self) {
        if self.mapped {
            self.staging.unmap();
        }
    }
}
impl Calibration {
    pub fn new(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        project: &Project,
        workgroup: u32,
    ) -> Result<Self> {
        let engine = Engine::new(d, q, sizing_for(PROGRESSION[1].0, 1.0)?, project, workgroup)?;
        Ok(Self {
            engine,
            project: project.clone(),
            rung: 1,
            best: 0,
            warmup: 25,
            samples: Vec::new(),
            working: Duration::ZERO,
            visibility_changed: Instant::now(),
            visible: true,
            pending: false,
            status: Arc::new(Mutex::new(None)),
            timing: Timing::new(d),
            workgroup,
            cancelled: false,
        })
    }
    pub fn progress(&self) -> usize {
        self.rung
    }
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
    pub fn set_visible(&mut self, visible: bool) {
        if self.visible {
            self.working += self.visibility_changed.elapsed();
        }
        self.visibility_changed = Instant::now();
        self.visible = visible;
    }
    pub fn tick(&mut self, d: &wgpu::Device, q: &wgpu::Queue) -> Result<Option<(f64, u32)>> {
        if !self.visible {
            return Ok(None);
        }
        if self.pending {
            let status = self.status.lock().unwrap().take();
            let Some(status) = status else {
                return Ok(None);
            };
            self.pending = false;
            let ms = if let Some(timing) = &mut self.timing {
                if let Err(e) = status {
                    timing.mapped = false;
                    bail!("Calibration readback: {e}");
                }
                let range = timing.staging.slice(..).get_mapped_range()?;
                let a = u64::from_le_bytes(range[..8].try_into().unwrap());
                let b = u64::from_le_bytes(range[8..16].try_into().unwrap());
                drop(range);
                timing.staging.unmap();
                timing.mapped = false;
                b.saturating_sub(a) as f64 * q.get_timestamp_period() as f64 / 1e6
            } else {
                status.map_err(anyhow::Error::msg)?
            };
            if self.warmup > 0 {
                self.warmup -= 1;
            } else {
                self.samples.push(ms);
            }
            if self.samples.len() == 10 {
                self.samples.sort_by(f64::total_cmp);
                if self.samples[5] > BUDGET_MS {
                    return Ok(Some(PROGRESSION[self.best]));
                }
                self.best = self.rung;
                // A passing rung counts even if its final sample crosses the
                // deadline. Time and cancellation gate the NEXT rung.
                if self.working.as_secs_f64() > 8.0 || self.cancelled {
                    return Ok(Some(PROGRESSION[self.best]));
                }
                self.rung += 1;
                if self.rung == PROGRESSION.len() {
                    return Ok(Some(PROGRESSION[self.best]));
                }
                let previous = PROGRESSION[self.rung - 1];
                if previous.0 != PROGRESSION[self.rung].0 {
                    self.engine = Engine::new(
                        d,
                        q,
                        sizing_for(PROGRESSION[self.rung].0, 1.0)?,
                        &self.project,
                        self.workgroup,
                    )?;
                    self.warmup = 25;
                } else {
                    self.warmup = 2;
                }
                self.samples.clear();
            }
        }
        self.engine.prepare_frame(
            q,
            &self.project.world,
            PROGRESSION[self.rung].1,
            None,
            [0.01, 0.001],
        )?;
        let begin = Instant::now();
        let mut encoder = d.create_command_encoder(&Default::default());
        if let Some(t) = &self.timing {
            t.mark(&mut encoder, 0);
        }
        for i in 0..PROGRESSION[self.rung].1 {
            self.engine.encode_step(&mut encoder, i);
        }
        if let Some(t) = &self.timing {
            t.mark(&mut encoder, 1);
            encoder.resolve_query_set(&t.queries, 0..2, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.staging, 0, 16);
        }
        q.submit([encoder.finish()]);
        self.pending = true;
        let status = self.status.clone();
        if let Some(t) = &mut self.timing {
            t.mapped = true;
            t.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    *status.lock().unwrap() = Some(r.map(|_| 0.0).map_err(|e| e.to_string()));
                });
        } else {
            q.on_submitted_work_done(move || {
                *status.lock().unwrap() = Some(Ok(begin.elapsed().as_secs_f64() * 1000.0));
            });
        }
        Ok(None)
    }
}
