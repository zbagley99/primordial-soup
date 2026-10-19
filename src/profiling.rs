//! Optional portable GPU timestamps. Three bounded readbacks never wait in the UI.
//! Empty compute-pass boundaries work with baseline TIMESTAMP_QUERY on WebGPU.
use std::sync::{Arc, Mutex};

const QUERY_COUNT: u32 = wgpu::QUERY_SET_MAX_QUERIES;
const RING_SIZE: usize = 3;
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    Entity,
    Trails,
    Drawing,
    Bloom,
    Assembly,
    CellsDensity,
    CellsSprites,
    CellsResolve,
    CellsPlanning,
    CellsRaster,
    CellsAverage,
    CellsMembranes,
    CellsTiled,
}
impl Stage {
    /// Diagnostic subspans within the inclusive CellsSprites stage.
    pub const DETAIL_NAMES: [&'static str; 5] = [
        "cells_planning",
        "cells_raster",
        "cells_average",
        "cells_membranes",
        "cells_tiled",
    ];
    pub const NAMES: [&'static str; 8] = [
        "entity",
        "trails",
        "drawing",
        "bloom",
        "assembly",
        "cells_density",
        "cells_sprites",
        "cells_resolve",
    ];
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuSample {
    pub total_ms: f64,
    /// Sum of every repetition, in Stage::NAMES order. Total also includes gaps.
    pub stages_ms: [f64; 8],
    /// Diagnostic subspans in Stage::DETAIL_NAMES order; do not add to stages_ms.
    pub details_ms: [f64; 5],
    /// Simulation ticks in the measured frame: one entity span is recorded per tick.
    pub steps: u32,
}
impl GpuSample {
    /// Entity update plus trail processing, per simulation tick.
    pub fn physics_ms_per_step(&self) -> Option<f64> {
        (self.steps > 0).then(|| (self.stages_ms[0] + self.stages_ms[1]) / self.steps as f64)
    }
}
type Status = Arc<Mutex<Option<Result<(), String>>>>;
struct Slot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    pending: Option<ProfileFrame>,
}
pub struct GpuProfiler {
    slots: Vec<Slot>,
    pub skipped_frames: u64,
    /// Entire samples discarded because their stage spans exceeded the query budget.
    pub capacity_skipped_frames: u64,
}
pub struct ProfileFrame {
    slot: usize,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    status: Status,
    cursor: u32,
    ranges: Vec<(Stage, u32, u32)>,
    finished: bool,
    cancel_on_drop: bool,
    capacity_exhausted: bool,
}
impl GpuProfiler {
    pub fn new(d: &wgpu::Device) -> Option<Self> {
        if !d.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let slots = (0..RING_SIZE)
            .map(|_| Slot {
                queries: d.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("stage timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERY_COUNT,
                }),
                resolve: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stage timestamp resolve"),
                    size: u64::from(QUERY_COUNT) * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stage timestamp readback"),
                    size: u64::from(QUERY_COUNT) * 8,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                pending: None,
            })
            .collect();
        Some(Self {
            slots,
            skipped_frames: 0,
            capacity_skipped_frames: 0,
        })
    }
    /// Begin immediately before the first GPU command being measured.
    pub fn begin_frame(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<ProfileFrame> {
        let Some(slot) = self.slots.iter().position(|s| s.pending.is_none()) else {
            self.skipped_frames += 1;
            return None;
        };
        let s = &self.slots[slot];
        let mut frame = ProfileFrame {
            slot,
            queries: s.queries.clone(),
            resolve: s.resolve.clone(),
            readback: s.readback.clone(),
            status: Arc::new(Mutex::new(None)),
            cursor: 2,
            ranges: Vec::new(),
            finished: false,
            cancel_on_drop: true,
            capacity_exhausted: false,
        };
        frame.mark(encoder, 0);
        // Reserve the slot even before submission. One outstanding frame per caller.
        self.slots[slot].pending = Some(frame.reservation());
        Some(frame)
    }
    /// True only while a frame reservation or readback needs servicing.
    pub fn has_pending(&self) -> bool {
        self.slots.iter().any(|s| s.pending.is_some())
    }
    /// Release an abandoned frame before its command encoder is submitted.
    pub fn cancel_frame(&mut self, mut frame: ProfileFrame) {
        frame.cancel_on_drop = false;
        self.slots[frame.slot].readback.unmap();
        self.slots[frame.slot].pending = None;
    }
    /// Call once, after submitting the encoder containing frame.finish().
    pub fn after_submit(&mut self, mut frame: ProfileFrame) {
        assert!(frame.finished, "finish GPU profile before submission");
        frame.cancel_on_drop = false;
        if frame.capacity_exhausted {
            self.skipped_frames += 1;
            self.capacity_skipped_frames += 1;
            self.slots[frame.slot].pending = None;
            return;
        }
        let status = frame.status.clone();
        frame
            .readback
            .slice(..u64::from(frame.cursor) * 8)
            .map_async(wgpu::MapMode::Read, move |r| {
                *status.lock().unwrap() = Some(r.map_err(|e| e.to_string()));
            });
        let slot = frame.slot;
        self.slots[slot].pending = Some(frame);
    }
    /// Nonblocking: pending readbacks are retained; failed mappings are discarded.
    pub fn poll(&mut self, d: &wgpu::Device, q: &wgpu::Queue) -> Vec<GpuSample> {
        let _ = d.poll(wgpu::PollType::Poll);
        let mut samples = Vec::new();
        for s in &mut self.slots {
            let ready = s
                .pending
                .as_ref()
                .and_then(|f| f.status.lock().unwrap().take());
            if let Some(result) = ready {
                let f = s.pending.take().unwrap();
                if result.is_ok() {
                    if let Ok(bytes) = s
                        .readback
                        .slice(..u64::from(f.cursor) * 8)
                        .get_mapped_range()
                    {
                        let ticks: Vec<u64> = bytes
                            .chunks_exact(8)
                            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                            .collect();
                        samples.push(summarize(&ticks, &f.ranges, q.get_timestamp_period()));
                    }
                }
                s.readback.unmap();
            }
        }
        samples
    }
}
fn reserve_pair(cursor: &mut u32) -> Option<u32> {
    if *cursor + 2 > QUERY_COUNT {
        return None;
    }
    let start = *cursor;
    *cursor += 2;
    Some(start)
}
fn summarize(ticks: &[u64], ranges: &[(Stage, u32, u32)], period: f32) -> GpuSample {
    let scale = f64::from(period) / 1e6;
    let mut sample = GpuSample {
        total_ms: ticks[1].saturating_sub(ticks[0]) as f64 * scale,
        stages_ms: [0.0; 8],
        details_ms: [0.0; 5],
        steps: ranges
            .iter()
            .filter(|r| matches!(r.0, Stage::Entity))
            .count() as u32,
    };
    for &(stage, a, b) in ranges {
        let duration = ticks[b as usize].saturating_sub(ticks[a as usize]) as f64 * scale;
        let index = stage as usize;
        if index < Stage::NAMES.len() {
            sample.stages_ms[index] += duration;
        } else {
            sample.details_ms[index - Stage::NAMES.len()] += duration;
        }
    }
    sample
}
impl ProfileFrame {
    fn reservation(&self) -> Self {
        Self {
            slot: self.slot,
            queries: self.queries.clone(),
            resolve: self.resolve.clone(),
            readback: self.readback.clone(),
            status: self.status.clone(),
            cursor: 2,
            ranges: Vec::new(),
            finished: false,
            cancel_on_drop: false,
            capacity_exhausted: false,
        }
    }
    fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("portable timestamp boundary"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        });
    }
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        let Some(start) = reserve_pair(&mut self.cursor) else {
            self.capacity_exhausted = true;
            return None;
        };
        self.mark(encoder, start);
        Some(start)
    }
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder, stage: Stage, start: Option<u32>) {
        if let Some(start) = start {
            self.mark(encoder, start + 1);
            self.ranges.push((stage, start, start + 1));
        }
    }
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        assert!(!self.finished, "GPU profile already finished");
        self.mark(encoder, 1);
        self.finished = true;
        // Never publish partial stage totals. The simulation/render commands still run.
        if self.capacity_exhausted {
            return;
        }
        encoder.resolve_query_set(&self.queries, 0..self.cursor, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &self.readback,
            0,
            u64::from(self.cursor) * 8,
        );
        self.finished = true;
    }
}

fn timestamp_boundary(encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet, index: u32) {
    let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("frame timestamp boundary"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: None,
        }),
    });
}

/// Always-on frame total: two portable timestamps around a scene submission. Unlike
/// GpuProfiler it adds only two empty pass boundaries, so it never locks calibration.
/// Readbacks are bounded by the ring and never waited on; a full ring skips frames.
const TIMER_SLOTS: usize = 3;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimerState {
    Recording,
    Mapping,
    Ready,
    Failed,
    Abandoned,
}
type TimerStatus = Arc<Mutex<TimerState>>;
struct TimerSlot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: Option<TimerStatus>,
}
pub struct FrameTimer {
    slots: Vec<TimerSlot>,
    period: f32,
    /// Scene submissions that found every slot still awaiting readback.
    pub skipped_frames: u64,
}
pub struct FrameTiming {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    state: TimerStatus,
    finished: bool,
    submitted: bool,
}
impl FrameTimer {
    pub fn new(d: &wgpu::Device, q: &wgpu::Queue) -> Option<Self> {
        if !d.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let slots = (0..TIMER_SLOTS)
            .map(|_| TimerSlot {
                queries: d.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("frame timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolve: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame timestamp resolve"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame timestamp readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: None,
            })
            .collect();
        Some(Self {
            slots,
            period: q.get_timestamp_period(),
            skipped_frames: 0,
        })
    }
    /// Begin immediately before the first measured command.
    pub fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<FrameTiming> {
        let Some(slot) = self.slots.iter().position(|s| s.state.is_none()) else {
            self.skipped_frames += 1;
            return None;
        };
        let state = Arc::new(Mutex::new(TimerState::Recording));
        let s = &mut self.slots[slot];
        s.state = Some(state.clone());
        let timing = FrameTiming {
            queries: s.queries.clone(),
            resolve: s.resolve.clone(),
            readback: s.readback.clone(),
            state,
            finished: false,
            submitted: false,
        };
        timestamp_boundary(encoder, &timing.queries, 0);
        Some(timing)
    }
    /// True only while a recorded frame awaits submission or readback.
    pub fn has_pending(&self) -> bool {
        self.slots.iter().any(|s| s.state.is_some())
    }
    /// Call once, after submitting the encoder containing `FrameTiming::finish`.
    pub fn after_submit(&mut self, mut timing: FrameTiming) {
        assert!(timing.finished, "finish frame timing before submission");
        timing.submitted = true;
        *timing.state.lock().unwrap() = TimerState::Mapping;
        let state = timing.state.clone();
        timing
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                *state.lock().unwrap() = if r.is_ok() {
                    TimerState::Ready
                } else {
                    TimerState::Failed
                };
            });
    }
    /// Nonblocking: completed frame totals in milliseconds. Pending readbacks are retained.
    pub fn poll(&mut self) -> Vec<f64> {
        let scale = f64::from(self.period) / 1e6;
        let mut totals = Vec::new();
        for s in &mut self.slots {
            let Some(state) = s.state.as_ref() else {
                continue;
            };
            let current = *state.lock().unwrap();
            match current {
                TimerState::Recording | TimerState::Mapping => {}
                // Never mapped, so there is nothing to unmap.
                TimerState::Failed | TimerState::Abandoned => s.state = None,
                TimerState::Ready => {
                    if let Ok(bytes) = s.readback.slice(..).get_mapped_range() {
                        let tick =
                            |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
                        totals.push(tick(8).saturating_sub(tick(0)) as f64 * scale);
                    }
                    s.readback.unmap();
                    s.state = None;
                }
            }
        }
        totals
    }
}
impl FrameTiming {
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        assert!(!self.finished, "frame timing already finished");
        timestamp_boundary(encoder, &self.queries, 1);
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, 16);
        self.finished = true;
    }
}
impl Drop for FrameTiming {
    fn drop(&mut self) {
        if !self.submitted {
            *self.state.lock().unwrap() = TimerState::Abandoned;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn repeated_stage_spans_are_summed_with_timestamp_period() {
        let sample = super::summarize(
            &[0, 100, 10, 20, 30, 50, 60, 80],
            &[
                (super::Stage::Entity, 2, 3),
                (super::Stage::Entity, 4, 5),
                (super::Stage::Drawing, 6, 7),
            ],
            1000.0,
        );
        assert!((sample.total_ms - 0.1).abs() < 1e-12);
        assert!((sample.stages_ms[0] - 0.03).abs() < 1e-12);
        assert!((sample.stages_ms[2] - 0.02).abs() < 1e-12);
        assert_eq!(sample.stages_ms[1], 0.0);
        // One entity span is recorded per simulation tick.
        assert_eq!(sample.steps, 2);
        assert!((sample.physics_ms_per_step().unwrap() - 0.015).abs() < 1e-12);
        let idle = super::summarize(&[0, 10], &[], 1000.0);
        assert_eq!(idle.physics_ms_per_step(), None);
    }
    #[test]
    fn query_capacity_obeys_portable_wgpu_limit_and_bounded_workloads() {
        assert_eq!(super::QUERY_COUNT, wgpu::QUERY_SET_MAX_QUERIES);
        assert_eq!(super::QUERY_COUNT, 4096);
        // Benchmark max128 steps, each with entity, trail and camera spans.
        assert!(128 * 6 + 6 <= super::QUERY_COUNT);
        let mut cursor = super::QUERY_COUNT - 2;
        assert_eq!(
            super::reserve_pair(&mut cursor),
            Some(super::QUERY_COUNT - 2)
        );
        assert_eq!(super::reserve_pair(&mut cursor), None);
        assert_eq!(cursor, super::QUERY_COUNT);
        // MAX_STEPS is legal simulation work, but cannot fit one profiled sample.
        assert!(crate::gpu::MAX_STEPS * 6 + 6 > super::QUERY_COUNT);
    }
}

impl Drop for ProfileFrame {
    fn drop(&mut self) {
        if self.cancel_on_drop {
            *self.status.lock().unwrap() =
                Some(Err("GPU profile encoder abandoned before submission".into()));
        }
    }
}
