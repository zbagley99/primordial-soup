//! Frame-rate feedback from Fluoddity's perf/fpsBand.ts and main.ts.
//! MIT Copyright (c) 2026 Jesse Gelders; see assets/presets/LICENSE.txt.
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Red,
    Yellow,
    Green,
}
impl Band {
    fn for_fps(fps: f64) -> Self {
        if fps >= 50.0 {
            Self::Green
        } else if fps >= 35.0 {
            Self::Yellow
        } else {
            Self::Red
        }
    }
    fn clears_margin(self, fps: f64) -> bool {
        let (floor, ceiling) = match self {
            Self::Red => (0.0, 35.0),
            Self::Yellow => (35.0, 50.0),
            Self::Green => (50.0, f64::INFINITY),
        };
        fps < floor - 3.0 || fps > ceiling + 3.0
    }
    pub fn badge_color(self) -> [u8; 3] {
        match self {
            Self::Red => [245, 163, 163],
            Self::Yellow => [242, 217, 160],
            Self::Green => [168, 220, 176],
        }
    }
    pub fn label_color(self) -> [u8; 3] {
        match self {
            Self::Red => [255, 128, 128],
            Self::Yellow => [255, 200, 87],
            Self::Green => [110, 222, 138],
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Red => "GPU struggling",
            Self::Yellow => "GPU under strain",
            Self::Green => "Holding the frame rate",
        }
    }
    pub fn help(self) -> &'static str {
        match self {
            Self::Green => {
                "The frame rate is holding. Try raising Physics Rate, enabling Motion Blur, or increasing World Size. Click for performance settings."
            }
            _ => {
                "The frame rate is falling behind. Try disabling Motion Blur, reducing Physics Rate, or lowering World Size. Click for performance settings."
            }
        }
    }
}

pub struct Meter {
    samples: VecDeque<(f64, f64)>,
    pub band: Band,
    pub readout: u32,
    pending: Option<(Band, f64)>,
}
impl Default for Meter {
    fn default() -> Self {
        Self {
            samples: VecDeque::new(),
            band: Band::Green,
            readout: 60,
            pending: None,
        }
    }
}
impl Meter {
    pub fn sample(&mut self, now: f64, ms: f64) {
        if ms > 0.0 && ms < 500.0 {
            self.samples.push_back((now, ms));
        }
        while self.samples.front().is_some_and(|s| now - s.0 > 250.0) {
            self.samples.pop_front();
        }
    }
    /// Paused/recording frames are sampled but must not change the displayed measurement.
    pub fn update(&mut self, now: f64, adjusting: bool) {
        let cutoff =
            self.samples.back().map_or(now, |s| s.0) - if adjusting { 80.0 } else { 250.0 };
        let (total, count) = self
            .samples
            .iter()
            .filter(|s| s.0 >= cutoff)
            .fold((0.0, 0), |(total, count), s| (total + s.1, count + 1));
        if count == 0 {
            return;
        }
        let fps = 1000.0 / (total / count as f64);
        if !fps.is_finite() {
            return;
        }
        self.readout = fps.round().clamp(1.0, 60.0) as u32;
        let target = Band::for_fps(fps);
        if target == self.band || !self.band.clears_margin(fps) {
            self.pending = None;
        } else if adjusting
            || self
                .pending
                .is_some_and(|(b, since)| b == target && now - since >= 250.0)
        {
            self.band = target;
            self.pending = None;
        } else if !self.pending.is_some_and(|(b, _)| b == target) {
            self.pending = Some((target, now));
        }
    }
}

/// GPU frame-time budget for the reference workload (world size 1, physics rate 15).
pub const TARGET_GPU_MS: f64 = 5.0;
/// The meter needs a full sampling window after resuming before its colour means anything.
pub const MEASURING_MS: f64 = 250.0;
/// GPU totals are summarized over roughly this much recent scene-submission time.
pub const GPU_WINDOW_MS: f64 = 1000.0;
const GPU_WINDOW_LIMIT: usize = 512;

/// What the status dot beside the frame-rate readout shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DotState {
    /// The meter's current, debounced verdict.
    Live(Band),
    /// The simulation resumed too recently for the meter to have fresh samples.
    Measuring,
    /// Paused, recording or exporting: the meter is not updating.
    Idle,
}
/// `running_since` is when the meter last started receiving live frames; None while it is frozen.
pub fn dot_state(band: Band, running_since: Option<f64>, now: f64) -> DotState {
    match running_since {
        None => DotState::Idle,
        Some(since) if now - since < MEASURING_MS => DotState::Measuring,
        Some(_) => DotState::Live(band),
    }
}

fn one_decimal(value: f64, unit: &str) -> String {
    let rounded = (value * 10.0).round() / 10.0;
    if rounded.fract() == 0.0 {
        format!("{rounded:.0}{unit}")
    } else {
        format!("{rounded:.1}{unit}")
    }
}
/// Short particle counts for one-line readouts: 950, 1.5k, 600k, 1.2M, 27M.
pub fn compact_count(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    if n < 10_000 {
        return one_decimal(n as f64 / 1e3, "k");
    }
    let thousands = (n as f64 / 1e3).round();
    if thousands < 1_000.0 {
        return format!("{thousands:.0}k");
    }
    let millions = n as f64 / 1e6;
    if millions < 10.0 {
        one_decimal(millions, "M")
    } else {
        format!("{:.0}M", millions.round())
    }
}
/// Full counts with thousands separators: 600,000.
pub fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One-line badge: `60 fps · 4.8 ms GPU · 600k particles · ×15`. The GPU segment is
/// omitted when no timestamps are available.
pub fn badge_text(head: &str, gpu_ms: Option<f64>, particles: u64, rate: &str) -> String {
    let gpu = gpu_ms.map_or(String::new(), |ms| format!(" · {ms:.1} ms GPU"));
    format!(
        "{head}{gpu} · {} particles · {rate}",
        compact_count(particles)
    )
}

/// Nearest-rank percentile; `q` is in 0..=1.
pub fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}
/// Chrome rounds WebGPU timestamps to 100 µs unless developer features are enabled,
/// so every measured interval lands on a multiple of 0.1 ms.
pub fn quantized(values: &[f64]) -> bool {
    values.len() >= 4
        && values
            .iter()
            .all(|v| ((v * 10.0) - (v * 10.0).round()).abs() < 1e-6)
}

/// GPU frame totals from roughly the last second of scene submissions. The window is
/// relative to the newest sample, so a paused app keeps showing its last activity.
#[derive(Default)]
pub struct GpuWindow {
    samples: VecDeque<(f64, f64)>,
}
impl GpuWindow {
    pub fn push(&mut self, now: f64, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        self.samples.push_back((now, ms));
        while self
            .samples
            .front()
            .is_some_and(|s| now - s.0 > GPU_WINDOW_MS)
            || self.samples.len() > GPU_WINDOW_LIMIT
        {
            self.samples.pop_front();
        }
    }
    pub fn values(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.1).collect()
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    pub fn median(&self) -> Option<f64> {
        percentile(&self.values(), 0.5)
    }
    pub fn p95(&self) -> Option<f64> {
        percentile(&self.values(), 0.95)
    }
    pub fn quantized(&self) -> bool {
        quantized(&self.values())
    }
}

/// Achieved rate of a monotonically increasing counter over roughly the last second.
#[derive(Default)]
pub struct RateWindow {
    samples: VecDeque<(f64, u64)>,
}
impl RateWindow {
    pub fn push(&mut self, now: f64, count: u64) {
        // A reset or restart lowers the counter; earlier samples no longer describe it.
        if self
            .samples
            .back()
            .is_some_and(|s| count < s.1 || now < s.0)
        {
            self.samples.clear();
        }
        self.samples.push_back((now, count));
        while self.samples.len() > 2 && now - self.samples[1].0 >= GPU_WINDOW_MS {
            self.samples.pop_front();
        }
    }
    /// Units per second, once at least 100 ms has been observed.
    pub fn rate(&self) -> Option<f64> {
        let (first, last) = (self.samples.front()?, self.samples.back()?);
        let span = last.0 - first.0;
        (span >= 100.0).then(|| (last.1 - first.1) as f64 * 1000.0 / span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_and_grouped_counts() {
        for (n, s) in [
            (0, "0"),
            (950, "950"),
            (1_500, "1.5k"),
            (2_000, "2k"),
            (9_999, "10k"),
            (60_000, "60k"),
            (150_000, "150k"),
            (600_000, "600k"),
            (999_600, "1M"),
            (1_200_000, "1.2M"),
            (2_400_000, "2.4M"),
            (16_200_000, "16M"),
        ] {
            assert_eq!(compact_count(n), s, "{n}");
        }
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_000), "1,000");
        assert_eq!(grouped(600_000), "600,000");
        assert_eq!(grouped(16_200_000), "16,200,000");
        assert_eq!(
            badge_text("60 fps", Some(4.83), 600_000, "×15"),
            "60 fps · 4.8 ms GPU · 600k particles · ×15"
        );
        assert_eq!(
            badge_text("Paused", None, 1_200_000, "300 ticks/s"),
            "Paused · 1.2M particles · 300 ticks/s"
        );
    }
    #[test]
    fn rolling_gpu_window_reports_median_and_p95() {
        let mut w = GpuWindow::default();
        assert_eq!(w.median(), None);
        for i in 0..20 {
            w.push(i as f64 * 16.0, (i + 1) as f64);
        }
        assert_eq!(w.len(), 20);
        assert_eq!(w.median(), Some(10.0));
        assert_eq!(w.p95(), Some(19.0));
        // Older than one second relative to the newest sample is discarded.
        w.push(2_000.0, 4.0);
        assert_eq!(w.len(), 1);
        assert_eq!(w.median(), Some(4.0));
        w.push(2_016.0, f64::NAN);
        w.push(2_016.0, -1.0);
        assert_eq!(w.len(), 1);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.0), Some(1.0));
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 1.0), Some(3.0));
    }
    #[test]
    fn quantization_detects_tenth_millisecond_steps() {
        assert!(quantized(&[0.1, 4.8, 5.0, 0.0]));
        assert!(!quantized(&[0.1, 4.8, 5.0]), "too few samples to judge");
        assert!(!quantized(&[0.1, 4.8, 5.0, 4.8123]));
        let chrome: Vec<f64> = [48u64, 50, 51, 47]
            .iter()
            .map(|k| (k * 100_000) as f64 * 1.0 / 1e6)
            .collect();
        assert!(quantized(&chrome));
    }
    #[test]
    fn dot_follows_meter_state() {
        assert_eq!(dot_state(Band::Green, None, 10_000.0), DotState::Idle);
        assert_eq!(
            dot_state(Band::Red, Some(1_000.0), 1_100.0),
            DotState::Measuring
        );
        assert_eq!(
            dot_state(Band::Red, Some(1_000.0), 1_000.0 + MEASURING_MS),
            DotState::Live(Band::Red)
        );
    }
    #[test]
    fn tick_rate_window() {
        let mut r = RateWindow::default();
        r.push(0.0, 0);
        assert_eq!(r.rate(), None);
        r.push(50.0, 45);
        assert_eq!(r.rate(), None, "needs 100 ms");
        for i in 1..=120 {
            r.push(i as f64 * 1000.0 / 60.0, i * 15);
        }
        let rate = r.rate().unwrap();
        assert!((rate - 900.0).abs() < 1e-9, "{rate}");
        r.push(2_100.0, 3);
        assert_eq!(r.rate(), None, "counter reset clears the window");
    }
}
