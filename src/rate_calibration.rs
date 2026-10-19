//! Current-project rate search from Fluoddity perf/{rateSearch,autoCalibrate}.ts.
//! Copyright Jesse Gelders, 2026, MIT; see LICENSES/Fluoddity-MIT.txt.
use serde::{Deserialize, Serialize};

pub const TARGET_MS: f64 = 16.7;
pub const MAX_PROBES: usize = 8;
pub const CEILING_SECONDS: u64 = 20;
const PLATEAU_MS: f64 = 17.6;

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Probe {
    pub rate: u32,
    #[serde(rename = "frameMs")]
    pub frame_ms: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum SearchStep {
    Probe { rate: u32 },
    Done { rate: u32 },
}
pub fn next_step(history: &[Probe], lo: u32, hi: u32) -> SearchStep {
    let clamp = |rate: f64| (rate.round() as u32).clamp(lo, hi);
    let best = || {
        let rate = history
            .iter()
            .filter(|p| p.frame_ms <= PLATEAU_MS)
            .map(|p| p.rate)
            .max()
            .unwrap_or(0);
        if rate == 0 {
            lo
        } else {
            clamp(rate as f64 * 0.9)
        }
    };
    let Some(last) = history.last() else {
        return SearchStep::Probe {
            rate: clamp((lo as f64 + hi as f64) / 2.0),
        };
    };
    if history.len() >= MAX_PROBES {
        return SearchStep::Done { rate: best() };
    }
    if last.frame_ms <= PLATEAU_MS {
        let climbed = clamp(last.rate as f64 * 1.7);
        if climbed <= last.rate || last.rate >= hi {
            return SearchStep::Done {
                rate: clamp(last.rate as f64),
            };
        }
        return if history.iter().any(|p| p.rate == climbed) {
            SearchStep::Done { rate: best() }
        } else {
            SearchStep::Probe { rate: climbed }
        };
    }
    let estimate = clamp(last.rate as f64 * (TARGET_MS / last.frame_ms).clamp(1.0 / 2.5, 2.5));
    if estimate == last.rate || history.iter().any(|p| p.rate == estimate) {
        SearchStep::Done { rate: best() }
    } else {
        SearchStep::Probe { rate: estimate }
    }
}

pub struct AutoCalibration {
    start: u32,
    rate: u32,
    lo: u32,
    hi: u32,
    seen: u32,
    samples: Vec<f64>,
    history: Vec<Probe>,
    finished: bool,
    result: Option<u32>,
}
impl AutoCalibration {
    pub fn new(start: u32, lo: u32, hi: u32) -> Self {
        Self {
            start,
            rate: start.clamp(lo, hi),
            lo,
            hi,
            seen: 0,
            samples: Vec::with_capacity(15),
            history: Vec::with_capacity(MAX_PROBES),
            finished: false,
            result: None,
        }
    }
    pub fn rate(&self) -> u32 {
        self.rate
    }
    pub fn probe_count(&self) -> usize {
        self.history.len()
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn result(&self) -> Option<u32> {
        self.result
    }
    /// Feed one real rendered frame, including camera, motion blur and bloom.
    pub fn on_frame(&mut self, frame_ms: f64) {
        if self.finished {
            return;
        }
        self.seen += 1;
        if self.seen <= 5 {
            return;
        }
        if frame_ms > 0.0 && frame_ms < 500.0 {
            self.samples.push(frame_ms);
        }
        if self.samples.len() < 15 {
            return;
        }
        self.history.push(Probe {
            rate: self.rate,
            frame_ms: self.samples.iter().sum::<f64>() / self.samples.len() as f64,
        });
        match next_step(&self.history, self.lo, self.hi) {
            SearchStep::Done { rate } => {
                self.rate = rate;
                self.result = Some(rate);
                self.finished = true;
            }
            SearchStep::Probe { rate } => {
                self.rate = rate;
                self.seen = 0;
                self.samples.clear();
            }
        }
    }
    /// Cancellation returns the original choice rather than an unfinished probe.
    pub fn cancel(&mut self) -> u32 {
        if !self.finished {
            self.finished = true;
            self.result = None;
            self.rate = self.start;
        }
        self.rate
    }
}
