//! Wall-clock scheduling only: each dispatched tick retains the existing numerical update.
use serde::{Deserialize, Serialize};
use std::time::Duration;

const NANOS_PER_SECOND: u128 = 1_000_000_000;
pub const DEFAULT_MAX_STEPS_PER_UPDATE: u32 = 120;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationMode {
    /// Compatibility mode: the existing Physics Rate is steps per rendered frame.
    #[default]
    PerFrame,
    FixedRate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SimulationSettings {
    pub mode: SimulationMode,
    pub ticks_per_second: u32,
    pub presentation_fps: u32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            mode: SimulationMode::PerFrame,
            ticks_per_second: 300,
            presentation_fps: 60,
        }
    }
}

impl SimulationSettings {
    pub fn sanitize(&mut self) {
        self.ticks_per_second = self.ticks_per_second.clamp(1, 7_200);
        self.presentation_fps = self.presentation_fps.clamp(1, 60);
    }

    pub fn frame_interval(self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.presentation_fps.max(1)))
    }

    /// Wake often enough to dispatch the requested tick rate within the bounded
    /// GPU batch, even when presentation itself is configured to run slowly.
    pub fn service_interval(self, max_steps: u32) -> Duration {
        let presentation = self.frame_interval();
        match self.mode {
            SimulationMode::PerFrame => presentation,
            SimulationMode::FixedRate => presentation.min(Duration::from_secs_f64(
                f64::from(max_steps.max(1)) / f64::from(self.ticks_per_second.max(1)),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickPlan {
    pub steps: u32,
    /// Exact work remaining after this batch, independent of the automatic clock.
    pub pending_exact_steps: u64,
    /// Whole automatic ticks still owed after this batch; overload is never discarded.
    pub wall_debt_ticks: u64,
}

#[derive(Debug, Default)]
pub struct SimulationClock {
    // Units are nanoseconds multiplied by ticks/second. Integer accumulation avoids
    // a truncated tick period (e.g. 60 Hz) drifting across different frame cadences.
    wall_units: u128,
    pending_exact: u64,
    completed: u64,
    was_active: bool,
    previous_rate: Option<(SimulationMode, u32)>,
}

impl SimulationClock {
    pub fn queue_exact_steps(&mut self, steps: u64) {
        self.pending_exact = self.pending_exact.saturating_add(steps);
    }

    pub fn pending_exact_steps(&self) -> u64 {
        self.pending_exact
    }

    pub fn cancel_exact_steps(&mut self) {
        self.pending_exact = 0;
        self.clear_wall_debt();
    }

    pub fn completed_ticks(&self) -> u64 {
        self.completed
    }

    /// Call only after the application has actually submitted these simulation steps.
    /// Initialization of GPU resources is deliberately not a logical simulation tick.
    pub fn complete_steps(&mut self, steps: u32) {
        self.completed = self.completed.saturating_add(u64::from(steps));
    }

    pub fn wall_debt_ticks(&self) -> u64 {
        (self.wall_units / NANOS_PER_SECOND).min(u128::from(u64::MAX)) as u64
    }

    /// Invalidate elapsed time at pause, focus/background, or reset boundaries.
    /// The next active update establishes a fresh epoch, without a catch-up burst.
    pub fn clear_wall_debt(&mut self) {
        self.wall_units = 0;
        self.was_active = false;
    }

    /// Background operation preserves queued exact work but starts a new wall epoch.
    pub fn suspend(&mut self) {
        self.clear_wall_debt();
    }

    /// Clear logical progress and both kinds of queued work for a new experiment.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Reserve at most `max_steps` for one update. The caller must submit the returned
    /// work once and then call `complete_steps`; planning itself is not completion.
    /// `active` controls automatic ticking; explicit queued work also runs while paused.
    pub fn plan(
        &mut self,
        elapsed: Duration,
        active: bool,
        settings: SimulationSettings,
        steps_per_frame: u32,
        max_steps: u32,
    ) -> TickPlan {
        let rate = (settings.mode, settings.ticks_per_second);
        if self.previous_rate != Some(rate) {
            self.clear_wall_debt();
            self.previous_rate = Some(rate);
        }
        if !active {
            self.clear_wall_debt();
        } else if self.was_active && settings.mode == SimulationMode::FixedRate {
            self.wall_units = self.wall_units.saturating_add(
                elapsed
                    .as_nanos()
                    .saturating_mul(u128::from(settings.ticks_per_second)),
            );
        }
        self.was_active = active;

        let steps = if self.pending_exact > 0 {
            let steps = self.pending_exact.min(u64::from(max_steps)) as u32;
            self.pending_exact -= u64::from(steps);
            steps
        } else if active {
            match settings.mode {
                SimulationMode::PerFrame => steps_per_frame.min(max_steps),
                SimulationMode::FixedRate => {
                    let steps = self.wall_debt_ticks().min(u64::from(max_steps)) as u32;
                    self.wall_units -= u128::from(steps) * NANOS_PER_SECOND;
                    steps
                }
            }
        } else {
            0
        };
        TickPlan {
            steps,
            pending_exact_steps: self.pending_exact,
            wall_debt_ticks: self.wall_debt_ticks(),
        }
    }
}

/// Each optional deadline is a remaining duration supplied by its owner. Idle pause
/// with no real outstanding work has no timer. Services and deferred saves may still
/// request a bounded wake-up, independent of simulation or presentation.
pub fn repaint_after(
    running: bool,
    pending_exact: bool,
    settings: SimulationSettings,
    service_deadline: Option<Duration>,
    save_deadline: Option<Duration>,
) -> Option<Duration> {
    let presentation = if pending_exact {
        Some(Duration::ZERO)
    } else if running {
        Some(settings.frame_interval())
    } else {
        None
    };
    [presentation, service_deadline, save_deadline]
        .into_iter()
        .flatten()
        .min()
}
