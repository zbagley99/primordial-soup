//! Fluoddity bounded project history (MIT, Jesse Gelders, 2026).
use crate::config::Project;
pub const MAX_HISTORY: usize = 100;
pub const COALESCE_WINDOW_MS: f64 = 1500.0;
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub project: Project,
    pub label: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordOutcome {
    Appended,
    Coalesced,
}
#[derive(Clone, Debug)]
pub struct History {
    states: Vec<HistoryEntry>,
    cursor: usize,
    last_key: Option<String>,
    last_time: f64,
}
impl Default for History {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            cursor: 0,
            last_key: None,
            last_time: 0.0,
        }
    }
}
impl History {
    pub fn seed(&mut self, p: Project) {
        self.states = vec![HistoryEntry {
            project: p,
            label: String::new(),
        }];
        self.cursor = 0;
        self.break_coalescing();
    }
    pub fn break_coalescing(&mut self) {
        self.last_key = None;
    }
    pub fn record(
        &mut self,
        before: Project,
        after: Project,
        label: impl Into<String>,
        key: Option<&str>,
        now_ms: f64,
    ) -> RecordOutcome {
        let label = label.into();
        if key.is_some()
            && key == self.last_key.as_deref()
            && self.cursor > 0
            && now_ms - self.last_time <= COALESCE_WINDOW_MS
        {
            self.states[self.cursor] = HistoryEntry {
                project: after,
                label,
            };
            self.last_time = now_ms;
            return RecordOutcome::Coalesced;
        }
        if self.states.is_empty() {
            self.states.push(HistoryEntry {
                project: before,
                label: String::new(),
            });
            self.cursor = 0;
        } else {
            self.states.truncate(self.cursor + 1);
            self.states[self.cursor].project = before;
        }
        self.states.push(HistoryEntry {
            project: after,
            label,
        });
        if self.states.len() > MAX_HISTORY {
            self.states.drain(..self.states.len() - MAX_HISTORY);
        }
        self.cursor = self.states.len() - 1;
        self.last_key = key.map(str::to_owned);
        self.last_time = now_ms;
        RecordOutcome::Appended
    }
    pub fn can_undo(&self) -> bool {
        !self.states.is_empty() && self.cursor > 0
    }
    pub fn can_redo(&self) -> bool {
        !self.states.is_empty() && self.cursor + 1 < self.states.len()
    }
    pub fn undo(&mut self) -> Option<Project> {
        self.break_coalescing();
        if !self.can_undo() {
            return None;
        }
        self.cursor -= 1;
        Some(self.states[self.cursor].project.clone())
    }
    pub fn redo(&mut self) -> Option<Project> {
        self.break_coalescing();
        if !self.can_redo() {
            return None;
        }
        self.cursor += 1;
        Some(self.states[self.cursor].project.clone())
    }
    pub fn depth(&self) -> usize {
        self.states.len()
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn undo_label(&self) -> &str {
        if self.can_undo() {
            &self.states[self.cursor].label
        } else {
            ""
        }
    }
    pub fn redo_label(&self) -> &str {
        if self.can_redo() {
            &self.states[self.cursor + 1].label
        } else {
            ""
        }
    }
}
