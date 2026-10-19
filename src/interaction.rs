//! Pure input decisions ported from Fluoddity (MIT, Jesse Gelders, 2026).
//! Sources: orchestrator/{drawingCommands,shoveCommands}.ts and
//! selection/cohortHighlight.ts; license in assets/presets/LICENSE.txt.

pub const SHOVE_GAIN: f64 = 0.004;
pub const SHOVE_REFERENCE_STEPS: f64 = 30.0;
pub const SHOVE_RATE_EXPONENT: f64 = 0.75;
pub const LINE_STROKE_GAIN: f32 = 1.0;
pub const WALLS_FIELD_GAIN: f32 = 0.01;
pub const TRAILS_FIELD_GAIN: f32 = 0.001;

/// Per-substep world displacement, with the reference's rate falloff.
/// Compute in f64 like JavaScript, then narrow once at the GPU boundary.
pub fn shove_strength(draw_power: f32, steps: u32, pulling: bool) -> f32 {
    let falloff = (steps.max(1) as f64).powf(SHOVE_RATE_EXPONENT)
        / SHOVE_REFERENCE_STEPS.powf(SHOVE_RATE_EXPONENT - 1.0);
    let strength = (SHOVE_GAIN * draw_power as f64) / falloff;
    (if pulling { -strength } else { strength }) as f32
}

/// Brush sigma in world units; drawing uses the same sigma in corrected UV.
pub fn shove_radius(draw_size: f32) -> f32 {
    draw_size * 2.0
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StrokeInput {
    pub uv: [f32; 2],
    pub left_dragging: bool,
    pub right_dragging: bool,
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub shift: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub uv: [f32; 2],
    pub previous: [f32; 2],
    pub erase: bool,
    pub is_line: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StrokeState {
    pub previous: Option<[f32; 2]>,
    pub anchor: Option<[f32; 2]>,
}

impl StrokeState {
    /// One rendered frame's segment, never one segment per physics substep.
    /// Button flags must describe canvas-owned gestures, matching InputState.
    pub fn update(&mut self, input: StrokeInput) -> Option<Stroke> {
        let drawing = input.left_dragging;
        let erasing = input.right_dragging && !drawing;
        let uv = input.uv;
        if input.shift {
            let Some(anchor) = self.anchor else {
                if drawing || erasing {
                    // The source's line branch owns this frame: no paint yet.
                    self.previous = Some(uv);
                    self.anchor = None;
                } else {
                    self.previous = None;
                    self.anchor = Some(uv);
                }
                return None;
            };
            self.previous = None;
            if input.left_pressed || input.right_pressed {
                self.anchor = Some(uv);
                return Some(Stroke {
                    uv,
                    previous: anchor,
                    erase: !input.left_pressed,
                    is_line: true,
                });
            }
            return None;
        }
        self.anchor = None;
        if !drawing && !erasing {
            self.previous = None;
            return None;
        }
        let previous = self.previous.unwrap_or(uv);
        self.previous = Some(uv);
        Some(Stroke {
            uv,
            previous,
            erase: erasing,
            is_line: false,
        })
    }
}

/// A valid hit commits only when it matches the already highlighted cohort,
/// or when one-click selection is enabled. Pass `one_click || cohorts <= 1`
/// to reproduce the source orchestrator's single-cohort exemption.
/// Negative/nonfinite picks denote invalid results and cannot confirm.
pub fn should_confirm(highlight: Option<f32>, picked: f32, one_click: bool) -> bool {
    if !picked.is_finite() || picked < 0.0 {
        return false;
    }
    one_click || highlight.is_some_and(|h| h.is_finite() && h >= 0.0 && h == picked)
}
/// Global commands consume their key events so a focused GUI button cannot
/// also activate (e.g. Space to resume after pressing Start Export).
pub fn command_keys(
    ctx: &eframe::egui::Context,
) -> Vec<(eframe::egui::Key, eframe::egui::Modifiers)> {
    use eframe::egui::{Event, Key};
    ctx.input_mut(|input| {
        let pressed: Vec<Key> = [
            Key::Space,
            Key::R,
            Key::F,
            Key::B,
            Key::Z,
            Key::C,
            Key::V,
            Key::M,
            Key::G,
            Key::OpenBracket,
            Key::CloseBracket,
            Key::Comma,
            Key::Period,
            Key::Home,
            Key::X,
            Key::Enter,
            Key::H,
            Key::Slash,
            Key::P,
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::ArrowRight,
            Key::ArrowLeft,
        ]
        .into_iter()
        .filter(|k| input.key_pressed(*k))
        .collect();
        let commands = input
            .events
            .iter()
            .filter_map(|event| {
                if let Event::Key {
                    key,
                    modifiers,
                    pressed: true,
                    ..
                } = event
                {
                    pressed.contains(key).then_some((*key, *modifiers))
                } else {
                    None
                }
            })
            .collect();
        input.events.retain(
            |event| !matches!(event,Event::Key{key,pressed:true,..} if pressed.contains(key)),
        );
        commands
    })
}
