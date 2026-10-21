//! Studio controls ported from Fluoddity's settingsSpec and drawingSection.
//! MIT Copyright (c) 2026 Jesse Gelders; see assets/presets/LICENSE.txt.
use crate::config::{Preferences, Project};
use eframe::egui::{self, Response, Ui};

pub fn begin_performance_frame(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("performance drag"), false));
}
pub fn performance_dragging(ctx: &egui::Context) -> bool {
    ctx.data_mut(|d| {
        d.get_temp::<bool>(egui::Id::new("performance drag"))
            .unwrap_or(false)
    })
}
pub fn observe_performance_drag(ctx: &egui::Context, response: &Response) {
    if response.dragged() || response.is_pointer_button_down_on() {
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("performance drag"), true));
    }
}
/// Measurements the Preferences panel uses to explain what its settings cost.
pub struct PerformanceHints {
    pub band: crate::performance::Band,
    /// Entity update plus trails per tick, from the latest stage-profiled frame.
    pub physics_ms_per_step: Option<f64>,
    pub presentation_fps: u32,
    /// Ticks per second when the clock is fixed-rate rather than frame-based.
    pub fixed_rate: Option<u32>,
}
/// "15 steps/frame ≈ 900 ticks/s at 60 fps", plus the measured physics cost when known.
pub fn physics_rate_readout(
    steps: u32,
    presentation_fps: u32,
    fixed_rate: Option<u32>,
    physics_ms_per_step: Option<f64>,
) -> String {
    if let Some(rate) = fixed_rate {
        return format!(
            "Fixed-rate clock: {rate} ticks/s; Physics Rate applies in frame-based mode"
        );
    }
    let mut text = format!(
        "{steps} steps/frame ≈ {} ticks/s at {presentation_fps} fps",
        u64::from(steps) * u64::from(presentation_fps)
    );
    if let Some(per_step) = physics_ms_per_step {
        text += &format!(
            "\n≈ {:.1} ms GPU physics per frame ({per_step:.3} ms/step measured)",
            per_step * f64::from(steps)
        );
    }
    text
}
/// Settings whose label takes the frame-rate band colour because they decide GPU cost.
fn costly(id: &str) -> bool {
    matches!(
        id,
        "world_size"
            | "physics_steps"
            | "motion_blur_samples"
            | "cell_style"
            | "cell_scale"
            | "cell_sharpen"
            | "cell_chromatic"
            | "cell_detailed"
            | "cell_fast"
    )
}
fn performance_label(ui: &Ui, id: &str, label: &str) -> egui::RichText {
    let text = egui::RichText::new(label);
    if costly(id) {
        if let Some(band) = ui
            .ctx()
            .data_mut(|d| d.get_temp::<crate::performance::Band>(egui::Id::new("performance band")))
        {
            if band != crate::performance::Band::Green {
                let [r, g, b] = band.label_color();
                return text.color(egui::Color32::from_rgb(r, g, b));
            }
        }
    }
    text
}

#[derive(Clone, Copy, Default)]
struct GateSession {
    forced: bool,
    held: bool,
}
fn active(ui: &Ui, r: &Response) -> bool {
    r.dragged()
        || r.is_pointer_button_down_on()
        || (r.has_focus()
            && ui.input(|i| {
                i.events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))
            }))
}
pub(crate) fn physics_speed_bar(ui: &mut Ui, value: &mut u32, fixed_rate: bool) -> Response {
    ui.push_id("dock physics speed", |ui| {
        ui.spacing_mut().slider_width = ui.available_width();
        let response = ui.add(
            egui::Slider::new(value, 1..=if fixed_rate { 7200 } else { 60 })
                .clamping(egui::SliderClamping::Edits)
                .logarithmic(fixed_rate)
                .show_value(false)
                .trailing_fill(true),
        );
        observe_performance_drag(ui.ctx(), &response);
        response.on_hover_text(format!(
            "Simulation speed: {value} {}. Drag to adjust.",
            if fixed_rate { "ticks/s" } else { "steps/frame" }
        ))
    })
    .inner
}
fn position(value: f64, lo: f64, hi: f64, curve: f64) -> f64 {
    ((value - lo) / (hi - lo)).clamp(0.0, 1.0).powf(1.0 / curve)
}
fn value_at(pos: f64, lo: f64, hi: f64, curve: f64) -> f64 {
    lo + (hi - lo) * pos.clamp(0.0, 1.0).powf(curve)
}
fn flip(value: f64, lo: f64, hi: f64, inverted: bool) -> f64 {
    if inverted { lo + hi - value } else { value }
}
fn off(value: f64, lo: f64, hi: f64, curve: f64, base: f64, integer: bool) -> bool {
    if integer {
        value.round() == base.round()
    } else {
        (position(value, lo, hi, curve) - position(base, lo, hi, curve)).abs() <= 1e-4
    }
}
/// Slider travel maps to the stored value only when a widget changes.
/// The readout displays real units, preserving imported values beyond UI bounds.
pub(crate) fn slider(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut f64,
    lo: f64,
    hi: f64,
    curve: f64,
    inverted: bool,
    integer: bool,
    help: &str,
) -> bool {
    ui.push_id(id, |ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.spacing_mut().interact_size.y = 24.0;
        let mut shown = flip(*value, lo, hi, inverted);
        let (number, label_id) = ui
            .horizontal(|ui| {
                let width = (ui.available_width() - 88.0).max(0.0);
                let label = ui
                    .allocate_ui_with_layout(
                        egui::vec2(width, ui.spacing().interact_size.y),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.set_min_width(width);
                            ui.add(
                                egui::Label::new(performance_label(ui, id, label))
                                    .wrap()
                                    .halign(egui::Align::Min),
                            )
                        },
                    )
                    .inner
                    .on_hover_text(help);
                let number = ui
                    .allocate_ui_with_layout(
                        egui::vec2(80.0, ui.spacing().interact_size.y),
                        egui::Layout::centered_and_justified(ui.layout().main_dir()),
                        |ui| {
                            let muted = ui.visuals().weak_text_color();
                            let inactive = &mut ui.visuals_mut().widgets.inactive;
                            inactive.bg_fill = egui::Color32::TRANSPARENT;
                            inactive.bg_stroke = egui::Stroke::NONE;
                            inactive.fg_stroke.color = muted;
                            ui.add(
                                egui::DragValue::new(&mut shown)
                                    .speed((hi - lo) / 500.0)
                                    .max_decimals(if integer { 0 } else { 6 }),
                            )
                        },
                    )
                    .inner
                    .on_hover_text(help)
                    .labelled_by(label.id);
                (number, label.id)
            })
            .inner;
        if number.changed() {
            *value = flip(
                if integer {
                    shown.round().clamp(lo, hi)
                } else {
                    shown.clamp(lo, hi)
                },
                lo,
                hi,
                inverted,
            );
        }
        let mut pos = position(flip(*value, lo, hi, inverted), lo, hi, curve);
        let r = ui
            .scope(|ui| {
                ui.spacing_mut().interact_size.y = 20.0;
                ui.spacing_mut().slider_width = ui.available_width();
                ui.add(egui::Slider::new(&mut pos, 0.0..=1.0).show_value(false))
                    .on_hover_text(help)
                    .labelled_by(label_id)
            })
            .inner;
        if r.changed() {
            let shown = value_at(pos, lo, hi, curve);
            *value = flip(
                if integer { shown.round() } else { shown },
                lo,
                hi,
                inverted,
            );
        }
        if matches!(id, "physics_steps" | "motion_blur_samples") {
            observe_performance_drag(ui.ctx(), &r);
            observe_performance_drag(ui.ctx(), &number);
        }
        active(ui, &r) || active(ui, &number)
    })
    .inner
}

fn f32_slider(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut f32,
    lo: f64,
    hi: f64,
    curve: f64,
    help: &str,
) {
    let mut v = *value as f64;
    slider(ui, id, label, &mut v, lo, hi, curve, false, false, help);
    *value = v as f32;
}
fn integer_slider(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut u32,
    lo: f64,
    hi: f64,
    help: &str,
) {
    let mut v = *value as f64;
    slider(ui, id, label, &mut v, lo, hi, 1.0, false, true, help);
    *value = v as u32;
}
fn gated(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut f64,
    lo: f64,
    hi: f64,
    curve: f64,
    inverted: bool,
    base: f64,
    integer: bool,
    help: &str,
) {
    let key = ui.make_persistent_id(("gate", id));
    let mut session = ui
        .ctx()
        .data_mut(|d| d.get_temp::<GateSession>(key).unwrap_or_default());
    let mut enabled = !off(*value, lo, hi, curve, base, integer) || session.held;
    if !enabled {
        let toggle = ui.checkbox(&mut enabled, label).on_hover_text(help);
        if toggle.changed() && enabled {
            *value = if integer {
                base.trunc() + 1.0
            } else {
                let p = position(base, lo, hi, curve);
                value_at(p + if p <= 0.5 { 2e-4 } else { -2e-4 }, lo, hi, curve)
            };
        }
    }
    if enabled {
        session.held = slider(ui, id, label, value, lo, hi, curve, inverted, integer, help);
    } else {
        session.held = false;
    }
    if !session.held && off(*value, lo, hi, curve, base, integer) {
        *value = base;
    }
    ui.ctx().data_mut(|d| d.insert_temp(key, session));
}
fn gated_integer(
    ui: &mut Ui,
    id: &str,
    label: &str,
    value: &mut u32,
    lo: f64,
    hi: f64,
    base: f64,
    help: &str,
) {
    let mut v = *value as f64;
    gated(ui, id, label, &mut v, lo, hi, 1.0, false, base, true, help);
    *value = v as u32;
}
fn choice(ui: &mut Ui, id: &str, label: &str, value: &mut u32, options: &[&str], help: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label(label).on_hover_text(help);
        egui::ComboBox::from_id_salt(id)
            .selected_text(options.get(*value as usize).copied().unwrap_or("Unknown"))
            .show_ui(ui, |ui| {
                for (i, s) in options.iter().enumerate() {
                    ui.selectable_value(value, i as u32, *s).on_hover_text(help);
                }
            })
            .response
            .on_hover_text(help);
    });
}
fn input(ui: &mut Ui, id: &str, label: &str, value: &mut f64, lo: f64, hi: f64, help: &str) {
    let key = ui.make_persistent_id(("input", id));
    let mut text = ui.ctx().data_mut(|d| {
        d.get_temp::<String>(key)
            .unwrap_or_else(|| value.to_string())
    });
    ui.vertical(|ui| {
        ui.label(performance_label(ui, id, label))
            .on_hover_text(help);
        let r = ui
            .add(egui::TextEdit::singleline(&mut text).desired_width(ui.available_width()))
            .on_hover_text(help);
        if (r.has_focus() || r.lost_focus()) && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let Ok(v) = text.parse::<f64>() {
                if v.is_finite() && (lo..=hi).contains(&v) {
                    *value = v;
                    r.surrender_focus();
                }
            }
        }
        if !r.has_focus() {
            text = value.to_string();
        }
    });
    ui.ctx().data_mut(|d| d.insert_temp(key, text));
}

pub fn project_panel(ui: &mut Ui, project: &mut Project, prefs: &mut Preferences) {
    ui.checkbox(&mut prefs.advanced_project, "Advanced")
        .on_hover_text("Show/Hide the advanced project settings");
    let advanced = prefs.advanced_project;
    // Loaded projects clear session-only holds; stable scoped IDs retain active gestures.
    let identity = ui.make_persistent_id("project gate identity");
    let current = (project.name.clone(), project.selected);
    let generation_id = ui.make_persistent_id("project gate generation");
    let previous = ui
        .ctx()
        .data_mut(|d| d.get_temp::<(String, usize)>(identity));
    let mut generation = ui
        .ctx()
        .data_mut(|d| d.get_temp::<u64>(generation_id).unwrap_or(0));
    if previous.as_ref() != Some(&current) {
        generation = generation.wrapping_add(1);
        ui.ctx().data_mut(|d| {
            d.insert_temp(identity, current);
            d.insert_temp(generation_id, generation);
        });
    }
    ui.push_id(generation,|ui| {
    egui::CollapsingHeader::new("Population").id_salt("Population").default_open(true).show(ui,|ui| {
        integer_slider(ui,"cohorts","Cohorts",&mut project.config_mut().cohorts,1.0,64.0,"How many groups the population is divided into. When mutation Scale > 0, each cohort gets its own mutation of the parent behavior.");
        if advanced {
        choice(ui,"boundary_conditions","Boundary Conditions",&mut project.world.boundary_conditions,&["Bounce","Wrap","Reset","Kill"],"What happens when a particle reaches the edge of the world: Bounce reflects it, reset returns it to starting position, wrap carries it around to the opposite edge, and kill destroys it.");
        }
        choice(ui,"initial_conditions","Initial Conditions",&mut project.config_mut().initial_conditions,&["Grid","Random","Center","Ring"],"How particles are arranged when the simulation starts: Grid and ring lay the cohorts out in a regular pattern, Random spreads them evenly, and Center starts them all in a dense clump in the middle.");
        ui.add_enabled(project.config().initial_conditions==0,egui::Checkbox::new(&mut project.config_mut().cohort_fences,"Cohort Fences")).on_hover_text("Cohort fences: When enabled, particles are forced to stay close to their initial locations (Grid only).");
        if advanced {
        gated(ui,"hazard_rate","Hazard Rate",&mut project.config_mut().hazard_rate,0.0,0.01,3.0,false,0.0,false,"Applies a small probability each tick for a particle to \"die\" and be \"reincarnated\" at its initial conditions");
        }
    });
    egui::CollapsingHeader::new("Sensing").id_salt("Sensors").default_open(true).show(ui,|ui| {
        slider(ui,"sensor_angle","Sensor Angle",&mut project.config_mut().sensor_angle,-1.0,1.0,1.0,false,false,"Defines how wide the angle is between a particle's two sensor locations. Values near 0.0 are looking straight ahead, values near 0.5 look to the left and right, while values near 1.0 look behind.");
        if advanced {
        gated(ui,"sensor_angle_jitter","Sensor Angle Jitter",&mut project.config_mut().sensor_angle_jitter,0.0,1.0,1.0,false,0.0,false,"Adds random, per-tick variation to each particle's sensor angle");
        }
        slider(ui,"sensor_distance","Sensor Distance",&mut project.config_mut().sensor_distance,0.0,5.0,1.0,false,false,"Defines how far away from a particle's center it reads the trail. Higher values tend to result in larger, more global patterns");
        if advanced {
        gated(ui,"sensor_distance_jitter","Sensor Distance Jitter",&mut project.config_mut().sensor_distance_jitter,0.0,1.0,1.0,false,0.0,false,"Adds random, per-tick variation to each particle's sensor distance. Distance jitter often produces softer edges and less noticeable grid-aligned aliasing");
        }
        if advanced {
        slider(ui,"sensor_gain","Sensor Gain",&mut project.config_mut().sensor_gain,0.0,8.0,1.0,false,false,"Controls sensitivity to vector flow. Current MLP/GRU models also observe occupancy and their own speed independently of this gain.");
        }
    });
    egui::CollapsingHeader::new("Movement").id_salt("Forces").default_open(true).show(ui,|ui| {
        if advanced {
        slider(ui,"global_force_mult","Global Force",&mut project.config_mut().global_force_mult,0.0,1.0,1.0,false,false,"Determines the strength with which particles turn quickly, accelerate, and strafe");
        }
        if advanced {
        slider(ui,"drag","Momentum",&mut project.config_mut().drag,0.0,1.0,1.0,false,false,"How much velocity carries over between ticks. At high values, particles will be able to build up momentum");
        }
        let key=ui.make_persistent_id(("gate","gravity"));let mut gravity=ui.ctx().data_mut(|d|d.get_temp::<GateSession>(key).unwrap_or_default());
        let c=project.config();let derived=c.gravity_strafe!=0.0 || c.gravity_force!=0.0 || c.gravity_trails!=0.0;
        if derived{gravity.forced=false;}
        let mut gravity_open=derived||gravity.forced||gravity.held;
        if ui.checkbox(&mut gravity_open,"Gravity").on_hover_text("Enables vertical/radial gravity controls").changed(){gravity.forced=gravity_open;gravity.held=false;if !gravity_open {let c=project.config_mut();c.gravity_strafe=0.0;c.gravity_force=0.0;c.gravity_trails=0.0;}}
        gravity.held=false;
        if gravity_open {
        gravity.held|=slider(ui,"gravity_strafe","Gravity (Strafe)",&mut project.config_mut().gravity_strafe,-1.0,1.0,1.0,false,false,"Applies a fixed offset to particles each frame, shifting them downward");
        }
        if advanced {
        if gravity_open {
        gravity.held|=slider(ui,"gravity_trails","Gravity (Trails)",&mut project.config_mut().gravity_trails,-1.0,1.0,1.0,false,false,"Biases what every particle senses, as though a trail lay downhill. Nothing is pushed -- each rule decides whether to follow that trail or flee it, so this can make a population fall or rise, and does nothing at all to one that ignores its sensors");
        }
        }
        if advanced {
        if gravity_open {
        gravity.held|=slider(ui,"gravity_force","Gravity (Force)",&mut project.config_mut().gravity_force,-1.0,1.0,1.0,false,false,"Applies a fixed force to particles each frame, accelerating them downward");
        }
        }
        if advanced {
        if gravity_open {
        ui.checkbox(&mut project.config_mut().radial_gravity,"Radial Gravity").on_hover_text("Makes gravity push particles towards the center of the simulation instead of straight down");
        }
        }
        ui.ctx().data_mut(|d|d.insert_temp(key,gravity));
    });
    if advanced {
    egui::CollapsingHeader::new("Shared Trails").id_salt("Trails").default_open(true).show(ui,|ui| {
        slider(ui,"trail_persistence","Trail Persistence",&mut project.world.trail_persistence,0.5,0.999,1.0,false,false,"Determines how long particle trails remain detectable. At high values, trails will spread widely and decay slowly.");
    });
    }
    egui::CollapsingHeader::new("Color").id_salt("Appearance").default_open(true).show(ui,|ui| {
        if advanced {
        slider(ui,"color_sensitivity","Color Sensitivity",&mut project.config_mut().color_sensitivity,-1.0,1.0,1.0,false,false,"Determines how sensitive the hue of a particle is to its brain outputs. At high values particle hue becomes chaotic and regions will become a random mix of hues: appearing pale/white. Pairs with Color Offset, which decides where on the wheel that response starts");
        }
        if advanced {
        slider(ui,"color_offset","Color Offset",&mut project.config_mut().color_offset,0.0,1.0,1.0,false,false,"Shifts this config’s hue around the colour wheel, before Color Sensitivity swings it. Lets each config sit in its own colour instead of every one fanning out from red");
        }
        ui.checkbox(&mut project.config_mut().color_by_cohort,"Color By Cohort (G)").on_hover_text("Assign each cohort a unique color instead of basing hue on brain outputs");
    });
    if advanced {
    egui::CollapsingHeader::new("Advanced").default_open(true).show(ui,|ui| {
        slider(ui,"axial_force","Axial Force",&mut project.config_mut().axial_force,-2.0,2.0,1.0,false,false,"Determines the relative strength of acceleration/braking forces and forward/backward strafing");
        slider(ui,"lateral_force","Lateral Force",&mut project.config_mut().lateral_force,-2.0,2.0,1.0,false,false,"Determines the relative strength of turning forces and left-right strafing");
        slider(ui,"strafe_power","Strafe Power",&mut project.config_mut().strafe_power,0.0,0.5,1.0,false,false,"Determines the relative strength of particle strafe, which directly shifts particle positions, bypassing momentum");
        gated(ui,"trail_diffusion","Trail Stiffness",&mut project.world.trail_diffusion,0.0,1.0,1.0,true,1.0,false,"Allows you to slow or stop the pace at which particle trails spread through the environment. At stiffness 1.0, trails do not diffuse at all, and simply decay over time.");
    });
    }
});
}

pub fn preferences_panel(ui: &mut Ui, prefs: &mut Preferences, hints: &PerformanceHints) {
    ui.ctx()
        .data_mut(|d| d.insert_temp(egui::Id::new("performance band"), hints.band));
    ui.checkbox(&mut prefs.advanced_preferences, "Advanced")
        .on_hover_text("Show advanced run and interaction settings");
    let advanced = prefs.advanced_preferences;
    egui::CollapsingHeader::new("Simulation").default_open(true).show(ui,|ui| {
        input(ui,"world_size","World Size",&mut prefs.world_size,0.05,27.0,"(Expensive) Determines the particle count and resolution of the trail map. Overall density -- Particles/Trail pixel -- is kept constant.");
        if let Ok(s) = crate::coords::sizing_for(prefs.world_size, prefs.canvas_aspect) {
            ui.small(format!("≈ {} particles · {} × {} trails", crate::performance::grouped(u64::from(s.entity_count)), s.canvas[0], s.canvas[1]));
        }
        if advanced {
        input(ui,"canvas_aspect","Canvas Aspect",&mut prefs.canvas_aspect,0.1,10.0,"Determines Width/Height ratio of simulation area.");
        }
        integer_slider(ui,"physics_steps","Physics Rate",&mut prefs.physics_steps,1.0,60.0,"(Expensive) Determines how many physics substeps are performed per frame. Higher values run the simulation faster.");
        ui.small(physics_rate_readout(prefs.physics_steps, hints.presentation_fps, hints.fixed_rate, hints.physics_ms_per_step));
    });
    egui::CollapsingHeader::new("Performance").default_open(true).show(ui,|ui| {
        gated_integer(ui,"motion_blur_samples","Motion Blur",&mut prefs.motion_blur_samples,1.0,16.0,1.0,"(Expensive) Renders multiple images at different substeps and blends them together. Results in a smoother, less noisy image");
        render_scale(ui, prefs);
        ui.checkbox(&mut prefs.show_fps_counter,"Show FPS Counter").on_hover_text("The quiet frame-rate readout above the playback controls. Click it to open Settings; hover for GPU timing and the full readout. Measurement pauses while the simulation is paused, recording, or exporting.\n\nThe number is capped at 60, the simulation's target frame rate.\n\nWorld Size, Physics Rate, Motion Blur, and costly Cells settings show yellow or red warnings when the frame rate falls behind.");
        {
            ui.checkbox(&mut prefs.show_physics_slider,"Show Physics Rate").on_hover_text("The quiet rate readout above the speed bar. Click it to open Settings; right-click to calibrate or cancel calibration.\n\nThe speed bar stays visible until X hides the controls.");
        }
    });
    if advanced {
        egui::CollapsingHeader::new("Behavior").default_open(true).show(ui,|ui| {
        ui.checkbox(&mut prefs.reset_on_behavior_change,"Reset on Behavior Change").on_hover_text("Resets the simulation whenever particles are given new behaviors");
        ui.checkbox(&mut prefs.one_click_selection,"1-Click Selection").on_hover_text("Allow a single click to bypass cohort selection and generate children immediately");
        choice(ui,"mobile_mode","Touch Layout",&mut prefs.mobile_mode,&["Auto","Always Touch","Always Desktop"],"Which layout to build: Auto picks the touch layout on touchscreens, including tablets. Change this only if the automatic choice is wrong for your device -- it takes effect after you reload the page.");
    });
    }
    if advanced {
        egui::CollapsingHeader::new("Archive").default_open(true).show(ui,|ui| {
        ui.checkbox(&mut prefs.strong_logging,"Strong Logging").on_hover_text("Record every project state you visit to a local archive, so the exploration can be studied later. Changes nothing on screen");
    });
    }
}

pub fn appearance_panel(
    ui: &mut Ui,
    prefs: &mut Preferences,
    hints: &PerformanceHints,
    allow_style_change: bool,
) {
    ui.ctx()
        .data_mut(|d| d.insert_temp(egui::Id::new("performance band"), hints.band));
    ui.checkbox(&mut prefs.advanced_preferences, "Advanced")
        .on_hover_text("Show advanced appearance settings");
    let advanced = prefs.advanced_preferences;
    egui::CollapsingHeader::new("Appearance").default_open(true).show(ui,|ui| {
        ui.add_enabled_ui(allow_style_change, |ui| {
            ui.horizontal_wrapped(|ui| {ui.label(performance_label(ui,"cell_style","Style")).on_hover_text("(Expensive) Cells redraws every particle as an integer-accumulated sprite and shades the image several times per pixel, costing several times the GPU time of Classic. Its cost grows with particle count and screen resolution.");ui.selectable_value(&mut prefs.cells.style,crate::cells::Style::Classic,"Classic");ui.selectable_value(&mut prefs.cells.style,crate::cells::Style::Cells,"Cells");});
        });
        if prefs.cells.style == crate::cells::Style::Cells {
            ui.horizontal_wrapped(|ui| {ui.label("Theme");ui.selectable_value(&mut prefs.cells.theme,crate::cells::Theme::Dark,"Dark");ui.selectable_value(&mut prefs.cells.theme,crate::cells::Theme::Light,"Light");});
            f32_slider(ui,"cell_scale","Cell Scale",&mut prefs.cells.scale,0.5,4.0,1.0,"(Expensive) Particle size in the cellular image; larger cells cover more pixels and take more GPU time. Does not change physics");
            f32_slider(ui,"cell_membranes","Cell Membranes",&mut prefs.cells.membranes,0.0,1.0,1.0,"Soft cell walls and shaded interiors revealed as you zoom in. 0 restores solid pigment. Uses the current particle colors.");
            f32_slider(ui,"cell_density","Density Response",&mut prefs.cells.density,0.0,1.0,1.0,"Crowded regions show smaller grains");
            f32_slider(ui,"cell_structure_threshold","Structure Threshold",&mut prefs.cells.structure_threshold,0.0,4.0,1.0,"Fade scattered grain below this local density. 0 = off; try 1–2. Denser strands and clusters remain; physics and Color Wash are unchanged.");
            f32_slider(ui,"cell_relief","Cell Relief",&mut prefs.cells.relief,0.0,0.9,1.0,"Bounded image displacement, in reference pixels");
            f32_slider(ui,"cell_wash","Color Wash",&mut prefs.cells.wash,0.0,1.0,1.0,"Broad unthresholded background color");
        }
    });
    egui::CollapsingHeader::new("Light & Detail").default_open(true).show(ui,|ui| {
        f32_slider(ui,"brightness","Brightness",&mut prefs.brightness,0.1,4.0,1.0,"Determines overall intensity of each particle");
        ui.checkbox(&mut prefs.bloom_enabled,"Bloom").on_hover_text("Adds a glowing halo effect to over-bright regions");
        if advanced {
            f32_slider(ui,"tonemap_softness","Tonemap Softness",&mut prefs.tonemap_softness,0.1,5.0,1.0,"Determines how aggressively the bright highlights are dimmed. High values have reduced contrast between bright and dim regions.");
            if prefs.bloom_enabled {
                f32_slider(ui,"bloom_threshold","Threshold",&mut prefs.bloom_threshold,0.0,2.0,1.0,"Determines the brightness above which bloom is applied");
                f32_slider(ui,"bloom_intensity","Intensity",&mut prefs.bloom_intensity,0.0,0.5,1.0,"Determines the strength of the bloom effect");
                f32_slider(ui,"bloom_radius","Radius",&mut prefs.bloom_radius,0.1,1.0,1.0,"Determines the size of the bloom halo");
            }
            if prefs.cells.style == crate::cells::Style::Cells {
                ui.checkbox(&mut prefs.cells.reference_palette,"Green / pink palette");
                f32_slider(ui,"cell_contrast","Cell Edge Contrast",&mut prefs.cells.contrast,0.0,2.0,1.0,"Adds soft relief to cell edges");
                f32_slider(ui,"cell_chromatic","Chromatic Separation",&mut prefs.cells.chromatic,0.0,0.5,1.0,"(Expensive) Subpixel color separation; UI stays sharp. Shades each pixel two extra times.");
                f32_slider(ui,"cell_sharpen","Cell Sharpening",&mut prefs.cells.sharpen,0.0,0.5,1.0,"(Expensive) Restrained sharpening before editor overlays. Shades each pixel four extra times, roughly tripling the Cells finishing cost.");
                f32_slider(ui,"cell_zoom","Zoom Compensation",&mut prefs.cells.zoom_response,0.2,1.0,1.0,"How sprite size responds to zoom");
                ui.add_enabled_ui(allow_style_change, |ui| {
                    ui.checkbox(&mut prefs.cells.fast,performance_label(ui,"cell_fast","Fast Cells")).on_hover_text("Faster rendering with small cell-edge differences and more GPU memory.");
                    ui.checkbox(&mut prefs.cells.detailed,performance_label(ui,"cell_detailed","Detailed density grid")).on_hover_text("(Expensive) Quadruples the density grid behind cell sizing.");
                });
            }
        }
    });
}

/// Id of the canvas's physical pixel size, published by the app each frame.
pub const PHYSICAL_CANVAS_ID: &str = "physical canvas";
/// Id of the working size actually rendered, after any GPU-limit fit.
pub const WORKING_CANVAS_ID: &str = "working canvas";

fn render_scale(ui: &mut Ui, prefs: &mut Preferences) {
    use crate::config::RenderScale;
    ui.horizontal_wrapped(|ui| {
        ui.label("Render Scale").on_hover_text(
            "(Expensive) Working resolution of the live view, relative to the screen's pixels. \
             Auto keeps Classic at full resolution and caps Cells at a 1080-pixel short side. \
             Lower scales are faster and softer. Screenshots use this resolution; video export keeps its own size.",
        );
        for scale in RenderScale::ALL {
            ui.selectable_value(&mut prefs.render_scale, scale, scale.label());
        }
    });
    let physical = ui
        .ctx()
        .data_mut(|d| d.get_temp::<[u32; 2]>(egui::Id::new(PHYSICAL_CANVAS_ID)));
    if let Some(physical) = physical {
        let cells = prefs.cells.style == crate::cells::Style::Cells;
        let requested = prefs.render_scale.size(physical, cells);
        let working = ui
            .ctx()
            .data_mut(|d| d.get_temp::<[u32; 2]>(egui::Id::new(WORKING_CANVAS_ID)))
            .filter(|w| cells && *w != requested && w.iter().zip(requested).all(|(a, b)| *a <= b))
            .unwrap_or(requested);
        let [w, h] = working;
        ui.small(format!(
            "Internal {w} × {h} of {} × {} screen pixels{}",
            physical[0],
            physical[1],
            if working != requested {
                " · limited by GPU buffer size"
            } else {
                ""
            }
        ));
    }
}

pub fn drawing_panel(ui: &mut Ui, prefs: &mut Preferences) {
    ui.checkbox(&mut prefs.advanced_drawing, "Advanced")
        .on_hover_text("Show/Hide the advanced drawing controls settings");
    egui::CollapsingHeader::new("Brush").default_open(true).show(ui, |ui| {
    f32_slider(
        ui,
        "draw_size",
        "Brush Size",
        &mut prefs.draw_size,
        0.001,
        0.25,
        2.5,
        "Determines the radius of the brush reticle",
    );
    f32_slider(
        ui,
        "draw_power",
        "Brush Power",
        &mut prefs.draw_power,
        0.1,
        5.0,
        1.0,
        "Determines the strength of the position offsets applied by shoving and drawn barriers",
    );
    if prefs.advanced_drawing {
        choice(
            ui,
            "brush_mode",
            "Brush Mode",
            &mut prefs.brush_mode,
            &["Out / Diverge", "In / Converge", "Stroke", "Fixed"],
            "Which way a stroke pushes. Out/Diverge and In/Converge push away from and toward the stroke; Stroke pushes along the direction you drag; Fixed pushes one way everywhere, set by Draw Angle",
        );
        f32_slider(
            ui,
            "draw_angle",
            "Draw Angle",
            &mut prefs.draw_angle,
            -std::f64::consts::PI,
            std::f64::consts::PI,
            1.0,
            "The direction the Fixed brush pushes. 0 is up. The arrow on the brush reticle turns with it",
        );
    }
    });
    if prefs.advanced_drawing {
        egui::CollapsingHeader::new("Painted Fields").default_open(true).show(ui, |ui| {
        f32_slider(
            ui,
            "walls_strength",
            "Walls Field Strength",
            &mut prefs.walls_strength,
            0.0,
            4.0,
            1.0,
            "Scales how hard the painted walls push, applied as they are read rather than as they are drawn -- so it retunes walls you painted earlier, and 0 mutes them without erasing them",
        );
        f32_slider(
            ui,
            "trails_strength",
            "Trails Field Strength",
            &mut prefs.trails_strength,
            0.0,
            4.0,
            1.0,
            "Scales how strongly the painted trails register on particle sensors. Like Walls Field Strength, it applies at read time, so a drawn pattern can be tuned or muted without redrawing it",
        );
        f32_slider(
            ui,
            "field_opacity",
            "Field Opacity",
            &mut prefs.field_opacity,
            0.0,
            1.0,
            1.0,
            "How strongly the painted walls and trails are drawn on screen",
        );
        ui.checkbox(&mut prefs.field_always_show,"Always Show Walls").on_hover_text("Keep the painted walls visible in every tool, including Trails. The Walls tool shows them either way");
        ui.checkbox(&mut prefs.trails_always_show,"Always Show Trails").on_hover_text("Keep the painted trails visible in every tool, including Walls. The Trails tool shows them either way");
    });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn displaying_controls_preserves_imported_values_without_input() {
        let mut project = Project::default();
        project.config_mut().hazard_rate = 0.02;
        project.config_mut().sensor_distance = 7.0;
        project.config_mut().strafe_power = 0.8;
        project.config_mut().gravity_force = 1.5;
        project.world.trail_diffusion = -0.25;
        let mut prefs = Preferences {
            advanced_project: true,
            advanced_preferences: true,
            advanced_drawing: true,
            physics_steps: 80,
            motion_blur_samples: 31,
            draw_size: 0.4,
            ..Default::default()
        };
        prefs.cells.style = crate::cells::Style::Cells;
        prefs.cells.scale = 5.0;
        prefs.cells.fast = false;
        let before_project = project.clone();
        let before_prefs = prefs.clone();
        let hints = PerformanceHints {
            band: crate::performance::Band::Green,
            physics_ms_per_step: None,
            presentation_fps: 60,
            fixed_rate: None,
        };
        let ctx = egui::Context::default();
        let mut fixed_rate = 7200;
        for width in [336.0, 720.0] {
            for _ in 0..2 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 10000.0),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        ui.push_id("project", |ui| project_panel(ui, &mut project, &mut prefs));
                        ui.push_id("run", |ui| preferences_panel(ui, &mut prefs, &hints));
                        ui.push_id("appearance", |ui| {
                            appearance_panel(ui, &mut prefs, &hints, true)
                        });
                        ui.push_id("drawing", |ui| drawing_panel(ui, &mut prefs));
                        physics_speed_bar(ui, &mut prefs.physics_steps, false);
                        physics_speed_bar(ui, &mut fixed_rate, true);
                    },
                );
                output.textures_delta.clear();
                assert_eq!(project, before_project);
                assert_eq!(prefs, before_prefs);
                assert_eq!(fixed_rate, 7200);
            }
        }
    }
    #[test]
    fn physics_rate_readout_reports_ticks_and_measured_cost() {
        assert_eq!(
            physics_rate_readout(15, 60, None, None),
            "15 steps/frame ≈ 900 ticks/s at 60 fps"
        );
        assert_eq!(
            physics_rate_readout(15, 60, None, Some(0.332)),
            "15 steps/frame ≈ 900 ticks/s at 60 fps\n≈ 5.0 ms GPU physics per frame (0.332 ms/step measured)"
        );
        assert!(
            physics_rate_readout(15, 60, Some(300), Some(0.3)).starts_with("Fixed-rate clock: 300")
        );
        assert!(costly("cell_sharpen") && costly("world_size") && !costly("brightness"));
    }
    #[test]
    fn hazard_gate_zone_is_measured_in_cubed_travel() {
        assert!(off(0.01 * 0.5e-4_f64.powi(3), 0.0, 0.01, 3.0, 0.0, false));
        assert!(!off(0.01 * 2e-4_f64.powi(3), 0.0, 0.01, 3.0, 0.0, false));
        assert!((value_at(0.5, 0.0, 0.01, 3.0) - 0.00125).abs() < 1e-15);
    }
    #[test]
    fn stiffness_inverts_display_and_blur_off_is_one() {
        assert_eq!(flip(1.0, 0.0, 1.0, true), 0.0);
        assert_eq!(flip(0.0, 0.0, 1.0, true), 1.0);
        assert!(off(1.0, 0.0, 1.0, 1.0, 1.0, false));
        assert!(!off(0.9998, 0.0, 1.0, 1.0, 1.0, false));
        assert!(off(1.0, 1.0, 16.0, 1.0, 1.0, true));
        assert!(!off(2.0, 1.0, 16.0, 1.0, 1.0, true));
        assert!((value_at(0.5, 0.001, 0.25, 2.5) - 0.045017397).abs() < 1e-8);
    }
}
