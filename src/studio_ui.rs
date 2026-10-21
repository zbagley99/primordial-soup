use eframe::egui::{self, Color32, FontId, Rect, Response, Stroke, Vec2};

pub const PANEL: Color32 = Color32::from_rgb(21, 21, 24);
pub const LINE: Color32 = Color32::from_rgb(46, 46, 51);
pub const SPEED_ROW_HEIGHT: f32 = 32.0;
pub const COHORT_BUTTON_SIZE: f32 = 44.0;
pub const TEXT: Color32 = Color32::from_rgb(236, 236, 239);
pub const MUTED: Color32 = Color32::from_rgb(146, 146, 157);
pub const ACCENT: Color32 = Color32::from_rgb(210, 212, 222);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum InspectorTab {
    #[default]
    World,
    Appearance,
    Settings,
    Brush,
}

impl InspectorTab {
    pub fn label(self) -> &'static str {
        match self {
            Self::World => "World",
            Self::Appearance => "Look",
            Self::Settings => "Settings",
            Self::Brush => "Brush",
        }
    }
}

pub struct Layout {
    pub margin: f32,
    pub narrow: bool,
    pub compact: bool,
    pub dock: Rect,
    pub cohorts: Rect,
    pub identity: Rect,
    pub inspector: Rect,
    pub dialogs: Rect,
}

impl Layout {
    pub fn new(view: Rect, inspector_open: bool, readout_visible: bool) -> Self {
        let narrow = view.width() < 720.0;
        let compact = view.width() < 980.0 || view.height() < 540.0;
        let margin = if narrow { 12.0 } else { 20.0 };
        let available = (view.width() - margin * 2.0).max(0.0);
        let identity = Rect::from_min_size(
            view.min + Vec2::splat(margin),
            egui::vec2(230.0f32.min(available), 36.0),
        );
        let inspector_width = 320.0f32.min(available);
        let inspector_left = view.right() - margin - inspector_width;
        let readout_height = if readout_visible { 22.0 } else { 0.0 };
        let dock_height = if narrow { 56.0 } else { 48.0 } + readout_height + SPEED_ROW_HEIGHT;
        let scene_left = view.left() + margin;
        let scene_right = if inspector_open && !narrow {
            inspector_left - margin
        } else {
            view.right() - margin
        };
        let scene_width = (scene_right - scene_left).max(0.0);
        let dock_width = (if narrow { 280.0f32 } else { 336.0 }).min(scene_width);
        let cohort_width = 280.0f32.min(scene_width);
        let cohort_height = COHORT_BUTTON_SIZE + 12.0;
        let gap = 12.0;
        let beside = !narrow && scene_width >= dock_width + gap + cohort_width;
        let group_width = if beside {
            cohort_width + gap + dock_width
        } else {
            dock_width
        };
        let group_left = (scene_left + scene_right - group_width) * 0.5;
        let dock = Rect::from_min_size(
            egui::pos2(
                group_left + if beside { cohort_width + gap } else { 0.0 },
                view.bottom() - margin - dock_height,
            ),
            egui::vec2(dock_width, dock_height),
        );
        let cohorts = Rect::from_min_size(
            egui::pos2(
                if beside {
                    group_left
                } else {
                    dock.center().x - cohort_width * 0.5
                },
                if beside {
                    dock.bottom() - cohort_height
                } else {
                    dock.top() - gap - cohort_height
                },
            ),
            egui::vec2(cohort_width, cohort_height),
        );
        let controls_top = dock.top().min(cohorts.top());
        let top = view.top() + margin;
        let bottom = if narrow {
            controls_top - margin
        } else {
            view.bottom() - margin
        };
        let inspector = Rect::from_min_size(
            egui::pos2(inspector_left, top),
            egui::vec2(inspector_width, (bottom - top).max(0.0)),
        );
        Self {
            margin,
            narrow,
            compact,
            dock,
            cohorts,
            identity,
            inspector,
            dialogs: Rect::from_min_max(
                view.min + Vec2::splat(margin),
                egui::pos2(view.right() - margin, controls_top - margin),
            ),
        }
    }
}

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.set_visuals(egui::Visuals::dark());
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style
            .text_styles
            .insert(egui::TextStyle::Heading, FontId::proportional(17.0));
        style
            .text_styles
            .insert(egui::TextStyle::Body, FontId::proportional(13.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, FontId::proportional(13.0));
        style
            .text_styles
            .insert(egui::TextStyle::Small, FontId::proportional(11.0));
        style
            .text_styles
            .insert(egui::TextStyle::Monospace, FontId::monospace(12.0));
        style.spacing.item_spacing = egui::vec2(6.0, 7.0);
        style.spacing.button_padding = egui::vec2(8.0, 5.0);
        style.spacing.interact_size.y = 32.0;
        style.spacing.slider_width = 180.0;
        let v = &mut style.visuals;
        v.panel_fill = PANEL;
        v.window_fill = PANEL;
        v.window_stroke = Stroke::new(1.0, LINE);
        v.window_corner_radius = 8.into();
        v.menu_corner_radius = 8.into();
        v.window_shadow = egui::epaint::Shadow {
            offset: [0, 3],
            blur: 10,
            spread: 0,
            color: Color32::from_black_alpha(45),
        };
        v.popup_shadow = v.window_shadow;
        v.weak_text_color = Some(MUTED);
        v.extreme_bg_color = Color32::from_rgb(13, 13, 15);
        v.faint_bg_color = Color32::from_rgb(26, 26, 30);
        v.selection.bg_fill = Color32::from_rgb(52, 52, 60);
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.hyperlink_color = ACCENT;
        v.text_cursor.blink = false;
        for widget in [
            &mut v.widgets.noninteractive,
            &mut v.widgets.inactive,
            &mut v.widgets.hovered,
            &mut v.widgets.active,
            &mut v.widgets.open,
        ] {
            widget.corner_radius = 6.into();
            widget.fg_stroke = Stroke::new(1.0, TEXT);
            widget.bg_stroke = Stroke::new(1.0, LINE);
        }
        v.widgets.noninteractive.bg_fill = PANEL;
        v.widgets.noninteractive.fg_stroke.color = MUTED;
        v.widgets.inactive.bg_fill = Color32::from_rgb(29, 29, 33);
        v.widgets.inactive.weak_bg_fill = Color32::from_rgb(29, 29, 33);
        v.widgets.hovered.bg_fill = Color32::from_rgb(38, 38, 44);
        v.widgets.hovered.weak_bg_fill = v.widgets.hovered.bg_fill;
        v.widgets.hovered.bg_stroke.color = Color32::from_rgb(83, 83, 94);
        v.widgets.active.bg_fill = Color32::from_rgb(49, 49, 57);
        v.widgets.active.weak_bg_fill = v.widgets.active.bg_fill;
        v.widgets.active.bg_stroke.color = ACCENT;
        v.widgets.open = v.widgets.active;
    });
}

pub fn surface() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(8)
        .inner_margin(12)
}

pub fn window<'open>(title: &'static str, bounds: Rect) -> egui::Window<'open> {
    egui::Window::new(title)
        .order(egui::Order::Foreground)
        .constrain_to(bounds)
        .max_width((bounds.width() - 28.0).max(160.0))
        .max_height((bounds.height() - 64.0).max(64.0))
        .vscroll(true)
}

pub fn dialog_controls(ui: &mut egui::Ui) {
    let width = ui.available_width();
    let style = ui.style_mut();
    style.wrap_mode = Some(egui::TextWrapMode::Wrap);
    style.spacing.slider_width = (width - 160.0).clamp(64.0, 180.0);
    style.spacing.text_edit_width = width;
}

pub fn menu<R>(
    ui: &mut egui::Ui,
    label: &str,
    popup_top: f32,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> Response {
    let response = ui.add(egui::Button::new(label).frame_when_inactive(false));
    show_menu(ui, &response, popup_top, content);
    response
}

pub fn icon_menu<R>(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    width: f32,
    popup_top: f32,
    content: impl FnOnce(&mut egui::Ui) -> R,
) -> Response {
    let response = action(ui, icon, label, width, false);
    show_menu(ui, &response, popup_top, content);
    response
}

fn show_menu<R>(
    ui: &egui::Ui,
    response: &Response,
    popup_top: f32,
    content: impl FnOnce(&mut egui::Ui) -> R,
) {
    let config = egui::containers::menu::MenuConfig::new()
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
    egui::Popup::menu(response)
        .anchor(Rect::from_min_max(
            egui::pos2(response.rect.left(), popup_top),
            response.rect.max,
        ))
        .align(egui::emath::RectAlign::TOP_START)
        .gap(6.0)
        .close_behavior(config.close_behavior)
        .style(config.style.clone())
        .info(
            egui::UiStackInfo::new(egui::UiKind::Menu)
                .with_tag_value(egui::containers::menu::MenuConfig::MENU_CONFIG_TAG, config),
        )
        .show(content);
    if egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(response)) {
        ui.painter().rect_stroke(
            response.rect,
            6,
            Stroke::new(1.0, ACCENT),
            egui::StrokeKind::Inside,
        );
    }
}

#[derive(Clone, Copy)]
pub enum Icon {
    Play,
    Pause,
    Select,
    Shove,
    Walls,
    Trails,
    Settings,
    Close,
    Step,
    More,
}

fn draw_icon(ui: &egui::Ui, rect: Rect, icon: Icon, color: Color32) {
    let p = |x, y| rect.min + rect.size() * egui::vec2(x, y);
    let stroke = Stroke::new(1.5, color);
    let line = |a, b| {
        ui.painter().line_segment([a, b], stroke);
    };
    match icon {
        Icon::More => {
            for x in [0.2, 0.5, 0.8] {
                ui.painter().circle_filled(p(x, 0.5), 1.6, color);
            }
        }
        Icon::Play | Icon::Step => {
            ui.painter().add(egui::Shape::convex_polygon(
                vec![p(0.25, 0.15), p(0.8, 0.5), p(0.25, 0.85)],
                color,
                Stroke::NONE,
            ));
            if matches!(icon, Icon::Step) {
                line(p(0.9, 0.15), p(0.9, 0.85));
            }
        }
        Icon::Pause => {
            line(p(0.3, 0.15), p(0.3, 0.85));
            line(p(0.7, 0.15), p(0.7, 0.85));
        }
        Icon::Close => {
            line(p(0.25, 0.25), p(0.75, 0.75));
            line(p(0.75, 0.25), p(0.25, 0.75));
        }
        Icon::Select => {
            ui.painter().add(egui::Shape::closed_line(
                vec![p(0.2, 0.1), p(0.82, 0.55), p(0.53, 0.6), p(0.38, 0.88)],
                stroke,
            ));
        }
        Icon::Shove => {
            for (a, b) in [
                ((0.1, 0.5), (0.9, 0.5)),
                ((0.5, 0.1), (0.5, 0.9)),
                ((0.1, 0.5), (0.3, 0.3)),
                ((0.1, 0.5), (0.3, 0.7)),
                ((0.9, 0.5), (0.7, 0.3)),
                ((0.9, 0.5), (0.7, 0.7)),
            ] {
                line(p(a.0, a.1), p(b.0, b.1));
            }
        }
        Icon::Walls => {
            for x in [0.2, 0.5, 0.8] {
                line(p(x, 0.15), p(x, 0.85));
            }
            line(p(0.2, 0.5), p(0.8, 0.5));
        }
        Icon::Trails => {
            ui.painter().add(egui::Shape::line(
                vec![
                    p(0.1, 0.75),
                    p(0.2, 0.35),
                    p(0.4, 0.25),
                    p(0.6, 0.7),
                    p(0.8, 0.65),
                    p(0.9, 0.2),
                ],
                stroke,
            ));
        }
        Icon::Settings => {
            for (y, x) in [(0.2, 0.35), (0.5, 0.7), (0.8, 0.4)] {
                line(p(0.1, y), p(0.9, y));
                ui.painter().circle_filled(p(x, y), 2.0, color);
            }
        }
    }
}

pub fn action(ui: &mut egui::Ui, icon: Icon, label: &str, width: f32, primary: bool) -> Response {
    let height = if ui.ctx().content_rect().width() < 720.0 {
        44.0
    } else {
        36.0
    };
    let response = ui.add_sized([width, height], egui::Button::new("").frame(false));
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
    });
    let enabled = ui.is_enabled();
    let fill = if primary && enabled {
        ACCENT
    } else if enabled && response.is_pointer_button_down_on() {
        Color32::from_rgb(49, 49, 57)
    } else if enabled && response.hovered() {
        Color32::from_rgb(38, 38, 44)
    } else {
        Color32::TRANSPARENT
    };
    let stroke = if enabled && response.hovered() && !primary {
        Stroke::new(1.0, LINE)
    } else {
        Stroke::NONE
    };
    ui.painter()
        .rect(response.rect, 6, fill, stroke, egui::StrokeKind::Inside);
    if response.has_focus() {
        ui.painter().rect_stroke(
            response.rect.expand(2.0),
            8,
            Stroke::new(1.0, ACCENT),
            egui::StrokeKind::Inside,
        );
    }
    let color = if !ui.is_enabled() {
        MUTED.gamma_multiply(0.5)
    } else if primary {
        PANEL
    } else {
        TEXT
    };
    let text = !label.is_empty() && width > 48.0;
    let icon_rect = Rect::from_center_size(
        egui::pos2(
            if text {
                response.rect.left() + 18.0
            } else {
                response.rect.center().x
            },
            response.rect.center().y,
        ),
        Vec2::splat(16.0),
    );
    draw_icon(ui, icon_rect, icon, color);
    if text {
        ui.painter().text(
            egui::pos2(response.rect.left() + 34.0, response.rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            FontId::proportional(13.0),
            color,
        );
    }
    response
}

pub(crate) fn cohort_button(ui: &mut egui::Ui, count: u32, selected: bool) -> Response {
    let size = COHORT_BUTTON_SIZE;
    let response = ui.add_sized([size, size], egui::Button::new("").frame(false));
    let label = format!("{count} {}", if count == 1 { "cohort" } else { "cohorts" });
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), selected, &label)
    });
    let color = if !ui.is_enabled() {
        MUTED.gamma_multiply(0.4)
    } else if selected || response.hovered() {
        TEXT
    } else {
        MUTED
    };
    let rect = Rect::from_center_size(response.rect.center(), Vec2::splat(32.0));
    let fill = if ui.is_enabled() && (selected || response.hovered()) {
        Color32::from_rgb(38, 38, 44)
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect(
        rect,
        7,
        fill,
        Stroke::new(1.2, color),
        egui::StrokeKind::Inside,
    );
    let side = count.isqrt();
    let grid = rect.shrink(6.0);
    let radius = match side {
        1 => 4.5,
        2 => 3.1,
        _ => 1.6,
    };
    for y in 0..side {
        for x in 0..side {
            let center = grid.min
                + grid.size()
                    * egui::vec2(
                        (x as f32 + 0.5) / side as f32,
                        (y as f32 + 0.5) / side as f32,
                    );
            ui.painter()
                .circle_stroke(center, radius, Stroke::new(1.1, color));
        }
    }
    if response.has_focus() {
        ui.painter().rect_stroke(
            rect.expand(3.0),
            9,
            Stroke::new(1.0, ACCENT),
            egui::StrokeKind::Inside,
        );
    }
    response.on_hover_text(format!("{label} · restart the world in a grid"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimal_chrome_fits_viewport_and_keeps_dock_clear_of_inspector() {
        for size in [
            [320.0, 568.0],
            [390.0, 844.0],
            [844.0, 390.0],
            [1440.0, 900.0],
        ] {
            let view = Rect::from_min_size(egui::Pos2::ZERO, size.into());
            for open in [false, true] {
                let layout = Layout::new(view, open, true);
                assert!(view.contains_rect(layout.dock));
                assert!(view.contains_rect(layout.cohorts));
                assert!(view.contains_rect(layout.identity));
                assert!(view.contains_rect(layout.inspector));
                assert!(view.contains_rect(layout.dialogs));
                assert!(!layout.dialogs.intersects(layout.dock));
                assert!(!layout.dialogs.intersects(layout.cohorts));
                assert!(!layout.cohorts.intersects(layout.dock));
                if layout.cohorts.right() < layout.dock.left() {
                    assert_eq!(layout.cohorts.bottom(), layout.dock.bottom());
                }
                assert!(layout.inspector.width() <= 320.0);
                assert!(layout.dock.width() <= 344.0);
                if open {
                    assert!(!layout.dock.intersects(layout.inspector));
                    assert!(!layout.cohorts.intersects(layout.inspector));
                } else {
                    assert!(!layout.dock.intersects(layout.identity));
                    assert_eq!(
                        layout.dock.union(layout.cohorts).center().x,
                        view.center().x
                    );
                    if size[0] == 320.0 {
                        assert!(layout.cohorts.bottom() < layout.dock.top());
                    } else if size[0] == 1440.0 {
                        assert!(layout.cohorts.right() < layout.dock.left());
                    }
                }
            }
        }
    }
}
