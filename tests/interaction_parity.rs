use primordial_soup::interaction::{
    LINE_STROKE_GAIN, SHOVE_GAIN, SHOVE_RATE_EXPONENT, SHOVE_REFERENCE_STEPS, Stroke, StrokeInput,
    StrokeState, TRAILS_FIELD_GAIN, WALLS_FIELD_GAIN, should_confirm, shove_radius, shove_strength,
};
fn input(uv: [f32; 2]) -> StrokeInput {
    StrokeInput {
        uv,
        ..Default::default()
    }
}

#[test]
fn shove_matches_frozen_javascript_f32_upload_values() {
    // Frozen from shoveCommands.ts's exact JavaScript expression, narrowed as
    // the source Float32Array uniform upload does (2026-10-14).
    let cases: [(u32, f32); 6] = [
        (0, 0.0042728702537715435),
        (1, 0.0042728702537715435),
        (5, 0.0012778862146660686),
        (10, 0.0007598357042297721),
        (30, 0.00033333332976326346),
        (60, 0.00019820118905045092),
    ];
    for (steps, expected) in cases {
        assert_eq!(
            shove_strength(2.5, steps, false).to_bits(),
            expected.to_bits(),
            "steps={steps}"
        );
        assert_eq!(
            shove_strength(2.5, steps, true).to_bits(),
            (-expected).to_bits()
        );
    }
    assert_eq!(SHOVE_GAIN, 0.004);
    assert_eq!(SHOVE_REFERENCE_STEPS, 30.0);
    assert_eq!(SHOVE_RATE_EXPONENT, 0.75);
    assert_eq!(shove_radius(0.03125), 0.0625);
    assert!(shove_strength(2.5, 1, false) > shove_strength(2.5, 30, false));
    assert!(shove_strength(2.5, 30, false) > shove_strength(2.5, 60, false));
}
#[test]
fn painting_uniform_power_and_layer_gain_contract() {
    // strafeUniforms.ts lane9 packs drawPower unchanged. Shader applies
    // 0.01*(draw_power/5)*kernel*line_gain/draw_size independently.
    assert_eq!(WALLS_FIELD_GAIN, 0.01);
    assert_eq!(TRAILS_FIELD_GAIN, 0.001);
    // Current fieldLayer.ts uses1 despite earlier comments about boosted lines.
    assert_eq!(LINE_STROKE_GAIN, 1.0);
}
#[test]
fn freehand_release_starts_a_fresh_splat_and_left_wins() {
    let mut s = StrokeState::default();
    let a = s
        .update(StrokeInput {
            left_dragging: true,
            ..input([0.1, 0.9])
        })
        .unwrap();
    assert_eq!(
        a,
        Stroke {
            uv: [0.1, 0.9],
            previous: [0.1, 0.9],
            erase: false,
            is_line: false
        }
    );
    let b = s
        .update(StrokeInput {
            left_dragging: true,
            right_dragging: true,
            ..input([0.9, 0.1])
        })
        .unwrap();
    assert_eq!(b.previous, a.uv);
    assert!(!b.erase);
    assert!(s.update(input([0.5, 0.5])).is_none());
    assert_eq!(s, StrokeState::default());
    let c = s
        .update(StrokeInput {
            right_dragging: true,
            ..input([0.8, 0.8])
        })
        .unwrap();
    assert!(c.erase);
    assert_eq!(c.previous, c.uv);
    assert!(
        StrokeState::default()
            .update(StrokeInput {
                left_pressed: true,
                ..input([0.2, 0.2])
            })
            .is_none()
    );
}
#[test]
fn shift_arms_then_click_commits_and_chains_without_smearing() {
    // The literal UV endpoints are the reference drawingCommands.test.ts cases.
    let mut s = StrokeState::default();
    assert!(
        s.update(StrokeInput {
            shift: true,
            ..input([0.1, 0.9])
        })
        .is_none()
    );
    assert_eq!(s.anchor, Some([0.1, 0.9]));
    assert_eq!(s.previous, None);
    let line = s
        .update(StrokeInput {
            shift: true,
            left_pressed: true,
            left_dragging: true,
            ..input([0.9, 0.1])
        })
        .unwrap();
    assert_eq!(
        line,
        Stroke {
            uv: [0.9, 0.1],
            previous: [0.1, 0.9],
            erase: false,
            is_line: true
        }
    );
    assert_eq!(s.previous, None);
    assert_eq!(s.anchor, Some(line.uv));
    assert!(
        s.update(StrokeInput {
            shift: true,
            left_dragging: true,
            ..input([0.8, 0.8])
        })
        .is_none()
    );
    let erase = s
        .update(StrokeInput {
            shift: true,
            right_pressed: true,
            ..input([0.3, 0.4])
        })
        .unwrap();
    assert!(erase.erase && erase.is_line);
    assert_eq!(erase.previous, line.uv);
    let both = s
        .update(StrokeInput {
            shift: true,
            left_pressed: true,
            right_pressed: true,
            ..input([0.6, 0.6])
        })
        .unwrap();
    assert!(!both.erase);
    assert_eq!(both.previous, erase.uv);
    assert!(s.update(input([0.5, 0.5])).is_none());
    assert_eq!(s, StrokeState::default());
}
#[test]
fn shift_mid_drag_waits_for_release_before_taking_an_anchor() {
    for right in [false, true] {
        let mut s = StrokeState {
            previous: Some([0.1, 0.1]),
            anchor: None,
        };
        assert!(
            s.update(StrokeInput {
                shift: true,
                left_dragging: !right,
                right_dragging: right,
                ..input([0.5, 0.5])
            })
            .is_none()
        );
        assert_eq!(s.previous, Some([0.5, 0.5]));
        assert_eq!(s.anchor, None);
        assert!(
            s.update(StrokeInput {
                shift: true,
                ..input([0.7, 0.2])
            })
            .is_none()
        );
        assert_eq!(s.previous, None);
        assert_eq!(s.anchor, Some([0.7, 0.2]));
    }
}
#[test]
fn confirmation_needs_the_same_cohort_not_merely_some_highlight() {
    assert!(!should_confirm(None, 3.0, false));
    assert!(!should_confirm(Some(2.0), 3.0, false));
    assert!(should_confirm(Some(3.0), 3.0, false));
    assert!(should_confirm(None, 3.0, true));
    assert!(should_confirm(Some(2.0), 3.0, true));
    for picked in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!should_confirm(Some(picked), picked, false));
        assert!(!should_confirm(Some(2.0), picked, true));
    }
    assert!(!should_confirm(Some(f32::NAN), 3.0, false));
}
#[test]
fn global_space_command_does_not_also_click_a_focused_button() {
    use eframe::egui::{Context, Event, Key, Modifiers, RawInput};
    let ctx = Context::default();
    let mut button_id = None;
    let mut output = ctx.run_ui(RawInput::default(), |ui| {
        button_id = Some(ui.button("End Video").id);
    });
    output.textures_delta.clear();
    ctx.memory_mut(|m| m.request_focus(button_id.unwrap()));
    let input = RawInput {
        events: vec![Event::Key {
            key: Key::Space,
            physical_key: Some(Key::Space),
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        }],
        ..Default::default()
    };
    let mut command_seen = false;
    let mut clicked = false;
    let mut output = ctx.run_ui(input, |ui| {
        command_seen |= primordial_soup::interaction::command_keys(ui.ctx())
            .iter()
            .any(|(key, _)| *key == Key::Space);
        clicked |= ui.button("End Video").clicked();
    });
    output.textures_delta.clear();
    assert!(command_seen);
    assert!(
        !clicked,
        "Global resume must not activate a focused End Video button"
    );
}

#[test]
fn commands_keep_event_time_shift_when_released_or_pressed_before_the_frame() {
    use eframe::egui::{Context, Event, Key, Modifiers, RawInput};
    for (event_shift, frame_shift) in [(true, false), (false, true)] {
        for key in [Key::V, Key::Z, Key::C, Key::P] {
            let event_modifiers = Modifiers {
                shift: event_shift,
                ..Modifiers::NONE
            };
            let frame_modifiers = Modifiers {
                shift: frame_shift,
                ..Modifiers::NONE
            };
            let input = RawInput {
                events: vec![
                    Event::Key {
                        key,
                        physical_key: Some(key),
                        pressed: true,
                        repeat: false,
                        modifiers: event_modifiers,
                    },
                    Event::ModifiersChanged(frame_modifiers),
                ],
                ..Default::default()
            };
            let ctx = Context::default();
            let mut seen = Vec::new();
            let mut output = ctx.run_ui(input, |ui| {
                assert_eq!(ui.input(|i| i.modifiers.shift), frame_shift);
                seen.extend(primordial_soup::interaction::command_keys(ui.ctx()));
                assert!(!ui.input(|i| i.key_pressed(key)));
            });
            output.textures_delta.clear();
            assert_eq!(seen, vec![(key, event_modifiers)]);
        }
    }
}
