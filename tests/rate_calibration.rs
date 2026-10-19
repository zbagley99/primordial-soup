use primordial_soup::rate_calibration::{AutoCalibration, Probe, SearchStep, next_step};
use serde::Deserialize;

#[test]
fn source_decisions_match_frozen_synthetic_machines_and_bounds() {
    #[derive(Deserialize)]
    struct Case {
        history: Vec<Probe>,
        lo: u32,
        hi: u32,
        expected: SearchStep,
    }
    let value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/rate.fixture.json")).unwrap();
    let cases: Vec<Case> = serde_json::from_value(value["cases"].clone()).unwrap();
    assert!(cases.len() > 200);
    for c in cases {
        assert_eq!(next_step(&c.history, c.lo, c.hi), c.expected);
    }
}
#[test]
fn settling_outliers_and_cancel_preserve_the_original_rate() {
    let mut run = AutoCalibration::new(20, 1, 60);
    for _ in 0..5 {
        run.on_frame(400.0);
    }
    for value in [0.0, -1.0, 500.0, 1000.0, f64::INFINITY, f64::NAN] {
        run.on_frame(value);
    }
    for _ in 0..14 {
        run.on_frame(16.7);
    }
    assert_eq!(run.probe_count(), 0);
    run.on_frame(16.7);
    assert_eq!(run.probe_count(), 1);
    assert_eq!(run.rate(), 34);
    assert!(!run.finished());
    assert_eq!(run.cancel(), 20);
    assert_eq!(run.result(), None);
    run.on_frame(1.0);
    assert_eq!(run.rate(), 20);
}
#[test]
fn affordable_ceiling_commits_without_a_margin_and_cannot_be_cancelled() {
    let mut run = AutoCalibration::new(60, 1, 60);
    for _ in 0..20 {
        run.on_frame(16.7);
    }
    assert!(run.finished());
    assert_eq!(run.result(), Some(60));
    assert_eq!(run.cancel(), 60);
}

#[test]
fn cancellation_restores_a_preexisting_rate_above_the_slider_bounds() {
    let mut run = AutoCalibration::new(120, 1, 60);
    assert_eq!(run.rate(), 60);
    assert_eq!(run.cancel(), 120);
    assert_eq!(run.result(), None);
}
