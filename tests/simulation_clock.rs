use primordial_soup::simulation::{
    SimulationClock, SimulationMode, SimulationSettings, repaint_after,
};
use std::time::Duration;

fn fixed(ticks_per_second: u32) -> SimulationSettings {
    SimulationSettings {
        mode: SimulationMode::FixedRate,
        ticks_per_second,
        ..Default::default()
    }
}

#[test]
fn fixed_ticks_are_equal_across_render_cadences() {
    for fps in [30u64, 60, 144] {
        let mut clock = SimulationClock::default();
        let settings = fixed(317);
        clock.plan(Duration::ZERO, true, settings, 5, 120);
        let mut previous = Duration::ZERO;
        // Derive cumulative timestamps instead of truncating each frame duration.
        for frame in 1..=fps * 10 {
            let now = Duration::from_nanos(frame * 1_000_000_000 / fps);
            let plan = clock.plan(now - previous, true, settings, 5, 120);
            clock.complete_steps(plan.steps);
            previous = now;
        }
        assert_eq!(clock.completed_ticks(), 3_170, "render cadence {fps}");
        assert_eq!(clock.wall_debt_ticks(), 0);
    }
}

#[test]
fn slow_presentation_services_ticks_without_exceeding_the_batch_budget() {
    let settings = SimulationSettings {
        presentation_fps: 1,
        ..fixed(300)
    };
    let interval = settings.service_interval(120);
    assert_eq!(interval, Duration::from_millis(400));
    assert_eq!(settings.frame_interval(), Duration::from_secs(1));
    assert_eq!(
        SimulationSettings {
            mode: SimulationMode::PerFrame,
            ..settings
        }
        .service_interval(120),
        Duration::from_secs(1)
    );
    let mut clock = SimulationClock::default();
    clock.plan(Duration::ZERO, true, settings, 5, 120);
    // Two ordinary service wakes, followed by the presentation deadline at 1s.
    for elapsed in [interval, interval, Duration::from_millis(200)] {
        let plan = clock.plan(elapsed, true, settings, 5, 120);
        assert!(plan.steps <= 120);
        assert_eq!(plan.wall_debt_ticks, 0);
        clock.complete_steps(plan.steps);
    }
    assert_eq!(clock.completed_ticks(), 300);
}

#[test]
fn exact_zero_one_and_many_preserve_counts_across_gpu_batches() {
    let mut clock = SimulationClock::default();
    let settings = fixed(300);
    for count in [0u64, 1, 17, 1_003] {
        clock.queue_exact_steps(count);
        let before = clock.completed_ticks();
        let mut total = 0u64;
        loop {
            let plan = clock.plan(Duration::from_secs(100), false, settings, 5, 120);
            assert!(plan.steps <= 120);
            assert_eq!(clock.completed_ticks(), before + total);
            total += u64::from(plan.steps);
            clock.complete_steps(plan.steps);
            if plan.pending_exact_steps == 0 {
                break;
            }
        }
        assert_eq!(total, count);
        assert_eq!(clock.completed_ticks(), before + count);
    }
}

#[test]
fn overload_retains_debt_instead_of_losing_time() {
    let mut clock = SimulationClock::default();
    let settings = fixed(1_000);
    clock.plan(Duration::ZERO, true, settings, 5, 120);
    let first = clock.plan(Duration::from_secs(1), true, settings, 5, 120);
    assert_eq!(first.steps, 120);
    assert_eq!(first.wall_debt_ticks, 880);
    let mut total = first.steps;
    while clock.wall_debt_ticks() > 0 {
        let plan = clock.plan(Duration::ZERO, true, settings, 5, 120);
        total += plan.steps;
    }
    assert_eq!(total, 1_000);
}

#[test]
fn pause_resume_and_hidden_transitions_have_no_catchup_burst() {
    let mut clock = SimulationClock::default();
    let settings = fixed(100);
    clock.plan(Duration::ZERO, true, settings, 5, 120);
    assert_eq!(
        clock
            .plan(Duration::from_secs(2), true, settings, 5, 120)
            .wall_debt_ticks,
        80
    );
    assert_eq!(
        clock
            .plan(Duration::from_secs(9), false, settings, 5, 120)
            .steps,
        0
    );
    assert_eq!(clock.wall_debt_ticks(), 0);
    assert_eq!(
        clock
            .plan(Duration::from_secs(30), true, settings, 5, 120)
            .steps,
        0
    );
    assert_eq!(
        clock
            .plan(Duration::from_millis(10), true, settings, 5, 120)
            .steps,
        1
    );
    clock.queue_exact_steps(400);
    clock.suspend(); // Hidden app deliberately does not call plan/dispatch.
    assert_eq!(clock.pending_exact_steps(), 400);
    assert_eq!(clock.wall_debt_ticks(), 0);
    assert_eq!(
        clock
            .plan(Duration::from_secs(30), true, settings, 5, 120)
            .steps,
        120
    );
    assert_eq!(clock.pending_exact_steps(), 280);
    assert_eq!(clock.wall_debt_ticks(), 0);
    clock.reset();
    assert_eq!(clock.pending_exact_steps(), 0);
    assert_eq!(clock.completed_ticks(), 0);
}

#[test]
fn mode_or_rate_changes_discard_old_wall_epoch_but_presentation_changes_do_not() {
    let mut clock = SimulationClock::default();
    let mut settings = fixed(100);
    clock.plan(Duration::ZERO, true, settings, 5, 120);
    clock.plan(Duration::from_millis(5), true, settings, 5, 120);
    settings.presentation_fps = 30;
    assert_eq!(
        clock
            .plan(Duration::from_millis(5), true, settings, 5, 120)
            .steps,
        1
    );
    settings.ticks_per_second = 200;
    assert_eq!(
        clock
            .plan(Duration::from_secs(1), true, settings, 5, 120)
            .steps,
        0
    );
    assert_eq!(
        clock
            .plan(Duration::from_millis(5), true, settings, 5, 120)
            .steps,
        1
    );
    settings.mode = SimulationMode::PerFrame;
    assert_eq!(
        clock
            .plan(Duration::from_secs(3), true, settings, 7, 120)
            .steps,
        7
    );
    assert_eq!(clock.plan(Duration::ZERO, true, settings, 7, 120).steps, 7);
}

#[test]
fn zero_budget_retains_exact_queue_and_fixed_debt() {
    let mut clock = SimulationClock::default();
    let settings = fixed(300);
    clock.plan(Duration::ZERO, true, settings, 5, 0);
    let plan = clock.plan(Duration::from_secs(1), true, settings, 5, 0);
    assert_eq!(plan.steps, 0);
    assert_eq!(plan.wall_debt_ticks, 300);
    clock.queue_exact_steps(12);
    assert_eq!(
        clock
            .plan(Duration::ZERO, true, settings, 5, 0)
            .pending_exact_steps,
        12
    );
}

#[test]
fn paused_idle_has_no_timer_and_real_deadlines_choose_earliest_wake() {
    let settings = SimulationSettings::default();
    assert_eq!(repaint_after(false, false, settings, None, None), None);
    assert_eq!(
        repaint_after(false, true, settings, None, None),
        Some(Duration::ZERO)
    );
    assert_eq!(
        repaint_after(true, false, settings, None, None),
        Some(settings.frame_interval())
    );
    assert_eq!(
        repaint_after(
            false,
            false,
            settings,
            Some(Duration::from_millis(100)),
            Some(Duration::from_millis(40))
        ),
        Some(Duration::from_millis(40))
    );
}

#[test]
fn settings_defaults_roundtrip_and_sanitize_persisted_values() {
    let mut settings: SimulationSettings = serde_json::from_str("{}").unwrap();
    assert_eq!(settings, SimulationSettings::default());
    settings.ticks_per_second = u32::MAX;
    settings.presentation_fps = 0;
    settings.sanitize();
    assert_eq!(settings.ticks_per_second, 7_200);
    assert_eq!(settings.presentation_fps, 1);
    let encoded = serde_json::to_string(&settings).unwrap();
    assert_eq!(
        serde_json::from_str::<SimulationSettings>(&encoded).unwrap(),
        settings
    );
}
