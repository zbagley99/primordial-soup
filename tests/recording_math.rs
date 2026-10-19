use primordial_soup::record::{
    CropRect, DriverAction, FULL_FRAME, MIN_RECORDING_DIM, RECORDING_FPS, RecordingSettings,
    clamp_resolution, crop_rect, driver_action, even_dim, is_full_frame, rescale_samples,
};
#[test]
fn light_export_defaults_and_output_time_are_independent_of_physics_rate() {
    let mut s = RecordingSettings::default();
    assert_eq!(s.resolution, [512; 2]);
    assert_eq!(s.duration, 5);
    assert_eq!(s.quality, 3);
    assert_eq!(s.physics_steps, 60);
    assert_eq!(s.motion_blur_samples, 1);
    assert_eq!(s.frame_count(), 300);
    assert_eq!(s.physics_frame_count(), 18000);
    s.set_physics_steps(240);
    assert_eq!(s.motion_blur_samples, 1);
    assert_eq!(s.frame_count(), 300);
    assert_eq!(s.physics_frame_count(), 72000);
    assert_eq!(RECORDING_FPS, 60);
}

#[test]
fn world_recording_dimensions_do_not_depend_on_the_display_window() {
    let mut s = RecordingSettings::default();
    s.resolution = [1920, 1080];
    assert_eq!(s.output_size([724, 724]).unwrap(), [1920, 1080]);
    s.resolution = [FULL_FRAME; 2];
    assert_eq!(s.output_size([1024, 512]).unwrap(), [1024, 512]);
    assert!(s.output_size([8192, 1024]).is_err());
}

#[test]
fn source_worked_blur_rescale_example_and_collapsed_ceiling() {
    // recordingSettings.ts's independent specification example:4/9 travel.
    assert_eq!(rescale_samples(5, 10, 15), 7);
    assert_eq!(rescale_samples(8, 10, 1), 1);
    assert_eq!(rescale_samples(1, 1, 480), 480);
    assert_eq!(rescale_samples(1, 0, 0), 1);
    for from in [2, 5, 10, 15, 60, 120, 480] {
        for to in [1, 2, 5, 10, 15, 60, 120, 480] {
            assert_eq!(rescale_samples(1, from, to), 1);
            assert_eq!(rescale_samples(from, from, to), to);
            for value in 1..=from {
                let next = rescale_samples(value, from, to);
                assert!((1..=to).contains(&next));
                if to > 1 {
                    assert!(
                        (next - 1) as f64 / (to - 1) as f64
                            <= (value - 1) as f64 / (from - 1) as f64 + 1e-12
                    );
                }
            }
        }
    }
    assert_eq!(rescale_samples(u32::MAX, u32::MAX, u32::MAX), 480);
}
#[test]
fn source_even_dimensions_center_crop_and_floor_priority() {
    assert_eq!(MIN_RECORDING_DIM, 128);
    assert_eq!(even_dim(127), 128);
    assert_eq!(even_dim(129), 128);
    assert_eq!(even_dim(1921), 1920);
    assert_eq!(
        clamp_resolution([FULL_FRAME; 2], [1921, 1081]),
        [1920, 1080]
    );
    assert_eq!(
        crop_rect([1280, 720], [1921, 1081]),
        CropRect {
            x: 320,
            y: 180,
            width: 1280,
            height: 720
        }
    );
    assert!(is_full_frame([FULL_FRAME; 2], [1921, 1081]));
    assert!(!is_full_frame([1280, 720], [1921, 1081]));
    // Reference helper keeps its floor even if a minimized window is smaller.
    // Recorder::start separately rejects such a window before a GPU copy.
    assert_eq!(clamp_resolution([1, 1], [64, 64]), [128, 128]);
    assert_eq!(
        crop_rect([1, 1], [63, 63]),
        CropRect {
            x: -33,
            y: -33,
            width: 128,
            height: 128
        }
    );
}
#[test]
fn finalization_outranks_pause_and_pause_does_not_encode() {
    assert_eq!(driver_action(false, false), DriverAction::Encode);
    assert_eq!(driver_action(false, true), DriverAction::Suspend);
    assert_eq!(driver_action(true, false), DriverAction::Finalize);
    assert_eq!(driver_action(true, true), DriverAction::Finalize);
}
#[test]
fn settings_reject_impossible_encoder_and_schedule_inputs() {
    let mut s = RecordingSettings::default();
    s.duration = 0;
    assert!(s.validate().is_err());
    s.duration = 121;
    assert!(s.validate().is_err());
    s.duration = 120;
    s.quality = 5;
    assert!(s.validate().is_err());
    s.quality = 4;
    s.physics_steps = 481;
    assert!(s.validate().is_err());
    s.physics_steps = 480;
    s.motion_blur_samples = 481;
    assert!(s.validate().is_err());
    s.motion_blur_samples = 480;
    assert!(s.validate().is_ok());
    // Legal recording rates fit the engine's uniform scheduling ring.
    assert!(s.physics_steps <= primordial_soup::gpu::MAX_STEPS);
    s.resolution = [127, 1080];
    assert!(s.validate().is_err());
}
