//! Numerical contracts for the optional appearance; no simulation equations here.
use primordial_soup::{
    cells::{
        PARTICLES_PER_BATCH, Settings, Style, Theme, fit_working_size, largest_binding_bytes,
        padding, quantum,
    },
    config::Preferences,
};

#[test]
fn old_preferences_keep_classic_and_cells_settings_roundtrip() {
    let old: Preferences = serde_json::from_str("{}").unwrap();
    assert_eq!(old.cells.style, Style::Classic);
    assert_eq!(old.cells.theme, Theme::Dark);
    assert_eq!(old.cells.structure_threshold, 0.0);
    assert_eq!(old.cells.membranes, 1.0);
    assert!(!old.cells.reference_palette);
    let mut settings = Settings::default();
    settings.style = Style::Cells;
    settings.detailed = true;
    settings.structure_threshold = 1.75;
    settings.membranes = 0.6;
    assert_eq!(
        settings,
        serde_json::from_str(&serde_json::to_string(&settings).unwrap()).unwrap()
    );
    settings.relief = f32::NAN;
    settings.scale = -1.0;
    settings.chromatic = f32::INFINITY;
    settings.structure_threshold = f32::NAN;
    settings.membranes = f32::INFINITY;
    settings.sanitize();
    assert_eq!(settings.structure_threshold, 0.0);
    assert_eq!(settings.membranes, 1.0);
    settings.structure_threshold = 10.0;
    settings.sanitize();
    assert_eq!(settings.structure_threshold, 4.0);
    assert!(settings.relief.is_finite() && settings.relief <= 0.9);
    assert_eq!(settings.scale, 0.5);
    assert_eq!(settings.chromatic, 0.0);
}

#[test]
fn fast_cells_migrates_old_preferences_and_preserves_explicit_choice() {
    // Both a missing Cells object and older stored Cells settings enable the
    // production fast path without changing the selected rendering style.
    let old_preferences: Preferences = serde_json::from_str("{}").unwrap();
    assert!(old_preferences.cells.fast);
    let old_cells: Settings =
        serde_json::from_str(r#"{"style":"cells","theme":"dark","detailed":true}"#).unwrap();
    assert!(old_cells.fast);
    assert_eq!(old_cells.style, Style::Cells);
    assert!(old_cells.detailed);

    let mut stored_preferences = serde_json::to_value(&old_preferences).unwrap();
    stored_preferences["cells"]["style"] = serde_json::json!("cells");
    stored_preferences["cells"]["fast"] = serde_json::json!(false);
    let explicit: Preferences = serde_json::from_value(stored_preferences).unwrap();
    assert!(!explicit.cells.fast);
    let encoded = serde_json::to_string(&explicit).unwrap();
    let restored: Preferences = serde_json::from_str(&encoded).unwrap();
    assert!(!restored.cells.fast);
    assert_eq!(restored.cells, explicit.cells);
}

#[test]
fn legacy_cells_use_dark_and_original_hues_but_explicit_theme_choices_survive() {
    for palette in [false, true] {
        let old: Settings = serde_json::from_value(serde_json::json!({
            "style":"cells", "reference_palette":palette, "scale":2.0, "wash":0.6
        }))
        .unwrap();
        assert_eq!(old.theme, Theme::Dark);
        assert!(!old.reference_palette);
        assert_eq!(old.scale, 2.0);
        assert_eq!(old.wash, 0.6);
        for theme in [Theme::Dark, Theme::Light] {
            let mut explicit = old.clone();
            explicit.theme = theme;
            explicit.reference_palette = palette;
            assert_eq!(
                explicit,
                serde_json::from_str::<Settings>(&serde_json::to_string(&explicit).unwrap())
                    .unwrap()
            );
        }
    }
    let restored: Settings =
        serde_json::from_str(r#"{"theme":"dark","style":"cells","microscope":true,"scale":2.0}"#)
            .unwrap();
    assert_eq!(restored.style, Style::Cells);
    assert_eq!(restored.scale, 2.0);
    assert!(
        !serde_json::to_string(&restored)
            .unwrap()
            .contains("microscope")
    );
    assert!(serde_json::from_str::<Settings>(r#"{"theme":"sepia"}"#).is_err());
    assert!(serde_json::from_str::<Settings>(r#"{"theme":"dark","typo":1}"#).is_err());
}

#[test]
fn integer_batch_bound_includes_rounding_at_maximum_population() {
    for n in [
        1,
        65_536,
        600_000,
        4_194_304,
        8_388_608,
        16_200_000,
        (1 << 24) - 1,
    ] {
        let q = quantum(n, PARTICLES_PER_BATCH).unwrap() as f64;
        assert!(q >= 26_420.0, "precision fell with total population: {n}");
        let maximum = (0.62 * q).ceil() as u64 + 1;
        assert!(maximum * u64::from(n.min(PARTICLES_PER_BATCH)) <= u64::from(u32::MAX));
    }
}

#[test]
fn overscan_contains_gaussian_cubic_and_finishing_support_even_at_odd_sizes() {
    for short in [
        64u32, 128, 180, 181, 359, 360, 361, 720, 721, 768, 1080, 1440, 2160, 4320,
    ] {
        let size = [short + 37, short];
        let pad = padding(size) as f64;
        let scale = short as f64 / 1080.0;
        let highest = ((24.0 * scale).max(1.0).log2() - 2.0)
            .ceil()
            .clamp(0.0, 5.0) as u32;
        for axis in size {
            let full = axis + 2 * pad as u32;
            let resolutions: Vec<_> = (0..=highest).map(|l| (full >> (l + 1)).max(1)).collect();
            for p in [
                pad - scale - scale.max(1.0),
                full as f64 - pad + scale + scale.max(1.0),
            ] {
                let last = *resolutions.last().unwrap() as f64;
                let q = (p * last / full as f64 - 0.5).floor();
                let mut lo = q - 1.0;
                let mut hi = q + 2.0;
                for l in (0..=highest as usize).rev() {
                    let src = if l == 0 { full } else { resolutions[l - 1] } as f64;
                    let dst = resolutions[l] as f64;
                    lo = ((lo + 0.5) * src / dst - 0.5).floor() - 3.0;
                    hi = ((hi + 0.5) * src / dst - 0.5).floor() + 4.0;
                }
                let sprite = (24.0 * scale).max(0.75);
                assert!(
                    lo - sprite >= 0.0 && hi + sprite < full as f64,
                    "clipped filter at {size:?}: {lo}..{hi}, pad{pad}"
                );
            }
        }
    }
}

#[test]
fn widest_sprite_retains_mass_and_color_through_integer_batches() {
    for radius in [1.5f64, 3.0, 6.0, 12.0, 24.0] {
        let batch = if radius > 6.0 {
            65_536
        } else {
            PARTICLES_PER_BATCH
        };
        let q = quantum((1 << 24) - 1, batch).unwrap() as f64;
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let mut exact = [0.0; 4];
            let mut fixed = [0.0; 4];
            for y in -25..=25 {
                for x in -25..=25 {
                    let rr =
                        ((x as f64 + phase).powi(2) + (y as f64 + phase).powi(2)) / radius.powi(2);
                    if rr > 1.0 {
                        continue;
                    }
                    let mass = (0.4 * 0.85896 / radius.powi(2) * (-rr / (2.0 * 0.45 * 0.45)).exp())
                        .min(0.62);
                    let quantized = (mass * q).round();
                    for (i, color) in [0.28, 0.43, 0.04, 1.0].iter().enumerate() {
                        exact[i] += mass * color;
                        fixed[i] += (mass * q * color).round().min(quantized) / q;
                    }
                }
            }
            assert!(
                (fixed[3] / exact[3] - 1.0).abs() < 0.01,
                "mass loss at radius {radius}, phase {phase}"
            );
            // Absolute pigment error per unit mass handles near-zero color channels.
            for i in 0..3 {
                assert!(
                    (fixed[i] - exact[i]).abs() / exact[3] < 0.01,
                    "color loss at radius {radius}"
                );
            }
        }
    }
}

#[test]
fn coarse_map_is_contracting_and_ten_iterations_recover_pixel() {
    for size in [[64u32, 64u32], [1920, 1080], [3840, 2160], [600, 1800]] {
        let scale = size[0].min(size[1]) as f64 / 1080.0;
        let pad = padding(size) as f64;
        let dimensions = size.map(|v| v as f64 + 2.0 * pad);
        let spacing = (8.0 * scale).max(1.0);
        let cells = dimensions.map(|v| (v / spacing).floor());
        let step = [dimensions[0] / cells[0], dimensions[1] / cells[1]];
        assert!(3.6 * scale / step[0].min(step[1]) <= 0.4500001);
        // An adversarial checkerboard reaches the largest allowed map derivatives.
        let delta = |p: [f64; 2]| -> [f64; 2] {
            let p = [
                p[0].clamp(0.0, dimensions[0]) / step[0],
                p[1].clamp(0.0, dimensions[1]) / step[1],
            ];
            let q = [p[0].floor(), p[1].floor()];
            let f = [p[0] - q[0], p[1] - q[1]];
            let mut d = [0.0; 2];
            for y in 0..2 {
                for x in 0..2 {
                    let weight = if x == 0 { 1.0 - f[0] } else { f[0] }
                        * if y == 0 { 1.0 - f[1] } else { f[1] };
                    let sign = if (q[0] as i64 + q[1] as i64 + x + y) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    d[0] += sign * 0.9 * scale * weight;
                    d[1] -= sign * 0.9 * scale * weight;
                }
            }
            d
        };
        for i in 0..200 {
            let p = [
                dimensions[0] * (i as f64 / 199.0),
                dimensions[1] * ((i * 73 % 199) as f64 / 199.0),
            ];
            let d = delta(p);
            let q = [p[0] + d[0], p[1] + d[1]];
            let mut inverse = q;
            for _ in 0..10 {
                let d = delta(inverse);
                inverse = [q[0] - d[0], q[1] - d[1]];
            }
            assert!((inverse[0] - p[0]).abs().max((inverse[1] - p[1]).abs()) / scale < 0.003);
        }
    }
}

#[test]
fn fractional_output_scales_preserve_small_footprint_energy_and_peak_bound() {
    for short_edge in [
        64.0f64, 128.0, 720.0, 1080.0, 1436.0, 1620.0, 2160.0, 2872.0, 3240.0,
    ] {
        let s = short_edge / 1080.0;
        for radius in [s.max(0.75), (1.5 * s).max(0.75), (2.25 * s).max(0.75)] {
            if radius > 3.0 {
                continue;
            }
            for px in [0.0, 0.25, 0.5, 0.75] {
                for py in [0.0, 0.25, 0.5, 0.75] {
                    let mut weights = Vec::new();
                    for y in -4..=4 {
                        for x in -4..=4 {
                            let rr = ((x as f64 + px).powi(2) + (y as f64 + py).powi(2))
                                / radius.powi(2);
                            if rr <= 1.0 {
                                weights.push((-rr / (2.0 * 0.45 * 0.45)).exp());
                            }
                        }
                    }
                    let sum: f64 = weights.iter().sum();
                    let target = 0.4 * s * s;
                    let peak = weights
                        .iter()
                        .map(|w| w / sum * target)
                        .fold(0.0f64, f64::max);
                    assert!(
                        peak <= 0.62,
                        "peak bound failed at scale{s}, radius{radius}, phase{px},{py}: {peak}"
                    );
                    let q = quantum(
                        (1 << 24) - 1,
                        if radius / s > 6.0 {
                            65_536
                        } else {
                            PARTICLES_PER_BATCH
                        },
                    )
                    .unwrap() as f64;
                    let quantized: f64 = weights
                        .iter()
                        .map(|w| (w / sum * target * q).round() / q)
                        .sum();
                    assert!(
                        (quantized / target - 1.0).abs() < 0.01,
                        "small footprint energy changed"
                    );
                }
            }
        }
    }
}

#[test]
fn working_size_fits_the_storage_binding_limit_at_the_same_aspect() {
    const DEFAULT_BINDING: u64 = 128 << 20;
    // 1080p already fits the WebGPU default binding size and is left alone.
    assert_eq!(
        fit_working_size([1920, 1080], DEFAULT_BINDING),
        [1920, 1080]
    );
    // A 4K working size does not; it shrinks until it fits, keeping its aspect.
    assert!(largest_binding_bytes([3840, 2160]) > DEFAULT_BINDING);
    let fitted = fit_working_size([3840, 2160], DEFAULT_BINDING);
    assert!(largest_binding_bytes(fitted) <= DEFAULT_BINDING);
    assert!(fitted[0] < 3840 && fitted[0] >= 3000, "{fitted:?}");
    assert!((fitted[0] as f64 / fitted[1] as f64 - 16.0 / 9.0).abs() < 0.01);
    // A generous limit changes nothing.
    assert_eq!(fit_working_size([3840, 2160], u64::MAX), [3840, 2160]);
}
