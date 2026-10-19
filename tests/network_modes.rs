use primordial_soup::{
    archive,
    config::Project,
    network::{self, Bank, Mode},
    share,
};
fn bank(mode: Mode) -> Bank {
    Bank {
        mode,
        revision: 1,
        parent: None,
        seed: 0.25,
        scale: 0.1,
    }
}
#[test]
fn layouts_and_canonical_generation_are_bounded_and_repeatable() {
    for mode in [Mode::Mlp, Mode::Gru] {
        let b = bank(mode);
        let p = network::derive(&b, 0);
        assert_eq!(p.len(), mode.parameters());
        assert!(p.iter().all(|v| v.is_finite() && v.abs() <= 8.0));
        assert_eq!(p, network::derive(&b, 0));
        assert_ne!(p, network::derive(&b, 1));
        let mut b = Bank {
            parent: Some(p.clone()),
            scale: 0.0,
            ..b
        };
        assert_eq!(network::derive(&b, 12), p);
        b.scale = 0.6;
        for _ in 0..1000 {
            b.parent = Some(network::derive(&b, 3));
        }
        assert!(
            b.parent
                .unwrap()
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 8.0)
        );
    }
}
#[test]
fn analytical_feedforward_and_recurrent_fixtures() {
    let mut p = vec![0.0; 76];
    p[0] = 1.0;
    p[40] = 2.0;
    p[72] = 0.5;
    let y = network::evaluate(Mode::Mlp, &p, [1.0, 0.0, 0.0, 0.0], &mut [0.0; 4]);
    assert!((y[0] - (2.0 * 1.0f32.tanh() + 0.5)).abs() < 1e-6);
    let mut p = vec![0.0; 128];
    p[104] = 1.0;
    p[108] = 1.0;
    let mut h = [0.0; 4];
    for tick in 1..=4 {
        let y = network::evaluate(Mode::Gru, &p, [0.0; 4], &mut h);
        assert!((y[0] - 1.0f32.tanh() * (1.0 - 0.5f32.powi(tick))).abs() < 1e-6);
    }
}

#[test]
fn revision_three_observes_occupancy_speed_and_has_signed_zero_motors() {
    let x = network::encode_observations([0.0; 4], [1.0, 1.0], 0.001, 1.0, 0.001, 1.0);
    assert_eq!(x, [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]);
    assert_eq!(
        network::encode_observations([0.0; 4], [0.0; 2], 0.0, 1.0, 0.001, 1.0),
        [0.0; 7]
    );
    let x = network::encode_observations(
        [0.002, -0.002, 0.001, -0.001],
        [1.0, 0.0],
        0.0,
        1.0,
        0.001,
        1.0,
    );
    assert_eq!(&x[..4], &[0.5, -0.5, 0.5, -0.5]);
    for mode in [Mode::Mlp, Mode::Gru] {
        let count = if mode == Mode::Mlp { 100 } else { 164 };
        let p = vec![0.0; count];
        assert_eq!(mode.parameters_at(3), count);
        assert_eq!(
            network::evaluate_revision(mode, 3, &p, x, &mut [0.0; 4]),
            [0.0; 4]
        );
        let mut p = p;
        p[count - 4] = -1.0;
        p[count - 3] = 1.0;
        let y = network::evaluate_revision(mode, 3, &p, x, &mut [0.0; 4]);
        assert!(y[0] < 0.0 && y[1] > 0.0 && y.iter().all(|v| v.abs() <= 2.0));
    }
}

#[test]
fn revision_three_matches_independent_equations_and_legacy_adoption_keeps_semantics() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/network-io-oracle.fixture.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mode: Mode = serde_json::from_value(case["mode"].clone()).unwrap();
        let p: Vec<f32> = serde_json::from_value(case["weights"].clone()).unwrap();
        let mut h: [f32; 4] = serde_json::from_value(case["initial_hidden"].clone()).unwrap();
        for step in case["steps"].as_array().unwrap() {
            let x = serde_json::from_value(step["input"].clone()).unwrap();
            let y = network::evaluate_revision(mode, 3, &p, x, &mut h);
            let expected: [f64; 4] = serde_json::from_value(step["output"].clone()).unwrap();
            assert!(
                y.iter()
                    .zip(expected)
                    .all(|(a, b)| (f64::from(*a) - b).abs() < 3e-6)
            );
        }
        let mut project = Project::default();
        project.config_mut().switch_mode(mode, 0.25);
        for revision in [1, 2] {
            project.config_mut().set_parent(None);
            project
                .config_mut()
                .networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|b| b.mode == mode)
                .unwrap()
                .revision = revision;
            let parent = network::derive(&project.config().bank().unwrap(), 0);
            project.config_mut().adopt_parent(parent, 0.75);
            assert_eq!(project.config().network_revision(), 2);
            assert_eq!(project.config().network_parameters(), mode.parameters());
            assert!(project.validate().is_ok());
        }
        project.config_mut().randomize_network();
        assert_eq!(project.config().network_revision(), 3);
        assert!(project.config().bank().unwrap().parent.is_none());
        assert!(project.validate().is_ok());
    }
}
#[test]
fn banks_survive_switches_project_share_archive_and_zip() {
    let mut p = Project::default();
    let original = p.config().rule.clone();
    p.config_mut().switch_mode(Mode::Mlp, 0.25);
    p.config_mut().set_parent(Some(vec![0.0; 100])); // zero is a valid authored rule
    assert!(!p.config().generated());
    p.config_mut().mutation_scale = 0.23;
    p.config_mut().switch_mode(Mode::Gru, 0.5);
    p.config_mut().mutation_seed = 0.75;
    p.config_mut().switch_mode(Mode::Mlp, 0.0);
    assert_eq!(p.config().mutation_scale, 0.23);
    assert_eq!(p.config().bank().unwrap().parent, Some(vec![0.0; 100]));
    let json = p.to_json().unwrap();
    assert!(json.contains("\"version\": 10"));
    let q = Project::from_json(&json).unwrap();
    assert_eq!(q.to_json().unwrap(), json);
    let link = share::encode_link(&p).unwrap();
    assert_eq!(
        share::decode_text(&link)
            .unwrap()
            .unwrap()
            .to_json()
            .unwrap(),
        json
    );
    let bytes = share::encode_binary(&p).unwrap();
    assert!(bytes.starts_with(b"PS10"));
    assert_eq!(archive::state(&p)["version"], 10);
    assert!(archive::hash_state(&p).starts_with("ps10-"));
    let mut named = p.clone();
    named.name = "Networks".into();
    let saves = std::collections::BTreeMap::from([("Networks".into(), named.clone())]);
    let zip = primordial_soup::export::export_zip(&saves).unwrap();
    assert_eq!(
        primordial_soup::export::import_zip(&zip).unwrap()["Networks"]
            .to_json()
            .unwrap(),
        named.to_json().unwrap()
    );
    p.config_mut().switch_mode(Mode::Fourier, 0.0);
    assert_eq!(
        p.config().rule,
        original
            .iter()
            .map(|v| (*v as f32) as f64)
            .collect::<Vec<_>>()
    );
    assert_eq!(p.config().normalized_networks().unwrap().banks.len(), 3);
}
#[test]
fn malformed_network_banks_fail_without_fallback() {
    let mut p = Project::default();
    p.config_mut().switch_mode(Mode::Gru, 0.25);
    let doc: serde_json::Value = serde_json::from_str(&p.to_json().unwrap()).unwrap();
    for (field, value) in [
        ("revision", serde_json::json!(99)),
        ("parent", serde_json::json!([0.0])),
        ("mode", serde_json::json!("unknown")),
    ] {
        let mut v = doc.clone();
        v["configs"][0]["networks"]["banks"][1][field] = value;
        assert!(Project::from_json(&v.to_string()).is_err());
    }
    let count = p.config().network_parameters();
    p.config_mut().set_parent(Some(vec![9.0; count]));
    assert!(p.validate().is_err());
}

#[test]
fn revision_one_matches_frozen_coefficients() {
    for revision in [1, 2] {
        let fixtures: serde_json::Value = serde_json::from_str(if revision == 1 {
            include_str!("fixtures/network-v1.fixture.json")
        } else {
            include_str!("fixtures/network-v2.fixture.json")
        })
        .unwrap();
        for f in fixtures.as_array().unwrap() {
            let mode: Mode = serde_json::from_value(f["mode"].clone()).unwrap();
            let cohort = f["cohort"].as_u64().unwrap() as u32;
            let b = Bank {
                revision,
                ..bank(mode)
            };
            let generated = network::derive(&b, cohort);
            assert_eq!(
                serde_json::json!(generated.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                f["generated_bits"]
            );
            let mutated = network::derive(
                &Bank {
                    parent: Some(generated),
                    ..b
                },
                cohort,
            );
            assert_eq!(
                serde_json::json!(mutated.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                f["mutated_bits"]
            );
        }
    }
}

#[test]
fn active_bank_adapters_and_legacy_versions_cannot_disagree() {
    let mut p = Project::default();
    p.config_mut().switch_mode(Mode::Mlp, 0.25);
    p.config_mut().switch_mode(Mode::Fourier, 0.5);
    let doc: serde_json::Value = serde_json::from_str(&p.to_json().unwrap()).unwrap();
    for version in [8, 9] {
        let mut bad = doc.clone();
        bad["version"] = serde_json::json!(version);
        assert!(Project::from_json(&bad.to_string()).is_err());
    }
    for field in ["seed", "scale"] {
        let mut bad = doc.clone();
        bad["configs"][0]["networks"]["banks"][0][field] = serde_json::json!(0.314159);
        assert!(Project::from_json(&bad.to_string()).is_err());
    }
    for count in [0, 1, 47, 79] {
        let mut bad = doc.clone();
        bad["configs"][0]["rule"] = serde_json::json!(vec![0.0; count]);
        assert!(Project::from_json(&bad.to_string()).is_err());
    }
    let mut bad = doc.clone();
    bad["configs"][0]["networks"]["banks"][0]["parent"][0] = serde_json::json!(7.0);
    assert!(Project::from_json(&bad.to_string()).is_err());
    bad["configs"][0]["networks"]["banks"][0]["parent"] = serde_json::Value::Null;
    assert!(Project::from_json(&bad.to_string()).is_err());
}

#[test]
fn dense_networks_match_independent_double_precision_fixtures() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/network-oracle.fixture.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mode: Mode = serde_json::from_value(case["mode"].clone()).unwrap();
        let weights: Vec<f32> = serde_json::from_value(case["weights"].clone()).unwrap();
        let mut hidden: [f32; 4] = serde_json::from_value(case["initial_hidden"].clone()).unwrap();
        for step in case["steps"].as_array().unwrap() {
            let input = serde_json::from_value(step["input"].clone()).unwrap();
            let output = network::evaluate(mode, &weights, input, &mut hidden);
            let expected: [f64; 4] = serde_json::from_value(step["output"].clone()).unwrap();
            for i in 0..4 {
                assert!((f64::from(output[i]) - expected[i]).abs() < 2e-6);
            }
            if mode == Mode::Gru {
                let expected: [f64; 4] = serde_json::from_value(step["hidden"].clone()).unwrap();
                for i in 0..4 {
                    assert!((f64::from(hidden[i]) - expected[i]).abs() < 2e-6);
                }
            }
        }
    }
}

#[test]
fn revision_two_mutations_are_not_radial_and_adoption_advances_streams() {
    for mode in [Mode::Mlp, Mode::Gru] {
        let mut aligned = 0;
        let mut total = 0;
        for seed in 0..32 {
            let b = Bank {
                revision: 2,
                seed: seed as f64 / 32.0,
                ..bank(mode)
            };
            let p = network::derive(&b, 0);
            let q = network::derive(
                &Bank {
                    parent: Some(p.clone()),
                    ..b
                },
                0,
            );
            for (offset, count, fan) in network::tensors(mode) {
                if fan == 0 {
                    continue;
                }
                for i in offset..offset + count {
                    aligned += usize::from(p[i] * (q[i] - p[i]) > 0.0);
                    total += 1;
                }
            }
        }
        assert!(
            (0.4..0.6).contains(&(aligned as f64 / total as f64)),
            "biased mutation {aligned}/{total}"
        );
        let mut p = Project::default();
        p.config_mut().switch_mode(mode, 0.25);
        p.config_mut()
            .networks
            .as_mut()
            .unwrap()
            .banks
            .retain(|b| b.mode == mode);
        assert!(p.validate().is_ok());
        let mut only_new_mode = p.clone();
        only_new_mode.config_mut().switch_mode(Mode::Fourier, 0.25);
        assert!(only_new_mode.validate().is_ok());
        assert_eq!(only_new_mode.config().bank().unwrap().revision, 1);
        let mut previous_delta: Option<Vec<f32>> = None;
        for _ in 0..8 {
            let before = p.config().bank().unwrap();
            let parent = network::derive(&before, 0);
            // Deliberately offer the old seed: the collision fallback must advance it.
            p.config_mut().adopt_parent(parent.clone(), before.seed);
            let after = p.config().bank().unwrap();
            assert_eq!(after.revision, network::CURRENT_REVISION);
            let fold = |s: f64| s.to_bits() as u32 ^ (s.to_bits() >> 32) as u32;
            assert_ne!(fold(before.seed), fold(after.seed));
            let child = network::derive(&after, 0);
            let delta: Vec<_> = child.iter().zip(parent).map(|(a, b)| a - b).collect();
            if let Some(old) = previous_delta {
                assert_ne!(old, delta);
            }
            previous_delta = Some(delta);
            assert!(p.validate().is_ok());
            assert_eq!(
                Project::from_json(&p.to_json().unwrap())
                    .unwrap()
                    .config()
                    .bank(),
                Some(after)
            );
        }
    }
}

#[test]
fn finite_extreme_inputs_are_bounded_and_overflowing_gain_is_rejected() {
    for mode in [Mode::Mlp, Mode::Gru] {
        let mut project = Project::default();
        project.config_mut().switch_mode(mode, 0.25);
        project.config_mut().sensor_gain = f32::MAX as f64;
        assert!(project.validate().is_err());
        project.config_mut().sensor_gain = network::INPUT_LIMIT as f64;
        assert!(project.validate().is_ok());
        project
            .config_mut()
            .networks
            .as_mut()
            .unwrap()
            .banks
            .iter_mut()
            .find(|b| b.mode == mode)
            .unwrap()
            .revision = 2;
        let weights = network::derive(&project.config().bank().unwrap(), 0);
        let mut hidden = [0.0; 4];
        for tick in 0..100_000 {
            let magnitude = if tick % 2 == 0 { f32::MAX } else { -f32::MAX };
            let output = network::evaluate(
                mode,
                &weights,
                [magnitude, -magnitude, magnitude, -magnitude],
                &mut hidden,
            );
            let bound = if mode == Mode::Gru { 40.0 } else { 72.0 };
            assert!(output.iter().all(|v| v.is_finite() && v.abs() <= bound));
            assert!(hidden.iter().all(|v| v.is_finite() && v.abs() <= 1.000001));
        }
    }
}
