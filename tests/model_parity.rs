use primordial_soup::{
    config::{Project, WorldSettings, mutate_rule_v1, presets},
    coords::*,
    history::{History, RecordOutcome},
};
use serde_json::{Value, json};
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/parity.fixture.json")).unwrap()
}
fn n(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap()
}
fn v2(v: &Value, k: &str) -> Vec2 {
    [v[k][0].as_f64().unwrap(), v[k][1].as_f64().unwrap()]
}
fn close(actual: Vec2, expected: Vec2, tol: f64) {
    for k in 0..2 {
        assert!(
            (actual[k] - expected[k]).abs() < tol,
            "{actual:?} != {expected:?}"
        );
    }
}
fn camel_to_snake(k: &str) -> String {
    let mut s = String::new();
    for c in k.chars() {
        if c.is_ascii_uppercase() {
            s.push('_');
            s.push(c.to_ascii_lowercase());
        } else {
            s.push(c);
        }
    }
    s
}
fn reference_config(v: &Value) -> primordial_soup::config::SimulationConfig {
    let mut m = serde_json::Map::new();
    for (k, v) in v.as_object().unwrap() {
        let key = camel_to_snake(k);
        let val = if key == "cohort_fences" {
            json!(v.as_f64().unwrap() > 0.0)
        } else {
            v.clone()
        };
        m.insert(key, val);
    }
    for (k, v) in [
        ("gravity_trails", 0.0),
        ("color_offset", 0.0),
        ("trail_weight", 1.0),
    ] {
        m.entry(k).or_insert(json!(v));
    }
    serde_json::from_value(Value::Object(m)).unwrap()
}
fn hex_words(s: &str) -> Vec<u32> {
    s.as_bytes()
        .chunks_exact(8)
        .map(|c| {
            let c = std::str::from_utf8(c).unwrap();
            let mut b = [0; 4];
            for k in 0..4 {
                b[k] = u8::from_str_radix(&c[k * 2..k * 2 + 2], 16).unwrap();
            }
            u32::from_le_bytes(b)
        })
        .collect()
}
#[test]
fn frozen_python_coordinate_oracle() {
    let f = fixture();
    let c = &f["coords"];
    for r in c["worldHalfExtent"].as_array().unwrap() {
        close(world_half_extent(v2(r, "canvasSize")), v2(r, "out"), 1e-12);
    }
    for r in c["letterboxScale"].as_array().unwrap() {
        close(
            letterbox_scale(v2(r, "canvasSize"), v2(r, "windowSize")),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["worldToUv"].as_array().unwrap() {
        close(
            world_to_uv(v2(r, "p"), v2(r, "canvasSize")),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["worldToNdc"].as_array().unwrap() {
        close(
            world_to_ndc(v2(r, "p"), v2(r, "canvasSize")),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["worldToScreenNdc"].as_array().unwrap() {
        close(
            world_to_screen_ndc(
                v2(r, "p"),
                v2(r, "canvasSize"),
                v2(r, "windowSize"),
                v2(r, "pan"),
                n(r, "zoom"),
            ),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["screenToWorld"].as_array().unwrap() {
        close(
            screen_to_world(
                v2(r, "pixel"),
                v2(r, "windowSize"),
                v2(r, "canvasSize"),
                v2(r, "pan"),
                n(r, "zoom"),
            ),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["screenToNdc"].as_array().unwrap() {
        close(
            screen_to_ndc(v2(r, "pixel"), v2(r, "windowSize")),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in c["uvRadiusToWorld"].as_array().unwrap() {
        assert_eq!(uv_radius_to_world(n(r, "radius")), n(r, "out"));
    }
}
#[test]
fn frozen_python_sizing_and_camera_oracle() {
    let f = fixture();
    for r in f["sizing"]["sizingFor"].as_array().unwrap() {
        let s = sizing_for(n(r, "worldSize"), 1.0).unwrap();
        assert_eq!(s.entity_count, r["out"][0].as_u64().unwrap() as u32);
        assert_eq!(s.canvas, [r["out"][1].as_u64().unwrap() as u32; 2]);
    }
    for r in f["sizing"]["canvasDimensions"].as_array().unwrap() {
        close(
            canvas_dimensions(n(r, "aspect"), r["dim"].as_f64().unwrap_or(1024.0)),
            v2(r, "out"),
            1e-12,
        );
    }
    for r in f["camera"]["zoomAtPixel"].as_array().unwrap() {
        let mut c = Camera {
            pan: v2(&r["start"], "pan").map(|v| v as f32),
            zoom: n(&r["start"], "zoom") as f32,
            ..Default::default()
        };
        c.zoom_at_pixel(
            n(r, "notches"),
            v2(r, "pixel"),
            v2(r, "windowSize"),
            v2(r, "canvasSize"),
        );
        close(c.pan.map(f64::from), v2(&r["end"], "pan"), 2e-6);
        assert!((c.zoom as f64 - n(&r["end"], "zoom")).abs() < 1e-5);
    }
    for r in f["camera"]["panByFraction"].as_array().unwrap() {
        let mut c = Camera {
            pan: v2(&r["start"], "pan").map(|v| v as f32),
            zoom: n(&r["start"], "zoom") as f32,
            ..Default::default()
        };
        c.pan_by_fraction(v2(r, "fraction"), v2(r, "canvasSize"));
        close(c.pan.map(f64::from), v2(&r["end"], "pan"), 2e-6);
    }
    for (w, a) in [(f64::NAN, 1.0), (1.0, 0.0), (-1.0, 1.0), (1e300, 1.0)] {
        assert!(sizing_for(w, a).is_err());
    }
}
#[test]
fn frozen_gpu_v8_migration_is_bit_exact() {
    let f: Value = serde_json::from_str(include_str!("fixtures/mutationV1.fixture.json")).unwrap();
    for r in f["cases"].as_array().unwrap() {
        let rule: Vec<f64> = serde_json::from_value(r["rule"].clone()).unwrap();
        let actual = mutate_rule_v1(&rule, n(r, "mutationScale"), n(r, "mutationSeed") as f32);
        let expected: Vec<f64> = serde_json::from_value(r["expected"].clone()).unwrap();
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                (*a as f32).to_bits(),
                (b as f32).to_bits(),
                "{} coefficient {i}",
                r["name"]
            );
        }
    }
}
#[test]
fn frozen_layout_and_packed_bytes() {
    let f = fixture();
    let c = reference_config(&f["packing"]["referenceConfig"]);
    let mut golden = hex_words(f["packing"]["configRecordHex"].as_str().unwrap());
    // Historical fence strength became a flag; final reserved lanes gained these defaults.
    golden[95] = 1;
    golden[103] = 1.0f32.to_bits();
    assert_eq!(c.pack().as_slice(), golden);
    let w = &f["packing"]["referenceWorld"];
    let actual = WorldSettings {
        trail_persistence: n(w, "trailPersistence"),
        trail_diffusion: n(w, "trailDiffusion"),
        boundary_conditions: n(w, "boundaryConditions") as u32,
    }
    .pack(n(w, "sqrtWorldSize") as f32, n(w, "configCount") as u32);
    assert_eq!(
        actual.as_slice(),
        hex_words(f["packing"]["worldRecordHex"].as_str().unwrap())
    );
    let l: Value = serde_json::from_str(include_str!("fixtures/layout.fixture.json")).unwrap();
    assert_eq!(l["structs"]["ConfigData"]["size"], 416);
    assert_eq!(l["structs"]["WorldData"]["size"], 32);
    let members = l["structs"]["ConfigData"]["members"].as_array().unwrap();
    assert_eq!(
        members
            .iter()
            .map(|r| r["floatIndex"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 80, 84, 88, 92, 96, 100]
    );
}
#[test]
fn frozen_historical_preset_reader_oracle() {
    let f: Value = serde_json::from_str(include_str!("fixtures/presets.fixture.json")).unwrap();
    for (name, r) in f["presets"].as_object().unwrap() {
        let text = std::fs::read_to_string(format!(
            "{}/tests/fixtures/historical/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let mut doc: Value = serde_json::from_str(&text).unwrap();
        // Bypass intentional v8 mutation migration to test the historical reader independently.
        doc["version"] = json!(9);
        let p = Project::from_json(&doc.to_string()).unwrap();
        let expected = reference_config(&r["config"]);
        assert_eq!(p.configs.len(), r["configCount"].as_u64().unwrap() as usize);
        assert_eq!(p.config().pack(), expected.pack(), "{name}");
        assert_eq!(
            p.world.trail_persistence,
            n(&r["world"], "trailPersistence")
        );
        assert_eq!(p.world.trail_diffusion, n(&r["world"], "trailDiffusion"));
    }
}
#[test]
fn preset_menu_categories_cover_all_entries_without_renaming_projects() {
    let catalog = primordial_soup::config::preset_catalog();
    let manifest: Value =
        serde_json::from_str(include_str!("../assets/presets/manifest.json")).unwrap();
    assert_eq!(catalog.len(), 131);
    assert_eq!(catalog.iter().filter(|p| p.category == "Core").count(), 22);
    assert_eq!(
        catalog.iter().filter(|p| p.category == "Archive").count(),
        109
    );
    for category in manifest["categories"].as_array().unwrap() {
        let name = category["name"].as_str().unwrap();
        let expected: Vec<_> = category["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        let visible: Vec<_> = catalog
            .iter()
            .filter(|p| p.category == name)
            .map(|p| {
                assert_eq!(p.project.name, p.name);
                p.name
            })
            .collect();
        assert!(!visible.is_empty());
        assert_eq!(visible, expected, "File → Load → {name}");
    }
}
#[test]
fn all_shipped_presets_and_rejection_boundary() {
    let all = presets();
    assert_eq!(all.len(), 131);
    for (name, p) in all {
        p.validate().unwrap();
        let round = Project::from_json(&p.to_json().unwrap()).unwrap();
        assert_eq!(round.configs, p.configs, "{name}");
        assert_eq!(round.world, p.world, "{name}");
    }
    let mut doc: Value = serde_json::from_str(&Project::default().to_json().unwrap()).unwrap();
    for (path, bad) in [
        (vec!["configs", "0", "force2"], json!(5)),
        (
            vec!["configs", "0", "misc", "cohorts"],
            json!(2147483648u64),
        ),
        (
            vec!["configs", "0", "force2", "initial_conditions"],
            json!(1.5),
        ),
        (vec!["world", "boundary_conditions"], json!(4)),
        (vec!["configs", "0", "rule"], json!([0, 1])),
        (vec!["configs", "0", "sensor", "gain"], json!(1e100)),
    ] {
        let mut d = doc.clone();
        let mut v = &mut d;
        for k in path {
            v = if k == "0" { &mut v[0] } else { &mut v[k] };
        }
        *v = bad;
        assert!(Project::from_json(&d.to_string()).is_err());
    }
    doc["configs"] = json!([]);
    assert!(Project::from_json(&doc.to_string()).is_err());
}
#[test]
fn history_gesture_preview_branch_and_bound() {
    let mut h = History::default();
    let p = Project::default();
    h.seed(p.clone());
    let mut q = p.clone();
    q.name = "first".into();
    assert_eq!(
        h.record(p.clone(), q.clone(), "first", Some("gain"), 0.0),
        RecordOutcome::Appended
    );
    let mut r = q.clone();
    r.name = "second".into();
    assert_eq!(
        h.record(q, r.clone(), "second", Some("gain"), 1400.0),
        RecordOutcome::Coalesced
    );
    assert_eq!(h.depth(), 2);
    assert_eq!(h.undo().unwrap(), p);
    assert_eq!(h.redo_label(), "second");
    assert_eq!(h.redo().unwrap(), r);
    let mut preview = r.clone();
    preview.name = "preview".into();
    let mut final_p = preview.clone();
    final_p.name = "committed".into();
    h.record(preview.clone(), final_p, "commit", None, 1500.0);
    assert_eq!(h.undo().unwrap(), preview);
    assert_eq!(h.undo_label(), "second");
    let mut last = preview;
    for i in 0..150 {
        let mut next = last.clone();
        next.name = i.to_string();
        h.record(last, next.clone(), "new", None, 2000.0 + i as f64);
        last = next;
    }
    assert_eq!(h.depth(), 100);
    assert!(!h.can_redo());
    let mut undone = 0;
    while h.undo().is_some() {
        undone += 1;
    }
    assert_eq!(undone, 99);
}
