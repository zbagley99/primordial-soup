use primordial_soup::{
    archive::{self, Event},
    config::Project,
    export,
};
use serde_json::{Value, json};
#[test]
fn source_archive_hash_and_metadata_exclusion() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/archive.fixture.json")).unwrap();
    for item in fixture["fixtures"].as_array().unwrap() {
        let mut p = Project::from_json(&item["document"].to_string()).unwrap();
        assert_eq!(archive::hash_state(&p), item["hash"].as_str().unwrap());
        let hash = archive::hash_state(&p);
        p.name = "renamed".into();
        p.notes = "changed notes".into();
        assert_eq!(archive::hash_state(&p), hash);
        p.config_mut().sensor_gain += 0.125;
        assert_ne!(archive::hash_state(&p), hash);
    }
}
#[test]
fn graph_preserves_first_parent_and_undo_branch_visits() {
    let p = Project::default();
    let mut b = p.clone();
    b.config_mut().sensor_gain += 1.0;
    let mut c = p.clone();
    c.config_mut().sensor_gain += 2.0;
    let ah = archive::hash_state(&p);
    let bh = archive::hash_state(&b);
    let ch = archive::hash_state(&c);
    let event = |project: &Project, from: Option<&str>, kind: &str| Event {
        session: "fixture-session".into(),
        at: 1000.0,
        kind: kind.into(),
        label: kind.into(),
        hash: Some(archive::hash_state(project)),
        from: from.map(str::to_owned),
        state: Some(archive::state(project)),
        field: None,
        value: None,
        previous: None,
    };
    let events = [
        event(&p, None, "enter"),
        event(&b, Some(&ah), "act"),
        event(&p, Some(&bh), "undo"),
        event(&c, Some(&ah), "act"),
        event(&b, Some(&ch), "enter"),
    ];
    let doc: Value = serde_json::from_str(&archive::document(&events).unwrap()).unwrap();
    assert_eq!(doc["version"], 2);
    assert_eq!(doc["nodeCount"], 3);
    assert_eq!(doc["rootCount"], 1);
    assert_eq!(doc["visitCount"], 5);
    assert_eq!(doc["nodes"][1]["parent"], ah);
    assert_eq!(doc["nodes"][2]["parent"], ah);
    assert_eq!(doc["visits"][4]["visit"]["repeat"], true);
    assert_eq!(
        doc["nodes"][1]["delta"]["configs"][0]["sensorGain"],
        b.config().sensor_gain
    );
    assert_eq!(
        doc["derivation"]["rule.wgsl"].as_str().unwrap(),
        include_str!("../shaders/rule.wgsl")
    );
    assert!(doc["exportedAt"].as_str().unwrap().contains('T'));
}
#[test]
fn freely_positioned_crop_preserves_pixels_and_rejects_outside_origin() {
    let pixels: Vec<u8> = (0..16).flat_map(|n| [n, 0, 0, 255]).collect();
    let crop = export::crop_rgba_at(&pixels, [4, 4], [1, 2], [2, 2]).unwrap();
    assert_eq!(
        crop.chunks_exact(4).map(|p| p[0]).collect::<Vec<_>>(),
        vec![9, 10, 13, 14]
    );
    assert!(export::crop_rgba_at(&pixels, [4, 4], [3, 0], [2, 2]).is_err());
    assert!(export::crop_rgba_at(&pixels, [4, 4], [u32::MAX, 0], [1, 1]).is_err());
}
#[test]
fn reference_calibration_ladder_and_budget() {
    assert_eq!(
        primordial_soup::calibration::PROGRESSION,
        [
            (0.1, 1),
            (0.25, 1),
            (0.25, 10),
            (0.6, 10),
            (0.6, 15),
            (1.0, 15),
            (1.0, 20)
        ]
    );
    assert!((primordial_soup::calibration::BUDGET_MS - 11.69).abs() < 1e-12);
    let _ = json!({"provenance":"Fluoddity progression.ts"});
}
