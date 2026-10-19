use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use primordial_soup::{
    config::Project,
    share::{decode_binary, decode_text, encode_binary, encode_link},
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/share.fixture.json")).unwrap()
}
fn document(project: &Project) -> Value {
    serde_json::from_str(&project.to_json().unwrap()).unwrap()
}

#[test]
fn original_codec_vectors_and_scalar_defaults() {
    for vector in fixture()["vectors"].as_array().unwrap() {
        let expected = Project::from_json(&vector["document"].to_string()).unwrap();
        let hash = vector["hash"].as_str().unwrap();
        let actual = decode_text(hash).unwrap().unwrap();
        assert_eq!(
            document(&actual),
            document(&expected),
            "codec {}",
            vector["codec"]
        );
        assert_eq!(actual.name, "Shared Link");
        assert_eq!(
            actual.configs[1].mutation_seed.to_bits(),
            0.024925940576650873_f64.to_bits()
        );
        if vector["codec"].as_u64().unwrap() < 4 {
            assert_eq!(actual.configs[1].trail_weight, 1.0);
        } else {
            assert_eq!(
                encode_link(&actual).unwrap(),
                hash,
                "exact original encoded bytes"
            );
        }
    }
}

#[test]
fn original_legacy_unicode_and_clipboard_transports() {
    let f = fixture();
    let expected = Project::from_json(&f["document"].to_string()).unwrap();
    assert_eq!(
        document(&decode_text(f["legacy"].as_str().unwrap()).unwrap().unwrap()),
        document(&expected)
    );
    let hash = f["vectors"][3]["hash"].as_str().unwrap();
    let wrapped = format!(
        " https://fluoddity.com/?debug{}\n{} ",
        &hash[..50],
        &hash[50..]
    );
    for text in [
        hash.to_owned(),
        hash[1..].to_owned(),
        wrapped,
        f["document"].to_string(),
    ] {
        assert_eq!(
            document(&decode_text(&text).unwrap().unwrap()),
            document(&expected)
        );
    }
    for text in [
        "",
        "#about",
        "https://fluoddity.com/#about",
        "some other text",
    ] {
        assert!(decode_text(text).unwrap().is_none());
    }
}

#[test]
fn original_v8_link_preserves_reader_migration() {
    let f = fixture();
    let expected = Project::from_json(&f["historical"]["document"].to_string()).unwrap();
    let actual = decode_text(f["historical"]["hash"].as_str().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(document(&actual), document(&expected));
}

#[test]
fn malformed_and_nonfinite_payloads_fail() {
    let f = fixture();
    let hash = f["vectors"][3]["hash"].as_str().unwrap();
    let bytes = URL_SAFE_NO_PAD.decode(&hash[3..]).unwrap();
    // Every possible truncation, including within Unicode notes, must fail.
    for end in 0..bytes.len() {
        assert!(decode_binary(&bytes[..end]).is_err(), "truncation at {end}");
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_binary(&trailing).is_err());
    for (offset, nonfinite) in [
        (4, f64::INFINITY.to_le_bytes().to_vec()),
        (21, f32::NAN.to_le_bytes().to_vec()),
        (341, f64::NAN.to_le_bytes().to_vec()),
    ] {
        let mut damaged = bytes.clone();
        damaged[offset..offset + nonfinite.len()].copy_from_slice(&nonfinite);
        assert!(decode_binary(&damaged).is_err());
    }
    let mut future = bytes.clone();
    future[0] = 5;
    assert!(decode_binary(&future).is_err());
    for text in ["#b=", "#b=***", "#b=A", "#c=", "#c=***", "#c=A"] {
        assert!(decode_text(text).is_err(), "{text}");
    }
    let legacy = f["legacy"].as_str().unwrap();
    assert!(decode_text(&legacy[..legacy.len() / 2]).is_err());
    assert!(decode_binary(&vec![0; 4 * 1024 * 1024 + 1]).is_err());
    let mut project = Project::from_json(&f["document"].to_string()).unwrap();
    project.notes = "x".repeat(4 * 1024 * 1024);
    assert!(encode_binary(&project).is_err());
}
