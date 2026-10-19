use base64::{Engine, engine::general_purpose::STANDARD};
use primordial_soup::{
    config::Project,
    export::{
        crop_rgba, encode_png, export_saves, export_zip, import_saves, import_zip, qr_geometry,
        sanitize_name, stamp_qr,
    },
};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn centered_crop_and_png_preserve_exact_rgba() {
    let pixels: Vec<u8> = (0..5 * 3).flat_map(|i| [i as u8, 22, 33, 255]).collect();
    let cropped = crop_rgba(&pixels, [5, 3], [3, 1]).unwrap();
    assert_eq!(
        cropped,
        vec![6, 22, 33, 255, 7, 22, 33, 255, 8, 22, 33, 255]
    );
    let png = encode_png(&cropped, [3, 1]).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().unwrap();
    let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut decoded).unwrap();
    assert_eq!(&decoded[..info.buffer_size()], &cropped);
    assert!(crop_rgba(&pixels, [5, 3], [6, 1]).is_err());
    assert!(crop_rgba(&pixels[..3], [5, 3], [1, 1]).is_err());
    assert!(encode_png(&[], [0, 1]).is_err());
}

#[test]
fn original_stamp_geometry_white_plate_and_failure_atomicity() {
    // Byte-mode ECC M version 2 has 25 modules. Original qrStamp.ts arithmetic:
    // (25+2*quiet4)*module4 + 2*pad8 = 148; overlay adds 40.
    let url = "https://fluoddity.com/";
    assert_eq!(qr_geometry(url).unwrap(), (148, 188));
    let mut image = vec![17; 512 * 512 * 4];
    stamp_qr(&mut image, [512, 512], url).unwrap();
    let origin = 512 - 148 - 16;
    let pixel = |x: usize, y: usize| &image[(y * 512 + x) * 4..(y * 512 + x) * 4 + 4];
    assert_eq!(pixel(origin, origin), &[255, 255, 255, 255]);
    assert_eq!(pixel(origin + 23, origin + 24), &[255, 255, 255, 255]);
    assert_eq!(pixel(origin + 24, origin + 24), &[0, 0, 0, 255]);
    assert_eq!(pixel(511, 511), &[17, 17, 17, 17]);
    let mut too_small = vec![17; 179 * 179 * 4];
    let before = too_small.clone();
    assert!(stamp_qr(&mut too_small, [179, 179], url).is_err());
    assert_eq!(too_small, before);
    assert!(qr_geometry(&"x".repeat(2332)).is_err());
}

#[test]
fn original_sanitization_first_duplicate_wins_and_atomic_validation() {
    assert_eq!(sanitize_name("  a:b/c*?\"<>|\\...  "), "abc");
    assert_eq!(sanitize_name(&"x".repeat(150)).len(), 120);
    assert_eq!(sanitize_name("..."), "");
    let project = Project::default();
    let text = project.to_json().unwrap();
    let archive = json!({"version":1,"files":[
        {"filename":"nested/a:b.JSON","text":text},
        {"filename":"ab.json","text":"invalid duplicate is skipped"},
        {"filename":"?.json","text":"unnamed is skipped"}
    ]});
    let imported = import_saves(&archive.to_string()).unwrap();
    assert_eq!(imported.len(), 1);
    assert_eq!(imported["ab"].name, "ab");
    let mut broken = archive;
    broken["files"]
        .as_array_mut()
        .unwrap()
        .push(json!({"filename":"different.json","text":"invalid"}));
    assert!(import_saves(&broken.to_string()).is_err());
    let saves = BTreeMap::from([("a:b".to_string(), project)]);
    assert_eq!(
        import_saves(&export_saves(&saves).unwrap()).unwrap()["ab"]
            .to_json()
            .unwrap(),
        saves["a:b"].to_json().unwrap()
    );
    assert_eq!(
        import_zip(&export_zip(&saves).unwrap()).unwrap()["ab"]
            .to_json()
            .unwrap(),
        saves["a:b"].to_json().unwrap()
    );
    assert!(import_saves(r#"{"version":2,"files":[]}"#).is_err());
}

#[test]
fn independent_fflate_deflate_archive_is_compatible() {
    // Frozen fflate 0.8.3 zipSync(level:6), generated outside this Rust codec.
    // Contains original Starcrossed JSON under nested/a:b.JSON, invalid ab.json
    // duplicate and readme.txt. Matches source filename/duplicate policy.
    let bytes = STANDARD.decode(FFLATE_ZIP).unwrap();
    let imported = import_zip(&bytes).unwrap();
    let expected =
        Project::from_json(include_str!("../assets/presets/Core/Starcrossed.json")).unwrap();
    assert_eq!(imported.len(), 1);
    assert_eq!(
        imported["ab"].to_json().unwrap(),
        expected.to_json().unwrap()
    );
    for end in [0, 21, 30, bytes.len() / 2, bytes.len() - 1] {
        assert!(import_zip(&bytes[..end]).is_err());
    }
    let mut corrupt = bytes.clone();
    corrupt[100] ^= 1;
    assert!(import_zip(&corrupt).is_err());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn external_zbar_decodes_share_stamp_into_original_project() {
    let project = Project::default();
    let url = format!(
        "https://fluoddity.com/{}",
        primordial_soup::share::encode_link(&project).unwrap()
    );
    let mut pixels = vec![29; 800 * 800 * 4];
    stamp_qr(&mut pixels, [800, 800], &url).unwrap();
    let png = encode_png(&pixels, [800, 800]).unwrap();
    let path = std::env::temp_dir().join(format!(
        "soup-qr-export-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, png).unwrap();
    let decoded = primordial_soup::export::decode_image_file(&path);
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        decoded.unwrap().to_json().unwrap(),
        project.to_json().unwrap()
    );
}

const FFLATE_ZIP: &str = "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";
