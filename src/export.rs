//! Screenshot and custom-save transfer helpers, with original Fluoddity geometry.
use crate::{
    config::Project,
    qr::{QrCode, QrCodeEcc, QrSegment, Version},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const MAX_IMAGE_BYTES: usize = 256 * 1024 * 1024;
const MAX_ARCHIVE_BYTES: usize = 16 * 1024 * 1024;
const MAX_SAVES: usize = 4096;

fn image_len(size: [u32; 2]) -> Result<usize> {
    ensure!(
        size[0] > 0 && size[1] > 0,
        "image dimensions must be positive"
    );
    let bytes = (size[0] as usize)
        .checked_mul(size[1] as usize)
        .and_then(|n| n.checked_mul(4))
        .context("image dimensions overflow")?;
    ensure!(
        bytes <= MAX_IMAGE_BYTES,
        "image exceeds 256 MiB pixel limit"
    );
    Ok(bytes)
}

/// Center crop of row-major RGBA pixels. Output has the requested exact size.
pub fn crop_rgba(pixels: &[u8], size: [u32; 2], crop: [u32; 2]) -> Result<Vec<u8>> {
    ensure!(
        crop[0] <= size[0] && crop[1] <= size[1],
        "crop exceeds image dimensions"
    );
    crop_rgba_at(
        pixels,
        size,
        [(size[0] - crop[0]) / 2, (size[1] - crop[1]) / 2],
        crop,
    )
}

/// Pixel-exact freely positioned crop, with no changes to the source buffer.
pub fn crop_rgba_at(
    pixels: &[u8],
    size: [u32; 2],
    origin: [u32; 2],
    crop: [u32; 2],
) -> Result<Vec<u8>> {
    ensure!(
        pixels.len() == image_len(size)?,
        "RGBA byte length does not match dimensions"
    );
    let output_len = image_len(crop)?;
    ensure!(
        crop[0] <= size[0] && crop[1] <= size[1],
        "crop exceeds source image"
    );
    ensure!(
        origin[0] <= size[0] - crop[0] && origin[1] <= size[1] - crop[1],
        "crop origin exceeds source image"
    );
    let row_bytes = crop[0] as usize * 4;
    let mut output = Vec::with_capacity(output_len);
    for y in 0..crop[1] {
        let start = ((origin[1] + y) as usize * size[0] as usize + origin[0] as usize) * 4;
        output.extend_from_slice(&pixels[start..start + row_bytes]);
    }
    Ok(output)
}

fn make_qr(url: &str) -> Result<QrCode> {
    ensure!(!url.is_empty(), "QR payload is empty");
    ensure!(
        url.len() <= 2331,
        "share link exceeds QR capacity at ECC M; use fewer configurations or shorter notes"
    );
    let segments = [QrSegment::make_bytes(url.as_bytes())];
    QrCode::encode_segments_advanced(
        &segments,
        QrCodeEcc::Medium,
        Version::MIN,
        Version::MAX,
        None,
        false,
    )
    .map_err(|error| anyhow::anyhow!("share link exceeds QR capacity at ECC M: {error}. Export project JSON instead; no network modes were removed"))
}

/// Stamp side and overlay selection minimum, using source module4/quiet4/pad8.
pub fn qr_geometry(url: &str) -> Result<(u32, u32)> {
    let qr = make_qr(url)?;
    let stamp = (qr.size() as u32 + 8) * 4 + 16;
    Ok((stamp, stamp + 40))
}

/// Stamp bottom-right with a white plate, full quiet zone, and 16px edge inset.
/// All checks run before pixels change; a small crop returns a useful error.
pub fn stamp_qr(pixels: &mut [u8], size: [u32; 2], url: &str) -> Result<()> {
    ensure!(
        pixels.len() == image_len(size)?,
        "RGBA byte length does not match dimensions"
    );
    let qr = make_qr(url)?;
    let count = qr.size() as u32;
    let stamp = (count + 8) * 4 + 16;
    let needed = stamp + 32;
    ensure!(
        size[0] >= needed && size[1] >= needed,
        "a {}x{} image cannot carry a {stamp}px stamp; needs at least {needed}x{needed}",
        size[0],
        size[1]
    );
    let origin = [size[0] - stamp - 16, size[1] - stamp - 16];
    let fill = |pixels: &mut [u8], x: u32, y: u32, side: u32, value: u8| {
        for row in y..y + side {
            let start = (row as usize * size[0] as usize + x as usize) * 4;
            for pixel in pixels[start..start + side as usize * 4].chunks_exact_mut(4) {
                pixel.copy_from_slice(&[value, value, value, 255]);
            }
        }
    };
    fill(pixels, origin[0], origin[1], stamp, 255);
    let grid = [origin[0] + 24, origin[1] + 24];
    for row in 0..count {
        for col in 0..count {
            if qr.get_module(col as i32, row as i32) {
                fill(pixels, grid[0] + col * 4, grid[1] + row * 4, 4, 0);
            }
        }
    }
    Ok(())
}

/// Encode RGBA without metadata or palette conversion; portable to wasm.
pub fn encode_png(pixels: &[u8], size: [u32; 2]) -> Result<Vec<u8>> {
    ensure!(
        pixels.len() == image_len(size)?,
        "RGBA byte length does not match dimensions"
    );
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, size[0], size[1]);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .context("PNG header encoding failed")?;
        writer
            .write_image_data(pixels)
            .context("PNG pixel encoding failed")?;
    }
    Ok(bytes)
}

/// Same forbidden filename characters, trim, trailing-dot removal and 120 UTF16
/// unit limit as persistence.ts. Never returns a dangling surrogate pair.
pub fn sanitize_name(name: &str) -> String {
    let clean: String = name
        .trim()
        .chars()
        .filter(|c| !"\\/:*?\"<>|".contains(*c))
        .collect();
    let clean = clean.trim().trim_end_matches('.');
    let mut result = String::new();
    let mut units = 0;
    for character in clean.chars() {
        units += character.len_utf16();
        if units > 120 {
            break;
        }
        result.push(character);
    }
    result
}

/// Bulk envelope around original saveTransfer.ts {filename,text} records.
/// Config format is carried inside each text; this envelope has its own version.
pub fn export_saves(saves: &BTreeMap<String, Project>) -> Result<String> {
    ensure!(saves.len() <= MAX_SAVES, "too many custom saves");
    let mut files = Vec::new();
    let mut names = BTreeSet::new();
    let mut text_bytes = 0usize;
    for (name, project) in saves {
        let name = sanitize_name(name);
        if name.is_empty() || !names.insert(name.clone()) {
            continue;
        }
        let text = format!("{}\n", project.to_json()?);
        text_bytes = text_bytes
            .checked_add(text.len())
            .context("archive size overflow")?;
        ensure!(
            text_bytes <= MAX_ARCHIVE_BYTES,
            "custom saves exceed 16 MiB archive limit"
        );
        files.push(json!({"filename":format!("{name}.json"),"text":text}));
    }
    let text = serde_json::to_string_pretty(&json!({"version":1,"files":files}))?;
    ensure!(
        text.len() <= MAX_ARCHIVE_BYTES,
        "custom saves exceed 16 MiB archive limit"
    );
    Ok(format!("{text}\n"))
}

/// Validate all eligible incoming files before returning any saves. First named
/// file wins a sanitized duplicate, matching original planImport order.
pub fn import_saves(text: &str) -> Result<BTreeMap<String, Project>> {
    ensure!(
        text.len() <= MAX_ARCHIVE_BYTES,
        "custom saves exceed 16 MiB archive limit"
    );
    let envelope: Value = serde_json::from_str(text).context("invalid custom-save archive JSON")?;
    ensure!(
        envelope["version"].as_u64() == Some(1),
        "unsupported custom-save archive version"
    );
    let files = envelope["files"]
        .as_array()
        .context("custom-save archive requires files array")?;
    ensure!(files.len() <= MAX_SAVES, "too many custom saves");
    let mut saves = BTreeMap::new();
    for file in files {
        let filename = file["filename"]
            .as_str()
            .context("archive file requires filename")?;
        let base = filename.rsplit('/').next().unwrap_or(filename);
        let stem = if base.to_ascii_lowercase().ends_with(".json") {
            &base[..base.len() - 5]
        } else {
            base
        };
        let name = sanitize_name(stem);
        if name.is_empty() || saves.contains_key(&name) {
            continue;
        }
        let text = file["text"]
            .as_str()
            .context("archive file requires text")?;
        ensure!(
            text.len() <= 4 * 1024 * 1024,
            "save {filename} exceeds 4 MiB"
        );
        let mut project =
            Project::from_json(text).with_context(|| format!("invalid save {filename}"))?;
        project.name = name.clone();
        saves.insert(name, project);
    }
    Ok(saves)
}

/// Optional native import through the system QR decoder. No shell is invoked.
#[cfg(not(target_arch = "wasm32"))]
pub fn decode_image_file(path: &std::path::Path) -> Result<Project> {
    ensure!(path.is_file(), "image file does not exist");
    ensure!(
        std::fs::metadata(path)?.len() <= 64 * 1024 * 1024,
        "image file exceeds 64 MiB"
    );
    let decoded = std::process::Command::new("zbarimg")
        .args(["--quiet", "--raw", "--"])
        .arg(path)
        .output()
        .context("image import needs zbarimg installed")?;
    ensure!(decoded.status.success(), "no readable QR code in image");
    ensure!(
        decoded.stdout.len() <= 8 * 1024 * 1024,
        "decoded QR output exceeds input limit"
    );
    let output = std::str::from_utf8(&decoded.stdout).context("QR did not contain UTF-8 text")?;
    for text in output.lines() {
        if let Some(project) = crate::share::decode_text(text)? {
            return Ok(project);
        }
    }
    anyhow::bail!("image QR code does not contain a Fluoddity project")
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}
fn put16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend(value.to_le_bytes());
}
fn put32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend(value.to_le_bytes());
}
fn get16(bytes: &[u8], at: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .context("truncated ZIP header")?
            .try_into()
            .unwrap(),
    ))
}
fn get32(bytes: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .context("truncated ZIP header")?
            .try_into()
            .unwrap(),
    ))
}

/// Original bulk-export transport: ZIP with one v9 JSON file per named save.
/// Stored entries avoid costly compression and work in original fflate readers.
pub fn export_zip(saves: &BTreeMap<String, Project>) -> Result<Vec<u8>> {
    let archive: Value = serde_json::from_str(&export_saves(saves)?)?;
    let files = archive["files"].as_array().unwrap();
    let mut bytes = Vec::new();
    let mut central = Vec::new();
    for file in files {
        let filename = file["filename"].as_str().unwrap().as_bytes();
        let text = file["text"].as_str().unwrap().as_bytes();
        let crc = crc32(text);
        let offset = bytes.len() as u32;
        // Local file header, UTF8 filename, no encryption or descriptor.
        put32(&mut bytes, 0x04034b50);
        put16(&mut bytes, 20);
        put16(&mut bytes, 0x800);
        put16(&mut bytes, 0);
        put16(&mut bytes, 0);
        put16(&mut bytes, 0x21);
        put32(&mut bytes, crc);
        put32(&mut bytes, text.len() as u32);
        put32(&mut bytes, text.len() as u32);
        put16(&mut bytes, filename.len() as u16);
        put16(&mut bytes, 0);
        bytes.extend(filename);
        bytes.extend(text);
        put32(&mut central, 0x02014b50);
        put16(&mut central, 20);
        put16(&mut central, 20);
        put16(&mut central, 0x800);
        put16(&mut central, 0);
        put16(&mut central, 0);
        put16(&mut central, 0x21);
        put32(&mut central, crc);
        put32(&mut central, text.len() as u32);
        put32(&mut central, text.len() as u32);
        put16(&mut central, filename.len() as u16);
        put16(&mut central, 0);
        put16(&mut central, 0);
        put16(&mut central, 0);
        put16(&mut central, 0);
        put32(&mut central, 0);
        put32(&mut central, offset);
        central.extend(filename);
        ensure!(
            bytes.len() + central.len() + 22 <= MAX_ARCHIVE_BYTES,
            "ZIP exceeds 16 MiB archive limit"
        );
    }
    let central_at = bytes.len() as u32;
    let central_len = central.len() as u32;
    bytes.extend(central);
    put32(&mut bytes, 0x06054b50);
    put16(&mut bytes, 0);
    put16(&mut bytes, 0);
    put16(&mut bytes, files.len() as u16);
    put16(&mut bytes, files.len() as u16);
    put32(&mut bytes, central_len);
    put32(&mut bytes, central_at);
    put16(&mut bytes, 0);
    Ok(bytes)
}

/// Read source ZIPs (stored or deflated), with central-directory sizes so ZIP
/// data descriptors work. No path extraction: names only become sanitized keys.
pub fn import_zip(bytes: &[u8]) -> Result<BTreeMap<String, Project>> {
    use std::io::Read;
    ensure!(
        bytes.len() <= MAX_ARCHIVE_BYTES && bytes.len() >= 22,
        "ZIP is truncated or exceeds 16 MiB"
    );
    let search_start = bytes.len().saturating_sub(65557);
    let end = (search_start..=bytes.len() - 22)
        .rev()
        .find(|&at| {
            bytes[at..at + 4] == [0x50, 0x4b, 0x05, 0x06]
                && get16(bytes, at + 20).is_ok_and(|len| at + 22 + len as usize == bytes.len())
        })
        .context("ZIP end directory not found")?;
    ensure!(
        get16(bytes, end + 4)? == 0 && get16(bytes, end + 6)? == 0,
        "multi-disk ZIP unsupported"
    );
    let count = get16(bytes, end + 10)? as usize;
    ensure!(
        count <= MAX_SAVES && get16(bytes, end + 8)? as usize == count,
        "invalid ZIP entry count"
    );
    let central_len = get32(bytes, end + 12)? as usize;
    let mut at = get32(bytes, end + 16)? as usize;
    let central_end = at
        .checked_add(central_len)
        .context("ZIP directory overflow")?;
    ensure!(central_end == end, "invalid ZIP directory bounds");
    let mut files = Vec::new();
    let mut total = 0usize;
    for _ in 0..count {
        ensure!(
            at + 46 <= central_end && get32(bytes, at)? == 0x02014b50,
            "invalid ZIP central header"
        );
        let flags = get16(bytes, at + 8)?;
        ensure!(flags & 1 == 0, "encrypted ZIP unsupported");
        let method = get16(bytes, at + 10)?;
        ensure!(
            method == 0 || method == 8,
            "unsupported ZIP compression method {method}"
        );
        let crc = get32(bytes, at + 16)?;
        let compressed = get32(bytes, at + 20)? as usize;
        let size = get32(bytes, at + 24)? as usize;
        let name_len = get16(bytes, at + 28)? as usize;
        let extra_len = get16(bytes, at + 30)? as usize;
        let comment_len = get16(bytes, at + 32)? as usize;
        ensure!(
            get16(bytes, at + 34)? == 0,
            "multi-disk ZIP entry unsupported"
        );
        let local = get32(bytes, at + 42)? as usize;
        let next = at + 46 + name_len + extra_len + comment_len;
        ensure!(next <= central_end, "truncated ZIP filename or metadata");
        let filename = std::str::from_utf8(&bytes[at + 46..at + 46 + name_len])
            .context("ZIP filenames must be UTF-8")?;
        at = next;
        if filename.ends_with('/') || !filename.to_ascii_lowercase().ends_with(".json") {
            continue;
        }
        ensure!(
            size <= 4 * 1024 * 1024 && compressed <= MAX_ARCHIVE_BYTES,
            "ZIP save exceeds size limit"
        );
        total = total.checked_add(size).context("ZIP total size overflow")?;
        ensure!(total <= MAX_ARCHIVE_BYTES, "ZIP expands beyond 16 MiB");
        ensure!(
            local.checked_add(30).is_some_and(|end| end <= bytes.len())
                && get32(bytes, local)? == 0x04034b50,
            "invalid ZIP local header"
        );
        ensure!(
            get16(bytes, local + 8)? == method && get16(bytes, local + 6)? == flags,
            "ZIP header disagreement"
        );
        let local_name_len = get16(bytes, local + 26)? as usize;
        let start = local + 30 + local_name_len + get16(bytes, local + 28)? as usize;
        let data_end = start
            .checked_add(compressed)
            .context("ZIP entry overflow")?;
        ensure!(data_end <= end && start <= data_end, "truncated ZIP entry");
        ensure!(
            bytes.get(local + 30..local + 30 + local_name_len) == Some(filename.as_bytes()),
            "ZIP filename disagreement"
        );
        let data = &bytes[start..data_end];
        let text = if method == 0 {
            ensure!(compressed == size, "stored ZIP size disagreement");
            data.to_vec()
        } else {
            let mut output = Vec::with_capacity(size);
            flate2::read::DeflateDecoder::new(data)
                .take(size as u64 + 1)
                .read_to_end(&mut output)
                .context("invalid ZIP deflate stream")?;
            ensure!(output.len() == size, "ZIP expansion size disagreement");
            output
        };
        ensure!(crc32(&text) == crc, "ZIP save checksum mismatch");
        let text = String::from_utf8(text).context("ZIP save is not UTF-8 JSON")?;
        files.push(json!({"filename":filename,"text":text}));
    }
    ensure!(
        at == central_end,
        "ZIP entry count disagrees with directory"
    );
    import_saves(&serde_json::to_string(&json!({"version":1,"files":files}))?)
}
