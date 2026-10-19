//! Fluoddity share fragments: positional codec 1..4 and legacy lz-string links.
//! Scalars stay f64: rounding a mutation seed changes the generated rule.
use crate::config::Project;
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde_json::{Value, json};

const LIMIT: usize = 4 * 1024 * 1024;
const HEADER: usize = 21;
const SCALARS: [(&str, &str); 19] = [
    ("sensor", "gain"),
    ("sensor", "angle"),
    ("sensor", "distance"),
    ("sensor", "mutation_scale"),
    ("force", "global_mult"),
    ("force", "drag"),
    ("force", "strafe"),
    ("force", "axial"),
    ("misc", "lateral"),
    ("misc", "hazard_rate"),
    ("misc", "mutation_seed"),
    ("force2", "gravity_force"),
    ("force2", "gravity_strafe"),
    ("misc2", "color_sensitivity"),
    ("misc2", "sensor_angle_jitter"),
    ("misc2", "sensor_distance_jitter"),
    ("misc3", "gravity_trails"),
    ("misc3", "color_offset"),
    ("misc3", "trail_weight"),
];
const FLAGS: [(&str, &str); 3] = [
    ("force2", "cohort_fences"),
    ("misc2", "color_by_cohort"),
    ("misc3", "radial_gravity"),
];

/// Encode the current project with the original codec 4 byte layout.
pub fn encode_binary(project: &Project) -> Result<Vec<u8>> {
    if project.extended() {
        let json = project.to_json()?;
        ensure!(json.len() + 4 <= LIMIT, "share payload exceeds 4 MiB");
        let mut bytes = b"PS10".to_vec();
        bytes.extend(json.as_bytes());
        return Ok(bytes);
    }
    let doc: Value = serde_json::from_str(&project.to_json()?)?;
    let configs = doc["configs"].as_array().context("missing configs")?;
    ensure!(
        configs.len() <= u16::MAX as usize,
        "too many configurations"
    );
    let notes = doc["notes"].as_str().unwrap_or("").as_bytes();
    let size = HEADER + configs.len() * 478 + 4 + notes.len();
    ensure!(size <= LIMIT, "share payload exceeds 4 MiB");
    let mut out = Vec::with_capacity(size);
    out.extend([4, doc["version"].as_u64().context("missing version")? as u8]);
    out.extend((configs.len() as u16).to_le_bytes());
    for key in ["trail_persistence", "trail_diffusion"] {
        out.extend(
            doc["world"][key]
                .as_f64()
                .context("missing world scalar")?
                .to_le_bytes(),
        );
    }
    out.push(
        doc["world"]["boundary_conditions"]
            .as_u64()
            .context("missing boundary")? as u8,
    );
    for config in configs {
        let rule = config["rule"].as_array().context("missing rule")?;
        ensure!(rule.len() == 80, "rule requires 80 floats");
        for value in rule {
            let value = value.as_f64().context("invalid rule float")? as f32;
            ensure!(
                value.is_finite(),
                "rule float cannot be represented as float32"
            );
            out.extend(value.to_le_bytes());
        }
        for (group, key) in SCALARS {
            out.extend(
                config[group][key]
                    .as_f64()
                    .context("missing config scalar")?
                    .to_le_bytes(),
            );
        }
        out.extend(
            (config["misc"]["cohorts"]
                .as_u64()
                .context("missing cohorts")? as u32)
                .to_le_bytes(),
        );
        out.push(
            config["force2"]["initial_conditions"]
                .as_u64()
                .context("missing initial conditions")? as u8,
        );
        let mut flags = 0;
        for (bit, (group, key)) in FLAGS.into_iter().enumerate() {
            if config[group][key].as_bool().unwrap_or(false) {
                flags |= 1 << bit;
            }
        }
        out.push(flags);
    }
    out.extend((notes.len() as u32).to_le_bytes());
    out.extend(notes);
    Ok(out)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.at.checked_add(N).context("payload length overflow")?;
        let bytes = self
            .bytes
            .get(self.at..end)
            .context("truncated share payload")?;
        self.at = end;
        Ok(bytes.try_into().unwrap())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn f64(&mut self) -> Result<f64> {
        let value = f64::from_le_bytes(self.take()?);
        ensure!(value.is_finite(), "nonfinite share scalar");
        Ok(value)
    }
}

/// Decode old scalar tables without inventing defaults; the project reader owns those.
pub fn decode_binary(bytes: &[u8]) -> Result<Project> {
    ensure!(bytes.len() <= LIMIT, "share payload exceeds 4 MiB");
    if bytes.starts_with(b"PS10") {
        return Project::from_json(std::str::from_utf8(&bytes[4..])?);
    }
    ensure!(bytes.len() >= HEADER, "truncated share header");
    let mut r = Reader { bytes, at: 0 };
    let codec = r.u8()?;
    ensure!((1..=4).contains(&codec), "unsupported share codec {codec}");
    let scalars = 15 + codec as usize;
    let version = r.u8()?;
    let count = u16::from_le_bytes(r.take()?) as usize;
    ensure!(
        (1..=256).contains(&count),
        "project requires 1..=256 configurations"
    );
    ensure!(
        bytes.len() >= HEADER + count * (326 + scalars * 8) + 4,
        "truncated configuration payload"
    );
    let mut doc = json!({"version":version,"world":{
        "trail_persistence":r.f64()?,"trail_diffusion":r.f64()?,"boundary_conditions":r.u8()?
    },"configs":[]});
    let mut configs = Vec::with_capacity(count);
    for _ in 0..count {
        let mut rule = Vec::with_capacity(80);
        for _ in 0..80 {
            let value = f32::from_le_bytes(r.take()?);
            ensure!(value.is_finite(), "nonfinite share rule");
            rule.push(value as f64);
        }
        let mut config =
            json!({"rule":rule,"sensor":{},"force":{},"misc":{},"force2":{},"misc2":{},"misc3":{}});
        for (group, key) in SCALARS.iter().take(scalars) {
            config[group][key] = json!(r.f64()?);
        }
        config["misc"]["cohorts"] = json!(u32::from_le_bytes(r.take()?));
        config["force2"]["initial_conditions"] = json!(r.u8()?);
        let flags = r.u8()?;
        for (bit, (group, key)) in FLAGS.into_iter().enumerate() {
            config[group][key] = json!(flags & (1 << bit) != 0);
        }
        configs.push(config);
    }
    let len = u32::from_le_bytes(r.take()?) as usize;
    ensure!(
        len <= LIMIT && r.at.checked_add(len) == Some(bytes.len()),
        "truncated notes or trailing share bytes"
    );
    let notes = std::str::from_utf8(&bytes[r.at..]).context("invalid UTF-8 share notes")?;
    doc["configs"] = json!(configs);
    if !notes.is_empty() {
        doc["notes"] = json!(notes);
    }
    let mut project = Project::from_json(&serde_json::to_string(&doc)?)?;
    project.name = "Shared Link".into();
    Ok(project)
}

pub fn encode_link(project: &Project) -> Result<String> {
    Ok(format!(
        "#b={}",
        URL_SAFE_NO_PAD.encode(encode_binary(project)?)
    ))
}

/// Accept full URLs, fragments, hard-wrapped links, and plain project JSON.
/// Unrelated text returns `None`; a damaged recognized link returns an error.
pub fn decode_text(text: &str) -> Result<Option<Project>> {
    ensure!(text.len() <= LIMIT * 2, "share text exceeds input limit");
    let text = text.trim();
    if text.starts_with('{') {
        ensure!(text.len() <= LIMIT, "project JSON exceeds 4 MiB");
        return Ok(Some(Project::from_json(text)?));
    }
    let fragment = text.split_once('#').map_or(text, |(_, tail)| tail);
    let fragment: String = fragment.chars().filter(|c| !c.is_whitespace()).collect();
    if let Some(payload) = fragment.strip_prefix("b=") {
        ensure!(!payload.is_empty(), "share link carries no data");
        let bytes = URL_SAFE_NO_PAD
            .decode(payload)
            .context("damaged share base64")?;
        return Ok(Some(decode_binary(&bytes)?));
    }
    if let Some(payload) = fragment.strip_prefix("c=") {
        ensure!(!payload.is_empty(), "share link carries no data");
        let json = decompress_legacy(payload)?;
        let mut project = Project::from_json(&json)?;
        project.name = "Shared Link".into();
        return Ok(Some(project));
    }
    Ok(None)
}

// Legacy decoder implements lz-string's algorithm (MIT, Copyright (c) 2013 pieroxy).
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// lz-string's URI transport stores a little-endian code stream in big-endian
// six-bit characters. Decode UTF-16 units before turning its JSON into UTF-8.
fn decompress_legacy(payload: &str) -> Result<String> {
    struct Bits {
        data: Vec<u8>,
        at: usize,
    }
    impl Bits {
        fn read(&mut self, width: usize) -> Result<usize> {
            ensure!(width <= 23, "legacy dictionary exceeds limit");
            let mut value = 0;
            for bit in 0..width {
                let digit = self
                    .data
                    .get(self.at / 6)
                    .context("truncated legacy link")?;
                value |= (((digit >> (5 - self.at % 6)) & 1) as usize) << bit;
                self.at += 1;
            }
            Ok(value)
        }
    }
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let data = payload
        .bytes()
        .map(|c| {
            ALPHABET
                .iter()
                .position(|v| *v == c)
                .map(|v| v as u8)
                .context("invalid legacy alphabet")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut bits = Bits { data, at: 0 };
    let first = match bits.read(2)? {
        0 => bits.read(8)? as u16,
        1 => bits.read(16)? as u16,
        _ => bail!("empty or invalid legacy link"),
    };
    let mut dictionary = vec![Vec::<u16>::new(); 3];
    dictionary.push(vec![first]);
    let mut previous = vec![first];
    let mut result = previous.clone();
    let mut width = 3;
    let mut enlarge = 4usize;
    let mut allocated = 1usize;
    loop {
        let mut code = bits.read(width)?;
        if code == 2 {
            break;
        }
        if code == 0 || code == 1 {
            let unit = bits.read(if code == 0 { 8 } else { 16 })? as u16;
            code = dictionary.len();
            dictionary.push(vec![unit]);
            allocated += 1;
            enlarge -= 1;
        }
        if enlarge == 0 {
            enlarge = 1 << width;
            width += 1;
        }
        let entry = if code < dictionary.len() {
            ensure!(
                !dictionary[code].is_empty(),
                "invalid legacy dictionary code"
            );
            dictionary[code].clone()
        } else if code == dictionary.len() {
            let mut entry = previous.clone();
            entry.push(previous[0]);
            entry
        } else {
            bail!("invalid legacy dictionary code");
        };
        ensure!(
            result.len() + entry.len() <= LIMIT / 2,
            "legacy output exceeds 4 MiB"
        );
        result.extend(&entry);
        previous.push(entry[0]);
        allocated += previous.len();
        ensure!(allocated <= LIMIT / 2, "legacy dictionary exceeds 4 MiB");
        dictionary.push(previous);
        previous = entry;
        enlarge -= 1;
        if enlarge == 0 {
            enlarge = 1 << width;
            width += 1;
        }
    }
    let text = String::from_utf16(&result).context("invalid UTF-16 legacy JSON")?;
    ensure!(text.len() <= LIMIT, "legacy JSON exceeds 4 MiB");
    Ok(text)
}
