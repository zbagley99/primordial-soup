//! Source-compatible state identity and graph exports. Full deltas preserve
//! exact rules without CPU recomputation of GPU-derived mutations.
use crate::config::Project;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Serialize, Deserialize)]
pub struct Event {
    pub session: String,
    pub at: f64,
    pub kind: String,
    pub label: String,
    pub hash: Option<String>,
    pub from: Option<String>,
    pub state: Option<Value>,
    pub field: Option<String>,
    pub value: Option<Value>,
    pub previous: Option<Value>,
}
pub fn hash_state(project: &Project) -> String {
    if project.extended() {
        // Distinct hash domain; legacy projects retain their exact identity.
        let mut hash = 0xcbf29ce484222325u64;
        for byte in b"primordial-network-v10"
            .iter()
            .copied()
            .chain(project.to_json().unwrap_or_default().bytes())
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        return format!("ps10-{hash:016x}");
    }
    let mut lanes = [
        0x811c9dc5u32,
        0x01000193,
        0x811c9dc5 ^ 0x5bf03635,
        0x9e3779b9,
    ];
    let configs: Vec<u32> = project.configs.iter().flat_map(|c| c.pack()).collect();
    let world = project.world.pack(0.0, 0);
    for chunk in [configs.as_slice(), world.as_slice()] {
        for word in chunk {
            for byte in word.to_le_bytes() {
                for (i, lane) in lanes.iter_mut().enumerate() {
                    *lane = (*lane ^ (u32::from(byte) + i as u32)).wrapping_mul(0x01000193);
                }
            }
        }
        for lane in &mut lanes {
            *lane = (*lane ^ (chunk.len() * 4) as u32).wrapping_mul(0x01000193);
        }
    }
    lanes.iter().map(|v| format!("{v:08x}")).collect()
}
fn camel_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| {
                    let mut parts = k.split('_');
                    let mut name = parts.next().unwrap_or_default().to_owned();
                    for part in parts {
                        let mut c = part.chars();
                        if let Some(first) = c.next() {
                            name.extend(first.to_uppercase());
                            name.extend(c);
                        }
                    }
                    (name, camel_keys(v))
                })
                .collect(),
        ),
        Value::Array(array) => Value::Array(array.into_iter().map(camel_keys).collect()),
        other => other,
    }
}
pub fn state(project: &Project) -> Value {
    if project.extended() {
        return serde_json::from_str(&project.to_json().expect("valid archive project")).unwrap();
    }
    camel_keys(json!({"configs":project.configs,"world":project.world}))
}
pub fn epoch_ms() -> f64 {
    web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}
fn iso_time(milliseconds: f64) -> String {
    let seconds = (milliseconds / 1000.0) as i64;
    let z = seconds / 86400 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    let clock = seconds % 86400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        clock / 3600,
        clock % 3600 / 60,
        clock % 60,
        milliseconds as i64 % 1000
    )
}
pub fn document(events: &[Event]) -> Result<String> {
    let mut known = BTreeSet::new();
    let (mut nodes, mut roots, mut visits) = (Vec::new(), Vec::new(), Vec::new());
    for event in events {
        let visit = if let (Some(hash), Some(state)) = (&event.hash, &event.state) {
            let repeat = known.contains(hash);
            let parent = event.from.as_ref().filter(|h| known.contains(*h));
            if !repeat {
                let delta = parent.map(
                    |_| json!({"kind":"full","configs":state["configs"],"world":state["world"]}),
                );
                nodes.push(json!({"hash":hash,"parent":parent,"delta":delta,"label":event.label,"visitedAt":event.at,"session":event.session}));
                if parent.is_none() {
                    roots.push(json!({"hash":hash,"configs":state["configs"],"world":state["world"],"reason":event.label,"createdAt":event.at}));
                }
                known.insert(hash.clone());
            }
            json!({"type":"project","kind":event.kind,"hash":hash,"from":event.from,"label":event.label,"repeat":repeat})
        } else if let Some(field) = &event.field {
            json!({"type":"preference","field":field,"value":event.value,"previous":event.previous,"label":event.label})
        } else {
            json!({"type":"command","command":event.kind,"label":event.label})
        };
        visits.push(
            json!({"seq":visits.len()+1,"session":event.session,"at":event.at,"visit":visit}),
        );
    }
    let rule = include_str!("../shaders/rule.wgsl");
    let common = include_str!("../shaders/common.wgsl");
    let mut hash = 0x811c9dc5u32;
    for c in rule.encode_utf16().chain([0]).chain(common.encode_utf16()) {
        hash = (hash ^ c as u32).wrapping_mul(0x01000193);
    }
    let extended = events
        .iter()
        .filter_map(|e| e.state.as_ref())
        .any(|s| s["version"] == 10);
    let mut doc = json!({"version":if extended {3}else{2},"exportedAt":iso_time(epoch_ms()),"nodeCount":nodes.len(),"rootCount":roots.len(),"visitCount":visits.len(),"derivation":{"hash":format!("{hash:08x}"),"rule.wgsl":rule,"common.wgsl":common},"roots":roots,"nodes":nodes,"visits":visits});
    if extended {
        doc["application"] = json!("primordial-soup");
        doc["projects"] = json!(
            events
                .iter()
                .filter_map(|e| e.state.as_ref())
                .filter(|s| s["version"] == 10)
                .collect::<Vec<_>>()
        );
        doc["network_derivation"] =
            json!({"revisions":[1,2,3],"implementation":include_str!("network.rs")});
    }
    Ok(serde_json::to_string_pretty(&doc)?)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Writer(std::sync::mpsc::SyncSender<String>);
#[cfg(not(target_arch = "wasm32"))]
impl Writer {
    pub fn start(notices: std::sync::Arc<std::sync::Mutex<Vec<String>>>) -> Result<Self> {
        Self::start_with_wakeup(notices, std::sync::Arc::new(|| {}))
    }
    pub fn start_with_wakeup(
        notices: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        wakeup: std::sync::Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(256);
        std::thread::Builder::new()
            .name("state archive writer".into())
            .spawn(move || {
                use std::io::Write;
                let notify = |message: String| {
                    notices.lock().unwrap().push(message);
                    // Release the notice lock before waking the UI to consume it.
                    wakeup();
                };
                let result = (|| -> Result<()> {
                    std::fs::create_dir_all("local-data")?;
                    let mut file = std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .open("local-data/state-archive.jsonl")?;
                    for text in rx {
                        if text == "\0clear" {
                            file.set_len(0)?;
                            notify("State archive cleared".into());
                        } else if text == "\0export" {
                            let document = document(&read()?)?;
                            std::fs::create_dir_all("exports")?;
                            let output = format!(
                                "exports/Fluoddity-State-Archive-{}.json",
                                epoch_ms() as u64
                            );
                            std::fs::write(&output, document)?;
                            notify(format!("State archive saved: {output}"));
                        } else {
                            file.write_all(text.as_bytes())?;
                            file.write_all(b"\n")?;
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = result {
                    notify(format!(
                        "Archive storage failed: {e}; export your current project"
                    ));
                }
            })?;
        Ok(Self(tx))
    }
    pub fn append(&self, event: &Event) -> Result<()> {
        self.0.try_send(serde_json::to_string(event)?)?;
        Ok(())
    }
    pub fn export(&self) -> Result<()> {
        self.0.try_send("\0export".into())?;
        Ok(())
    }
    pub fn clear(&self) -> Result<()> {
        self.0.try_send("\0clear".into())?;
        Ok(())
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn read() -> Result<Vec<Event>> {
    use std::io::BufRead;
    let file = std::fs::File::open("local-data/state-archive.jsonl")?;
    anyhow::ensure!(
        file.metadata()?.len() <= 128 * 1024 * 1024,
        "Archive exceeds 128 MiB export limit; preserve the JSONL file for offline analysis"
    );
    std::io::BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
