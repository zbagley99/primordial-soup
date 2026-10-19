//! Configuration and persistence port of Fluoddity (MIT, Jesse Gelders, 2026).
//! Original license accompanies assets/presets/LICENSE.txt.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub networks: Option<crate::network::Networks>,
    pub mutation_seed: f64,
    pub sensor_gain: f64,
    pub sensor_angle: f64,
    pub sensor_distance: f64,
    pub mutation_scale: f64,
    pub global_force_mult: f64,
    pub drag: f64,
    pub strafe_power: f64,
    pub axial_force: f64,
    pub lateral_force: f64,
    pub hazard_rate: f64,
    pub gravity_force: f64,
    pub gravity_strafe: f64,
    pub gravity_trails: f64,
    pub color_sensitivity: f64,
    pub color_offset: f64,
    pub sensor_angle_jitter: f64,
    pub sensor_distance_jitter: f64,
    pub trail_weight: f64,
    pub cohorts: u32,
    pub initial_conditions: u32,
    pub cohort_fences: bool,
    pub color_by_cohort: bool,
    pub radial_gravity: bool,
    pub rule: Vec<f64>,
}
impl SimulationConfig {
    pub fn network_revision(&self) -> u32 {
        self.bank().map_or(1, |b| b.revision)
    }
    pub fn network_parameters(&self) -> usize {
        self.mode().parameters_at(self.network_revision())
    }
    pub fn randomize_network(&mut self) {
        self.set_parent(None);
        let mode = self.mode();
        if mode != crate::network::Mode::Fourier {
            self.networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|b| b.mode == mode)
                .unwrap()
                .revision = crate::network::CURRENT_REVISION;
        }
    }
    pub fn mode(&self) -> crate::network::Mode {
        self.networks
            .as_ref()
            .map_or(crate::network::Mode::Fourier, |n| n.active)
    }
    pub fn generated(&self) -> bool {
        if self.mode() == crate::network::Mode::Fourier {
            [0, 1, 2, 3, 44, 45, 46, 47]
                .into_iter()
                .all(|i| self.rule.get(i) == Some(&0.0))
        } else {
            self.bank().unwrap().parent.is_none()
        }
    }
    // Legacy scalar controls remain the active seed/scale adapters. Banks are
    // normalized at every serialization/switch boundary, never read stale.
    pub fn bank(&self) -> Option<crate::network::Bank> {
        let mut b = self
            .networks
            .as_ref()?
            .banks
            .iter()
            .find(|b| b.mode == self.mode())?
            .clone();
        b.seed = self.mutation_seed;
        b.scale = self.mutation_scale;
        if b.mode == crate::network::Mode::Fourier {
            b.parent = if self.generated() {
                None
            } else {
                Some(self.rule.iter().map(|v| *v as f32).collect())
            };
        }
        Some(b)
    }
    pub fn normalized_networks(&self) -> Option<crate::network::Networks> {
        let mut n = self.networks.clone()?;
        let active = self.bank().unwrap();
        *n.banks.iter_mut().find(|b| b.mode == n.active).unwrap() = active;
        Some(n)
    }
    pub fn switch_mode(&mut self, mode: crate::network::Mode, seed: f64) {
        use crate::network::{Bank, Mode, Networks};
        if mode == self.mode() {
            return;
        }
        let mut n = self.normalized_networks().unwrap_or_else(|| Networks {
            active: Mode::Fourier,
            banks: vec![Bank {
                mode: Mode::Fourier,
                revision: 1,
                parent: if self.generated() {
                    None
                } else {
                    Some(self.rule.iter().map(|v| *v as f32).collect())
                },
                seed: self.mutation_seed,
                scale: self.mutation_scale,
            }],
        });
        if !n.banks.iter().any(|b| b.mode == mode) {
            n.banks.push(Bank {
                mode,
                revision: if mode == Mode::Fourier {
                    1
                } else {
                    crate::network::CURRENT_REVISION
                },
                parent: None,
                seed,
                scale: 0.1,
            });
        }
        let b = n.banks.iter().find(|b| b.mode == mode).unwrap();
        self.mutation_seed = b.seed;
        self.mutation_scale = b.scale;
        if mode == Mode::Fourier {
            self.rule = b
                .parent
                .as_ref()
                .map(|p| p.iter().map(|v| *v as f64).collect())
                .unwrap_or_else(|| vec![0.0; 80]);
        }
        n.active = mode;
        self.networks = Some(n);
    }
    pub fn set_parent(&mut self, parent: Option<Vec<f32>>) {
        if self.mode() == crate::network::Mode::Fourier {
            self.rule = parent
                .map(|p| p.into_iter().map(f64::from).collect())
                .unwrap_or_else(|| vec![0.0; 80]);
        } else {
            let mode = self.mode();
            self.networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|b| b.mode == mode)
                .unwrap()
                .parent = parent;
        }
    }
    /// Adoption starts a fresh generation. Old revision-1 saves reproduce until
    /// explicitly edited; adopting their effective coefficients upgrades safely.
    pub fn adopt_parent(&mut self, parent: Vec<f32>, seed: f64) {
        let revision = self.network_revision().max(2);
        self.set_parent(Some(parent));
        if self.mode() != crate::network::Mode::Fourier {
            let bits = self.mutation_seed.to_bits();
            let new_bits = seed.to_bits();
            let same_stream =
                (bits as u32 ^ (bits >> 32) as u32) == (new_bits as u32 ^ (new_bits >> 32) as u32);
            self.mutation_seed = if same_stream {
                let bits = self.mutation_seed.abs().to_bits();
                f64::from_bits(if self.mutation_seed == 1.0 {
                    bits - 1
                } else {
                    bits + 1
                })
            } else {
                seed
            };
            let mode = self.mode();
            self.networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|b| b.mode == mode)
                .unwrap()
                .revision = revision;
        }
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(n) = &self.networks {
            n.validate()?;
            if n.active != crate::network::Mode::Fourier {
                ensure!(
                    self.sensor_gain <= f64::from(crate::network::INPUT_LIMIT),
                    "network sensor_gain must be at most 1000000"
                );
                ensure!(
                    self.cohorts <= 4096,
                    "network modes support at most 4096 cohorts"
                );
            }
        }
        ensure!(
            self.rule.len() == 80,
            "rule must contain exactly 80 coefficients"
        );
        ensure!(
            (1..=i32::MAX as u32).contains(&self.cohorts),
            "cohorts must be in 1..=2147483647 (the shader signed integer range)"
        );
        ensure!(
            self.initial_conditions <= 3,
            "unknown initial_conditions mode"
        );
        finite(self.mutation_seed, "mutation_seed")?;
        finite(self.sensor_gain, "sensor_gain")?;
        finite(self.sensor_angle, "sensor_angle")?;
        finite(self.sensor_distance, "sensor_distance")?;
        finite(self.mutation_scale, "mutation_scale")?;
        finite(self.global_force_mult, "global_force_mult")?;
        finite(self.drag, "drag")?;
        finite(self.strafe_power, "strafe_power")?;
        finite(self.axial_force, "axial_force")?;
        finite(self.lateral_force, "lateral_force")?;
        finite(self.hazard_rate, "hazard_rate")?;
        finite(self.gravity_force, "gravity_force")?;
        finite(self.gravity_strafe, "gravity_strafe")?;
        finite(self.gravity_trails, "gravity_trails")?;
        finite(self.color_sensitivity, "color_sensitivity")?;
        finite(self.color_offset, "color_offset")?;
        finite(self.sensor_angle_jitter, "sensor_angle_jitter")?;
        finite(self.sensor_distance_jitter, "sensor_distance_jitter")?;
        finite(self.trail_weight, "trail_weight")?;
        for &v in &self.rule {
            finite(v, "rule coefficient")?;
        }
        ensure!(
            (0.0..=1.0).contains(&self.mutation_seed),
            "mutation_seed must be in 0..=1"
        );
        ensure!(
            self.sensor_distance >= 0.0 && self.sensor_gain >= 0.0 && self.mutation_scale >= 0.0,
            "negative sensor or mutation scale"
        );
        ensure!(
            (-1.0..=1.0).contains(&self.drag) && (0.0..=1.0).contains(&self.hazard_rate),
            "drag must be in -1..=1 and hazard_rate in 0..=1"
        );
        ensure!(
            (0.0..=1.0).contains(&self.sensor_angle_jitter)
                && (0.0..=1.0).contains(&self.sensor_distance_jitter),
            "sensor jitter must be in 0..=1"
        );
        Ok(())
    }
    /// Float lanes are explicitly narrowed to f32; integer lanes are bitcast.
    pub fn pack(&self) -> [u32; 104] {
        assert_eq!(self.rule.len(), 80, "invalid rule length");
        let mut out = [0; 104];
        for (dst, &v) in out[..80].iter_mut().zip(&self.rule) {
            *dst = (v as f32).to_bits();
        }
        out[80] = (self.sensor_gain as f32).to_bits();
        out[81] = (self.sensor_angle as f32).to_bits();
        out[82] = (self.sensor_distance as f32).to_bits();
        out[83] = (self.mutation_scale as f32).to_bits();
        out[84] = (self.global_force_mult as f32).to_bits();
        out[85] = (self.drag as f32).to_bits();
        out[86] = (self.strafe_power as f32).to_bits();
        out[87] = (self.axial_force as f32).to_bits();
        out[88] = (self.lateral_force as f32).to_bits();
        out[89] = (self.hazard_rate as f32).to_bits();
        out[90] = self.cohorts;
        out[91] = (self.mutation_seed as f32).to_bits();
        out[92] = (self.gravity_force as f32).to_bits();
        out[93] = (self.gravity_strafe as f32).to_bits();
        out[94] = self.initial_conditions;
        out[95] = u32::from(self.cohort_fences);
        out[96] = (self.color_sensitivity as f32).to_bits();
        out[97] = u32::from(self.color_by_cohort);
        out[98] = (self.sensor_angle_jitter as f32).to_bits();
        out[99] = (self.sensor_distance_jitter as f32).to_bits();
        out[100] = u32::from(self.radial_gravity);
        out[101] = (self.gravity_trails as f32).to_bits();
        out[102] = (self.color_offset as f32).to_bits();
        out[103] = (self.trail_weight as f32).to_bits();
        out
    }
}
fn finite(v: f64, field: &str) -> Result<()> {
    ensure!(
        v.is_finite() && (v as f32).is_finite(),
        "{field} must be finite and representable as f32"
    );
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSettings {
    pub trail_persistence: f64,
    pub trail_diffusion: f64,
    pub boundary_conditions: u32,
}
impl Default for WorldSettings {
    fn default() -> Self {
        Self {
            trail_persistence: 0.94,
            trail_diffusion: 1.0,
            boundary_conditions: 1,
        }
    }
}
impl WorldSettings {
    pub fn validate(&self) -> Result<()> {
        finite(self.trail_persistence, "trail_persistence")?;
        finite(self.trail_diffusion, "trail_diffusion")?;
        ensure!(
            (0.0..=1.0).contains(&self.trail_persistence),
            "trail_persistence must be in 0..=1"
        );
        ensure!(
            (0.0..=1.0).contains(&self.trail_diffusion),
            "trail_diffusion must be in 0..=1"
        );
        ensure!(
            self.boundary_conditions <= 3,
            "unknown boundary_conditions mode"
        );
        Ok(())
    }
    pub fn pack(&self, sqrt_world_size: f32, config_count: u32) -> [u32; 8] {
        [
            (self.trail_persistence as f32).to_bits(),
            (self.trail_diffusion as f32).to_bits(),
            sqrt_world_size.to_bits(),
            config_count,
            self.boundary_conditions,
            0,
            0,
            0,
        ]
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub configs: Vec<SimulationConfig>,
    pub world: WorldSettings,
    pub name: String,
    pub selected: usize,
    pub notes: String,
}
impl Project {
    pub fn extended(&self) -> bool {
        self.configs.iter().any(|c| c.networks.is_some())
    }
    pub fn config(&self) -> &SimulationConfig {
        &self.configs[0]
    }
    pub fn config_mut(&mut self) -> &mut SimulationConfig {
        &mut self.configs[0]
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.configs.is_empty() && self.configs.len() <= 256,
            "project requires 1..=256 configurations"
        );
        ensure!(
            self.selected < self.configs.len(),
            "selected index outside configs"
        );
        self.world.validate()?;
        for c in &self.configs {
            c.validate()?;
        }
        Ok(())
    }
    pub fn from_json(text: &str) -> Result<Self> {
        ensure!(text.len() <= 4 * 1024 * 1024, "project JSON exceeds 4 MiB");
        let doc: Value = serde_json::from_str(text).context("invalid project JSON")?;
        let version = doc
            .get("version")
            .and_then(Value::as_u64)
            .context("missing format version")?;
        ensure!(
            version == 8 || version == 9 || version == 10,
            "unsupported format version {version}"
        );
        let w = object(&doc, "world", false)?;
        let world = WorldSettings {
            trail_persistence: number(&w, "trail_persistence", None)?,
            trail_diffusion: number(&w, "trail_diffusion", None)?,
            boundary_conditions: integer(&w, "boundary_conditions", Some(1))?,
        };
        let raw = doc
            .get("configs")
            .and_then(Value::as_array)
            .context("missing configs array")?;
        ensure!(
            (1..=256).contains(&raw.len()),
            "project requires 1..=256 configurations"
        );
        let mut configs = Vec::with_capacity(raw.len());
        for (i, r) in raw.iter().enumerate() {
            ensure!(
                version == 10 || r.get("networks").is_none(),
                "network banks require project version 10"
            );
            configs.push(read_config(r).with_context(|| format!("configs[{i}]"))?);
        }
        let mut p = Self {
            configs,
            world,
            name: doc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("Untitled")
                .into(),
            selected: 0,
            notes: doc
                .get("notes")
                .and_then(Value::as_str)
                .unwrap_or("")
                .into(),
        };
        p.validate()?;
        if version == 8 {
            for c in &mut p.configs {
                migrate_v8(c);
            }
        }
        p.validate()?;
        Ok(p)
    }
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        let mut d = json!({"version":9,"world":self.world,"configs":self.configs.iter().map(write_config).collect::<Vec<_>>()});
        if self.extended() {
            d["version"] = json!(10);
            d["name"] = json!(self.name);
        }
        if !self.notes.is_empty() {
            d["notes"] = json!(self.notes);
        }
        Ok(serde_json::to_string_pretty(&d)?)
    }
}
impl Default for Project {
    fn default() -> Self {
        let mut p = Project::from_json(include_str!("../assets/presets/Core/Starcrossed.json"))
            .expect("shipped default preset must be valid");
        p.name = "Starcrossed".into();
        p
    }
}
pub struct Preset {
    pub category: &'static str,
    pub name: &'static str,
    pub project: Project,
}
pub fn preset_catalog() -> Vec<Preset> {
    let embedded: &[(&str, &str, &str)] = include!("../assets/presets/mod.rs");
    embedded
        .iter()
        .map(|(category, name, text)| {
            let mut p = Project::from_json(text).expect("shipped preset must be valid");
            p.name = (*name).into();
            Preset {
                category,
                name,
                project: p,
            }
        })
        .collect()
}
pub fn presets() -> Vec<(String, Project)> {
    preset_catalog()
        .into_iter()
        .map(|p| (p.name.into(), p.project))
        .collect()
}
fn object(raw: &Value, key: &str, optional: bool) -> Result<Value> {
    match raw.get(key) {
        Some(v) if v.is_object() => Ok(v.clone()),
        None if optional => Ok(json!({})),
        _ => bail!("missing or malformed {key} block"),
    }
}
fn number(raw: &Value, key: &str, fallback: Option<f64>) -> Result<f64> {
    let v = match raw.get(key) {
        None => fallback.context(format!("missing {key}"))?,
        Some(v) => v.as_f64().context(format!("{key} is not a number"))?,
    };
    finite(v, key)?;
    Ok(v)
}
fn integer(raw: &Value, key: &str, fallback: Option<u32>) -> Result<u32> {
    let v = number(raw, key, fallback.map(f64::from))?;
    ensure!(
        v >= 0.0 && v <= u32::MAX as f64 && v.fract() == 0.0,
        "{key} is not an unsigned integer"
    );
    Ok(v as u32)
}
fn boolean(raw: &Value, key: &str, default: bool) -> Result<bool> {
    match raw.get(key) {
        None => Ok(default),
        Some(v) => v.as_bool().context(format!("{key} is not a boolean")),
    }
}
fn read_config(raw: &Value) -> Result<SimulationConfig> {
    let sensor = object(raw, "sensor", false)?;
    let force = object(raw, "force", false)?;
    let misc = object(raw, "misc", false)?;
    let force2 = object(raw, "force2", true)?;
    let misc2 = object(
        raw,
        if raw.get("misc2").is_some() {
            "misc2"
        } else {
            "appearance"
        },
        true,
    )?;
    let misc3 = object(raw, "misc3", true)?;
    let fences = match force2.get("cohort_fences") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(v) => {
            let n = v
                .as_f64()
                .context("cohort_fences must be boolean or number")?;
            finite(n, "cohort_fences")?;
            n > 0.0
        }
    };
    let rule: Vec<f64> = serde_json::from_value(raw.get("rule").context("missing rule")?.clone())
        .context("invalid rule")?;
    let config = SimulationConfig {
        networks: raw
            .get("networks")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .context("invalid network banks")?,
        mutation_seed: number(
            &misc,
            if misc.get("mutation_seed").is_some() {
                "mutation_seed"
            } else {
                "rule_seed"
            },
            Some(0.0),
        )?,
        sensor_gain: number(&sensor, "gain", None)?,
        sensor_angle: number(&sensor, "angle", None)?,
        sensor_distance: number(&sensor, "distance", None)?,
        mutation_scale: number(&sensor, "mutation_scale", None)?,
        global_force_mult: number(&force, "global_mult", None)?,
        drag: number(&force, "drag", None)?,
        strafe_power: number(&force, "strafe", None)?,
        axial_force: number(&force, "axial", None)?,
        lateral_force: number(&misc, "lateral", None)?,
        hazard_rate: number(&misc, "hazard_rate", None)?,
        gravity_force: number(&force2, "gravity_force", Some(0.0))?,
        gravity_strafe: number(&force2, "gravity_strafe", Some(0.0))?,
        gravity_trails: number(&misc3, "gravity_trails", Some(0.0))?,
        color_sensitivity: number(&misc2, "color_sensitivity", Some(0.5))?,
        color_offset: number(&misc3, "color_offset", Some(0.0))?,
        sensor_angle_jitter: number(&misc2, "sensor_angle_jitter", Some(0.0))?,
        sensor_distance_jitter: number(&misc2, "sensor_distance_jitter", Some(0.0))?,
        trail_weight: number(&misc3, "trail_weight", Some(1.0))?,
        cohorts: integer(&misc, "cohorts", None)?,
        initial_conditions: integer(&force2, "initial_conditions", Some(2))?,
        cohort_fences: fences,
        color_by_cohort: boolean(&misc2, "color_by_cohort", false)?,
        radial_gravity: boolean(&misc3, "radial_gravity", false)?,
        rule,
    };
    config.validate()?;
    if let Some(n) = &config.networks {
        let active = n
            .banks
            .iter()
            .find(|b| b.mode == n.active)
            .context("missing active bank")?;
        ensure!(
            active.seed == config.mutation_seed && active.scale == config.mutation_scale,
            "active bank seed/scale disagree with active scalar controls"
        );
        if n.active == crate::network::Mode::Fourier {
            let parent = if config.generated() {
                None
            } else {
                Some(config.rule.iter().map(|v| *v as f32).collect::<Vec<_>>())
            };
            ensure!(
                active.parent == parent,
                "active Fourier bank disagrees with legacy rule/source"
            );
        }
    }
    Ok(config)
}
fn write_config(c: &SimulationConfig) -> Value {
    let mut value = json!({
        "rule":c.rule,
        "sensor":{"gain":c.sensor_gain,"angle":c.sensor_angle,"distance":c.sensor_distance,"mutation_scale":c.mutation_scale},
        "force":{"global_mult":c.global_force_mult,"drag":c.drag,"strafe":c.strafe_power,"axial":c.axial_force},
        "misc":{"lateral":c.lateral_force,"hazard_rate":c.hazard_rate,"cohorts":c.cohorts,"mutation_seed":c.mutation_seed},
        "force2":{"gravity_force":c.gravity_force,"gravity_strafe":c.gravity_strafe,"initial_conditions":c.initial_conditions,"cohort_fences":c.cohort_fences},
        "misc2":{"color_sensitivity":c.color_sensitivity,"color_by_cohort":c.color_by_cohort,"sensor_angle_jitter":c.sensor_angle_jitter,"sensor_distance_jitter":c.sensor_distance_jitter},
        "misc3":{"radial_gravity":c.radial_gravity,"gravity_trails":c.gravity_trails,"color_offset":c.color_offset,"trail_weight":c.trail_weight},
    });
    if let Some(n) = c.normalized_networks() {
        value["networks"] = serde_json::to_value(n).unwrap();
    }
    value
}
fn pcg_hash(seed: u32) -> u32 {
    let state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}
fn hash(x: f32, y: f32) -> f32 {
    pcg_hash(x.to_bits() ^ pcg_hash(y.to_bits())) as f32 / 4294967296.0
}
fn hash4(x: f32, y: f32) -> [f32; 4] {
    [
        hash(x, y),
        hash(-x + 5.0, -y + 5.0),
        hash(y - 100.0, x - 100.0),
        hash(-y + 25.0, -x + 25.0),
    ]
}
/// Exact legacy f32 arithmetic, used only to bake v8 single-cohort saves.
pub fn mutate_rule_v1(rule: &[f64], amount: f64, cohort: f32) -> Vec<f64> {
    assert_eq!(rule.len(), 80);
    let mut r: Vec<f32> = rule.iter().map(|v| *v as f32).collect();
    let a = amount as f32;
    let sx = (r[32] + r[61]) + r[10];
    let sy = (r[33] + r[60]) + r[11];
    let seed = hash(sx, sy) + cohort;
    let half_a = a * 0.5;
    for i in 0..10 {
        let h = hash4(-0.5 + (-(i as f32) + seed), -0.5 + i as f32);
        for k in 0..4 {
            let m = a * (-1.0 + 2.0 * h[k]);
            r[i * 8 + 4 + k] += m;
        }
        let factor = 1.0 + half_a * (hash(seed, i as f32) - 0.5);
        for k in 0..4 {
            r[i * 8 + k] *= factor;
        }
    }
    r.into_iter().map(f64::from).collect()
}
fn migrate_v8(c: &mut SimulationConfig) {
    let sentinel = [0, 1, 2, 3, 44, 45, 46, 47]
        .iter()
        .all(|&i| c.rule[i] == 0.0);
    if c.mutation_scale != 0.0 && !sentinel && c.cohorts == 1 {
        c.rule = mutate_rule_v1(&c.rule, c.mutation_scale, c.mutation_seed as f32);
        c.mutation_scale = 0.0;
    }
}

/// Live-view working resolution relative to the canvas's physical pixels.
/// Video export keeps its own size; screenshots use the working resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderScale {
    /// Full resolution for Classic; Cells is capped at a 1080-pixel short side.
    #[default]
    Auto,
    Full,
    ThreeQuarters,
    Half,
}
impl RenderScale {
    pub const ALL: [Self; 4] = [Self::Auto, Self::Full, Self::ThreeQuarters, Self::Half];
    /// Short side above which Auto scales Cells down; Cells cost grows with pixels.
    pub const CELLS_AUTO_SHORT_SIDE: u32 = 1080;
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Full => "1×",
            Self::ThreeQuarters => "0.75×",
            Self::Half => "0.5×",
        }
    }
    /// Working size for a physical canvas. Aspect is kept; each side is at least one pixel.
    pub fn size(self, physical: [u32; 2], cells: bool) -> [u32; 2] {
        let factor = match self {
            Self::Auto if cells => (Self::CELLS_AUTO_SHORT_SIDE as f64
                / physical[0].min(physical[1]).max(1) as f64)
                .min(1.0),
            Self::Auto | Self::Full => 1.0,
            Self::ThreeQuarters => 0.75,
            Self::Half => 0.5,
        };
        physical.map(|v| ((v as f64 * factor).round() as u32).max(1))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub cells: crate::cells::Settings,
    pub render_scale: RenderScale,
    pub brightness: f32,
    pub physics_steps: u32,
    pub tonemap_softness: f32,
    pub motion_blur_samples: u32,
    pub show_fps_counter: bool,
    pub physics_slider_open: bool,
    pub show_physics_slider: bool,
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub bloom_radius: f32,
    pub draw_size: f32,
    pub draw_power: f32,
    pub brush_mode: u32,
    pub draw_angle: f32,
    pub walls_strength: f32,
    pub trails_strength: f32,
    pub field_opacity: f32,
    pub field_always_show: bool,
    pub trails_always_show: bool,
    pub show_reticle: bool,
    pub reset_on_behavior_change: bool,
    pub one_click_selection: bool,
    pub strong_logging: bool,
    pub world_size: f64,
    pub canvas_aspect: f64,
    pub advanced_project: bool,
    pub advanced_preferences: bool,
    pub advanced_drawing: bool,
    pub mobile_mode: u32,
    pub calibration_version: u32,
}
impl Default for Preferences {
    fn default() -> Self {
        Self {
            cells: Default::default(),
            render_scale: RenderScale::Auto,
            brightness: 2.0,
            physics_steps: 5,
            tonemap_softness: 2.5,
            motion_blur_samples: 1,
            show_fps_counter: true,
            physics_slider_open: true,
            show_physics_slider: true,
            bloom_enabled: true,
            bloom_threshold: 0.2,
            bloom_intensity: 0.1,
            bloom_radius: 1.0,
            draw_size: 0.01,
            draw_power: 2.5,
            brush_mode: 0,
            draw_angle: 0.0,
            walls_strength: 1.0,
            trails_strength: 1.0,
            field_opacity: 0.10,
            field_always_show: false,
            trails_always_show: false,
            show_reticle: true,
            reset_on_behavior_change: true,
            one_click_selection: false,
            strong_logging: false,
            world_size: 0.50,
            canvas_aspect: 1.0,
            advanced_project: false,
            advanced_preferences: false,
            advanced_drawing: false,
            mobile_mode: 0,
            calibration_version: 0,
        }
    }
}

#[cfg(test)]
mod render_scale_tests {
    use super::*;
    #[test]
    fn auto_caps_only_cells_and_keeps_aspect() {
        let s = RenderScale::Auto;
        assert_eq!(s.size([3840, 2160], false), [3840, 2160]);
        assert_eq!(s.size([3840, 2160], true), [1920, 1080]);
        assert_eq!(s.size([2560, 1440], true), [1920, 1080]);
        assert_eq!(s.size([1920, 1080], true), [1920, 1080]);
        assert_eq!(s.size([1280, 800], true), [1280, 800]);
        assert_eq!(RenderScale::Half.size([1921, 1081], false), [961, 541]);
        assert_eq!(
            RenderScale::ThreeQuarters.size([2000, 1000], true),
            [1500, 750]
        );
        assert_eq!(RenderScale::Half.size([1, 1], true), [1, 1]);
    }
    #[test]
    fn saves_without_render_scale_still_load() {
        let mut value = serde_json::to_value(Preferences::default()).unwrap();
        value.as_object_mut().unwrap().remove("render_scale");
        let prefs: Preferences = serde_json::from_value(value).unwrap();
        assert_eq!(prefs.render_scale, RenderScale::Auto);
        let half = serde_json::json!({"render_scale": "half"});
        let prefs: Preferences = serde_json::from_value(half).unwrap();
        assert_eq!(prefs.render_scale, RenderScale::Half);
    }
}
