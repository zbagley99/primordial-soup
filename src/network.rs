//! Versioned particle brains. Matrices are row-major, biases follow each matrix.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const CURRENT_REVISION: u32 = 3;
pub const WEIGHT_STRIDE: usize = 192;
/// Finite inputs are saturated before any network dot product.
pub const INPUT_LIMIT: f32 = 1_000_000.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Fourier,
    Mlp,
    Gru,
}
impl Mode {
    pub const ALL: [Self; 3] = [Self::Fourier, Self::Mlp, Self::Gru];
    pub fn label(self) -> &'static str {
        match self {
            Self::Fourier => "Fourier",
            Self::Mlp => "MLP",
            Self::Gru => "GRU",
        }
    }
    pub fn parameters(self) -> usize {
        match self {
            Self::Fourier => 80,
            Self::Mlp => 76,
            Self::Gru => 128,
        }
    }
    pub fn parameters_at(self, revision: u32) -> usize {
        if revision >= 3 {
            match self {
                Self::Mlp => 100,
                Self::Gru => 164,
                Self::Fourier => 80,
            }
        } else {
            self.parameters()
        }
    }
    pub fn id(self) -> u32 {
        match self {
            Self::Fourier => 0,
            Self::Mlp => 1,
            Self::Gru => 2,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bank {
    pub mode: Mode,
    pub revision: u32,
    /// None means independent random search, including for a valid all-zero parent.
    pub parent: Option<Vec<f32>>,
    pub seed: f64,
    pub scale: f64,
}
impl Bank {
    pub fn parameters(&self) -> usize {
        self.mode.parameters_at(self.revision)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Networks {
    pub active: Mode,
    pub banks: Vec<Bank>,
}
impl Networks {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=3).contains(&self.banks.len()),
            "requires 1..=3 network banks"
        );
        for (i, b) in self.banks.iter().enumerate() {
            ensure!(
                !self.banks[..i].iter().any(|a| a.mode == b.mode),
                "duplicate network bank"
            );
            ensure!(
                b.revision == 1
                    || (b.mode != Mode::Fourier && (2..=CURRENT_REVISION).contains(&b.revision)),
                "unsupported network revision {}",
                b.revision
            );
            ensure!(
                b.seed.is_finite() && (0.0..=1.0).contains(&b.seed),
                "invalid network seed"
            );
            ensure!(
                b.scale.is_finite() && b.scale >= 0.0 && (b.scale as f32).is_finite(),
                "invalid mutation scale"
            );
            if let Some(p) = &b.parent {
                ensure!(p.len() == b.parameters(), "wrong network tensor length");
                ensure!(
                    p.iter()
                        .all(|v| v.is_finite() && (b.mode == Mode::Fourier || v.abs() <= 8.0)),
                    "invalid network coefficient"
                );
            }
        }
        ensure!(
            self.banks.iter().any(|b| b.mode == self.active),
            "missing active network bank"
        );
        Ok(())
    }
}
// Revision 1 generator: SplitMix32, seed = XOR of f64 seed halves, mode,
// cohort and fixed domain constants. Uniform uses the high 24 bits / 2^24.
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e3779b9);
        let mut x = self.0;
        x = (x ^ (x >> 16)).wrapping_mul(0x21f0aaad);
        x = (x ^ (x >> 15)).wrapping_mul(0x735a2d97);
        x ^ (x >> 15)
    }
    fn signed(&mut self) -> f32 {
        ((self.next() >> 8) as f32 / 16777216.0) * 2.0 - 1.0
    }
}
/// Canonical tensor spans: (offset, count, matrix fan-in; 0 means bias).
pub fn tensors(mode: Mode) -> Vec<(usize, usize, usize)> {
    tensors_at(mode, 2)
}
pub fn tensors_at(mode: Mode, revision: u32) -> Vec<(usize, usize, usize)> {
    if revision >= 3 {
        return match mode {
            Mode::Mlp => vec![(0, 56, 7), (56, 8, 0), (64, 32, 8), (96, 4, 0)],
            Mode::Gru => vec![
                (0, 28, 7),
                (28, 16, 4),
                (44, 4, 0),
                (48, 28, 7),
                (76, 16, 4),
                (92, 4, 0),
                (96, 28, 7),
                (124, 16, 4),
                (140, 4, 0),
                (144, 16, 4),
                (160, 4, 0),
            ],
            Mode::Fourier => vec![],
        };
    }
    match mode {
        Mode::Mlp => vec![(0, 32, 4), (32, 8, 0), (40, 32, 8), (72, 4, 0)],
        Mode::Gru => vec![
            (0, 16, 4),
            (16, 16, 4),
            (32, 4, 0),
            (36, 16, 4),
            (52, 16, 4),
            (68, 4, 0),
            (72, 16, 4),
            (88, 16, 4),
            (104, 4, 0),
            (108, 16, 4),
            (124, 4, 0),
        ],
        Mode::Fourier => vec![],
    }
}
pub fn derive(bank: &Bank, cohort: u32) -> Vec<f32> {
    assert!(bank.mode != Mode::Fourier);
    let bits = bank.seed.to_bits();
    let mut seed_base = bits as u32
        ^ (bits >> 32) as u32
        ^ bank.mode.id().wrapping_mul(0x85ebca6b)
        ^ cohort.wrapping_mul(0xc2b2ae35)
        ^ 0x62726169;
    // Keep revision 1 byte-for-byte reproducible. Revision 2 separates random
    // initialization from mutation, so adopting a generated rule has no radial bias.
    if bank.revision >= 2 {
        seed_base ^= if bank.parent.is_some() {
            0x6d757461
        } else {
            0x696e6974
        };
    }
    let mut p = bank
        .parent
        .clone()
        .unwrap_or_else(|| vec![0.0; bank.parameters()]);
    for (tensor, (offset, count, fan)) in
        tensors_at(bank.mode, bank.revision).into_iter().enumerate()
    {
        for i in 0..count {
            let mut rng = Rng(seed_base
                ^ (tensor as u32).wrapping_mul(0x27d4eb2d)
                ^ (i as u32).wrapping_mul(0x165667b1));
            let v = if bank.parent.is_some() {
                let gain = if fan == 0 {
                    0.25
                } else {
                    1.0 / (fan as f32).sqrt()
                };
                p[offset + i] + rng.signed() * bank.scale as f32 * gain
            } else if fan > 0 {
                let rows = count / fan;
                rng.signed() * (6.0 / (fan + rows) as f32).sqrt()
            } else if bank.mode == Mode::Gru && tensor == 2 {
                1.0
            } else if bank.mode == Mode::Gru && tensor == 5 {
                0.0
            } else {
                rng.signed() * 0.25
            };
            p[offset + i] = v.clamp(-8.0, 8.0);
        }
    }
    p
}
fn dense(p: &[f32], offset: usize, bias: usize, x: &[f32], rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let mut v = p[bias + r];
            for (j, &input) in x.iter().enumerate() {
                v += p[offset + r * x.len() + j] * input;
            }
            v
        })
        .collect()
}
/// Independent host reference. GRU updates one stream, once.
pub fn evaluate(mode: Mode, p: &[f32], x: [f32; 4], h: &mut [f32; 4]) -> [f32; 4] {
    evaluate_revision(mode, 2, p, [x[0], x[1], x[2], x[3], 0.0, 0.0, 0.0], h)
}
pub fn encode_observations(
    flow: [f32; 4],
    mass: [f32; 2],
    speed: f32,
    reference_mass: f32,
    reference_speed: f32,
    gain: f32,
) -> [f32; 7] {
    let mut x = [0.0; 7];
    for j in 0..4 {
        let v = f64::from(flow[j]) * f64::from(gain)
            / (f64::from(mass[j / 2]) + f64::from(reference_mass))
            / f64::from(reference_speed);
        x[j] = (v / (1.0 + v.abs())) as f32;
    }
    for j in 0..2 {
        x[4 + j] = mass[j] / (mass[j] + reference_mass);
    }
    x[6] = speed / (speed + reference_speed);
    x
}
pub fn evaluate_revision(
    mode: Mode,
    revision: u32,
    p: &[f32],
    x: [f32; 7],
    h: &mut [f32; 4],
) -> [f32; 4] {
    let x = x.map(|v| v.clamp(-INPUT_LIMIT, INPUT_LIMIT));
    let x = &x[..if revision >= 3 { 7 } else { 4 }];
    let decode = |y: Vec<f32>| -> [f32; 4] {
        y.into_iter()
            .map(|v| if revision >= 3 { 2.0 * v.tanh() } else { v })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    };
    let t = tensors_at(mode, revision);
    if mode == Mode::Mlp {
        let hidden: Vec<_> = dense(p, 0, t[1].0, x, 8)
            .into_iter()
            .map(f32::tanh)
            .collect();
        return decode(dense(p, t[2].0, t[3].0, &hidden, 4));
    }
    assert_eq!(mode, Mode::Gru);
    let gate = |w, u, b| {
        let mut a = dense(p, w, b, x, 4);
        for r in 0..4 {
            for j in 0..4 {
                a[r] += p[u + r * 4 + j] * h[j];
            }
            a[r] = 1.0 / (1.0 + (-a[r]).exp());
        }
        a
    };
    let z = gate(t[0].0, t[1].0, t[2].0);
    let r = gate(t[3].0, t[4].0, t[5].0);
    let rh: Vec<_> = r.iter().zip(h.iter()).map(|(r, h)| r * h).collect();
    let mut n = dense(p, t[6].0, t[8].0, x, 4);
    for i in 0..4 {
        for j in 0..4 {
            n[i] += p[t[7].0 + i * 4 + j] * rh[j];
        }
        h[i] = z[i] * h[i] + (1.0 - z[i]) * n[i].tanh();
    }
    decode(dense(p, t[9].0, t[10].0, h, 4))
}
