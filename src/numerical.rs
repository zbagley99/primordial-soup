//! Bounded native numerical checks against an independent analytical reference.
//!
//! Supported fixture: zero sensor gain, one supplied Fourier center, mutation,
//! hazards, jitter, gravity, fences and painted fields disabled. Its constant
//! axial force and strafe preserve heading: v' = drag*v + force*unit(v),
//! p' = p + v' + strafe*unit(v). We check wrap, bounce and kill separately.
//! Trails use a supplied impulse, zero-size (non-depositing) particles, the five-tap diffusion
//! stencil and persistence; the reference uses f64 without fp16 quantization.
//! Position/velocity tolerance is 3e-5 (4e-3 beyond the uniform batch boundary);
//! trail absolute tolerance is 2e-3, allowing fp16 storage rounding. Discrete
//! config/death and same-device batched/serial results must agree exactly.
//! This does not certify hash generation, stochastic deposits, every shader
//! branch, long chaotic trajectories, or cross-device equality. Run on each
//! actual adapter and retain the checker's adapter metadata with the report.
use crate::{
    config::Project,
    coords::Sizing,
    gpu::{Engine, MAX_STEPS},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{sync::mpsc, time::Duration};

const SIDE: u32 = 16;
const COUNT: u32 = 67; // Deliberately not a multiple of any tested workgroup.
const DRAG: f64 = 0.999;

fn project(boundary: u32) -> Project {
    let mut p = Project::default();
    p.configs.truncate(1);
    let c = &mut p.configs[0];
    c.rule.fill(0.0);
    c.rule[0] = 1.0; // Avoid the shader's all-zero-rule generation sentinel.
    c.rule[4..8].copy_from_slice(&[0.25, 0.0625, 0.125, 0.25]);
    c.sensor_gain = 0.0;
    c.mutation_scale = 0.0;
    c.global_force_mult = 0.25;
    c.axial_force = 1.0;
    c.lateral_force = 1.0;
    c.strafe_power = 0.125;
    c.drag = DRAG;
    c.hazard_rate = 0.0;
    c.gravity_force = 0.0;
    c.gravity_strafe = 0.0;
    c.gravity_trails = 0.0;
    c.sensor_angle_jitter = 0.0;
    c.sensor_distance_jitter = 0.0;
    c.trail_weight = 1.0; // Studio normalizes this to one; never divide sensors by zero.
    c.cohort_fences = false;
    c.cohorts = 1;
    p.world.boundary_conditions = boundary;
    p.world.trail_diffusion = 1.0; // K=4/(5^1-1)=1: five equal taps.
    p.world.trail_persistence = 0.5;
    p
}

fn seed() -> Vec<[u32; 8]> {
    (0..COUNT)
        .map(|i| {
            let p = if i % 3 == 0 {
                [0.999f32, -0.999]
            } else {
                [-0.5 + i as f32 / 128.0, 0.125]
            };
            [
                p[0].to_bits(),
                p[1].to_bits(),
                0.001f32.to_bits(),
                (-0.002f32).to_bits(),
                0, // Zero particle size makes the production deposit amount exactly zero.
                0,
                0,
                0,
            ]
        })
        .collect()
}

fn impulse() -> Vec<[f64; 2]> {
    let mut v = vec![[0.0; 2]; (SIDE * SIDE) as usize];
    // Distinct signs, channels and asymmetric edges catch flips and wrapping.
    v[0] = [0.5, -0.25];
    v[(3 * SIDE + 5) as usize] = [-0.125, 0.5];
    v
}

fn reference_particle(mut e: [u32; 8], boundary: u32, steps: u32) -> [f64; 8] {
    let mut s = e.map(|x| f32::from_bits(x) as f64);
    let phase: f64 = 0.25 * 3.14159;
    let color = 0.5 * phase.sin();
    let force = color * 0.25 / 400.0;
    let strafe = 0.25 * (phase * 1.3).sin() * 0.25 / 20.0 * 0.125;
    // DRAG is narrowed on upload; the reference follows that input precision.
    let drag = DRAG as f32 as f64;
    for _ in 0..steps {
        let len = s[2].hypot(s[3]);
        let dir = [s[2] / len, s[3] / len];
        for axis in 0..2 {
            s[axis + 2] = s[axis + 2] * drag + force * dir[axis];
            s[axis] += s[axis + 2] + strafe * dir[axis];
        }
        if boundary == 3 && (s[0].abs() > 1.0 || s[1].abs() > 1.0) {
            e = [0; 8];
            e[5] = (-1i32) as u32;
            return e.map(|x| f32::from_bits(x) as f64);
        }
        for axis in 0..2 {
            if boundary == 1 {
                s[axis] = (s[axis] + 1.0).rem_euclid(2.0) - 1.0;
            } else {
                if s[axis].abs() > 1.0 {
                    s[axis + 2] = -s[axis + 2];
                }
                s[axis] = s[axis].signum() * (1.0 - (1.0 - s[axis].abs() % 2.0).abs());
            }
        }
        s[6] = color;
        s[7] = 0.0;
    }
    s
}

fn reference_trails(boundary: u32, steps: u32) -> Vec<[f64; 2]> {
    let mut v = impulse();
    for _ in 0..steps {
        let get = |x: i32, y: i32| {
            let coord = |n: i32| {
                if boundary == 1 {
                    n.rem_euclid(SIDE as i32)
                } else {
                    n.clamp(0, SIDE as i32 - 1)
                }
            };
            v[(coord(y) * SIDE as i32 + coord(x)) as usize]
        };
        let mut next = v.clone();
        for y in 0..SIDE as i32 {
            for x in 0..SIDE as i32 {
                let taps = [
                    get(x, y),
                    get(x + 1, y),
                    get(x - 1, y),
                    get(x, y + 1),
                    get(x, y - 1),
                ];
                for c in 0..2 {
                    next[(y * SIDE as i32 + x) as usize][c] =
                        taps.iter().map(|v| v[c]).sum::<f64>() * 0.1;
                }
            }
        }
        v = next;
    }
    v
}

#[cfg(not(target_arch = "wasm32"))]
fn mapped(d: &wgpu::Device, b: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let result = (|| {
        d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })?;
        rx.recv_timeout(Duration::from_secs(1))??;
        Ok(b.slice(..).get_mapped_range()?.to_vec())
    })();
    b.unmap();
    result
}

fn half(v: u16) -> f64 {
    let sign = if v & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (v >> 10) & 31;
    let fraction = (v & 1023) as f64;
    if exponent == 31 {
        return if fraction == 0.0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        };
    }
    sign * if exponent == 0 {
        fraction * 2f64.powi(-24)
    } else {
        (1.0 + fraction / 1024.0) * 2f64.powi(exponent as i32 - 15)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    workgroup: u32,
    boundary: u32,
    steps: u32,
    serial: bool,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let p = project(boundary);
    let mut e = Engine::new(
        d,
        q,
        Sizing {
            entity_count: COUNT,
            canvas: [SIDE, SIDE],
            sqrt_world_size: 1.0,
        },
        &p,
        workgroup,
    )?;
    q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&seed()));
    // The seed values are exactly representable in binary16.
    let mut pixels = vec![0u16; (SIDE * SIDE * 4) as usize];
    pixels[..2].copy_from_slice(&[0x3800, 0xb400]);
    pixels[(3 * SIDE + 5) as usize * 4..(3 * SIDE + 5) as usize * 4 + 2]
        .copy_from_slice(&[0xb000, 0x3800]);
    q.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &e.canvas[e.current_canvas],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIDE * 8),
            rows_per_image: Some(SIDE),
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
    e.frame_count = 1; // Preserve supplied particles/trails; never exercise random reset.
    let mut remaining = steps;
    while remaining > 0 {
        let n = remaining.min(if serial { 1 } else { MAX_STEPS });
        e.prepare_frame(q, &p.world, n, None, [0.0, 0.0])?;
        let mut commands = d.create_command_encoder(&Default::default());
        for index in 0..n {
            e.encode_step(&mut commands, index);
        }
        q.submit([commands.finish()]);
        remaining -= n;
    }
    let entities = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("numerical particles"),
        size: COUNT as u64 * 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let trails = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("numerical trails"),
        size: SIDE as u64 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut c = d.create_command_encoder(&Default::default());
    c.copy_buffer_to_buffer(&e.entities, 0, &entities, 0, COUNT as u64 * 32);
    c.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &e.canvas[e.current_canvas],
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &trails,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(SIDE),
            },
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
    q.submit([c.finish()]);
    let entity_bytes = mapped(d, &entities)?;
    let raw = mapped(d, &trails)?;
    Ok((
        entity_bytes,
        raw.chunks_exact(256)
            .flat_map(|row| {
                row[..(SIDE * 8) as usize]
                    .chunks_exact(8)
                    .flat_map(|pixel| pixel[..4].iter().copied())
            })
            .collect(),
    ))
}

/// Run through production entity and canvas pipelines. Adapter identity belongs
/// in the enclosing checker report, alongside this returned numerical evidence.
#[cfg(not(target_arch = "wasm32"))]
pub fn validate(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<Value> {
    let mut groups = vec![64, 128, workgroup];
    groups.sort_unstable();
    groups.dedup();
    groups.retain(|g| {
        *g <= d.limits().max_compute_workgroup_size_x
            && *g <= d.limits().max_compute_invocations_per_workgroup
    });
    let mut cases = Vec::new();
    for group in groups {
        let mut specs = vec![(1, 7)];
        if group == workgroup {
            specs.extend([(0, 7), (3, 7), (1, MAX_STEPS + 1)]);
        }
        for (boundary, steps) in specs {
            let actual = run(d, q, group, boundary, steps, false)?;
            let serial = run(d, q, group, boundary, steps, true)?;
            ensure!(
                actual == serial,
                "Numerical serial/batched mismatch, workgroup {group}, boundary {boundary}, steps {steps}"
            );
            let tolerance = if steps > MAX_STEPS { 4e-3 } else { 3e-5 };
            let mut max_particle_error = 0f64;
            for (index, chunk) in actual.0.chunks_exact(32).enumerate() {
                let words: Vec<u32> = chunk
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let expected = reference_particle(seed()[index], boundary, steps);
                let dead = expected[5].is_nan(); // -1 config is an integer bitcast, not a float.
                ensure!(
                    words[5] == if dead { u32::MAX } else { 0 },
                    "Discrete config/death mismatch at {index}"
                );
                for lane in [0, 1, 2, 3, 4, 6, 7] {
                    let got = f32::from_bits(words[lane]) as f64;
                    ensure!(got.is_finite(), "Nonfinite particle {index} lane {lane}");
                    let error = (got - expected[lane]).abs();
                    max_particle_error = max_particle_error.max(error);
                    ensure!(
                        error <= tolerance,
                        "Particle reference mismatch entity {index} lane {lane}: {got} vs {}, error {error}, tolerance {tolerance}",
                        expected[lane]
                    );
                }
            }
            let expected = reference_trails(boundary, steps);
            let mut max_trail_error = 0f64;
            for (i, bytes) in actual.1.chunks_exact(2).enumerate() {
                let got = half(u16::from_le_bytes(bytes.try_into().unwrap()));
                ensure!(got.is_finite(), "Nonfinite trail component {i}");
                let error = (got - expected[i / 2][i % 2]).abs();
                max_trail_error = max_trail_error.max(error);
                ensure!(
                    error <= 2e-3,
                    "Trail reference mismatch {i}: {got}, error {error}"
                );
            }
            cases.push(json!({"workgroup":group,"boundary":boundary,"steps":steps,"particles":COUNT,"particle_absolute_tolerance":tolerance,"max_particle_error":max_particle_error,"trail_absolute_tolerance":2e-3,"max_trail_error":max_trail_error,"serial_batch_exact":true}));
        }
    }
    Ok(
        json!({"reference":"independent f64 constant-Fourier motion and five-tap trail recurrence","limits":"zero gain and particle size, no stochastic deposits/hazards/jitter/gravity/fences; not full shader certification or cross-device equality","cases":cases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn impulse_wraps_but_clamped_edges_keep_energy_local() {
        let wrap = reference_trails(1, 1);
        let clamp = reference_trails(0, 1);
        assert_eq!(wrap[(SIDE - 1) as usize], [0.05, -0.025]);
        assert_eq!(clamp[(SIDE - 1) as usize], [0.0, 0.0]);
        assert!((clamp[0][0] - 0.15).abs() < 1e-15);
    }
    #[test]
    fn boundary_reference_kills_crossers_and_preserves_survivors() {
        assert!(reference_particle(seed()[0], 3, 1)[5].is_nan());
        assert_eq!(reference_particle(seed()[1], 3, 1)[5], 0.0);
        let bounce = reference_particle(seed()[0], 0, 1);
        assert!(bounce[0] <= 1.0 && bounce[1] >= -1.0 && bounce[2] < 0.0 && bounce[3] > 0.0);
    }
    #[test]
    fn half_decoder_distinguishes_finite_values_from_invalid() {
        assert_eq!(half(0xb400), -0.25);
        assert_eq!(half(1), 2f64.powi(-24));
        assert!(!half(0x7c00).is_finite());
        assert!(half(0x7e00).is_nan());
    }
}
