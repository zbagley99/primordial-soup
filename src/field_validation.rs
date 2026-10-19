//! Native same-device contracts for skipping a provably pristine painted field.
//! The control always samples its zero field; particle and trail equations are
//! production code on both sides. This checks the optimization's equivalence,
//! not an independent physical model or cross-device reproducibility.
#![cfg(not(target_arch = "wasm32"))]
use crate::{
    config::Project,
    coords::Sizing,
    gpu::{Engine, MAX_STEPS, Paint},
    network::Mode,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{sync::mpsc, time::Duration};

const SIDE: u32 = 32;
const COUNT: u32 = 67; // Dispatch tail, live/dead slots and asymmetric coordinates.
const STRENGTHS: [f32; 2] = [0.01, 0.001];

fn wait(d: &wgpu::Device) -> Result<()> {
    d.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })?;
    Ok(())
}
fn mapped(d: &wgpu::Device, b: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let result = (|| {
        wait(d)?;
        rx.recv_timeout(Duration::from_secs(1))??;
        Ok(b.slice(..).get_mapped_range()?.to_vec())
    })();
    b.unmap();
    result
}
fn read_buffer(d: &wgpu::Device, q: &wgpu::Queue, source: &wgpu::Buffer) -> Result<Vec<u8>> {
    let target = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field contract buffer readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = d.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(source, 0, &target, 0, source.size());
    q.submit([commands.finish()]);
    mapped(d, &target)
}
fn read_texture(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    source: &wgpu::Texture,
    size: [u32; 2],
    pixel_bytes: u32,
) -> Result<Vec<u8>> {
    let pitch = (size[0] * pixel_bytes).next_multiple_of(256);
    let target = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field contract texture readback"),
        size: u64::from(pitch) * u64::from(size[1]),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = d.create_command_encoder(&Default::default());
    commands.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &target,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(size[1]),
            },
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
    q.submit([commands.finish()]);
    Ok(mapped(d, &target)?
        .chunks_exact(pitch as usize)
        .flat_map(|row| row[..(size[0] * pixel_bytes) as usize].iter().copied())
        .collect())
}
fn finite_half(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(2)
        .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7c00 != 0x7c00)
}
#[derive(PartialEq, Eq)]
struct Snapshot {
    entities: Vec<u8>,
    splat: Vec<u8>,
    canvases: [Vec<u8>; 2],
    memory: Vec<u8>,
    frame: u32,
    canvas: usize,
}
fn snapshot(d: &wgpu::Device, q: &wgpu::Queue, engine: &Engine) -> Result<Snapshot> {
    let result = Snapshot {
        entities: read_buffer(d, q, &engine.entities)?,
        splat: read_buffer(d, q, engine.splat_buffer())?,
        canvases: [
            read_texture(d, q, &engine.canvas[0], engine.sizing.canvas, 8)?,
            read_texture(d, q, &engine.canvas[1], engine.sizing.canvas, 8)?,
        ],
        memory: read_buffer(d, q, &engine.brain_memory)?,
        frame: engine.frame_count,
        canvas: engine.current_canvas,
    };
    for particle in result.entities.chunks_exact(32) {
        for lane in [0, 1, 2, 3, 4, 6, 7] {
            ensure!(
                f32::from_le_bytes(particle[lane * 4..lane * 4 + 4].try_into().unwrap())
                    .is_finite(),
                "Nonfinite field contract particle lane {lane}"
            );
        }
    }
    ensure!(
        result.canvases.iter().all(|v| finite_half(v)),
        "Nonfinite field contract trails"
    );
    ensure!(
        result
            .memory
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "Nonfinite field contract brain memory"
    );
    ensure!(
        result.splat.iter().all(|v| *v == 0),
        "Field contract splat not drained"
    );
    Ok(result)
}
fn equal(d: &wgpu::Device, q: &wgpu::Queue, a: &Engine, b: &Engine, case: &str) -> Result<()> {
    let a = snapshot(d, q, a)?;
    let b = snapshot(d, q, b)?;
    ensure!(a.entities == b.entities, "{case}: particle bytes differ");
    ensure!(a.splat == b.splat, "{case}: splat bytes differ");
    ensure!(a.canvases == b.canvases, "{case}: canvas bytes differ");
    ensure!(a.memory == b.memory, "{case}: brain memory bytes differ");
    ensure!(
        (a.frame, a.canvas) == (b.frame, b.canvas),
        "{case}: tick/canvas metadata differ"
    );
    Ok(())
}
fn project(boundary: u32, hazard: f64) -> Project {
    let mut p = Project::default();
    p.configs.truncate(1);
    let c = p.config_mut();
    c.rule.fill(0.0);
    c.rule[0..8].copy_from_slice(&[1.0, 0.25, -0.5, 0.125, 0.25, 0.0625, 0.125, 0.25]);
    c.sensor_gain = 1.0;
    c.mutation_scale = 0.0;
    c.global_force_mult = 0.25;
    c.axial_force = 1.0;
    c.lateral_force = 1.0;
    c.strafe_power = 0.125;
    c.drag = 0.999;
    c.hazard_rate = hazard;
    c.gravity_force = 0.0;
    c.gravity_strafe = 0.0;
    c.gravity_trails = 0.0;
    c.sensor_angle_jitter = 0.0;
    c.sensor_distance_jitter = 0.0;
    c.trail_weight = 1.0;
    c.cohort_fences = false;
    c.cohorts = 4;
    p.world.boundary_conditions = boundary;
    p.world.trail_diffusion = 0.5;
    p.world.trail_persistence = 0.5;
    p
}
fn seeded(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    p: &Project,
    workgroup: u32,
    force: bool,
) -> Result<Engine> {
    let mut e = Engine::new(
        d,
        q,
        Sizing {
            entity_count: COUNT,
            canvas: [SIDE, SIDE],
            sqrt_world_size: 1.0,
        },
        p,
        workgroup,
    )?;
    ensure!(
        !e.field_sampling_active(),
        "Fresh field must be provably pristine"
    );
    if force {
        let _ = e.field_texture(q);
    }
    let particles: Vec<[u32; 8]> = (0..COUNT)
        .map(|i| {
            let x = if i % 7 == 0 {
                0.9995
            } else {
                -0.3 + (i % 13) as f32 * 0.05
            };
            [
                x.to_bits(),
                (-0.25 + (i % 11) as f32 * 0.05).to_bits(),
                0.001f32.to_bits(),
                (-0.002f32).to_bits(),
                0.001f32.to_bits(),
                if i % 17 == 0 { u32::MAX } else { 0 },
                0,
                ((i % 4) as f32).to_bits(),
            ]
        })
        .collect();
    q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
    e.frame_count = 1; // Preserve injected particles, including deliberately dead slots.
    Ok(e)
}
fn encode_prepared(d: &wgpu::Device, q: &wgpu::Queue, e: &mut Engine, steps: u32) {
    let mut commands = d.create_command_encoder(&Default::default());
    for index in 0..steps {
        e.encode_step(&mut commands, index);
    }
    q.submit([commands.finish()]);
}
fn advance(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    e: &mut Engine,
    p: &Project,
    mut steps: u32,
) -> Result<()> {
    while steps > 0 {
        let n = steps.min(MAX_STEPS);
        e.prepare_frame(q, &p.world, n, None, STRENGTHS)?;
        encode_prepared(d, q, e, n);
        steps -= n;
    }
    wait(d)
}
fn upload(d: &wgpu::Device, q: &wgpu::Queue, e: &Engine, layers: [u16; 4]) -> Result<()> {
    let pixels: Vec<u16> = (0..e.field_size[0] * e.field_size[1])
        .flat_map(|_| layers)
        .collect();
    q.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: e.field_texture(q),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(e.field_size[0] * 8),
            rows_per_image: Some(e.field_size[1]),
        },
        wgpu::Extent3d {
            width: e.field_size[0],
            height: e.field_size[1],
            depth_or_array_layers: 1,
        },
    );
    wait(d)
}
fn paint(d: &wgpu::Device, q: &wgpu::Queue, e: &Engine, layer: u32) -> Result<()> {
    let mut commands = d.create_command_encoder(&Default::default());
    e.paint(
        q,
        &mut commands,
        Paint {
            uv: [0.5, 0.5],
            previous_uv: [0.25, 0.4],
            size: 1.0,
            power: 8.0,
            erase: false,
            mode: 0,
            layer,
            angle: 0.0,
            line_gain: 1.0,
        },
    )?;
    q.submit([commands.finish()]);
    wait(d)
}

pub fn validate(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<Value> {
    let mut cases = Vec::new();
    for (boundary, hazard) in [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (1, 0.02)] {
        let p = project(boundary, hazard);
        let mut optimized = seeded(d, q, &p, workgroup, false)?;
        let mut control = seeded(d, q, &p, workgroup, true)?;
        let mut completed = 0;
        for target in [1, 7, 120, 1025] {
            advance(d, q, &mut optimized, &p, target - completed)?;
            advance(d, q, &mut control, &p, target - completed)?;
            equal(
                d,
                q,
                &optimized,
                &control,
                "pristine versus forced zero-field sampling",
            )?;
            ensure!(
                !optimized.field_sampling_active() && control.field_sampling_active(),
                "Zero-field branch coverage lost"
            );
            cases.push(json!({"case":"pristine","boundary":boundary,"hazard_rate":hazard,"ticks":target,"byte_exact":true}));
            completed = target;
        }
    }
    // First mutation may happen after prepare_frame has already uploaded every
    // substep's flags. Force the control active before preparing, to expose an
    // omitted or incomplete uniform-ring backpatch on the optimized engine.
    let p = project(1, 0.0);
    for (name, layer, raw, after_prepare) in [
        ("paint walls", 0, false, false),
        ("paint trails", 1, false, false),
        ("prepare then paint walls", 0, false, true),
        ("prepare then paint trails", 1, false, true),
        ("raw field upload", 0, true, false),
        ("prepare then raw field upload", 0, true, true),
    ] {
        let mut optimized = seeded(d, q, &p, workgroup, false)?;
        let mut control = seeded(d, q, &p, workgroup, true)?;
        let mut unmodified = seeded(d, q, &p, workgroup, false)?;
        let steps = 7;
        if after_prepare {
            optimized.prepare_frame(q, &p.world, steps, None, STRENGTHS)?;
            control.prepare_frame(q, &p.world, steps, None, STRENGTHS)?;
        }
        for e in [&optimized, &control] {
            if raw {
                upload(d, q, e, [0x3400, 0xb000, 0x3000, 0x3400])?;
            } else {
                paint(d, q, e, layer)?;
            }
            ensure!(
                e.field_sampling_active(),
                "{name}: first mutation did not activate field"
            );
            let field = read_texture(d, q, e.field_texture(q), e.field_size, 8)?;
            ensure!(
                finite_half(&field) && field.iter().any(|v| *v != 0),
                "{name}: mutation was empty/nonfinite"
            );
        }
        if after_prepare {
            encode_prepared(d, q, &mut optimized, steps);
            encode_prepared(d, q, &mut control, steps);
        } else {
            advance(d, q, &mut optimized, &p, steps)?;
            advance(d, q, &mut control, &p, steps)?;
        }
        advance(d, q, &mut unmodified, &p, steps)?;
        equal(d, q, &optimized, &control, name)?;
        ensure!(
            snapshot(d, q, &optimized)?.entities != snapshot(d, q, &unmodified)?.entities,
            "{name}: field had no particle effect"
        );
        for e in [&optimized, &control] {
            let mut commands = d.create_command_encoder(&Default::default());
            e.clear_field(&mut commands);
            q.submit([commands.finish()]);
            ensure!(
                e.field_sampling_active(),
                "Clearing must not restore pristine assumption"
            );
            let field = read_texture(d, q, e.field_texture(q), e.field_size, 8)?;
            ensure!(
                field.iter().all(|v| *v == 0),
                "Full field clear left content"
            );
        }
        advance(d, q, &mut optimized, &p, 7)?;
        advance(d, q, &mut control, &p, 7)?;
        equal(d, q, &optimized, &control, "cleared field remains active")?;
        optimized.reset();
        control.reset();
        ensure!(
            optimized.field_sampling_active() && control.field_sampling_active(),
            "Reset disabled an exposed field"
        );
        advance(d, q, &mut optimized, &p, 7)?;
        advance(d, q, &mut control, &p, 7)?;
        equal(d, q, &optimized, &control, "reset after field mutation")?;
        cases.push(json!({"case":name,"prepared_substeps":steps,"nonzero_particle_effect":true,"clear_and_reset_monotonic":true,"byte_exact":true}));
    }
    // Already-recorded commands still reference uploaded uniform slots even
    // after a later prepare_frame(0). First access must backpatch those slots.
    let mut recorded_optimized = seeded(d, q, &p, workgroup, false)?;
    let mut recorded_control = seeded(d, q, &p, workgroup, true)?;
    let mut recorded_zero = seeded(d, q, &p, workgroup, false)?;
    let mut recorded = Vec::new();
    for e in [
        &mut recorded_optimized,
        &mut recorded_control,
        &mut recorded_zero,
    ] {
        e.prepare_frame(q, &p.world, 7, None, STRENGTHS)?;
        let mut commands = d.create_command_encoder(&Default::default());
        for index in 0..7 {
            e.encode_step(&mut commands, index);
        }
        e.prepare_frame(q, &p.world, 0, None, STRENGTHS)?;
        recorded.push(commands.finish());
    }
    ensure!(
        !recorded_optimized.field_sampling_active(),
        "Recorded-command fixture lost pristine coverage"
    );
    upload(d, q, &recorded_optimized, [0x3400, 0xb000, 0x3000, 0x3400])?;
    upload(d, q, &recorded_control, [0x3400, 0xb000, 0x3000, 0x3400])?;
    q.submit(recorded);
    wait(d)?;
    equal(
        d,
        q,
        &recorded_optimized,
        &recorded_control,
        "encoded steps then prepare zero then raw upload",
    )?;
    ensure!(
        snapshot(d, q, &recorded_optimized)?.entities != snapshot(d, q, &recorded_zero)?.entities,
        "Recorded-command raw upload had no particle effect"
    );
    cases.push(json!({"case":"prepare7_encode7_prepare0_raw_upload_submit","byte_exact":true,"nonzero_particle_effect":true}));
    // The camera is a different texture and must keep working while painted
    // field sampling remains disabled.
    let camera = crate::gpu::texture(
        d,
        "field validation positive camera",
        [SIDE, SIDE],
        wgpu::TextureFormat::Rg16Float,
    );
    let camera_pixels = vec![[0x3400u16, 0xb000u16]; (SIDE * SIDE) as usize];
    q.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &camera,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&camera_pixels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIDE * 4),
            rows_per_image: Some(SIDE),
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
    let view = camera.create_view(&Default::default());
    let mut camera_optimized = seeded(d, q, &p, workgroup, false)?;
    let mut camera_control = seeded(d, q, &p, workgroup, true)?;
    let mut camera_off = seeded(d, q, &p, workgroup, false)?;
    for e in [&mut camera_optimized, &mut camera_control] {
        e.set_camera_field(d, &view);
        e.set_camera_strengths([0.01, 0.0]);
    }
    advance(d, q, &mut camera_optimized, &p, 7)?;
    advance(d, q, &mut camera_control, &p, 7)?;
    advance(d, q, &mut camera_off, &p, 7)?;
    equal(
        d,
        q,
        &camera_optimized,
        &camera_control,
        "camera independent of painted-field flag",
    )?;
    ensure!(
        !camera_optimized.field_sampling_active(),
        "Camera-only input activated painted field"
    );
    ensure!(
        snapshot(d, q, &camera_optimized)?.entities != snapshot(d, q, &camera_off)?.entities,
        "Camera-only input had no particle effect"
    );
    cases.push(json!({"case":"camera_only","painted_field_inactive":true,"nonzero_particle_effect":true,"byte_exact":true}));
    // Replacing a brain and reset cannot accidentally invalidate either field
    // activity state. Check both zero-field execution routes with real networks.
    let mut brain_project = p.clone();
    let mut optimized = seeded(d, q, &brain_project, workgroup, false)?;
    let mut control = seeded(d, q, &brain_project, workgroup, true)?;
    for mode in [Mode::Mlp, Mode::Gru, Mode::Fourier] {
        brain_project.config_mut().switch_mode(mode, 0.375);
        optimized.upload_project(d, q, &brain_project)?;
        control.upload_project(d, q, &brain_project)?;
        ensure!(
            !optimized.field_sampling_active() && control.field_sampling_active(),
            "Project replacement changed field activity"
        );
        optimized.reset();
        control.reset();
        advance(d, q, &mut optimized, &brain_project, 7)?;
        advance(d, q, &mut control, &brain_project, 7)?;
        equal(
            d,
            q,
            &optimized,
            &control,
            "brain replacement/reset pristine field",
        )?;
        cases.push(json!({"case":"brain_replacement_reset","mode":mode,"byte_exact":true}));
    }
    Ok(
        json!({"particles":COUNT,"canvas":[SIDE,SIDE],"workgroup":workgroup,"equivalence":"byte-exact same-device production pipelines","finite_state":true,"compared":["entities","splat","both_canvases","brain_memory","frame_count","current_canvas"],"cases":cases}),
    )
}
