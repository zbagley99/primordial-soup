//! Optional native observations of the actual update shader; never enabled by the app.
use super::*;
use serde_json::{Value, json};
use std::{sync::mpsc, time::Duration};

#[path = "paired_contracts.rs"]
mod paired;
pub use paired::check as paired_contracts;

#[path = "occupancy_contracts.rs"]
mod occupancy;
pub use occupancy::check as occupancy_contracts;

#[path = "gru_history.rs"]
mod history;
pub use history::{interleave_gru_history, planar_gru_history};

fn interleaved_history_values(
    values: &[f32],
    particles: usize,
    streams: usize,
) -> Result<Vec<f32>> {
    Ok(
        interleave_gru_history(bytemuck::cast_slice(values), particles, streams)?
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
    )
}

fn read_brain_history(d: &wgpu::Device, q: &wgpu::Queue, engine: &Engine) -> Result<Vec<f32>> {
    let values = read(d, q, &engine.brain_memory)?;
    if engine.mode == crate::network::Mode::Gru {
        interleaved_history_values(
            &values,
            engine.sizing.entity_count as usize,
            if engine.revision >= 3 { 3 } else { 2 },
        )
    } else {
        Ok(values)
    }
}

const WORDS: usize = 48;

impl Engine {
    pub fn audit_field_texture(&self) -> &wgpu::Texture {
        &self.field
    }
}

pub struct Trace {
    buffer: wgpu::Buffer,
}

fn insert(source: &mut String, anchor: &str, addition: &str) {
    assert_eq!(source.matches(anchor).count(), 1, "trace anchor: {anchor}");
    *source = source.replacen(anchor, &format!("{anchor}\n{addition}"), 1);
}

impl Trace {
    pub fn enable(d: &wgpu::Device, engine: &mut Engine, ablation: u32) -> Self {
        let buffer = buffer(
            d,
            "network observations",
            u64::from(engine.sizing.entity_count) * WORDS as u64 * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut source = shader("entityUpdate.wgsl").to_owned();
        source.push_str(
            "\n@group(3) @binding(0) var<storage, read_write> observations: array<vec4f>;\n",
        );
        insert(
            &mut source,
            "let e = entities[index];",
            "for(var j=0u;j<12u;j++){observations[index*12u+j]=vec4f(0.0);}\nobservations[index*12u]=e.pos_vel;",
        );
        insert(
            &mut source,
            "vel = MIN_VELOCITY * vec2f(cos(kick), sin(kick));",
            "observations[index*12u+9u].z=1.0;",
        );
        insert(
            &mut source,
            "let R = vec2f(dot(R_in, forward), dot(R_in, left));",
            "observations[index*12u+1u]=vec4f(L,R);observations[index*12u+5u]=vec4f(L,R);observations[index*12u+11u]=vec4f(sensor_mass,speed,occupancy_reference());",
        );
        insert(
            &mut source,
            "let mirrorterm = terms.mirror;",
            "observations[index*12u+2u]=baseterm; observations[index*12u+3u]=mirrorterm;",
        );
        insert(
            &mut source,
            "strafe *= 1.0 / sqrt_world_size * cfg_global_force_mult(config) / 20.0;",
            "observations[index*12u+4u]=vec4f(force,strafe*cfg_strafe_power(config));\nobservations[index*12u+9u].w=1.0;",
        );
        insert(
            &mut source,
            "let x = clamp(input, vec4f(-BRAIN_INPUT_LIMIT), vec4f(BRAIN_INPUT_LIMIT));",
            "if(stream==0u){observations[index*12u+5u]=x;observations[index*12u+10u]=vec4f(extra,0.0);}",
        );
        insert(
            &mut source,
            "hidden[i]=tanh(brain_bias(slot,select(32u,56u,modern),i)+brain_input_row(slot,0u,i,x,extra));",
            "if(stream==0u && abs(hidden[i])>0.99){observations[index*12u+9u].x+=0.125;}",
        );
        insert(
            &mut source,
            "let h=brain_memory[brain_state_index(index,stream)];",
            "if(stream==0u){observations[index*12u+6u]=h;}",
        );
        insert(
            &mut source,
            "brain_memory[brain_state_index(index,stream)]=next;",
            "if(stream==0u){observations[index*12u+7u]=next;observations[index*12u+8u]=z;}",
        );
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("observed entity update"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let trace_layout = layout(
            d,
            &[buf_layout(
                0,
                wgpu::BufferBindingType::Storage { read_only: false },
                false,
                0,
                wgpu::ShaderStages::COMPUTE,
            )],
        );
        let state = engine.entity_pipeline.get_bind_group_layout(0);
        let textures = engine.entity_pipeline.get_bind_group_layout(1);
        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("observed network update"),
            bind_group_layouts: &[
                Some(&state),
                Some(&textures),
                Some(&engine.interaction_layout),
                Some(&trace_layout),
            ],
            immediate_size: 0,
        });
        engine.entity_pipeline = compute(
            d,
            &m,
            "main",
            Some(&pl),
            &[
                ("CONFIG_PER_COHORT", 1.0),
                ("WORKGROUP_SIZE", engine.workgroup as f64),
                ("NETWORK_MODE", engine.mode.id() as f64),
                ("BRAIN_REVISION", engine.revision as f64),
                ("BRAIN_ABLATION", ablation as f64),
                ("WIDE_OCCUPANCY", f64::from(engine.splat_stride() == 4)),
            ],
        );
        engine.diagnostic_group = Some(group(d, &trace_layout, &[entry(0, &buffer, None)]));
        Self { buffer }
    }

    pub fn read(&self, d: &wgpu::Device, q: &wgpu::Queue) -> Result<Vec<f32>> {
        read(d, q, &self.buffer)
    }
}

pub fn read(d: &wgpu::Device, q: &wgpu::Queue, src: &wgpu::Buffer) -> Result<Vec<f32>> {
    let b = buffer(
        d,
        "network observation readback",
        src.size(),
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(src, 0, &b, 0, src.size());
    q.submit([e.finish()]);
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })?;
    rx.recv_timeout(Duration::from_secs(1))??;
    let data = b.slice(..).get_mapped_range()?;
    let out = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    drop(data);
    b.unmap();
    Ok(out)
}

fn distribution(mut values: Vec<f32>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    assert!(values.iter().all(|v| v.is_finite()));
    values.sort_by(f32::total_cmp);
    let quantile = |q: f32| values[((values.len() - 1) as f32 * q) as usize];
    json!({"mean": values.iter().map(|v| f64::from(*v)).sum::<f64>()/values.len() as f64, "p10":quantile(0.1),"p50":quantile(0.5),"p90":quantile(0.9),"p99":quantile(0.99),"max":values.last()})
}

pub fn summarize(trace: &[f32], entities: &[f32]) -> Value {
    let mut inputs = Vec::new();
    let mut speed = Vec::new();
    let mut thrust = Vec::new();
    let mut strafe = Vec::new();
    let mut memory_delta = Vec::new();
    let mut gates = Vec::new();
    let mut actual_rescues = 0;
    let mut occupancy_inputs = Vec::new();
    let mut speed_inputs = Vec::new();
    let mut rescues = 0;
    let mut reversals = 0;
    let mut low_input = 0;
    let mut saturation = 0.0;
    let mut count = 0;
    let mut bins = vec![0u32; 32 * 32];
    for (t, e) in trace.chunks_exact(WORDS).zip(entities.chunks_exact(8)) {
        if t[39] != 1.0 {
            continue;
        }
        count += 1;
        actual_rescues += usize::from(t[38] > 0.0);
        occupancy_inputs.extend_from_slice(&t[40..42]);
        speed_inputs.push(t[42]);
        let before = t[2].hypot(t[3]);
        let after = e[2].hypot(e[3]);
        rescues += usize::from(before < 5e-5);
        reversals += usize::from(t[2] * e[2] + t[3] * e[3] < 0.0);
        let axis = if before > 0.0 {
            [t[2] / before, t[3] / before]
        } else {
            [0.0; 2]
        };
        let max_input = t[20..24].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        inputs.push(max_input);
        low_input += usize::from(max_input < 0.1);
        speed.push(after);
        thrust.push(t[16] * axis[0] + t[17] * axis[1]);
        strafe.push(t[18].hypot(t[19]));
        memory_delta.push(
            t[24..28]
                .iter()
                .zip(&t[28..32])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt(),
        );
        gates.extend_from_slice(&t[32..36]);
        saturation += t[36];
        let x = (((e[0] + 1.0) * 0.5 * 32.0) as usize).min(31);
        let y = (((e[1] + 1.0) * 0.5 * 32.0) as usize).min(31);
        bins[y * 32 + x] += 1;
    }
    let n = count.max(1) as f64;
    let mean = n / bins.len() as f64;
    let cv = (bins.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / bins.len() as f64)
        .sqrt()
        / mean;
    json!({"evaluated":count,"low_speed_fraction":rescues as f64/n,"random_heading_rescue_fraction":actual_rescues as f64/n,"occupancy_input":distribution(occupancy_inputs),"speed_input":distribution(speed_inputs),"reversal_fraction":reversals as f64/n,"input_abs_max":distribution(inputs),"input_below_point_one_fraction":low_input as f64/n,"speed":distribution(speed),"signed_axial_acceleration":distribution(thrust),"strafe_displacement":distribution(strafe),"mlp_hidden_saturated_fraction":saturation as f64/n,"memory_delta":distribution(memory_delta),"update_gate":distribution(gates),"occupancy_cv":cv})
}

pub struct Evaluation {
    pub revision: u32,
    pub ablation: u32,
    pub world: f64,
    pub ticks: u32,
}
pub fn evaluate(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    workgroup: u32,
    directory: &str,
    options: Evaluation,
) -> Result<Value> {
    use crate::{
        config::Preferences,
        coords::Camera,
        network::Mode,
        render::{Overlays, Renderer},
    };
    ensure!(
        (1..=3).contains(&options.revision) && options.ablation <= 4,
        "unsupported evaluation revision/ablation"
    );
    ensure!(
        [256, 1024, 2048].contains(&options.ticks),
        "evaluation ticks must be 256, 1024 or 2048"
    );
    std::fs::create_dir_all(directory)?;
    let mut runs = Vec::new();
    for mode in Mode::ALL {
        if options.ablation > 0 && mode == Mode::Fourier {
            continue;
        }
        for seed in [0.125, 0.25, 0.5, 0.875] {
            if options.world >= 1.0 && seed != 0.25 {
                continue;
            }
            let mut project = Project::default();
            if mode != Mode::Fourier {
                project.config_mut().switch_mode(mode, seed);
            } else {
                project.config_mut().rule.fill(0.0);
                project.config_mut().mutation_seed = seed;
            }
            project.config_mut().cohorts = 4;
            if mode != Mode::Fourier {
                project
                    .config_mut()
                    .networks
                    .as_mut()
                    .unwrap()
                    .banks
                    .iter_mut()
                    .find(|b| b.mode == mode)
                    .unwrap()
                    .revision = options.revision;
            }
            let mut engine = Engine::new(
                d,
                q,
                crate::coords::sizing_for(options.world, 1.0)?,
                &project,
                workgroup,
            )?;
            let trace = Trace::enable(d, &mut engine, options.ablation);
            let mut samples = Vec::new();
            for target in [2, 17, 65, 257, 1025, 2049]
                .into_iter()
                .filter(|t| *t <= options.ticks + 1)
            {
                while engine.frame_count < target {
                    let n = (target - engine.frame_count).min(120);
                    engine.prepare_frame(q, &project.world, n, None, [0.01, 0.001])?;
                    let mut encoder = d.create_command_encoder(&Default::default());
                    for i in 0..n {
                        engine.encode_step(&mut encoder, i);
                    }
                    q.submit([encoder.finish()]);
                    d.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(30)),
                    })?;
                }
                let entities = read(d, q, &engine.entities)?;
                let hash = entities.iter().fold(0xcbf29ce484222325u64, |h, v| {
                    (h ^ u64::from(v.to_bits())).wrapping_mul(0x100000001b3)
                });
                samples.push(json!({"ticks":target-1,"particle_hash":format!("{hash:016x}"),"measurements":summarize(&trace.read(d,q)?,&entities)}));
            }
            let mut preferences = Preferences::default();
            preferences.cells.structure_threshold = 0.0;
            let mut renderer = Renderer::new(d, &engine, [512, 512])?;
            let mut encoder = d.create_command_encoder(&Default::default());
            renderer.begin_frame(q, &engine, &Camera::default(), &project, 1, -1.0);
            renderer.clear_accumulator(&mut encoder);
            renderer.sample(&mut encoder, &engine, &Camera::default());
            renderer.present(
                q,
                &mut encoder,
                &engine,
                &Camera::default(),
                &preferences,
                Overlays::default(),
            );
            renderer.present_clean(
                d,
                q,
                &mut encoder,
                &engine,
                &Camera::default(),
                &preferences,
            );
            let path = format!("{directory}/{}-{seed}.png", mode.label().to_lowercase());
            let rb = readback_rgba(
                d,
                &mut encoder,
                renderer.clean_texture()?,
                [0, 0],
                [512, 512],
            )?;
            q.submit([encoder.finish()]);
            let (tx, rx) = mpsc::channel();
            rb.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            d.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })?;
            rx.recv_timeout(Duration::from_secs(1))??;
            let bytes = rb.buffer.slice(..).get_mapped_range()?;
            std::fs::write(
                &path,
                crate::export::encode_png(&rb.rgba(&bytes)?, [512, 512])?,
            )?;
            drop(bytes);
            rb.buffer.unmap();
            runs.push(
                json!({"mode":mode,"seed":seed,"project":project,"samples":samples,"render":path}),
            );
        }
    }
    Ok(
        json!({"world_size":options.world,"revision":options.revision,"ablation":options.ablation,"ticks":options.ticks,"instrumented":true,"timing_claim":false,"runs":runs}),
    )
}

fn uniform_scratch_contract(
    engine: &mut Engine,
    q: &wgpu::Queue,
    world: &WorldSettings,
) -> Result<Value> {
    let saved_frame = engine.frame_count;
    engine.frame_count = u32::MAX - 3;
    let mut stable = None;
    for (index, steps) in [1, 31, 3, 0, 17].into_iter().enumerate() {
        engine.entity_uniform_words.fill(0xdead_beef);
        engine.canvas_uniform_words.fill(0xdead_beef);
        let shove = (index % 2 == 0).then_some(Shove {
            center: [-0.25, 0.75],
            strength: -1.25,
            size: 0.125,
        });
        let strengths = [index as f32 * 0.125, -0.25];
        engine.prepare_frame(q, world, steps, shove, strengths)?;
        let stride = engine.uniform_stride as usize / 4;
        let mut entity = vec![0u32; steps as usize * stride];
        let mut canvas = entity.clone();
        for step in 0..steps as usize {
            let e = &mut entity[step * stride..step * stride + 24];
            e[..8].copy_from_slice(&world.pack(engine.sizing.sqrt_world_size, engine.config_count));
            e[8..12].copy_from_slice(&[
                (engine.sizing.canvas[0] as f32).to_bits(),
                (engine.sizing.canvas[1] as f32).to_bits(),
                (engine.field_size[0] as f32).to_bits(),
                (engine.field_size[1] as f32).to_bits(),
            ]);
            if let Some(shove) = shove {
                e[12..16].copy_from_slice(&[
                    shove.center[0].to_bits(),
                    shove.center[1].to_bits(),
                    shove.strength.to_bits(),
                    shove.size.to_bits(),
                ]);
            }
            e[16] = engine.frame_count.wrapping_add(step as u32);
            e[17] = u32::from(engine.field_sampling_active());
            e[18..22].copy_from_slice(&[
                strengths[0].to_bits(),
                strengths[1].to_bits(),
                engine.camera_strengths[0].to_bits(),
                engine.camera_strengths[1].to_bits(),
            ]);
            let c = &mut canvas[step * stride..step * stride + 12];
            c[..8].copy_from_slice(&e[..8]);
            c[8] = e[16];
            c[9] = engine.splat_stride();
        }
        ensure!(
            engine.entity_uniform_words == entity && engine.canvas_uniform_words == canvas,
            "Reused uniform packing differs from zero-initialized packing at steps={steps}, shove={shove:?}"
        );
        let allocation = (
            engine.entity_uniform_words.as_ptr() as usize,
            engine.entity_uniform_words.capacity(),
            engine.canvas_uniform_words.as_ptr() as usize,
            engine.canvas_uniform_words.capacity(),
        );
        if index == 1 {
            stable = Some(allocation);
        } else if index > 1 {
            ensure!(
                stable == Some(allocation),
                "Uniform scratch reallocates after maximum test size"
            );
        }
    }
    engine.frame_count = saved_frame;
    Ok(
        json!({"steps":[1,31,3,0,17],"sentinel_prefill":true,"all_packed_words_exact":true,"shove_then_none":true,"frame_counter_wrap":true,"stable_after_grow":true,"entity_capacity_words":engine.entity_uniform_words.capacity(),"canvas_capacity_words":engine.canvas_uniform_words.capacity()}),
    )
}

pub fn contracts(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<Value> {
    use crate::network::{self, Mode};
    let cases: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/network-io-oracle.fixture.json"
    ))?;
    let mut results = Vec::new();
    let mut uniform_scratch = Vec::new();
    for mode in [Mode::Mlp, Mode::Gru] {
        let case = cases
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["mode"] == serde_json::to_value(mode).unwrap())
            .unwrap();
        let mut project = Project::default();
        project.config_mut().switch_mode(mode, 0.25);
        let oracle: Vec<f32> = serde_json::from_value(case["weights"].clone())?;
        let bank = project.config().bank().unwrap();
        let weights = [oracle, network::derive(&bank, 1), network::derive(&bank, 2)];
        let make = |size| {
            buffer(
                d,
                "IO contract",
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            )
        };
        let input = make(256);
        let output = make(128);
        let weight = make((3 * network::WEIGHT_STRIDE * 4) as u64);
        let memory = make(192);
        let mut packed = Vec::new();
        for w in &weights {
            packed.extend_from_slice(w);
            packed.resize(packed.len() + network::WEIGHT_STRIDE - w.len(), 0.0);
        }
        q.write_buffer(&weight, 0, bytemuck::cast_slice(&packed));
        let m = module(d, "networkProbe3.wgsl");
        let pipeline = compute(
            d,
            &m,
            "main",
            None,
            &[("NETWORK_MODE", mode.id() as f64), ("BRAIN_REVISION", 3.0)],
        );
        let bg = group(
            d,
            &pipeline.get_bind_group_layout(0),
            &[
                entry(0, &input, None),
                entry(1, &output, None),
                entry(5, &weight, None),
                entry(6, &memory, None),
            ],
        );
        let mut hs: [[f32; 4]; 8] =
            std::array::from_fn(|i| std::array::from_fn(|j| (i as f32 - j as f32) * 0.03));
        hs[0] = serde_json::from_value(case["initial_hidden"].clone())?;
        hs[6] = hs[1];
        hs[7] = hs[0];
        let mut memory_values = vec![0.0f32; 48];
        for i in 0..8 {
            memory_values[(i / 2 * 3 + i % 2) * 4..(i / 2 * 3 + i % 2 + 1) * 4]
                .copy_from_slice(&hs[i]);
        }
        let native_memory = if mode == Mode::Gru {
            planar_gru_history(bytemuck::cast_slice(&memory_values), 4, 3)?
        } else {
            bytemuck::cast_slice(&memory_values).to_vec()
        };
        q.write_buffer(&memory, 0, &native_memory);
        let mut max_error = 0.0f32;
        for tick in 0..40 {
            let mut xs: [[f32; 7]; 8] = std::array::from_fn(|i| {
                [
                    0.2 + i as f32 * 0.1,
                    -0.3,
                    0.4,
                    -0.2,
                    0.5,
                    0.7,
                    (tick % 7) as f32 * 0.1,
                ]
            });
            if tick < 4 {
                xs[0] = serde_json::from_value(case["steps"][tick]["input"].clone())?;
            }
            if tick >= 32 {
                xs.fill([f32::MAX, -f32::MAX, f32::MAX, -f32::MAX, 0.5, 0.5, 1.0]);
            }
            xs[6] = xs[1];
            xs[7] = xs[0];
            let mut values = Vec::new();
            for x in &xs {
                values.extend_from_slice(&x[..4]);
                values.extend_from_slice(&x[4..]);
                values.push(0.0);
            }
            q.write_buffer(&input, 0, bytemuck::cast_slice(&values));
            let mut encoder = d.create_command_encoder(&Default::default());
            {
                let mut p = encoder.begin_compute_pass(&Default::default());
                p.set_pipeline(&pipeline);
                p.set_bind_group(0, &bg, &[]);
                p.dispatch_workgroups(4, 1, 1);
            }
            q.submit([encoder.finish()]);
            let ys = read(d, q, &output)?;
            for i in 0..8 {
                let expected =
                    network::evaluate_revision(mode, 3, &weights[(i / 2) % 3], xs[i], &mut hs[i]);
                for j in 0..4 {
                    let error = (ys[i * 4 + j] - expected[j]).abs();
                    max_error = max_error.max(error);
                    ensure!(
                        error < 3e-5,
                        "revision 3 {} evaluator {tick}/{i}/{j}: {error}",
                        mode.label()
                    );
                }
                if i == 0 && tick < 4 {
                    let expected: [f64; 4] =
                        serde_json::from_value(case["steps"][tick]["output"].clone())?;
                    for j in 0..4 {
                        ensure!(
                            (f64::from(ys[j]) - expected[j]).abs() < 3e-5,
                            "independent revision 3 equation mismatch"
                        );
                    }
                }
            }
            ensure!(
                ys[..4] == ys[28..32] && ys[4..8] == ys[24..28],
                "revision 3 stream reflection mismatch"
            );
            if mode == Mode::Gru {
                let got = interleaved_history_values(&read(d, q, &memory)?, 4, 3)?;
                for i in 0..8 {
                    for j in 0..4 {
                        ensure!(
                            (got[(i / 2 * 3 + i % 2) * 4 + j] - hs[i][j]).abs() < 3e-5,
                            "revision 3 state mismatch"
                        );
                    }
                }
            }
        }
        // A zero controller can rest exactly while leaving positive occupancy.
        project
            .config_mut()
            .set_parent(Some(vec![0.0; mode.parameters_at(3)]));
        project.config_mut().mutation_scale = 0.0;
        project.config_mut().hazard_rate = 0.0;
        project.config_mut().cohort_fences = false;
        project.config_mut().cohorts = 1;
        project.config_mut().sensor_angle_jitter = 0.0;
        project.config_mut().sensor_distance_jitter = 0.0;
        let mut engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [32, 32],
                entity_count: 4096,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        uniform_scratch.push(uniform_scratch_contract(&mut engine, q, &project.world)?);
        let mut particles = vec![[0u32; 8]; 4096];
        for e in &mut particles {
            e[0] = 0.03125f32.to_bits();
            e[1] = 0.03125f32.to_bits();
            e[4] = 0.003f32.to_bits();
        }
        q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
        engine.frame_count = 1;
        for persistence in [0.001, 0.938, 0.9999] {
            for texture in &engine.canvas {
                q.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &vec![0u8; 32 * 32 * 8],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(32 * 8),
                        rows_per_image: Some(32),
                    },
                    wgpu::Extent3d {
                        width: 32,
                        height: 32,
                        depth_or_array_layers: 1,
                    },
                );
            }
            project.world.trail_persistence = persistence;
            engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
            let mut encoder = d.create_command_encoder(&Default::default());
            engine.encode_step(&mut encoder, 0);
            q.submit([encoder.finish()]);
            let got = read(d, q, &engine.entities)?;
            for e in got.chunks_exact(8) {
                ensure!(
                    e[0] == 0.03125 && e[1] == 0.03125 && e[2] == 0.0 && e[3] == 0.0,
                    "zero controller injected movement"
                );
            }
            let pixels = canvas_pixels(d, q, &engine)?;
            ensure!(
                pixels
                    .iter()
                    .all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2].is_finite() && p[2] >= 0.0),
                "occupancy or momentum invalid at rest"
            );
            ensure!(
                pixels.iter().any(|p| p[2] > 0.0),
                "resting particles disappeared from observations"
            );
        }
        project.world.trail_persistence = 0.938;
        for (i, e) in particles.iter_mut().enumerate() {
            e[2] = (if i % 2 == 0 { 0.001f32 } else { -0.001f32 }).to_bits();
        }
        q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
        let trace = Trace::enable(d, &mut engine, 0);
        engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
        let mut encoder = d.create_command_encoder(&Default::default());
        engine.encode_step(&mut encoder, 0);
        q.submit([encoder.finish()]);
        let observations = trace.read(d, q)?;
        for t in observations.chunks_exact(WORDS) {
            if t[39] != 1.0 {
                continue;
            }
            let x = network::encode_observations(
                t[4..8].try_into().unwrap(),
                [t[44], t[45]],
                t[46],
                t[47],
                0.001,
                project.config().sensor_gain as f32,
            );
            for j in 0..4 {
                ensure!(
                    (x[j] - t[20 + j]).abs() < 3e-5,
                    "observation encoding mismatch"
                );
            }
            for j in 0..3 {
                ensure!(
                    (x[j + 4] - t[40 + j]).abs() < 3e-5,
                    "scalar observation encoding mismatch"
                );
            }
        }
        let pixels = canvas_pixels(d, q, &engine)?;
        ensure!(
            pixels.iter().all(|p| p[0] == 0.0 && p[1] == 0.0),
            "opposing momentum did not cancel"
        );
        ensure!(
            pixels.iter().any(|p| p[2] > 0.0),
            "occupancy cancelled with momentum"
        );
        // Observe actual nonzero flow separately from cancelling occupancy.
        for e in &mut particles {
            e[2] = 0.001f32.to_bits();
        }
        q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
        for _ in 0..2 {
            engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
            let mut encoder = d.create_command_encoder(&Default::default());
            engine.encode_step(&mut encoder, 0);
            q.submit([encoder.finish()]);
        }
        let obs = trace.read(d, q)?;
        ensure!(
            obs.chunks_exact(WORDS).any(|t| t[20].abs() > 0.0),
            "nonzero momentum never reached network"
        );
        for t in obs.chunks_exact(WORDS) {
            if t[39] != 1.0 {
                continue;
            }
            let x = network::encode_observations(
                t[4..8].try_into().unwrap(),
                [t[44], t[45]],
                t[46],
                t[47],
                0.001,
                project.config().sensor_gain as f32,
            );
            for j in 0..4 {
                ensure!(
                    (x[j] - t[20 + j]).abs() < 3e-5,
                    "nonzero flow encoding mismatch"
                );
            }
        }
        let frozen = texture(d, "occupancy fixture snapshot", [32, 32], CANVAS_FORMAT);
        let mut encoder = d.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            engine.canvas[engine.current_canvas].as_image_copy(),
            frozen.as_image_copy(),
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
        );
        q.submit([encoder.finish()]);
        for e in &mut particles {
            e[2] = 0.0f32.to_bits();
        }
        q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
        let mut previous_density: Option<Vec<[f32; 2]>> = None;
        for weight in [0.02f32, 1.0, 5.0] {
            q.write_buffer(&engine.configs, 412, bytemuck::bytes_of(&weight));
            let mut encoder = d.create_command_encoder(&Default::default());
            encoder.copy_texture_to_texture(
                frozen.as_image_copy(),
                engine.canvas[engine.current_canvas].as_image_copy(),
                wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
            );
            engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
            engine.encode_step(&mut encoder, 0);
            q.submit([encoder.finish()]);
            let density: Vec<[f32; 2]> = trace
                .read(d, q)?
                .chunks_exact(WORDS)
                .map(|t| [t[40], t[41]])
                .collect();
            if let Some(previous) = previous_density.as_ref() {
                ensure!(
                    *previous == density,
                    "trail weight changed unweighted occupancy observation"
                );
            }
            previous_density = Some(density);
        }
        project.config_mut().set_parent(Some(weights[0].clone()));
        let mut live = Engine::new(
            d,
            q,
            Sizing {
                canvas: [32, 32],
                entity_count: 67,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        let tick = |e: &mut Engine, p: &Project, n: u32| -> Result<()> {
            e.prepare_frame(q, &p.world, n, None, [0.0; 2])?;
            let mut encoder = d.create_command_encoder(&Default::default());
            for i in 0..n {
                e.encode_step(&mut encoder, i);
            }
            q.submit([encoder.finish()]);
            Ok(())
        };
        tick(&mut live, &project, 7)?;
        let state = read_brain_history(d, q, &live)?;
        ensure!(
            state.iter().all(|v| v.is_finite()) && state.iter().any(|v| *v != 0.0),
            "revision 3 state did not advance"
        );
        let particles_before = read(d, q, &live.entities)?;
        let mut display = project.clone();
        display.config_mut().color_offset += 0.1;
        live.upload_project(d, q, &display)?;
        ensure!(
            read_brain_history(d, q, &live)? == state,
            "display upload changed revision 3 state"
        );
        project.config_mut().adopt_parent(weights[0].clone(), 0.75);
        live.upload_project(d, q, &project)?;
        ensure!(
            read_brain_history(d, q, &live)?.iter().all(|v| *v == 0.0),
            "adoption retained revision 3 state"
        );
        ensure!(
            read(d, q, &live.entities)? == particles_before,
            "weight edit moved particles before a tick"
        );
        tick(&mut live, &project, 1)?;
        ensure!(
            live.request_pick(q, &project.world, [0.0; 2], 4.0, -1.0),
            "revision 3 pick rejected"
        );
        let mut encoder = d.create_command_encoder(&Default::default());
        ensure!(live.encode_pick(&mut encoder), "pick not encoded");
        q.submit([encoder.finish()]);
        live.begin_pick_readback();
        d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })?;
        let pick = live
            .take_pick()?
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("revision 3 pick missed"))?;
        ensure!(
            pick.rule == weights[0].iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            "revision 3 pick lost coefficients"
        );
        live.reset();
        tick(&mut live, &project, 1)?;
        let reset = read_brain_history(d, q, &live)?;
        if mode == Mode::Gru {
            ensure!(
                reset
                    .chunks_exact(12)
                    .all(|v| v[..8].iter().all(|h| *h == 0.0) && v[8].hypot(v[9]) > 0.99),
                "reset did not separate hidden and heading state"
            );
        }
        project.config_mut().hazard_rate = 1.0;
        live.upload_project(d, q, &project)?;
        tick(&mut live, &project, 1)?;
        if mode == Mode::Gru {
            ensure!(
                read_brain_history(d, q, &live)?
                    .chunks_exact(12)
                    .all(|v| v[..8].iter().all(|h| *h == 0.0)),
                "hazard retained GRU history"
            );
        }
        project.config_mut().hazard_rate = 0.0;
        project.world.boundary_conditions = 3;
        live.upload_project(d, q, &project)?;
        let mut outside = read(d, q, &live.entities)?;
        for e in outside.chunks_exact_mut(8) {
            e[0] = 10.0;
            e[1] = 10.0;
        }
        q.write_buffer(&live.entities, 0, bytemuck::cast_slice(&outside));
        tick(&mut live, &project, 1)?;
        ensure!(
            read_brain_history(d, q, &live)?.iter().all(|v| *v == 0.0),
            "death retained revision 3 state"
        );
        // Same-mode revision switches must replace evaluator and state layout.
        let mut legacy = project.clone();
        legacy.config_mut().set_parent(None);
        legacy
            .config_mut()
            .networks
            .as_mut()
            .unwrap()
            .banks
            .iter_mut()
            .find(|b| b.mode == mode)
            .unwrap()
            .revision = 2;
        live.upload_project(d, q, &legacy)?;
        ensure!(
            live.frame_count == 0 && live.revision == 2,
            "legacy transition did not reset"
        );
        tick(&mut live, &legacy, 3)?;
        live.upload_project(d, q, &project)?;
        ensure!(
            live.frame_count == 0 && live.revision == 3,
            "current transition did not reset"
        );
        tick(&mut live, &project, 3)?;
        ensure!(
            read_brain_history(d, q, &live)?
                .iter()
                .all(|v| v.is_finite()),
            "revision transition corrupted state"
        );
        results.push(json!({"mode":mode,"max_output_error":max_error,"independent_oracle":true,"ticks":40,"mirrored_histories":true,"zero_motor_rest":true,"stationary_occupancy":true,"cancelling_flow":true,"nonzero_flow":true,"concentrated_particles":4096,"persistence":[0.001,0.938,0.9999],"encoded_observations":true,"lifecycle":true,"exact_pick":true,"same_mode_revision_transition":true}));
    }
    Ok(json!({"contracts":results,"uniform_scratch":uniform_scratch}))
}

pub fn canvas_pixels(d: &wgpu::Device, q: &wgpu::Queue, e: &Engine) -> Result<Vec<[f32; 4]>> {
    texture_pixels(d, q, &e.canvas[e.current_canvas])
}
fn texture_storage_bytes(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<u8>> {
    let [w, h] = [texture.width(), texture.height()];
    let pixel_bytes = texture
        .format()
        .block_copy_size(None)
        .context("Unsupported canvas readback format")?;
    let pitch = (w * pixel_bytes).next_multiple_of(256);
    let b = buffer(
        d,
        "occupancy readback",
        u64::from(pitch) * u64::from(h),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut encoder = d.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &b,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    q.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })?;
    rx.recv_timeout(Duration::from_secs(1))??;
    let raw = b.slice(..).get_mapped_range()?;
    let bytes = raw
        .chunks_exact(pitch as usize)
        .flat_map(|row| row[..w as usize * pixel_bytes as usize].iter().copied())
        .collect();
    drop(raw);
    b.unmap();
    Ok(bytes)
}
fn texture_pixels(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<[f32; 4]>> {
    let pixel_bytes = texture
        .format()
        .block_copy_size(None)
        .context("Unsupported half-float readback format")?;
    ensure!(
        matches!(
            texture.format(),
            wgpu::TextureFormat::Rg16Float | wgpu::TextureFormat::Rgba16Float
        ),
        "Expected half-float vector texture"
    );
    let raw = texture_storage_bytes(d, q, texture)?;
    let half = |v: u16| {
        let sign = if v & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = (v >> 10) & 31;
        let frac = f32::from(v & 1023);
        sign * if exp == 0 {
            frac * 2f32.powi(-24)
        } else if exp == 31 {
            f32::INFINITY
        } else {
            (1.0 + frac / 1024.0) * 2f32.powi(i32::from(exp) - 15)
        }
    };
    Ok(raw
        .chunks_exact(pixel_bytes as usize)
        .map(|px| {
            std::array::from_fn(|j| {
                if j * 2 < px.len() {
                    half(u16::from_le_bytes([px[j * 2], px[j * 2 + 1]]))
                } else {
                    0.0
                }
            })
        })
        .collect())
}
