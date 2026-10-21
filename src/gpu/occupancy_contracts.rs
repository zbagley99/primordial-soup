use super::*;

pub fn check(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    const MAX: u32 = 7_680_000;
    const PIXELS: usize = 11;
    let source = probe_source()?;
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("preclear occupancy contracts"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let mut reports = Vec::new();
    for mode in 0..=2 {
        for wide in [false, true] {
            let stride = if wide { 4 } else { 3 };
            let pipelines = ["reference", "candidate"].map(|entry| {
                let candidate = entry == "candidate";
                compute(
                    d,
                    &module,
                    entry,
                    None,
                    &[
                        ("NETWORK_MODE", mode as f64),
                        ("WIDE_OCCUPANCY", f64::from(wide && candidate)),
                        ("STRIDE", if candidate { stride as f64 } else { 3.0 }),
                    ],
                )
            });
            for (name, count, initial, additions) in [
                ("zero", 67, 0u64, vec![0]),
                ("unsaturated", 67, 31, vec![1, 7, 127]),
                ("threshold", 67, u64::from(MAX - 9), vec![0, 1, 4, 31]),
                ("signed-boundary", 67, 0x7fff_fffe, vec![0, 1, MAX]),
                ("single-carry", 67, 0xffff_fffe, vec![0, 1, 31]),
                ("multiple-carry", 4099, 0, vec![MAX]),
                ("mixed-contended", 4099, 0, vec![0, 1, 31, MAX, MAX + 1024]),
            ] {
                let mut sums = vec![initial; PIXELS];
                let lanes = |stride: usize, sums: &[u64]| {
                    let mut words = Vec::new();
                    for (pixel, sum) in sums.iter().enumerate() {
                        words.extend([0x1234_0000 + pixel as u32, 0x2345_0000 + pixel as u32]);
                        if stride == 4 {
                            words.extend([*sum as u32, (*sum >> 32) as u32]);
                        } else {
                            words.push((*sum).min(u64::from(MAX)) as u32);
                        }
                    }
                    words
                };
                let initial_lanes = [lanes(3, &sums), lanes(stride, &sums)];
                let mut inputs = Vec::new();
                for i in 0..count {
                    let pixel = if name == "multiple-carry" || name == "mixed-contended" {
                        0
                    } else {
                        i % 7
                    };
                    let amount = additions[i % additions.len()];
                    inputs.extend([pixel as u32, (amount as f32 / 65536.0).to_bits()]);
                    sums[pixel] += u64::from(amount.min(MAX));
                }
                let make = |size| {
                    buffer(
                        d,
                        "occupancy contract",
                        size,
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    )
                };
                let input = make((inputs.len() * 4) as u64);
                q.write_buffer(&input, 0, bytemuck::cast_slice(&inputs));
                let outputs = [
                    make((PIXELS * 12) as u64),
                    make((PIXELS * stride * 4) as u64),
                ];
                let expected = [lanes(3, &sums), lanes(stride, &sums)];
                let mut raw = Vec::new();
                for repeat in 0..3 {
                    for i in 0..2 {
                        q.write_buffer(&outputs[i], 0, bytemuck::cast_slice(&initial_lanes[i]));
                    }
                    let mut encoder = d.create_command_encoder(&Default::default());
                    for i in 0..2 {
                        let bindings = group(
                            d,
                            &pipelines[i].get_bind_group_layout(0),
                            &[entry(0, &outputs[i], None), entry(1, &input, None)],
                        );
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipelines[i]);
                        pass.set_bind_group(0, &bindings, &[]);
                        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
                    }
                    q.submit([encoder.finish()]);
                    raw.clear();
                    for (i, output) in outputs.iter().enumerate() {
                        let actual: Vec<u32> =
                            read(d, q, output)?.iter().map(|v| v.to_bits()).collect();
                        ensure!(
                            actual == expected[i],
                            "Occupancy mode {mode} wide={wide} {name} repeat {repeat}, pipeline {i}: raw preclear limbs differ: {actual:?} != {:?}",
                            expected[i]
                        );
                        raw.push(actual);
                    }
                }
                for pixel in 0..PIXELS {
                    let low = raw[1][pixel * stride + 2];
                    let sum = if wide {
                        u64::from(low) | (u64::from(raw[1][pixel * stride + 3]) << 32)
                    } else {
                        u64::from(low)
                    };
                    ensure!(
                        sum.min(u64::from(MAX)) as u32 == raw[0][pixel * 3 + 2],
                        "Wide decoded occupancy differs from frozen scalar reference"
                    );
                }
                reports.push(json!({"mode":mode,"wide":wide,"stride":stride,"case":name,
                    "invocations":count,"repeats":3,"partial_workgroup":true,"raw_lanes_exact":true,"decoded_occupancy_exact":true,
                    "xy_sentinels_unchanged":true,"untouched_pixels":4,"host_u64_sums":sums,
                    "reference_raw":raw[0],"candidate_raw":raw[1]}));
            }
        }
    }
    Ok(
        json!({"preclear_occupancy":reports,"frozen_scalar_reference":true,"actual_production_function":true,"layout_transitions":layout_transitions(d,q)?,"canvas_decode":canvas_decode(d,q)?,"fourier_preclear":fourier_preclear(d,q)?,"renderer_transitions":renderer_transitions(d,q)?}),
    )
}

fn layout_transitions(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    use crate::network::Mode;
    let mut reports = Vec::new();
    for mode in [Mode::Mlp, Mode::Gru] {
        let mut project = Project::default();
        project.config_mut().switch_mode(mode, 0.25);
        let sizing = Sizing {
            canvas: [32, 32],
            entity_count: 67,
            sqrt_world_size: 1.0,
        };
        let mut engines = [
            Engine::new_with_splat_lanes(d, q, sizing, &project, 64, false)?,
            Engine::new_with_splat_lanes(d, q, sizing, &project, 64, true)?,
        ];
        ensure!(
            engines[0].splat_capacity_lanes() == 3 && engines[1].splat_capacity_lanes() == 4,
            "Native small-layout fixture must exercise forced fallback and wide allocation"
        );
        let camera = texture(
            d,
            "transition camera",
            [1, 1],
            wgpu::TextureFormat::Rg16Float,
        );
        let camera_view = camera.create_view(&Default::default());
        q.write_texture(
            camera.as_image_copy(),
            bytemuck::cast_slice(&[0x2800u16, 0]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let field: Vec<u16> = (0..1024).flat_map(|_| [0x2000u16, 0, 0, 0]).collect();
        for engine in &mut engines {
            engine.set_camera_field(d, &camera_view);
            engine.set_camera_strengths([0.5, 0.25]);
            q.write_texture(
                engine.field_texture(q).as_image_copy(),
                bytemuck::cast_slice(&field),
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
        let exact = |a: &wgpu::Buffer, b: &wgpu::Buffer| -> Result<()> {
            let a: Vec<u32> = read(d, q, a)?.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = read(d, q, b)?.iter().map(|v| v.to_bits()).collect();
            ensure!(a == b, "Wide/fallback particle or recurrent state differs");
            Ok(())
        };
        for (active_mode, revision) in [
            (Mode::Fourier, 1),
            (mode, 1),
            (Mode::Fourier, 1),
            (mode, 2),
            (Mode::Fourier, 1),
            (mode, 3),
            (Mode::Fourier, 1),
            (mode, 3),
            (mode, 3),
            (Mode::Fourier, 1),
            (Mode::Fourier, 1),
        ] {
            project.config_mut().switch_mode(active_mode, 0.25);
            if active_mode != Mode::Fourier {
                project
                    .config_mut()
                    .networks
                    .as_mut()
                    .unwrap()
                    .banks
                    .iter_mut()
                    .find(|bank| bank.mode == active_mode)
                    .unwrap()
                    .revision = revision;
            }
            for engine in &mut engines {
                engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
                let before = (
                    engine.frame_count,
                    engine.splat_stride(),
                    engine.prepared_steps,
                );
                let identities = (
                    engine.canvas_views.clone(),
                    engine.field_view.clone(),
                    engine.camera_field_view.clone(),
                    engine.entities.clone(),
                    engine.interaction_group.clone(),
                );
                let held_field = texture_pixels(d, q, &engine.field)?;
                let expected_replacement = trail_format(active_mode) != engine.canvas[0].format();
                let preview = engine.prepare_project(d, q, &project)?;
                ensure!(
                    preview.canvas_views().is_some() == expected_replacement,
                    "Prepared trail replacement decision incorrect"
                );
                drop(preview);
                let mut invalid = project.clone();
                invalid.world.trail_persistence = f64::NAN;
                ensure!(
                    engine.prepare_project(d, q, &invalid).is_err(),
                    "Invalid preparation unexpectedly succeeded"
                );
                ensure!(
                    engine.canvas_views == identities.0
                        && engine.field_view == identities.1
                        && engine.camera_field_view == identities.2
                        && engine.entities == identities.3
                        && engine.interaction_group == identities.4,
                    "Cancelled/failed preparation replaced active resources"
                );
                ensure!(
                    before
                        == (
                            engine.frame_count,
                            engine.splat_stride(),
                            engine.prepared_steps
                        ),
                    "Abandoned preview changed active layout or prepared frame"
                );
                let old_stride = engine.splat_stride();
                engine.upload_project(d, q, &project)?;
                ensure!(
                    (engine.canvas_views != identities.0) == expected_replacement,
                    "Commit replaced same-format trails or retained old-format trails"
                );
                ensure!(
                    engine.canvas[0].format() == trail_format(active_mode),
                    "Committed trail format incorrect"
                );
                ensure!(
                    engine.field_view == identities.1
                        && engine.camera_field_view == identities.2
                        && engine.entities == identities.3
                        && engine.interaction_group == identities.4,
                    "Trail transition replaced unrelated active resources"
                );
                ensure!(
                    texture_pixels(d, q, &engine.field)? == held_field,
                    "Trail transition erased painted field"
                );
                if old_stride != engine.splat_stride() {
                    ensure!(
                        engine.prepared_steps == 0,
                        "Layout transition retained stale prepared uniforms"
                    );
                }
                let paused_frame = engine.frame_count;
                engine.prepare_frame(q, &project.world, 0, None, [0.0; 2])?;
                ensure!(
                    engine.prepared_steps == 0 && engine.frame_count == paused_frame,
                    "Zero-step frame advanced simulation"
                );
                for _ in 0..3 {
                    engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
                    ensure!(
                        engine.canvas_uniform_words[9] == engine.splat_stride(),
                        "Canvas stride uniform stale"
                    );
                    let mut encoder = d.create_command_encoder(&Default::default());
                    engine.encode_step(&mut encoder, 0);
                    q.submit([encoder.finish()]);
                }
            }
            exact(&engines[0].entities, &engines[1].entities)?;
            exact(&engines[0].brain_memory, &engines[1].brain_memory)?;
            let canvases: Vec<_> = engines
                .iter()
                .map(|e| canvas_pixels(d, q, e))
                .collect::<Result<_>>()?;
            ensure!(
                canvases[0]
                    .iter()
                    .flatten()
                    .map(|v| v.to_bits())
                    .eq(canvases[1].iter().flatten().map(|v| v.to_bits())),
                "Wide/fallback canvas differs after revision transition"
            );
            reports.push(json!({"mode":active_mode,"revision":revision,"active_lanes":engines[1].splat_stride(),
                "forced_fallback":true,"partial_workgroup":true,"zero_steps":true,"abandoned_preview":true,
                "stale_uniform_guard":true,"entities_memory_canvas_exact":true}));
        }
    }
    // A reset must clear spare capacity too, even while the active path is narrow.
    for allow_wide in [false, true] {
        let project = Project::default();
        let mut engine = Engine::new_with_splat_lanes(
            d,
            q,
            Sizing {
                canvas: [8, 8],
                entity_count: 67,
                sqrt_world_size: 1.0,
            },
            &project,
            64,
            allow_wide,
        )?;
        q.write_buffer(&engine.splat, 0, &vec![0x55; engine.splat.size() as usize]);
        engine.reset();
        engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
        let mut encoder = d.create_command_encoder(&Default::default());
        engine.encode_step(&mut encoder, 0);
        q.submit([encoder.finish()]);
        ensure!(
            read(d, q, &engine.splat)?.iter().all(|v| v.to_bits() == 0),
            "Reset left stale splat allocation lanes"
        );
        reports.push(json!({"reset_full_allocation_clear":true,"layout":engine.splat_layout()}));
    }
    Ok(json!(reports))
}

fn probe_source() -> Result<String> {
    let production = shader("entityUpdate.wgsl");
    let start = production
        .find("fn add_occupancy(")
        .context("Missing occupancy function")?;
    let end = production[start..]
        .find("fn footprint_amount(")
        .context("Missing occupancy function end")?
        + start;
    let occupancy_source = format!(
        "{}\n{}",
        production
            .lines()
            .filter(|line| line.starts_with("const DENSITY_FIXED_SCALE:")
                || line.starts_with("const DENSITY_FIXED_MAX:"))
            .collect::<Vec<_>>()
            .join("\n"),
        &production[start..end]
    );
    Ok(format!(
        "override NETWORK_MODE:u32=0u; override WIDE_OCCUPANCY:bool=false; override STRIDE:u32=3u;\n@group(0) @binding(0) var<storage,read_write> splat:array<atomic<i32>>;\n\
         @group(0) @binding(1) var<storage,read> inputs:array<vec2u>;\n{}\n{}\n\
         @compute @workgroup_size(64) fn reference(@builtin(global_invocation_id) id:vec3u) {{\n\
         if(id.x>=arrayLength(&inputs)){{return;}} let v=inputs[id.x];add_occupancy_reference(v.x*STRIDE+2u,bitcast<f32>(v.y));}}\n\
         @compute @workgroup_size(64) fn candidate(@builtin(global_invocation_id) id:vec3u) {{\n\
         if(id.x>=arrayLength(&inputs)){{return;}} let v=inputs[id.x];add_occupancy(v.x*STRIDE+2u,bitcast<f32>(v.y));}}",
        occupancy_source,
        include_str!("occupancy_reference.wgsl"),
    ))
}

fn renderer_transitions(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    use crate::{
        config::Preferences,
        coords::Camera,
        network::Mode,
        render::{Overlays, Renderer},
    };
    let mut project = Project::default();
    let mut engine = Engine::new(
        d,
        q,
        Sizing {
            canvas: [32, 32],
            entity_count: 67,
            sqrt_world_size: 1.0,
        },
        &project,
        64,
    )?;
    let mut reused = Renderer::new(d, &engine, [64, 64])?;
    let camera = Camera {
        trail_mode: true,
        ..Default::default()
    };
    let prefs = Preferences::default();
    let mut count = 0u16;
    let mut previous_output = None;
    for mode in [
        Mode::Fourier,
        Mode::Mlp,
        Mode::Gru,
        Mode::Fourier,
        Mode::Gru,
        Mode::Fourier,
    ] {
        project.config_mut().switch_mode(mode, 0.25);
        let prepared = engine.prepare_project(d, q, &project)?;
        let bindings = prepared
            .canvas_views()
            .map(|views| reused.prepare_trail_bindings(d, views));
        let replaced = bindings.is_some();
        engine.commit_project(prepared);
        if let Some(bindings) = bindings {
            reused.commit_trail_bindings(bindings);
        }
        if replaced {
            for canvas in &engine.canvas {
                ensure!(
                    texture_storage_bytes(d, q, canvas)?.iter().all(|v| *v == 0),
                    "Paused new-format canvas not initialized zero"
                );
            }
        }
        engine.prepare_frame(q, &project.world, 0, None, [0.0; 2])?;
        ensure!(
            engine.prepared_steps == 0,
            "Paused format transition prepared simulation steps"
        );
        for _ in 0..3 {
            engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
            let mut encoder = d.create_command_encoder(&Default::default());
            engine.encode_step(&mut encoder, 0);
            q.submit([encoder.finish()]);
        }
        // Nonzero held vectors make a stale reference to the discarded canvas observable.
        let channels = engine.canvas[engine.current_canvas]
            .format()
            .block_copy_size(None)
            .unwrap() as usize
            / 2;
        let data: Vec<u16> = (0..1024)
            .flat_map(|_| {
                (0..channels).map(move |lane| {
                    if lane == 0 {
                        0x4000u16 + count * 0x100
                    } else {
                        0
                    }
                })
            })
            .collect();
        q.write_texture(
            engine.canvas[engine.current_canvas].as_image_copy(),
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(32 * channels as u32 * 2),
                rows_per_image: Some(32),
            },
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
        );
        let mut fresh = Renderer::new(d, &engine, [64, 64])?;
        for renderer in [&mut reused, &mut fresh] {
            renderer.begin_frame(q, &engine, &camera, &project, 1, -1.0);
            let mut encoder = d.create_command_encoder(&Default::default());
            renderer.clear_accumulator(&mut encoder);
            renderer.sample(&mut encoder, &engine, &camera);
            renderer.present(
                q,
                &mut encoder,
                &engine,
                &camera,
                &prefs,
                Overlays::default(),
            );
            q.submit([encoder.finish()]);
        }
        let output = texture_storage_bytes(d, q, reused.output_texture())?;
        ensure!(
            output == texture_storage_bytes(d, q, fresh.output_texture())?,
            "Reused Renderer samples stale trails after format transition"
        );
        if let Some(previous) = previous_output {
            ensure!(
                output != previous,
                "Trail sentinel did not distinguish old and new bindings"
            );
        }
        previous_output = Some(output);
        reused.rebind_trails(d, &engine.canvas_views);
        count += 1;
    }
    Ok(
        json!({"transitions":count,"prepared_renderer_bindings":true,"reused_vs_fresh_rgba8_exact":true,"paused_transition":true,"nonzero_trail_sentinel":true}),
    )
}

fn fourier_preclear(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    let project = Project::default();
    let source = shader("entityUpdate.wgsl");
    let stride = "let stride = select(2u, select(3u, 4u, WIDE_OCCUPANCY), NETWORK_MODE != 0u);";
    ensure!(
        source.contains(stride),
        "Fourier reference stride anchor missing"
    );
    let reference = source.replace(stride, "let stride = select(3u, 4u, WIDE_OCCUPANCY);");
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("original three-lane Fourier deposit"),
        source: wgpu::ShaderSource::Wgsl(reference.into()),
    });
    let mut reports = Vec::new();
    for allow_wide in [false, true] {
        let sizing = Sizing {
            canvas: [32, 32],
            entity_count: 67,
            sqrt_world_size: 1.0,
        };
        let mut engines = [
            Engine::new_with_splat_lanes(d, q, sizing, &project, 64, allow_wide)?,
            Engine::new_with_splat_lanes(d, q, sizing, &project, 64, allow_wide)?,
        ];
        let layouts = [
            engines[0].entity_pipeline.get_bind_group_layout(0),
            engines[0].entity_pipeline.get_bind_group_layout(1),
        ];
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("original Fourier deposit layout"),
            bind_group_layouts: &[
                Some(&layouts[0]),
                Some(&layouts[1]),
                Some(&engines[0].interaction_layout),
            ],
            immediate_size: 0,
        });
        engines[0].entity_pipeline = compute(
            d,
            &module,
            "main",
            Some(&layout),
            &[
                ("CONFIG_PER_COHORT", 1.0),
                ("WORKGROUP_SIZE", 64.0),
                ("NETWORK_MODE", 0.0),
                ("BRAIN_REVISION", 1.0),
                ("WIDE_OCCUPANCY", 0.0),
            ],
        );
        for tick in 0..4 {
            if tick == 1 {
                let particles: Vec<f32> = (0..67)
                    .flat_map(|index| {
                        [
                            (index % 8) as f32 * 0.25 - 0.875,
                            (index / 8) as f32 * 0.22 - 0.875,
                            0.05,
                            -0.03,
                            0.025,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    })
                    .collect();
                for engine in &engines {
                    q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
                }
            }
            for engine in &mut engines {
                engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
                let mut encoder = d.create_command_encoder(&Default::default());
                engine.encode_bake(&mut encoder);
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&engine.entity_pipeline);
                    pass.set_bind_group(0, &engine.entity_state, &[0]);
                    pass.set_bind_group(
                        1,
                        &engine.entity_textures[engine.wrap][engine.current_canvas],
                        &[],
                    );
                    pass.set_bind_group(2, &engine.interaction_group, &[]);
                    pass.dispatch_workgroups(2, 1, 1);
                }
                q.submit([encoder.finish()]);
                engine.frame_count += 1;
            }
            let raw = [
                read(d, q, &engines[0].splat)?,
                read(d, q, &engines[1].splat)?,
            ];
            for pixel in 0..1024 {
                for lane in 0..2 {
                    ensure!(
                        raw[0][pixel * 3 + lane].to_bits() == raw[1][pixel * 2 + lane].to_bits(),
                        "Fourier preclear momentum differs at tick {tick}, pixel {pixel}, lane {lane}"
                    );
                }
                ensure!(
                    raw[0][pixel * 3 + 2].to_bits() == 0,
                    "Fourier reference wrote occupancy"
                );
            }
            ensure!(
                raw[1][2048..].iter().all(|v| v.to_bits() == 0),
                "Two-lane entity wrote unused capacity"
            );
            if tick > 0 {
                ensure!(
                    raw[1][..2048].iter().any(|v| v.to_bits() != 0),
                    "Preclear probe deposited no momentum"
                );
            }
            let mut encoder = d.create_command_encoder(&Default::default());
            for engine in &engines {
                encoder.clear_buffer(&engine.splat, 0, None);
            }
            q.submit([encoder.finish()]);
        }
        reports.push(json!({"capacity_lanes":engines[1].splat_capacity_lanes(),"ticks":4,
            "partial_workgroup":true,"actual_entity_shader":true,"preclear_xy_exact":true,"unused_capacity_zero":true}));
    }
    Ok(json!(reports))
}

fn canvas_decode(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    use crate::network::Mode;
    let mut project = Project::default();
    project.config_mut().switch_mode(Mode::Gru, 0.25);
    let mut outputs = Vec::new();
    for (allow_wide, fourier) in [(false, false), (true, false), (false, true), (true, true)] {
        project
            .config_mut()
            .switch_mode(if fourier { Mode::Fourier } else { Mode::Gru }, 0.25);
        let mut engine = Engine::new_with_splat_lanes(
            d,
            q,
            Sizing {
                canvas: [32, 32],
                entity_count: 67,
                sqrt_world_size: 1.0,
            },
            &project,
            64,
            allow_wide,
        )?;
        let stride = engine.splat_stride();
        ensure!(
            stride
                == if fourier {
                    2
                } else if allow_wide {
                    4
                } else {
                    3
                },
            "Canvas oracle layout unavailable"
        );
        let sums = [
            0u64,
            1,
            7_679_999,
            7_680_000,
            0x7fff_ffff,
            0xffff_ffff,
            0x1_0000_0007,
            0x9_0000_001f,
        ];
        let mut words = Vec::new();
        for pixel in 0..1024 {
            words.extend([1234u32, (-4321i32) as u32]);
            let sum = sums[pixel % sums.len()];
            if !fourier {
                if allow_wide {
                    words.extend([sum as u32, (sum >> 32) as u32]);
                } else {
                    words.push(sum.min(7_680_000) as u32);
                }
            }
        }
        q.write_buffer(&engine.splat, 0, &vec![0x55; engine.splat.size() as usize]);
        q.write_buffer(&engine.splat, 0, bytemuck::cast_slice(&words));
        engine.frame_count = 1;
        engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
        let back = 1 - engine.current_canvas;
        let mut encoder = d.create_command_encoder(&Default::default());
        draw(
            &mut encoder,
            &engine.canvas_views[back],
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            &engine.canvas_pipeline,
            &[
                (&engine.canvas_state, &[0]),
                (
                    &engine.canvas_textures[engine.wrap][engine.current_canvas],
                    &[],
                ),
            ],
        );
        q.submit([encoder.finish()]);
        engine.current_canvas = back;
        outputs.push(
            canvas_pixels(d, q, &engine)?
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
    }
    ensure!(
        outputs[0] == outputs[1],
        "Actual canvas decoder differs from legacy saturation at unsigned/carry boundaries"
    );
    ensure!(
        outputs[2] == outputs[3],
        "Fourier canvas differs across allocation capacities"
    );
    for pixel in 0..1024 {
        let offset = pixel * 4;
        ensure!(
            outputs[2][offset..offset + 2] == outputs[0][offset..offset + 2],
            "Two-lane momentum decode differs at pixel {pixel}"
        );
        ensure!(
            outputs[2][offset + 2..offset + 4] == [0, 0],
            "Two-lane decoder consumed unused splat data at pixel {pixel}"
        );
    }
    Ok(
        json!({"actual_canvas_shader":true,"raw_rgba16f_exact":true,"cases":8,"pixels":1024,"momentum_unchanged":true,
            "fourier_two_lane":true,"capacity_controls":[3,4],"unused_tail_sentinels":true,"last_texel":true}),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn occupancy_probe_composes_actual_production_and_frozen_reference() {
        let source = super::probe_source().unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
