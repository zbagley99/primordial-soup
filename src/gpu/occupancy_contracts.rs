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
        json!({"preclear_occupancy":reports,"frozen_scalar_reference":true,"actual_production_function":true,"layout_transitions":layout_transitions(d,q)?,"canvas_decode":canvas_decode(d,q)?}),
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
            engines[0].splat_stride() == 3 && engines[1].splat_capacity_lanes() == 4,
            "Native small-layout fixture must exercise forced fallback and wide allocation"
        );
        let exact = |a: &wgpu::Buffer, b: &wgpu::Buffer| -> Result<()> {
            let a: Vec<u32> = read(d, q, a)?.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = read(d, q, b)?.iter().map(|v| v.to_bits()).collect();
            ensure!(a == b, "Wide/fallback particle or recurrent state differs");
            Ok(())
        };
        for revision in [2, 3, 2, 3] {
            project
                .config_mut()
                .networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|bank| bank.mode == mode)
                .unwrap()
                .revision = revision;
            for engine in &mut engines {
                engine.prepare_frame(q, &project.world, 1, None, [0.0; 2])?;
                let before = (
                    engine.frame_count,
                    engine.splat_stride(),
                    engine.prepared_steps,
                );
                let preview = engine.prepare_project(d, q, &project)?;
                drop(preview);
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
            reports.push(json!({"mode":mode,"revision":revision,"active_lanes":engines[1].splat_stride(),
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

fn canvas_decode(d: &wgpu::Device, q: &wgpu::Queue) -> Result<Value> {
    use crate::network::Mode;
    let mut project = Project::default();
    project.config_mut().switch_mode(Mode::Gru, 0.25);
    let mut outputs = Vec::new();
    for allow_wide in [false, true] {
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
            stride == if allow_wide { 4 } else { 3 },
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
            if allow_wide {
                words.extend([sum as u32, (sum >> 32) as u32]);
            } else {
                words.push(sum.min(7_680_000) as u32);
            }
        }
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
    Ok(
        json!({"actual_canvas_shader":true,"raw_rgba16f_exact":true,"cases":8,"pixels":1024,"momentum_unchanged":true}),
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
