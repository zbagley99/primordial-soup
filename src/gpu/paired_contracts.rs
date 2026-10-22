use super::*;

pub fn check(d: &wgpu::Device, q: &wgpu::Queue, path: Option<&str>) -> Result<Value> {
    use crate::network::{self, Bank, Mode};
    const COUNT: usize = 67;
    const TICKS: usize = 8;
    let candidate = match path {
        Some(path) => std::fs::read_to_string(path)?,
        None => shader("network.wgsl").to_owned(),
    };
    let planar_history = candidate.contains("fn brain_state_index(");
    let reference = include_str!("network_reference.wgsl").trim_end();
    let candidate =
        if candidate.starts_with(reference) && candidate.contains("fn brain_evaluate_pair(") {
            candidate
        } else {
            let start = candidate
                .find("fn brain_row(")
                .context("Missing candidate brain_row")?;
            let reference_start = reference.find("fn brain_row(").unwrap();
            ensure!(
                candidate[..start] == reference[..reference_start],
                "Candidate must preserve network bindings, overrides and input limit"
            );
            let mut functions = candidate[start..].to_owned();
            for name in [
                "brain_row",
                "brain_input_row",
                "brain_bias",
                "brain_state_stride",
                "brain_state_index",
                "brain_heading_index",
                "brain_clear",
                "brain_motor",
                "brain_mlp_output",
                "brain_evaluate",
            ] {
                functions = functions.replace(&format!("{name}("), &format!("{name}_candidate("));
            }
            format!(
                "{reference}\n{functions}\nstruct BrainPair {{ base:vec4f, mirror:vec4f }}\n\
             fn brain_evaluate_pair(slot:i32,a:vec4f,b:vec4f,index:u32,extra:vec3f)->BrainPair {{\n\
             return BrainPair(brain_evaluate_candidate(slot,a,index,0u,extra),\n\
             brain_evaluate_candidate(slot,b,index,1u,extra.yxz));\n}}"
            )
        };
    let source = format!("{candidate}\n{}", include_str!("network_pair_probe.wgsl"));
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paired network contract"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let make = |words: usize| {
        buffer(
            d,
            "paired network contract",
            (words * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    };
    let exact = |reference: &[f32], paired: &[f32], context: &str| -> Result<()> {
        ensure!(reference.len() == paired.len(), "{context}: length differs");
        if let Some(i) = reference
            .iter()
            .zip(paired)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            anyhow::bail!(
                "{context}: word {i} differs: scalar={:08x}, paired={:08x}",
                reference[i].to_bits(),
                paired[i].to_bits()
            );
        }
        ensure!(
            reference.iter().all(|v| v.is_finite()),
            "{context}: nonfinite state"
        );
        Ok(())
    };
    let mut results = Vec::new();
    for mode in [Mode::Mlp, Mode::Gru] {
        for revision in 1..=3 {
            let bank = Bank {
                mode,
                revision,
                parent: None,
                seed: 0.125,
                scale: 0.17,
            };
            let mut weights = Vec::new();
            for cohort in 0..3 {
                let mut row = network::derive(&bank, cohort);
                if cohort == 2 {
                    for (i, value) in row.iter_mut().enumerate() {
                        *value = if i % 2 == 0 { -8.0 } else { 8.0 };
                    }
                }
                weights.extend_from_slice(&row);
                weights.resize(weights.len() + network::WEIGHT_STRIDE - row.len(), 0.0);
            }
            let weight = make(weights.len());
            q.write_buffer(&weight, 0, bytemuck::cast_slice(&weights));
            let input = make(COUNT * 12);
            let outputs = [make(COUNT * 8), make(COUNT * 8)];
            let stride = if revision >= 3 {
                if mode == Mode::Gru { 3 } else { 1 }
            } else {
                2
            };
            let memories = [make(COUNT * stride * 4), make(COUNT * stride * 4)];
            let pipelines = ["reference", "paired"].map(|entry| {
                compute(
                    d,
                    &module,
                    entry,
                    None,
                    &[
                        ("NETWORK_MODE", mode.id() as f64),
                        ("BRAIN_REVISION", revision as f64),
                    ],
                )
            });
            let groups: [_; 2] = std::array::from_fn(|i| {
                group(
                    d,
                    &pipelines[i].get_bind_group_layout(0),
                    &[
                        entry(0, &input, None),
                        entry(1, &outputs[i], None),
                        entry(5, &weight, None),
                        entry(6, &memories[i], None),
                    ],
                )
            });
            for zero in [false, true] {
                let initial: Vec<f32> = (0..COUNT * stride * 4)
                    .map(|i| {
                        if zero {
                            0.0
                        } else {
                            (i % 29) as f32 * 0.031 - 0.4
                        }
                    })
                    .collect();
                for (i, memory) in memories.iter().enumerate() {
                    let native = if i == 1 && mode == Mode::Gru && planar_history {
                        planar_gru_history(bytemuck::cast_slice(&initial), COUNT, stride)?
                    } else {
                        bytemuck::cast_slice(&initial).to_vec()
                    };
                    q.write_buffer(memory, 0, &native);
                }
                for tick in 0..TICKS {
                    let mut values = Vec::with_capacity(COUNT * 12);
                    for i in 0..COUNT {
                        let normal = 0.17 + i as f32 * 0.013 + tick as f32 * 0.021;
                        let a = match i % 7 {
                            0 => [0.0, -0.0, 0.0, -0.0],
                            1 => [f32::MAX, -f32::MAX, f32::MAX, -f32::MAX],
                            2 => [network::INPUT_LIMIT, -network::INPUT_LIMIT, 1.0, -1.0],
                            3 => [1e-30, -1e-30, 1e-20, -1e-20],
                            _ => [normal, -0.31, 0.43, normal * -0.7],
                        };
                        let b = [-a[2], a[0], -a[3], a[1]];
                        values.extend(a);
                        values.extend(b);
                        values.extend([
                            (i % 5) as f32 / 4.0,
                            (i % 7) as f32 / 6.0,
                            tick as f32 / 8.0,
                            0.0,
                        ]);
                    }
                    q.write_buffer(&input, 0, bytemuck::cast_slice(&values));
                    let mut encoder = d.create_command_encoder(&Default::default());
                    for i in 0..2 {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipelines[i]);
                        pass.set_bind_group(0, &groups[i], &[]);
                        pass.dispatch_workgroups((COUNT as u32).div_ceil(64), 1, 1);
                    }
                    q.submit([encoder.finish()]);
                    let context = format!("{mode:?} revision {revision}, zero={zero}, tick={tick}");
                    let reference_output = read(d, q, &outputs[0])?;
                    let paired_output = read(d, q, &outputs[1])?;
                    ensure!(
                        reference_output.iter().any(|v| *v != 0.0),
                        "{context}: fixture did not exercise nonzero outputs"
                    );
                    exact(
                        &reference_output,
                        &paired_output,
                        &format!("{context} outputs"),
                    )?;
                    let reference_memory = read(d, q, &memories[0])?;
                    let native_memory = read(d, q, &memories[1])?;
                    let paired_memory = if mode == Mode::Gru && planar_history {
                        interleaved_history_values(&native_memory, COUNT, stride)?
                    } else {
                        native_memory
                    };
                    exact(
                        &reference_memory,
                        &paired_memory,
                        &format!("{context} memory"),
                    )?;
                    if mode == Mode::Gru && tick == 0 {
                        ensure!(
                            initial
                                .iter()
                                .zip(&reference_memory)
                                .any(|(a, b)| a.to_bits() != b.to_bits()),
                            "{context}: fixture did not exercise recurrent updates"
                        );
                    }
                    if mode == Mode::Mlp {
                        exact(
                            &initial,
                            &paired_memory,
                            &format!("{context} MLP memory unchanged"),
                        )?;
                    } else if revision >= 3 {
                        for i in 0..COUNT {
                            let start = (i * stride + 2) * 4;
                            exact(
                                &initial[start..start + 4],
                                &paired_memory[start..start + 4],
                                &format!("{context} heading {i} unchanged"),
                            )?;
                        }
                    }
                }
                results.push(json!({"mode":mode,"revision":revision,"zero_initial_state":zero,"particles":COUNT,"ticks":TICKS,"outputs_byte_exact":true,"memory_byte_exact":true}));
            }
        }
    }
    Ok(
        json!({"candidate":path.unwrap_or("embedded production shader"),"embedded_production":path.is_none(),"same_device_scalar_reference":true,"native_diagnostic_only":true,"asymmetric_mirror_inputs":true,"mirrored_occupancy_inputs":true,"unequal_nonzero_histories":true,"extreme_finite_inputs_clamped":true,"partial_workgroup":true,"cases":results}),
    )
}
