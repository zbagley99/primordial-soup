//! Bounded native Vulkan validation and reproducible end-to-end GPU timing.
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use anyhow::{Context, Result, ensure};
    use primordial_soup::{
        config::{Preferences, Project},
        coords::{Camera, Sizing, sizing_for},
        gpu::{Engine, Paint},
        render::{Overlays, Renderer, blur_schedule},
    };
    use serde_json::json;
    use std::{
        path::Path,
        sync::{Arc, Mutex, mpsc},
        time::{Duration, Instant},
    };
    fn wait(d: &wgpu::Device) -> Result<()> {
        d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(20)),
        })?;
        Ok(())
    }
    fn read(d: &wgpu::Device, q: &wgpu::Queue, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounded validation readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = d.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(src, 0, &b, 0, size);
        q.submit([e.finish()]);
        mapped(d, &b)
    }
    fn mapped(d: &wgpu::Device, b: &wgpu::Buffer) -> Result<Vec<u8>> {
        let (tx, rx) = mpsc::channel();
        b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let result = (|| -> Result<Vec<u8>> {
            wait(d)?;
            rx.recv_timeout(Duration::from_secs(1))??;
            let view = b.slice(..).get_mapped_range()?;
            Ok(view.to_vec())
        })();
        // The view is dropped and even pending/error mappings are canceled before reuse.
        b.unmap();
        result
    }
    fn tex(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        t: &wgpu::Texture,
        size: [u32; 2],
        pixel: u32,
    ) -> Result<Vec<u8>> {
        let native_pixel = t
            .format()
            .block_copy_size(None)
            .context("Unsupported readback format")?;
        ensure!(
            native_pixel == pixel || (t.format() == wgpu::TextureFormat::Rg16Float && pixel == 8),
            "Texture readback byte width mismatch"
        );
        let pitch = (size[0] * native_pixel).next_multiple_of(256);
        let b = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture readback"),
            size: pitch as u64 * size[1] as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = d.create_command_encoder(&Default::default());
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: t,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &b,
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
        q.submit([e.finish()]);
        let raw = mapped(d, &b)?;
        let native: Vec<u8> = raw
            .chunks_exact(pitch as usize)
            .flat_map(|r| r[..(size[0] * native_pixel) as usize].iter().copied())
            .collect();
        // Stored-channel comparison: absent RG16 B/A are canonical zero, not sampled alpha.
        Ok(if native_pixel == pixel {
            native
        } else {
            native
                .chunks_exact(4)
                .flat_map(|rg| [rg[0], rg[1], rg[2], rg[3], 0, 0, 0, 0])
                .collect()
        })
    }
    fn step(d: &wgpu::Device, q: &wgpu::Queue, e: &mut Engine, p: &Project, n: u32) -> Result<()> {
        e.prepare_frame(q, &p.world, n, None, [0.01, 0.001])?;
        let mut c = d.create_command_encoder(&Default::default());
        for i in 0..n {
            e.encode_step(&mut c, i);
        }
        q.submit([c.finish()]);
        wait(d)
    }
    fn floats(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    fn finite_particles(b: &[u8]) -> Result<()> {
        for (i, e) in b.chunks_exact(32).enumerate() {
            let f = floats(e);
            ensure!(
                [0, 1, 2, 3, 4, 6, 7].iter().all(|j| f[*j].is_finite()),
                "Nonfinite entity {i}"
            );
        }
        Ok(())
    }
    fn finite_half(b: &[u8]) -> Result<()> {
        ensure!(
            b.chunks_exact(2)
                .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7c00 != 0x7c00),
            "Nonfinite fp16 field/canvas"
        );
        Ok(())
    }
    fn camera_smoke(d: &wgpu::Device, q: &wgpu::Queue) -> Result<()> {
        let mut camera = primordial_soup::webcam::Webcam::new(d);
        camera.settings.mirror = false;
        camera.settings.blur = 0.0;
        camera.settings.mapping = 0;
        let bytes: Vec<u8> = (0..64 * 64)
            .flat_map(|i| {
                let value = ((i % 64) * 4) as u8;
                [value, value, value, 255]
            })
            .collect();
        let mut commands = d.create_command_encoder(&Default::default());
        camera.ingest_rgba(d, q, &mut commands, [64, 64], &bytes, 1.0)?;
        q.submit([commands.finish()]);
        wait(d)?;
        let across = tex(d, q, camera.field_texture(), [256, 256], 4)?;
        finite_half(&across)?;
        let center = (128 * 256 + 128) * 4;
        let r = u16::from_le_bytes(across[center..center + 2].try_into().unwrap());
        let g = u16::from_le_bytes(across[center + 2..center + 4].try_into().unwrap());
        ensure!(
            r & 0x7fff > 0 && r & 0x8000 == 0 && g & 0x7fff == 0,
            "Camera across edge points incorrectly"
        );
        camera.settings.mapping = 1;
        let mut commands = d.create_command_encoder(&Default::default());
        camera.ingest_rgba(d, q, &mut commands, [64, 64], &bytes, 1.0)?;
        q.submit([commands.finish()]);
        wait(d)?;
        let along = tex(d, q, camera.field_texture(), [256, 256], 4)?;
        finite_half(&along)?;
        ensure!(
            u16::from_le_bytes(along[center..center + 2].try_into().unwrap()) & 0x7fff == 0
                && u16::from_le_bytes(along[center + 2..center + 4].try_into().unwrap()) == r,
            "Camera along edge is not quarter-turn of across"
        );
        camera.settings.mapping = 2;
        let mut commands = d.create_command_encoder(&Default::default());
        camera.ingest_rgba(d, q, &mut commands, [64, 64], &bytes, 1.0)?;
        q.submit([commands.finish()]);
        wait(d)?;
        let still = tex(d, q, camera.field_texture(), [256, 256], 4)?;
        ensure!(
            still
                .chunks_exact(2)
                .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7fff == 0),
            "Still camera frame generates motion"
        );
        let black: Vec<u8> = (0..64 * 64).flat_map(|_| [0, 0, 0, 255]).collect();
        let mut commands = d.create_command_encoder(&Default::default());
        camera.ingest_rgba(d, q, &mut commands, [64, 64], &black, 1.0)?;
        q.submit([commands.finish()]);
        wait(d)?;
        let motion = tex(d, q, camera.field_texture(), [256, 256], 4)?;
        finite_half(&motion)?;
        ensure!(
            motion
                .chunks_exact(2)
                .any(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7fff != 0),
            "Camera change generates no motion field"
        );
        Ok(())
    }
    fn smoke(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<serde_json::Value> {
        camera_smoke(d, q)?;
        export_target_smoke(d, q, workgroup)?;
        let renderer_uniforms = renderer_uniform_smoke(d, q, workgroup)?;
        let s = Sizing {
            entity_count: 4096,
            canvas: [64, 64],
            sqrt_world_size: 1.0,
        };
        let mut p = Project::default();
        let mut a = Engine::new(d, q, s, &p, workgroup)?;
        let mut b = Engine::new(d, q, s, &p, workgroup)?;
        step(d, q, &mut a, &p, 4)?;
        for _ in 0..4 {
            step(d, q, &mut b, &p, 1)?;
        }
        let particles = read(d, q, &a.entities, s.entity_count as u64 * 32)?;
        finite_particles(&particles)?;
        ensure!(
            particles == read(d, q, &b.entities, s.entity_count as u64 * 32)?,
            "Uniform ring differs from serial substeps"
        );
        let canvas = tex(d, q, &a.canvas[a.current_canvas], s.canvas, 8)?;
        finite_half(&canvas)?;
        ensure!(
            canvas == tex(d, q, &b.canvas[b.current_canvas], s.canvas, 8)?,
            "Canvas differs across uniform schedule"
        );
        ensure!(canvas.iter().any(|v| *v != 0), "Trails never deposited");
        ensure!(
            read(
                d,
                q,
                a.splat_buffer(),
                s.canvas[0] as u64 * s.canvas[1] as u64 * 12
            )?
            .iter()
            .all(|v| *v == 0),
            "Splat accumulator not drained"
        );
        // Freeze rule forces, then inject a known outside-domain entity to distinguish all boundaries.
        p.configs[0].global_force_mult = 0.0;
        p.configs[0].gravity_force = 0.0;
        p.configs[0].gravity_strafe = 0.0;
        p.configs[0].gravity_trails = 0.0;
        p.configs[0].hazard_rate = 0.0;
        p.configs[0].drag = 1.0;
        p.configs[0].cohort_fences = false;
        a.upload_project(d, q, &p)?;
        for bc in 0..4 {
            p.world.boundary_conditions = bc;
            a.reset();
            step(d, q, &mut a, &p, 1)?;
            let mut lanes = [0u32; 8];
            for (i, v) in [(0, 1.25f32), (1, 0.25), (2, 0.01), (3, 0.0), (4, 0.001)] {
                lanes[i] = v.to_bits();
            }
            q.write_buffer(&a.entities, 0, bytemuck::cast_slice(&lanes));
            step(d, q, &mut a, &p, 1)?;
            let bytes = read(d, q, &a.entities, 32)?;
            let f = floats(&bytes);
            let config = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as i32;
            if bc == 3 {
                ensure!(config < 0, "Kill boundary did not kill");
                a.reset();
                step(d, q, &mut a, &p, 1)?;
                let reset = read(d, q, &a.entities, 32)?;
                ensure!(
                    u32::from_le_bytes(reset[20..24].try_into().unwrap()) as i32 >= 0,
                    "Reset did not resurrect killed entity"
                );
            } else {
                ensure!(
                    f[0].abs() <= 1.0 && f[1].abs() <= 1.0,
                    "Boundary {bc} left entity outside domain: {:?}",
                    &f[..4]
                );
                if bc == 0 {
                    ensure!(f[2] < 0.0, "Bounce did not reflect velocity");
                }
                if bc == 1 {
                    ensure!(f[0] < 0.0, "Wrap did not cross to opposite edge");
                }
            }
        }
        // Paused rule bake/pick returns the baked GPU rule, with a full 24-bit index.
        p = Project::default();
        a.upload_project(d, q, &p)?;
        a.reset();
        step(d, q, &mut a, &p, 1)?;
        let bytes = read(d, q, &a.entities, s.entity_count as u64 * 32)?;
        let first = floats(&bytes[..32]);
        ensure!(
            a.request_pick(q, &p.world, [first[0], first[1]], 0.01, -1.0),
            "Pick request rejected"
        );
        let mut c = d.create_command_encoder(&Default::default());
        ensure!(a.encode_pick(&mut c), "Paused pick not encoded");
        q.submit([c.finish()]);
        a.begin_pick_readback();
        wait(d)?;
        let hit = a
            .take_pick()?
            .context("Picker not ready")?
            .context("Picker missed exact particle position")?;
        ensure!(
            hit.index == 0,
            "Exact-position deterministic picker did not select index0: {}",
            hit.index
        );
        ensure!(
            hit.rule.len() == 80 && hit.rule.iter().all(|v| v.is_finite()),
            "Invalid GPU-adopted rule"
        );
        let configs = read(d, q, &a.configs, a.config_count as u64 * 416)?;
        let slot = (hit.cohort as usize).min(a.config_count as usize - 1);
        let baked = floats(&configs[slot * 416..slot * 416 + 320]);
        ensure!(
            hit.rule.iter().zip(baked).all(|(a, b)| *a == b as f64),
            "Pick adopted a different rule from baked cohort"
        );
        let brush = Paint {
            uv: [0.5, 0.5],
            previous_uv: [0.3, 0.4],
            size: 0.08,
            power: 2.0,
            erase: false,
            mode: 0,
            layer: 0,
            angle: 0.0,
            line_gain: 1.0,
        };
        let mut c = d.create_command_encoder(&Default::default());
        a.clear_field(&mut c);
        a.paint(q, &mut c, brush)?;
        q.submit([c.finish()]);
        wait(d)?;
        let walls = tex(d, q, a.field_texture(q), a.field_size, 8)?;
        finite_half(&walls)?;
        ensure!(
            walls
                .chunks_exact(8)
                .all(|v| v[4..].iter().all(|b| *b == 0)),
            "Walls paint modified trails channels"
        );
        ensure!(walls.iter().any(|v| *v != 0), "Walls paint was empty");
        let mut c = d.create_command_encoder(&Default::default());
        a.paint(q, &mut c, Paint { layer: 1, ..brush })?;
        q.submit([c.finish()]);
        wait(d)?;
        let both = tex(d, q, a.field_texture(q), a.field_size, 8)?;
        finite_half(&both)?;
        ensure!(
            both.chunks_exact(8)
                .zip(walls.chunks_exact(8))
                .all(|(a, b)| a[..4] == b[..4]),
            "Trails paint modified walls channels"
        );
        ensure!(
            both.chunks_exact(8).any(|v| v[4..].iter().any(|b| *b != 0)),
            "Trails paint was empty"
        );
        Ok(
            json!({"renderer_uniform_cache":renderer_uniforms,"uniform_ring_matches_serial":true,"finite_particles_and_trails":true,"splat_drained":true,"boundary_modes":["bounce","wrap","reset","kill"],"reset_resurrects":true,"paused_gpu_pick_and_adopt":true,"independent_painted_layers":true,"synthetic_webcam_edges_motion":true,"lazy_export_target_matches_display":true,"repeated_exports_release_and_recreate":true,"resize_invalidates_export_target":true,"cropped_png_readback_matches_cpu_oracle":true,"particles":4096,"canvas":[64,64]}),
        )
    }
    fn renderer_uniform_smoke(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::Style;
        let mut project = Project::default();
        project
            .config_mut()
            .switch_mode(primordial_soup::network::Mode::Gru, 0.875);
        let mut engine = Engine::new(
            d,
            q,
            Sizing {
                entity_count: 4096,
                canvas: [64, 64],
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        step(d, q, &mut engine, &project, 48)?;
        let frame = engine.frame_count;
        let state = || -> Result<Vec<Vec<u8>>> {
            let mut bytes = Vec::new();
            for buffer in [
                &engine.entities,
                &engine.configs,
                &engine.brain_memory,
                engine.splat_buffer(),
            ] {
                bytes.push(read(d, q, buffer, buffer.size())?);
            }
            bytes.push(mapped(d, &engine.snapshot_rules(d, q))?);
            for canvas in &engine.canvas {
                bytes.push(tex(d, q, canvas, engine.sizing.canvas, 8)?);
            }
            bytes.push(tex(
                d,
                q,
                engine.audit_field_texture(),
                engine.field_size,
                8,
            )?);
            Ok(bytes)
        };
        let before = state()?;
        let mut prefs = Preferences::default();
        prefs.cells.style = Style::Classic;
        let mut camera = Camera::default();
        let mut samples = 1;
        let mut highlight = -1.0;
        let mut size = [160, 128];
        let mut reused = Renderer::new(d, &engine, size)?;
        let cases = [
            "warm",
            "unchanged",
            "pan_zoom",
            "palette",
            "highlight",
            "temporal_multi",
            "temporal_single",
            "bloom_parameters",
            "bloom_off",
            "bloom_on",
            "resize",
            "cells",
            "trail_view",
            "classic",
            "replacement",
        ];
        for name in cases {
            match name {
                "pan_zoom" => {
                    camera.pan = [0.2, -0.15];
                    camera.zoom = 2.0;
                }
                "palette" => {
                    project.config_mut().color_sensitivity = 0.35;
                    project.config_mut().color_offset = 0.25;
                }
                "highlight" => highlight = 0.0,
                "temporal_multi" => samples = 3,
                "temporal_single" => samples = 1,
                "bloom_parameters" => {
                    prefs.bloom_threshold = 0.15;
                    prefs.bloom_radius = 1.7;
                }
                "bloom_off" => prefs.bloom_enabled = false,
                "bloom_on" => prefs.bloom_enabled = true,
                "resize" => {
                    size = [192, 96];
                    reused.resize(d, &engine, size)?;
                }
                "cells" => prefs.cells.style = Style::Cells,
                "trail_view" => camera.trail_mode = true,
                "classic" => {
                    camera.trail_mode = false;
                    prefs.cells.style = Style::Classic;
                }
                "replacement" => reused = Renderer::new(d, &engine, size)?,
                _ => {}
            }
            let mut fresh = Renderer::new(d, &engine, size)?;
            for renderer in [&mut reused, &mut fresh] {
                if prefs.cells.style == Style::Cells && renderer.cells.is_none() {
                    renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
                }
                cells_frame(
                    d, q, &engine, renderer, &project, &prefs, &camera, samples, highlight,
                )?;
            }
            ensure!(
                tex(d, q, reused.output_texture(), size, 4)?
                    == tex(d, q, fresh.output_texture(), size, 4)?,
                "Cached renderer display differs from fresh renderer: {name}"
            );
            ensure!(
                tex(d, q, reused.clean_texture()?, size, 4)?
                    == tex(d, q, fresh.clean_texture()?, size, 4)?,
                "Cached renderer clean export differs from fresh renderer: {name}"
            );
        }
        ensure!(
            engine.frame_count == frame && state()? == before,
            "Rendering changed frozen simulation state"
        );
        Ok(
            json!({"cases":cases,"display_and_clean_exact":true,"simulation_state_unchanged":true,
            "mode":"gru","ticks":frame,"particles":4096}),
        )
    }
    fn export_target_smoke(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<()> {
        let p = Project::default();
        let mut e = Engine::new(
            d,
            q,
            Sizing {
                entity_count: 4096,
                canvas: [64, 64],
                sqrt_world_size: 1.0,
            },
            &p,
            workgroup,
        )?;
        step(d, q, &mut e, &p, 1)?;
        let mut r = Renderer::new(d, &e, [160, 128])?;
        let prefs = Preferences::default();
        let camera = Camera::default();
        ensure!(
            r.clean_output_bytes() == 0 && r.clean_texture().is_err(),
            "Idle renderer allocated an export target"
        );
        for size in [[160, 128], [192, 96]] {
            if r.resize(d, &e, size)? {
                ensure!(
                    r.clean_output_bytes() == 0 && r.clean_texture().is_err(),
                    "Resize retained an export target"
                );
            }
            for _ in 0..2 {
                r.begin_frame(q, &e, &camera, &p, 1, -1.0);
                let mut encoder = d.create_command_encoder(&Default::default());
                r.clear_accumulator(&mut encoder);
                r.sample(&mut encoder, &e, &camera);
                r.present(q, &mut encoder, &e, &camera, &prefs, Overlays::default());
                r.present_clean(d, q, &mut encoder, &e, &camera, &prefs);
                q.submit([encoder.finish()]);
                ensure!(
                    r.clean_output_bytes() == size[0] as u64 * size[1] as u64 * 4,
                    "Export target has wrong logical storage"
                );
                let clean = tex(d, q, r.clean_texture()?, size, 4)?;
                ensure!(
                    clean == tex(d, q, r.output_texture(), size, 4)?,
                    "Clean export differs from display without overlays"
                );
                ensure!(
                    clean
                        .chunks_exact(4)
                        .any(|p| p[..3].iter().any(|v| *v != 0)),
                    "Export rendered no particles"
                );
                for (origin, crop) in [
                    ([7, 9], [33, 17]),
                    ([0, 0], size),
                    ([size[0] - 31, size[1] - 13], [31, 13]),
                ] {
                    let mut encoder = d.create_command_encoder(&Default::default());
                    let readback = primordial_soup::gpu::readback_rgba(
                        d,
                        &mut encoder,
                        r.clean_texture()?,
                        origin,
                        crop,
                    )?;
                    q.submit([encoder.finish()]);
                    let pixels = readback.rgba(&mapped(d, &readback.buffer)?)?;
                    ensure!(
                        pixels
                            == primordial_soup::export::crop_rgba_at(&clean, size, origin, crop)?,
                        "GPU crop differs from CPU oracle at {origin:?} size {crop:?}"
                    );
                }
                let mut encoder = d.create_command_encoder(&Default::default());
                ensure!(
                    primordial_soup::gpu::readback_rgba(
                        d,
                        &mut encoder,
                        r.clean_texture()?,
                        [size[0], 0],
                        [1, 1]
                    )
                    .is_err(),
                    "Out-of-bounds readback was accepted"
                );
                ensure!(
                    primordial_soup::gpu::readback_rgba(
                        d,
                        &mut encoder,
                        r.clean_texture()?,
                        [0, 0],
                        [0, 1]
                    )
                    .is_err(),
                    "Empty readback was accepted"
                );
                r.release_clean();
                ensure!(
                    r.clean_output_bytes() == 0 && r.clean_texture().is_err(),
                    "Finished export retained its texture"
                );
            }
            // Resize must also invalidate a target left alive by an export session.
            let mut encoder = d.create_command_encoder(&Default::default());
            r.present_clean(d, q, &mut encoder, &e, &camera, &prefs);
            q.submit([encoder.finish()]);
        }
        r.release_clean();
        Ok(())
    }
    fn percentiles(mut v: Vec<f64>) -> serde_json::Value {
        if v.is_empty() {
            return serde_json::Value::Null;
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        json!({"min":v[0],"p50":v[(n-1)/2],"p95":v[((n-1) as f64*0.95).ceil() as usize],"max":v[n-1],"mean":v.iter().sum::<f64>()/n as f64})
    }
    #[derive(Clone, Copy, serde::Serialize)]
    struct CaptureSample {
        transfer_wall_ms: f64,
        pixels_cpu_ms: f64,
        png_cpu_ms: f64,
        total_wall_ms: f64,
    }
    fn timed_capture(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        texture: &wgpu::Texture,
        window: [u32; 2],
        origin: [u32; 2],
        crop: [u32; 2],
        legacy: bool,
    ) -> Result<(Vec<u8>, CaptureSample)> {
        let started = Instant::now();
        let mut encoder = d.create_command_encoder(&Default::default());
        let readback = primordial_soup::gpu::readback_rgba(
            d,
            &mut encoder,
            texture,
            if legacy { [0, 0] } else { origin },
            if legacy { window } else { crop },
        )?;
        q.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        readback
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let mapped_ready = (|| -> Result<()> {
            wait(d)?;
            rx.recv_timeout(Duration::from_secs(1))??;
            Ok(())
        })();
        if let Err(error) = mapped_ready {
            readback.buffer.unmap();
            return Err(error);
        }
        let transfer_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let pixels_start = Instant::now();
        // Read the production mapped view directly: mapped() would add an artificial padded Vec copy.
        let pixels = (|| -> Result<Vec<u8>> {
            let view = readback.buffer.slice(..).get_mapped_range()?;
            let pixels = readback.rgba(&view);
            drop(view);
            pixels
        })();
        readback.buffer.unmap();
        let pixels = pixels?;
        // Legacy retained both full-frame and crop buffers through PNG encoding.
        let cropped = if legacy {
            Some(primordial_soup::export::crop_rgba_at(
                &pixels, window, origin, crop,
            )?)
        } else {
            None
        };
        let output = cropped.as_deref().unwrap_or(&pixels);
        let pixels_cpu_ms = pixels_start.elapsed().as_secs_f64() * 1000.0;
        let png_start = Instant::now();
        let png = primordial_soup::export::encode_png(output, crop)?;
        let sample = CaptureSample {
            transfer_wall_ms,
            pixels_cpu_ms,
            png_cpu_ms: png_start.elapsed().as_secs_f64() * 1000.0,
            total_wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((png, sample))
    }
    fn capture_stats(samples: &[CaptureSample]) -> serde_json::Value {
        let summary = |get: fn(&CaptureSample) -> f64| {
            let mut values: Vec<_> = samples.iter().map(get).collect();
            values.sort_by(f64::total_cmp);
            let middle = values.len() / 2;
            json!({"mean":values.iter().sum::<f64>()/values.len() as f64,"median":(values[middle-1]+values[middle])/2.0,"min":values[0],"max":values[values.len()-1]})
        };
        json!({"transfer_wall_ms":summary(|s|s.transfer_wall_ms),"pixels_cpu_ms":summary(|s|s.pixels_cpu_ms),"png_cpu_ms":summary(|s|s.png_cpu_ms),"total_wall_ms":summary(|s|s.total_wall_ms)})
    }
    fn capture_benchmark(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let p = primordial_soup::config::presets()
            .into_iter()
            .find(|(name, _)| name == "Tangle")
            .context("Tangle preset missing")?
            .1;
        let sizing = sizing_for(0.5, 1.0)?;
        let mut e = Engine::new(d, q, sizing, &p, workgroup)?;
        step(d, q, &mut e, &p, 64)?;
        let window = [2537, 1218];
        let mut renderer = Renderer::new(d, &e, window)?;
        let camera = Camera::default();
        let prefs = Preferences::default();
        renderer.begin_frame(q, &e, &camera, &p, 1, -1.0);
        let mut encoder = d.create_command_encoder(&Default::default());
        renderer.clear_accumulator(&mut encoder);
        renderer.sample(&mut encoder, &e, &camera);
        renderer.present(q, &mut encoder, &e, &camera, &prefs, Overlays::default());
        renderer.present_clean(d, q, &mut encoder, &e, &camera, &prefs);
        q.submit([encoder.finish()]);
        wait(d)?;
        let texture = renderer.clean_texture()?;
        let mut cases = Vec::new();
        for (name, origin, crop) in [
            ("offset_crop", [855, 278], [631, 577]),
            ("full_frame", [0, 0], window),
        ] {
            for _ in 0..3 {
                let before = timed_capture(d, q, texture, window, origin, crop, true)?.0;
                let after = timed_capture(d, q, texture, window, origin, crop, false)?.0;
                ensure!(before == after, "Capture benchmark warmup pixel mismatch");
            }
            let mut before = Vec::new();
            let mut after = Vec::new();
            let mut pairs = Vec::new();
            let mut png_bytes = 0;
            for pair in 0..12 {
                let (legacy, current) = if pair % 2 == 0 {
                    let legacy = timed_capture(d, q, texture, window, origin, crop, true)?;
                    (
                        legacy,
                        timed_capture(d, q, texture, window, origin, crop, false)?,
                    )
                } else {
                    let current = timed_capture(d, q, texture, window, origin, crop, false)?;
                    (
                        timed_capture(d, q, texture, window, origin, crop, true)?,
                        current,
                    )
                };
                ensure!(
                    legacy.0 == current.0,
                    "Capture benchmark pixel mismatch at pair {pair}"
                );
                png_bytes = current.0.len();
                before.push(legacy.1);
                after.push(current.1);
                pairs.push(json!({"pair":pair,"first":if pair%2==0{"legacy"}else{"cropped"},"legacy":legacy.1,"cropped":current.1}));
            }
            let stride = |width: u32| {
                (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) as u64
            };
            let full_pixels = window[0] as u64 * window[1] as u64 * 4;
            let crop_pixels = crop[0] as u64 * crop[1] as u64 * 4;
            cases.push(json!({"case":name,"origin":origin,"crop":crop,"warmup_pairs":3,"measured_pairs":12,"png_bytes":png_bytes,"byte_identical":true,"legacy":capture_stats(&before),"cropped":capture_stats(&after),"legacy_staging_bytes":stride(window[0])*window[1] as u64,"cropped_staging_bytes":stride(crop[0])*crop[1] as u64,"legacy_cpu_pixel_buffers_bytes":full_pixels+crop_pixels,"cropped_cpu_pixel_buffers_bytes":crop_pixels,"pairs":pairs}));
        }
        Ok(
            json!({"window":window,"preset":"Tangle","world":0.5,"particles":sizing.entity_count,"frozen_after_physics_steps":64,"workgroup":workgroup,"rendered_once":true,"cases":cases,"measurement_note":"Native production-equivalent readback and PNG paths, alternating pairs on one unchanged GPU texture. Transfer wall time includes allocation, encoding, submission and map wait. Pixel and PNG times are CPU wall times on one low-priority CPU. Buffer counts are logical allocations, not RSS or driver heap usage."}),
        )
    }
    fn benchmark_project(args: &[String]) -> Result<Project> {
        use primordial_soup::network::Mode;
        let option = |key: &str| -> Result<Option<&str>> {
            args.iter()
                .position(|v| v == key)
                .map(|i| {
                    args.get(i + 1)
                        .map(String::as_str)
                        .with_context(|| format!("{key} needs a value"))
                })
                .transpose()
        };
        let file = option("--project-file")?;
        let preset = option("--preset")?;
        let mode = option("--network-mode")?;
        let revision = option("--network-revision")?;
        let seed = option("--network-seed")?;
        let cohorts = option("--cohorts")?;
        ensure!(
            file.is_none()
                || [preset, mode, revision, seed, cohorts]
                    .iter()
                    .all(Option::is_none),
            "--project-file cannot be combined with preset or network overrides"
        );
        if let Some(path) = file {
            return Project::from_json(
                &std::fs::read_to_string(path)
                    .with_context(|| format!("Read benchmark project {path}"))?,
            );
        }
        let mut project = if let Some(name) = preset {
            primordial_soup::config::presets()
                .into_iter()
                .find(|(preset, _)| *preset == name)
                .with_context(|| format!("Unknown preset {name}"))?
                .1
        } else {
            Project::default()
        };
        let seed = seed.map(str::parse::<f64>).transpose()?;
        ensure!(
            seed.is_none_or(|v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "Network seed must be finite and in 0..=1"
        );
        let config = project.config_mut();
        if let Some(mode) = mode {
            let mode = match mode {
                "fourier" => Mode::Fourier,
                "mlp" => Mode::Mlp,
                "gru" => Mode::Gru,
                _ => anyhow::bail!("Network mode must be fourier, mlp or gru"),
            };
            config.switch_mode(mode, seed.unwrap_or(config.mutation_seed));
        }
        if let Some(revision) = revision {
            let revision: u32 = revision.parse()?;
            ensure!(
                config.mode() != Mode::Fourier && (1..=3).contains(&revision),
                "Network revision must be 1..=3 for MLP or GRU"
            );
            let mode = config.mode();
            config
                .networks
                .as_mut()
                .context("Network revision needs an active bank")?
                .banks
                .iter_mut()
                .find(|bank| bank.mode == mode)
                .context("Missing active network bank")?
                .revision = revision;
        }
        if let Some(seed) = seed {
            config.mutation_seed = seed;
        }
        if let Some(cohorts) = cohorts {
            let cohorts: u32 = cohorts.parse()?;
            ensure!(
                (1..=64).contains(&cohorts),
                "Benchmark cohorts must be 1..=64"
            );
            config.cohorts = cohorts;
        }
        config.networks = config.normalized_networks();
        project.validate()?;
        Ok(project)
    }

    fn bytes_hash(bytes: &[u8]) -> String {
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        format!("{hash:016x}")
    }

    fn canonical_splat(native: &[u8], pixels: usize, stride: usize) -> Result<Vec<u8>> {
        ensure!((2..=4).contains(&stride), "Invalid splat stride {stride}");
        ensure!(
            native.len() >= pixels * stride * 4,
            "Splat allocation shorter than active layout"
        );
        let mut canonical = Vec::with_capacity(pixels * 12);
        for pixel in native[..pixels * stride * 4].chunks_exact(stride * 4) {
            canonical.extend_from_slice(&pixel[..8]);
            let low = if stride >= 3 {
                u32::from_le_bytes(pixel[8..12].try_into().unwrap())
            } else {
                0
            };
            let occupancy = if stride == 4 {
                let high = u32::from_le_bytes(pixel[12..16].try_into().unwrap());
                if high != 0 {
                    7_680_000
                } else {
                    low.min(7_680_000)
                }
            } else {
                low
            };
            canonical.extend_from_slice(&occupancy.to_le_bytes());
        }
        Ok(canonical)
    }

    #[allow(clippy::too_many_arguments)]
    fn benchmark_state(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        engine: &Engine,
        renderer: &Renderer,
        project: &Project,
        path: &str,
        rendered: bool,
        entities: &[u8],
    ) -> Result<serde_json::Value> {
        let directory = Path::new(path);
        std::fs::create_dir_all(directory)?;
        let mut files = serde_json::Map::new();
        let mut save = |name: &str, bytes: &[u8]| -> Result<()> {
            std::fs::write(directory.join(name), bytes)?;
            files.insert(
                name.into(),
                json!({"bytes":bytes.len(),"fnv1a64":bytes_hash(bytes)}),
            );
            Ok(())
        };
        save("entities.bin", entities)?;
        save(
            "configs.bin",
            &read(d, q, &engine.configs, engine.configs.size())?,
        )?;
        let native_memory = read(d, q, &engine.brain_memory, engine.brain_memory.size())?;
        let brain_memory_representation = if project.config().mode()
            == primordial_soup::network::Mode::Gru
        {
            let particles = engine.sizing.entity_count as usize;
            let streams = if project.config().network_revision() >= 3 {
                3
            } else {
                2
            };
            let canonical = primordial_soup::gpu::audit::interleave_gru_history(
                &native_memory,
                particles,
                streams,
            )?;
            std::fs::write(directory.join("brain-memory-native.bin"), &native_memory)?;
            save("brain-memory.bin", &canonical)?;
            Some(
                json!({"layout":"gru-stream-planar-vec4","particles":particles,"streams":streams,
                "native_file":"brain-memory-native.bin","native_bytes":native_memory.len(),
                "native_fnv1a64":bytes_hash(&native_memory),"canonical_file":"brain-memory.bin",
                "canonical_encoding":"particle-major vec4 streams: base, mirror, revision3 heading; scalar bytes unchanged"}),
            )
        } else {
            save("brain-memory.bin", &native_memory)?;
            None
        };
        let layout = engine.splat_layout();
        let native_splat = read(d, q, engine.splat_buffer(), layout.allocated_bytes)?;
        let canonical = canonical_splat(
            &native_splat,
            engine.sizing.canvas[0] as usize * engine.sizing.canvas[1] as usize,
            layout.active_lanes as usize,
        )?;
        save("splat.bin", &canonical)?;
        std::fs::write(directory.join("splat-native.bin"), &native_splat)?;
        let rules = engine.snapshot_rules(d, q);
        save("rules.bin", &mapped(d, &rules)?)?;
        let mut canvas_representation = Vec::new();
        for (index, canvas) in engine.canvas.iter().enumerate() {
            let rgba = if canvas.format() == wgpu::TextureFormat::Rg16Float {
                let native = tex(d, q, canvas, engine.sizing.canvas, 4)?;
                let native_file = format!("canvas-{index}.rg16f");
                std::fs::write(directory.join(&native_file), &native)?;
                canvas_representation.push(json!({"format":"rg16float","native_file":native_file,
                    "native_bytes":native.len(),"native_fnv1a64":bytes_hash(&native),
                    "canonical_file":format!("canvas-{index}.rgba16f"),
                    "absent_storage_channels":["b","a"],"sampled_missing_channels":[0,1],
                    "canonical_missing_channels":[0,0]}));
                native
                    .chunks_exact(4)
                    .flat_map(|rg| [rg[0], rg[1], rg[2], rg[3], 0, 0, 0, 0])
                    .collect()
            } else {
                tex(d, q, canvas, engine.sizing.canvas, 8)?
            };
            save(&format!("canvas-{index}.rgba16f"), &rgba)?;
        }
        save(
            "field.rgba16f",
            &tex(d, q, engine.audit_field_texture(), engine.field_size, 8)?,
        )?;
        if rendered {
            save(
                "output.rgba8",
                &tex(d, q, renderer.output_texture(), renderer.size, 4)?,
            )?;
        }
        let project_json = project.to_json()?;
        std::fs::write(directory.join("project.json"), &project_json)?;
        let metadata = json!({
            "format":"primordial-benchmark-state-v1","byte_order":"little_endian",
            "canvas_representation":canvas_representation,
            "brain_memory_representation":brain_memory_representation,
            "hash":"fnv1a64 (comparison aid; compare raw bytes for exactness)",
            "project":project,"project_hash":primordial_soup::archive::hash_state(project),
            "project_json_fnv1a64":bytes_hash(project_json.as_bytes()),
            "completed_tick_count":engine.frame_count,"current_canvas":engine.current_canvas,
            "particles":engine.sizing.entity_count,"canvas":engine.sizing.canvas,
            "field":engine.field_size,"painted_field_sampling_active":engine.field_sampling_active(),
            "rendered":rendered,"output":rendered.then_some(renderer.size),
            "brain_memory_note":"entire buffer: both mirrored hidden streams and revision-3 heading state",
            "captured_after_timing":true,"files":files,
            "splat_representation": {
                "layout":layout,"native_file":"splat-native.bin",
                "native_bytes":native_splat.len(),"native_fnv1a64":bytes_hash(&native_splat),
                "canonical_file":"splat.bin",
                "canonical_encoding":"three i32 lanes: unchanged momentum x/y, occupancy saturated to 7680000; wide low/high unsigned bits"
            }
        });
        std::fs::write(
            directory.join("state.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;
        Ok(metadata)
    }

    #[cfg(test)]
    mod benchmark_project_tests {
        use super::*;

        fn project(flags: &[&str]) -> Result<Project> {
            benchmark_project(&flags.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>())
        }

        #[test]
        fn canonical_splat_preserves_momentum_and_saturates_unsigned_limbs() {
            let values = [
                0x8000_0001u32,
                0xffff_ffff,
                31,
                0,
                7,
                8,
                0x8000_0000,
                0,
                9,
                10,
                1,
                2,
            ];
            let native: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let actual = canonical_splat(&native, 3, 4).unwrap();
            let expected: Vec<u8> = [
                0x8000_0001u32,
                0xffff_ffff,
                31,
                7,
                8,
                7_680_000,
                9,
                10,
                7_680_000,
            ]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
            assert_eq!(actual, expected);
            assert_eq!(canonical_splat(&expected, 3, 3).unwrap(), expected);
            assert!(canonical_splat(&native[..8], 1, 4).is_err());
            let narrow: Vec<u8> = [0x8000_0001u32, 0xffff_ffff, 7, 8]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let expected: Vec<u8> = [0x8000_0001u32, 0xffff_ffff, 0, 7, 8, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(canonical_splat(&narrow, 2, 2).unwrap(), expected);
        }

        #[test]
        fn default_project_is_unchanged_and_explicit_network_banks_are_reproducible() {
            assert_eq!(project(&[]).unwrap(), Project::default());
            for mode in ["mlp", "gru"] {
                for revision in ["1", "2", "3"] {
                    let flags = [
                        "--network-mode",
                        mode,
                        "--network-revision",
                        revision,
                        "--network-seed",
                        "0.25",
                        "--cohorts",
                        "16",
                    ];
                    let first = project(&flags).unwrap();
                    assert_eq!(first, project(&flags).unwrap());
                    assert_eq!(first.config().mode().label().to_lowercase(), mode);
                    assert_eq!(first.config().network_revision().to_string(), revision);
                    assert_eq!(first.config().bank().unwrap().seed, 0.25);
                    assert_eq!(first.config().cohorts, 16);
                    assert_eq!(
                        first,
                        Project::from_json(&first.to_json().unwrap()).unwrap()
                    );
                }
            }
        }

        #[test]
        fn invalid_network_overrides_and_mixed_project_sources_are_rejected() {
            for flags in [
                vec!["--network-seed", "NaN"],
                vec!["--network-seed", "1.1"],
                vec!["--network-mode", "gru", "--network-revision", "4"],
                vec!["--network-revision", "3"],
                vec!["--cohorts", "0"],
                vec!["--network-mode"],
                vec!["--project-file", "unused.json", "--network-mode", "gru"],
                vec!["--project-file", "unused.json", "--preset", "Starcrossed"],
            ] {
                assert!(project(&flags).is_err(), "Accepted {flags:?}");
            }
        }

        #[test]
        fn project_file_preserves_the_serialized_project() {
            let mut expected =
                project(&["--network-mode", "gru", "--network-seed", "0.125"]).unwrap();
            expected.notes = "Exact imported benchmark fixture".into();
            let path = std::env::temp_dir().join(format!(
                "primordial-benchmark-project-{}.json",
                std::process::id()
            ));
            std::fs::write(&path, expected.to_json().unwrap()).unwrap();
            let imported = project(&["--project-file", path.to_str().unwrap()]);
            std::fs::remove_file(path).unwrap();
            assert_eq!(imported.unwrap(), expected);
        }
    }

    fn benchmark(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        world: f64,
        steps: u32,
        frames: u32,
        workgroup: u32,
        shot: Option<&str>,
        eager_export: bool,
    ) -> Result<serde_json::Value> {
        let args: Vec<String> = std::env::args().collect();
        let value = |key: &str, default: &str| -> Result<String> {
            match args.iter().position(|v| v == key) {
                Some(i) => Ok(args
                    .get(i + 1)
                    .with_context(|| format!("{key} needs a value"))?
                    .clone()),
                None => Ok(default.to_owned()),
            }
        };
        ensure!(
            (1..=3600).contains(&frames),
            "Benchmark frames must be 1..3600"
        );
        ensure!((1..=128).contains(&steps), "Benchmark steps must be 1..128");
        let render_only = args.iter().any(|v| v == "--render-only");
        let physics_only = args.iter().any(|v| v == "--physics-only");
        ensure!(
            !(render_only && physics_only),
            "Choose render-only or physics-only"
        );
        ensure!(
            !physics_only || shot.is_none(),
            "Physics-only has no rendered screenshot"
        );
        let style = value("--style", "classic")?;
        ensure!(
            ["classic", "cells"].contains(&style.as_str()),
            "Style must be classic or cells"
        );
        let cellular = style == "cells";
        let warmup: u32 = value("--warmup", "3")?.parse()?;
        let setup_ticks: u32 =
            value("--setup-ticks", if cellular { "120" } else { "0" })?.parse()?;
        ensure!(
            warmup <= 3600 && setup_ticks <= 100_000,
            "Excessive benchmark warmup/setup"
        );
        ensure!(
            !render_only || setup_ticks > 0,
            "Frozen rendering needs at least one setup tick to initialize particles"
        );
        let size = [
            value("--width", "1280")?.parse::<u32>()?,
            value("--height", "720")?.parse::<u32>()?,
        ];
        ensure!(
            size.iter()
                .all(|v| *v > 0 && *v <= d.limits().max_texture_dimension_2d),
            "Invalid benchmark output dimensions"
        );
        let zoom: f32 = value("--zoom", "1")?.parse()?;
        ensure!(
            zoom.is_finite() && zoom > 0.0,
            "Zoom must be finite and positive"
        );
        let label = args
            .iter()
            .position(|v| v == "--label")
            .map(|_| value("--label", ""))
            .transpose()?;
        let s = sizing_for(world, 1.0)?;
        let p = benchmark_project(&args)?;
        let mut e = Engine::new(d, q, s, &p, workgroup)?;
        if args.iter().any(|v| v == "--active-empty-field") {
            let _ = e.field_texture(q);
        }
        // Setup is outside every measured interval, and identical for all workload variants.
        for chunk in (0..setup_ticks).step_by(128) {
            step(d, q, &mut e, &p, (setup_ticks - chunk).min(128))?;
        }
        let mut camera = Camera {
            zoom,
            ..Default::default()
        };
        let mut anchor_count = 0;
        if zoom > 1.0 && setup_ticks > 0 {
            // Deterministic densest 64x64 world bin; ties select the lowest bin.
            let entities = read(d, q, &e.entities, s.entity_count as u64 * 32)?;
            let mut bins = vec![(0u32, [0.0f64; 2]); 64 * 64];
            for entity in entities.chunks_exact(32) {
                if i32::from_le_bytes(entity[20..24].try_into().unwrap()) < 0 {
                    continue;
                }
                let position = floats(&entity[..8]);
                if !position.iter().all(|v| v.is_finite()) {
                    continue;
                }
                let cell = position
                    .iter()
                    .map(|v| (((v + 1.0) * 32.0).floor() as i32).clamp(0, 63) as usize)
                    .collect::<Vec<_>>();
                let bin = &mut bins[cell[1] * 64 + cell[0]];
                bin.0 += 1;
                for axis in 0..2 {
                    bin.1[axis] += f64::from(position[axis]);
                }
            }
            let mut best = &bins[0];
            for bin in &bins[1..] {
                if bin.0 > best.0 {
                    best = bin;
                }
            }
            ensure!(best.0 > 0, "No live particles for close-view anchor");
            camera.pan = best.1.map(|sum| (sum / f64::from(best.0)) as f32);
            anchor_count = best.0;
        }
        let mut r = if eager_export {
            Renderer::new_eager_for_benchmark(d, &e, size)?
        } else {
            Renderer::new(d, &e, size)?
        };
        let idle_export_bytes = r.clean_output_bytes();
        ensure!(
            idle_export_bytes
                == if eager_export {
                    u64::from(size[0]) * u64::from(size[1]) * 4
                } else {
                    0
                },
            "Unexpected idle export allocation in policy control"
        );
        let mut prefs = Preferences::default();
        prefs.cells.fast = !args.iter().any(|v| v == "--reference-cells");
        ensure!(
            !(args.iter().any(|v| v == "--reference-cells") && args.iter().any(|v| v == "--tiled")),
            "Reference Cells and tiled are mutually exclusive"
        );
        prefs.world_size = world;
        prefs.physics_steps = steps;
        prefs.render_scale = primordial_soup::config::RenderScale::Full;
        prefs.cells.style = if cellular {
            primordial_soup::cells::Style::Cells
        } else {
            primordial_soup::cells::Style::Classic
        };
        prefs.motion_blur_samples = value("--blur-samples", &value("--blur", "4")?)?.parse()?;
        ensure!(
            (1..=16).contains(&prefs.motion_blur_samples),
            "Benchmark blur samples must be 1..16"
        );
        if cellular && !physics_only {
            let mut construction = prefs.cells.clone();
            if args.iter().any(|v| v == "--tiled") {
                construction.fast = false;
            }
            r.commit_cells(r.prepare_cells(d, &e, &construction)?);
        }
        configure_tiled(d, &e, &mut r)?;
        r.set_preferences(&prefs);
        let schedule = blur_schedule(steps, prefs.motion_blur_samples);
        let profile_requested = !args.iter().any(|v| v == "--no-profile");
        let total_only = args.iter().any(|v| v == "--total-only");
        ensure!(
            !total_only || profile_requested,
            "Total-only and no-profile are mutually exclusive"
        );
        let mut timing = if profile_requested && !total_only {
            primordial_soup::profiling::GpuProfiler::new(d)
        } else {
            None
        };
        let mut total_timer = if total_only {
            primordial_soup::profiling::FrameTimer::new(d, q)
        } else {
            None
        };
        let instrumented = timing.is_some() || total_timer.is_some();
        let mut preparation = Vec::new();
        let mut stages = Vec::new();
        let mut details = Vec::new();
        let mut submits = Vec::new();
        let mut waits = Vec::new();
        let (mut gpu, mut cpu, mut complete) = (Vec::new(), Vec::new(), Vec::new());
        let mut measured_start_tick = e.frame_count;
        for frame in 0..frames + warmup {
            if frame == warmup {
                measured_start_tick = e.frame_count;
            }
            let started = Instant::now();
            if !render_only {
                e.prepare_frame(q, &p.world, steps, None, [0.01, 0.001])?;
            }
            if !physics_only {
                r.begin_frame(q, &e, &camera, &p, schedule.samples, -1.0);
            }
            let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
            let encoding_started = Instant::now();
            let mut c = d.create_command_encoder(&Default::default());
            let mut profile = timing.as_mut().and_then(|t| t.begin_frame(&mut c));
            let mut total = total_timer.as_mut().and_then(|t| t.begin(&mut c));
            if !physics_only {
                r.clear_accumulator(&mut c);
            }
            if render_only {
                for _ in 0..schedule.samples {
                    r.sample_profiled(&mut c, &e, &camera, profile.as_mut());
                }
            } else {
                for i in 0..steps {
                    e.encode_step_profiled(&mut c, i, profile.as_mut());
                    if !physics_only && i % schedule.stride == schedule.sample_at {
                        r.sample_profiled(&mut c, &e, &camera, profile.as_mut());
                    }
                }
            }
            if !physics_only {
                r.present_profiled(
                    q,
                    &mut c,
                    &e,
                    &camera,
                    &prefs,
                    Overlays::default(),
                    profile.as_mut(),
                );
            }
            if let Some(t) = &mut profile {
                t.finish(&mut c);
            }
            if let Some(t) = &mut total {
                t.finish(&mut c);
            }
            let commands = c.finish();
            let encode = encoding_started.elapsed().as_secs_f64() * 1000.0;
            let submitted = Instant::now();
            q.submit([commands]);
            let submit_ms = submitted.elapsed().as_secs_f64() * 1000.0;
            if let (Some(t), Some(frame)) = (&mut timing, profile) {
                t.after_submit(frame);
            }
            if let (Some(t), Some(frame)) = (&mut total_timer, total) {
                t.after_submit(frame);
            }
            let waiting = Instant::now();
            wait(d)?;
            let wait_ms = waiting.elapsed().as_secs_f64() * 1000.0;
            let completed = started.elapsed().as_secs_f64() * 1000.0;
            let elapsed = timing.as_mut().and_then(|t| t.poll(d, q).pop());
            let total_elapsed = total_timer.as_mut().and_then(|t| t.poll().pop());
            ensure!(
                !instrumented || elapsed.is_some() || total_elapsed.is_some(),
                "Missing GPU timing for benchmark frame {frame}"
            );
            if frame >= warmup {
                preparation.push(prepare_ms);
                cpu.push(encode);
                complete.push(completed);
                submits.push(submit_ms);
                waits.push(wait_ms);
                if let Some(sample) = elapsed {
                    gpu.push(sample.total_ms);
                    stages.push(sample.stages_ms);
                    details.push(sample.details_ms);
                }
                if let Some(ms) = total_elapsed {
                    gpu.push(ms);
                }
            }
        }
        ensure!(
            !instrumented || gpu.len() == frames as usize,
            "Incomplete GPU timing samples"
        );
        let measured_ticks = e.frame_count.wrapping_sub(measured_start_tick);
        ensure!(
            measured_ticks == if render_only { 0 } else { frames * steps },
            "Unexpected measured physics tick count"
        );
        let entities = read(d, q, &e.entities, u64::from(s.entity_count) * 32)?;
        finite_particles(&entities)?;
        let live_particles = entities
            .chunks_exact(32)
            .filter(|entity| i32::from_le_bytes(entity[20..24].try_into().unwrap()) >= 0)
            .count();
        let state = args
            .iter()
            .position(|v| v == "--benchmark-state")
            .map(|_| {
                let path = value("--benchmark-state", "")?;
                benchmark_state(d, q, &e, &r, &p, &path, !physics_only, &entities)
            })
            .transpose()?;
        if let Some(path) = shot {
            save_png(d, q, r.output_texture(), size, path)?;
        }
        // Inspect the final temporal sample after all frame intervals have completed.
        let sprite_plan = if !physics_only {
            if let Some(cells) = r.cells.as_ref() {
                let bytes = read(d, q, cells.sprite_plan_buffer(), 16)?;
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                    .collect();
                Some(
                    json!({"overlap_bound":words[0],"visible_copies":words[1],"mode":words[2],"compact":words[2]==0,"maximum_encoded_batches":cells.max_sprite_batches(),"sample":"final temporal sample after timing"}),
                )
            } else {
                None
            }
        } else {
            None
        };
        let mean = if gpu.is_empty() {
            None
        } else {
            Some(gpu.iter().sum::<f64>() / gpu.len() as f64)
        };
        let simulation_memory = e.allocated_bytes();
        let entity_per_tick = if render_only {
            Vec::new()
        } else {
            stages
                .iter()
                .map(|sample| sample[0] / f64::from(steps))
                .collect()
        };
        let raw_samples = json!({"gpu_ms":gpu,"cpu_encode_ms":cpu,"cpu_prepare_ms":preparation,"cpu_submit_ms":submits,"completed_wait_ms":waits,"completed_frame_ms":complete,"gpu_stage_ms":stages,"gpu_detail_ms":details});
        let mut report = json!({
            "label":label,"preset":p.name,"project":p,"preferences":prefs,"style":style,
            "project_hash":primordial_soup::archive::hash_state(&p),
            "project_json_fnv1a64":bytes_hash(p.to_json()?.as_bytes()),
            "network_mode":p.config().mode(),"network_revision":p.config().network_revision(),
            "network_seed":p.config().mutation_seed,"cohorts":p.config().cohorts,
            "network_configurations":p.configs.iter().map(|config| json!({"mode":config.mode(),"revision":config.network_revision(),"seed":config.mutation_seed,"cohorts":config.cohorts,"parameters":config.network_parameters()})).collect::<Vec<_>>(),
            "actual_live_particles":live_particles,"actual_dead_particles":s.entity_count as usize-live_particles,
            "benchmark_state":state,"entity_update_ms_per_tick":percentiles(entity_per_tick),
            "entity_update_timing_note":"Detailed-profile entity stage divided by simulation ticks; null for render-only or total-only runs. Do not compare detailed-profile totals against total-only acceptance timing.",
            "workload":if render_only {"render-only"} else if physics_only {"physics-only"} else {"combined"},
            "camera":{"zoom":camera.zoom,"pan":camera.pan,"trail_mode":camera.trail_mode,"anchor_method":if zoom>1.0 && setup_ticks>0 {"densest_64x64_world_bin_mean"}else{"world_center"},"anchor_bin_population":anchor_count,"fixed_during_run":true},
            "setup_ticks":setup_ticks,"measured_start_tick":measured_start_tick,"completed_tick_count":e.frame_count,"measured_ticks":measured_ticks,
            "painted_field_sampling_active":e.field_sampling_active(),
            "initialization_note":"Engine tick zero initializes particle state; logical completed ticks include that initialization tick. Setup and warmup are excluded from measured_ticks.",
            "requested_motion_blur_samples":prefs.motion_blur_samples,"timestamp_query_supported":d.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "world":world,"particles":s.entity_count,"canvas":s.canvas,"output":size,"steps":steps,"frames":frames,"warmup":warmup,"workgroup":workgroup,
            "motion_blur_samples":if physics_only {0}else{schedule.samples},"bloom":prefs.bloom_enabled
        });
        let detail_percentiles: Vec<_> = (0..primordial_soup::profiling::Stage::DETAIL_NAMES.len())
            .map(|index| percentiles(details.iter().map(|sample| sample[index]).collect()))
            .collect();
        let tiled = tiled_state(d, q, &r)?;
        let cells_sizes = r
            .cells
            .as_ref()
            .map(|cells| json!({"working":cells.size,"padded":cells.padded,"framebuffer":size}));
        let measurements = json!({
            "gpu_ms":percentiles(gpu),"cpu_encode_ms":percentiles(cpu),"completed_frame_ms":percentiles(complete),
            "particle_updates_per_second":if render_only {None}else{mean.map(|ms|s.entity_count as f64*steps as f64*1000.0/ms)},
            "fast_cells_requested":prefs.cells.fast,"analytic_raster_requested":args.iter().any(|v|v=="--analytic-raster"),"distance_mode":if r.cells.as_ref().is_some_and(|cells|cells.tiled_state_buffer().is_some()) || args.iter().any(|v|v=="--analytic-raster") {"analytic_pixel"}else{"raster_interpolated"},"tiled_requested":args.iter().any(|v| v=="--tiled"),"tiled_state":tiled,"cells_sizes":cells_sizes,"sprite_plan":sprite_plan,"simulation_allocated_bytes":simulation_memory,"renderer_texture_bytes":r.texture_bytes(),"memory_note":"Logical allocations; driver heap overhead additional",
            "profile_requested":profile_requested,"profiled":timing.is_some(),"total_only":total_only,"instrumented":instrumented,
            "timestamp_method":if total_timer.is_some(){Some("FrameTimer: two portable empty compute pass boundaries")}else{timing.as_ref().map(|_|"GpuProfiler: total and detailed stage boundaries")},
            "timestamp_overhead_note":"Compare identical instrumentation across runs; total-only is the acceptance timer. CPU completion includes synchronous readback waiting, with no presentation.",
            "gpu_stage_order":primordial_soup::profiling::Stage::NAMES,"gpu_detail_order":primordial_soup::profiling::Stage::DETAIL_NAMES,"gpu_detail_ms":detail_percentiles,"automatic_calibration":false,"cpu_prepare_ms":percentiles(preparation),"cpu_submit_ms":percentiles(submits),"completed_wait_ms":percentiles(waits),
            "screenshot":shot,"eager_export_target":eager_export,"idle_export_bytes":idle_export_bytes,"raw_samples":raw_samples
        });
        report
            .as_object_mut()
            .unwrap()
            .extend(measurements.as_object().unwrap().clone());
        Ok(report)
    }

    fn allocation_control(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let mut blocks = Vec::new();
        for (block, eager) in [true, false, false, true].into_iter().enumerate() {
            wait(d)?;
            let unix_ms = || {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("Clock precedes Unix epoch")
                    .as_millis()
            };
            let start = unix_ms();
            let result = benchmark(d, q, 1.0, 30, 12, workgroup, None, eager)?;
            // benchmark's engine/renderer are dropped before constructing the next block.
            wait(d)?;
            blocks.push(json!({"block":block,"policy":if eager{"eager"}else{"lazy"},"started_unix_ms":start,"ended_unix_ms":unix_ms(),"result":result}));
        }
        Ok(
            json!({"order":"eager-lazy-lazy-eager","blocks":blocks,"measurement_note":"One executable/device; fresh identical simulations; eager clean texture allocated immediately after the display output before bloom textures, retained without export draws, uploads or clears. Each block uses three warmups and twelve measurements with identical frame commands. Submissions are drained and block resources dropped before the next block. This isolates the allocation policy, not every difference from a historical executable."}),
        )
    }
    fn video_smoke(d: &wgpu::Device, q: &wgpu::Queue, workgroup: u32) -> Result<serde_json::Value> {
        use primordial_soup::record::{ExportSession, Recorder, RecordingSettings};
        let sizing = Sizing {
            entity_count: 4096,
            canvas: [64, 64],
            sqrt_world_size: 1.0,
        };
        let project = primordial_soup::config::presets()
            .into_iter()
            .find(|(name, _)| name == "Tangle")
            .context("Tangle missing")?
            .1;
        let mut engine = Engine::new(d, q, sizing, &project, workgroup)?;
        step(d, q, &mut engine, &project, 64)?;
        let size = [160, 128];
        let mut camera = Camera {
            zoom: 2.0,
            pan: [0.3, -0.2],
            ..Default::default()
        };
        let cellular = std::env::args().any(|v| v == "--cells-video");
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        if cellular {
            prefs.cells.style = primordial_soup::cells::Style::Cells;
            if std::env::args().any(|v| v == "--cells-light") {
                prefs.cells.theme = primordial_soup::cells::Theme::Light;
            }
        }
        let settings = RecordingSettings {
            duration: 1,
            resolution: size,
            physics_steps: 2,
            motion_blur_samples: if cellular { 2 } else { 1 },
            ..Default::default()
        };
        let mut session =
            ExportSession::start(d, &engine, &settings, camera, &prefs, &project, true)?;
        ensure!(
            session.camera.pan == [0.0; 2] && session.camera.zoom == 1.0 && session.was_paused,
            "World camera was not independent of preview"
        );
        let allocated = session.allocated_bytes();
        let mut preview = Renderer::new(d, &engine, [240, 180])?;
        let mut reference = Renderer::new(d, &engine, size)?;
        for r in [&mut preview, &mut reference] {
            if cellular {
                r.commit_cells(r.prepare_cells(d, &engine, &prefs.cells)?);
            }
            r.set_preferences(&prefs);
        }
        let mut last_reference = Vec::new();
        let mut preview_prefs = prefs.clone();
        let start = Instant::now();
        let mut compared = 0;
        let output = loop {
            ensure!(
                start.elapsed() < Duration::from_secs(30),
                "Video smoke exceeded30 seconds"
            );
            let progress = session.pump(d)?;
            if progress.finished {
                break progress.output.context("Video finished without output")?;
            }
            if !session.ready() {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let frame = session.frames_done();
            if frame == 10 {
                camera.pan = [-0.4, 0.25];
                camera.zoom = 3.0;
                if cellular {
                    preview_prefs.cells.theme = match prefs.cells.theme {
                        primordial_soup::cells::Theme::Dark => primordial_soup::cells::Theme::Light,
                        primordial_soup::cells::Theme::Light => primordial_soup::cells::Theme::Dark,
                    };
                    preview.set_preferences(&preview_prefs);
                }
            }
            if frame == 20 {
                preview.resize(d, &engine, [320, 128])?;
                if cellular {
                    preview.commit_cells(preview.prepare_cells(d, &engine, &prefs.cells)?);
                    preview.set_preferences(&preview_prefs);
                }
            }
            engine.prepare_frame(
                q,
                &project.world,
                settings.physics_steps,
                None,
                [0.01, 0.001],
            )?;
            let schedule = blur_schedule(settings.physics_steps, settings.motion_blur_samples);
            preview.begin_frame(q, &engine, &camera, &project, schedule.samples, 0.0);
            session.render.begin_frame(
                q,
                &engine,
                &session.camera,
                &project,
                schedule.samples,
                -1.0,
            );
            reference.begin_frame(
                q,
                &engine,
                &Camera::default(),
                &project,
                schedule.samples,
                -1.0,
            );
            let mut commands = d.create_command_encoder(&Default::default());
            preview.clear_accumulator(&mut commands);
            session.render.clear_accumulator(&mut commands);
            reference.clear_accumulator(&mut commands);
            for i in 0..settings.physics_steps {
                engine.encode_step(&mut commands, i);
                if i % schedule.stride == schedule.sample_at {
                    preview.sample(&mut commands, &engine, &camera);
                    session
                        .render
                        .sample(&mut commands, &engine, &session.camera);
                    reference.sample(&mut commands, &engine, &Camera::default());
                }
            }
            preview.present(
                q,
                &mut commands,
                &engine,
                &camera,
                &preview_prefs,
                Overlays {
                    walls: true,
                    trails: true,
                    radius: 0.2,
                    ..Default::default()
                },
            );
            session.render.present(
                q,
                &mut commands,
                &engine,
                &session.camera,
                &session.prefs,
                Overlays::default(),
            );
            reference.present(
                q,
                &mut commands,
                &engine,
                &Camera::default(),
                &prefs,
                Overlays::default(),
            );
            q.submit([commands.finish()]);
            if frame % 20 == 0 {
                ensure!(
                    tex(d, q, session.render.output_texture(), size, 4)?
                        == tex(d, q, reference.output_texture(), size, 4)?,
                    "Preview navigation changed world output"
                );
                compared += 1;
            }
            ensure!(
                session.allocated_bytes() == allocated && session.render.clean_output_bytes() == 0,
                "Export allocated redundant clean output or changed size"
            );
            if cellular && frame == 59 {
                last_reference = tex(d, q, reference.output_texture(), size, 4)?;
            }
            session
                .recorder
                .frame(d, q, session.render.output_texture(), size, false)?;
        };
        let elapsed = start.elapsed().as_secs_f64();
        ensure!(
            engine.frame_count == 184,
            "Simulation advanced more than once per recording frame"
        );
        // Exercise startup, GPU mapping and encoder delivery cancellation with
        // retained handles, then start an unrelated session immediately.
        let mut cancellation = Vec::new();
        for stage in ["startup", "mapping", "encoding"] {
            let mut cancelled = Recorder::start(d, &settings, size)?;
            if stage != "startup" {
                let deadline = Instant::now() + Duration::from_secs(10);
                while !cancelled.ready() {
                    cancelled.pump(d)?;
                    ensure!(Instant::now() < deadline, "Cancel fixture did not start");
                    std::thread::sleep(Duration::from_millis(1));
                }
                cancelled.frame(d, q, session.render.output_texture(), size, false)?;
                if stage == "encoding" {
                    wait(d)?;
                    cancelled.pump(d)?;
                }
            }
            cancelled.discard();
            ensure!(
                cancelled.finished() && !cancelled.ready(),
                "Discarded recorder revived"
            );
            let mut restart = Recorder::start(d, &settings, size)?;
            let deadline = Instant::now() + Duration::from_secs(10);
            while !restart.ready() {
                restart.pump(d)?;
                ensure!(Instant::now() < deadline, "Restart blocked");
                std::thread::sleep(Duration::from_millis(1));
            }
            ensure!(
                cancelled.pump(d)?.finished && !cancelled.ready(),
                "Late callback revived discarded recording"
            );
            restart.cancel();
            while !restart.finished() {
                restart.pump(d)?;
                ensure!(Instant::now() < deadline, "Empty finish blocked");
                std::thread::sleep(Duration::from_millis(1));
            }
            ensure!(
                restart.progress().output.as_deref() == Some(""),
                "Zero-frame recording saved an output"
            );
            cancellation.push(stage);
        }
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                &output,
            ])
            .output()?;
        ensure!(probe.status.success(), "ffprobe rejected NVENC output");
        let probe: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
        let stream = &probe["streams"][0];
        ensure!(
            stream["codec_name"] == "h264"
                && stream["width"] == size[0]
                && stream["height"] == size[1]
                && stream["nb_read_frames"] == "60"
                && stream["avg_frame_rate"] == "60/1",
            "Unexpected offline video dimensions, codec or timing"
        );
        let mut decoded_error = None;
        if cellular {
            let suffix = if prefs.cells.theme == primordial_soup::cells::Theme::Light {
                "-light"
            } else {
                ""
            };
            std::fs::copy(&output, format!("artifacts/cells-native-video{suffix}.mp4"))?;
            let decoded = std::process::Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    &output,
                    "-vf",
                    "select=eq(n\\,59)",
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "pipe:1",
                ])
                .output()?;
            ensure!(
                decoded.status.success() && decoded.stdout.len() == last_reference.len(),
                "Could not decode Cells video"
            );
            let error = decoded
                .stdout
                .iter()
                .zip(&last_reference)
                .enumerate()
                .filter(|(i, _)| i % 4 != 3)
                .map(|(_, (a, b))| a.abs_diff(*b) as f64)
                .sum::<f64>()
                / (size[0] * size[1] * 3) as f64;
            ensure!(error < 5.0, "Decoded Cells color error: {error}");
            decoded_error = Some(error);
            std::fs::write(
                format!("artifacts/cells-native-video{suffix}-last.png"),
                primordial_soup::export::encode_png(&decoded.stdout, size)?,
            )?;
        } else {
            std::fs::copy(&output, "artifacts/world-native-video.mp4")?;
        }
        Ok(
            json!({"appearance":prefs.cells.style,"theme":prefs.cells.theme,"decoded_mean_rgb_error":decoded_error,"encoder":"h264_nvenc", "output":output, "probe":probe, "frames":60, "elapsed_seconds":elapsed, "export_logical_gpu_bytes":allocated, "clean_output_bytes":session.render.clean_output_bytes(), "world_output_reference_matches":compared, "preview_changes":{"pan_zoom_and_theme_at_frame":10,"resized_at_frame":20}, "simulation_steps":184,"discard_and_restart_stages":cancellation, "bounded_readback":true}),
        )
    }
    fn save_png(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        t: &wgpu::Texture,
        size: [u32; 2],
        path: &str,
    ) -> Result<()> {
        let bytes = tex(d, q, t, size, 4)?;
        ensure!(
            bytes
                .chunks_exact(4)
                .any(|p| p[..3].iter().any(|v| *v != 0)),
            "Rendered screenshot is empty"
        );
        if let Some(p) = Path::new(path).parent() {
            std::fs::create_dir_all(p)?;
        }
        let file = std::fs::File::create(path)?;
        let mut encoder = png::Encoder::new(file, size[0], size[1]);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&bytes)?;
        Ok(())
    }
    fn network_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::network::{self, Mode};
        let mut results = Vec::new();
        for mode in [Mode::Mlp, Mode::Gru] {
            let mut p = Project::default();
            p.config_mut().switch_mode(mode, 0.25);
            let c = p.config_mut();
            c.networks
                .as_mut()
                .unwrap()
                .banks
                .iter_mut()
                .find(|b| b.mode == mode)
                .unwrap()
                .revision = 2;
            c.cohorts = 1;
            c.mutation_scale = 0.0;
            c.hazard_rate = 0.0;
            c.cohort_fences = false;
            c.sensor_angle_jitter = 0.0;
            c.sensor_distance_jitter = 0.0;
            let cases: serde_json::Value = serde_json::from_str(include_str!(
                "../../tests/fixtures/network-oracle.fixture.json"
            ))?;
            let case = cases
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["mode"] == serde_json::to_value(mode).unwrap())
                .unwrap();
            let weights: Vec<f32> = serde_json::from_value(case["weights"].clone())?;
            let probe_weights = [
                weights.clone(),
                network::derive(&c.bank().unwrap(), 1),
                network::derive(&c.bank().unwrap(), 2),
            ];
            c.set_parent(Some(weights.clone()));
            // Compare the actual GPU evaluator on unequal base/mirror streams.
            let storage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let make = |size| {
                d.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("network probe"),
                    size,
                    usage: storage,
                    mapped_at_creation: false,
                })
            };
            let input = make(128);
            let output = make(128);
            let w = make(2304);
            let memory = make(128);
            let mut padded = Vec::new();
            for weights in &probe_weights {
                let mut p = weights.clone();
                p.resize(primordial_soup::network::WEIGHT_STRIDE, 0.0);
                padded.extend(p);
            }
            q.write_buffer(&w, 0, bytemuck::cast_slice(&padded));
            let module = primordial_soup::gpu::module(d, "networkProbe.wgsl");
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("network probe"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("NETWORK_MODE", mode.id() as f64)],
                    ..Default::default()
                },
                cache: None,
            });
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: w.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: memory.as_entire_binding(),
                    },
                ],
            });
            let mut hs: [[f32; 4]; 8] =
                std::array::from_fn(|i| std::array::from_fn(|j| (i as f32 - j as f32) * 0.05));
            hs[0] = serde_json::from_value(case["initial_hidden"].clone())?;
            // Particle 3 duplicates particle 0 with swapped stream histories.
            hs[6] = hs[1];
            hs[7] = hs[0];
            let native_history = if mode == Mode::Gru {
                primordial_soup::gpu::audit::planar_gru_history(bytemuck::cast_slice(&hs), 4, 2)?
            } else {
                bytemuck::cast_slice(&hs).to_vec()
            };
            q.write_buffer(&memory, 0, &native_history);
            let mut maximum_error = 0.0f32;
            for tick in 0..40 {
                let mut inputs: [[f32; 4]; 8] = std::array::from_fn(|i| {
                    [
                        0.25 + tick as f32 * 0.1 + i as f32,
                        -0.5,
                        0.75 - i as f32 * 0.2,
                        0.125,
                    ]
                });
                if tick < 4 {
                    inputs[0] = serde_json::from_value(case["steps"][tick]["input"].clone())?;
                }
                if tick >= 8 {
                    for input in &mut inputs {
                        *input = [f32::MAX, -f32::MAX, f32::MAX, -f32::MAX];
                    }
                }
                inputs[6] = inputs[1];
                inputs[7] = inputs[0];
                q.write_buffer(&input, 0, bytemuck::cast_slice(&inputs));
                let mut encoder = d.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(4, 1, 1);
                }
                q.submit([encoder.finish()]);
                let got = floats(&read(d, q, &output, 128)?);
                for stream in 0..8 {
                    let expected = network::evaluate(
                        mode,
                        &probe_weights[(stream / 2) % 3],
                        inputs[stream],
                        &mut hs[stream],
                    );
                    if stream == 0 && tick < 4 {
                        let oracle: [f64; 4] =
                            serde_json::from_value(case["steps"][tick]["output"].clone())?;
                        for j in 0..4 {
                            ensure!(
                                (f64::from(got[j]) - oracle[j]).abs() < 3e-5,
                                "GPU independent oracle mismatch"
                            );
                        }
                    }
                    for j in 0..4 {
                        let error = (expected[j] - got[stream * 4 + j]).abs();
                        maximum_error = maximum_error.max(error);
                        ensure!(
                            error < 3e-5,
                            "{} evaluator mismatch tick {tick}: {error}",
                            mode.label()
                        );
                    }
                }
                ensure!(
                    got[0..4] == got[28..32] && got[4..8] == got[24..28],
                    "stream swap broke reflection equivalence"
                );
                if mode == Mode::Gru {
                    let native = read(d, q, &memory, 128)?;
                    let canonical =
                        primordial_soup::gpu::audit::interleave_gru_history(&native, 4, 2)?;
                    let got = floats(&canonical);
                    for i in 0..32 {
                        ensure!(
                            (got[i] - hs[i / 4][i % 4]).abs() < 3e-5,
                            "GRU hidden state mismatch"
                        );
                    }
                }
            }
            // Real update pipeline: initialize, compare batched and serial work.
            let size = Sizing {
                canvas: [32, 32],
                entity_count: 67,
                sqrt_world_size: 1.0,
            };
            let mut extreme = p.clone();
            extreme.config_mut().sensor_gain = f32::MAX as f64;
            ensure!(
                extreme.validate().is_err(),
                "overflowing sensor gain accepted"
            );
            extreme.config_mut().sensor_gain = network::INPUT_LIMIT as f64;
            let mut bounded = Engine::new(d, q, size, &extreme, workgroup)?;
            step(d, q, &mut bounded, &extreme, 120)?;
            finite_particles(&read(d, q, &bounded.entities, 67 * 32)?)?;
            let bounded_memory = floats(&read(
                d,
                q,
                &bounded.brain_memory,
                bounded.brain_memory.size(),
            )?);
            ensure!(
                bounded_memory
                    .iter()
                    .all(|v| v.is_finite() && v.abs() <= 1.000001),
                "bounded sensor gain corrupted memory"
            );
            let mut e = Engine::new(d, q, size, &p, workgroup)?;
            step(d, q, &mut e, &p, 1)?;
            let initial = read(d, q, &e.entities, 67 * 32)?;
            step(d, q, &mut e, &p, 6)?;
            let batch = read(d, q, &e.entities, 67 * 32)?;
            finite_particles(&batch)?;
            let batch_memory = read(d, q, &e.brain_memory, e.brain_memory.size())?;
            let memory_before = batch_memory.clone();
            let mut display = p.clone();
            display.config_mut().color_offset += 0.1;
            e.upload_project(d, q, &display)?;
            ensure!(
                read(d, q, &e.brain_memory, e.brain_memory.size())? == memory_before,
                "display upload erased memory"
            );
            e.reset();
            step(d, q, &mut e, &p, 1)?;
            ensure!(
                read(d, q, &e.entities, 67 * 32)? == initial,
                "reset did not restore baseline"
            );
            if mode == Mode::Gru {
                ensure!(
                    read(d, q, &e.brain_memory, 67 * 32)?
                        .iter()
                        .all(|b| *b == 0),
                    "reset retained memory"
                );
            }
            for _ in 0..6 {
                step(d, q, &mut e, &p, 1)?;
            }
            ensure!(
                read(d, q, &e.entities, 67 * 32)? == batch,
                "serial/batched network particles differ"
            );
            ensure!(
                read(d, q, &e.brain_memory, e.brain_memory.size())? == batch_memory,
                "serial/batched memory differs"
            );
            ensure!(
                e.request_pick(q, &p.world, [0.0, 0.0], 4.0, -1.0),
                "pick request rejected"
            );
            let mut encoder = d.create_command_encoder(&Default::default());
            ensure!(e.encode_pick(&mut encoder), "pick not encoded");
            q.submit([encoder.finish()]);
            e.begin_pick_readback();
            wait(d)?;
            let picked = e.take_pick()?.flatten().context("network pick missed")?;
            ensure!(
                picked.mode == mode
                    && picked.rule == weights.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                "wrong adopted coefficients"
            );
            // Same-mode reroll must replace memory, whereas pause/no tick preserves it.
            let mut changed = p.clone();
            changed.config_mut().mutation_scale = 0.2;
            e.upload_project(d, q, &changed)?;
            ensure!(
                read(d, q, &e.brain_memory, e.brain_memory.size())?
                    .iter()
                    .all(|b| *b == 0),
                "rule edit retained memory"
            );
            for boundary in 0..4 {
                let mut next = p.clone();
                next.world.boundary_conditions = boundary;
                next.config_mut().hazard_rate = 1.0;
                e.upload_project(d, q, &next)?;
                step(d, q, &mut e, &next, 1)?;
                if mode == Mode::Gru {
                    ensure!(
                        read(d, q, &e.brain_memory, 67 * 32)?
                            .iter()
                            .all(|b| *b == 0),
                        "respawn retained memory"
                    );
                }
            }
            // Real boundary respawn/death must erase only the affected memory.
            if mode == Mode::Gru {
                for boundary in [2, 3] {
                    let mut next = p.clone();
                    next.world.boundary_conditions = boundary;
                    e.upload_project(d, q, &next)?;
                    let mut particles = vec![[0u32; 8]; 67];
                    for particle in &mut particles {
                        particle[0] = 2.0f32.to_bits();
                        particle[1] = 2.0f32.to_bits();
                        particle[2] = 0.001f32.to_bits();
                        particle[4] = 0.001f32.to_bits();
                    }
                    q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
                    q.write_buffer(
                        &e.brain_memory,
                        0,
                        bytemuck::cast_slice(&vec![0.5f32; 67 * 8]),
                    );
                    e.frame_count = 1;
                    step(d, q, &mut e, &next, 1)?;
                    ensure!(
                        read(d, q, &e.brain_memory, 67 * 32)?
                            .iter()
                            .all(|b| *b == 0),
                        "boundary retained recurrent state"
                    );
                }
            }
            // Inject a scoped validation error while preparing a valid project.
            // Dropping the failed transaction must preserve the active buffers.
            let before = read(d, q, &e.entities, 67 * 32)?;
            let before_memory = read(d, q, &e.brain_memory, e.brain_memory.size())?;
            let frame = e.frame_count;
            let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
            let prepared = e.prepare_project(d, q, &p)?;
            let _invalid = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("intentional rejection fixture"),
                size: 32,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            ensure!(
                pollster::block_on(scope.pop()).is_some(),
                "injected validation error was not caught"
            );
            drop(prepared);
            ensure!(
                e.frame_count == frame
                    && read(d, q, &e.entities, 67 * 32)? == before
                    && read(d, q, &e.brain_memory, e.brain_memory.size())? == before_memory,
                "failed preparation mutated live state"
            );
            // Render a bounded random search for visual inspection.
            let mut visual = Project::default();
            visual.config_mut().switch_mode(mode, 0.25);
            visual.config_mut().cohorts = 4;
            let mut world = Engine::new(
                d,
                q,
                Sizing {
                    canvas: [128, 128],
                    entity_count: 16384,
                    sqrt_world_size: 1.0,
                },
                &visual,
                workgroup,
            )?;
            let mut renderer = Renderer::new(d, &world, [512, 512])?;
            for _ in 0..8 {
                step(d, q, &mut world, &visual, 120)?;
            }
            let mut encoder = d.create_command_encoder(&Default::default());
            renderer.begin_frame(q, &world, &Camera::default(), &visual, 1, -1.0);
            renderer.clear_accumulator(&mut encoder);
            renderer.sample(&mut encoder, &world, &Camera::default());
            renderer.present(
                q,
                &mut encoder,
                &world,
                &Camera::default(),
                &Preferences::default(),
                Overlays::default(),
            );
            renderer.present_clean(
                d,
                q,
                &mut encoder,
                &world,
                &Camera::default(),
                &Preferences::default(),
            );
            q.submit([encoder.finish()]);
            wait(d)?;
            let pixels = tex(
                d,
                q,
                renderer.clean_texture().context("no clean render")?,
                [512, 512],
                4,
            )?;
            let path = format!("artifacts/network-{}.png", mode.label().to_lowercase());
            std::fs::write(
                &path,
                primordial_soup::export::encode_png(&pixels, [512, 512])?,
            )?;
            results.push(json!({"mode":mode,"cpu_gpu_max_abs_error":maximum_error,"independent_dense_oracle":true,"probe_particles":4,"probe_cohorts":3,"probe_ticks":40,"stream_swap_equivalent":true,"extreme_finite_input":true,"bounded_gain_ticks":120,"serial_batch_equal":true,"baseline_repeatable":true,"exact_pick":true,"memory_lifecycle":true,"scoped_failure_preserves_state":true,"render":path}));
        }
        Ok(json!({"networks":results}))
    }
    fn configure_tiled(d: &wgpu::Device, engine: &Engine, renderer: &mut Renderer) -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        if args.iter().any(|v| v == "--cells-tiled-check") {
            return Ok(());
        }
        if args.iter().any(|v| v == "--analytic-raster") {
            if let Some(cells) = renderer.cells.as_mut() {
                cells.use_analytic_distances();
            }
        }
        if !args.iter().any(|v| v == "--tiled") {
            return Ok(());
        }
        let limit = args
            .iter()
            .position(|v| v == "--tiled-index-limit")
            .map(|i| {
                args.get(i + 1)
                    .context("--tiled-index-limit needs a value")?
                    .parse::<u32>()
                    .context("Invalid tiled index limit")
            })
            .transpose()?;
        if let Some(cells) = renderer.cells.as_mut() {
            cells.enable_tiled(d, engine, limit);
        }
        Ok(())
    }
    fn tiled_state(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        renderer: &Renderer,
    ) -> Result<Option<serde_json::Value>> {
        let Some(buffer) = renderer
            .cells
            .as_ref()
            .and_then(|cells| cells.tiled_state_buffer())
        else {
            return Ok(None);
        };
        let bytes = read(d, q, buffer, 16)?;
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        Ok(Some(
            json!({"overflow":words[0],"references":words[1],"enabled":words[2],"maximum_occupancy":words[3]}),
        ))
    }
    fn cells_tiled_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::Style;
        let args: Vec<String> = std::env::args().collect();
        let directory = |key: &str| -> Result<Option<&String>> {
            args.iter()
                .position(|arg| arg == key)
                .map(|index| {
                    args.get(index + 1)
                        .filter(|path| !path.is_empty() && !path.starts_with("--"))
                        .with_context(|| format!("{key} needs a directory"))
                })
                .transpose()
        };
        let capture_dir = directory("--capture-dir")?;
        let baseline_dir = directory("--baseline-dir")?;
        if let Some(dir) = capture_dir {
            std::fs::create_dir_all(dir)?;
        }
        let n = 262_145u32;
        let mut project = Project::default();
        project.world.boundary_conditions = 1;
        project.config_mut().color_sensitivity = 0.0;
        project.config_mut().color_offset = 0.0;
        let mut engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: n,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        step(d, q, &mut engine, &project, 1)?;
        let initial_tick = engine.frame_count;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.bloom_enabled = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.reference_palette = false;
        prefs.cells.density = 0.0;
        prefs.cells.scale = 4.0;
        prefs.cells.zoom_response = 1.0;
        prefs.cells.membranes = 1.0;
        let mut renderers = Vec::new();
        for limit in [None, None, Some(1)] {
            let mut renderer = Renderer::new(d, &engine, [256, 256])?;
            renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
            renderer.cells.as_mut().unwrap().use_analytic_distances();
            if !renderers.is_empty() {
                ensure!(
                    renderer
                        .cells
                        .as_mut()
                        .unwrap()
                        .enable_tiled(d, &engine, limit),
                    "Adapter does not support tiled parity fixture"
                );
            }
            renderers.push(renderer);
        }
        let mut particles = vec![[0u32; 8]; n as usize];
        let mut results = Vec::new();
        for (name, count, edge) in [
            ("compact32", 32u32, false),
            ("compact4096", 4096, false),
            ("fallback_dense", n, false),
            ("wrapped_edge", 32, true),
            ("empty", 0, false),
        ] {
            for (i, particle) in particles.iter_mut().enumerate() {
                particle[0] = if edge { 0.999f32.to_bits() } else { 0 };
                particle[2] = if i % 2 == 0 { 0.001f32 } else { -0.001f32 }.to_bits();
                particle[4] = 0.001f32.to_bits();
                particle[5] = if i < count as usize { 0 } else { u32::MAX };
                particle[7] = ((i % 4) as f32).to_bits();
            }
            q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
            let camera = Camera {
                zoom: if edge { 1.0 } else { 100.0 },
                ..Default::default()
            };
            for (samples, highlight, membranes) in [
                (1, 0.0f32, 1.0),
                (4, -1.0, 1.0),
                (1, 1.0, 1.0),
                (1, 0.0, 1.0),
                (1, -1.0, 0.0),
                (1, -1.0, 1.0),
            ] {
                prefs.cells.membranes = membranes;
                let mut snapshots = Vec::new();
                for renderer in &mut renderers {
                    cells_frame(
                        d, q, &engine, renderer, &project, &prefs, &camera, samples, highlight,
                    )?;
                    let cells = renderer.cells.as_ref().unwrap();
                    let image = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
                    let selection = floats(&tex(d, q, cells.selected_texture(), cells.padded, 4)?);
                    let membrane = floats(&read(
                        d,
                        q,
                        &cells.membrane_history,
                        cells.membrane_history.size(),
                    )?);
                    ensure!(
                        image
                            .iter()
                            .chain(&selection)
                            .chain(&membrane)
                            .all(|v| v.is_finite()),
                        "Nonfinite tiled parity fixture"
                    );
                    let pixels = tex(d, q, renderer.output_texture(), [256, 256], 4)?;
                    let plan = read(d, q, cells.sprite_plan_buffer(), 16)?;
                    let mode = u32::from_le_bytes(plan[8..12].try_into().unwrap());
                    snapshots.push((
                        image,
                        selection,
                        membrane,
                        pixels,
                        tiled_state(d, q, renderer)?,
                        mode,
                        read(
                            d,
                            q,
                            cells.membrane_sites_buffer(),
                            cells.membrane_sites_buffer().size(),
                        )?,
                    ));
                }
                let reference = &snapshots[0];
                if capture_dir.is_some() || baseline_dir.is_some() {
                    for (variant, snapshot) in snapshots.iter().enumerate() {
                        let label =
                            format!("{name}-case{}-variant{variant}.bin", results.len() / 2);
                        let bytes = [
                            bytemuck::cast_slice(snapshot.0.as_slice()),
                            bytemuck::cast_slice(snapshot.1.as_slice()),
                            bytemuck::cast_slice(snapshot.2.as_slice()),
                            snapshot.3.as_slice(),
                            snapshot.6.as_slice(),
                        ]
                        .concat();
                        if let Some(dir) = baseline_dir {
                            ensure!(
                                bytes == std::fs::read(Path::new(dir).join(&label))?,
                                "Campaign baseline differs at {label}"
                            );
                        }
                        if let Some(dir) = capture_dir {
                            std::fs::write(Path::new(dir).join(label), bytes)?;
                        }
                    }
                }
                ensure!(
                    if count == n {
                        matches!(reference.5, 1 | 2)
                    } else {
                        reference.5 == 0
                    },
                    "Unexpected planner mode in tiled fixture {name}"
                );
                for (index, candidate) in snapshots.iter().enumerate().skip(1) {
                    let maximum_error = |a: &[f32], b: &[f32]| {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| f64::from((x - y).abs()))
                            .fold(0.0, f64::max)
                    };
                    let image_error = maximum_error(&reference.0, &candidate.0);
                    let selection_error = maximum_error(&reference.1, &candidate.1);
                    let membrane_error = maximum_error(&reference.2, &candidate.2);
                    let mass = |v: &[f32]| v.chunks_exact(4).map(|p| f64::from(p[3])).sum::<f64>();
                    let reference_mass = mass(&reference.0);
                    let mass_error = if reference_mass == 0.0 {
                        mass(&candidate.0).abs()
                    } else {
                        (mass(&candidate.0) / reference_mass - 1.0).abs()
                    };
                    let max_pixel_error = reference
                        .3
                        .iter()
                        .zip(&candidate.3)
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap_or(0);
                    let scale = reference
                        .0
                        .iter()
                        .map(|v| f64::from(v.abs()))
                        .fold(1e-10, f64::max);
                    let selection_scale = reference
                        .1
                        .iter()
                        .map(|v| f64::from(v.abs()))
                        .fold(1e-10, f64::max);
                    ensure!(
                        mass_error < 0.01
                            && image_error / scale < 0.01
                            && selection_error / selection_scale < 0.01,
                        "Tiled raw mass/color/selection mismatch at {name}"
                    );
                    ensure!(
                        membrane_error <= 1e-5,
                        "Tiled membrane mismatch at {name}: {membrane_error}"
                    );
                    ensure!(
                        max_pixel_error <= 1,
                        "Tiled final pixel mismatch at {name}: {max_pixel_error}"
                    );
                    ensure!(
                        reference.6 == candidate.6,
                        "Tiled nearest membrane site keys differ at {name}"
                    );
                    let state = candidate.4.as_ref().context("Missing tiled state")?;
                    if index == 2 && count > 0 {
                        ensure!(
                            state["enabled"] == 0,
                            "Forced-capacity fixture did not fall back"
                        );
                        ensure!(
                            reference.0 == candidate.0
                                && reference.1 == candidate.1
                                && reference.2 == candidate.2
                                && reference.3 == candidate.3,
                            "Forced capacity fallback differs from raster"
                        );
                    } else if reference.5 == 0 {
                        ensure!(
                            state["enabled"] == 1,
                            "Compact fixture did not exercise tiled path"
                        );
                    }
                    results.push(json!({"fixture":name,"samples":samples,"highlight":highlight,"membranes":membranes,"variant":if index==1 {"tiled"}else{"forced_capacity_fallback"},"planner_mode":candidate.5,"state":state,"image_max_abs_error":image_error,"selection_max_abs_error":selection_error,"membrane_max_abs_error":membrane_error,"membrane_site_keys_exact":true,"mass_relative_error":mass_error,"final_pixel_max_error_255":max_pixel_error}));
                }
            }
        }
        ensure!(
            engine.frame_count == initial_tick,
            "Tiled rendering advanced simulation"
        );
        Ok(
            json!({"reference":"analytic pixel distances; compatibility with original separate","campaign_baseline_exact":baseline_dir.is_some(),"same_engine_particles":true,"simulation_unchanged":true,"cases":results}),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn capture_cells_frame(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        engine: &Engine,
        renderer: &Renderer,
        project: &Project,
        prefs: &Preferences,
        camera: &Camera,
        samples: u32,
        highlight: f32,
    ) -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let args: Vec<_> = std::env::args().collect();
        let Some(index) = args.iter().position(|arg| arg == "--cells-capture-dir") else {
            return Ok(());
        };
        let directory = args
            .get(index + 1)
            .filter(|path| !path.is_empty() && !path.starts_with("--"))
            .context("--cells-capture-dir needs a directory")?;
        let Some(cells) = renderer.cells.as_ref().filter(|_| renderer.cells_active()) else {
            return Ok(());
        };
        let directory =
            Path::new(directory).join(format!("{:04}", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&directory)?;
        let mut files = serde_json::Map::new();
        let mut save = |name: &str, bytes: Vec<u8>| -> Result<()> {
            std::fs::write(directory.join(name), &bytes)?;
            files.insert(
                name.into(),
                json!({"bytes":bytes.len(),"fnv1a64":bytes_hash(&bytes)}),
            );
            Ok(())
        };
        save(
            "image.rgba32f",
            tex(d, q, &cells.images[cells.active], cells.padded, 16)?,
        )?;
        save(
            "selection.r32f",
            tex(d, q, cells.selected_texture(), cells.padded, 4)?,
        )?;
        save(
            "map.rgba32f",
            tex(d, q, &cells.maps[cells.displayed_map], cells.map_size, 16)?,
        )?;
        save(
            "output.rgba8",
            tex(d, q, renderer.output_texture(), renderer.size, 4)?,
        )?;
        save(
            "clean.rgba8",
            tex(d, q, renderer.clean_texture()?, renderer.size, 4)?,
        )?;
        let transition = ((camera.zoom - 2.0) / 6.0).clamp(0.0, 1.0);
        let membranes =
            prefs.cells.membranes * transition * transition * (3.0 - 2.0 * transition) > 0.0;
        if membranes {
            let sites = cells.membrane_sites_buffer();
            save("sites.u32", read(d, q, sites, sites.size())?)?;
            save(
                "membranes.f32",
                read(d, q, &cells.membrane_history, cells.membrane_history.size())?,
            )?;
        }
        let plan = read(d, q, cells.sprite_plan_buffer(), 16)?;
        let words: Vec<_> = plan
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let batch_settings = read(d, q, cells.sprite_batch_settings_buffer(), 32)?;
        let batch_settings_words: Vec<_> = batch_settings
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        let metadata = json!({"format":"cells-frame-capture-v1","byte_order":"little_endian",
            "files":files,"output":renderer.size,"padded":cells.padded,"map_size":cells.map_size,
            "samples":samples,"highlight_bits":highlight.to_bits(),"membranes_valid":membranes,
            "camera":{"pan_bits":camera.pan.map(f32::to_bits),"zoom_bits":camera.zoom.to_bits(),"trail_mode":camera.trail_mode},
            "cells_settings":prefs.cells,"boundary":project.world.boundary_conditions,
            "particles":engine.sizing.entity_count,"simulation_tick":engine.frame_count,
            "planner_words":words,"batch_settings_words":batch_settings_words,"tiled_state":tiled_state(d, q, renderer)?,
            "note":"Compare every raw file exactly. Planner execution mode may explicitly change 1 to 2; other metadata and raw values are not normalized."});
        std::fs::write(
            directory.join("capture.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
        Ok(())
    }
    fn cells_frame(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        engine: &Engine,
        renderer: &mut Renderer,
        project: &Project,
        prefs: &Preferences,
        camera: &Camera,
        samples: u32,
        highlight: f32,
    ) -> Result<()> {
        configure_tiled(d, engine, renderer)?;
        renderer.set_preferences(prefs);
        renderer.begin_frame(q, engine, camera, project, samples, highlight);
        let mut c = d.create_command_encoder(&Default::default());
        renderer.clear_accumulator(&mut c);
        for _ in 0..samples {
            renderer.sample(&mut c, engine, camera);
        }
        renderer.present(q, &mut c, engine, camera, prefs, Overlays::default());
        renderer.present_clean(d, q, &mut c, engine, camera, prefs);
        q.submit([c.finish()]);
        wait(d)?;
        capture_cells_frame(
            d, q, engine, renderer, project, prefs, camera, samples, highlight,
        )
    }
    fn cells_fixture_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::Style;
        let mut p = Project::default();
        p.world.boundary_conditions = 1;
        p.config_mut().color_sensitivity = 0.0;
        p.config_mut().color_offset = 0.0;
        let n = 262_145;
        let mut e = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: n,
                sqrt_world_size: 1.0,
            },
            &p,
            workgroup,
        )?;
        step(d, q, &mut e, &p, 1)?;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.bloom_enabled = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.reference_palette = false;
        prefs.cells.density = 0.0;
        prefs.cells.scale = 4.0;
        prefs.cells.zoom_response = 1.0;
        let camera = Camera {
            zoom: 100.0,
            ..Default::default()
        };
        let mut r = Renderer::new(d, &e, [256, 256])?;
        r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
        let mut particles = vec![[0u32; 8]; n as usize];
        let mut dense = Vec::new();
        for count in [1u32, 32, 4096, 65536, n] {
            for (i, v) in particles.iter_mut().enumerate() {
                v[2] = if i % 2 == 0 { 0.001f32 } else { -0.001f32 }.to_bits();
                v[4] = 0.001f32.to_bits();
                v[5] = if i < count as usize { 0 } else { u32::MAX };
                v[7] = ((i % 4) as f32).to_bits();
            }
            q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
            cells_frame(d, q, &e, &mut r, &p, &prefs, &camera, 1, 0.0)?;
            let cells = r.cells.as_ref().unwrap();
            let image = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
            ensure!(image.iter().all(|v| v.is_finite()), "Dense pile nonfinite");
            let scale = 256.0f64 / 1080.0;
            let radius = 24.0 * scale;
            let mut expected = 0.0;
            let mut actual = 0.0;
            let mut rgb = [0.0; 3];
            for (i, v) in image.chunks_exact(4).enumerate() {
                let x = (i % cells.padded[0] as usize) as f64 + 0.5 - cells.padded[0] as f64 * 0.5;
                let y = (i / cells.padded[0] as usize) as f64 + 0.5 - cells.padded[1] as f64 * 0.5;
                let rr = (x * x + y * y) / (radius * radius);
                if rr <= 1.0 {
                    expected += count as f64 * 0.4 * scale * scale * 0.85896 / (radius * radius)
                        * (-rr / (2.0 * 0.45 * 0.45)).exp();
                }
                actual += v[3] as f64;
                for j in 0..3 {
                    rgb[j] += v[j] as f64;
                }
            }
            let error = (actual / expected - 1.0).abs();
            ensure!(error < 0.01, "Dense mass error {error} at {count}");
            for (j, color) in [1.0, 0.2, 0.2].iter().enumerate() {
                ensure!(
                    (rgb[j] / actual - color).abs() < 0.01,
                    "Dense pigment error"
                );
            }
            let counts = read(d, q, &cells.counts, cells.counts.size())?;
            let total: u64 = counts
                .chunks_exact(4)
                .map(|a| u32::from_le_bytes(a.try_into().unwrap()) as u64)
                .sum();
            ensure!(
                total == count as u64,
                "Opposing velocities/dead slots changed scalar occupancy"
            );
            dense.push(json!({"live":count,"mass_relative_error":error,"mass":actual}));
        }
        let single = floats(&tex(
            d,
            q,
            &r.cells.as_ref().unwrap().images[r.cells.as_ref().unwrap().active],
            r.cells.as_ref().unwrap().padded,
            16,
        )?);
        let highlighted_clean = tex(d, q, r.clean_texture()?, [256, 256], 4)?;
        cells_frame(d, q, &e, &mut r, &p, &prefs, &camera, 4, 0.0)?;
        let cells = r.cells.as_ref().unwrap();
        let blurred = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
        ensure!(
            single
                .iter()
                .zip(&blurred)
                .all(|(a, b)| (a - b).abs() <= 1e-4 * a.abs().max(1.0)),
            "Temporal sample normalization drift"
        );
        ensure!(
            tex(d, q, r.clean_texture()?, [256, 256], 4)? == highlighted_clean,
            "Clean output changed under static blur"
        );
        let cached = tex(d, q, r.clean_texture()?, [256, 256], 4)?;
        r.clear_cells_highlight();
        let mut c = d.create_command_encoder(&Default::default());
        r.present(q, &mut c, &e, &camera, &prefs, Overlays::default());
        r.present_clean(d, q, &mut c, &e, &camera, &prefs);
        q.submit([c.finish()]);
        wait(d)?;
        ensure!(
            tex(d, q, r.clean_texture()?, [256, 256], 4)? == cached,
            "Clearing highlight lost held blur"
        );
        ensure!(
            tex(d, q, r.output_texture(), [256, 256], 4)? == cached,
            "Clean/display mismatch without overlays"
        );
        // Exercise selection sum read/reset across real raster passes, temporal
        // samples and consecutive frames in both planner execution modes.
        let mut selection_transitions = Vec::new();
        for (count, expected_mode) in [(32u32, 0u32), (n, 1u32)] {
            for (i, particle) in particles.iter_mut().enumerate() {
                particle[5] = if i < count as usize { 0 } else { u32::MAX };
            }
            q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
            for highlight in [0.0f32, -1.0, 1.0, 0.0] {
                cells_frame(d, q, &e, &mut r, &p, &prefs, &camera, 4, highlight)?;
                let cells = r.cells.as_ref().unwrap();
                let plan = read(d, q, cells.sprite_plan_buffer(), 16)?;
                let mode = u32::from_le_bytes(plan[8..12].try_into().unwrap());
                ensure!(
                    mode == expected_mode || (expected_mode == 1 && mode == 2),
                    "Selection fixture expected planner mode {expected_mode} (or listed batching for 1), got {mode} with {count} live particles"
                );
                let image = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
                let selected = floats(&tex(d, q, cells.selected_texture(), cells.padded, 4)?);
                ensure!(
                    image.iter().chain(&selected).all(|v| v.is_finite()),
                    "Selection transition produced nonfinite mass"
                );
                let mass: f64 = image.chunks_exact(4).map(|v| f64::from(v[3])).sum();
                let selected_mass: f64 = selected.iter().map(|v| f64::from(*v)).sum();
                let chosen_count = (0..count)
                    .filter(|i| highlight < 0.0 || (*i % 4) as f32 == highlight)
                    .count();
                // All coincident particles have identical shape and mass. The
                // cohort count is computed independently of GPU selection data.
                let expected_selected = mass * chosen_count as f64 / f64::from(count);
                let relative_error = (selected_mass / expected_selected - 1.0).abs();
                ensure!(
                    mass > 0.0 && expected_selected > 0.0 && relative_error < 0.01,
                    "Selection mass error {relative_error}, mode {mode}, highlight {highlight}"
                );
                // Compare total mass with the independently checked dense fixture
                // too, so matching selection/color accumulation drift cannot pass.
                let reference = dense
                    .iter()
                    .find(|case| case["live"].as_u64() == Some(u64::from(count)))
                    .unwrap()["mass"]
                    .as_f64()
                    .unwrap();
                ensure!(
                    (mass / reference - 1.0).abs() < 0.01,
                    "Selection transition changed total mass"
                );
                selection_transitions.push(json!({"live":count,"planner_mode":mode,"highlight":highlight,"samples":4,"chosen_count":chosen_count,"selection_mass_relative_error":relative_error}));
            }
        }
        // Empty live population preserves the background and one-pixel world outline.
        for v in &mut particles {
            v[5] = u32::MAX;
        }
        q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
        cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 1, -1.0)?;
        let empty = tex(d, q, r.clean_texture()?, [256, 256], 4)?;
        let background_index = (8 * 256 + 8) * 4;
        let background = &empty[background_index..background_index + 4];
        ensure!(background[3] == 255, "Empty background is not opaque");
        for (i, pixel) in empty.chunks_exact(4).enumerate() {
            let x = i % 256;
            let y = i / 256;
            if x == 0 || x == 255 || y == 0 || y == 255 {
                ensure!(
                    pixel == [255, 255, 255, 255],
                    "Empty world outline is not opaque white at {x},{y}"
                );
            } else {
                ensure!(pixel == background, "Empty background differs at {x},{y}");
            }
        }
        // A wrapped edge particle must contribute on both edges; nonwrap must not.
        particles[0][0] = 0.999f32.to_bits();
        particles[0][5] = 0;
        q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
        let edge_cam = Camera::default();
        cells_frame(d, q, &e, &mut r, &p, &prefs, &edge_cam, 1, -1.0)?;
        let cells = r.cells.as_ref().unwrap();
        let edge = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
        let pad = (cells.padded[0] - 256) / 2;
        let left: f32 = (pad..pad + 2)
            .flat_map(|x| {
                (0..cells.padded[1]).map(move |y| (y * cells.padded[0] + x) as usize * 4 + 3)
            })
            .map(|i| edge[i])
            .sum();
        ensure!(left > 0.0, "Wrapped sprites lost opposite-edge support");
        p.world.boundary_conditions = 0;
        cells_frame(d, q, &e, &mut r, &p, &prefs, &edge_cam, 1, -1.0)?;
        let cells = r.cells.as_ref().unwrap();
        let edge = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
        let left: f32 = (pad..pad + 2)
            .flat_map(|x| {
                (0..cells.padded[1]).map(move |y| (y * cells.padded[0] + x) as usize * 4 + 3)
            })
            .map(|i| edge[i])
            .sum();
        ensure!(left == 0.0, "Nonwrapped sprite leaked across world edge");
        p.world.boundary_conditions = 1;
        prefs.cells.scale = 1.5;
        for (i, v) in particles.iter_mut().enumerate() {
            v[0] = 0.96f32.to_bits();
            v[5] = if i < 4096 { 0 } else { u32::MAX };
        }
        q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
        cells_frame(d, q, &e, &mut r, &p, &prefs, &edge_cam, 1, -1.0)?;
        let cells = r.cells.as_ref().unwrap();
        let image = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
        let exterior: f32 = (0..pad)
            .flat_map(|x| {
                (0..cells.padded[1]).map(move |y| (y * cells.padded[0] + x) as usize * 4 + 3)
            })
            .map(|i| image[i])
            .sum();
        ensure!(
            exterior > 0.0,
            "Wrapped wash lost ghost source beyond the sprite radius"
        );
        cells_frame(
            d,
            q,
            &e,
            &mut r,
            &p,
            &prefs,
            &Camera {
                zoom: 0.1,
                ..Default::default()
            },
            1,
            -1.0,
        )?;
        let cells = r.cells.as_ref().unwrap();
        ensure!(
            floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?)
                .iter()
                .all(|v| v.is_finite()),
            "Zoom-out wrap nonfinite"
        );
        // Rejected resource preparation leaves the active render and simulation usable.
        let before = read(d, q, &e.entities, e.entities.size())?;
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let prepared = r.prepare_cells(d, &e, &prefs.cells)?;
        let _invalid = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cells rejected preparation fixture"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        ensure!(
            pollster::block_on(scope.pop()).is_some(),
            "Injected allocation error escaped scope"
        );
        drop(prepared);
        ensure!(
            read(d, q, &e.entities, e.entities.size())? == before,
            "Failed preparation changed particles"
        );
        ensure!(
            Renderer::new(d, &e, [d.limits().max_texture_dimension_2d + 1, 256]).is_err(),
            "Oversize target accepted"
        );
        Ok(
            json!({"dense_piles":dense,"opposite_velocities_and_dead_slots":true,"four_sample_normalization":true,"cached_highlight_clean_parity":true,"selection_transition_compact_and_fallback":true,"selection_transitions":selection_transitions,"empty_paper":true,"wrap_and_nonwrap_sprites":true,"wrapped_exterior_filter_support":true,"minimum_zoom_finite":true,"scoped_preparation_failure":true}),
        )
    }
    fn cells_timing(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let mut p = Project::default();
        p.config_mut().cohorts = 4;
        let mut e = Engine::new(d, q, sizing_for(1.0, 1.0)?, &p, workgroup)?;
        step(d, q, &mut e, &p, 120)?;
        std::fs::write("artifacts/cells-timing-project.json", p.to_json()?)?;
        let mut results = Vec::new();
        // Zoom 4 exceeds the wide-sprite radius, where particle batches shrink to 65,536.
        for (size, samples, zoom) in [
            ([1920, 1080], 1, 1.0),
            ([1920, 1080], 4, 1.0),
            ([3840, 2160], 1, 1.0),
            ([1920, 1080], 1, 4.0),
        ] {
            let camera = Camera {
                zoom,
                ..Default::default()
            };
            for cellular in [false, true] {
                let mut prefs = Preferences::default();
                prefs.cells.fast = false;
                prefs.cells.style = if cellular {
                    primordial_soup::cells::Style::Cells
                } else {
                    primordial_soup::cells::Style::Classic
                };
                let mut r = Renderer::new(d, &e, size)?;
                if cellular {
                    r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
                }
                r.set_preferences(&prefs);
                let mut profiler = primordial_soup::profiling::GpuProfiler::new(d)
                    .context("GPU timestamps unavailable")?;
                let mut times = Vec::new();
                let mut stages = Vec::new();
                let mut cpu = Vec::new();
                for frame in 0..11 {
                    let started = Instant::now();
                    r.begin_frame(q, &e, &camera, &p, samples, -1.0);
                    let mut c = d.create_command_encoder(&Default::default());
                    let mut profile = profiler
                        .begin_frame(&mut c)
                        .context("Profiler unavailable")?;
                    r.clear_accumulator(&mut c);
                    for _ in 0..samples {
                        r.sample_profiled(&mut c, &e, &camera, Some(&mut profile));
                    }
                    r.present_profiled(
                        q,
                        &mut c,
                        &e,
                        &camera,
                        &prefs,
                        Overlays::default(),
                        Some(&mut profile),
                    );
                    profile.finish(&mut c);
                    let commands = c.finish();
                    let encode = started.elapsed().as_secs_f64() * 1000.0;
                    q.submit([commands]);
                    profiler.after_submit(profile);
                    wait(d)?;
                    let sample = profiler.poll(d, q).pop().context("Missing GPU sample")?;
                    if frame >= 3 {
                        times.push(sample.total_ms);
                        stages.push(sample.stages_ms);
                        cpu.push(encode);
                    }
                }
                let median = |v: &[f64]| {
                    let mut v = v.to_vec();
                    v.sort_by(f64::total_cmp);
                    (v[3] + v[4]) * 0.5
                };
                let stage: Vec<_> = (0..8)
                    .map(|i| median(&stages.iter().map(|v| v[i]).collect::<Vec<_>>()))
                    .collect();
                results.push(json!({"style":prefs.cells.style,"size":size,"samples":samples,"zoom":zoom,"population":600000,"simulation_frame":e.frame_count,"simulation_ticks_during_measurement":0,"gpu_ms_median":median(&times),"cpu_encode_ms_median":median(&cpu),"stages_ms_median":stage,"stage_names":primordial_soup::profiling::Stage::NAMES,"logical_render_bytes":r.texture_bytes()}));
            }
        }
        Ok(json!({"frames_per_case":8,"warmup_frames":3,"fixed_snapshot":true,"cases":results}))
    }
    fn cells_mapping_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let mut p = Project::default();
        p.world.boundary_conditions = 1;
        let mut e = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: 2,
                sqrt_world_size: 1.0,
            },
            &p,
            workgroup,
        )?;
        step(d, q, &mut e, &p, 1)?;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = primordial_soup::cells::Style::Cells;
        prefs.bloom_enabled = false;
        let mut r = Renderer::new(d, &e, [256, 256])?;
        r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
        cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 1, -1.0)?;
        let cells = r.cells.as_ref().unwrap();
        let [mw, mh] = cells.map_size;
        let mut map = Vec::new();
        for y in 0..mh {
            for x in 0..mw {
                map.extend_from_slice(&[
                    0.9 * (x as f32 * 0.8).sin(),
                    0.9 * (y as f32 * 0.7).cos(),
                    0.0,
                    0.0,
                ]);
            }
        }
        q.write_texture(
            cells.maps[cells.displayed_map].as_image_copy(),
            bytemuck::cast_slice(&map),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mw * 16),
                rows_per_image: Some(mh),
            },
            wgpu::Extent3d {
                width: mw,
                height: mh,
                depth_or_array_layers: 1,
            },
        );
        let padded = cells.padded;
        let scale = 256.0f64 / 1080.0;
        let pad = (padded[0] - 256) as f64 * 0.5;
        let delta = |p: [f64; 2]| {
            let pos = [
                (p[0] / padded[0] as f64).clamp(0.0, 1.0) * (mw - 1) as f64,
                (p[1] / padded[1] as f64).clamp(0.0, 1.0) * (mh - 1) as f64,
            ];
            let ix = pos[0].floor() as u32;
            let iy = pos[1].floor() as u32;
            let f = [pos[0] - ix as f64, pos[1] - iy as f64];
            let mut result = [0.0; 2];
            for y in 0..2 {
                for x in 0..2 {
                    let i = (((iy + y).min(mh - 1) * mw + (ix + x).min(mw - 1)) * 4) as usize;
                    let w = if x == 0 { 1.0 - f[0] } else { f[0] }
                        * if y == 0 { 1.0 - f[1] } else { f[1] };
                    for j in 0..2 {
                        result[j] += map[i + j] as f64 * w * scale;
                    }
                }
            }
            result
        };
        let displayed_uv = |uv: [f64; 2]| {
            let q = [uv[0] * 256.0 + pad, (1.0 - uv[1]) * 256.0 + pad];
            let mut pixel = q;
            for _ in 0..30 {
                let d = delta(pixel);
                pixel = [q[0] - d[0], q[1] - d[1]];
            }
            [(pixel[0] - pad) / 256.0, 1.0 - (pixel[1] - pad) / 256.0]
        };
        e.set_interaction_map(d, q, Some(cells));
        let mut particles = [[0u32; 8]; 2];
        particles[0][0] = 0.17f32.to_bits();
        particles[0][1] = (-0.23f32).to_bits();
        particles[0][4] = 0.001f32.to_bits();
        particles[1][5] = u32::MAX;
        q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&particles));
        let uv = displayed_uv([0.585, 0.385]);
        let target = [(uv[0] * 2.0 - 1.0) as f32, (uv[1] * 2.0 - 1.0) as f32];
        ensure!(
            e.request_pick(q, &p.world, target, 0.00001, -1.0),
            "Mapped pick rejected"
        );
        let mut c = d.create_command_encoder(&Default::default());
        e.encode_pick(&mut c);
        q.submit([c.finish()]);
        e.begin_pick_readback();
        wait(d)?;
        ensure!(
            e.take_pick()?.flatten().is_some_and(|hit| hit.index == 0),
            "Mapped pick missed independently inverted point"
        );
        let mut checked = 0;
        let mut changed = 0;
        for line in [false, true] {
            let data: Vec<u16> = (0..128 * 128).flat_map(|_| [0x3c00u16, 0, 0, 0]).collect();
            q.write_texture(
                e.field_texture(q).as_image_copy(),
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(128 * 8),
                    rows_per_image: Some(128),
                },
                wgpu::Extent3d {
                    width: 128,
                    height: 128,
                    depth_or_array_layers: 1,
                },
            );
            let a = if line { [0.2, 0.3] } else { [0.55, 0.5] };
            let b = if line { [0.85, 0.6] } else { a };
            let radius = 0.23;
            let mut c = d.create_command_encoder(&Default::default());
            e.paint(
                q,
                &mut c,
                Paint {
                    previous_uv: a.map(|v| v as f32),
                    uv: b.map(|v| v as f32),
                    size: (radius * 0.5) as f32,
                    power: 1.0,
                    erase: true,
                    mode: 0,
                    layer: 0,
                    angle: 0.0,
                    line_gain: 1.0,
                },
            )?;
            q.submit([c.finish()]);
            wait(d)?;
            let field = tex(d, q, e.field_texture(q), [128, 128], 8)?;
            for y in 0..128 {
                for x in 0..128 {
                    let original = [(x as f64 + 0.5) / 128.0, (y as f64 + 0.5) / 128.0];
                    let uv = displayed_uv(original);
                    let distance = |v: [f64; 2]| {
                        let pa = [v[0] - a[0], v[1] - a[1]];
                        let ba = [b[0] - a[0], b[1] - a[1]];
                        let den = ba[0] * ba[0] + ba[1] * ba[1];
                        let t = if den > 0.0 {
                            ((pa[0] * ba[0] + pa[1] * ba[1]) / den).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        ((pa[0] - ba[0] * t).powi(2) + (pa[1] - ba[1] * t).powi(2)).sqrt()
                    };
                    let dist = distance(uv);
                    if (dist - radius).abs() < 1e-6 {
                        continue;
                    }
                    let erased = u16::from_le_bytes(
                        field[((y * 128 + x) * 8) as usize..((y * 128 + x) * 8 + 2) as usize]
                            .try_into()
                            .unwrap(),
                    ) == 0;
                    ensure!(
                        erased == (dist < radius),
                        "Mapped {line:?} footprint mismatch at{x},{y}"
                    );
                    changed += usize::from((distance(original) < radius) != (dist < radius));
                    checked += 1;
                }
            }
        }
        ensure!(
            changed > 0,
            "Footprint fixture did not distinguish mapping from cursor-only mapping"
        );
        // Exercise the real assembly uniform and texture path, independently of
        // picking/painting: alternating field bars magnify a wrong map origin.
        let base = tex(d, q, r.clean_texture()?, [256, 256], 4)?;
        let bars: Vec<u16> = (0..128 * 128)
            .flat_map(|i| [if i % 2 == 0 { 0 } else { 0x2800 }, 0, 0, 0])
            .collect();
        q.write_texture(
            e.field_texture(q).as_image_copy(),
            bytemuck::cast_slice(&bars),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128 * 8),
                rows_per_image: Some(128),
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );
        prefs.field_opacity = 1.0;
        let mut c = d.create_command_encoder(&Default::default());
        r.present(
            q,
            &mut c,
            &e,
            &Camera::default(),
            &prefs,
            Overlays {
                walls: true,
                ..Default::default()
            },
        );
        q.submit([c.finish()]);
        wait(d)?;
        let output = tex(d, q, r.output_texture(), [256, 256], 4)?;
        let overlay_strength = |x: f64| {
            let pos = x / 2.0 - 0.5;
            let ix = pos.floor() as i32;
            let f = pos - pos.floor();
            let value = |i: i32| f64::from(i.clamp(0, 127) % 2);
            // 0x2800 is exactly 1/32 in f16; avoid saturating the overlay.
            1.0 - (-1.25 * (value(ix) * (1.0 - f) + value(ix + 1) * f)).exp()
        };
        let mut max_overlay_error = 0.0f64;
        let mut distinguishing_padding = 0usize;
        for y in 0..256 {
            for x in 0..256 {
                if x == 0 || x == 255 || y == 0 || y == 255 {
                    let i = (y * 256 + x) * 4;
                    ensure!(
                        base[i..i + 4] == [255, 255, 255, 255]
                            && output[i..i + 4] == [255, 255, 255, 255],
                        "Field overlay changed opaque white world outline at {x},{y}"
                    );
                    continue;
                }
                let p = [x as f64 + 0.5 + pad, y as f64 + 0.5 + pad];
                let displaced = delta(p);
                let g = overlay_strength(x as f64 + 0.5 + displaced[0]);
                let old_pad = (64.0 * scale).ceil();
                let legacy_p = [
                    (x as f64 + 0.5 + old_pad) / (256.0 + 2.0 * old_pad) * padded[0] as f64,
                    (y as f64 + 0.5 + old_pad) / (256.0 + 2.0 * old_pad) * padded[1] as f64,
                ];
                let legacy = overlay_strength(x as f64 + 0.5 + delta(legacy_p)[0]);
                for channel in 0..3 {
                    let i = (y * 256 + x) * 4 + channel;
                    let color = base[i] as f64 / 255.0;
                    let expected = (color * (1.0 - g) + g * g) * 255.0;
                    max_overlay_error = max_overlay_error.max((output[i] as f64 - expected).abs());
                    let wrong = (color * (1.0 - legacy) + legacy * legacy) * 255.0;
                    distinguishing_padding += usize::from((wrong - expected).abs() > 2.0);
                }
            }
        }
        ensure!(
            distinguishing_padding > 100,
            "Overlay fixture cannot detect stale padding"
        );
        ensure!(
            max_overlay_error < 1.5,
            "Field overlay disagrees with displayed map: {max_overlay_error} channel levels"
        );
        Ok(
            json!({"independent_mapped_pick":true,"circle_and_line_footprint_texels":checked,"texels_distinguishing_cursor_only":changed,"map_max_reference_displacement":0.9,"field_overlay":{"border_pixels":1020,"interior_pixels":64516,"max_rgb_error_255":max_overlay_error,"channels_distinguishing_stale_padding":distinguishing_padding}}),
        )
    }
    fn cells_density_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let mut p = Project::default();
        p.world.boundary_conditions = 1;
        let mut e = Engine::new(
            d,
            q,
            Sizing {
                canvas: [256, 128],
                entity_count: 1,
                sqrt_world_size: 1.0,
            },
            &p,
            workgroup,
        )?;
        step(d, q, &mut e, &p, 1)?;
        let entity = [0u32, 0, 0.001f32.to_bits(), 0, 0.001f32.to_bits(), 0, 0, 0];
        q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&entity));
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = primordial_soup::cells::Style::Cells;
        let mut results = Vec::new();
        for detailed in [false, true] {
            prefs.cells.detailed = detailed;
            q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&entity));
            let mut r = Renderer::new(d, &e, [256, 256])?;
            r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
            cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 1, -1.0)?;
            let c = r.cells.as_ref().unwrap();
            let values = floats(&tex(d, q, &c.density, c.grid, 4)?);
            let center = c.grid.map(|v| v / 2);
            let mut mass = 0.0f64;
            let mut variance = [0.0; 2];
            for (i, value) in values.iter().enumerate() {
                let offset = [
                    i as i64 % c.grid[0] as i64 - center[0] as i64,
                    i as i64 / c.grid[0] as i64 - center[1] as i64,
                ];
                let world = [
                    offset[0] as f64 * 2.0 * 2.0f64.sqrt() / c.grid[0] as f64,
                    offset[1] as f64 * 2.0 / 2.0f64.sqrt() / c.grid[1] as f64,
                ];
                mass += *value as f64;
                for j in 0..2 {
                    variance[j] += *value as f64 * world[j] * world[j];
                }
            }
            for v in &mut variance {
                *v /= mass;
                ensure!(
                    (*v / (1.0 / 128.0f64).powi(2) - 1.0).abs() < 0.05,
                    "Density world kernel drift at quality{detailed}: {variance:?}"
                );
            }
            let density_grid = c.grid;
            prefs.cells.chromatic = 0.5;
            prefs.cells.sharpen = 0.5;
            prefs.bloom_enabled = true;
            prefs.bloom_intensity = 0.5;
            let mut edge = entity;
            edge[1] = 0.7f32.to_bits();
            q.write_buffer(&e.entities, 0, bytemuck::cast_slice(&edge));
            cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 1, -1.0)?;
            let output = tex(d, q, r.clean_texture()?, [256, 256], 4)?;
            for (i, v) in output.chunks_exact(4).enumerate() {
                if i / 256 < 64 || i / 256 >= 192 {
                    ensure!(v == &output[..4], "Finishing/bloom leaked into letterbox");
                }
            }
            results
                .push(json!({"detailed":detailed,"grid":density_grid,"world_variance":variance}));
        }
        Ok(
            json!({"rectangular_world_kernel":results,"letterbox_mask_with_bloom_chromatic_and_sharpen":true}),
        )
    }
    fn cells_network_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let mut modes = Vec::new();
        for mode in [
            primordial_soup::network::Mode::Fourier,
            primordial_soup::network::Mode::Mlp,
            primordial_soup::network::Mode::Gru,
        ] {
            let mut p = Project::default();
            p.config_mut().switch_mode(mode, 0.25);
            p.config_mut().cohorts = 4;
            let mut e = Engine::new(
                d,
                q,
                Sizing {
                    canvas: [64, 64],
                    entity_count: 4096,
                    sqrt_world_size: 1.0,
                },
                &p,
                workgroup,
            )?;
            step(d, q, &mut e, &p, 120)?;
            let before = read(d, q, &e.entities, e.entities.size())?;
            let memory = read(d, q, &e.brain_memory, e.brain_memory.size())?;
            let trails = [
                tex(d, q, &e.canvas[0], e.sizing.canvas, 8)?,
                tex(d, q, &e.canvas[1], e.sizing.canvas, 8)?,
            ];
            let field = tex(d, q, e.field_texture(q), e.field_size, 8)?;
            let configs = read(d, q, &e.configs, e.configs.size())?;
            let frame = e.frame_count;
            let mut prefs = Preferences::default();
            prefs.cells.fast = false;
            prefs.cells.style = primordial_soup::cells::Style::Cells;
            let mut r = Renderer::new(d, &e, [256, 256])?;
            r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
            cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 2, -1.0)?;
            prefs.cells.theme = primordial_soup::cells::Theme::Light;
            cells_frame(d, q, &e, &mut r, &p, &prefs, &Camera::default(), 2, -1.0)?;
            ensure!(
                read(d, q, &e.entities, e.entities.size())? == before
                    && read(d, q, &e.brain_memory, e.brain_memory.size())? == memory,
                "Appearance changed {mode:?} simulation"
            );
            ensure!(
                frame == e.frame_count
                    && read(d, q, &e.configs, e.configs.size())? == configs
                    && tex(d, q, e.field_texture(q), e.field_size, 8)? == field,
                "Appearance changed coefficients, field or frame"
            );
            for i in 0..2 {
                ensure!(
                    tex(d, q, &e.canvas[i], e.sizing.canvas, 8)? == trails[i],
                    "Appearance changed vector trails"
                );
            }
            let mut classic = Renderer::new(d, &e, [256, 256])?;
            cells_frame(
                d,
                q,
                &e,
                &mut r,
                &p,
                &Preferences::default(),
                &Camera::default(),
                1,
                -1.0,
            )?;
            cells_frame(
                d,
                q,
                &e,
                &mut classic,
                &p,
                &Preferences::default(),
                &Camera::default(),
                1,
                -1.0,
            )?;
            ensure!(
                tex(d, q, r.clean_texture()?, [256, 256], 4)?
                    == tex(d, q, classic.clean_texture()?, [256, 256], 4)?,
                "Switching back did not restore Classic"
            );
            modes.push(mode);
        }
        Ok(
            json!({"unchanged_positions_velocities_trails_fields_coefficients_memory_and_frame":modes,"both_themes":true,"restored_classic_matches_fresh_renderer":true}),
        )
    }
    fn cells_wash_source(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        e: &Engine,
        p: &Project,
        r: &mut Renderer,
        prefs: &Preferences,
        size: [u32; 2],
        center: [f64; 2],
        peak: f64,
        camera: &Camera,
    ) -> Result<Vec<u8>> {
        let scale = size[0].min(size[1]) as f64 / 768.0;
        r.set_preferences(prefs);
        r.begin_frame(q, e, camera, p, 1, -1.0);
        let mut commands = d.create_command_encoder(&Default::default());
        r.clear_accumulator(&mut commands);
        q.submit([commands.finish()]);
        let cells = r.cells.as_ref().unwrap();
        let pad = [
            (cells.padded[0] - size[0]) as f64 * 0.5,
            (cells.padded[1] - size[1]) as f64 * 0.5,
        ];
        let mut source = vec![[0.0f32; 4]; (cells.padded[0] * cells.padded[1]) as usize];
        for y in 0..cells.padded[1] {
            for x in 0..cells.padded[0] {
                let dx = x as f64 + 0.5 - pad[0] - center[0];
                let dy = y as f64 + 0.5 - pad[1] - center[1];
                let mass = (peak * (-(dx * dx + dy * dy) / (8.0 * scale * scale)).exp()) as f32;
                source[(y * cells.padded[0] + x) as usize] = [mass * 0.2, mass * 0.65, mass, mass];
            }
        }
        q.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cells.images[0],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&source),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cells.padded[0] * 16),
                rows_per_image: Some(cells.padded[1]),
            },
            wgpu::Extent3d {
                width: cells.padded[0],
                height: cells.padded[1],
                depth_or_array_layers: 1,
            },
        );
        let mut commands = d.create_command_encoder(&Default::default());
        r.present(q, &mut commands, e, camera, prefs, Overlays::default());
        r.present_clean(d, q, &mut commands, e, camera, prefs);
        q.submit([commands.finish()]);
        wait(d)?;
        tex(d, q, r.clean_texture()?, size, 4)
    }
    /// Isolate the wash from sprites and simulation. These output-space gates
    /// reject the original square halo and a disabled/thresholded wash alike.
    fn cells_wash_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::{Style, Theme};
        let p = Project::default();
        let e = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: 1,
                sqrt_world_size: 1.0,
            },
            &p,
            workgroup,
        )?;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.relief = 0.0;
        prefs.cells.contrast = 0.0;
        prefs.cells.wash = 1.0;
        prefs.bloom_enabled = false;
        let mut results = Vec::new();
        for size in [[768, 768], [769, 577]] {
            let scale = size[0].min(size[1]) as f64 / 768.0;
            let mut r = Renderer::new(d, &e, size)?;
            r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
            for theme in [Theme::Dark, Theme::Light] {
                prefs.cells.theme = theme;
                for phase in [[0.25, 0.25], [0.75, 0.75], [7.75, 3.25]] {
                    let center = [
                        size[0] as f64 * 0.5 + phase[0],
                        size[1] as f64 * 0.5 + phase[1],
                    ];
                    let pixels = cells_wash_source(
                        d,
                        q,
                        &e,
                        &p,
                        &mut r,
                        &prefs,
                        size,
                        center,
                        80.0,
                        &Camera::default(),
                    )?;
                    let channel = if theme == Theme::Dark { 2 } else { 0 };
                    let pigment = if theme == Theme::Dark { 1.0 } else { 0.2 };
                    let background_index = ((8 * size[0] + 8) * 4) as usize;
                    let background = pixels[background_index + channel] as f64 / 255.0;
                    let normalization = pigment - background;
                    ensure!(
                        normalization.is_finite() && normalization != 0.0,
                        "Wash background normalization invalid: {background}, pigment {pigment}, {size:?} {theme:?} {phase:?}"
                    );
                    let alpha: Vec<f64> = pixels
                        .chunks_exact(4)
                        .map(|v| {
                            ((v[channel] as f64 / 255.0 - background) / normalization).max(0.0)
                        })
                        .collect();
                    let mut mass = 0.0;
                    let mut centroid = [0.0; 2];
                    for (i, a) in alpha.iter().enumerate() {
                        let xy = [
                            (i as u32 % size[0]) as f64 + 0.5,
                            (i as u32 / size[0]) as f64 + 0.5,
                        ];
                        let radius = (xy[0] - center[0]).hypot(xy[1] - center[1]);
                        if radius > 12.0 * scale && radius < 90.0 * scale {
                            mass += a;
                            for j in 0..2 {
                                centroid[j] += xy[j] * a;
                            }
                        }
                    }
                    ensure!(mass > 1.0, "Wash was removed");
                    for v in &mut centroid {
                        *v /= mass;
                    }
                    let error = (centroid[0] - center[0]).hypot(centroid[1] - center[1]) / scale;
                    ensure!(
                        error < 0.75,
                        "Wash drifts with its mip grid: {error} reference pixels, {size:?} {theme:?} {phase:?}"
                    );
                    let sample = |xy: [f64; 2]| {
                        let pos = xy.map(|v| v - 0.5);
                        let base = pos.map(|v| v.floor() as u32);
                        let f = pos.map(|v| v - v.floor());
                        let mut value = 0.0;
                        for y in 0..2 {
                            for x in 0..2 {
                                value += alpha[((base[1] + y) * size[0] + base[0] + x) as usize]
                                    * if x == 0 { 1.0 - f[0] } else { f[0] }
                                    * if y == 0 { 1.0 - f[1] } else { f[1] };
                            }
                        }
                        value
                    };
                    let mut rings = Vec::new();
                    for radius in [24.0, 32.0] {
                        let samples: Vec<_> = (0..72)
                            .map(|i| {
                                let a = i as f64 * std::f64::consts::TAU / 72.0;
                                sample([
                                    centroid[0] + radius * scale * a.cos(),
                                    centroid[1] + radius * scale * a.sin(),
                                ])
                            })
                            .collect();
                        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                        let spread = (samples.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                            - samples.iter().copied().fold(f64::INFINITY, f64::min))
                            / mean;
                        ensure!(
                            mean > if radius == 24.0 { 0.08 } else { 0.015 },
                            "Wash halo was suppressed"
                        );
                        // The faint outer ring has only a few 8-bit levels;
                        // bound its absolute opacity variation instead.
                        ensure!(
                            if radius == 24.0 {
                                spread < 0.2
                            } else {
                                spread * mean < 0.025
                            },
                            "Square wash halo: {spread} at r{radius}, {size:?} {theme:?} {phase:?}"
                        );
                        rings.push(json!({"radius_reference_pixels":radius,"mean_opacity":mean,"angular_relative_spread":spread}));
                    }
                    if size == [768, 768] && phase == [0.25, 0.25] {
                        let path = format!(
                            "artifacts/cells-wash-{}.png",
                            if theme == Theme::Dark {
                                "dark"
                            } else {
                                "light"
                            }
                        );
                        std::fs::write(&path, primordial_soup::export::encode_png(&pixels, size)?)?;
                    }
                    // Pending UI preferences cannot recolor a committed source's
                    // letterbox or selection. Clean capture keeps the held image.
                    let mut pending = prefs.clone();
                    pending.cells.theme = if theme == Theme::Dark {
                        Theme::Light
                    } else {
                        Theme::Dark
                    };
                    r.cells.as_mut().unwrap().highlight = 0.0;
                    let mut commands = d.create_command_encoder(&Default::default());
                    r.present(
                        q,
                        &mut commands,
                        &e,
                        &Camera::default(),
                        &pending,
                        Overlays::default(),
                    );
                    r.present_clean(d, q, &mut commands, &e, &Camera::default(), &pending);
                    q.submit([commands.finish()]);
                    wait(d)?;
                    ensure!(
                        tex(d, q, r.clean_texture()?, size, 4)? == pixels,
                        "Theme/highlight changed clean held output"
                    );
                    let selected = tex(d, q, r.output_texture(), size, 4)?;
                    ensure!(
                        selected[..4] == pixels[..4],
                        "Selection changed the world border"
                    );
                    ensure!(
                        selected[background_index..background_index + 4]
                            == pixels[background_index..background_index + 4],
                        "Selection/letterbox uses pending theme"
                    );
                    let index = ((center[1].floor() as u32 * size[0] + center[0].floor() as u32)
                        * 4) as usize;
                    for j in 0..3 {
                        let expected = pixels[background_index + j] as f64
                            + (pixels[index + j] as f64 - pixels[background_index + j] as f64)
                                * 0.125;
                        ensure!(
                            (selected[index + j] as f64 - expected).abs() < 1.5,
                            "Selection theme mismatch"
                        );
                    }
                    results.push(json!({"size":size,"theme":theme,"phase":phase,"centroid_error_reference_pixels":error,"rings":rings}));
                }
            }
        }
        let mut edge_results = Vec::new();
        for theme in [Theme::Dark, Theme::Light] {
            prefs.cells.theme = theme;
            let mut images = Vec::new();
            let camera = Camera {
                zoom: 2.0,
                ..Default::default()
            };
            for (size, center) in [([768, 768], [-65.0, 384.25]), ([1024, 768], [63.0, 384.25])] {
                let mut r = Renderer::new(d, &e, size)?;
                r.commit_cells(r.prepare_cells(d, &e, &prefs.cells)?);
                images.push(cells_wash_source(
                    d,
                    q,
                    &e,
                    &p,
                    &mut r,
                    &prefs,
                    size,
                    center,
                    1_000_000.0,
                    &camera,
                )?);
            }
            let mut max_error = 0u8;
            let mut signal = 0u8;
            for y in 0..768 {
                for x in 0..64 {
                    for channel in 0..3 {
                        let a = images[0][(y * 768 + x) * 4 + channel];
                        let b = images[1][(y * 1024 + x + 128) * 4 + channel];
                        max_error = max_error.max(a.abs_diff(b));
                        signal = signal.max(a.abs_diff(images[0][channel]));
                    }
                }
            }
            ensure!(signal > 2, "Edge fixture lost its exterior wash source");
            ensure!(
                max_error <= 2,
                "Viewport clipped the wash: {max_error} channel levels"
            );
            edge_results.push(json!({"theme":theme,"source_center_outside_view_pixels":65,"peak_source_mass":1_000_000,"max_rgb_error_255":max_error,"max_signal_255":signal}));
        }
        Ok(
            json!({"compact_round_sources":results,"viewport_crop_invariance":edge_results,"theme_correct_cached_clean_and_selected_output":true}),
        )
    }
    // Frozen uniform grain, a compact cluster and a narrow strand share the
    // same population normalization. No simulation steps between comparisons.
    fn cells_structure_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::{Style, Theme};
        let mut project = Project::default();
        project.world.boundary_conditions = 1;
        let engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: 65536,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        let mut particles = vec![[0u32; 8]; 65536];
        for (i, p) in particles.iter_mut().enumerate() {
            let (x, y) = if i < 32768 {
                (
                    -1.0 + 2.0 * ((i % 256) as f32 + 0.5) / 256.0,
                    -1.0 + 2.0 * ((i / 256) as f32 + 0.5) / 128.0,
                )
            } else if i < 49152 {
                let j = i - 32768;
                (
                    0.45 + 0.15 * ((j % 128) as f32 / 127.0 - 0.5),
                    -0.4 + 0.15 * ((j / 128) as f32 / 127.0 - 0.5),
                )
            } else {
                let j = i - 49152;
                (
                    0.1 + 0.7 * (j % 512) as f32 / 511.0,
                    0.4 + 0.012 * ((j / 512) as f32 / 31.0 - 0.5),
                )
            };
            p[0] = x.to_bits();
            p[1] = y.to_bits();
            p[2] = 0.001f32.to_bits();
            p[4] = 0.001f32.to_bits();
            p[7] = 0.2f32.to_bits();
        }
        q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
        let before = read(d, q, &engine.entities, engine.entities.size())?;
        let memory = read(d, q, &engine.brain_memory, engine.brain_memory.size())?;
        let size = [768, 768];
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = Style::Cells;
        prefs.bloom_enabled = false;
        prefs.cells.wash = 0.0;
        let mut renderer = Renderer::new(d, &engine, size)?;
        renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
        let mut results = Vec::new();
        let mut dead = particles.clone();
        for particle in &mut dead {
            particle[5] = u32::MAX;
        }
        for theme in [Theme::Dark, Theme::Light] {
            prefs.cells.theme = theme;
            q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&dead));
            cells_frame(
                d,
                q,
                &engine,
                &mut renderer,
                &project,
                &prefs,
                &Camera::default(),
                1,
                -1.0,
            )?;
            let background_pixels = tex(d, q, renderer.clean_texture()?, size, 4)?;
            let background_index = ((8 * size[0] + 8) * 4) as usize;
            let bg = &background_pixels[background_index..background_index + 3];
            q.write_buffer(&engine.entities, 0, &before);
            let mut signals = Vec::new();
            let mut base_mass = None;
            for threshold in [0.0, 0.4, 1.0, 4.0] {
                prefs.cells.structure_threshold = threshold;
                cells_frame(
                    d,
                    q,
                    &engine,
                    &mut renderer,
                    &project,
                    &prefs,
                    &Camera::default(),
                    1,
                    -1.0,
                )?;
                let pixels = tex(d, q, renderer.clean_texture()?, size, 4)?;
                ensure!(
                    pixels == tex(d, q, renderer.output_texture(), size, 4)?,
                    "Threshold clean/display mismatch"
                );
                let cells = renderer.cells.as_ref().unwrap();
                let mass = tex(d, q, &cells.images[cells.active], cells.padded, 16)?;
                if let Some(base) = &base_mass {
                    ensure!(*base == mass, "Threshold changed accumulated mass");
                } else {
                    base_mass = Some(mass);
                }
                let energy = |rect: [usize; 4]| -> f64 {
                    let mut sum = 0u64;
                    for y in rect[1]..rect[3] {
                        for x in rect[0]..rect[2] {
                            for c in 0..3 {
                                sum += pixels[(y * 768 + x) * 4 + c].abs_diff(bg[c]) as u64;
                            }
                        }
                    }
                    sum as f64
                };
                let sparse = energy([32, 32, 320, 736]);
                // Subtract dispersed-grain energy so removal around a structure
                // does not count as loss of the structure itself.
                let dispersed = sparse / (288.0 * 704.0);
                let cluster = energy([530, 510, 580, 565]) - dispersed * 50.0 * 55.0;
                let strand = energy([425, 225, 685, 237]) - dispersed * 260.0 * 12.0;
                let path = format!(
                    "artifacts/cells-structure-{}-{threshold}.png",
                    if theme == Theme::Dark {
                        "dark"
                    } else {
                        "light"
                    }
                );
                std::fs::write(&path, primordial_soup::export::encode_png(&pixels, size)?)?;
                signals.push([sparse, cluster, strand]);
                results.push(json!({"theme":theme,"threshold":threshold,"sparse_signal":sparse,"cluster_signal":cluster,"strand_signal":strand,"render":path}));
            }
            ensure!(
                signals[0].iter().all(|v| *v > 100.0),
                "Empty structure fixture"
            );
            ensure!(
                signals[1][0] > signals[0][0] * 0.05 && signals[1][0] < signals[0][0] * 0.95,
                "Threshold shoulder did not partially fade grain"
            );
            for signal in &signals[2..] {
                ensure!(
                    signal[0] < signals[0][0] * 0.15,
                    "Scattered grain not suppressed"
                );
                ensure!(signal[1] > signals[0][1] * 0.85, "Compact cluster lost");
                ensure!(
                    signal[2] > signals[0][2] * 0.85,
                    "Narrow strand lost: {:?} vs {:?}",
                    signal,
                    signals[0]
                );
            }
        }
        ensure!(
            read(d, q, &engine.entities, engine.entities.size())? == before,
            "Threshold changed particles"
        );
        ensure!(
            read(d, q, &engine.brain_memory, engine.brain_memory.size())? == memory,
            "Threshold changed memory"
        );
        ensure!(engine.frame_count == 0, "Threshold advanced simulation");
        Ok(
            json!({"renders":results,"clean_display_equal":true,"accumulated_mass_unchanged":true,"simulation_unchanged":true}),
        )
    }
    fn cells_membrane_fixtures(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::Style;
        let mut project = Project::default();
        project.world.boundary_conditions = 1;
        let n = 65_537; // The second contact is in a different sprite batch.
        let engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: n,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        let size = [512, 512];
        let camera = Camera {
            zoom: 100.0,
            ..Default::default()
        };
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.density = 0.0;
        prefs.cells.wash = 0.0;
        prefs.cells.relief = 0.0;
        prefs.bloom_enabled = false;
        let mut renderer = Renderer::new(d, &engine, size)?;
        renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
        let make_particles = |positions: &[[f32; 2]]| {
            let mut particles = vec![[0u32; 8]; n as usize];
            for particle in &mut particles {
                particle[5] = u32::MAX;
            }
            for (j, pos) in positions.iter().enumerate() {
                let index = if j == 1 { n as usize - 1 } else { j };
                let p = &mut particles[index];
                p[0] = pos[0].to_bits();
                p[1] = pos[1].to_bits();
                p[2] = 0.001f32.to_bits();
                p[4] = 0.001f32.to_bits();
                p[5] = 0;
                p[6] = (j as f32 * 0.123).to_bits();
            }
            particles
        };
        let history = |renderer: &Renderer| -> Result<Vec<f32>> {
            Ok(floats(&read(
                d,
                q,
                &renderer.cells.as_ref().unwrap().membrane_history,
                renderer.cells.as_ref().unwrap().membrane_history.size(),
            )?))
        };
        let mut packed = Vec::new();
        for y in -7..=7 {
            for x in -7..=7 {
                let jitter = ((x * 17 + y * 31) as f32).sin() * 0.00012;
                packed.push([x as f32 * 0.0006 + jitter, y as f32 * 0.0006 + jitter * 0.7]);
            }
        }
        let mut results = Vec::new();
        for (name, positions) in [
            ("isolated", vec![[0.0, 0.0]]),
            ("contact", vec![[-0.0003, 0.0], [0.0003, 0.0]]),
            ("coincident", vec![[0.0, 0.0]; 32]),
            ("packed", packed),
            ("wrapped", vec![[0.9998, 0.0], [-0.9998, 0.0]]),
        ] {
            let particles = make_particles(&positions);
            q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(&particles));
            let camera = if name == "wrapped" {
                Camera {
                    pan: [0.9998, 0.0],
                    ..camera
                }
            } else {
                camera
            };
            cells_frame(
                d,
                q,
                &engine,
                &mut renderer,
                &project,
                &prefs,
                &camera,
                1,
                -1.0,
            )?;
            let h = history(&renderer)?;
            ensure!(
                h.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
                "Invalid membrane field: {name}"
            );
            let cells = renderer.cells.as_ref().unwrap();
            let pad = (cells.padded[0] - size[0]) / 2;
            let center = (((256 + pad) * cells.padded[0] + 256 + pad) * 2) as usize;
            if name == "isolated" || name == "coincident" {
                ensure!(
                    h[center + 1] < 0.01,
                    "Coincident/isolated interior became a membrane"
                );
                ensure!(
                    h.chunks_exact(2).any(|v| v[1] > 0.3),
                    "Missing exterior membrane"
                );
            }
            if name == "contact" {
                ensure!(h[center + 1] > 0.5, "Cross-batch contact membrane missing");
            }
            let mass = tex(d, q, &cells.images[cells.active], cells.padded, 16)?;
            let map = tex(d, q, &cells.maps[cells.displayed_map], cells.map_size, 16)?;
            let pixels = tex(d, q, renderer.clean_texture()?, size, 4)?;
            let path = format!("artifacts/cells-membrane-{name}.png");
            std::fs::write(&path, primordial_soup::export::encode_png(&pixels, size)?)?;
            prefs.cells.membranes = 0.0;
            cells_frame(
                d,
                q,
                &engine,
                &mut renderer,
                &project,
                &prefs,
                &camera,
                1,
                -1.0,
            )?;
            let cells = renderer.cells.as_ref().unwrap();
            ensure!(
                mass == tex(d, q, &cells.images[cells.active], cells.padded, 16)?,
                "Membranes changed raw mass"
            );
            ensure!(
                map == tex(d, q, &cells.maps[cells.displayed_map], cells.map_size, 16)?,
                "Membranes changed interaction map"
            );
            prefs.cells.membranes = 1.0;
            results.push(json!({"fixture":name,"render":path,"mass_and_map_unchanged":true}));
        }
        // Independent temporal oracle: shade two translated states separately,
        // then compare their mean with two actual sequential render samples.
        let states = [
            make_particles(&[[-0.0004, 0.0]]),
            make_particles(&[[0.0004, 0.0]]),
        ];
        let mut expected = Vec::new();
        for state in &states {
            q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(state));
            cells_frame(
                d,
                q,
                &engine,
                &mut renderer,
                &project,
                &prefs,
                &camera,
                1,
                -1.0,
            )?;
            expected.push(history(&renderer)?);
        }
        renderer.begin_frame(q, &engine, &camera, &project, 2, -1.0);
        for (sample, state) in states.iter().enumerate() {
            q.write_buffer(&engine.entities, 0, bytemuck::cast_slice(state));
            let mut commands = d.create_command_encoder(&Default::default());
            if sample == 0 {
                renderer.clear_accumulator(&mut commands);
            }
            renderer.sample(&mut commands, &engine, &camera);
            if sample == 1 {
                renderer.present(
                    q,
                    &mut commands,
                    &engine,
                    &camera,
                    &prefs,
                    Overlays::default(),
                );
                renderer.present_clean(d, q, &mut commands, &engine, &camera, &prefs);
            }
            q.submit([commands.finish()]);
        }
        wait(d)?;
        let actual = history(&renderer)?;
        let error = actual
            .iter()
            .zip(expected[0].iter().zip(&expected[1]))
            .map(|(a, (b, c))| (a - (b + c) * 0.5).abs())
            .fold(0.0f32, f32::max);
        ensure!(error < 1e-6, "Moving membrane temporal error {error}");
        std::fs::write(
            "artifacts/cells-membrane-moving.png",
            primordial_soup::export::encode_png(
                &tex(d, q, renderer.clean_texture()?, size, 4)?,
                size,
            )?,
        )?;
        Ok(
            json!({"fixtures":results,"moving_sample_mean_max_error":error,"cross_batch_contact":true,"finite_fields":true}),
        )
    }
    fn cells_membrane_timing(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        engine: &Engine,
        project: &Project,
        anchor: [f32; 2],
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::Style;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.structure_threshold = 1.0;
        prefs.bloom_enabled = false;
        let mut renderer = Renderer::new(d, engine, [1920, 1080])?;
        renderer.commit_cells(renderer.prepare_cells(d, engine, &prefs.cells)?);
        let mut profiler = primordial_soup::profiling::GpuProfiler::new(d)
            .context("GPU timestamps unavailable")?;
        let mut cases = Vec::new();
        for zoom in [1.0, 32.0] {
            let camera = Camera {
                zoom,
                pan: if zoom > 1.0 { anchor } else { [0.0, 0.0] },
                ..Default::default()
            };
            for sharpen in [0.0, 0.35] {
                for membranes in [0.0, 1.0] {
                    prefs.cells.membranes = membranes;
                    prefs.cells.sharpen = sharpen;
                    renderer.set_preferences(&prefs);
                    let mut total = Vec::new();
                    let mut sprites = Vec::new();
                    let mut resolve = Vec::new();
                    for frame in 0..10 {
                        renderer.begin_frame(q, engine, &camera, project, 1, -1.0);
                        let mut commands = d.create_command_encoder(&Default::default());
                        let mut profile = profiler
                            .begin_frame(&mut commands)
                            .context("Membrane timing unavailable")?;
                        renderer.clear_accumulator(&mut commands);
                        renderer.sample_profiled(
                            &mut commands,
                            engine,
                            &camera,
                            Some(&mut profile),
                        );
                        renderer.present_profiled(
                            q,
                            &mut commands,
                            engine,
                            &camera,
                            &prefs,
                            Overlays::default(),
                            Some(&mut profile),
                        );
                        profile.finish(&mut commands);
                        q.submit([commands.finish()]);
                        profiler.after_submit(profile);
                        wait(d)?;
                        let sample = profiler
                            .poll(d, q)
                            .pop()
                            .context("Missing membrane timing sample")?;
                        if frame >= 2 {
                            total.push(sample.total_ms);
                            sprites.push(sample.stages_ms[6]);
                            resolve.push(sample.stages_ms[7]);
                        }
                    }
                    cases.push(json!({"zoom":zoom,"membranes":membranes,"sharpen":sharpen,"gpu_ms":percentiles(total),"sprites_and_membranes_ms":percentiles(sprites),"resolve_ms":percentiles(resolve)}));
                }
            }
        }
        Ok(
            json!({"size":[1920,1080],"population":engine.sizing.entity_count,"samples":1,"frames_per_case":8,"warmup_frames":2,"simulation_ticks_during_measurement":0,"cases":cases,"render_bytes":renderer.texture_bytes(),"note":"One executable, same frozen scene and allocations; membranes=0 skips distance and membrane work. GPU timestamps only; not simulation throughput or browser FPS."}),
        )
    }
    fn cells_zoom_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::{Style, Theme};
        let mut project = Project::default();
        project.config_mut().cohorts = 32;
        project.config_mut().initial_conditions = 0;
        project.config_mut().hazard_rate = 0.0;
        let mut engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: 262144,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        for _ in 0..8 {
            step(d, q, &mut engine, &project, 120)?;
        }
        let entities = read(d, q, &engine.entities, engine.entities.size())?;
        let memory = read(d, q, &engine.brain_memory, engine.brain_memory.size())?;
        let first = floats(&entities[..32]);
        let size = [768, 768];
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = Style::Cells;
        prefs.cells.structure_threshold = 1.0;
        prefs.bloom_enabled = false;
        let mut renderer = Renderer::new(d, &engine, size)?;
        renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
        let mut results = Vec::new();
        let mut static_error = 0;
        for theme in [Theme::Dark, Theme::Light] {
            prefs.cells.theme = theme;
            for zoom in [1.0, 4.0, 12.0, 32.0, 100.0] {
                let camera = Camera {
                    zoom,
                    pan: if zoom > 1.0 {
                        [first[0], first[1]]
                    } else {
                        [0.0, 0.0]
                    },
                    ..Default::default()
                };
                cells_frame(
                    d,
                    q,
                    &engine,
                    &mut renderer,
                    &project,
                    &prefs,
                    &camera,
                    1,
                    -1.0,
                )?;
                let pixels = tex(d, q, renderer.clean_texture()?, size, 4)?;
                ensure!(
                    pixels == tex(d, q, renderer.output_texture(), size, 4)?,
                    "Zoom clean/display mismatch"
                );
                let path = format!(
                    "artifacts/cells-membranes-{}-{zoom:.0}.png",
                    if theme == Theme::Dark {
                        "dark"
                    } else {
                        "light"
                    }
                );
                std::fs::write(&path, primordial_soup::export::encode_png(&pixels, size)?)?;
                if zoom == 32.0 {
                    cells_frame(
                        d,
                        q,
                        &engine,
                        &mut renderer,
                        &project,
                        &prefs,
                        &camera,
                        4,
                        -1.0,
                    )?;
                    let blurred = tex(d, q, renderer.clean_texture()?, size, 4)?;
                    let error = pixels
                        .iter()
                        .zip(&blurred)
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    static_error = static_error.max(error);
                    ensure!(
                        error <= 1,
                        "Membrane static blur changed image by {error}/255"
                    );
                }
                results.push(json!({"theme":theme,"zoom":zoom,"render":path}));
            }
        }
        ensure!(
            entities == read(d, q, &engine.entities, engine.entities.size())?,
            "Membranes changed particles"
        );
        ensure!(
            memory == read(d, q, &engine.brain_memory, engine.brain_memory.size())?,
            "Membranes changed memory"
        );
        let fixtures = cells_membrane_fixtures(d, q, workgroup)?;
        let timing = cells_membrane_timing(d, q, &engine, &project, [first[0], first[1]])?;
        Ok(
            json!({"renders":results,"fixtures":fixtures,"timing":timing,"static_blur_max_channel_error_255":static_error,"simulation_unchanged":true,"cells_bytes":renderer.cells.as_ref().unwrap().total_bytes}),
        )
    }
    /// Resolve-stage regression fixtures: raw display captures (optionally
    /// diffed against a previous capture directory) and resolve timings.
    fn cells_resolve_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
        capture_dir: &str,
        baseline_dir: Option<&str>,
        timing_only: bool,
    ) -> Result<serde_json::Value> {
        use primordial_soup::cells::{Style, Theme};
        let mut project = Project::default();
        project.config_mut().cohorts = 4;
        let mut engine = Engine::new(d, q, sizing_for(1.0, 1.0)?, &project, workgroup)?;
        step(d, q, &mut engine, &project, 120)?;
        let entities = read(d, q, &engine.entities, engine.entities.size())?;
        let first = floats(&entities[..32]);
        let anchor = [first[0], first[1]];
        // The default project wraps; one fixture exercises the bounded path.
        let mut wrapped = project.clone();
        wrapped.world.boundary_conditions = 0;
        std::fs::create_dir_all(capture_dir)?;
        // (size, theme, zoom, sharpen, chromatic, structure, highlight, bounded)
        let mut cases = Vec::new();
        for theme in [Theme::Dark, Theme::Light] {
            for zoom in [1.0, 4.0, 32.0] {
                for (sharpen, chromatic) in [(0.0, 0.0), (0.35, 0.0), (0.0, 0.3), (0.35, 0.3)] {
                    cases.push((
                        [1920, 1080],
                        theme,
                        zoom,
                        sharpen,
                        chromatic,
                        0.0,
                        -1.0,
                        false,
                    ));
                }
            }
        }
        for (zoom, sharpen) in [(1.0, 0.0), (32.0, 0.35)] {
            cases.push((
                [1920, 1080],
                Theme::Dark,
                zoom,
                sharpen,
                0.0,
                1.0,
                -1.0,
                false,
            ));
        }
        for sharpen in [0.0, 0.35] {
            cases.push((
                [1920, 1080],
                Theme::Dark,
                1.0,
                sharpen,
                0.0,
                0.0,
                1.0,
                false,
            ));
        }
        cases.push(([1920, 1080], Theme::Dark, 1.0, 0.0, 0.0, 0.0, -1.0, true));
        cases.push(([1920, 1080], Theme::Dark, 4.0, 0.35, 0.3, 0.0, -1.0, true));
        for size in [[2560, 1440], [3840, 2160]] {
            for (theme, zoom) in [(Theme::Dark, 1.0), (Theme::Light, 32.0)] {
                for (sharpen, chromatic) in [(0.0, 0.0), (0.35, 0.0), (0.35, 0.3)] {
                    cases.push((size, theme, zoom, sharpen, chromatic, 0.0, -1.0, false));
                }
            }
        }
        let mut renderers: Vec<([u32; 2], Renderer)> = Vec::new();
        let mut results = Vec::new();
        let mut worst = 0u8;
        for (size, theme, zoom, sharpen, chromatic, structure, highlight, wrap) in
            cases.into_iter().filter(|_| !timing_only)
        {
            let mut prefs = Preferences::default();
            prefs.cells.fast = false;
            prefs.cells.style = Style::Cells;
            prefs.cells.theme = theme;
            prefs.cells.sharpen = sharpen;
            prefs.cells.chromatic = chromatic;
            prefs.cells.structure_threshold = structure;
            if !renderers.iter().any(|(s, _)| *s == size) {
                let mut r = Renderer::new(d, &engine, size)?;
                r.commit_cells(r.prepare_cells(d, &engine, &prefs.cells)?);
                renderers.push((size, r));
            }
            let renderer = &mut renderers.iter_mut().find(|(s, _)| *s == size).unwrap().1;
            let camera = Camera {
                zoom,
                pan: if zoom > 1.0 { anchor } else { [0.0, 0.0] },
                ..Default::default()
            };
            cells_frame(
                d,
                q,
                &engine,
                renderer,
                if wrap { &wrapped } else { &project },
                &prefs,
                &camera,
                1,
                highlight,
            )?;
            let pixels = tex(d, q, renderer.output_texture(), size, 4)?;
            let label = format!(
                "{}x{}-{}-z{zoom}-s{sharpen}-c{chromatic}-t{structure}-h{highlight}{}",
                size[0],
                size[1],
                if theme == Theme::Dark {
                    "dark"
                } else {
                    "light"
                },
                if wrap { "-bounded" } else { "" }
            );
            std::fs::write(format!("{capture_dir}/{label}.rgba"), &pixels)?;
            let diff = baseline_dir
                .map(|dir| -> Result<serde_json::Value> {
                    let old = std::fs::read(format!("{dir}/{label}.rgba"))?;
                    ensure!(old.len() == pixels.len(), "Baseline size mismatch");
                    let (mut max, mut sum, mut changed, mut over1, mut over2) =
                        (0u8, 0u64, 0u64, 0u64, 0u64);
                    for (a, b) in pixels.chunks_exact(4).zip(old.chunks_exact(4)) {
                        for c in 0..3 {
                            let e = a[c].abs_diff(b[c]);
                            max = max.max(e);
                            sum += e as u64;
                            changed += u64::from(e > 0);
                            over1 += u64::from(e > 1);
                            over2 += u64::from(e > 2);
                        }
                    }
                    worst = worst.max(max);
                    let n = (pixels.len() / 4 * 3) as f64;
                    Ok(json!({"max":max,"mean":sum as f64/n,"changed_fraction":changed as f64/n,"over_1_fraction":over1 as f64/n,"over_2_fraction":over2 as f64/n}))
                })
                .transpose()?;
            results.push(json!({"label":label,"diff":diff}));
        }
        let mut profiler = primordial_soup::profiling::GpuProfiler::new(d)
            .context("GPU timestamps unavailable")?;
        let mut timing = Vec::new();
        for (size, zoom, sharpen, chromatic) in [
            ([1920, 1080], 1.0, 0.0, 0.0),
            ([1920, 1080], 1.0, 0.35, 0.0),
            ([1920, 1080], 1.0, 0.0, 0.3),
            ([1920, 1080], 1.0, 0.35, 0.3),
            ([1920, 1080], 32.0, 0.0, 0.0),
            ([1920, 1080], 32.0, 0.35, 0.0),
            ([3840, 2160], 1.0, 0.0, 0.0),
            ([3840, 2160], 1.0, 0.35, 0.0),
        ] {
            let mut prefs = Preferences::default();
            prefs.cells.fast = false;
            prefs.cells.style = Style::Cells;
            prefs.cells.sharpen = sharpen;
            prefs.cells.chromatic = chromatic;
            if !renderers.iter().any(|(s, _)| *s == size) {
                let mut r = Renderer::new(d, &engine, size)?;
                r.commit_cells(r.prepare_cells(d, &engine, &prefs.cells)?);
                renderers.push((size, r));
            }
            let renderer = &mut renderers.iter_mut().find(|(s, _)| *s == size).unwrap().1;
            renderer.set_preferences(&prefs);
            let camera = Camera {
                zoom,
                pan: if zoom > 1.0 { anchor } else { [0.0, 0.0] },
                ..Default::default()
            };
            let (mut total, mut resolve, mut post) = (Vec::new(), Vec::new(), Vec::new());
            for frame in 0..43 {
                renderer.begin_frame(q, &engine, &camera, &project, 1, -1.0);
                let mut c = d.create_command_encoder(&Default::default());
                let mut profile = profiler
                    .begin_frame(&mut c)
                    .context("Profiler unavailable")?;
                renderer.clear_accumulator(&mut c);
                renderer.sample_profiled(&mut c, &engine, &camera, Some(&mut profile));
                renderer.present_profiled(
                    q,
                    &mut c,
                    &engine,
                    &camera,
                    &prefs,
                    Overlays::default(),
                    Some(&mut profile),
                );
                profile.finish(&mut c);
                q.submit([c.finish()]);
                profiler.after_submit(profile);
                wait(d)?;
                let sample = profiler.poll(d, q).pop().context("Missing GPU sample")?;
                if frame >= 3 {
                    total.push(sample.total_ms);
                    resolve.push(sample.stages_ms[7]);
                    post.push(sample.stages_ms[3] + sample.stages_ms[4]);
                }
            }
            timing.push(json!({"size":size,"zoom":zoom,"sharpen":sharpen,"chromatic":chromatic,"gpu_ms":percentiles(total),"resolve_ms":percentiles(resolve.clone()),"resolve_raw_ms":resolve,"bloom_assembly_ms":percentiles(post)}));
        }
        Ok(
            json!({"mode":"cells-resolve","population":engine.sizing.entity_count,"simulation_frame":engine.frame_count,"captures":capture_dir,"baseline":baseline_dir,"max_channel_difference_255":baseline_dir.map(|_|worst),"cases":results,"timing":timing}),
        )
    }
    fn cells_planner_boundaries(d: &wgpu::Device, q: &wgpu::Queue) -> Result<serde_json::Value> {
        use primordial_soup::gpu;
        let listed = gpu::shader("cellsPlan.wgsl").contains("LISTED_BATCHES");
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cells planner boundary contract"),
            layout: None,
            module: &gpu::module(d, "cellsPlan.wgsl"),
            entry_point: Some("plan_batches"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, size, usage| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let uniforms = buffer(
            "planner screen",
            176,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let settings = buffer(
            "planner settings",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let plan = buffer("planner counters", 16, storage);
        let indirect = buffer("planner indirect", 128, storage);
        let bindings = [(0, &uniforms), (4, &plan), (5, &settings), (9, &indirect)].map(
            |(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            },
        );
        let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planner boundary bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings,
        });
        let mut screen = [0u32; 44];
        screen[8] = 19.0f32.to_bits();
        screen[9] = 17.0f32.to_bits();
        q.write_buffer(&uniforms, 0, bytemuck::cast_slice(&screen));
        // Eleven particles split 4/4/3; two spare indirect slots must remain untouched.
        let cases = [
            ("compact_at_bound", 4u32, 10u32, 10u32, 9u32, 0u32),
            ("list_capacity_exact", 5, 10, 10, 9, 2),
            ("list_capacity_overflow", 5, 11, 10, 9, 1),
            ("empty_unsafe_bound", 5, 0, 10, 9, 2),
            ("benefit_equal", 5, 33, 40, 9, 1),
            ("benefit_one_below", 5, 32, 40, 9, 2),
            ("nonwrap_benefit", 5, 2, 10, 1, 2),
            ("nonwrap_equal", 5, 3, 10, 1, 1),
        ];
        let mut results = Vec::new();
        for (name, bound, entries, capacity, copies, candidate_mode) in cases {
            let mode = if listed {
                candidate_mode
            } else {
                candidate_mode.min(1)
            };
            q.write_buffer(
                &settings,
                0,
                bytemuck::cast_slice(&[4u32, 3, capacity, copies, 11, 4, 5, 0]),
            );
            q.write_buffer(
                &plan,
                0,
                bytemuck::cast_slice(&[bound, entries, u32::MAX, 0]),
            );
            q.write_buffer(&indirect, 0, bytemuck::cast_slice(&[0xfeedfacdu32; 32]));
            let mut encoder = d.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            q.submit([encoder.finish()]);
            let counters = read(d, q, &plan, 16)?;
            ensure!(
                counters == bytemuck::cast_slice::<u32, u8>(&[bound, entries, mode, 0]),
                "Planner mode/counters differ at {name}"
            );
            let mut expected = [[0xfeedfacdu32; 4]; 8];
            for (batch, original_count) in [4u32, 4, 3].into_iter().enumerate() {
                let instances = match mode {
                    0 => {
                        if batch == 0 {
                            entries
                        } else {
                            0
                        }
                    }
                    2 => entries,
                    _ => original_count * copies,
                };
                expected[batch] = [4, instances, 0, 0];
                expected[5 + batch] = if mode == 0 && batch > 0 {
                    [0, 1, 1, 0]
                } else {
                    [3, 3, 1, 0]
                };
            }
            let actual = read(d, q, &indirect, 128)?;
            ensure!(
                actual == bytemuck::cast_slice::<[u32; 4], u8>(&expected),
                "Planner draw/average/guard slots differ at {name}"
            );
            results.push(json!({"case":name,"bound":bound,"entries":entries,"capacity":capacity,
                "copies":copies,"mode":mode,"draw_args":&expected[..3],"average_args":&expected[5..],"guard_slots_unchanged":true}));
        }
        Ok(
            json!({"implementation":if listed {"listed-batches"} else {"original-batches"},
            "production_shader":"cellsPlan.wgsl:plan_batches","particle_batch_counts":[4,4,3],"cases":results}),
        )
    }
    fn cells_checks(
        d: &wgpu::Device,
        q: &wgpu::Queue,
        workgroup: u32,
    ) -> Result<serde_json::Value> {
        let planner_boundaries = cells_planner_boundaries(d, q)?;
        let wash = cells_wash_checks(d, q, workgroup)?;
        let fixtures = cells_fixture_checks(d, q, workgroup)?;
        let mapping = cells_mapping_checks(d, q, workgroup)?;
        let density = cells_density_checks(d, q, workgroup)?;
        let networks = cells_network_checks(d, q, workgroup)?;
        let mut project = Project::default();
        project.config_mut().cohorts = 4;
        project.config_mut().hazard_rate = 0.0;
        let mut engine = Engine::new(
            d,
            q,
            Sizing {
                canvas: [128, 128],
                entity_count: 65536,
                sqrt_world_size: 1.0,
            },
            &project,
            workgroup,
        )?;
        for _ in 0..8 {
            step(d, q, &mut engine, &project, 120)?;
        }
        let entities = read(d, q, &engine.entities, engine.entities.size())?;
        let memory = read(d, q, &engine.brain_memory, engine.brain_memory.size())?;
        let mut renderer = Renderer::new(d, &engine, [768, 768])?;
        let mut prefs = Preferences::default();
        prefs.cells.fast = false;
        prefs.cells.style = primordial_soup::cells::Style::Cells;
        renderer.commit_cells(renderer.prepare_cells(d, &engine, &prefs.cells)?);
        renderer.set_preferences(&prefs);
        let mut encoder = d.create_command_encoder(&Default::default());
        renderer.begin_frame(q, &engine, &Camera::default(), &project, 1, -1.0);
        renderer.clear_accumulator(&mut encoder);
        renderer.sample(&mut encoder, &engine, &Camera::default());
        renderer.present(
            q,
            &mut encoder,
            &engine,
            &Camera::default(),
            &prefs,
            Overlays::default(),
        );
        renderer.present_clean(d, q, &mut encoder, &engine, &Camera::default(), &prefs);
        q.submit([encoder.finish()]);
        wait(d)?;
        let pixels = tex(d, q, renderer.clean_texture()?, [768, 768], 4)?;
        std::fs::write(
            "artifacts/cells-prototype.png",
            primordial_soup::export::encode_png(&pixels, [768, 768])?,
        )?;
        ensure!(
            read(d, q, &engine.entities, engine.entities.size())? == entities,
            "Cells mutated particles"
        );
        ensure!(
            read(d, q, &engine.brain_memory, engine.brain_memory.size())? == memory,
            "Cells mutated brain memory"
        );
        let cells = renderer.cells.as_ref().unwrap();
        let counts = read(d, q, &cells.counts, cells.counts.size())?;
        let total: u64 = counts
            .chunks_exact(4)
            .map(|a| u32::from_le_bytes(a.try_into().unwrap()) as u64)
            .sum();
        ensure!(total == 65536, "Density lost particles: {total}");
        let image = floats(&tex(d, q, &cells.images[cells.active], cells.padded, 16)?);
        ensure!(image.iter().all(|v| v.is_finite()), "Nonfinite Cells image");
        let extra_bytes = cells.total_bytes;
        let first = floats(&entities[..32]);
        let mut visuals = Vec::new();
        for zoom in [1.0, 8.0, 32.0] {
            let camera = Camera {
                zoom,
                pan: if zoom == 1.0 {
                    [0.0, 0.0]
                } else {
                    [first[0], first[1]]
                },
                ..Default::default()
            };
            cells_frame(
                d,
                q,
                &engine,
                &mut renderer,
                &project,
                &prefs,
                &camera,
                1,
                -1.0,
            )?;
            let path = format!("artifacts/cells-zoom-{zoom:.0}.png");
            std::fs::write(
                &path,
                primordial_soup::export::encode_png(
                    &tex(d, q, renderer.clean_texture()?, [768, 768], 4)?,
                    [768, 768],
                )?,
            )?;
            visuals.push(path);
            if zoom == 8.0 {
                let mut light = prefs.clone();
                light.cells.theme = primordial_soup::cells::Theme::Light;
                cells_frame(
                    d,
                    q,
                    &engine,
                    &mut renderer,
                    &project,
                    &light,
                    &camera,
                    1,
                    -1.0,
                )?;
                std::fs::write(
                    "artifacts/cells-light.png",
                    primordial_soup::export::encode_png(
                        &tex(d, q, renderer.clean_texture()?, [768, 768], 4)?,
                        [768, 768],
                    )?,
                )?;
                let mut classic = Renderer::new(d, &engine, [768, 768])?;
                cells_frame(
                    d,
                    q,
                    &engine,
                    &mut classic,
                    &project,
                    &Preferences::default(),
                    &camera,
                    1,
                    -1.0,
                )?;
                std::fs::write(
                    "artifacts/cells-classic-comparison.png",
                    primordial_soup::export::encode_png(
                        &tex(d, q, classic.clean_texture()?, [768, 768], 4)?,
                        [768, 768],
                    )?,
                )?;
            }
        }
        Ok(
            json!({"scalar_count":total,"finite_accumulation":true,"unchanged_simulation":true,"extra_bytes":extra_bytes,"render":"artifacts/cells-prototype.png","visuals":visuals,"planner_boundaries":planner_boundaries,"wash":wash,"fixtures":fixtures,"mapping":mapping,"density":density,"networks":networks,"population":65536,"simulation_frame":engine.frame_count,"project":project}),
        )
    }
    pub fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().collect();
        let value = |key: &str, default: &str| -> String {
            args.iter()
                .position(|v| v == key)
                .and_then(|i| args.get(i + 1))
                .map_or(default.to_owned(), Clone::clone)
        };
        let benchmark_mode = args.iter().any(|v| v == "--benchmark");
        let capture_mode = args.iter().any(|v| v == "--capture-benchmark");
        let allocation_mode = args.iter().any(|v| v == "--allocation-control");
        let video_mode = args
            .iter()
            .any(|v| v == "--video-smoke" || v == "--cells-video");
        let require_nvidia = video_mode || args.iter().any(|v| v == "--require-nvidia");
        let world = value("--world", "1").parse()?;
        let steps = value("--steps", "30").parse()?;
        let frames = value("--frames", "12").parse()?;
        let output = args
            .iter()
            .position(|v| v == "--output")
            .and_then(|i| args.get(i + 1));
        let shot = args
            .iter()
            .position(|v| v == "--shot")
            .and_then(|i| args.get(i + 1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = if args.iter().any(|v| v == "--adapter") {
            let selector = value("--adapter", "");
            let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
            if let Ok(index) = selector.parse::<usize>() {
                adapters
                    .into_iter()
                    .nth(index)
                    .context("Adapter index out of range")?
            } else {
                adapters
                    .into_iter()
                    .find(|a| {
                        a.get_info()
                            .name
                            .to_lowercase()
                            .contains(&selector.to_lowercase())
                    })
                    .context("Adapter name not found")?
            }
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            }))?
        };
        let workgroup = value(
            "--workgroup",
            &primordial_soup::gpu::default_workgroup(&adapter.limits()).to_string(),
        )
        .parse()?;
        let info = adapter.get_info();
        ensure!(
            info.device_type != wgpu::DeviceType::Cpu,
            "Refusing CPU/software GPU fallback"
        );
        ensure!(
            !require_nvidia || info.vendor == 0x10de,
            "NVIDIA GPU required; selected {}",
            info.name
        );
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (d, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("bounded gpu-check"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))?;
        let errors = Arc::new(Mutex::new(Vec::<String>::new()));
        let captured = errors.clone();
        d.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            captured.lock().unwrap().push(e.to_string());
        }));
        let checks = if args.iter().any(|v| v == "--network-pair-contracts") {
            primordial_soup::gpu::audit::paired_contracts(
                &d,
                &q,
                args.windows(2)
                    .find(|pair| pair[0] == "--pair-shader")
                    .map(|pair| pair[1].as_str()),
            )?
        } else if args.iter().any(|v| v == "--network-contracts") {
            let mut checks = primordial_soup::gpu::audit::contracts(&d, &q, workgroup)?;
            checks["occupancy"] = primordial_soup::gpu::audit::occupancy_contracts(&d, &q)?;
            checks
        } else if args.iter().any(|v| v == "--network-evaluation") {
            primordial_soup::gpu::audit::evaluate(
                &d,
                &q,
                workgroup,
                &value("--evaluation-dir", "artifacts/network-evaluation"),
                primordial_soup::gpu::audit::Evaluation {
                    revision: value("--evaluation-revision", "3").parse()?,
                    ablation: value("--ablation", "0").parse()?,
                    world: value("--world", "0.05").parse()?,
                    ticks: value("--ticks", "2048").parse()?,
                },
            )?
        } else if args.iter().any(|v| v == "--field-check") {
            primordial_soup::field_validation::validate(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--cells-tiled-check") {
            cells_tiled_checks(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--cells-resolve") {
            cells_resolve_checks(
                &d,
                &q,
                workgroup,
                &value("--capture-dir", "artifacts/local/cells-resolve"),
                args.iter()
                    .position(|v| v == "--baseline-dir")
                    .and_then(|i| args.get(i + 1))
                    .map(String::as_str),
                args.iter().any(|v| v == "--timing-only"),
            )?
        } else if args.iter().any(|v| v == "--cells-zoom") {
            cells_zoom_checks(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--cells-structure") {
            cells_structure_checks(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--cells-timing") {
            cells_timing(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--cells") {
            cells_checks(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--networks") {
            network_checks(&d, &q, workgroup)?
        } else if args.iter().any(|v| v == "--numerical") {
            primordial_soup::numerical::validate(&d, &q, workgroup)?
        } else if allocation_mode {
            allocation_control(&d, &q, workgroup)?
        } else if capture_mode {
            capture_benchmark(&d, &q, workgroup)?
        } else if benchmark_mode {
            benchmark(
                &d,
                &q,
                world,
                steps,
                frames,
                workgroup,
                shot.map(String::as_str),
                false,
            )?
        } else if video_mode {
            video_smoke(&d, &q, workgroup)?
        } else {
            let results = smoke(&d, &q, workgroup)?;
            if let Some(path) = shot {
                benchmark(&d, &q, 0.05, 5, 1, workgroup, Some(path), false)?;
            }
            results
        };
        wait(&d)?;
        ensure!(
            errors.lock().unwrap().is_empty(),
            "GPU validation errors: {:?}",
            *errors.lock().unwrap()
        );
        let report = json!({"mode":if args.iter().any(|v|v == "--network-pair-contracts"){"network-pair-contracts"}else if args.iter().any(|v|v == "--network-evaluation"){"network-evaluation"}else if args.iter().any(|v|v == "--network-contracts"){"network-contracts"}else if args.iter().any(|v|v == "--field-check"){"field-check"}else if args.iter().any(|v|v == "--cells-tiled-check"){"cells-tiled-check"}else if args.iter().any(|v|v == "--cells-zoom"){"cells-zoom"}else if args.iter().any(|v|v == "--cells-structure"){"cells-structure"}else if args.iter().any(|v|v == "--cells-video"){"cells-video"}else if args.iter().any(|v|v == "--cells-timing"){"cells-timing"}else if args.iter().any(|v|v == "--cells"){"cells"}else if args.iter().any(|v|v == "--networks"){"networks"}else if args.iter().any(|v|v == "--numerical"){"numerical"}else if allocation_mode{"allocation-control"}else if capture_mode{"capture-benchmark"}else if benchmark_mode{"benchmark"}else if video_mode{"video-smoke"}else{"smoke"},"adapter":{"name":info.name,"vendor":info.vendor,"device":info.device,"backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type),"driver":info.driver,"driver_info":info.driver_info},"checks":checks,"uncaptured_errors":[]});
        let text = serde_json::to_string_pretty(&report)?;
        if let Some(path) = output {
            if let Some(p) = Path::new(path).parent() {
                std::fs::create_dir_all(p)?;
            }
            std::fs::write(path, &text)?;
        }
        println!("{text}");
        Ok(())
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::run()
}
