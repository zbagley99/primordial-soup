//! Fast Cells compact tile gathering; capacity failures retain sprite drawing.
use crate::gpu;
use wgpu::util::DeviceExt;

pub struct Tiled {
    pipelines: Vec<wgpu::ComputePipeline>,
    groups: Vec<Vec<wgpu::BindGroup>>,
    counts: wgpu::Buffer,
    state: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
    tiles: [u32; 2],
    blocks: u32,
    bytes: u64,
}
impl Tiled {
    /// Returns None on adapters that cannot support the bounded working set.
    /// `index_limit` is a number of u32 tile references (None uses 64 MiB).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: &wgpu::Device,
        uniform: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        plan: &wgpu::Buffer,
        sprites: &wgpu::Buffer,
        radii: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
        _accum: &wgpu::Buffer,
        _selected: &wgpu::Buffer,
        sites: &wgpu::Buffer,
        size: [u32; 2],
        capacity: u32,
        index_limit: Option<u32>,
        image_views: [&wgpu::TextureView; 2],
        mask_views: [&wgpu::TextureView; 2],
        average_slot: u32,
    ) -> Option<Self> {
        let limits = d.limits();
        if limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_size_y < 16
            || limits.max_compute_workgroup_storage_size < 3080
            || size.contains(&0)
        {
            return None;
        }
        let max_buffer = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let tiles = [size[0].div_ceil(8), size[1].div_ceil(8)];
        let tile_count = tiles[0].checked_mul(tiles[1])?;
        let blocks_n = tile_count.div_ceil(256);
        let entries = capacity.min(((128u64 << 20).min(max_buffer) / 48) as u32);
        let references = index_limit
            .unwrap_or(16 << 20)
            .min(((64u64 << 20).min(max_buffer) / 4) as u32)
            .max(1);
        if entries == 0
            || u64::from(tile_count) * 4 > max_buffer
            || tiles.iter().copied().max()? > limits.max_compute_workgroups_per_dimension
            || (entries + 1).div_ceil(256) > limits.max_compute_workgroups_per_dimension
            || blocks_n > limits.max_compute_workgroups_per_dimension
        {
            return None;
        }
        let make = |label, size| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            })
        };
        let records = make("Cells tiled records", u64::from(entries) * 48);
        let counts = make("Cells tiled counts", u64::from(tile_count) * 4);
        let offsets = make("Cells tiled offsets", u64::from(tile_count) * 4);
        let indices = make("Cells tiled indices", u64::from(references) * 4);
        let state = make("Cells tiled state", 32);
        let dispatch_args = make("Cells tiled dispatch args", 16);
        let blocks = make("Cells tiled block offsets", u64::from(blocks_n) * 4);
        let settings = [
            tiles[0],
            tiles[1],
            entries,
            references,
            tile_count,
            blocks_n,
            8192,
            average_slot,
        ];
        let config = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cells tiled settings"),
            contents: bytemuck::cast_slice(&settings),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let resources = [
            uniform, &config, particles, plan, sprites, radii, &records, &counts, &offsets,
            &indices, &state, &blocks, indirect, _accum, _selected, sites,
        ];
        // Auto layouts contain only resources statically reachable from each entry.
        let bindings: &[(&str, &[usize])] = &[
            ("setup", &[1, 3, 10, 20]),
            ("prepare", &[0, 1, 2, 3, 4, 5, 6, 7, 10]),
            ("scan", &[1, 7, 8, 10, 11]),
            ("scan_blocks", &[1, 10, 11]),
            ("add_offsets", &[1, 8, 11]),
            ("fill", &[1, 3, 6, 7, 8, 9, 10]),
            ("finalize", &[1, 3, 10, 12]),
            ("raster", &[0, 1, 6, 7, 8, 9, 10, 15]),
        ];
        let module = gpu::module(d, "cellsTiled.wgsl");
        let mut pipelines = Vec::new();
        let mut groups = Vec::new();
        for (entry, used) in bindings {
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let mut pipeline_groups = Vec::new();
            for active in 0..if *entry == "raster" { 2 } else { 1 } {
                let mut entries: Vec<_> = used
                    .iter()
                    .map(|&i| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: if i == 20 {
                            dispatch_args.as_entire_binding()
                        } else {
                            resources[i].as_entire_binding()
                        },
                    })
                    .collect();
                if *entry == "raster" {
                    for (binding, view) in [
                        (16, image_views[active]),
                        (17, mask_views[active]),
                        (18, image_views[active ^ 1]),
                        (19, mask_views[active ^ 1]),
                    ] {
                        entries.push(wgpu::BindGroupEntry {
                            binding,
                            resource: wgpu::BindingResource::TextureView(view),
                        });
                    }
                }
                pipeline_groups.push(d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(entry),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                }));
            }
            groups.push(pipeline_groups);
            pipelines.push(pipeline);
        }
        let bytes = [
            &records,
            &counts,
            &offsets,
            &indices,
            &state,
            &blocks,
            &config,
            &dispatch_args,
        ]
        .iter()
        .map(|b| b.size())
        .sum();
        Some(Self {
            pipelines,
            groups,
            counts,
            state,
            dispatch_args,
            tiles,
            blocks: blocks_n,
            bytes,
        })
    }
    /// Run after planner and the sample's site clear, before the original draw.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, active: usize) {
        encoder.clear_buffer(&self.counts, 0, None);
        encoder.clear_buffer(&self.state, 0, None);
        let dispatches = [
            Some([1, 1]), // Setup actual visible-entry dispatch arguments.
            None,         // Prepare.
            Some([self.blocks, 1]),
            Some([1, 1]),
            Some([self.blocks, 1]),
            None, // Fill.
            Some([1, 1]),
            Some(self.tiles),
        ];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cells tiled"),
            ..Default::default()
        });
        for ((pipeline, groups), n) in self.pipelines.iter().zip(&self.groups).zip(dispatches) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &groups[if groups.len() == 2 { active } else { 0 }], &[]);
            if let Some(n) = n {
                pass.dispatch_workgroups(n[0], n[1], 1);
            } else {
                pass.dispatch_workgroups_indirect(&self.dispatch_args, 0);
            }
        }
    }
    /// Four u32 values: overflow, references, enabled, maximum tile occupancy.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.bytes
    }
}
