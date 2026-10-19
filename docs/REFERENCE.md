# Reference and acceptance

The reference is the studio at <https://fluoddity.com/>, inspected on
2026-10-14. The simulation is neural particle dynamics with indirect vector
trail interactions, rather than a cellular Game of Life or pairwise particle
forces.

Source: <https://github.com/aphid91/Fluoddity-Web>, revision
`52787528700d49aff80a798b1e2490e4d537ea2b` (2026-10-08), MIT, Jesse Gelders.
The live root loaded `main-CPEML8xm.js` (SHA256
`d783aaf6b294224d85028b8de97ac95a55060732d779a9e2c1300b8a15dd9adc`) and
`shareCapture-D1nQEW_9.js`. Every non-comment line of the downloaded source's
entity update, canvas update, common definitions, rule derivation, particle
camera, frame assembly and field brush occurs literally in these live assets.
The source, rather than its older README, defines current defaults and presets.

The target is the linked root studio. The separately routed sand application
is a different application. The studio's features must be assessed individually:

| Area | Required behavior |
| --- | --- |
| Dynamics | Exact Fourier rule, sensors, drag/force/strafe, gravity, hazard, vector deposition, trail decay/diffusion |
| Populations | 1/4/16 cohort shortcuts, arbitrary cohorts, initial conditions, fences, reroll, randomize, GPU pick/adopt |
| Interaction | Select, shove, walls, trails; brush modes/erase/field strength; zoom/pan; keyboard shortcuts |
| Display | Particles/trails camera, cohort colors, brightness, tone curve, bloom, temporal motion blur |
| Projects | All shipped presets, v8 migration/v9 files, local saves, import/export, undo/redo, checkpoints, share |
| Platform | Native Vulkan hardware execution; WebAssembly + browser WebGPU; explicit unsupported/device errors |
| Validation | Independent frozen fixtures, actual GPU readbacks, rendered comparisons, measured release performance, real browser controls |

Long chaotic trajectories can diverge between GPU compilers. Short controlled
steps, layout/packing, and unmodified shader source are numerical evidence;
emergent character and coordinate/render correctness require visual evidence.
Tests do not establish an absolute absence of bugs.

## Architecture

`config`/`coords`/`history` are portable pure Rust. `gpu` retains GPU-resident
particles, baked cohort rules, ordered deposit/drain passes and ping-pong trail
textures. `render` reproduces the camera, temporal accumulation and bloom/tone
pipeline. `app` shares controls and commands using egui; platform-specific
startup and file services are thin adapters.

One encoder submits the simulation and render frame. Each physics substep has
an independently aligned uniform slice. Large state never crosses the CPU on
the frame path. Timestamp queries are optional. Vulkan and WebGPU retain one
WGSL model; workgroup choices are benchmarked with equivalent workloads.

## Static hosting contract

Cloudflare serves HTML, JS and WASM. The visitor's GPU performs all simulation
and rendering; no Worker functions, server GPU, database, or paid compute are
required. Workers Static Assets and Pages both support this model. Every asset
must stay below 25 MiB; static asset requests are free and unlimited.

Sources verified on 2026-10-14:

- <https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/>
- <https://developers.cloudflare.com/workers/static-assets/>
- <https://developers.cloudflare.com/pages/platform/limits/>

