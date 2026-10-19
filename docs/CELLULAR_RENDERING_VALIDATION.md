# Cells rendering validation

Generated reports and captures are local outputs under ignored `artifacts/`.
Paths below identify optional local evidence; these files are not distributed
with the repository.

2026-10-16. Optional rendering style implemented for native wgpu and browser
WebGPU. Classic remains the default. Cells defaults to Dark with normal particle hues; Light is selectable.
The artistic result is a first version;
it is not an exact reproduction of the reference application's renderer.

## Rendered evidence

Both native images below use the same 65,536-particle Starcrossed snapshot,
engine frame 960, camera zoom 8, and camera centered on the first live particle.
The complete project is retained in cells-validation.json (`artifacts/cells-validation.json`).

| Classic | Cells — Dark |
| --- | --- |
| Local capture (`artifacts/cells-classic-comparison.png`) | Local capture (`artifacts/cells-zoom-8.png`) |

The same frozen state also has a Light rendering (`artifacts/cells-light.png`).

Full world (`artifacts/cells-zoom-1.png`) and
32× zoom (`artifacts/cells-zoom-32.png`) show the continuous size response.
These change framing; they are not supposed to contain the same particles.

The compact-source regression reproduces the reported artifact directly:

| Before: coarse square wash | After: smooth Light wash |
| --- | --- |
| Local capture (`artifacts/cells-wash-before.png`) | Local capture (`artifacts/cells-wash-light.png`) |

The Dark fixture (`artifacts/cells-wash-dark.png`) uses the same source and
filter. Twelve GPU cases cover two themes, subpixel/coarse-grid translations,
and square/odd rectangular outputs. Maximum wash-centroid error is **0.136
reference pixels**, and the inner ring's maximum relative angular variation is
**13.71%**. Gates also require nonzero halo intensity, so removing the wash
cannot pass. The faint outer ring uses an absolute opacity bound to account for
8-bit output quantization.

A source 65 pixels outside a 768×768 viewport produces the same edge strip as
the corresponding crop from a 1024×768 viewport: **zero differing RGB levels**
in both themes. This verifies that wider filtering has sufficient overscan.

## Correctness checks

The optimized native application build and the WebAssembly build both succeed.
The staged web distribution validator, formatting check and diff checks pass.
Browser reload checks restore Cells and GRU preferences with no application
or console errors.

`cargo test --locked --lib --tests` passes all 81 tests, including seven Cells
numerical/persistence contracts, actual eframe RON SavedState reload/migration,
and validation of every new shader entry point.
The Cells GPU suite passes with no uncaptured validation errors on Vulkan:

| Check | Evidence / result |
| --- | --- |
| Independent occupancy | Dead slots excluded; opposing velocities still count separately; 65,536 live slots sum to 65,536 |
| Dense-pile precision | 1, 32, 4,096, 65,536 and 262,145 coincident live particles; integrated mass error below 0.60%; bounded pigment error; positive, increasing mass |
| Maximum-population arithmetic | CPU quantum bounds cover up to 2²⁴−1 slots; radius/phase/color oracle stays within 1% gates using bounded batches |
| Small output and fractional scale | 64/128px outputs plus fractional reference scales, including the prior 1436px failure case; normalized small kernels retain mass and respect the peak bound |
| Density aspect/quality | 2:1 world; grids 362×181 and 724×362; measured world variances around 6.03–6.08×10⁻⁵ versus target 6.10×10⁻⁵ |
| Temporal order | Four samples of a frozen state preserve color/mass relative to one sample; density is rebuilt with each sample |
| Clean capture | Highlighted clean output remains unhighlighted; clearing selection preserves the held temporal base; display/clean bytes agree without overlays |
| Inverse mapping | Adversarial coarse-map contraction oracle; GPU pick at an independently inverted point |
| Brush footprints | Circle and line erasers compare every field texel with an independent CPU inverse map: 32,768 checked texels; 27 distinguish full-footprint mapping from cursor-only mapping |
| Field overlay alignment | 65,536 pixels of an unsaturated field pattern agree with the displayed fine map within 1.02 / 255 RGB levels; 155,904 channel comparisons detect the stale-padding formula |
| Wash / themes | Twelve roundness/translation fixtures; exterior-source crop invariance; cached clean capture and selection use the committed theme |
| Boundaries | Wrapped/nonwrapped sprite behavior, exterior source beyond the sprite radius for filtering, finite minimum-zoom output, and theme background with Bloom/chromatic/sharpen enabled |
| Simulation isolation | Exact entity, both trail texture, paint field, coefficient and memory bytes plus frame count stay unchanged in Fourier, MLP and GRU |
| Classic restoration | Switching renderer preferences back to Classic matches a fresh Classic renderer byte-for-byte |
| Failure isolation | Scoped injected validation error during inactive resource preparation; active particle state survives; oversized dimensions rejected before activation |

Raw results: Cells (`artifacts/cells-validation.json`),
Classic regression (`artifacts/cells-classic-regression.json`),
network CPU/GPU oracles (`artifacts/cells-network-regression.json`),
numerical simulation regression (`artifacts/cells-numerical-regression.json`).

The dense GPU fixture reaches 262,145 particles, not 16 million. Larger counts
are covered by arithmetic bounds and scalar precision cases; this is not a
claim of measured maximum-population GPU performance.

## Image and video integration

Native video verification exports 60 frames at 160×128 with two physics steps
and two temporal samples per frame. Three uncompressed output comparisons match
an independent renderer exactly while preview pan/zoom and dimensions change.
The engine ends at frame 184, including 64 setup steps and exactly 120 recording
steps. Dark H.264 decoding has approximately **0.77 / 255 mean absolute RGB error**
relative to the final uncompressed frame. The preview switches to the opposite
theme during recording; the export still matches its frozen reference exactly.
The same checks also run in Light (`artifacts/cells-video-light-validation.json`),
with **1.41 / 255** mean absolute RGB decoding error. Startup, mapping and encoding
cancellation/restart cases also pass. See
cells-video-validation.json (`artifacts/cells-video-validation.json`),
video (`artifacts/cells-native-video.mp4`) and
decoded final frame (`artifacts/cells-native-video-last.png`).

Browser checks use a separate localhost origin. Observed behavior:

The initial prototype browser checks below predate the theme/filter correction;
the current correction is recorded separately below.

- Classic → Cells renders the pale pigment scene. Editing Cell Scale from
  1.5 to 2, resizing and navigating a paused view leave logical tick 640 unchanged.
- After pending work settles, status returns to `requested_ms=None`.
- Cells preferences survive reload. GRU switches transactionally and retains
  Cells; the paused Weave fixture remains at logical tick zero after initialization.
- A full-frame 923×923 clean PNG downloads without editor controls or ticking.
- A browser 512×512 recording completes 60 frames / 120 logical ticks, restores
  the previous paused state, releases export GPU resources, and returns to idle.
- The initial clip decoded as H.264, 60/1 fps, one second, 60 frames. Its binary
  and probe were superseded by the current theme-correction recording below;
  the original record (`artifacts/cells-browser-validation.json`) retains
  observed results and its original clean PNG.
- No browser console warnings/errors or application error status were observed.

The current correction also passes these browser checks:

- Existing prototype preferences load Cells with Dark selected and normal hues.
  Explicit Light survives a full reload with the GRU project preserved.
- The paired Dark PNG (`artifacts/cells-browser-dark.png`) and
  Light PNG (`artifacts/cells-browser-light.png`) are clean 2537×1284 captures
  of the same paused 30,000-particle, four-cohort Starcrossed state at tick 7500.
  Theme changes and zooming preserve that tick, with smooth wash around clumps.
- A new Dark browser video (`artifacts/cells-browser-video.mp4`) decodes to
  this frame (`artifacts/cells-browser-video-last.png`). It is H.264, 512×512,
  60 frames at 60 fps (probe (`artifacts/cells-browser-video-probe.json`)).
  Two physics steps and two temporal samples produce exactly
  120 new ticks (7500 → 7620), then restore pause, release export resources and
  settle to `requested_ms=None`.
- No console warnings/errors or application error status were observed.

Current results: browser theme checks (`artifacts/cells-theme-browser-validation.json`).

Browser status identifies the backend as `BrowserWebGpu`; these checks do not
establish a browser-specific adapter performance number. Browser codec error
was visually checked, not compared against an exact GPU readback oracle.

## Fixed-workload timing

Device: NVIDIA GeForce RTX 3090, Vulkan, driver 610.57.04. Commands use the native
`gpu-check` debug host; shaders run normally on the hardware GPU. Snapshot:
600,000 particles, Starcrossed with four cohorts, engine frame 120, default Cells
quality, Dark theme, original hues and full-world camera. Each case has three warmups and eight measured
frames. **No simulation steps occur during the timed frames.** Existing Bloom
and assembly are included for both styles.

| Output / temporal samples | Classic GPU median | Cells GPU median | Added work |
| --- | ---: | ---: | ---: |
| 1920×1080 / 1 | 0.55 ms | 6.39 ms | 5.84 ms |
| 1920×1080 / 4 | 1.88 ms | 20.91 ms | 19.03 ms |
| 3840×2160 / 1 | 0.58 ms | 11.06 ms | 10.49 ms |

The smoother filter and its larger overscan increase cost from the initial
prototype (5.98 ms / 190.17 MiB at 1080p, 9.76 ms / 754.59 MiB at 4K).
The proposed **3 ms added-work target is not met**. Most added cost is sprite
preparation, integer rasterization, clearing and per-batch float additions.
Density and final relief/wash are separate reported stages; the sprite stage
includes all its batches. CPU encoding medians for Cells are approximately
0.58 / 1.17 / 0.48 ms respectively. These numbers do not measure simulation ticks
per second, browser scheduling, end-to-end presentation FPS or energy use.

The fixed snapshot makes the comparison repeatable, not representative of every
rule, camera, concentrated population or host load. Other desktop/browser work
was not globally controlled. Raw medians/stages/storage are in
cells-timing.json (`artifacts/cells-timing.json`), with the
project (`artifacts/cells-timing-project.json`).

## Logical storage

| Output | Existing renderer | Cells-only additional | Combined renderer |
| --- | ---: | ---: | ---: |
| 1920×1080 | 46,995,368 B | 224,126,504 B (213.74 MiB) | 271,121,872 B |
| 3840×2160 | 187,984,328 B | 885,043,848 B (844.04 MiB) | 1,073,028,176 B |

Cells includes five integer screen buffers, two RGBA32F temporal images, two
R32F selection images, occupancy/density textures, wash pyramid, two coarse
float maps, final scene/mark targets, raster attachment, overscan and its frame
uniform. The estimate excludes driver heap/alignment, pipelines/bindings,
simulation buffers and clean screenshot staging. The old proposal's much smaller
RGBA16F estimate is superseded by these precision-preserving resources.

During preparation the prior renderer remains alive. Concurrent display,
recording, resize/quality changes and screenshot readback can therefore increase
peak allocation beyond either table row. Logical accounting is not a measured
process/device peak. Failed requests are cached; retry restores the requested
style and quality. Actual hardware OOM/device loss was not forced.

## Reproduction

```sh
cargo test --locked --lib --tests
cargo build --locked --bin gpu-check
./target/debug/gpu-check --cells --output artifacts/cells-validation.json
./target/debug/gpu-check --cells-timing --output artifacts/cells-timing.json
./target/debug/gpu-check --cells-video --output artifacts/cells-video-validation.json
./target/debug/gpu-check --cells-video --cells-light --output artifacts/cells-video-light-validation.json
./target/debug/gpu-check --output artifacts/cells-classic-regression.json
./target/debug/gpu-check --networks --output artifacts/cells-network-regression.json
./target/debug/gpu-check --numerical --output artifacts/cells-numerical-regression.json
cargo build --locked --release --bin primordial-soup
RUSTC="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc" bash scripts/build-web.sh
```

Run GPU checks serially. Video requires the existing native ffmpeg/NVENC path;
there is no software rendering fallback. The normal application remains built
from the same Rust/wgpu code for native and WebAssembly.

## Structure Threshold (2026-10-16)

The optional **Controls → Display → Structure Threshold** slider fades fine
pigment, edge shading and bloom marks using base-level world density. It leaves
wash, displacement, particle accumulation and simulation unchanged. Default 0
preserves the previous appearance; the range is 0–4, with 1–2 suggested initially.
Density is a clustering proxy; sparse connected strands can fade too.

`cargo run --locked --bin gpu-check -- --cells-structure --output
artifacts/cells-structure-validation.json` checks a frozen 65,536-particle scene
with dispersed grain, a compact cluster and a narrow strand. On the RTX 3090,
Vulkan, both themes pass at 0, 0.4, 1 and 4: 0.4 retains approximately half the
scattered signal; 1 and 4 remove it while retaining the dense structures. Clean
and display bytes match, accumulated mass is identical, particles and recurrent
memory are unchanged, and logical frame count stays zero. See
raw results (`artifacts/cells-structure-validation.json`) and the paired
Light before (`artifacts/cells-structure-light-0.png`) /
Light after (`artifacts/cells-structure-light-1.png`) renders.

Library, Cells math and WGSL validation tests pass (23 tests), including missing
field migration, nonzero JSON/RON round trips and invalid threshold clamping.
The web build passes using `RUSTC=$HOME/.cargo/bin/rustc` because the system
Rust installation lacks the Wasm standard library. An isolated Chrome preview
on port 8082 showed the slider changing from 0 to 1.5 while paused at tick zero,
then retaining 1.5 after reload; the application reported no error and settled
to `requested_ms=None`.

Strict Clippy remains blocked by existing project-wide lints (including
`possible_missing_else`, `collapsible_if` and `too_many_arguments`). Moving
motion-blur tails have not been visually checked: occupancy represents the last
temporal sample, while pigment averages samples.

## Zoom-dependent membranes (2026-10-16)

The paired browser wide view (`artifacts/cells-membranes-browser-wide.png`)
and browser close-up (`artifacts/cells-membranes-browser-close.png`) show the
same paused Starcrossed scene at tick 960: 150,000 particles, existing hues,
Structure Threshold 1.5 and Cell Membranes 1.0. The wide view retains the spiral
and compact body on a quiet background; wheel zoom reveals rounded cells,
contact walls and shaded interiors. Intermediate views and the return to Home
were inspected without an observed abrupt switch. Clean captures contain no
editor overlays. Tick 960 is unchanged throughout zooming and capture; the app
reports `error=false`. This was checked on a separate port 8082 origin. See
browser observations (`artifacts/cells-membranes-browser-validation.json`).

`gpu-check --cells-zoom --output artifacts/cells-membranes-validation.json`
checks both themes at 1×, 4×, 12×, 32× and 100×. Its native evolved grid-seeded
scene is a different fixture from the browser scene; the grid alone is not
visual evidence of a coherent organism. Additional isolated, contacting,
coincident, packed and wrapped fixtures check finite factors and unchanged raw
color/mass and interaction maps. Two contacting sites straddle a particle batch
boundary. A separately shaded two-state translation oracle matches the actual
two-sample membrane average exactly (maximum error zero). Four frozen samples
differ from one by at most one 8-bit channel level in the evolved fixture.
Raw native results (`artifacts/cells-membranes-validation.json`) include images
and all timing cases.

The pre-existing Cells GPU suite also passes: dense-pile mass/color precision,
four-sample normalization, selection/held output, world density, minimum zoom,
wrapping, letterbox finishing, picking/brush mapping, theme/wash fixtures,
Fourier/MLP/GRU simulation isolation and Classic restoration. Its temporary
captures were written under `/tmp/primordial-cells-membranes/`; the retained
regression report (`artifacts/cells-membranes-regression.json`) therefore has
image paths relative to that temporary run, not new repository captures.
Library, Cells math and WGSL tests pass (23 tests), including nonzero membrane
JSON/RON persistence. The WebAssembly build passes.

The 1080p timing cases use one executable, identical allocations, a frozen
262,144-particle scene, one sample, two warmups and eight measured frames per
case, with Cell Membranes off/on and sharpening 0/0.35. Both membrane work and
post resolve are measured separately. The shared GPU and short cases show
substantial variance, including the initial wide baseline, so they establish
an observed working path and do not establish a reliable speedup, universal
cost, simulation throughput or browser FPS. New buffers add 43.4 MiB at 1080p;
full renderer logical allocation in this run is 316,629,792 bytes. Wide zoom
skips membrane computation but retains those allocations.

Resemblance to the target look (soft, translucent cells with visible walls) is
approximate. Dense regions can remain
crowded, and colors still mix locally. One physical RTX 3090 was tested through
native Vulkan and browser WebGPU; no second physical device was tested.


## Fast Cells and the 5 ms target (2026-10-17)

The retained optimization reaches **4.343 ms mean / 5.122 ms pooled p95 at
zoom 8**, and **4.120 ms mean / 4.893 ms pooled p95 at zoom 32**. The strict
p95 ≤ 5 ms gate passes all three zoom-32 repetitions, but only one zoom-8
repetition. Zoom 8 still needs margin; these results establish a near-5 ms
GPU scene, not a guarantee that every frame or every scene finishes within 5 ms.

The target uses Cells, World Size 1, Physics Rate 15 (15 ticks per frame),
600,000 Starcrossed particles, a 1024-square trail canvas, bloom, one requested
and actual temporal sample, and a 1920 × 1080 framebuffer. Working size remains
1920 × 1080, padded to 2158 × 1318. No resolution, particle count or tick count
was reduced. Calibration is disabled. Setup advances 120 ticks, selects the
mean position in the densest 64 × 64 world bin, then fixes the camera. Thirty
warmup frames precede 300 measured frames, spanning ticks 570–5070 and 4,500
measured physics ticks per run.

Measurements ran serially under `/tmp/gpu-timing.lock`, with normal
CPU scheduling, on RTX 3090 / native Vulkan / driver 610.57.04. Both versions
use the same total-only FrameTimer boundaries. Each row pools three 300-frame
runs; the last column preserves the individual-run p95 range. The entry
baseline includes the local optimizations already present when this round
started. It is a retained executable, not the older historical frozen Cells
fixture above. A fresh entry-binary control interleaved with the final runs
reproduced 6.443/8.664 ms mean/p95 at zoom 8 and 5.191/6.247 ms at zoom 32.

| Complete GPU scene | Mean ms | p50 ms | Pooled p95 ms | Individual p95 range ms |
| --- | ---: | ---: | ---: | ---: |
| Entry baseline, zoom 8 | 6.445 | 6.363 | 8.767 | 8.666–8.878 |
| Fast Cells, zoom 8 | 4.343 | 4.278 | 5.122 | 4.945–5.158 |
| Entry baseline, zoom 32 | 5.226 | 5.302 | 6.240 | 6.183–6.318 |
| Fast Cells, zoom 32 | 4.120 | 4.053 | 4.893 | 4.839–4.921 |

Mean GPU time falls 32.6% at zoom 8 and 21.2% at zoom 32. GPU scene timestamps
exclude query readback, egui composition, surface presentation and scheduling.
CPU encoding remains separate in every raw report; this is not a browser FPS
or input-to-display latency claim. Only one physical GPU/backend was tested.

The approach and retained changes are:

1. Fix the workload and retain an entry executable; measure combined,
   physics-only and frozen render-only controls before changing kernels.
2. Gather visible Cells through bounded 8 × 8 screen tiles, retaining the
   original fixed-point quantum, selection sums and nearest-two site keys.
   Fuse color/mass averaging into that gather, dispatch preparation/fill from
   the actual visible-entry count, and place the eight compute dispatches in
   one compute pass. Capacity, occupancy and adapter failures retain raster
   fallback. Single-sample raster averaging also skips cleared history reads.
3. Skip painted-field texture reads only while a fresh engine's field is
   provably untouched. Paint or raw texture access permanently restores reads,
   including prepared uniform slots; clear/reset never re-arm the shortcut.
   Camera input remains independent. The same-binary forced-active zero-field
   control measured 5.050/6.012 ms combined mean/p95 and 3.171/4.059 ms
   physics-only, versus 2.440/3.143 ms physics-only with the shortcut.
4. Require numerical, state, fallback and image checks, then repeat total-only
   measurements and discard attempts without useful gains. Plain integer
   average reads/writes were slower; grouped trail deposits passed exactness
   checks but added time; workgroup 128 and a zero-hazard guard showed no useful
   target-workload gain. Those experiments are not retained. The 8 × 8 tile
   choice improved frozen dense rendering to roughly 2.61 ms mean; the prior
   16 × 16 fused version measured roughly 2.95–2.97 ms. This frozen tick-120
   control has a different state from the evolving combined case, so its cost
   must not be subtracted from combined timings.

Fast Cells is enabled by default, including older saved preferences. An
explicit disabled choice survives serialization. Controls → Display → Advanced
exposes **Fast Cells**; disabling it retains the original interpolated raster
appearance. Renderer replacement, pending-result rejection, retry and failure
rollback all track the preference.

The accepted visual tradeoff is analytical pixel distance instead of hardware
interpolation at cell edges. Across 42 separate compatibility captures, the
largest per-capture mean absolute channel difference was 0.0457/255; up to
4.30% of channels changed, mostly by one or less. At most 0.1546% of channels
changed by more than one, while rare individual channels differed by as much
as 63/255. This is not pixel identity with the original appearance. These small
visual changes were accepted because the Cells appearance stays comparable and
the numerical checks pass. See the compatibility report (`artifacts/local/perf-5ms-round1/shared-raster-resolve.json`).

Strict gather validation uses an explicitly shared analytical raster reference:
all 32 cases passed with zero color/mass/selection, membrane-history and final
pixel error, exact site keys, and exact forced capacity fallback. This does not
replace the separate original-arithmetic compatibility measurement. The
production Cells suite also passes selection transitions, temporal samples,
precision, mapping and density checks, and the zoom suite passes both themes.
The 31 pristine-field cases compare particle, splat, both canvas, recurrent
memory and tick state byte-for-byte, including prepared/encoded ordering,
paint/upload, clear/reset, positive camera input and network replacement.
The independent six-case f64 numerical reference also passes within its stated
limits. There were no uncaptured GPU validation errors.

The final target captures are byte-identical to the earlier fused, consolidated
and smaller-tile captures. A transient preview inconsistency was resolved with
full file hashes and independent PNG decoding; the image audit (`artifacts/local/perf-5ms-round1/image-integrity-audit.json`)
records identical content. Final rendered frame (`artifacts/local/perf-5ms-round1/final-a-combined8-1920x1080.png`).

The memory cost is material: logical renderer allocation rises from
382,979,736 to 584,663,476 bytes, an additional 201,683,740 bytes (192.3 MiB)
in this case. Simulation allocation remains 38,337,728 bytes. These include
logical buffers/textures, not driver allocation overhead. The GPU-bounded
record/index working set is reused; it does not grow each frame.

All 99 Rust tests including doctests, WGSL validation, formatting/diff checks,
the native release checker build and the WASM library check pass. WASM used
an explicit matching rustup compiler and Cargo because the system toolchain
lacks its WASM standard library. Browser execution/performance, other GPUs,
painted-session target timing, and longer chaotic trajectories need separate
acceptance; painted sessions intentionally retain field-sampling cost.

The next approach for consistent zoom-8 p95 below 5 ms is to remove the remaining
zero-work fallback render/average pass overhead while preserving current-frame
fallback decisions, then measure dense entity-update spikes. Previous-frame
readbacks must not suppress required work. Retest the same fixed cases after
each change, plus painted fields and adapter/capacity fallback; do not buy the
remaining margin by lowering Physics Rate, World Size or framebuffer resolution.

Summary and per-run links (`artifacts/local/perf-5ms-round1/final-summary.json`),
final source/binary identity (`artifacts/local/perf-5ms-round1/final-source.json`),
field exactness (`artifacts/local/perf-5ms-round1/field-current.json`),
gather exactness (`artifacts/local/perf-5ms-round1/tiled-current.json`),
Cells regressions (`artifacts/local/perf-5ms-round1/cells-current.json`),
zoom regressions (`artifacts/local/perf-5ms-round1/zoom-current.json`),
independent numerical checks (`artifacts/local/perf-5ms-round1/numerical-current.json`).
Reproduction commands and measurement boundaries are in
[VALIDATION.md](VALIDATION.md#cells-5-ms-target-benchmark). Evidence under
`artifacts/local` is local to this working tree and is not a published artifact set.
