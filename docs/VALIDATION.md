# Validation protocol

Run CPU checks with `python3 scripts/low-load.py cargo test --locked -- --test-threads=1`.
The launcher constrains compilation, the linker, and their descendants to one
low-priority CPU. Run one GPU validation or benchmark process at a time.

## Independent numerical evidence

Frozen upstream Python/JavaScript fixtures are checked directly for coordinate
transforms, camera transforms, sizing, GPU packing and preset migration. The
original share codec and legacy link fixtures establish interchange. Shader
validation checks Naga parsing, validation and struct offsets. Interaction
fixtures pin shove strengths, line gestures and cohort confirmation.

`gpu-check --numerical --adapter NAME --output FILE.json` runs the production
particle and trail pipelines against independent f64 motion and five-tap trail
recurrences. A supplied nonzero Fourier rule and seeded particles/fields avoid
random rule generation. The fixtures disable stochastic deposits, hazard,
jitter, gravity, fences and live input to isolate those equations. Continuous
particle lanes use absolute tolerance 3e-5 over seven ticks, or 4e-3 over 1025
ticks; trails use 2e-3 to allow fp16 rounding. Discrete death/cohort outcomes are
exact, all relevant floating lanes must be finite, and batched versus serial
execution must agree on the same device. Dispatch tails and 64/128/256-thread
workgroups are covered where supported. This does not certify every shader
equation or prove the absence of races.

Repeat on each available device/driver/backend and retain each report's adapter
identity, error maxima and tolerances. Unavailable devices remain untested.
Long chaotic trajectories or matching screenshots after thousands of ticks are
not a cross-device correctness oracle. Tiny permitted floating-point differences
can amplify; statistical equivalence also requires separate evidence.

`tests/simulation_clock.rs` injects elapsed durations to check tick counts at
different rendering cadences, exact 0/1/N work across batches, retained overload
debt, mode changes, pause/resume and hidden/visible epochs. The application must
also run a low-display-rate acceptance case (300 ticks/sec at 1 FPS), proving
simulation-only batches are serviced independently of presentation.

## Actual device evidence

`target/debug/gpu-check --smoke` requires a hardware Vulkan adapter. It checks
uniform-ring versus serial stepping, finite state, trail drainage, all boundary
modes, resurrection, paused picking/adoption, independent painted channels and
synthetic camera gradients/motion. It does not activate a physical camera.

`target/release/gpu-check --benchmark --workgroup 256 --output artifacts/benchmark-256.json --shot artifacts/benchmark-256.png`
uses 600,000 particles, a 1024-square canvas, 30 physics steps, four requested
temporal samples, bloom and a 1280-by-720 output. The reference's floor-stride
schedule takes **five actual samples** in that case; the stride is seven steps.
Three warmup frames precede twelve
measured frames. Repeat serially for 64 and 128 threads. Report GPU timestamps
separately from CPU encoding and completed-frame intervals; do not convert a
submission rate into GPU FPS. Hardware, driver, backend, preset and settings
belong with every result. Other work on the GPU can affect measured timings.

The checker always disables calibration and records fixed workload settings.
Optional stage timestamps sum all entity updates, trail processing, sample
drawing/accumulation, bloom and final assembly. Total GPU work includes gaps and
other scene commands, but excludes query resolution/readback, egui composition,
surface presentation and frame scheduling. Portable empty compute-pass timestamp
boundaries add overhead. Compare instrumented runs only with matching methods,
and use `--benchmark --no-profile` as an uninstrumented control. Missing query
support produces unavailable GPU timing, never an invented GPU estimate.

CPU preparation, command encoding, submission call, completion wait and completed
frame time are separate. The historical CPU encoding interval included preparation
and submission; the new `cpu_encode_ms` excludes both. Add the three new CPU
components when comparing that historical interval. Submission duration does not
measure GPU completion. Native results do not establish browser performance.

Workgroup 256 remains the default when supported, with 128/64 fallbacks by limits.
Optional workgroups need matching numerical checks and repeated fixed-workload
measurements on the target device before being recommended as an optimization.
No hardware-specific mathematical fast path was added in this change.

For paused power scheduling, compare app update and scene-submission counters
over bounded equal observation windows at matching settings, first after all
async work settles and again after profiling drains. The new idle policy has no
recurring app timer. Pending maps, recording startup/finalization, active camera,
calibration, navigation input and deferred saves are explicit wakeup reasons.
Hidden tabs/minimized windows suspend simulation and queued exact work; loss of
focus alone does not. Drain pending captures even when hidden. Wakeup counts
establish scheduling behavior; energy savings and workday responsiveness require
their own measurements. Browser debug memory sampling is a separate five-second
JavaScript timer and is not a simulation repaint.

Current acceptance and remaining coverage are recorded in
[RESULTS.md](RESULTS.md#tick-controls-stage-profiling-and-idle-scheduling--2026-10-15).
The final Chrome fixture returned to idle after profiling and a paused capture.
Actual hidden-tab service completion, native minimize/restore and another GPU
still require manual acceptance. The 300-tick/sec browser run was constrained by
animation callback delivery and is not a passing target-rate result; the separate
60-tick/sec, 1-FPS browser case passed with zero retained lag.

`python3 scripts/low-load.py python3 scripts/benchmark-allocation-control.py`
uses one executable/device and four fresh simulations in eager/lazy/lazy/eager
order, with the same three warmups and twelve measurements per block. The eager
control allocates the unused clean texture in the constructor before bloom
textures and retains it without adding commands. Block resources are dropped
and work drained before the next block. Read-only `nvidia-smi` sampling records
active clocks, power and utilization at 100 ms intervals; it changes no GPU
settings. The script saves the command and executable/source hashes as well as
raw per-frame results. This tests allocation policy, not every difference from
a historical executable. Keep historical differences explicit if the original
binary or source snapshot is unavailable.

## Cells 5 ms target benchmark

Build `gpu-check` in release mode with normal scheduling:

```sh
cargo build --release --locked --offline --bin gpu-check
```

Run every GPU measurement and validation serially under the shared timing lock.
A busy lock must stop the command; wait for its owner before retrying. Do not
remove or bypass the lock. The target case uses Cells, World Size 1, Physics
Rate 15, one temporal sample, a 1920 × 1080 framebuffer and a fixed close view:

```sh
flock -n /tmp/gpu-timing.lock target/release/gpu-check \
  --benchmark --style cells --world 1 --steps 15 --blur-samples 1 \
  --zoom 8 --width 1920 --height 1080 --warmup 30 --frames 300 \
  --total-only --adapter NVIDIA --require-nvidia \
  --label fast-cells-zoom8 --output artifacts/local/cells-target-zoom8.json
```

Repeat with `--zoom 32` and distinct labels/output paths. Fast Cells is enabled
by default. Add `--reference-cells` for the retained raster path and its original
interpolated distance arithmetic. Keep the executable/source identity and all
other settings with each report; run three repetitions of a candidate and its
control to assess variance. The command uses normal scheduling; resource
throttling is an explicit choice and must be recorded if used.

`--total-only` uses the lightweight FrameTimer around the complete scene, with
two portable timestamp boundaries. Require timestamp-query support and all 300
GPU samples before judging the proposed p95 ≤ 5 ms target; also report p50 and
retain raw samples. Omitting `--total-only` enables detailed profiling, including
Cells planning, raster, averaging and membrane subspans. Those extra timestamp
boundaries affect diagnostic runs, so compare matching instrumentation. With
`--no-profile`, GPU timing is unavailable; CPU completion time cannot replace
it. Native RTX/Vulkan measurements cover encoded GPU work, excluding browser
presentation and input-to-display latency. Validate browser behavior separately.

The benchmark disables automatic calibration, uses the default project (or the
shipped preset named by `--preset`) and performs 120 setup ticks for Cells. It
selects the mean position of the densest 64 × 64 world bin, then holds that
camera fixed. The report records the project,
preferences, camera, actual working/padded/framebuffer sizes, requested/actual
temporal samples, planner/tile state and logical memory allocations. Include
the additional Fast Cells allocation in comparisons; driver heap overhead is
separate. Unsupported adapters or tile-capacity exhaustion retain the raster
path, so check actual tile state when attributing a speedup.

Add `--render-only` or `--physics-only` to the same command for diagnostic
controls, using distinct output paths. Combined and physics-only cases start
measurement at logical tick 570 after setup and 30 × 15 warmup ticks, then
complete 4,500 measured ticks. Render-only remains frozen at tick 120 throughout
warmup and measurement. Their states differ, so subtracting their times does
not isolate a rendering cost. Setup/tick-zero particle initialization and warmup
are outside every measured interval. CPU preparation, encoding, submission and
completion waiting are reported separately from GPU time.

Run the focused gather regression under the same lock:

```sh
flock -n /tmp/gpu-timing.lock target/release/gpu-check \
  --cells-tiled-check --adapter NVIDIA --require-nvidia \
  --output artifacts/local/cells-tiled-parity.json
```

Its 32 comparisons cover compact/dense/wrapped fixtures, highlight transitions,
temporal samples and forced capacity fallback on identical frozen particles.
The strict reference uses the same analytical pixel distances as Fast Cells;
it verifies gather arithmetic, color/mass/selection, membrane state and final
pixels against that explicit raster reference. It does not establish pixel
identity with the original interpolated raster arithmetic. Keep the separate
42-capture original-to-analytical compatibility comparison and its error maxima
with the results. Small cell-edge differences are an accepted Fast Cells
tradeoff; numerical and regression gates still apply. Users can disable Fast
Cells to retain the original appearance.

The physics path skips painted-field texture reads only while a newly created
engine's field is known to be pristine. Any paint or raw texture access
permanently enables sampling for that engine. Clearing a field, resetting the
simulation, or replacing its network does not rearm the shortcut. A positive
camera input remains independent of that painted-field state. Add
`--active-empty-field` to an otherwise identical benchmark command to force
sampling of the empty field as a control; keep its label/output distinct and
compare the recorded actual field activity. This control changes the sampling
path without painting content.

Validate this shortcut under the same serial lock:

```sh
flock -n /tmp/gpu-timing.lock target/release/gpu-check \
  --field-check --adapter NVIDIA --require-nvidia \
  --output artifacts/local/pristine-field-validation.json
```

The 31 cases require byte-exact same-device equality of entities, splat storage,
both trail canvases, brain memory, frame count and current canvas, together with
finite-state checks. The control explicitly samples its zero field. Coverage
includes four boundary modes, dead slots, a positive hazard rate and checkpoints
at 1, 7, 120 and 1,025 ticks; first wall/trail painting and raw uploads before or
after preparing substeps; encoded commands followed by a zero-step preparation
and first raw access; clear/reset persistence; positive camera-only input; and
MLP/GRU/Fourier replacement. Mutation and camera cases also require a nonzero
particle effect. These paired production-pipeline checks establish equivalence
of the shortcut on the tested device, rather than an independent physical model
or cross-device equality.

## Paired capture and memory measurements

`python3 scripts/low-load.py target/release/gpu-check --capture-benchmark --output artifacts/capture-before-after.json`
requires an NVIDIA Vulkan adapter. It renders and freezes one developed Tangle
image, then compares the previous full-window readback/crop algorithm against
the current direct cropped readback. Legacy code stays inside the benchmark.
Both paths read the mapped view directly, with no artificial padded-vector
copy. Three warmup pairs precede twelve alternating pairs for an offset crop
and a full frame. PNG byte equality is checked outside the timed interval.

Report transfer wall time (including allocation, submission and mapping), CPU
unpadding/cropping, PNG encoding and total separately. Preserve raw pairs and
report medians and means. Compare both order strata when caching changes their
costs. Explicit staging/pixel-buffer bytes are logical allocations, not RSS.
Native capture measurements cannot establish browser FPS or paused-frame delays.

For tab RAM, ask for Chrome's hover-card reading with matching settings, physical
canvas dimensions and no exports. The debug HUD's WASM linear capacity and JS
heap are separate component readings. Keep pre/post bundle manifests, and do
not interpret a smaller WASM download as an equivalent RAM reduction. Fresh
tabs may reuse a renderer process; one manual pair remains an observation.

## Browser acceptance

Build the actual wasm bundle, serve `dist` on localhost, and test the generated
HTML/JS/WASM in a hardware-accelerated browser. Confirm rendered particles,
both camera modes, preset loading, pause/reset, selection, independent fields,
save/reload, malformed imports, share/QR interchange, crop exports and video
output. Test a narrow viewport and the unsupported-WebGPU error surface.

Physical camera capture remains a separate manual check. Synthetic camera
readbacks establish the mapping equations, not a particular camera driver.

`node tests/camera_services.test.mjs` checks permission errors, discovery,
device selection and late start/stop callbacks with mocked devices. It never
activates a physical camera. `node tests/recording_services.test.mjs` checks
video lifecycle failures and discard during finalization, with mocks that
model the pinned muxer's graceful-close and force-cancel distinction. It also
remuxes a real H264 fixture through the pinned library and stops a second remux
on a small injected payload budget before adding the full clip.

`python3 scripts/low-load.py target/release/gpu-check --video-smoke --output artifacts/world-video-smoke.json`
records one second through the production fixed world renderer and NVENC
backend. It compares output pixels with an independent world renderer while
the preview pans, zooms, resizes and draws overlays, checks that simulation
advances only once, and discards/restarts during startup, mapping and encoding.
Probe and independently decode the saved clip to verify dimensions, rate,
duration, frame count and temporal variation. Repeat an actual browser export
through the visible UI; mocks do not establish WebCodecs acceptance. Verify
that starting from pause returns to pause and export allocations return to zero.

## Evidence status

Evidence is recorded after execution, in `docs/RESULTS.md`. Passing a compile
or a test suite alone does not establish rendered or browser behavior.

## Native live frame cadence

Use `cadence-check` to measure the production `SoupApp` native UI and render
loop through an actual visible surface. The headless GPU benchmark measures
GPU work independently and cannot establish live frame cadence. Build with
normal scheduling and run all GPU measurements under the shared lock:

```sh
cargo build --release --locked --offline --bin cadence-check
flock -n /tmp/gpu-timing.lock timeout 360s \
  target/release/cadence-check \
  --rates 1,15,30,60 --zooms 1,8,32 \
  --warmup-frames 30 --frames 180 --fullscreen \
  --output artifacts/local/cadence-16ms/native-final-fullscreen.json
```

Keep World Size 1, Cells with Fast Cells enabled, one temporal sample, and
the requested Physics Rate throughout each phase. Check the recorded actual
working extent: the fullscreen acceptance run uses 1920 × 1080 rendering.
Fullscreen uses the actual monitor; the requested window dimensions alone do
not establish the render extent, especially under a tiling window manager.
Do not silently lower the physics rate, world size or rendering resolution to
meet the cadence target. On a 60 Hz display the nominal refresh interval is
16.67 ms; report its distribution and missed refreshes rather than treating
that interval as a strict 16.00 ms result.

The helper uses fresh isolated persistence and filters physical input events,
including wheel and modifier changes, and hovered/dropped files. Visibility,
resize and native close metadata remain available. It closes its own window
after the bounded matrix. Preserve its JSON, executable/source provenance,
camera, actual extent, adapter and complete raw samples. Verify that the
camera remains at each requested zoom and pan throughout measurement.

Report UI callback intervals and app scene-update intervals separately,
including p50/p95/p99 and the count above 20 ms. Scene updates derive from
increases in the app's presentation counter; neither counter nor callback
timing is a hardware surface-presentation timestamp. Record actual scene
presentations, submissions and completed ticks. In the frame-based workload,
completed ticks must equal measured scene presentations multiplied by the
requested Physics Rate. An unchanged callback cadence can still hide skipped
scene updates, so also report app presentations per second.

The default frame-count protocol warms up and measures by scene presentations.
Its first scene interval starts at the warmup scene update. Optional
`--warmup-seconds` / `--seconds` measurement omits the first scene interval
because the initial callback may not update the scene. Measured elapsed time
ends at the last recorded UI callback start, excluding report-writing work.
GPU timing reports are asynchronous rolling snapshots and can repeat across
callbacks; do not aggregate them as independent GPU samples. Phases share a
continuously evolving simulation, so different phases are not equal-state
GPU A/B comparisons. Use the fixed-workload GPU protocol above for that claim.

Browser acceptance remains separate. Rebuild the WASM bundle, serve the newly
built `dist` at `localhost:8080`, and measure foreground `requestAnimationFrame`
cadence with the actual application and matching settings. A previously
observed roughly one-second cadence on an automated plain page is an
automation/environment confound, not browser acceptance or evidence of the
application's rendering cost. Native cadence evidence does not replace a
foreground browser measurement.
