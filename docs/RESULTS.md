# Executed validation

Generated reports and captures are local outputs under ignored `artifacts/`.
Paths below identify optional local evidence; these files are not distributed
with the repository.

Validated on 2026-10-14, on Linux with an NVIDIA GeForce RTX 3090,
Vulkan driver 610.57.04, Rust 1.98.1 and Chrome WebGPU. Compilation,
linking and command-line validation were confined to one CPU at niceness 19.
GPU runs were short and serial. This is recorded device evidence, not a
guarantee of identical performance on another machine or an absence of bugs.

## Native GPU performance

All three release measurements used the same workload: 600,000 particles,
a 1024 × 1024 trail canvas, 30 physics steps per frame, bloom and a
1280 × 720 output. Four requested temporal samples produce **five actual
samples** under the reference's floor-stride schedule. Three warmup frames
preceded twelve measured frames.

| Workgroup | Mean GPU ms | GPU p95 ms | Mean CPU encode ms | Mean completed frame ms |
| --- | ---: | ---: | ---: | ---: |
| 64 | 14.4923 | 15.7338 | 0.5508 | 15.1743 |
| 128 | 14.0109 | 14.9780 | 0.6438 | 14.7954 |
| 256 | 13.6826 | 14.5306 | 0.5550 | 14.3646 |

The default 256-thread workgroup was the fastest observed in this bounded
comparison. It processed about 1.32 billion particle updates per second.
GPU time uses timestamp queries around the complete simulation/render encoder;
CPU time measures command encoding, not a full GUI frame. Completed-frame
time includes waiting for GPU completion. These numbers do not establish a
browser frame rate or sustained full-editor performance. Other work on the GPU
can change them. Simulation allocations were 38,337,728 bytes; renderer textures
and driver heap overhead are additional.

Raw evidence: 64 (`artifacts/benchmark-64.json`),
128 (`artifacts/benchmark-128.json`),
256 (`artifacts/benchmark-256.json`),
rendered benchmark frame (`artifacts/benchmark-256.png`).
Every run recorded an empty uncaptured GPU-error list.

Two later checks with matching settings measured 14.4943 and 14.4638 ms mean
GPU time, about 5.7–5.9% above the historical 256-thread result. The old executable
was not retained, so those separate-session numbers cannot isolate a code change.
See first rerun (`artifacts/benchmark-256-after-optimization.json`) and
repeat (`artifacts/benchmark-256-after-repeat.json`).

To test the changed idle export-target allocation, one executable/device ran
four fresh Starcrossed simulations in **eager / lazy / lazy / eager** order,
each with the same three warmups and twelve measured frames. The eager target
was allocated immediately after the display output, before bloom textures,
and retained without export draws, clears or uploads. Submissions were drained
and each block's resources dropped before the next.

| Allocation policy | Mean GPU ms across 24 frames | Median GPU ms | Mean CPU encode ms | Mean completed frame ms |
| --- | ---: | ---: | ---: | ---: |
| Previous eager target | 13.7778 | 13.7339 | 0.5373 | 14.4495 |
| Current lazy target | 13.7244 | 13.6402 | 0.6325 | 14.4918 |

This bounded control found no GPU slowdown attributable to removing the unused
texture; it does not establish a rendering speedup. CPU encoding was 0.0952 ms
higher in the lazy blocks; read-only hardware sampling shared that low-priority
CPU, so no general CPU-encoding improvement is claimed. Active GPU clocks rose
from 1695 to 1965 MHz during the run. That demonstrates changing hardware state
in this run, without explaining the older timings. The historical 5.7–5.9%
difference remains unresolved. Evidence:
allocation control and raw frames (`artifacts/allocation-control.json`),
hardware samples (`artifacts/allocation-control-hardware.csv`), and
commands and binary/source hashes (`artifacts/allocation-control-run.json`).

## Measured memory and capture changes

The follow-up focuses on measured costs rather than matching the reference's
tab-memory figure. Chrome's hover card was read manually on fresh
tabs before and after removing WASM name/producers metadata: **71.3 MB → 67 MB**
(4.3 MB, about 6%). Both used the same localhost origin, paused Starcrossed,
300,000 particles, a 2537 × 1218 physical canvas and no exports. This is one
paired observation, not a controlled process-RSS measurement: a fresh tab can
reuse a browser renderer process. WASM linear memory was identical at 8,847,360
bytes; JS heap readings were approximately 8 MB in both cases. The tab reading
includes costs that neither diagnostic accounts for. Evidence:
before (`artifacts/memory-pre-strip.json`),
after (`artifacts/memory-post-strip.json`).

The web build also excludes unused WebGL/native backend dependencies. Its WASM
module fell from 14,011,005 bytes before these changes to **8,516,669 bytes**
in the manually measured build. The final constructor refactor for the native
allocation control adds 77 bytes, giving a final module of 8,516,746 bytes.
Removing metadata alone saved 2,104,398 bytes; all retained executable, data and
other sections were byte-identical. That file-size saving is distinct from RAM.
See measured-build section comparison (`artifacts/wasm-strip-comparison.json`),
final-build section comparison (`artifacts/wasm-strip-final-comparison.json`) and
web dependency tree (`artifacts/wasm-dependencies.txt`).

An idle clean-export target is now absent and is released after screenshot or
video work. At the measured canvas size this eliminates **12,360,264 logical
texture bytes** while idle. Driver allocation overhead and reclamation timing
are additional; this number is not a claim about tab RAM. Vulkan checks cover
release/recreation, resize and clean output equality. Browser checks observed
the allocation retained during a paused recording and returned to zero after
ending or completing recording. See
recording lifecycle (`artifacts/browser-recording-memory.json`).

PNG export now transfers only the selected rectangle and directly unpads that
mapped buffer. The earlier path transferred the full window, allocated its
full RGBA image and then allocated a second cropped image. A release benchmark
rendered Tangle once after 64 physics steps, froze its 2537 × 1218 clean image,
warmed both implementations for three pairs and measured twelve alternating
pairs per case. Both paths read the production mapped view directly and use
the same PNG settings. PNG bytes matched in every warmup and measured pair.

| Native capture case | Previous median total ms | Current median total ms | Previous/current staging bytes | Previous/current CPU pixel-buffer bytes |
| --- | ---: | ---: | ---: | ---: |
| 631 × 577 crop at (855, 278) | 4.2576 | 1.5386 | 12,472,320 / 1,477,120 | 13,816,612 / 1,456,348 |
| Full 2537 × 1218 frame | 14.7137 | 12.1350 | 12,472,320 / 12,472,320 | 24,720,528 / 12,360,264 |

The balanced batch showed about **64% less crop-export time** and **18% less
full-frame export time**. CPU unpadding/cropping medians were 0.9988 → 0.0925 ms
for the crop and 4.4758 → 2.0026 ms for the full frame. Transfer medians were
2.0048 → 0.3347 ms and 2.1440 → 2.1452 ms respectively; full-frame transfer is
unchanged because its copied area is unchanged. PNG encoding itself was similar.

Order affects the full-frame result. With the previous path first, median total
was 15.7141 → 10.8681 ms; with the current path first it was 13.9747 → 12.9917 ms.
The current path won all twelve pairs in each case, but the aggregate percentage
is specific to this balanced batch. Timings include native allocation, copy,
submission, mapping and PNG work on one low-priority CPU. Pixel-buffer counts
are explicit logical allocations, excluding PNG buffers and driver heaps;
these results do not establish browser FPS or total RSS. Raw samples, means,
medians and stage timings are in
paired capture benchmark (`artifacts/capture-before-after.json`).
The timed paths exclude rendering, QR stamping and file delivery.

Actual browser exports also exercised the new path: repeated 631 × 577
QR-stamped crops were byte-identical, independently decoded with `zbarimg`, and
returned the clean-export allocation to zero. WASM linear memory grew from
about 8.4 to 9.8 MiB and stayed at 9.8 MiB through repeat captures and recording;
that committed high-water capacity is not reduced by freeing individual buffers.
See crop (`artifacts/browser-cropped-readback.png`),
repeat evidence (`artifacts/browser-crop-repeat.json`) and
capture memory (`artifacts/browser-crop-memory.json`).

## Correctness and interchange

- **45 unit/integration tests and three executable doctests passed**, with
  no ignored tests. Coverage includes all 131 shipped presets and their menu categories, frozen upstream
  coordinate/layout/packing and v8 migration fixtures, source share codecs
  1–4 and legacy links, malformed data, independent ZIP input, crop/PNG/QR
  geometry, shader validation, interaction gestures, recording schedules,
  archive hashes/branches, calibration constants and focused-button hotkeys.
  Shifted commands retain their event-time modifiers even when Shift changes
  before the next GUI frame. Current-project rate search matches 393 frozen
  upstream decisions across synthetic machines, bounds and probe histories;
  settling, outlier rejection, ceiling and cancellation are checked separately.
  FPS measurement and color feedback match 833 frozen upstream frames, including
  250/80 ms windows, hysteresis, dwell, discontinuities and paused samples.
- **Five browser-service checks passed**: denied image clipboard falls back
  to PNG; blocked/failed project storage rejects; archive storage remains
  independent; pending archive writes finish in order before export; video
  selection prefers hardware and retries when the preference is rejected.
- The original TypeScript archive loader reconstructed three exported Rust
  states with matching source content hashes and no reconstruction failures.
  The fixed independent fixtures are in `tests/fixtures/`.
- Actual Vulkan readbacks passed: uniform-ring stepping matches serial
  stepping exactly; state remains finite; deposit splats drain; bounce, wrap,
  reset and kill boundaries work; reset resurrects; picking/adoption works
  while paused; walls/trails remain independent; synthetic webcam gradient
  and motion equations match. See GPU smoke (`artifacts/gpu-smoke.json`).
- Both the native release executable and the optimized WASM bundle built
  successfully. Browser startup initializes particles even when paused.

## Real exports

Native export used **h264_nvenc**, with a bounded GPU readback. FFprobe decoded
60 H.264 frames at 160 × 128, 60 fps and one second, with no uncaptured GPU
errors. See native evidence (`artifacts/video-smoke.json`) and
the video (`artifacts/native-nvenc-video.mp4`).

An actual Chrome export produced 60 H.264 frames at 128 × 128, 60 fps and
one second. Hardware preference was rejected in this browser session, so the
browser's default encoder was used; browser NVENC is **not** claimed.
See probe (`artifacts/browser-video.json`),
video (`artifacts/browser-video.mp4`) and
decoded first frame (`artifacts/browser-video-frame.png`).

The browser exported a freely positioned **630 × 577** crop with a share QR
stamp. Independent `zbarimg` decoded the generated PNG into a project link.
The capture contains the simulation and QR plate, with editor controls and
crop/brush overlays excluded. The same PNG was then selected through Chrome's
image picker: browser jsQR decoding and the Rust link decoder loaded the project
and displayed "Imported preset". See crop (`artifacts/browser-qr-crop.png`)
and browser import proof (`artifacts/browser-qr-import.jpg`).

## Browser acceptance and hosting

The final memory-optimized bundle was opened in Chrome, rendered Tangle and
was left paused. Its debug status reported no application error and zero idle
clean-export bytes. File → Load visibly contained Core (22), Custom (1) and
Archive (109), with the Core preset list expanded. This follow-up fixes the
category lookup that had hidden presets; archive scrolling, custom loading,
preview cancellation and Core loading were exercised earlier in the same pass.
See final menu proof (`artifacts/browser-final-presets-menu.jpg`),
full browser view (`artifacts/browser-final-presets.jpg`) and
final paused status (`artifacts/browser-final-preview.json`).

Chrome rendered developed Starcrossed particle/trail structures, and the
reference calibration ladder selected world size 1.0 and 20 physics steps
per frame. Pause, preset menus, project controls, screenshot crop, video export
and saving/reloading a custom project were exercised. A narrow viewport was
checked and the toolbar repaired to stay within it. The final editor adapts
the reference controls to egui rather than copying the website's DOM layout.

Final-bundle acceptance used a separate localhost origin at world size 0.1
(60,000 particles). Right-click tuning resumed a paused simulation, completed
and restored the pause; the physics counter advanced from 1 to 141, preserving
the live state. A second run started at rate 60 and was cancelled by another
right-click: rate 60 and the pause were restored, with the counter at 321.
Fast Shift+V opened the import window. No browser warnings or errors were
captured in that run. Evidence:
completion (`artifacts/browser-rate-complete.jpg`),
cancellation and import-window proof (`artifacts/browser-rate-cancel.jpg`), and
developed desktop view (`artifacts/browser-desktop.jpg`).

The generated `dist/` passed the file/type/size audit. It is assets-only and
below Cloudflare's 25 MiB per-file and 20,000-file Free Pages limits. The exact
sizes and hashes are in static manifest (`artifacts/static-dist.json`).
The current controls bundle has 15 files totalling 10,373,240 bytes;
its largest asset, the WASM module, is 8,531,504 bytes. Earlier stripping and
tab-memory observations above refer to the preceding build.
Cloudflare will only serve assets; simulation and storage execute on the
visitor's machine.

## Webcam clarity and whole-world video

The webcam panel is now **Webcam Input** and explains that camera edges or
motion become particle forces or trail signals. Device discovery lists Default
and available video inputs without activating capture. Missing devices,
permission denial and busy-device failures have separate, actionable messages.
Native discovery queries V4L2 capture capabilities instead of assuming video0;
a listed device is still not a guarantee that its requested capture mode works.
Browser discovery and the visible explanation were checked; capture stayed off.

Recording is video-only through **Share → Record World (Video)**. It uses a
fixed world camera, a separate output renderer and one backpressured readback.
It follows the preview's initial speed/blur and restores the prior pause state.
Project/world replacement is rejected during a clip. Finish early saves a
partial clip; discard is terminal and suppresses browser delivery even during
finalization. The pinned muxer cannot interrupt an already-started finalization.

Updated verification passed: 48 Rust unit/integration checks plus 3 doctests,
9 mocked webcam checks, 5 browser-service checks and 11 recording checks. The
video checks cover delayed startup ownership, frame/finalization cancellation,
restart, errors and compressed-payload/container budgets. The actual pinned
Mediabunny muxer remuxed H264; an injected small budget stopped a second remux
before the full clip was added. No GIF encoder, worker or dependency remains.

The production world renderer and native NVENC backend saved a one-second,
160×128 H264 clip with 60 frames at 60 fps on the RTX 3090. Three output pixel
comparisons matched an independent world renderer despite preview pan, zoom,
resize and overlays. Exactly 120 simulation steps were added to 64 initialization
steps. Startup, mapping and encoding discard/restart checks passed. The small
fixture completed in 0.755 seconds, with 562,472 logical export GPU bytes and no
redundant clean-output texture. This is an acceptance fixture, not a general
performance benchmark. GPU evidence (`artifacts/world-video-smoke.json`).

The real Chrome UI saved a 512×512 H264 clip with 60 frames at 60 fps, duration 1s,
108,299 bytes. Independent decoding found 60 distinct frames; native decoding
found 52 distinct frames out of 60. The full native GPU smoke, including synthetic
webcam gradients/motion, also passed with no uncaptured errors.
Browser decode (`artifacts/video-decode-browser.json`),
native decode (`artifacts/video-decode-native.json`),
browser clip (`artifacts/world-browser-video.mp4`).

At a 2537×1218 physical viewport, 600,000 particles and 10 steps/frame, the browser
returned to pause after saving. A second recording paused at 0 frames, retained
6,989,832 logical GPU bytes, and returned to zero after discard. Committed WASM
capacity rose from 9,043,968 to 10,158,080 bytes during the completed export and
stayed there; JS heap readings varied. These are component observations,
not a new measurement of total tab RAM or a replacement for the prior 67 MB
hover-card observation. Lifecycle (`artifacts/browser-world-video-lifecycle.json`).

Browser acceptance exposed stale cached control labels. The build now versions
boot, bindings, snippets, WASM and lazy media imports by content, explicitly
initializes the versioned WASM URL, and revalidates HTML. Generated WASM import
dictionary keys retain their compiled names. Versioning is idempotent, published
hashes match the audited manifest, and a rejected staging build left existing
output in place. Versioned query URLs select cache entries; this does not archive
old releases or promise atomic upgrades.
Fresh and warm reloads showed the updated controls without console warnings
or errors. HTTP requests used versioned boot, bindings, snippets, WASM and
vendor URLs; the warm path reused validated cached modules.
Request evidence (`artifacts/browser-versioned-requests.json`).

## Compact menu and panel shortcuts

The top menu now fits its six labels, ending just after Help. File/Load popups
retain their normal width. **X** hides all panels, including the mutation toolbar,
dialogs, crop guide, performance controls and debug readouts. It restores their
previous open state without changing pause. A small, noninteractive **X for
panels** hint remains at the bottom-left while hidden. Hidden crop tools cannot
intercept canvas input, and hiding a hovered preset restores the original project.
**G** toggles Color by Cohort through normal project history; **C** still saves a
checkpoint. Text-entry protection remains in place.

Chrome acceptance checked menu popups, both directions of G, checkpoint feedback,
panel/dialog restoration, an unchanged crop after dragging with panels hidden,
continued simulation while hidden, and preset-preview rollback. No browser
warnings or errors were captured. Native all-target checking, eight interaction
tests, formatting, the native release build and the production WASM build passed.
Recorded observations (`artifacts/browser-panel-shortcuts.json`),
compact menu (`artifacts/browser-compact-menu.jpg`),
all panels hidden (`artifacts/browser-all-panels-hidden.jpg`),
cohort shortcut enabled (`artifacts/browser-cohort-shortcut-on.jpg`).
The subsequent panel hint was checked in Chrome in both visibility states;
native and WASM release builds passed. Hint (`artifacts/browser-panels-hint.jpg`).

Comma/period now step Physics Rate through 1, 5, 10, …, 60, using the adjacent
mark when the slider starts between marks. The previous 1 → 6 transition was
replaced with 1 → 5. Brackets were unbound and now lower/raise Mutation Scale
by 0.01 on hundredth marks within 0–0.6, through normal project undo history.
Chrome checked 1 → 5 → 10 for Physics Rate, mutation changes 0 → 0.01 → 0, and
repeated brackets stopping at both 0 and 0.6. The simulation stayed paused at
substep 1. Native/WASM release builds and formatting passed, with no browser
warnings or errors.
Physics minimum (`artifacts/browser-physics-minimum-five.jpg`),
mutation controls (`artifacts/browser-mutation-brackets.jpg`).

## Remaining device coverage

Physical camera capture was not activated; synthetic readbacks establish its
math, not a particular V4L2 device or browser permission/capture implementation.
The native window itself was not driven through desktop automation; native
GPU rendering/readbacks and video were run, and the shared egui interface was
exercised in Chrome. Other browsers, operating systems, GPUs and device-loss
recovery need their own acceptance runs. Very long chaotic trajectories can
diverge across GPU compilers.

## Tick controls, stage profiling and idle scheduling — 2026-10-15

The editor now retains the original steps-per-frame mode and adds a fixed tick
clock, separate presentation cadence, Step 1 and exact N-tick execution. A reset
initializes tick 0 separately from subsequent numerical updates. Simulation-only
batches execute entity and trail updates without drawing or bloom. Automatic
clock overload is retained and displayed, rather than concealed by calibration.
The shader update equations were not changed.

The native checker ran serially on the RTX 3090 / Vulkan / 610.57.04 adapter,
through the one-CPU low-priority launcher. Two instrumented runs used the same
600,000-particle Starcrossed workload, 1024-square trails, 30 ticks per frame,
four requested/five actual temporal samples, bloom and 1280×720 output. Each
had three warmups and twelve measured frames; calibration was disabled.

| GPU stage | Mean ms across 24 measured frames |
| --- | ---: |
| Entity update | 2.658 |
| Trail processing | 8.700 |
| Drawing / accumulation | 2.374 |
| Bloom | 0.048 |
| Final assembly | 0.011 |
| Complete scene encoder | 13.904 |

Trail processing is the largest measured stage for this fixture. Stage totals
exclude gaps, accumulator clearing and other unclassified commands. Portable
timestamp boundaries add work and exclude query readback, egui and display
presentation. The two instrumented runs averaged 0.983 ms CPU encoding and
15.126 ms completed-frame time. A matching uninstrumented control averaged
0.454 ms encoding, 0.015 ms preparation, 0.033 ms submission-call time and
14.694 ms completed-frame time. These bounded observations show why profiling
is optional; they do not establish a general performance improvement or browser
frame rate. Historical CPU encoding included preparation and submission, unlike
the new separate intervals. The historical 13.7 ms GPU result used a different
timestamp method, so this is not a controlled historical regression comparison.
Raw reports: stages (`artifacts/timing-stages.json`),
repeat (`artifacts/timing-stages-repeat.json`),
uninstrumented control (`artifacts/timing-unprofiled-c.json`).

The independent numerical reference passed six hardware cases: supplied-rule
motion, seeded trail diffusion/decay, wrap/bounce/kill outcomes, dispatch tails,
64/128/256-thread workgroups and 1,025 ticks across the uniform-batch boundary.
All relevant state was finite and same-device serial/batched results were exact.
Maximum particle error was about 2.30e-7 in the short wrap fixture and 1.53e-5
over 1,025 ticks; maximum trail error was 1.51e-6. Tolerances and deliberately
excluded stochastic behavior are recorded in the
numerical report (`artifacts/timing-numerical.json`). This is one GPU/backend,
not cross-device certification. The ordinary GPU smoke also passed, with no
uncaptured errors: report (`artifacts/timing-gpu-smoke.json`),
rendered frame (`artifacts/timing-native.png`).

Actual Chrome UI execution initialized tick 0, advanced Step 1 to tick 1, then
executed N=257 across bounded batches, finishing paused at tick 258 with no
pending work or application error. The GPU substep counter was 259 because it
also includes initialization. See exact-step evidence (`artifacts/timing-browser-exact.json`).

The first new paused bundle recorded zero additional scene submissions over
129.516 seconds, while 26 UI updates remained. The old bundle recorded about
1,800 UI updates over 90.091 seconds at the same paused 60,000-particle setting.
These different observation windows are reported directly, not as a paired
energy measurement. Subsequent inspection found that successful implicit saves
requested a repaint even when they produced no notice; those completion
wakeups are now suppressed, while explicit successes and errors still repaint.
Old observation (`artifacts/timing-paused-baseline.json`),
first new observation (`artifacts/timing-idle-first-pass.json`).

The final bundle then stayed paused with optional GPU profiling enabled and its
readback drained for 115.121 seconds. UI updates, logical ticks, scene submissions
and presentations all stayed unchanged; no application repaint deadline or
browser warning/error was present. This verifies that profiling and implicit
saves return to idle. Framework animation callbacks and the debug five-second
memory timer still run, so this does not claim zero browser wakeups or measured
power savings. Final idle observation (`artifacts/timing-idle-final.json`).

An actual paused Chrome PNG capture completed at tick 4,310 without advancing
the simulation; its temporary clean-frame GPU allocation returned to zero.
The browser download-event waiter timed out, but the newly written 1280×800 PNG
was independently verified on disk and visually inspected.
Capture evidence (`artifacts/timing-browser-capture.json`),
captured image (`artifacts/timing-browser-capture.png`).

The 300-tick/sec, 1-FPS browser fixture encountered a separate measured limit:
browser animation callbacks averaged roughly 886 ms despite an immediate app
repaint deadline, with the document reporting focused and visible. The five-second
timer remained near five seconds. That run retained clock lag and produced more
simulation submissions than presentations, but did not achieve 300 ticks/sec.
The cause of the browser callback constraint was not established; occlusion or
driver/browser policy is not proven. See callback evidence (`artifacts/timing-browser-callback-limit.json`).
It is not a successful target-rate benchmark or evidence of browser GPU speed.
A subsequent 60-tick/sec, 1-FPS run advanced 5,585 ticks over a 92.734-second
observation (about 60.23 ticks/sec, with up to one callback of sampling-phase
uncertainty), retained zero lag and used 92 scene submissions for 89
presentations. Fixed-clock evidence (`artifacts/timing-browser-fixed60.json`).

Native recording regression also passed: a one-second 160×128 NVENC clip had
60 decoded frames, 52 distinct frames, three independent world-output matches,
bounded readback and successful discard/restart during startup, mapping and
encoding. No uncaptured GPU errors occurred. This verifies the recording
pipeline separately from browser hidden-service routing.
Native lifecycle (`artifacts/timing-video-smoke.json`),
independent decode (`artifacts/timing-video-decode.json`).

Clock unit tests and framework/source review cover visibility transitions, but
the browser automation did not produce a genuinely hidden document. Actual
hide/return, pending-service completion while hidden, and native minimize/restore
acceptance remain manual coverage. Physical camera and second-device numerical
coverage are also pending; the paused capture above does not establish those
cases.

The Rust suite passed 63 unit/integration checks plus three doctests, and all
25 existing JavaScript service checks passed. Native release and the actual
optimized WASM bundle built. Tests cover explicit queues, zero/one/many ticks,
different presentation cadences, pause/visibility epochs, retained overload,
the portable timestamp query limit and stage aggregation. No commit or
deployment was performed.


## Fast Cells close-view target — 2026-10-17

At World Size 1, Physics Rate 15, one temporal sample and 1920 × 1080,
three 300-frame native RTX 3090/Vulkan runs measured 4.343 ms mean / 5.122 ms
pooled p95 at zoom 8, versus the retained entry baseline's 6.445 / 8.767 ms.
Zoom 32 measured 4.120 / 4.893 ms, versus 5.226 / 6.240 ms. Individual zoom-8
p95 results ranged 4.945–5.158 ms, so the strict all-repeat 5 ms p95 goal still
needs margin. Zoom 32 passed that gate in all three runs. These are complete
GPU scene timings, excluding readback, editor composition and presentation.

The retained changes are bounded tiled Cells gathering with fused averaging,
consolidated compute dispatches and a conservative pristine-field sampling
shortcut. Fast Cells defaults on and can be disabled. It permits the approved
small cell-edge differences and adds 192.3 MiB logical renderer memory in this
case. Painted/exposed fields retain their old sampling costs after clear/reset.
Grouped deposits and other unsuccessful attempts were removed.

All 99 Rust tests, 31 byte-exact field cases, 32 gather/reference/fallback cases,
Cells/zoom regression suites, six independent numerical cases, native release
build, WASM library check, formatting and diff checks passed. Browser GPU timing
and other devices remain unmeasured. See the [full approach, results and limits](CELLULAR_RENDERING_VALIDATION.md#fast-cells-and-the-5-ms-target-2026-10-17)
and local raw summary (`artifacts/local/perf-5ms-round1/final-summary.json`).


## Zoomed scene cadence — 2026-10-17

The live scheduler now preserves its scheduled presentation phase across UI
callback jitter instead of starting the next deadline after encoding finishes.
A running PerFrame update uses one deadline for its physics batch and scene
presentation. Input-only early redraws leave that deadline alone. Resume,
visibility suspension and presentation-cap changes start fresh cadence epochs;
a long stall earns one automatic update before the deadline rebases. Fixed-rate
physics still accumulates actual elapsed time and retains overload debt. These
changes do not reduce Physics Rate, World Size, render resolution or Cells
quality.

A finite production-app desktop probe passed all twelve combinations of Physics
Rate 1/15/30/60 and zoom 1/8/32, with Cells, World Size 1, one temporal sample,
calibration off and actual 1920 × 1080 working size. Each case recorded 180 UI
callbacks and 180 scene updates, with exactly 180 × Physics Rate completed ticks.
Across 2,160 measured scene intervals, mean cadence was approximately 16.667 ms;
per-case p95 ranged 16.823–17.071 ms, maximum was 18.679 ms, and none exceeded
20 ms. This verifies one-refresh native cadence on this 60 Hz display during the
finite run, not a strict 16 ms maximum or hardware presentation/input latency.
Raw fullscreen cadence (`artifacts/local/cadence-16ms/native-final-fullscreen.json`).

An additional ordinary-window probe held about 60 scene updates/sec at Physics
Rate 15 for zoom 8/32 (p95 16.887/16.898 ms, actual 1080 × 1212). Its Physics
Rate 60 cases still measured approximately 33 ms, with rolling GPU p95 reaching
21–23 ms. Concurrent desktop/browser load was not isolated in that run, so its
cause is unresolved; the result does not support a universal 16 ms maximum at
all workloads. Raw windowed cadence (`artifacts/local/cadence-16ms/native-final-window.json`).

The earlier windowed diagnostic reproduced approximately 33 ms scene gaps at
Physics Rate 15 despite regular UI callbacks, and additional physics-only
submissions. Its working dimensions differ from the final fullscreen run and
its input filtering differs, so those runs do not establish a controlled GPU
speedup. Phases share an evolving simulation; their GPU observations are rolling
windows rather than equal-state benchmark samples. The independent fixed-work
GPU results above retain their separate measurement scope.

The browser bundle in `dist` was rebuilt with Fast Cells and the cadence
changes, and so was the native app binary. Browser execution and zoom input were
exercised, but both the app and a plain page without WASM or GPU work showed
approximately 1,017 ms independent animation-frame stalls in the automation
window despite visible/focused DOM signals. Those measurements cannot certify
foreground browser cadence or attribute the throttle to the app. Browser
acceptance is separate from the automated-window timing data. In a visible,
manually used browser tab, the frame interval read about 16–17 ms at Cells /
World Size 1 / Physics Rate 15 / zoomed in. This is a manual reading, not a
browser percentile matrix. A later read-only browser capture also shows Cells
rendering and a 16.60 ms interval at Physics Rate 5.
Browser control diagnostic (`artifacts/local/cadence-16ms/browser-after-diagnostic.json`).

All 103 Rust checks including doctests pass, with deterministic coverage of
alternating late/early callbacks, 120 Hz callbacks at a 60 FPS cap, 55 FPS over
60 Hz callbacks, early input redraws, long stalls, resume/cap changes, exact-step
counts and fixed-rate debt. Native release and the optimized WASM bundle build;
formatting, diff and static distribution checks pass. Reproduction and metric
boundaries are in [VALIDATION.md](VALIDATION.md#native-live-frame-cadence).
