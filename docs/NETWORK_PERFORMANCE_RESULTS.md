# Network performance campaign, 2026-10-21

The completed primary native measurements reduce GPU frame time by 60.54–68.10%
for MLP and 40.65–51.13% for GRU, with essentially unchanged Fourier GPU time.
The retained candidate is wide integer occupancy accumulation with the original
network evaluator. Secondary native controls, long-run exactness and final
delivery GPU/browser checks and final independent review are complete.
Earlier paired and unrolled evaluators are not retained.
Local evidence is under ignored
`artifacts/local/network-performance-20261020/`; generated binaries, captures and
logs are not distributed in Git.

## Scope and measurement

This campaign measures Fourier, MLP and GRU simulation with Classic and Fast
Cells rendering. It follows the earlier Cells renderer campaign; its baseline
already contains those accepted rendering changes. Gains reported here must be
relative to this campaign's frozen baseline, not added to earlier percentages.

The fixed primary workload is 600,000 particles, world size 1, four cohorts,
15 physics steps per rendered frame, one motion sample and 1920×1080 output.
Each primary run initializes for 120 ticks, warms up for 30 frames and measures
300 frames. Three alternating baseline/candidate repetitions are required.
The runner also compares configuration, completed ticks, live/dead counts,
canonical state and exported PNG bytes. GPU work is serialized with
`/tmp/gpu-timing.lock`. Detailed stage profiles are separate from primary timing.
GPU frame time is not browser FPS, CPU encoding time or an energy measurement.

The completed `wide-primary` matrix contains 36 runs: six mode/style cases,
two binaries and three repetitions. `wide-pilot` was an earlier 96-frame,
single-repetition check. Dispersed, profile and stress controls are complete
below. Exact-checkpoint, long-run, regression and final rebuilt-delivery gates
are complete. Independent review verified the measurements and delivery evidence.

Hardware in the completed native contract report: NVIDIA GeForce RTX 3090,
Vulkan, NVIDIA driver 610.57.04. Results on this adapter do not establish gains
on other GPUs or browser implementations.

## Completed primary results

Times below are means of three per-run means, in milliseconds per rendered
frame. Change is candidate relative to baseline; lower is better. Paired range
is the minimum and maximum change among the three matched repetitions, not a
confidence interval or pooled percentile.

| Mode / renderer | Baseline GPU ms | Wide GPU ms | Change | Paired range |
| --- | ---: | ---: | ---: | ---: |
| Fourier / Classic | 3.077 | 3.096 | +0.64% | -0.11% to +1.14% |
| Fourier / Fast Cells | 4.593 | 4.595 | +0.03% | -0.67% to +0.63% |
| MLP / Classic | 17.038 | 5.435 | -68.10% | -68.78% to -66.76% |
| MLP / Fast Cells | 19.182 | 7.569 | -60.54% | -61.23% to -59.20% |
| GRU / Classic | 18.205 | 8.896 | -51.13% | -51.88% to -50.60% |
| GRU / Fast Cells | 23.380 | 13.875 | -40.65% | -42.18% to -39.35% |

Fourier GPU changes are small and cross the zero-change line between
repetitions. The evidence supports approximately unchanged Fourier performance,
not a Fourier speedup.

The GPU tail-time gate uses each run's p95. The table averages those three
per-run p95 values; it is not a pooled-frame p95. Every neural matched pair
improved its p95, while Fourier paired directions are mixed.

| Mode / renderer | Baseline GPU p95 ms | Wide GPU p95 ms | Change | Paired range |
| --- | ---: | ---: | ---: | ---: |
| Fourier / Classic | 4.023 | 4.017 | -0.13% | -1.16% to +0.61% |
| Fourier / Fast Cells | 5.536 | 5.529 | -0.14% | -2.43% to +1.52% |
| MLP / Classic | 20.906 | 6.147 | -70.60% | -71.60% to -69.21% |
| MLP / Fast Cells | 22.921 | 8.395 | -63.37% | -63.77% to -62.58% |
| GRU / Classic | 21.002 | 11.218 | -46.59% | -47.52% to -45.41% |
| GRU / Fast Cells | 26.633 | 16.041 | -39.77% | -43.68% to -37.14% |

CPU preparation and command encoding are measured separately. Each timing cell
is baseline → candidate. CPU changes cannot all be attributed to vector reuse:
faster GPU execution can reduce driver backpressure affecting host timings.

| Mode / renderer | Prepare ms | Change (paired range) | Encode ms | Change (paired range) |
| --- | ---: | ---: | ---: | ---: |
| Fourier / Classic | 0.0176 → 0.0165 | -6.62% (-10.12% to -1.89%) | 0.4233 → 0.4212 | -0.49% (-5.00% to +5.39%) |
| Fourier / Fast Cells | 0.0230 → 0.0232 | +0.82% (-8.90% to +9.62%) | 0.6767 → 0.6974 | +3.06% (-8.22% to +11.15%) |
| MLP / Classic | 0.0333 → 0.0198 | -40.57% (-44.29% to -37.42%) | 0.6126 → 0.4707 | -23.16% (-25.06% to -20.16%) |
| MLP / Fast Cells | 0.0400 → 0.0254 | -36.67% (-38.89% to -34.49%) | 0.9536 → 0.7488 | -21.47% (-25.14% to -18.19%) |
| GRU / Classic | 0.0329 → 0.0231 | -29.73% (-33.57% to -25.28%) | 0.5946 → 0.5119 | -13.91% (-17.60% to -11.25%) |
| GRU / Fast Cells | 0.0425 → 0.0362 | -14.84% (-19.82% to -11.06%) | 0.9950 → 0.9341 | -6.12% (-8.91% to -1.41%) |

Fourier Classic preparation improved, while Fourier Fast Cells preparation and
both Fourier encoding comparisons are mixed. There is no claim that all CPU
paths improve. Neural host-time reductions are observed end-to-end effects;
these runs do not isolate uniform packing as their sole cause.

All six first-repetition matched pairs preserved all nine canonical state files
exactly. PNG bytes matched in all 18 matched pairs. All primary runs retained
600,000 live particles. Machine-readable analysis and existing plots are
`wide-primary/baseline-vs-wide.json`, `.png` and `.svg` in the local evidence
directory. No plots or builds were run alongside timing to produce this report.

## Dispersed controls

The completed `wide-dispersed` matrix contains 24 runs: four neural mode/style
cases, two binaries and three repetitions of 120 measured frames. This separate
fixture uses seed 0.875 with random initialization, four cohorts and 600,000
particles. It retains the primary world's size, physics rate and output settings;
it does not replace the primary fixture or estimate its speedup.

| Mode / renderer | Baseline GPU ms | Wide GPU ms | Change | Paired range |
| --- | ---: | ---: | ---: | ---: |
| MLP / Classic | 7.019 | 6.239 | -11.11% | -11.29% to -10.79% |
| MLP / Fast Cells | 8.476 | 7.715 | -8.98% | -9.14% to -8.87% |
| GRU / Classic | 8.031 | 7.816 | -2.67% | -2.99% to -2.37% |
| GRU / Fast Cells | 9.430 | 9.205 | -2.38% | -2.77% to -1.98% |

These gains are substantially smaller than in the primary fixture, consistent
with the optimization depending on occupancy contention. All runs retained
600,000 live particles. All four first-repetition canonical state pairs and all
12 matched PNG pairs were exact. Other seeds, evolving population geometries
and other adapters can produce different timings.

## Stage attribution and stress control

The separate `wide-profile` matrix contains 24 runs: six cases, two binaries and
two repetitions of 64 measured frames with detailed timestamp instrumentation.
These instrumented runs locate the change; they do not replace the lightweight
total-timestamp primary timing table. Entity-update means below are milliseconds per physics tick.

| Mode / renderer | Baseline entity ms/tick | Wide entity ms/tick | Change |
| --- | ---: | ---: | ---: |
| Fourier / Classic | 0.0936 | 0.0944 | +0.91% |
| Fourier / Fast Cells | 0.0945 | 0.0964 | +2.07% |
| MLP / Classic | 1.1454 | 0.2407 | -78.99% |
| MLP / Fast Cells | 1.1788 | 0.2426 | -79.42% |
| GRU / Classic | 1.1051 | 0.4754 | -56.98% |
| GRU / Fast Cells | 1.0560 | 0.4805 | -54.50% |

The large neural change is in entity update, where the contended occupancy
compare-exchange loop was replaced. Trail-stage frame time increased by
16.56–21.55% in the neural profile cases: wider reads and clears are a cost,
not a free optimization. Fourier's profiled entity changes are small positive
deltas; primary total-GPU results remain approximately unchanged. These profiles
support occupancy contention as the bottleneck without implying that network
matrix multiplication itself became faster.

The `wide-stress` control uses GRU, Fast Cells, 64 cohorts, 60 physics steps per
frame, zoom 8 and 2,048 setup ticks. Three matched repetitions of 64 measured
frames reduced mean GPU frame time from 84.397 to 42.313 ms (−49.86%; matched
range −49.95% to −49.80%). CPU preparation changed −3.41% with mixed paired
directions, and encoding changed −0.35%, also mixed. It is a distinct workload,
not an additional percentage to combine with the primary result. First-repetition
canonical state and all three matched PNG pairs were exact.

Completed analyses and detailed plots are under `wide-dispersed/`,
`wide-profile/` and `wide-stress/`. The compact primary GPU-only visualization
is `summary-gpu.png` / `.svg`, generated by `plot-summary-gpu.py`.

## Retained implementation

Revision-3 MLP/GRU particles deposit nonnegative occupancy as well as signed
momentum. Previously, each occupancy deposit used a saturating compare-exchange
loop; particles sharing a texel could repeatedly retry that loop.

Where device buffer limits permit, the accumulator now has four integer lanes
per texel: unchanged signed momentum x/y, followed by unsigned low/high limbs
of the occupancy sum. A low-limb atomic add returns its old value. A carry adds
one to the high limb. The canvas pass saturates the completed integer sum before
performing the original float conversion and scaling. It runs after all entity
invocations, so it never consumes a partially updated pair of limbs.

Every individual addition retains the original clamp and rounding to
0…7,680,000. For nonnegative additions, repeatedly saturating each addition is
exactly equivalent to saturating their exact sum once. Each addition is below
2³² and produces at most one carry. Even the conservative bound of u32::MAX
particles contributing the maximum stays below 2⁵⁵. Concrete WGSL integer
arithmetic wraps modulo 2³², and atomicAdd returns the original value; see the
[integer rules](https://www.w3.org/TR/WGSL/#integer-types) and
[atomicAdd definition](https://www.w3.org/TR/WGSL/#atomic-add).

The original three-lane path remains for Fourier, legacy revisions and devices
where the four-lane allocation exceeds either actual buffer or storage-binding
limits. Its original compare-exchange retry structure is retained, with the
GRU saturation early return. The host reserves four-lane capacity even when
starting in Fourier, allowing later mode changes without reallocating the
accumulator. At a 1024² trail canvas this costs an additional 4 MiB, including
in Fourier. There are no extra buffers, dispatches or GPU feature requirements.

One host layout decision selects the entity specialization, canvas uniform and
clear extent. A layout-changing project commit invalidates prepared steps;
matching uniforms must be prepared before encoding again. Frame zero clears the
whole allocation after the canvas reset pass; other steps clear the active
three- or four-lane extent.

Reusable CPU uniform vectors also retain allocations across frames. Their
contents, padding and removed shove fields are zeroed before repacking. No
network width, layer, coefficient, activation, input, recurrence, history,
physics step, sensor, motor, rendering resolution or visual quality setting was
reduced. The production network shader is byte-identical to the frozen scalar
reference.

## Exactness evidence

`wide-contracts.json` completed without uncaptured GPU errors. Direct preclear
contracts use the actual production occupancy function, a frozen scalar
reference and a host u64 integer oracle. They cover zero/unsaturated deposits,
saturation crossing, signed-bit boundaries, single/multiple carries, 4,099
contending invocations, partial workgroups, untouched neighbors and momentum
sentinels. Both layouts and all three mode specializations are checked.
The actual canvas shader separately compares eight unsigned/carry boundary
cases against legacy saturation, with raw RGBA16F equality.

Native lifecycle checks compare forced narrow and optional wide engines through
revision 2→3→2→3, abandoned project previews, zero-step preparation, stale-uniform
invalidation and full-allocation reset. Uniform scratch checks compare every
packed word against fresh zero-filled packing after sentinel-prefilled
1→31→3→0→17 step preparations, including shove removal and frame-counter wrap.

Benchmark captures intentionally distinguish representation from semantics.
`splat-native.bin` retains the complete physical allocation and its declared
layout, size and hash. `splat.bin` always contains three canonical lanes:
unchanged momentum plus saturated occupancy. The runner independently decodes
native limbs and rejects inconsistent metadata or bytes before comparing the
canonical file. Every other state file and metadata field remains exact, as do
rendered output bytes and exported PNGs. End-step accumulators are cleared, so
these snapshots supplement rather than replace the direct preclear contracts.

The completed `wide-exact` matrix contains 80 executions and 40 matched
canonical-state comparisons at setup checkpoints 2, 17, 65 and 257. It covers
partial workgroups, Fourier and all three MLP/GRU revisions, positive/full hazard
and kill-boundary cases. Every compared canonical state file, including raw
rendered output, was exact. These cases do not export separate PNGs.

The completed `wide-long` matrix adds six executions and three matched
canonical-state comparisons at tick 8,193, one for each network mode with
600,000 live particles. All canonical files, including raw rendered output,
were exact. This is a full-size evolved-state check, distinct from the tiny
partial-workgroup fixtures despite inherited case names.

`wide-regressions` passed all three existing suites: networks, numerical and
field checks. `wide-pair-contracts.json` passed 12 direct evaluator cases without
uncaptured GPU errors using the binary's embedded production shader against the
frozen original scalar reference. The direct checker name reflects its two
input streams; no paired evaluator is retained in production.

Final CPU validation passed 112 Rust checks including three doctests, three
JavaScript suites and formatting. Strict Clippy remains blocked by 73 existing
findings, compared with 75 in the baseline. Five newly introduced findings were
corrected and the remaining signatures were checked against the baseline; this
is not a clean strict-Clippy result.

Completed pilot, primary, secondary, long-run and regression checks passed the
comparisons above. The rebuilt delivery then passed `final-delivery-equality`:
12 executions covering all six full-size mode/style cases, with all six matched
canonical state and PNG pairs exact against the measured wide binary. This
GPU-tested binary precedes the final UI-only window-layer correction below.
Its 64-frame runs are an equality gate, not a replacement acceptance timing
study.
`final-delivery-network-contracts.json` and
`final-delivery-network-pair-contracts.json` also passed with empty uncaptured
error lists. Final independent review passed with no remaining blockers.

## Rejected experiments

- Paired MLP/GRU evaluation shared coefficient loads while retaining two scalar
  accumulation streams. It passed direct and integrated exactness checks but
  did not provide a repeatable performance benefit; it is not retained.
- Full constant-index evaluation and hidden-vector variants changed MLP output
  bits in direct contracts. They were rejected without relaxing exactness.
- Smaller row/input and output-loop unrolls passed direct contracts, but the
  combined candidate did not establish a useful MLP improvement. All evaluator
  unrolling was removed from the retained candidate.
- Reusing compare-exchange failure values instead of reloading occupancy was
  rejected after a performance regression. The narrow fallback reloads inside
  its original retry loop.

## Provenance and remaining limits

`wide-source-before.json` and `wide-source-after.json` match across all 220
recorded inputs. The frozen `wide-gpu-check` SHA-256 is
`15ad313c3c6d00b040313ac71842cb442b079d11c5994e56ccca584033366b9d`.
`wide-pilot/manifest.json` records this binary and the matched frozen baseline
`5c7630af17d62e87dd490d4a65a45f83b3f2f07cb7ca08a937459919fb253c3c`.
The local design and diagnostic boundaries are documented in
`WIDE_OCCUPANCY_PLAN.md` and `WIDE_QA.md` in the campaign artifact directory.

Measured source and delivery source are recorded separately. After measurements,
four files received Clippy cleanup: `src/app.rs` collapsed an equivalent nested
narrow-layout condition; `src/controls.rs` changed fixture initialization to
struct defaults; `src/gpu.rs` removed a redundant u64 conversion on the wide
storage limit; and `src/gpu/occupancy_contracts.rs` flattened a nested format
expression and moved its test module. `delivery-source-before.json` and
`delivery-source.tar` record the delivery inputs. All production WGSL remains
byte-identical to the measured wide source, but the complete source trees are
not identical. The timing tables above belong to the measured `wide-gpu-check` binary.

A subsequent UI-only phone-layout correction in `src/app.rs` and
`src/studio_ui.rs` superseded the first delivery build's overflowing phone timing
capture. `final-delivery-source-before.json` and `final-delivery-source-after.json`
match across 220 recorded inputs. Final native and WASM builds passed.
`final-delivery-binaries.json` records the rebuilt GPU check SHA-256
`39cf07ccf137677028763350dcc7ce08ec172eb0c5231544eae59f880b2d1893`
and native app SHA-256
`fb0b7af0ad6e17b26f81eb8221041ab18a7dc35a29f9d56a47bcda1f5f2a07b1`.

The browser evidence directory is `artifacts/local/ui-minimal-20261020/`.
Its `accepted-build-source.json` matches the pre-layering final-delivery source
snapshot and records that build's served WASM distribution hashes. The `accepted` filename prefix labels the
QA session; it does not mean user approval. The closed `accepted-session.json`
records 18 captures with zero page, console or command errors. The valid
`accepted-clean.png` matched the pre-redesign reference with absolute pixel
error count zero. `accepted-desktop-default.png` and the phone control captures
show the tested UI: timing top/bottom, wrapped and scrollable landscape controls,
network settings and screenshot controls fit with readouts visible. Tools,
preset sections and the X shortcut were exercised. A 600,000-particle
Fourier→MLP→GRU→Fourier smoke advanced each neural mode for 180 ticks without
reported errors. The later `accepted-*-hero` and `accepted-cells-close` images
are excluded as showcase evidence: an automation coordinate missed the intended
preset selection. They do not establish the intended preset or close-view look.
These captures remain valid evidence of the preceding functional checks.

The latest delivered UI adds one window-order line in `src/studio_ui.rs`:
foreground dialog windows now occlude the quiet identity correctly. This is the
only source difference from the pre-layering final delivery.
`layered-source-before.json` and `layered-source-after.json` match across all
220 inputs. Native and WASM builds passed, and `layered-tests.log` records all
112 Rust checks including three doctests passing. All production WGSL and
simulation host source remain unchanged. The native GPU contracts, timing study
and six-pair delivery equality gate were not rerun for this UI-only correction,
following the reviewed validation scope.

`layered-binaries.json` records the latest rebuilt GPU check SHA-256
`cdd6fddf25cb63b2f435c2bf4691324da395cbad4e08ea53578181ee75039578`
and native app SHA-256
`46bc435813bbadc277e2988fef500ef5392fa132ed86bff6e9735374faf35ef8`.
`layered-build-source.json` in the UI evidence directory matches this source
snapshot and records the actual served distribution hashes. Its browser
captures show the corrected dialog occlusion, Share menu and nested dialog
combo popup ordering. `layered-clean.png` still matches the original reference
with absolute pixel error count zero. `layered-session.json` records seven
captures with zero reported page, command or console errors. Its desktop,
close-view and phone captures correctly identify Medusa, 600,000 particles and
1,800 evolved ticks. These supersede the invalid earlier showcase attempts.
Independent final review verified the latest source and served build, dialog and
popup ordering, clean-export equality and closed browser session. No blockers
remain; the review record is `final-review.json`.

Forced narrow allocation exercises fallback behavior on the RTX 3090. A real
adapter whose limits admit three lanes but reject four has not been tested.
Additional storage pressure, mobile/browser drivers and low-contention
workloads can affect the benefit. Dispersed MLP/GRU controls confirm smaller
gains on that fixture. Completed Fourier regression checks and the storage cost
bound the result; no universal speedup is claimed.
