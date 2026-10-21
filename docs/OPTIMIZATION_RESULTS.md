# Cells optimization results — 2026-10-20

This campaign reduces renderer memory and command overhead while requiring exact
agreement with the entry implementation. Simulation settings, render resolution,
particle count, sampling, shader appearance, and fallback behavior are preserved.
No dependencies were added. The measured runtime/executable predates the UI
redesign; `final-source.json` identifies that measured candidate. The Cells
renderer and shaders remain unchanged by the redesign. These native timings do
not measure the redesigned UI.

## Fixed workload and measurements

Baseline: commit `7a5f3881c9a592d99269d6770abdb0c91ee984dd`. Native Vulkan on an
NVIDIA GeForce RTX 3090, driver 610.57.04; Starcrossed, 600,000 particles,
1920 × 1080 output, world size 1, 15 simulation steps per rendered frame,
one temporal sample, Fast Cells and bloom enabled. Each process uses 120 setup
ticks and 30 warmup frames before measuring 300 frames. Three paired repetitions
at each zoom provide 900 measured frames per version per zoom. GPU jobs run
serially under the campaign lock.

The acceptance timer uses total-only GPU timestamps with identical
instrumentation. CPU encode time measures command encoding. Neither metric
measures browser scheduling, presentation, display frame rate, or energy use.
Logical allocations exclude driver heap overhead.

| Metric | Zoom 8 baseline → optimized | Change | Zoom 32 baseline → optimized | Change |
| --- | --- | --- | --- | --- |
| GPU mean | 5.146 → 5.017 ms | −2.52% | 4.989 → 4.820 ms | −3.38% |
| GPU p95 | 6.966 → 6.814 ms | −2.18% | 6.916 → 6.808 ms | −1.56% |
| CPU encode mean | 0.939 → 0.840 ms | −10.57% | 0.926 → 0.853 ms | −7.90% |
| Logical renderer allocation | 557.58 → 514.91 MiB | −7.65% | 557.58 → 514.91 MiB | −7.65% |

Values are the mean of three run statistics. Each run's p95 uses
`sorted_samples[ceil((n - 1) * 0.95)]`; the table averages those three p95 values,
not a pooled percentile. Individual runs and paired deltas remain in
`summary.json`. The zoom-32 p95 did not improve in every paired repetition, so
these results do not establish a universal tail-latency improvement.

![Native measurements with individual run values](../artifacts/local/optimization-campaign-20261020/performance.png)

## Shorter secondary controls

Each control uses two paired repetitions of 120 frames at each zoom: 240 frames
per version per zoom, compared with 900 in the primary workload. Every paired
PNG is exact and each pair's workload matches. Changes below compare baseline
and optimized within the same control; they are not cross-workload subtractions.

| Control | GPU mean change, zoom 8 | GPU mean change, zoom 32 |
| --- | --- | --- |
| Reference renderer | −3.03% | −2.77% |
| Forced-capacity fallback | −2.05% | −3.56% |
| Motion blur, requested 4 / actual 5 samples | −6.19% | −7.53% |
| Active empty painted field | −2.17% | −3.14% |

Motion-blur CPU encode means fall from 2.8613 to 2.4933 ms at zoom 8 (12.9%)
and 2.9941 to 2.4860 ms at zoom 32 (17.0%), pooling equal-length runs within
each version. The active-field timing control uses an **empty** field. Separate
painted-field correctness checks cover actual nonzero paint; this timing does
not establish performance for nonzero painted fields.

## Retained changes

- Tiled records hold geometry and pigment in 32 bytes instead of 48. Preparation
  and fill share the same bounds calculation. The original record-count cap and
  adapter limits remain unchanged. This saves 42.67 MiB in the measured renderer
  and reduces record traffic and workgroup cache size.
- Tiled gathering initializes nearest-site winners directly. On tiled failure,
  its existing raster dispatch zeros every valid padded site before the original
  sprite fallback. Unsupported adapters retain the host clear. This removes
  redundant clear/read traffic on successful tiled rendering.
- Single-sample membrane accumulation overwrites history, including explicit
  zeros for empty pixels. Multiple samples retain history clearing and addition.
- Consecutive density, planning, and resolve dispatches share their respective
  compute passes, preserving dispatch order and resource dependencies.

The `add_offsets` removal experiment passed exactness checks but was rejected
on performance evidence; the dedicated offset-adjustment dispatch remains.
Render bundles also passed all 90 exact comparisons but showed no convincing
timing gain and were removed. Historical splat-storage experiments are excluded.
The retained runtime is the pre-bundle implementation measured above.

## Quality and regression gates

- **90 baseline comparisons are exact**, covering intermediate color/mass,
  selection, membrane history, nearest-two site keys, and final output across
  reference, tiled, and forced-capacity fallback variants. Fixtures include
  compact and dense scenes, wrapped edges, selection changes, temporal samples,
  and membrane transitions. Rendering does not advance simulation state.
- **42 resolve captures are byte-exact** at 1080p, 1440p, and 4K, including dark
  and light themes, zoom, sharpening, chromatic effects, bounded worlds, and
  selection. Baseline and optimized RGBA bytes match without tolerance changes.
- All six paired benchmark screenshots match exactly, including across repeated
  runs at each zoom. Workload, project/preferences, camera, tile state, ticks,
  and adapter equality are checked before results are summarized.
- GPU numerical, painted-field, network-contract, and zoom checks pass.
- Rust checks pass: 105 tests including three doctests. All 25 JavaScript service
  tests pass. The WASM build and static distribution build also pass, using the
  explicit stable toolchain at normal scheduling priority.
- Full legacy `--cells` checks pass on both baseline and final runtime after
  correcting fixture assumptions about the white world-border background.
  Tolerances are unchanged.
- Strict Clippy reproduces the same existing 71 library / 75 library-test errors
  on baseline. They are outside this optimization change and remain unresolved.

The final Chromium 152.0.7977.82 smoke run uses actual RTX 3090
WebGPU/Vulkan (`isFallbackAdapter: false`), observes 60 distinct rendered frames,
advances simulation, and reports no application or console errors. After pause,
180 rendered frames with the camera control held preserve simulation substep
2146 exactly. Opaque 1920 × 1080 direct canvas captures were visually inspected:
the wide scene and close view retain detailed cell membranes. These browser
results apply to an isolated temporary profile launched with explicit
Vulkan/WebGPU flags and the GPU sandbox disabled. Actual browser zoom was not
measured. The headless full-page screenshot is all-zero RGBA on this
environment's compositor; direct canvas capture supplies the visual evidence.
This smoke check does not
measure browser frame rate or establish mobile performance.

## Reproduction and evidence

Local evidence is under `artifacts/local/optimization-campaign-20261020/` and is
not assumed to accompany a fresh checkout:

| Evidence | Files |
| --- | --- |
| Entry and candidate provenance | `baseline-source.tar`, `baseline-original`, `final-source.json` (measured runtime), `final-source.patch`, `final-delivery-source.json` (delivery provenance) |
| Paired raw timings, commands, hashes, hardware samples and screenshots | `final-comparison/manifest.json`, `final-comparison/{baseline,optimized}-z{8,32}-r{0,1,2}.{json,png}` |
| Secondary paired controls | `control-summary.json`, `reference-control/`, `fallback-control/`, `temporal-control/`, `field-control/` |
| Statistics and plot generation | `summarize.py`, `summary.json`, `performance.png`, `performance.svg` |
| Exact intermediate/final comparisons | `baseline-transitions/`, `baseline-transitions.json`, `transitions-final.json` |
| Exact resolve comparisons | `baseline-resolve.json`, `resolve-final.json`, `baseline-resolve/`, `final-resolve-captures/` |
| GPU regressions | `numerical-final.json`, `fields-final.json`, `networks-final.json`, `zoom-final.json`, `baseline-cells-final.json`, `cells-final.json` |
| Rejected experiments | `offsets-comparison/`, `candidate4-transitions.json`, `add-offsets.patch`, `bundle-ablation/` |
| Build and language checks | `tests-final.log`, `wasm-final-build.log`, `web-dist-final.log`, `build-web-normal.sh` |
| Browser smoke and direct canvas capture | `final-browser-smoke.json`, `final-canvas.png`, `final-close-canvas.png` |
| Existing lint failures | `baseline-clippy.log`, `clippy.log` |

The manifest records executable hashes and exact commands. To regenerate the
summary and plots from the paired reports:

```sh
/tmp/primordial-campaign-plots/bin/python \
  artifacts/local/optimization-campaign-20261020/summarize.py
```

The plotting interpreter is a campaign-local environment, not a project runtime
dependency. The summarizer rejects missing reports, mismatched workloads,
incorrect frame counts, GPU errors, and differing RGBA output.
