# Network modes: implementation and evidence

Generated reports and captures are local outputs under ignored `artifacts/`.
Paths below identify optional local evidence; these files are not distributed
with the repository.

Implemented 2026-10-16, against [the approved scope](NETWORK_MODES_SPEC.md).

## What is available

The mutation toolbar has direct Fourier / MLP / GRU buttons. Mode banks retain
parent/source, seed, and Mutation Scale; shared physics remains shared. Mode
switches initialize at tick 0. GRU has two independent four-value memory streams
per particle, evaluated once per simulation tick with shared cohort weights.
Adoption inherits exact coefficients and clears memory. Display-only settings
uploads preserve memory.

Timing & Experiments provides clean comparison restart, exact Run N ticks,
cancellation, and JSON reports. Reports include every mode bank, effective sizing,
completed tick count, an asynchronously read-back hash of effective cohort rules,
and available timing data. A cancelled run is incomplete. Simulation edits are
locked while work is pending. Completed wall duration ends after the GPU rule
snapshot has mapped, rather than when the final commands were submitted.

Project changes prepare resources inside validation/out-of-memory scopes and
commit the project, GPU resources, and history together after scope completion.
The browser awaits scope results without blocking its event loop. Saved-project and legacy preset
hover previews stay within the active architecture and prepare transactionally
(asynchronously in the browser); clicking a preset can change
architecture through the transactional load path.

Legacy projects retain v8/v9 migration, Fourier packing, original generator and
mutation arithmetic, binary share codecs, and legacy archive identity. Extended
projects use v10, a PS10 share envelope and archive version 3. All banks round-trip
through JSON, shares, custom-save ZIPs, history/checkpoints, and saved application
state. Startup now restores the saved project/checkpoints instead of discarding
them. Explicit launch presets/share links can still replace the startup project.

## Executed checks

- `cargo test --locked --lib --tests`: 73 tests passed, including existing preset, layout,
  mutation, share, archive, interaction, timing, recording, and export fixtures.
  New tests cover architecture arithmetic, frozen revision-1 and revision-2 coefficient bits,
  mutation bounds, valid all-zero parents, multi-bank switching/round-trips, ZIP,
  and malformed/inconsistent imports (including short Fourier rules).
- `node tests/browser_services.test.mjs`: passed.
- Native debug/release builds and optimized WebAssembly build/staged asset audit:
  passed. The system `/usr/bin/rustc` does not use rustup's installed wasm
  standard library; web builds used
  `RUSTC=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc`.
- Legacy hardware smoke (`artifacts/network-legacy-smoke.json`): passed on
  NVIDIA GeForce RTX 3090 / Vulkan / driver 610.57.04, with no uncaptured GPU errors.
- Legacy analytical GPU checks (`artifacts/network-legacy-numerical.json`):
  passed the independent motion/trail reference and serial/batched comparisons,
  including 64/128/256 workgroups and work across the uniform batch boundary.
- New hardware checks (`artifacts/network-modes-gpu.json`): passed on the same
  adapter. Independent host/GPU evaluator checks use unequal original/mirrored
  inputs over eight ticks. Maximum absolute output error was `1.1920929e-7` for
  MLP and `8.9406967e-8` for GRU (acceptance tolerance `3e-5`).
- New hardware checks also verified exact serial/batched particle and memory
  equality, repeatable reset baselines, exact picked coefficients, display-upload
  memory preservation, rule-edit clearing, hazard clearing, GRU boundary
  respawn/death clearing, and a deliberately injected scoped validation failure
  that left active particle/memory buffers and frame count unchanged.
- Native application startup ran for eight seconds with isolated temporary
  settings and no stderr, then the test timeout terminated it. Interactive native
  window controls were not separately exercised; shared UI behavior was checked
  in the browser.

Native GPU-rendered outputs, using derivation revision 2 and seed `0.25`, four cohorts,
16,384 particles, 128×128 trails, and 960 update submissions including initialization:

| MLP | GRU |
| --- | --- |
| Local capture (`artifacts/network-mlp.png`) | Local capture (`artifacts/network-gru.png`) |

These are rendered behavior examples, not visual similarity requirements or a
ranking of architectures. The first attempt to capture these omitted renderer
sampling and was black; the retained images were regenerated through the complete
sample/present pipeline and inspected.

## Regression checks after review

A review found correlated mutation directions, accepted sensor-gain overflow,
weak independent GRU coverage, and rejected browser MLP/GRU hover previews.
All four are addressed; compatibility and numerical boundaries are specified in
[the review addendum](NETWORK_MODES_SPEC.md#review-fixes-2026-10-16).

Expanded hardware results (`artifacts/network-fixes-gpu.json`) verify a dense,
independently calculated double-precision fixture, four particles across three
cohorts with distinct initial memories, 40 recurrent ticks, swapped mirrored
histories, and extreme finite inputs. MLP and GRU CPU/GPU maximum output errors
were each `1.7881393e-7`, below the `3e-5` tolerance. An overflowing imported gain
is rejected; a real 67-particle engine remains finite for 120 steps at the maximum
supported gain. Existing reset/death/adoption/pick/transaction checks also pass.

The Rust suite now includes independently generated Python equation fixtures,
mutation-direction decorrelation, forced adoption-seed collision handling,
revision compatibility, missing-Fourier-bank creation, original-project persistence
through previews, and 100,000 extreme finite-input steps per new architecture.
The original coefficient fixtures remain bit-exact.

Browser regression checks used isolated localhost:8092 with 30,000 particles:
hovering saved GRU B then C changed the active preview, reloading restored original
Starcrossed, empty-history undo preserved the restoration backup, leaving restored
Starcrossed, and clicking B followed by undo returned to Starcrossed. No GPU error
was shown. Screenshot (`artifacts/network-preview-fixed.png`) and
observation manifest (`artifacts/network-fixes-ui.json`) retain the evidence.
These new checks supplement the earlier comparison reports below; those older
reports still describe revision-1 rules. A physical OOM and a failed browser
preview-restore were not deliberately injected; preservation on failure was
source-reviewed and the underlying scoped GPU transaction rejection was tested.

## Browser interaction evidence

Chrome WebGPU was exercised on an isolated localhost:8091 origin, at world size
0.05: 30,000 particles, 228×228 trails, four cohorts, and matching shared physics.
Observed behavior included:

- One-click Fourier → MLP → GRU switching, reset to tick 0, preserved pause, and
  running/pausing GRU without a GPU error.
- MLP and GRU comparisons completing exactly 1,000 ticks across bounded batches,
  remaining paused with zero queued work; both reports downloaded successfully.
- Profile controls locked during a long comparison; rule hotkeys did not change
  the active mode or interrupt its tick progression. Cancellation at tick 8,880
  cleared queued work and displayed “Comparison incomplete (cancelled).”
- Saving/restoring an unchanged checkpoint reset tick 8,880 to tick 0.
- Returning to Fourier and undo/redo across modes, with resets and no GPU error.
- Reloading without a preset override restored the saved four-cohort project.
  Switching to its saved GRU bank and exporting a zero-tick comparison produced
  the exact same effective rule hash as before reload. All three bank payloads
  matched the earlier report exactly; see the
  reload verification (`artifacts/network-browser-reloaded-gru.json`).

Retained downloads: GRU comparison (`artifacts/network-browser-gru-comparison.json`)
and MLP comparison (`artifacts/network-browser-mlp-comparison.json`). Both contain
all three mode banks and effective rule hashes. Browser adapter name was withheld
by WebGPU (empty string); backend is recorded. GPU samples are asynchronous,
may describe different batch sizes, and are not whole-run averages. These two
reports do not establish a performance ordering between modes.

## Validation limits

Only this GPU/driver and Chrome were exercised. No physical out-of-memory
condition was forced; recoverable scoped validation rejection was injected.
Device loss cannot preserve a usable GPU world. Cross-device chaotic trajectories
are not promised to be identical. Native interactive controls, other browsers,
and new-mode video playback were not independently validated in this pass.
Physical camera capture is unchanged and was not activated.

The v1 feature does not include simultaneous worlds, mixed architectures,
configurable topology, training, saved runtime memory, or graph archive replay.

## Final checks

After the review fixes, the final test rerun passed. Both release targets were
rebuilt and the staged static distribution passed its asset audit.
