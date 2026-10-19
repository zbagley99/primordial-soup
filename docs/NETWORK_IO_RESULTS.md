# Network IO evaluation, 2026-10-17

Revision 3 implements [the IO specification](NETWORK_IO_SPEC.md). Generated JSON
and PNG evidence is under ignored `artifacts/network-io/` and is not distributed
in Git.

## What the baseline established

The production GPU shader was instrumented without extra network evaluations or
memory updates. Hardware: NVIDIA GeForce RTX 3090, Vulkan, driver 610.57.04.
The matched baseline uses Starcrossed shared physics, four cohorts, four seeds
(0.125, 0.25, 0.5, 0.875), 30,000 particles, 228² trails, world size 0.05, and
2,048 physics ticks after initialization. Samples occur at ticks 1,16,64,256,
1,024 and 2,048. Random Fourier searches provide context; they are not the
hand-selected Starcrossed Fourier rule.

For revision-2 MLP/GRU, median absolute vector-input maxima at tick 2,048 were
0.00167–0.01076 across the eight runs. No sampled MLP hidden units exceeded
|activation|=0.99. Thus weakly varying inputs, rather than tanh saturation,
were the leading observation problem in these configurations. GRU median
per-tick hidden changes ranged from 0.00046 to 0.00372. Low-speed rescues and
reversals varied substantially by seed; neither is unique to the new networks.

Signed trails could also hide opposing or resting populations. Strafe moves
positions independently of velocity deposits, so visible displacement did not
necessarily provide a strong vector observation. These are properties of the
original observation/motor contract, not evidence of a packing or GRU-equation
error. The initial sparse 16,384-particle/128² experiment is retained separately;
it is not used to infer normal application input scale.

Evidence: `baseline-scaled.json`, `baseline-scaled/*.png`.

## Controlled ablations

Each stage uses identical revision-3 coefficients, seeds, physics, initialization
and tick counts. Only the listed encoding/motor/orientation factors change.
This separates IO effects from the topology/initialization change between
revision 2 and revision 3.

| Stage | GRU median hidden change | GRU mean rescue fraction | MLP mean rescue fraction |
| --- | ---: | ---: | ---: |
| 1: original vector scaling; new scalar inputs zero | 0.0027 | 23.2% | 18.0% |
| 2: bounded physical flow encoding; scalars zero | 0.0007 | 25.3% | 18.7% |
| 3: add occupancy and own speed | 0.0069 | 10.9% | 1.2% |
| 4: also decode signed motors with 2*tanh | 0.0107 | 1.4% | 0.2% |
| Final: retain heading at numerical rest; inject no velocity | 0.0102 | 0% | 0% |

Hidden-change entries are medians of the four cohort-population medians at
tick 2,048. Rescue fractions are arithmetic means across the four seeds at that
tick, not cumulative lifetime rates. Final zero rescue is an intentional removal
of random velocity injection; it is not evidence that particles never stop.
Final GRU reversal fraction still averaged 66.7%; signed braking and reversal
remain valid outcomes. Scalar observations are a more consequential addition
than flow scaling alone in this experiment.

These ablations support the chosen finite motor unit and local observations;
they do not establish globally optimal scales. No automatic selection, training,
survival rule, positive thrust offset, global normalization or periodic stimulus
was added. Existing Glorot/bias distributions retain their definitions with the
new matrix dimensions. Inactive or converged random rules remain possible.

Evidence: `ablation-{1,2,3,4}.json`, corresponding PNG directories, and
`final.json` / `final/*.png`. Native captures show distinct streams, filaments,
clumps and diffuse populations depending on seed. Appearance is not a numerical
acceptance target or a ranking of network architectures.

## Correctness and compatibility

- Frozen revision-1/2 coefficient fixtures are unchanged. All **72** legacy
  particle snapshot hashes and **12** legacy rendered PNGs matched the baseline
  exactly on this device after the format/stride changes. Evidence:
  `legacy-after.json` versus `baseline-scaled.json` and their captures.
  These captures precede the separately requested white world outline, which
  intentionally changes display pixels in every mode.
- Independent Python equations check both revised evaluators, including
  nonzero GRU history. GPU checks cover four particles across three cohorts,
  both mirrored histories, 40 recurrent ticks, and extreme finite inputs.
  Maximum output error was below 3e-5 (approximately 4.2e-7 in the measured run).
- Real production updates verify that a zero controller can remain exactly at
  rest while depositing occupancy. Concentrated 4,096-particle fixtures clear
  the canvas before each requested persistence case (0.001, 0.938, 0.9999;
  effective clamps 0.01 and 0.999). The footprint is chosen to produce fresh
  representable occupancy even at the highest persistence. Opposing momentum
  cancels while occupancy stays positive. Nonzero flow/scalar encoding is
  checked against the host reference. Injected Trail Weights 0.02,1,5 preserve
  unweighted occupancy observations.
- Positive occupancy accumulation uses a bounded compare-exchange sum, avoiding
  signed atomic wrap under contention. A too-small diagnostic footprint exposed
  a quantization limit during fixture development; existing momentum deposits
  can round to zero for sufficiently slow/small particles. Occupancy does not
  claim unlimited resolution, and the positive high-persistence fixture starts
  from an empty field rather than relying on retained earlier deposits.
- Actual revision-3 engine checks exercise display-state preservation,
  adoption clearing without moving particles, reset/hazard/death clearing,
  exact coefficient picks, and same-mode revision-2↔3 pipeline/state transitions.
  Evidence: `contracts-reviewed.json`.
- Legacy analytical motion/trail and serial/batched checks pass. Field/camera
  equivalence checks pass after adapting canvas readbacks to eight-byte texels.
  Webcam intermediates retain their original RG16 vector format. Evidence:
  `legacy-numerical.json`, `legacy-networks.json`, `field-check.json`.
- All three modes also completed 256 physics ticks at 600,000 particles and
  1,024² trails, with finite observations/state and real rendered captures.
  Evidence: `full-size.json` / `full-size/*.png`. This is a short full-size check,
  not a long-run statistical survey.

## Cost and execution limits

Shared trail storage adds 12 bytes per texel: 12 MiB at 1,024². Revision-3 heading
state adds 16 bytes per particle: 9.16 MiB at 600,000. MLP now uses 16 bytes of
state per particle; GRU uses 48. Renderer targets and camera intermediates do not
grow. Preflight checks actual buffer sizes; normal application sizing remains
bounded by the particle allocation before the enlarged splat reaches the same
storage-binding limit. Arbitrary caller-supplied sizes are checked explicitly.

A separate pre-change checkout at commit 03d27f6 and the revised release binary
ran matched Fourier physics-only workloads: world size 1, 600,000 particles,
1,024² trails, 15 ticks/submission, four warmups, 12 measured submissions.
Before/after and reverse-order repeats measured mean GPU times:

| Run | Original | Revised storage |
| --- | ---: | ---: |
| First pair | 1.861 ms | 2.079 ms |
| Reverse-order repeat | 1.904 ms | 2.082 ms |

The measured legacy cost is roughly 0.20 ms per 15 ticks, about 10.5% in these
short runs. It is not an end-to-end editor FPS measurement or a performance
ranking of MLP and GRU. Instrumented observation runs are excluded from timing
claims. Evidence: `timing-{before,after}{,-second}.json`.

All 102 Rust tests, shader validation, browser service tests, native release and
optimized WebAssembly builds passed. Chrome WebGPU completed exactly
1,000 ticks for both revised models at 30,000 particles, paused with zero pending
work and no displayed GPU error. Downloaded reports identify MLP revision 3 / 100
parameters / 16 state bytes and GRU revision 3 / 164 / 48, with the effective
cohort rule hashes. Evidence: `browser-{mlp,gru}-comparison.json`.

The final Network Settings panel was opened, edited, closed and reopened in
Chrome. Both controls retained their values, and Mutation Scale matched the
existing toolbar. A mutation-only edit preserved tick 1; changing Sensor Gain
honored the existing reset-on-edit preference and reset the clock. The original
values were restored after this check. No displayed GPU error remained.

Only this GPU/driver and Chrome were exercised. Physical camera capture and
physical out-of-memory conditions were not tested. Cross-device chaotic
trajectories are not promised. How useful the new exploration space feels is a
visual judgement that these checks do not settle.

## Try the final models

Run `make run`, or use the rebuilt local browser distribution. Choose MLP or
GRU. A restored old bank keeps its original model: press **B / Randomize Behavior**
to start revision 3. The toolbar identifies **flow + occupancy + speed**.
The **Network** panel exposes existing Sensor Gain and Mutation Scale controls;
topology and gate equations stay fixed. Save an interesting rule before
randomizing.
