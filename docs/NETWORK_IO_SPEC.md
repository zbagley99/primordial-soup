# MLP and GRU observations and actions, revision 3

Status: implemented and evaluated, 2026-10-17. This revises the new-model
contract in NETWORK_MODES_SPEC.md; Fourier and revision-1/2 saved rules retain
their mathematical interpretation. Numerical and visual results are recorded
separately in NETWORK_IO_RESULTS.md.

## Outcome and scope

Give the small feedforward and recurrent networks meaningful local observations
and explicit motor units. Success means observable sensitivity to neighboring
particles and, for GRU, history; finite reversible state; reproducible legacy
projects; and measured comparisons through the actual GPU update path.
Interesting structures are assessed visually, not enforced by a fitness function.
An inactive, dispersed, or converged random rule remains a valid outcome.

The Network settings panel exposes the existing Sensor Gain and Mutation Scale settings
and explains the current observations. Their persistence, undo and reset behavior
remain the same as the corresponding controls in the main panel/toolbar.

Do not add attraction, survival, positive-only thrust, minimum speed, periodic
inputs, global population normalization, automatic gain adjustment, or preferred
patterns. Do not train or select weights automatically. Keep world settings,
sensor positions, tools, rendering, and user-driven selection/mutation.

## Measurements before changing behavior

The native `gpu-check --network-evaluation` runner instruments the actual entity
update shader without extra evaluations or recurrent updates. Four network seeds,
four cohorts, and equal completed tick counts sample local sensor inputs, raw
outputs, world acceleration/strafe, speed, low-speed events, reversals, hidden
changes and gates. It also captures occupancy dispersion and real rendered PNGs.
Instrumented execution is not used for timing claims. A sparsely populated
diagnostic world and the application's normal world-size scaling are recorded
separately; conclusions about typical input scale use normal scaling.

## Observations

Revision 3 has seven inputs in this order:

`[left.forward, left.sideways, right.forward, right.sideways,
  left.occupancy, right.occupancy, own.speed]`.

Occupancy is a decaying **occupancy trail**, not instantaneous density. Every
live particle deposits positive mass independent of velocity using the same
footprint, spatial stochastic kernel, persistence and diffusion as its vector
trail. Oppositely moving particles can cancel vector flow while remaining
observable through occupancy. Painted trails and camera vectors affect flow,
not particle occupancy. Resting particles still deposit occupancy.

Let `A` be the expected total footprint contribution from the existing deposit
kernel, `N` the allocated particle count, `T` the trail texel count, and `P` the
clamped persistence. For the initial particle size,
`rho_ref = (N / T) * A / P`. This is the expected uniform occupancy-trail value
under the existing decay-then-add-premultiplied-deposits convention. It depends only on known
world dimensions/population and user persistence, never observed population
statistics. Hazard/death does not automatically renormalize it.

Let `v_ref = 0.001 / sqrt(world_size)` in world units per physics tick.
At each sensor, let `m` be physical occupancy and `p` the physical vector trail.
Encode local flow from `gain * p / ((m + rho_ref) * v_ref)` with
`softsign(a) = a / (1 + abs(a))`, componentwise after local rotation.
The additive reference avoids division by zero and smoothly reduces confidence
in nearly empty measurements. It is not unit-vector normalization.

Encode occupancy as `m / (m + rho_ref)` and speed as
`|velocity| / (|velocity| + v_ref)`. Empty occupancy/speed encode exactly zero;
uniform occupancy encodes 0.5 within the representable field range. Flow retains sign; scalar observations are
nonnegative. Sensor Gain controls vector signal strength. It does not change
the meaning of occupancy or proprioception. Finite clamps precede products to
avoid overflow on extreme accepted external values.

The mirrored evaluation swaps both sensor pairs, negates the two sideways flow
components, and preserves own speed. GRU keeps independent mirrored histories.
No world-position, global time, cohort identity, or external noise is an input.

## Networks and actions

MLP remains one tanh hidden layer of width 8: 7 → 8 → 4, **100 coefficients**.
GRU remains four memory units and the existing reset-before equations:
7 → 4 memory → 4, **164 coefficients**. Matrix rows are output units; row-major
input widths and tensor offsets are revision-dependent. Canonical cohort upload
stride is 192 floats; picked/exported rules contain only actual coefficients.
Legacy coefficient generators remain bit-exact. New matrices use Glorot-uniform
with their actual dimensions; initial bias distributions remain unchanged.

Apply `2 * tanh(logit)` to each revision-3 output before mirrored combination.
Outputs represent signed local acceleration and signed local strafe, with the
existing axial/lateral controls and `/400` acceleration, `/20` displacement
conversion. Summing mirrored evaluations permits each combined local component
in [-4,4]. There is no positive propulsion offset. The factor 2 is a fixed motor
unit, subject to the documented ablation; it is not a live calibration.

Velocity trails continue to encode velocity, not total strafe displacement.
Occupancy makes strafe-only movement observable without redefining velocity.

## Low-speed orientation and state

Revision 3 removes the legacy random minimum-velocity injection. Store the last
nondegenerate heading (speed above `1e-10 / sqrt(world_size)` world units
per tick). Below this numerical floor, retain that axis while leaving velocity unchanged, including
exactly zero. Initialize the axis from reset velocity. Signed negative force
may still brake or reverse a particle; this is allowed. This is heading memory,
not an independently steerable body orientation or an angular motor.

Reuse the separate brain-state buffer: one vec4 per revision-3 MLP particle;
three vec4s per revision-3 GRU particle (two hidden vectors, one heading).
Reset/respawn initializes heading and zeros hidden state. Death clears state.
Rule changes with positions retained clear hidden state and allow heading to
recover from current velocity; exact rest receives a deterministic initial axis.
Pause, drawing/rendering, picks and unrelated uploads do not advance this state.

## Persistence and resources

Bank revision defines input encoding, tensors, motor decoding and state layout.
Revision-1/2 adoption remains in those semantics (rev1 may advance to rev2 for
the already-established mutation-stream correction). Randomize Behavior starts
a fresh revision-3 search for MLP/GRU; switching to a saved legacy bank restores
it unchanged. UI identifies legacy versus current observations. Undo, previews,
checkpoints, JSON, ZIP, links and archives preserve bank revision and weights.
Unknown revisions or coefficient lengths are rejected, never reinterpreted.

The shared trail texture becomes RGBA16Float; xy keeps legacy vector arithmetic,
z stores occupancy and w is unused. Splat storage uses three scalar components
per texel. This avoids replacing cached simulation/render bindings during
project transactions. Extra storage is 12 bytes per trail texel across the two
textures and splat, plus revision-3 heading state (16 bytes per particle).
Occupancy uses its own fixed scale, 65,536 counts per physical mass unit.
A compare-exchange accumulator saturates at 7,680,000 counts (60,000 canvas
units after scaling by 512). Each add and the sum fit in signed i32; density
cannot wrap under contention. The same canvas saturation limits apply at
extreme concentrations/persistence. Preflight validates actual storage limits. Legacy simulation hashes and rendered
comparisons must pass; measure the resource/timing cost explicitly.

## Validation and acceptance

1. Retain the original GPU measurements before implementation.
2. Compare staged ablations with identical revision-3 coefficients: original
   flow encoding; new flow encoding; added occupancy/speed; bounded motors;
   retained heading. Never promote an isolated zero-field recurrence to a
   reproduction of a spatial simulation failure.
3. Check CPU/GPU evaluator parity and independent analytical cases for all
   revisions, both reflected streams, nonzero histories and extreme finite data.
4. Test occupancy accumulation at rest and with cancelling velocities, symmetric
   encoding, zero signed motors, memory lifecycle, mode/revision transitions,
   exact adoption/picking, and persistence. Keep old coefficient fixtures frozen.
5. Compare multiple seeds at fixed ticks and inspect rendered results. Include
   normal application sizing and a full-size check; report converged outcomes.
6. Run project tests, native/WebAssembly builds, legacy GPU numerical checks,
   new-model GPU checks and browser interaction checks when available. Keep
   visual assessment separate from numerical acceptance.
