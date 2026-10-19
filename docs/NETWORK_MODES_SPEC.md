# Selectable network modes

Status: approved scope, implemented. See [implementation evidence and limits](NETWORK_MODES_RESULTS.md).

The four-input architectures below describe saved revisions 1 and 2. New searches
use [revision 3 observations and motors](NETWORK_IO_SPEC.md); older banks retain
their equations until the user starts a fresh search.

## Purpose

Explore how changing a particle's brain changes the behavior of the swarm.
Keep the sensors, movement, trails, tools, and rendering consistent so the
network is the main variable being explored.

The first version offers **Fourier / MLP / GRU** in one world. Switching modes
restores that mode's rule and restarts the simulation. This compares a periodic
response, a conventional feedforward network, and a network with individual
memory. It does not translate one network into an equivalent network of another
type.

This document proposes one world with a selector as the initial scope.
Side-by-side independent worlds and mixed architectures interacting in one
world are later features. Also deferred: training, fitness scoring, editable
network sizes, CNNs, graph networks, cellular automata, and runtime snapshots.

## User experience

Add a **Network** selector alongside the existing behavior controls:

| Mode | Description shown to the user | Parameters | Memory per particle |
| --- | --- | ---: | ---: |
| Fourier | Original sine/cosine behavior rule | 80 | None |
| MLP | Feedforward network, 4 inputs → 8 hidden → 4 outputs | 76 | None |
| GRU | Recurrent network, 4 inputs → 4 memory units → 4 outputs | 128 | 8 floats / 32 bytes |

The GRU has four memory units in each of two mirrored evaluation streams,
explained below. Parameter counts include weights and biases; mirrored
evaluations share weights and do not double the parameter count.

Existing projects open in Fourier mode. Selecting another mode for the first
time creates a random search for that architecture. Returning to a visited
mode restores its parent rule, generation state, mutation seed, and Mutation
Scale. Sensors, world settings, movement settings, layout, and cohort count
are shared across modes rather than silently restored with each brain.

The selector always resets particles, clears the swarm's trails and GRU memory,
and returns the logical clock to tick 0, while preserving pause state. This
overrides the ordinary reset-on-behavior-change preference. Explain this beside
the selector. Ordinary switching keeps painted fields; comparison restart clears
them explicitly. Camera/view settings stay put.

The selector adds one undoable project change. Undo restores project settings
and rules, then resets when the restored mode differs; it does not rewind the
simulation. Block mode changes during recording and calibration. Cancel stale
asynchronous particle picks and previews when a mode changes. A switch must
either succeed completely or leave the current project and simulation intact.

No new keyboard shortcuts are required. Existing Randomize (`B`), Reroll (`F`),
Adopt (`Enter`), mutation brackets, undo/redo, and checkpoints work in every mode.

## Shared sensing and movement contract

Every mode receives the same four local-coordinate sensor values:

`x = [left.forward, left.sideways, right.forward, right.sideways]`

Every mode returns four raw values:

`y = [force.forward, force.sideways, strafe.forward, strafe.sideways]`

Keep the current conversion to world coordinates, force scaling, drag, strafe
power, gravity, tool displacement, fences, boundaries, deposits, and trail-update
order. Keep the current color derivation from the two evaluations. Cohort color
remains a display option. Do not add mode-specific movement multipliers hidden
from the user.

Fourier retains its existing evaluator, generator, mutation arithmetic, and
zero-rule sentinel exactly. The new modes do not reinterpret those coefficients
or change the legacy behavior to match their output range.

### Mirrored evaluation

The current rule runs twice: once on the sensor readings, once with the sensors
swapped and their sideways components reflected. Its movement outputs combine
with the same reflection to avoid a built-in handedness.

Preserve that contract in all modes. MLP evaluates twice with shared weights.
GRU maintains **two independent hidden vectors per particle**, using the same
weights: one for original inputs and one for mirrored inputs. Each advances
exactly once per physics tick. Never update a shared hidden vector twice in
sequence: that would make the result depend on evaluation order.

Rendering, picking, pausing, and pipeline preparation must not advance memory.
Simulation-only ticks and recording ticks advance it normally.

## Fixed v1 architectures

These are proposed fixed definitions, not runtime configuration options.
Architecture identifiers include a revision so their interpretation cannot
change silently in saved projects.

### Fourier v1

The existing ten Fourier terms and 80 coefficients, unchanged.

### MLP v1

`h = tanh(W1 x + b1)`

`y = W2 h + b2`

Shapes: W1 is 8×4; b1 has 8 values; W2 is 4×8; b2 has 4 values.
Total: 32 + 8 + 32 + 4 = **76 parameters**.

### GRU v1

For each of the two streams, with four hidden values:

```text
z = sigmoid(Wz x + Uz h + bz)
r = sigmoid(Wr x + Ur h + br)
n = tanh(Wn x + Un (r * h) + bn)
h_next = z * h + (1 - z) * n
y = Wo h_next + bo
```

Multiplication between vectors is componentwise. This fixes the reset-before
candidate convention and makes a larger update gate retain more previous
memory. All W/U matrices are 4×4 and each bias has four values.
Total: 3 × (16 + 16 + 4) + 16 + 4 = **128 parameters**.

All memory starts at zero. Memory is private to each particle; cohorts share
weights, not hidden states. Memory duration is measured in simulation ticks,
so changing ticks per second changes its duration in wall-clock time.

This uses a gated recurrent unit based on the architecture introduced by
[Cho et al.](https://arxiv.org/abs/1406.1078); the equations above define this
app's specific variant.

### Initialization and mutation for MLP/GRU

Generate and mutate new-mode coefficients on the CPU only when rules change,
then upload finished cohort rules. Persist canonical f32 coefficient values.
Use a versioned integer PRNG with fixed test vectors and independent streams
for architecture, cohort, tensor, and coefficient. Freeze its exact seed mapping,
coefficient order, and distribution before the first format fixtures land.

Proposed initialization: Glorot-uniform matrices; MLP biases uniform in
[-0.25, 0.25]; GRU reset bias zero, update bias +1, and candidate/output biases
uniform in [-0.25, 0.25]. Nonzero biases allow activity when the trail field is
empty. These are exploration defaults, not a promise of interesting behavior.

For a parent rule, add independent uniform noise to each coefficient:
matrix delta in `[-s/sqrt(fan_in), +s/sqrt(fan_in)]`, bias delta in
`[-0.25s, +0.25s]`, where `s` is Mutation Scale. Clamp new-mode coefficients
to [-8, 8] after initialization/mutation and reject imported coefficients outside
that range. Use stable sigmoid/tanh evaluation and finite checks. These bounds
do not apply retroactively to Fourier.

Keep Mutation Scale's current UI range and shortcuts, but preserve its value
per mode. Equal slider values do **not** imply equal behavioral changes across
architectures. Do not claim calibrated equivalence. Initialization quality and
mutation sensitivity need a documented visual review before release; changes
after format stabilization require a derivation revision.

## Cohorts and recurrent-state lifecycle

One architecture is active for the whole simulated population. Each cohort gets
its own rule, with the existing equal population split and shared trail field.
Keep the existing layout and fence semantics.

Each mode bank has an explicit rule source: **random search** or **parent rule**.
For new modes, all-zero coefficients are a valid rule, never a generation flag.
Random search generates independent rules for cohorts; Mutation Scale does not
apply until adoption. Reroll is inactive during random search and at zero scale,
matching the current controls. Adoption copies the selected cohort's actual
coefficients as the parent and derives children from it. Only weights are adopted;
a particle's memories are not inherited.

| Event | GRU memory behavior |
| --- | --- |
| Tick | Advance both streams once per live particle that reaches behavior evaluation; respawn ticks initialize instead. |
| Pause, redraw, pick, hide, or resume | Preserve; no update without a tick. |
| Full reset, load, mode switch, world resize, checkpoint restore | Zero all memory. |
| Adoption, reroll, randomize, active weight change, or cohort reassignment | Zero all memory, even when positions are preserved by the reset preference. |
| Individual hazard/boundary respawn | Zero that particle's two streams before its next evaluation. |
| Particle death | Clear its memory; dead particles do not evaluate. |
| Display-only change or unrelated settings upload | Preserve; uploading a config alone is not a memory-reset reason. |

Undo/redo applies the corresponding mode/rule lifecycle above. Checkpoints and
project saves store settings and weights, not positions, trail contents, or GRU
memory. State this in the UI/docs instead of implying they resume a running world.

## Comparison workflow

Provide **Restart comparison** and **Run N ticks** in Timing & Experiments.
Use the existing exact-step scheduler rather than a separate simulation loop.

Restart comparison pauses, cancels queued exact work and elapsed clock debt,
initializes the selected mode at tick 0, zeros recurrent memory, clears swarm
trails **and painted Walls/Trails**, and disables webcam mapping and automatic
calibration. Ignore live drawing/shove input during an exact comparison run.
Explain field clearing on the control. Do not activate or stop camera capture
implicitly; disabling its influence is sufficient. Retain view and visual settings.

For the same world size, cohort count, and layout, use the current initialization
algorithm to start all modes at the same positions and velocities on that device.
Do not introduce a configurable world seed in v1: current reset is derived from
particle identity/cohort and changing its hashing would disturb legacy behavior.
Network generation seeds remain separate and are recorded.

Run N ticks starts from that clean baseline, advances exactly N ticks, and stays
paused. Restarting resets time-dependent random inputs to their original tick
indices. Overload may take longer but must never reduce the requested work.
Hidden/minimized suspension retains the existing scheduler behavior.

Reject comparison starts during recording or calibration. While comparison work
is pending, lock mode, rule, physics, world, cohort, load, undo/redo, checkpoint
restore, and timing edits; also block recording/calibration starts. View and
display-only controls remain available. An explicit Cancel stops remaining ticks,
leaves the partial result paused, and marks it incomplete; it must not emit a
completed comparison record. Restarting cancels any prior run before creating
the new baseline.

Show/export a small comparison record containing architecture/revision, parent
or random-search state, rule seeds and coefficient hash, Mutation Scale, shared
physics/world settings, effective dimensions and particle/cohort counts, tick
count, timing/batch settings, rendering settings, adapter/backend, and any
profiling results available. Include all mode banks in the project used to
reproduce it. Mark the result stale after a relevant edit. Existing screenshot
and video exports remain the way to capture appearance.

Compare at equal **completed tick counts**, not equal seconds spent rendering.
Report parameter count, recurrent-memory overhead, and measured timing alongside
the visuals. Keep automatic calibration off for timing comparisons, match
workloads, and separate particle-update timing from total frame cost. Do not
promise bit-identical trajectories across devices or call this a benchmark of
intrinsic network quality: architecture, initialization, weights, and mutation
all affect the result. There is no automatic cross-architecture weight mapping.

## Data and compatibility

Use a typed network definition rather than stretching `rule: Vec<f64>` to mean
different things. Each configuration owns its active-mode identifier and lazily
created mode banks. A bank contains architecture/derivation revision, rule source,
parent coefficients when present, generation/mutation seed, and Mutation Scale.
Shared sensor/physics settings stay outside the bank. Preserve inactive banks
through every persistence path.

The current engine uses the first configuration to build cohort slots; it does
not simulate the configs array as independent species. This feature must not
accidentally introduce a config selector or promise mixed species. Keep import
round-tripping of additional configurations intact and make the UI's active
configuration agree with the one being simulated.

- Read v8/v9 projects as Fourier, preserving existing migration behavior.
- Use a new project format, proposed **v10**, whenever network banks or comparison
  metadata exist. Preserve name/notes and relevant mode state explicitly.
- Keep legacy v9 output and original Fluoddity links only for projects with no
  extra state to lose. Provide an explicitly labeled active-Fourier-only export
  if users need to discard extra banks for compatibility; never do so silently.
- Add a distinct app-specific share envelope for v10, targeting this app rather
  than Fluoddity. JSON, links, QR import/export, custom saves, browser/native
  autosave, ZIP backups, and checkpoints must preserve all banks. Enforce bounded
  decoding before allocations. If a QR payload is too large, explain the limit
  and offer JSON; never drop inactive modes to make it fit.
- History snapshots, comparisons, and archive hashes include architecture,
  revisions, and all saved banks. Keep legacy hash identity for legacy projects;
  use a separate version/domain for extended state.
- Extend archive logging/export with an app-specific version for extended projects.
  Store exact adopted coefficients and derivation identifiers; never rebuild an
  MLP or GRU through Fourier's GPU derivation. Keep existing legacy archive output
  for legacy projects. The current app has no graph archive replay/import UI;
  adding one is outside v1. Validate that emitted full project states can be
  decoded losslessly, without promising graph replay.
- Reject unknown modes/revisions, wrong tensor lengths, out-of-range or nonfinite
  values, and oversized payloads with a clear error. No silent Fourier fallback.

## GPU and implementation boundaries

Keep the current 32-byte Entity layout. GRU state goes in a separate storage
buffer with two vec4 values per particle, allocated only for GRU; non-recurrent
pipelines use no full-size memory allocation. Account for 32 extra bytes per
particle, e.g. 32 MB for one million particles, and check device limits before
switching. An allocation failure must preserve the previous mode.

Keep the Fourier config packing and GPU mutation path intact. New modes use
separate typed weight storage and mode-specific update pipelines; their finished
CPU-derived rules bypass Fourier's cohort bake. Do not mutate twice. Sharing
sensing/integration helpers should not change the legacy arithmetic order.

Picking returns an architecture-tagged, exact coefficient payload from the
active cohort. Adoption, logging, and export use that payload rather than
reconstructing a potentially different rule. Requests carry a project/mode
generation identifier so stale readbacks cannot adopt a rule into a newer world.

Inspect every pipeline binding and adapter storage-buffer limit, including
browser WebGPU. Keep hardware-only execution and existing tick/render separation.
Preparing a pipeline must not execute settling ticks or leave a partially applied
mode switch. Profile before choosing optimizations or claiming mode costs.

## Delivery sequence and acceptance

1. **Formats and contract:** typed models/banks, immutable layout fixtures,
   legacy migrations, extended persistence/share/archive-export formats, reference
   evaluators, versioned generator/mutation fixtures, and lifecycle decisions.
2. **MLP:** GPU evaluation, cohort derivation, exact picking/adoption, selector,
   and UI/persistence integration. Prove Fourier remains unchanged.
3. **GRU:** dual memory streams, GPU evaluation, lifecycle integration, capacity
   handling, and analytical recurrence checks.
4. **Comparison:** clean restart, exact ticks, comparison records, and matched
   performance measurements using existing profiling facilities.
5. **Release review:** native and browser interaction, rendered evidence,
   documentation, and a final source review.

Required evidence before calling the feature complete:

- Existing Fourier format/codec fixtures, presets, numerical checks, and controls
  pass. Its shader arithmetic and legacy generator/mutator remain unchanged.
- Independent CPU references check GPU MLP output and multi-tick GRU memory/output
  against small fixed fixtures with stated tolerances; include mirrored-stream
  exchange and absence of evaluation-order dependence.
- Same-device serial and batched ticks agree within justified tolerances;
  drawing/picking without ticks leaves memory unchanged. Run finite-value tests
  at supported limits and repeated adoption/mutation bounds.
- Every reset/death/respawn/adoption/cohort/undo/checkpoint path follows the memory
  table. Display-only changes and incidental config uploads preserve memory.
- GPU-picked/adopted coefficients match the active cohort exactly. Reject stale
  readbacks during rapid switching and verify switch failure is atomic.
- Round-trip every bank through native/browser storage, JSON, share, ZIP,
  and checkpoint/history; validate exact full states and extended hashes in
  archive output. Reject malformed/unsupported project and share payloads.
- Clean comparison restart yields matching initial particle/trail state across
  modes on the same device. Exact tick counts survive batching and suspension.
  Pending comparisons reject conflicting edits/work, and cancellation cannot
  produce a completed record.
- Native and browser users can switch, breed, save/reload, undo, and capture all
  three modes. Retain screenshots/clips and adapter-tagged measurements. Do not
  substitute a successful shader compile for visible behavior verification.

The approved scope is implemented with fixed architecture equations and versioned coefficient derivation. The
selector uses three direct buttons to avoid popup click-through in the floating
toolbar. Canonical tensor layouts and independent SplitMix32-derived streams are
frozen in `src/network.rs` and the revision-1/revision-2 coefficient fixtures. Existing
scalar fields act as active seed/scale adapters; exported bank snapshots are
normalized and imports reject inconsistent adapters. New modes support at most
4096 cohorts. See the results document for executed validation and limits.

## Current source map

- [Config types, packing, and project formats](../src/config.rs)
- [GPU pipeline, cohort uploads, and picking](../src/gpu.rs)
- [Particle evaluator and mirrored behavior](../shaders/entityUpdate.wgsl)
- [Legacy generation and mutation](../shaders/rule.wgsl)
- [Legacy cohort bake](../shaders/cohortRules.wgsl)
- [Entity and config GPU layouts](../shaders/common.wgsl)
- [Editor actions and lifecycle](../src/app.rs)
- [History](../src/history.rs), [sharing](../src/share.rs), and [archives](../src/archive.rs)
- [Tick scheduling](../src/simulation.rs) and [numerical checks](../src/numerical.rs)
- [Plain-language simulation explanation](DETAILS.md)

## Review fixes (2026-10-16)

New MLP/GRU banks use derivation revision 2. Initialization and mutation have
separate RNG domains. Adoption chooses a fresh seed (with a different folded
32-bit stream identity even if the proposed seed collides), and upgrades an
adopted revision-1 bank to revision 2 using its exact effective coefficients.
Existing revision-1 banks retain their frozen derivation until adoption; loading,
switching, and exporting do not rewrite them. Fourier stays revision 1.
Architecture equations, tensor layout, project v10 and PS10 encoding stay the same.
Both derivation revisions round-trip and have frozen bit fixtures. Comparison
reports label the actual active bank revision; archives include both revisions.

For MLP/GRU, supported Sensor Gain is 0..1,000,000 (the UI range remains 0..8).
Imports and switches reject larger gains. Before sensor multiplication, finite
sensor taps and the world-size/gain operands are bounded to 1,000,000 in magnitude;
network inputs saturate to ±1,000,000 before any matrix operation. The CPU reference
uses the same input saturation. With valid coefficients and initially zero memory,
GRU memory remains in [-1,1], with raw outputs bounded by 40; MLP outputs are bounded
by 72. These guarantees concern finite network inputs and the identified sensor
overflow, not every extreme legacy physics setting or recovery of existing NaNs.
Fourier sensor arithmetic is unchanged.

Same-mode saved-project previews use transactional GPU preparation on both native
and browser builds; the browser awaits scope completion asynchronously. Hovering another candidate or leaving cancels obsolete pending
previews. Preview/restore operations do not record history. Autosave and exit save
retain the original project until an explicit load/edit successfully commits;
undo history for that commit starts from the original. Preparation failures leave
the current project and original-preview backup intact.
