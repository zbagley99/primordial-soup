# How Primordial Soup works

**Each dot senses the trails around it, follows its group's movement rule, and
leaves new trails. We vary those rules and select interesting results.**

That's the foundation. The shapes emerge from that feedback loop: particles
change their surroundings, then react to the surroundings they collectively
created.

## The dots and their trails

The moving dots are particles. Each has a position and velocity; they move
through continuous space rather than hopping between fixed screen pixels.
The trail field is a grid of pixels storing a two-dimensional vector at each
location: both a direction and a strength, rather than just brightness.

On each simulation tick, a particle:

1. Places two sensors to its left and right, relative to its direction of travel.
   Their angle and distance are adjustable, so they can look ahead, beside it,
   or behind it.
2. Reads the trail vectors at those two locations.
3. Runs those readings through its cohort's behavior rule.
4. Applies the resulting acceleration and direct movement, along with drag
   and any enabled external influences.
5. Deposits a signal based on its velocity at its new location.

After the particles update, the trail field spreads into neighboring pixels
and fades according to its diffusion and persistence settings. The next tick
reads that updated field.

Particles interact through this shared field. There is no particle-to-particle
collision calculation or built-in instruction to form a flock, spiral, or cell.
Opposing trail vectors can cancel; aligned vectors can reinforce each other.
Different behavior rules can respond to the same signal in different ways.

## What the behavior network actually is

The original/default mode is a **small Fourier network**, not an RNN. Here, Fourier means it combines
sine and cosine functions to make a flexible mathematical response to the
sensor readings.

- **Four inputs:** two components of the left sensor's vector and two of the
  right sensor's vector, expressed relative to the particle's heading.
- **Ten mathematical terms:** each has four frequency coefficients and four
  amplitude coefficients, for **80 adjustable numbers** in total.
- **Four raw outputs:** combined into a two-dimensional acceleration and a
  two-dimensional direct displacement, called *strafe*. The calculation also
  supplies a color signal for display.

The rule is evaluated for both the original readings and their mirrored
counterpart. Combining those results keeps motion from having a built-in
clockwise or counterclockwise preference.

The Fourier network has **no internal memory and no learning during simulation**.
For a fixed rule, identical sensor inputs produce identical raw outputs.
Memory lives in the particle's existing velocity and in the lingering trails.
The whole simulation has feedback; the network itself isn't recurrent.

## Switching the particle brain

The toolbar also offers **MLP** and **GRU** modes. All three use the same sensors,
movement outputs, and shared trail field:

| Mode | Brain | Internal memory |
| --- | --- | --- |
| Fourier | Ten sine/cosine terms, 80 coefficients | None |
| MLP | Four inputs → eight tanh neurons → four outputs; 76 parameters | None |
| GRU | Four inputs → four gated memory units → four outputs; 128 parameters | Four values per mirrored stream, eight per particle |

GRU is recurrent: its next decision depends on its sensor readings and its own
previous hidden state. Each particle has separate memory even when it shares
weights with its cohort. Adoption inherits the weights and clears memory.
None of the modes trains itself while running.

Switching modes restores that mode's saved rule and mutation settings, then
resets particles, swarm trails, and memory. Shared physics settings stay in place.
Use **Timing & Experiments → Run N ticks** for a comparison from clean starting
conditions. That action also clears painted fields and disables camera influence.
The same tick count makes a useful controlled comparison, but the three networks
have different capacities and no automatic equivalent-weight mapping.

## Changing the visual style

**Style → Classic / Cells** changes how the same world is drawn. It works with
Fourier, MLP and GRU and does not alter their rules or advance a paused world.

Cells counts particle positions into a separate density grid, adjusts visible
particle size, and draws color plus intensity. Small intensity slopes bend and
shade the image; a blurred copy adds soft background color. The result is pigment
on a dark background with the normal particle colors by default. **Light**
keeps the pale paper look; the green/pink palette is an optional advanced
setting. Smooth Gaussian filtering and cubic reconstruction keep the color wash
from showing its coarse grid as square patches. These apparent cells are a
visual effect, not new simulated organisms.

Cell Scale, Density Response, Cell Relief and Color Wash control the look.
Recordings and clean PNGs use the same style. Cells costs more GPU time and memory
than Classic; see the [rendering details](CELLULAR_RENDERING_SPEC.md) and
[measured validation](CELLULAR_RENDERING_VALIDATION.md).

## What a cohort is

A cohort is a group of particles sharing one behavior rule. Its members can
move differently because they have different positions, headings, and sensor
readings.

The cohort count splits the particle population into approximately equal
groups. Membership comes from particle identity, rather than where a particle
currently happens to be. Particles don't switch cohorts by crossing into
another group's area.

Usually, each cohort receives a different mutation of one parent rule. All
cohorts share the same trail field, so they can influence one another.
Grid, center, random, and ring layouts control where particles start.
Optional **Cohort Fences**, available with the grid layout, softly pull particles
back toward their starting areas; they don't create separate trail fields.

## Mutation and selection

This is an interactive breeding process: **you choose which behaviors to
explore further**. There is no automatic fitness score or winner.

| Control | What it does |
| --- | --- |
| Cohort count | Sets how many behavior variants share the population. |
| Mutation Scale | Sets how far the variants depart from the parent rule. Zero gives the same parent rule to every cohort. |
| Reroll Mutations (`F`) | Generates another set of variations around the current parent. |
| Adopt (click the highlighted cohort again, or press `Enter`) | Makes that cohort's actual rule the new parent, then derives children from it. |
| Randomize Behavior (`B`) | Starts a fresh random-rule search. |

Mutations change the network's amplitudes and frequencies. They stay fixed
while the simulation runs; particles don't continually mutate as they move.
Adopting changes the shared parent rule, rather than making the selected
particles physically reproduce. Whether a behavior change also resets particle
positions depends on the reset-on-behavior-change preference.

A fresh random-rule search generates independent rules for the cohorts.
Mutation Scale takes effect once you adopt a rule as the parent; it doesn't
modify those initial random rules. Reroll Mutations is inactive during that
initial search and when Mutation Scale is zero.

## Other controls that shape the result

- **Sensors:** distance, angle, gain, and optional jitter change what particles
  perceive. Gain controls how strongly they hear the field.
- **Movement:** force strength, strafe strength, and drag change how the rule's
  outputs translate into motion.
- **Trails:** persistence and diffusion control how long the shared memory
  lasts and how widely it spreads.
- **Boundaries:** particles can wrap around, bounce, reset, or die when leaving
  the world. A reset restores the population; Hazard Rate can also randomly
  respawn individual particles during a run.
- **Interaction:** Shove directly displaces particles. Painted Walls also
  directly displace them; painted Trails alter what their sensors read.
  Webcam input can feed either kind of influence, and gravity settings can
  affect acceleration, direct movement, or sensor readings.
- **Timing:** Physics Rate controls ticks per displayed frame in frame-based
  mode. Fixed ticking instead requests a number of simulation ticks per second,
  independently of presentation FPS.

## What the colors mean

Ordinary particle color comes from the behavior calculation. **Color by Cohort**
instead labels group membership. Switching that display option doesn't change
the movement rules. Trail view visualizes the underlying vector field; bloom
and motion blur affect presentation rather than the behavior network.

## Source pointers

The implementation behind this explanation is in:

- [Particle sensing, movement, and deposits](../shaders/entityUpdate.wgsl)
- [Rule generation and mutation](../shaders/rule.wgsl)
- [Per-cohort rule preparation](../shaders/cohortRules.wgsl)
- [Trail diffusion and fading](../shaders/canvas.wgsl)
- [Adoption, rerolling, and editor actions](../src/app.rs)

See [REFERENCE.md](REFERENCE.md) for provenance and [the README](../README.md)
for running the app and its controls.

### Reliability and saved rules

MLP/GRU adoption starts a fresh mutation generation. New banks use derivation
revision 2, separating initialization randomness from mutation randomness; existing
revision-1 saves reproduce until adopted. Fourier is unchanged. Extreme finite
sensor inputs are saturated before network math, and unsupported sensor gains are
rejected. The GRU gates use the original reset-before formulation, checked against
independent dense reference fixtures. See [validation results](NETWORK_MODES_RESULTS.md).
