# Minimal simulation interface

The living system is the main view. A small command dock, adjacent cohort panel and quiet scene
label form the default interface. The previous full-width header, permanent tool
rail, boxed performance card, and green surface treatment have been replaced.
The inspector stays closed until opened. There is no continuous UI animation.

## Hierarchy and interaction

| Layer | Content |
| --- | --- |
| Observe | Full-size simulation canvas; scene name, model, and optional quiet performance/rate readouts |
| Act | Compact bottom dock: playback, Presets, interaction tools, Inspect, and More |
| Edit | On-demand inspector with World, Look, Settings, and contextual Brush controls |
| Inspect further | More contains reset/variation, files, history, recording, timing, and help |

Phone dock actions retain 44-pixel touch targets. Reset and one-tick stepping
move into More on phones; desktop also exposes Step beside playback. The dock
moves into the available scene space when the desktop inspector is open. On
phones, the inspector replaces the quiet scene label and scrolls above the dock.
The Presets picker identifies Core and Archive as Fluoddity presets by Jesse
Gelders, separately from locally saved presets. Selecting a tool preserves the canvas view. Brush settings opens explicitly
from the tool menu. Selection actions replace the scene label rather than
adding another panel. X hides all controls and leaves a small restore action.

FPS and the configured physics rate sit above playback inside the dock, visible
with the inspector or a preset menu open. Settings → Performance has independent
toggles for both. A quiet “X to hide” hint stays at the right of that row, even
when both readouts are disabled. Clicking either readout opens Settings; right-clicking Physics
starts calibration. Paused FPS is labelled as paused rather than presented as a
fresh measurement. More → Editor → Diagnostic overlay exposes the detailed
developer text separately.

A full-width physics speed bar stays between this readout row and playback,
including when both readouts are disabled. It edits steps per frame in the
frame-based clock and ticks per second in the fixed clock. X hides the whole dock.
The extra row is included in the phone inspector and dialog bounds.

An adjacent panel contains outlined 1 / 4 / 16 cohort grid buttons inspired by the
supplied sketch, with 44-pixel targets and a selected state. They restart the world
through the existing layout transaction. A small Mutation label, decimal value
and slider sit beside them. The slider previews its value during a drag and
applies once on release, retaining one undoable edit. Both controls share the
existing operation guards, preserve imported values until edited, and hide with X.
The panel sits to the left of the physics dock, aligned to the same bottom edge,
when space permits. On narrow
screens, or beside an inspector that leaves too little width, it stacks above
the dock. Inspector, dialog and notice bounds account for both panels.

Fresh application state uses the captured modified Diversity project, with Cells
selected and world size 1. The Cells appearance and shared lighting defaults match
the local Look settings captured on 2026-10-21. Existing saved values take precedence.
Fresh sessions use Physics Rate 5 and Mutation 0.2, with automatic startup
calibration disabled so it cannot replace the requested rate. Manual calibration
and saved calibration preferences remain available.
Right-clicking the Select canvas with no highlight restores the previous project
and resets particles, swarm trails and the clock after that project applies.
Dismissing a highlight, an empty history and a rejected history change do not reset.

Neutral charcoal surfaces, subtle borders, restrained corners, and ghost buttons
keep emphasis on the simulation. Text size, contrast, spacing, and section
headings distinguish scene context, common actions, and detailed controls.
Sliders have a compact label/value row and a thin full-width track below.
Default success messages about appearance and network preparation are omitted; errors and
other actionable notices remain visible.

## Preserved contracts

The full CentralPanel canvas, camera mapping, shaders, simulation, preference
schema, imported values, control ranges/curves, integer and inverted
values, gate thresholds, changed-only writes, recording guards, and project
history transactions remain. Physics drag tracking and measured cost tooltips
remain. Rendering controls without input must not clamp imported values.

The 48 existing control call arguments are unchanged. A representative field
measured 81 pixels high in the rejected interface and 46 pixels after this pass
under the same inherited egui control metrics. This is a layout measurement,
not an application performance claim.

## Validation

Hardware WebGPU captures cover desktop, 320-pixel and 390-pixel phone layouts,
the inspector, preset picker and touch scrolling, and More submenus at 844 × 390.
The longest supported fixed-rate label, 7,200 ticks/s, fits the 320-pixel dock.
Independent FPS/rate toggles were exercised; the hide hint keeps its row visible
when both readouts are disabled.
The original clipped Inspect label had 34 pixels for 41.9 pixels of text; its
new slot leaves 50 pixels. Core phone dock actions retain 44-pixel targets;
secondary inspector fields are more compact. Touch checks use browser-emulated
touch events, not a physical phone.

A matched 1440 × 900 clean simulation export at 180 ticks is pixel-identical to
the pre-redesign reference. The on-demand inspector, touch gestures and camera
changes do not establish visual acceptance by the user.

Before the persistent speed bar, measured rectangular surface bounds with the inspector closed covered 11.1% →
1.9% of the desktop viewport and 23.8% → 6.9% on the phone, including performance
and rate controls. These are frame bounds, including rounded corner cutouts;
scene text, shadows, menus and temporary messages are excluded. Source captures,
component measurements and a plot are retained in the evidence directory.

The final rebuilt WebAssembly source matches its recorded inputs before and
after compilation. The Timing dialog scrolls at 320 × 568 and 844 × 390 while
both readouts remain visible. Narrow dialogs wrap text and use shorter slider
tracks; action rows wrap instead of widening their window. Timing, Network
Settings and Screenshot were inspected on the 320-pixel viewport. Editing a
numeric field leaves playback paused, and X restores the previous controls.
The final clean export again has zero differing pixels from the reference.

Dialogs use a foreground layer so scene text and the diagnostic overlay cannot
draw through them. The final landscape overlap capture, dock menu and a dropdown
inside Project Link Settings verify that ordering. The campaign source passes
all 112 Rust checks, including three doctests.

The campaign build provenance is `layered-build-source.json`; the final browser
observations are in `layered-session.json`, with desktop, phone and close Cells
captures at 1,800 evolved ticks. Paused zoom and view reset preserve that tick
count. `accepted-session.json` records the preceding controls, dialogs and neural
mode checks. These filenames identify QA artifacts, not visual approval by the
user. Earlier `delivery-*` captures precede the phone-dialog control sizing fix.

The subsequent hide hint was checked in rebuilt desktop and 320-pixel browser
views, including the 7,200 ticks/s label and both performance readouts disabled.
X hides and restores the controls without advancing the paused simulation.
The native check and WebAssembly distribution build pass. Evidence is under
`artifacts/local/hide-hint-20261021/`.

The subsequent captured defaults and persistent speed bar pass 113 Rust checks
and the WebAssembly build. Fresh browser state loads the exported Diversity
project in Cells at world size 1. Right-click backtracking resets an evolved
canvas from 2,805 completed ticks to zero; ordinary undo/redo, highlight dismissal
and empty history preserve progress. The bar edits both clocks while paused.
At 320 CSS pixels, the inspector clears the bar, including with both optional
readouts disabled. X hides and restores the controls without advancing the clock.
These checks use the intentionally updated appearance defaults, so the earlier
pixel-equality results describe the preceding UI campaign. Build provenance and
observations are under `artifacts/local/cell-defaults-20261021/`.

Cohort controls were checked on hardware WebGPU at desktop, 320 × 568 and
844 × 390. The 1 / 4 / 16 buttons produce the matching grids and restart time;
mutation release outside the track applies the final value, and one undo restores
the prior scale while paused. Both panels hide with X. The short-landscape Guide
scrolls above the controls. The 113 Rust checks, final bottom-alignment regression
and web build pass; evidence is under `artifacts/local/cohort-controls-20261021/`.

Local evidence is under `artifacts/local/ui-minimal-20261020/`. Entry source,
diff, and the rejected web build are archived before edits. No simulation FPS
improvement is claimed for the interface redesign.
