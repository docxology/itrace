# iTrace interface design

## 0. Research log and direction

The existing dependency-free dashboard supplies the visual language: light
instrument panels, blue actions, restrained status colour and system type.
The study view extends those primitives. It is an experimental instrument,
with the stimulus occupying the main visual field and setup/quality information
in a supporting pane. No remote fonts, UI framework or decorative animation.
The [supporting-pane pattern](https://github.com/changeroa/StyleGallery/blob/main/patterns/split-sidebar/supporting-pane.md)
informs reflow and source order; its prose/code is not vendored.

## 1. Users and task

An investigator prepares a private protocol and checks acquisition; a participant
then calibrates, answers trials and can stop at any time. The initial deployment
is a one-person self-experiment. Readiness is an observed state, never a default
green badge. Keep software checks, gaze quality and behavioral outcomes distinct.

## 2. Colour and material

These existing dashboard variables are the shared palette:

| Token | Value | Purpose |
| --- | --- | --- |
| `--ink` | `#18212f` | Primary text |
| `--muted` | `#607080` | Secondary text |
| `--line` | `#d7dee8` | Neutral separators |
| `--panel` | `#ffffff` | Instrument surface and stimulus background |
| `--page` | `#f4f7fb` | Page background |
| `--blue` | `#0072b2` | Primary action and links |
| `--green` | `#009e73` | Supporting success indicator |
| `--orange` | `#e69f00` | Supporting warning indicator |
| `--red` | `#d55e00` | Supporting error indicator |
| `--ok-bg` | `#eef8f4` | Success wash |
| `--warn-bg` | `#fff7e6` | Warning wash |
| `--bad-bg` | `#fff0ec` | Error wash |

Status text stays ink-coloured on its wash and includes a label. Panels use
neutral hairlines and an 8px radius. No coloured edge stripes or decorative
gradients. The task image retains its actual stimulus colours and white field.

## 3. Typography and spacing

Use system sans-serif; monospace is reserved for identifiers and timing values.
Body text is 16px, secondary labels 14px, section titles 20px, page titles 28px,
and the central preparation instruction 36px. Line heights are 1.5 for body and
1.2 for headings. Spacing tokens are 4, 8, 12, 16, 24, 32 and 48px. Buttons have
a minimum 44px target. Focus has a visible 3px blue outline with a 2px offset.

## 4. Layout and presentation

The page owns scrolling; there are no nested scroll regions. Desktop preparation
uses a dominant stimulus panel and narrower supporting pane, in that DOM order.
Small screens stack those regions. Long identifiers wrap. During trials, hide
the eye preview and changing diagnostics; preserve a stable, aspect-correct
stimulus rectangle and record its actual bounds. Resizing or hiding a presented
trial interrupts it rather than silently changing its geometry.

Setup uses numbered groups for identity, physical measurements and detected
display geometry. Window dimensions are live CSS pixels; display estimates
include device pixel ratio and are refreshed after fullscreen settles. The
supporting pane retains diagnostics, while technical paths live in a disclosure.
Recovery is an inline panel with one save-and-new-session action and a separately
visible saved-export path. Repeated requests reuse their acknowledged result;
new sessions always require renewed acknowledgements and calibration. Existing
instrument tokens and page-owned scrolling remain the layout contract.

Before collection, an expandable protocol guide exposes frozen task rules and
response keys without displaying stimuli or answers. The supporting pane shows
fresh sample availability separately from held-out calibration quality. Target
instructions include phase and progress; rest controls show remaining seconds.
Saved diagnostics use a compact definition list and retain explicit incomplete
and inadequate outcomes. No scored correctness is shown during collection.

## 5. Reusable primitives and states

- Panel: neutral surface, title and grouped content; empty/loading/error states
  use explicit text in the same region.
- Action: primary blue, secondary neutral and stop/abort outlined with a text
  label. Disabled actions explain the missing prerequisite next to the control.
- Field: persistent label, unit and validation message; participant identifiers
  and physical measurements are entered explicitly.
- Status: short labelled value with a semantic live region for transitions.
  Camera absent, no face, insufficient samples and calibrated are distinct states.
- Stimulus frame: decoded local image, exact aspect ratio, no crop, stretch,
  transition, interpolation of target motion or answer-key disclosure.
- Response row: equally sized, keyboard-labelled choices; first eligible answer
  locks the row. Timeout, abort and technical interruption have separate text.
- Progress: counts practice and scored trials separately; does not count frames
  or repeated sessions as participants.

## 6. Accessibility and verification

Setup is keyboard accessible with semantic forms, headings, buttons and status
text. Space must not activate multiple actions; Escape stops an active trial.
Respect reduced motion; experimental onset itself is instantaneous. Preserve
text zoom and report the resulting geometry. This is a visual task, so its gaze
measurement is not an equivalent nonvisual assessment; instructions, controls,
errors and exports still need accessible semantics.

The state harness must exercise idle, loading, camera failure, calibration
failure, active trial, timeout, interrupted and completed/export states. Inspect
desktop, tablet and narrow layouts in the real T3 browser before delivery.
Record actual keyboard and resize checks separately from screenshot inspection.
