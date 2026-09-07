# scripts/

Thin orchestrator scripts for iTrace. All business/data/plot/analysis logic
lives in `src/itrace/` (importable and tested); scripts only do argparse, path
bootstrap, logging, and a single delegated call. See `AGENTS.md` for the
contract and the full inventory table.

## Scripts

### `generate_figures.py`

Generates the complete publication figure set into `output/figures/` and writes
the provenance manifest `docs/figure_manifest.json`.

```bash
uv run python scripts/generate_figures.py
```

Delegates to:

- `src/itrace/viz/power_figure.py` — 3-panel noise-sensitivity figure, statistics table, sidecar metrics
- `src/itrace/viz/loop_animation.py` — closed-loop summary PNG + GIF
- `src/itrace/viz/orbs_animation.py` — floating-orb still + GIF
- `src/itrace/viz/graphical_abstract.py` — cover visual + graphical abstract
- `src/itrace/viz/empirical_pilot.py` — N=1 pilot summary figure
- `src/itrace/viz/empirical_sessions.py` — repeated-session intake figure
- `src/itrace/viz/gallery.py` — CLI gallery figures
- `src/itrace/power.py`, `src/itrace/scene.py` — Monte-Carlo sweep / closed-loop simulation

Outputs (`output/figures/` unless noted): `noise_power.png`, `noise_summary.md`,
`noise_metrics.json`, `closed_loop_summary.png`, `closed_loop.gif`,
`eye_orbs_still.png`, `eye_orbs.gif`, `empirical_pilot_summary.png`,
`empirical_sessions_summary.png`, `cover_visual.png`, `graphical_abstract.png`,
`main_sequence.png`, `direction_polar.png`, gallery figures, plus
`docs/figure_manifest.json` (sha256 + source-data provenance per artifact).

### `generate_power_figure.py`

Runs the noise sweep and renders the 3-panel recovery-vs-noise figure, the
Markdown statistics table, and the JSON sidecar metrics.

```bash
uv run python scripts/generate_power_figure.py
```

Module logic: `src/itrace/viz/power_figure.py` (figure/table/metrics over
`src/itrace/power.py` sweep results).

### `generate_loop_animation.py`

Renders the full closed loop (3-D eye → projection → recovered gaze/pupil) as a
300-dpi PNG and a GIF.

```bash
uv run python scripts/generate_loop_animation.py
```

Module logic: `src/itrace/viz/loop_animation.py` (layout only; simulation in
`src/itrace/scene.py`).

### `generate_orbs_animation.py`

Renders synthetic eyes as shaded floating 3-D orbs (dark editorial canvas) as a
still and a GIF.

```bash
uv run python scripts/generate_orbs_animation.py
```

Module logic: `src/itrace/viz/orbs_animation.py`.

### `generate_graphical_abstract.py`

Renders the PDF cover visual and the 16:9 graphical abstract from live repo
ledger files.

```bash
uv run python scripts/generate_graphical_abstract.py
```

Module logic: `src/itrace/viz/graphical_abstract.py`. Reads
`docs/verification_metrics.json` and `docs/empirical_sessions_summary.json` for
hydration.

### `summarize_empirical_pilot.py`

Summarizes one derived live empirical-pilot export (no raw video): writes
`docs/empirical_pilot_metrics.json` (manuscript token hydration) and
`output/figures/empirical_pilot_summary.png`. Absent pilot data produces an
explicit `available=false` payload.

```bash
uv run python scripts/summarize_empirical_pilot.py [--report P] [--metrics-out P] [--figure-out P] [--pilot-id ID]
```

Module logic: `src/itrace/empirical.py` (metrics + tokens, pure) and
`src/itrace/viz/empirical_pilot.py` (figure).

### `aggregate_empirical_sessions.py`

Aggregates manifest-listed empirical sessions into
`docs/empirical_sessions_summary.json` plus the v1-readiness figure.

```bash
uv run python scripts/aggregate_empirical_sessions.py [--manifest P] [--summary-out P] [--figure-out P]
```

Module logic: `src/itrace/empirical.py` (validation/aggregation, pure) and
`src/itrace/viz/empirical_sessions.py` (figure).

### `render_manuscript.py`

Standalone manuscript renderer for this working checkout: concatenates
renderable modules, hydrates `{{TOKENS}}` from the metrics ledgers, and runs
Pandoc (pandoc-crossref before citeproc) in `docs/manuscript/`.

```bash
uv run python scripts/render_manuscript.py --metrics-json docs/verification_metrics.json
```

Module logic: in-script token loading + Pandoc invocation (the canonical
template pipeline takes over after promotion to `active/`).

Outputs: `docs/manuscript/_build.{md,tex,pdf,txt}`.

### `audit_crossrefs.py`

Cross-reference and citation integrity audit over `docs/manuscript/`: dangling
cross-refs, duplicate/dead labels, missing/uncited citations, and manual
Figure/Table numbering. Exit non-zero on any build-fatal problem.

```bash
uv run python scripts/audit_crossrefs.py
```

Module logic: `src/itrace/manuscript.py`.
