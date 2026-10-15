# scripts/ — Agent Notes

## Design Contract

- Scripts are **thin orchestrators only**: argparse, path bootstrap, logging,
  and delegated calls into `src/itrace/` entrypoints.
- Business/data/plot/analysis logic lives in `src/itrace/` (importable and
  unit-tested without a camera; `src/itrace/viz/` is the matplotlib-gated
  subpackage).
- Scripts MUST NOT import each other. Shared repo-layout defaults live next to
  the logic that consumes them (`itrace.empirical.DEFAULT_PILOT_*` and
  `itrace.empirical.DEFAULT_SESSIONS_*`) and are resolved
  against each script's own `ROOT = Path(__file__).resolve().parents[1]`.
- All reusable script behavior is covered by `tests/` (the suite imports the
  script modules for their entry points; `uv run pytest tests/`).

## Script Inventory

| Script | Purpose | Delegates to | Command |
|---|---|---|---|
| `generate_figures.py` | Full figure set + `docs/figure_manifest.json` provenance manifest | `itrace.viz.{power_figure, loop_animation, orbs_animation, graphical_abstract, empirical_pilot, empirical_sessions, gallery}`, `itrace.{power, scene}` | `uv run python scripts/generate_figures.py` |
| `generate_power_figure.py` | 3-panel noise-sensitivity figure, stats table, sidecar metrics | `itrace.viz.power_figure` (over `itrace.power`) | `uv run python scripts/generate_power_figure.py` |
| `generate_loop_animation.py` | Closed-loop summary PNG + GIF | `itrace.viz.loop_animation` (over `itrace.scene`) | `uv run python scripts/generate_loop_animation.py` |
| `generate_orbs_animation.py` | Floating-orb 3-D eye still + GIF | `itrace.viz.orbs_animation` (over `itrace.scene`) | `uv run python scripts/generate_orbs_animation.py` |
| `generate_graphical_abstract.py` | Cover visual + 16:9 graphical abstract | `itrace.viz.graphical_abstract` | `uv run python scripts/generate_graphical_abstract.py` |
| `summarize_empirical_pilot.py` | Pilot metrics JSON + N=1 diagnostic figure | `itrace.empirical` (metrics), `itrace.viz.empirical_pilot` (figure) | `uv run python scripts/summarize_empirical_pilot.py` |
| `aggregate_empirical_sessions.py` | Repeated-session aggregate JSON + v1-readiness figure | `itrace.empirical` (aggregation), `itrace.viz.empirical_sessions` (figure) | `uv run python scripts/aggregate_empirical_sessions.py` |
| `render_manuscript.py` | Manuscript render (token hydration + Pandoc/crossref/citeproc) | `itrace.manuscript` | `uv run python scripts/render_manuscript.py --metrics-json docs/verification_metrics.json` |
| `audit_crossrefs.py` | Manuscript crossref/citation integrity audit (exit 1 on build-fatal) | `itrace.manuscript` | `uv run python scripts/audit_crossrefs.py` |
| `verify.py` | Fail-fast gates with logs and source/runtime identity | `itrace.verification` | `uv run python scripts/verify.py --out-dir output/verification/new-run` |
| `audit_release.py` | Release metadata and figure integrity | `itrace.release` | `uv run python scripts/audit_release.py` |

Run every script from the repo root: `uv run python scripts/<name>.py`.
