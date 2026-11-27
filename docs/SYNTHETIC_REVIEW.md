# Synthetic analysis and manuscript review

The pure NumPy/SciPy analysis runs with synthetic arrays. The capture and
dashboard extras are separate. Use Python 3.14 and the locked development
environment for the workflow below.

From a source checkout, run focused numerical and manuscript checks.

```bash
uv sync --locked --extra dev --python 3.14
uv run --no-sync python -m pytest tests/test_geometry.py tests/test_detection.py tests/test_velocity.py tests/test_pupil.py tests/test_synthetic_session.py tests/test_scene.py tests/test_manuscript_audit.py tests/test_manuscript_render.py
```

For a minimal offline review checkout, materialize the source, manuscript and
figure inputs below. A blobless sparse clone can select
`docs/`, `scripts/`, `src/` and `tests/` and the listed files under
`output/figures/`. Review preparation reads the software metrics in
`docs/verification_metrics.json`, the synthetic noise ledger, and these
synthetic figures.

- `closed_loop_summary.png`, `cover_visual.png`, `direction_polar.png`,
  `eye_orbs_still.png`, `graphical_abstract.png`, and `main_sequence.png`
- `noise_metrics.json`, `noise_power.png`, `resampling_sensitivity.png`,
  and `statistical_diagnostics.png`

The review generator labels historical participant-derived tokens and four
figure panels as withheld. The full authored sources and the
[published manuscript](https://doi.org/10.5281/zenodo.23419306) retain the
original paper. The review uses the recorded software and synthetic metrics.
Its PDF and HTML are separate from the published edition.

Use a clean, isolated project checkout with the locked environment above and
[docxology/template at the pinned
revision](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2).
Install the Template environment and put Pandoc, a matching pandoc-crossref,
XeLaTeX and BibTeX on PATH. Register it from a checkout whose physical directory
is named `itrace`.

```bash
export PROJECT_ROOT="$(pwd -P)"
export TEMPLATE_ROOT=/path/to/template
git -C "$TEMPLATE_ROOT" rev-parse HEAD
mkdir -p "$TEMPLATE_ROOT/projects/working"
ln -s "$PROJECT_ROOT" "$TEMPLATE_ROOT/projects/working/itrace"
cd "$TEMPLATE_ROOT"
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/itrace
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/itrace
```

The Template HEAD should equal the pin. Stage 03 invokes the local hydration
script and builds `output/pdf/itrace_combined.pdf` and `output/web/index.html`
in the project checkout. Stage 04 writes the validation and rendered
provenance receipts under `output/reports/`. Check that the intended
sections, bibliography and figures are present and the review's withheld
evidence is clearly marked.
