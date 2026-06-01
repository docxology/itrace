# manuscript/ — iTrace

Pandoc manuscript following the template conventions (see the repo
`CONVENTIONS.md`): one labelled H1 per renderable file
(`# Methods: architecture {#sec:methods}`), numbered sections via
`--number-sections` (never hand-numbered), Pandoc-style citations resolved
against `references.bib` (`fail_on_missing: true`), and cross-references with
`@sec:`, `@fig:`, `@eq:`, `@tbl:`.

## Modular section files

The manuscript is decomposed into small, composable, single-topic modules so each
can be edited, reviewed, and reordered independently.

| File | Section |
|------|---------|
| `00_abstract.md` | Abstract |
| `01_introduction.md` | Introduction |
| `02a_architecture.md` | Methods — architecture (core vs shell) |
| `02b_geometry.md` | Methods — gaze geometry |
| `02c_velocity_and_events.md` | Methods — velocity, I-VT/I-DT, microsaccades |
| `02d_dynamics_and_encoding.md` | Methods — main sequence + scanpath encoding |
| `02e_pupillometry.md` | Methods — pupillometry pipeline |
| `02f_capture_shell.md` | Methods — capture shell + live HTML orchestrator |
| `02g_forward_model.md` | Methods — 3-D forward model + closed loop |
| `02h_noise_model_and_statistics.md` | Methods — noise model + statistical design |
| `02i_descriptive_and_distribution_statistics.md` | Methods — descriptive/distribution statistics |
| `02j_advanced_detection_and_similarity.md` | Methods — adaptive detection + scanpath similarity |
| `03a_ground_truth_recovery.md` | Results — ground-truth recovery |
| `03b_pupillometry_results.md` | Results — pupillometry |
| `03c_closed_loop.md` | Results — closed-loop recovery |
| `03d_noise_sensitivity.md` | Results — noise sensitivity (figure + table) |
| `03e_reproducibility.md` | Results — cross-implementation + gates |
| `03f_figure_gallery.md` | Results — visualization gallery |
| `03g_scanpath_and_temporal.md` | Results — scanpath + temporal statistics |
| `04_discussion.md` | Discussion |
| `05_limitations.md` | Limitations + future work |
| `06_conclusion.md` | Conclusion |
| `S01_statistical_methods.md` | Supplement — statistical methods |
| `S02_software_architecture.md` | Supplement — software architecture |
| `99_references.md` | Reference list anchor |
| `config.yaml` · `references.bib` · `preamble.md` | build config · bibliography · LaTeX packages |

## `{{TOKEN}}` injection

`{{DEMO_AMPLITUDE}}`, `{{TEST_COUNT}}`, `{{COVERAGE_PCT}}` are placeholders the
template pipeline substitutes from computed run values so the prose cites the same
numbers as the run. Current ground-truth values (from `uv run pytest` and
`uv run itrace demo`): DEMO_AMPLITUDE = 10–12°, TEST_COUNT = 429,
COVERAGE_PCT = 91.18%. Figures and the statistics table are produced by
`scripts/generate_figures.py`, `scripts/generate_loop_animation.py`,
`scripts/generate_orbs_animation.py`, and `scripts/generate_power_figure.py`
(which also writes `noise_summary.md`) into `../output/figures/`. The CLI gallery
is rendered with `uv run itrace figures --out-dir output/figures --animations`.

Standalone manuscript renders must run `pandoc-crossref` before citeproc;
otherwise `[@sec:]`, `[@fig:]`, and `[@tbl:]` references are treated as missing
citations in rendered artifacts. The working-project renderer is:

```bash
uv run python scripts/render_manuscript.py \
  --demo-amplitude 10 --test-count 429 --coverage-pct 91.18%
```

The underlying Pandoc shape is:

```bash
pandoc manuscript_input.md --filter pandoc-crossref --citeproc \
  --bibliography=manuscript/references.bib --resource-path=.:manuscript \
  --number-sections -o manuscript/_build.pdf --pdf-engine=xelatex \
  -H manuscript/_render_preamble.tex
```
