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
| `03_graphical_abstract.md` | Graphical abstract |
| `03a_ground_truth_recovery.md` | Results — ground-truth recovery |
| `03b_pupillometry_results.md` | Results — pupillometry |
| `03c_closed_loop.md` | Results — closed-loop recovery |
| `03d_noise_sensitivity.md` | Results — noise sensitivity (figure + table) |
| `03e_reproducibility.md` | Results — cross-implementation + gates |
| `03ez_empirical_pilot.md` | Results — single-pilot empirical diagnostics |
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

`{{DEMO_AMPLITUDE}}`, `{{TEST_COUNT}}`, `{{COVERAGE_PCT}}`,
`{{EMPIRICAL_PILOT_*}}`, and `{{EMPIRICAL_SESSIONS_*}}` are placeholders the
working-project renderer substitutes from `docs/verification_metrics.json`,
`docs/empirical_pilot_metrics.json`, and
`docs/empirical_sessions_summary.json` so the prose cites the same numbers as
the gate, pilot, and repeated-session records. Do not duplicate current gate
numbers in this README; `docs/verification_metrics.json` is the single source
for version, demo amplitude, test count, coverage, repository URL, gate date,
and render evidence.
Figures and the statistics table are produced by
`scripts/generate_figures.py` into `../output/figures/`; that publication
refresh path calls the loop/orbs/power/empirical helpers and writes
`docs/figure_manifest.json`. The CLI gallery remains available with
`uv run itrace figures --out-dir output/figures --animations`, but it does not
refresh the graphical abstract, noise sidecars, empirical summary, or
manuscript manifest.
`scripts/summarize_empirical_pilot.py` reads the derived live
`experiment_report.json` and writes the pilot metrics JSON plus
`../output/figures/empirical_pilot_summary.png`; before a local recording exists,
the metrics file is explicit about unavailable values.
`scripts/aggregate_empirical_sessions.py` reads
`docs/empirical_sessions_manifest.json`, validates planned/available
single-participant/device repeated-session metadata, and writes
`docs/empirical_sessions_summary.json` plus
`../output/figures/empirical_sessions_summary.png` so new empirical replicates
can be added without weakening the v1 evidence boundary. The same publication
refresh path writes `../output/figures/synthetic_empirical_range_bridge.json`
and `.png`, which compare the N=1 pilot with synthetic-domain, idealized
landmark-noise, and statistical-diagnostic evidence while labelling
non-comparable quantities. It also writes
`../output/figures/statistical_interpretation_ledger.json` and `.png`, which map
the manuscript statistics to their estimands, source artifacts, scholarship
basis, and explicit non-claims.

Standalone manuscript renders must run `pandoc-crossref` before citeproc;
otherwise `[@sec:]`, `[@fig:]`, and `[@tbl:]` references are treated as missing
citations in rendered artifacts. The working-project renderer is:

```bash
uv run python scripts/render_manuscript.py \
  --metrics-json docs/verification_metrics.json
```

The underlying Pandoc shape is:

```bash
pandoc manuscript_input.md --filter pandoc-crossref --citeproc \
  --bibliography=manuscript/references.bib --resource-path=.:manuscript \
  --number-sections -o manuscript/_build.pdf --pdf-engine=xelatex \
  -H manuscript/_render_preamble.tex
```
