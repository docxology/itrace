# iTrace validation protocols

These protocols are implementation scaffolds, not completed validation studies.
iTrace is currently algorithmically verified against synthetic truth and an
independent 3-D closed loop. Device-level accuracy still requires real frames, calibrated
targets, and/or a reference eye tracker.

## Public webcam datasets

Goal: estimate real-frame gaze error without claiming hardware validation beyond
the dataset conditions.

Inputs:

- Subject/session split metadata.
- Frame-level gaze labels in degrees of visual angle or screen coordinates that
  can be converted to degrees.
- Camera/frame metadata when available.

Protocol:

1. Freeze an `AnalysisConfig` JSON file before evaluation.
2. Convert each dataset record into iTrace gaze/pupil/capture CSV artifacts
   without fitting on the test split.
3. Fit calibration only on the dataset's calibration or training split.
4. Evaluate held-out sessions with angular error, dropout fraction, finite
   fraction, sampling quality, and event plausibility.
5. Report subject-level and session-level bootstrap percentile intervals.
6. Publish exact split files, config JSON, generated validation JSON, and figure
   manifests.

Rejected claims:

- Do not call synthetic-closed-loop residuals dataset accuracy.
- Do not report aggregate error without subject/session uncertainty.
- Do not compare against papers unless preprocessing, units, and split policy
  match.

## Reference eye-tracker comparison

Goal: compare webcam-derived gaze/events/pupil proxy to a synchronized reference
device on the same participants and task.

Minimum data:

- Synchronized webcam frames, iTrace capture records, and reference samples.
- Clock-alignment markers or shared trigger events.
- Calibration targets and fitted calibration JSON.
- A preregistered event-matching rule.

Protocol:

1. Collect calibration and validation blocks separately.
2. Align clocks with trigger events, then verify residual clock drift.
3. Apply `itrace validate-recording` before analysis.
4. Match fixation/saccade intervals by temporal overlap; report precision,
   recall, F1, onset error, offset error, amplitude error, and direction error.
5. For pupil, report relative proxy correlations separately from physical
   diameter errors unless a calibrated segmentation path exists.
6. Include Bland-Altman plots for continuous gaze/pupil outputs and bootstrap
   percentile intervals across subjects.

Rejected claims:

- Do not report millimetre pupil accuracy from the current relative proxy.
- Do not pool all samples as independent observations when participants are the
  real sampling unit.

## Benchmarking against pymovements and REMoDNaV

Goal: compare event labels on shared fixtures without presenting library
agreement as biological ground truth.

Protocol:

1. Export common gaze fixtures with units in degrees and timestamps in seconds.
2. Run iTrace, pymovements, and REMoDNaV with recorded configuration files.
3. Normalize labels into fixation, saccade, PSO, smooth pursuit, blink, and
   unknown.
4. Compare interval overlap, event counts, onset/offset deltas, amplitude,
   direction, and disagreement clusters.
5. Use the disagreement report to improve edge-case handling, not to tune on the
   test fixtures silently.

iTrace support:

```bash
uv run itrace benchmark truth_events.csv \
  --gaze-csv itrace_input_gaze.csv \
  --comparator-events-csv remodnav_events.csv \
  --out output/benchmark.json
```

The `truth_events.csv`, `--predicted-events-csv`, and
`--comparator-events-csv` files use `onset_t,offset_t` columns in seconds, with
optional `amplitude_deg`, `direction_deg`, and `peak_velocity_deg_s` columns.
The output JSON includes the explicit truth boundary
`user-supplied event truth/comparators; no bundled real-device validation is
claimed`.

Rejected claims:

- Do not call agreement with another detector accuracy unless a reference
  annotation or reference device exists.
- Do not hide configuration differences between detectors.

## Guided webcam empirical sessions

Goal: estimate session-specific operating parameters from prompted real-eye
recordings without storing raw biometric video or claiming reference-device
accuracy.

Default protocol:

1. Fixed-center gaze for 30 s to estimate jitter, finite-gaze fraction, sampling
   regularity, pupil-valid fraction, and drift slope.
2. Natural reading for 30 s to estimate scanpath quality under a realistic task.
3. Center/four-corner saccade-grid prompting for 30 s; fit calibration from
   marked target intervals and report held-out target residuals and target
   acquisition latency.

iTrace support:

```bash
uv run --extra capture --extra web itrace live-html --camera 0 --output-dir output/live-html
uv run itrace experiment-report \
  output/live-html/experiment/experiment_manifest.json \
  output/live-html/live_capture_records.csv \
  --out output/live-html/experiment_report.json
```

Manuscript pilot support:

```bash
uv run --extra capture --extra web itrace live-html \
  --camera 0 --host 127.0.0.1 --port 8766 \
  --output-dir output/empirical_pilot \
  --empirical-manifest docs/empirical_sessions_manifest.json \
  --open-browser
uv run python scripts/summarize_empirical_pilot.py \
  --report output/empirical_pilot/local_pilot_XXX/experiment/experiment_report.json
```

The summarizer writes `docs/empirical_pilot_metrics.json` and
`output/figures/empirical_pilot_summary.png`; absent pilot data remains explicit
as unavailable, not estimated from a placeholder. With the empirical manifest
configured, the live dashboard allocates the next unused `local_pilot_###`
session ID, assigns the matching `R###` replicate ID, automatically writes the
experiment bundle under that session directory after the final required trial,
and upserts the manifest with the repo-relative report path. The same auto-save
refreshes `docs/empirical_sessions_summary.json`
so the current replicate, condition, and reference-evidence blockers are saved
with the data export. Rerun `scripts/aggregate_empirical_sessions.py` when the
PNG readiness figure also needs to be regenerated.

Multi-session intake:

```bash
uv run python scripts/aggregate_empirical_sessions.py \
  --manifest docs/empirical_sessions_manifest.json \
  --summary-out docs/empirical_sessions_summary.json \
  --figure-out output/figures/empirical_sessions_summary.png
```

`docs/empirical_sessions_manifest.json` is the source of truth for adding more
empirical data. The current paper v1 plan is a single-participant,
single-device, five-session diagnostic design: participant/device counts are
recorded as scope metadata, while the release gate requires five available
sessions, five replicate IDs, and the two collected lighting conditions. Each
available session needs
`session_id`, `participant_id`, `device_id`, `session_group`, `replicate_id`,
`condition`, `protocol_id`, `consent_scope`, `reference_kind`, and a
repo-relative `report` path. Dashboard auto-saves fill those fields automatically
when launched with `--empirical-manifest`; the browser saves the bundle
automatically on final-trial completion when `--output-dir` is configured. The
browser session panel includes controls for condition, participant, device,
session group, and reference kind, plus a status banner with the assigned
session ID, manifest path, auto-save state, and report path. Use
`status: "planned"` for sessions that are scheduled but not yet exported.

The aggregator rejects absolute paths, parent traversal, raw-video suffixes,
missing reports for available sessions, duplicate session IDs, duplicate
replicate IDs within the same `session_group`, and unknown reference kinds. A
non-`none` `reference_kind` is only an intent marker until the row also names a
valid repo-relative `reference_artifact` JSON. For `manual_annotation`, that
artifact uses `kind: "itrace_manual_annotation_evidence"` and must include
`version`, `session_id`, `source_report`, `source_records`, `annotation_scope`,
`annotator_id`, `created_at`, and nonempty sampled-window annotations with
`trial_id`, `target_label`, `start_s`, `end_s`, `quality`, and `target_hit`.
The artifact points to derived report/record files only; raw video paths are
rejected. The summary records the current session, replicate,
participant/device-scope, condition, reference-candidate, and validated
reference-evidence counts plus any current diagnostic-v1 status.

The checked-in v1 criteria now treat `R001`-`R005` as sufficient for diagnostic
release readiness: `R001`-`R002` are `indoor_office_daylight`, and
`R003`-`R005` are `indoor_office_dim`. The old 12-session, three-condition, and
validated-reference target is retained in the summary as future validation
scope, not as a current release blocker. That future scope still needs seven
more session exports, `indoor_office_backlit`, and at least one real
`reference_device`, `public_dataset`, or `manual_annotation` lane backed by a
valid `reference_artifact` before device-validation wording is allowed.

Exported artifacts are derived by default: per-trial gaze, pupil, and capture
CSV files, `target_schedule.csv`, `experiment_manifest.json`, and
`experiment_report.json`. The local browser may receive transient eye-crop JPEGs
for display during a live session, but raw eye video and persisted eye-crop
images are not written by the default workflow.

Rejected claims:

- Do not describe held-out target residuals as reference eye-tracker accuracy.
- Do not generalize a single-participant/device repeated-session result to other
  participants, cameras, displays, lighting regimes, head-pose distributions, or
  tasks without a separate design that samples those axes.
- Do not store raw eye video or persisted eye-crop images unless a separate
  consent/privacy workflow exists.

## Documentation, figure, and analysis traceability

Goal: make every public validation claim and rendered figure auditable back to a
tested Python method, generated evidence file, or explicit pending boundary.

Source of truth:

- `docs/verification_metrics.json`: current version, gate date, test count,
  coverage, lint/type/lock summaries, repository URL, license, and render
  evidence.
- `docs/empirical_pilot_metrics.json`: pending or recorded guided-pilot values
  for manuscript hydration; absent values remain unavailable.
- `docs/empirical_sessions_manifest.json`: machine-readable intake ledger for
  planned and available repeated empirical sessions, with privacy, replicate,
  session-group, condition, and reference-evidence fields.
- `docs/empirical_sessions_summary.json`: generated repeated-session readiness
  summary and v1 blocker ledger.
- `output/figures/synthetic_empirical_range_bridge.json`: generated comparison
  of N=1 local-session values, synthetic-domain summaries, idealized
  landmark-noise sweep values, and statistical diagnostics with explicit
  comparability labels.
- `output/figures/statistical_interpretation_ledger.json`: generated map from
  each statistical diagnostic to its source artifact, estimand, scholarship
  basis, and explicit non-claim boundary.
- `docs/TRACEABILITY_MATRIX.json`: machine-readable map from claims and figures
  to docs, methods, tests, evidence files, and truth boundaries.
- `docs/figure_manifest.json`: generated publication-figure manifest with source
  scripts/data, byte counts, SHA-256 hashes, dimensions, and nonblank
  pixel-variance checks.

Protocol:

1. Add or update a traceability row when a doc claim, manuscript result, figure,
   benchmark, live diagnostic, or analysis report changes public behavior.
2. Name the tested Python method or script that computes the value; browser
   Canvas/SVG is display-only unless it renders a Python-computed payload.
3. Name at least one test file and one evidence artifact for every public claim.
4. State whether the evidence is algorithmic verification, bounded diagnostic,
   external user-supplied truth, session diagnostic, or device validation.
5. Regenerate figures and manuscript outputs after the matrix or metrics ledger
   changes.

The statistical interpretation ledger is a readability and traceability layer:
it does not add new validation evidence, and it must keep descriptive
statistics, relative model diagnostics, synthetic stress tests, and the N=1
pilot case-study separated from device-level accuracy claims.

Rejected claims:

- Do not say a visualization proves anything beyond the tested data it renders.
- Do not let README, ISA, manuscript, or docs carry validation numbers that are
  absent from `docs/verification_metrics.json`.
- Do not treat UI plots, screenshots, or live webcam plausibility warnings as
  reference-device evidence.

## Release automation

Goal: make each release reproducible and import-safe.

Required release gate:

```bash
uv lock --check
uv run pytest --cov=itrace
uv run ruff format --check src/ tests/ scripts/
uv run ruff check src/ tests/ scripts/
uv run mypy src/itrace
uv run --extra capture --extra dashboard --extra figures --extra web \
  python -c "import cv2, mediapipe, streamlit, plotly, matplotlib, fastapi, uvicorn"
uv run python scripts/generate_figures.py
```

The `itrace figures --out-dir output/figures --animations` CLI remains the
gallery-only path; `scripts/generate_figures.py` is the publication refresh path
that also writes the graphical abstract, noise sidecars, empirical summary,
synthetic-to-empirical range bridge, statistical interpretation ledger, and
figure manifest.

Optional environmental smoke:

```bash
uv run --extra capture --extra web itrace camera-probe --camera 0 --frames 30
uv run --extra all itrace live-html --camera 0 --output-dir output/live-html
```

Hardware failure, camera permission denial, or no-face detection in the optional
smoke is environmental unless the deterministic synthetic and import-safety
gates fail.
