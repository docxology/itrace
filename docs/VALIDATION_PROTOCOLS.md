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

Rejected claims:

- Do not call agreement with another detector accuracy unless a reference
  annotation or reference device exists.
- Do not hide configuration differences between detectors.

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
uv run itrace figures --out-dir output/figures --animations
```

Optional environmental smoke:

```bash
uv run --extra capture --extra web itrace camera-probe --camera 0 --frames 30
uv run --extra all itrace live-html --camera 0 --output-dir output/live-html
```

Hardware failure, camera permission denial, or no-face detection in the optional
smoke is environmental unless the deterministic synthetic and import-safety
gates fail.
