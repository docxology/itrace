# TODO - iTrace Roadmap

This file tracks upcoming improvements after the v0.5.0 package refresh. Keep
items scoped, verifiable, and honest about the current validation boundary:
iTrace is algorithmically verified against synthetic truth, but device-level
accuracy still needs reference eye-tracker or public-dataset validation.

## Minor Updates

- [x] Add short examples for `AffineCalibration`, `robust_gaze_quality`, and
  `merge_gap_s` to the API docs and CLI help text.
- [x] Expose gaze-quality fields in `itrace stats` output with stable names:
  valid fraction, dropout fraction, median dt, sampling jitter, longest gap,
  large-gap count, and nonmonotonic timestamp count.
- [x] Add a small CSV fixture demonstrating calibrated vs uncalibrated gaze and
  expected calibration-error JSON output.
- [x] Tighten type aliases for report dictionaries where practical, especially
  pupil and quality summaries.
- [x] Add regression tests for direct `detect_ivt` validation errors:
  non-negative velocity threshold, non-negative minimum duration, and
  non-negative merge gap.
- [x] Add a `--config-json` option to CLI analysis commands for reproducible
  method settings.
- [x] Improve live-HTML status text for camera-permission, no-face, and
  low-quality landmark states without adding browser-side analysis logic.
- [x] Add small screenshots or generated stills for README sections that mention
  the live HTML orchestrator and figure gallery.
- [x] Audit public docs for every metric literal after each full gate run.
- [x] Keep optional dependency import-safety tests current when new extras are
  added.
- [x] Add a generated cover visual and graphical abstract to the manuscript
  render path, with integrity tests that check author/ORCID/layout metadata.
- [x] Add a machine-readable `docs/TRACEABILITY_MATRIX.json` that maps public
  claims and figures to tested methods, tests, generated evidence, and truth
  boundaries.
- [x] Add a generated `docs/figure_manifest.json` with publication-figure source
  scripts/data, hashes, dimensions, byte counts, and nonblank pixel checks.
- [x] Add a synthetic-to-empirical range-bridge JSON and publication figure that
  contextualize the N=1 pilot against synthetic/stress/statistical variables
  without device-validation wording.
- [x] Add a statistical interpretation ledger JSON and publication figure that
  map reported statistics to source artifacts, estimands, scholarship bases,
  and explicit non-claim boundaries.

## Medium Improvements

- [x] Add a reusable calibration-session workflow: backend-owned per-target
  sampling, affine fit, calibration diagnostics, and JSON export.
- [x] Add visible target presentation to the live calibration workflow.
- [x] Add calibration-aware live HTML mode that fits `AffineCalibration` from
  recent raw samples, applies it to subsequent gaze samples, exports
  calibration artifacts, and displays calibrated vs raw coordinates.
- [x] Expand robust event detection with configurable minimum inter-event gap,
  maximum event duration, and edge-event rejection, each reported in config.
- [x] Add smooth-pursuit candidate detection as a labelled, explicitly
  provisional event path, with tests against synthetic pursuit traces.
- [x] Add blink-aware gaze interpolation utilities that can mask or bridge short
  capture gaps without silently inventing long missing segments.
- [x] Add more pupil features: latency-to-peak, dilation area under curve,
  baseline-relative percentage change, and windowed pupil-phase summaries.
- [x] Add synthetic head-pose drift, lighting changes, rolling-shutter-like
  timestamp jitter, and correlated landmark noise to the scene/noise model.
- [x] Add event-level bootstrap summaries for fixation duration, saccade
  amplitude, peak velocity, and pupil-response metrics.
- [x] Add JSON schema or pydantic-free dataclass validators for exported reports
  so downstream tooling can rely on stable report structure.
- [x] Add a compact `itrace validate-recording` CLI that checks sampling,
  dropout, pupil validity, event plausibility, and calibration availability
  before analysis.
- [x] Improve visualization coverage for quality diagnostics: dropout raster,
  sampling interval histogram, calibration residual plot, and pupil velocity
  trace.
- [x] Add browser smoke checks for the live HTML controls using synthetic frame
  sources, including start/stop, adaptive/fixed I-VT toggle, export, and plot
  nonblank checks.
- [x] Add a pure synthetic-domain validation suite with within-domain repeat
  summaries, cross-domain stability metrics, and bootstrap intervals.
- [x] Add live HTML validation diagnostics that display Python-computed quality
  and synthetic-domain recovery results without browser-side analysis logic.
- [x] Harden adaptive I-VT threshold estimation against non-finite velocity
  samples from dropout-heavy synthetic/live traces.
- [x] Add a guided live empirical-session workflow with backend-owned trial
  windows, derived-only export, and a reproducible `itrace experiment-report`
  command.
- [x] Add a single-participant/device repeated-session empirical intake manifest,
  aggregate readiness summary, generated figure, and tests that block v1
  device-validation claims until replicate, condition, and reference-backed
  evidence criteria are satisfied.

## Large Improvements

- [x] Add 2D polynomial calibration with cross terms and regularisation for higher-order screen-space mapping.
- [x] Add first-class Area of Interest (AOI) dataclasses, configs, dwell statistics, and transition matrices.
- [x] Refactor recording validation payload generation out of CLI into core validation module.
- [ ] Run a public-dataset validation study on webcam gaze data such as MPIIGaze
  or GazeCapture, with subject/session splits and explicit device-validation
  language. Acceptance: dataset license cleared for redistribution of derived
  metrics; subject-level split script committed under scripts/ with fixed seeds;
  results land in a dated ledger under docs/ with device-validation claim
  language reviewed against TRACEABILITY_MATRIX.json boundaries.
- [x] Add reference-eye-tracker comparison protocol: synchronized capture,
  calibration, clock alignment, event matching, Bland-Altman plots, and
  preregistered acceptance thresholds.
- [x] Implement a pure eye-crop pupil segmentation path that reports pixels or
  pupil/iris-relative units only, with confidence/quality fields and no
  millimetre claim.
- [ ] Calibrate pupil segmentation against reference geometry or a learned model
  so live webcam pupil output can move beyond relative proxy measurements.
  Acceptance: calibrated units defined (mm from reference geometry or
  model-predicted diameter with stated CI); no millimetre claim ships without a
  reference-device comparison artifact; TRACEABILITY_MATRIX.json updated.
- [x] Add binocular modelling: per-eye gaze estimates, vergence, inter-eye
  consistency checks, and diagnostic asymmetry fields.
- [ ] Add binocular calibration and confidence down-weighting for live capture.
  Acceptance: per-eye affine fits with inter-eye consistency gate; down-weighting
  rule documented with its failure mode; headless tests against synthetic
  binocular fixtures.
- [ ] Add lens-distortion and camera-intrinsic calibration support, including
  checkerboard calibration import and distortion-aware projection/estimation.
  Acceptance: checkerboard import (OpenCV YAML/JSON) with round-trip test;
  distortion-aware projection stays out of the pure core (capture-shell only);
  import safety preserved (no top-level cv2 import).
- [ ] Add head-pose compensation from Face Mesh geometry while keeping the
  capture shell thin and the core testable with plain arrays. Acceptance:
  compensation is a pure function of plain float arrays (unit-testable headless);
  lazy MediaPipe import only in the capture shell.
- [ ] Add REMoDNaV-like richer event classification, including smooth pursuit,
  PSO/glissade classes, and explicit event-merging/splitting rules. Acceptance:
  each event class validated against synthetic ground truth with the
  reference-impl oracle; merging/splitting rules recorded in the report config.
- [x] Add a benchmark harness comparing iTrace event outputs against
  user-supplied truth/comparator files on shared fixtures.
- [ ] Add first-class pymovements and REMoDNaV runner integrations with recorded
  detector configuration files. Acceptance: optional extra with import-safe
  fallback; detector configs serialized with every benchmark run.
- [ ] Add a stable plugin/backend interface for alternative capture sources:
  browser WebRTC, recorded video files, IR cameras, and synthetic frame streams.
  Acceptance: `CaptureBackend` protocol frozen with conformance suite coverage
  for each shipped backend; no breaking change to existing `record` CLI.
- [ ] Add packaging/release automation: build artifacts, wheel smoke tests,
  optional-extra matrix, generated docs, and release notes. Acceptance: `uv
  build` wheel+sdist smoke-tested in a clean venv per optional-extra
  combination; release notes generated from CHANGELOG-style ledger.
- [ ] Add a manuscript-quality validation supplement with all protocols,
  datasets, statistical plans, limitations, and rendered figures. Acceptance:
  supplement renders via scripts/render_manuscript.py with all metrics
  token-injected from ledgers; no hand-typed statistic literals.
- [ ] Add long-running reliability tests for live capture: memory growth, FPS
  stability, dropped-frame handling, and export integrity over extended sessions.
  Acceptance: opt-in slow-marker suite (not default CI); memory growth bounded
  (assert RSS delta threshold); export byte-identical across replay.
- [ ] Promote guided empirical-session reports into a preregistered
  reference-device study with synchronized truth, consent/privacy handling, and
  subject-level uncertainty. Acceptance: preregistration document committed
  before data collection; consent/privacy handling documented per IRB/ethics
  decision (owner sign-off required); analysis plan sealed until collection ends.

## Follow-ups Discovered During Implementation

- [x] Replace the live HTML quick-fit calibration button with a backend-owned
  target-session workflow that records per-target finite sample windows.
- [x] Add visible target presentation and click/space confirmations to the live
  HTML calibration workflow (target dot accepts click or Space to advance the
  backend-owned target-session sampling; keyboard handler ignores form fields).
- [x] Add a validation JSON example fixture covering short-gap interpolation,
  low valid fraction warnings, and calibration availability.
- [x] Add a richer `CaptureBackend` conformance test suite for recorded video,
  synthetic frame streams, and future browser WebRTC adapters.
- [x] Add figure snapshots for the new quality plots and live HTML orchestrator
  to the README once the rendered figures are regenerated in the final gate.
- [x] Add a machine-readable validation-domain registry so downstream studies
  can extend the default synthetic domains without editing Python source. The
  existing `docs/TRACEABILITY_MATRIX.json` is the claim/figure evidence map, not
  a domain-extension registry.
- [x] Add acceptance-threshold presets for "demo", "webcam exploratory", and
  "reference-device comparison" validation reports.

## Maintenance Rules

- Keep every item tied to a testable artifact or command before marking it done.
- Prefer new pure-core methods over browser-side or dashboard-only
  implementations; the UI should orchestrate validated Python paths.
- Update `README.md`, `ISA.md`, manuscript source, and generated manuscript
  artifacts whenever a roadmap item changes public behavior or validation
  evidence.
- Preserve import safety: `import itrace` must not import optional capture, web,
  dashboard, or figure dependencies.
