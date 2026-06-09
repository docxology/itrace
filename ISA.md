---
task: "Original request: build a webcam eye gaze / saccade / pupil tracking package; implemented as algorithmically verified, not device-validated"
project: iTrace
effort: comprehensive
effort_source: classifier
phase: complete
progress: 132/132
mode: build
iteration: 7
started: 2026-05-29T00:00:00Z
updated: 2026-05-31T00:00:00Z
---

# ISA — iTrace: Open-Source Webcam Eye-Movement Analysis Toolkit

## Problem

`working/iTrace` contained exactly one artifact: a 833-line literature survey
(`docs/RESEARCH_BRIEF.md`) describing the open-source webcam eye-tracking
ecosystem — OpenCV/MediaPipe capture, L2CS-Net/EyeGestures gaze, REMoDNaV /
pymovements event classification, PupilEXT / EyeDentify / rtPupilPhase
pupillometry, and Streamlit/Dash dashboards. It was a *map*, not a *territory*:
no package, no code, no tests, nothing installable or runnable. The repo norm
(`CONVENTIONS.md`, the `template/` exemplar, the praised `BTC_Buffalo` run) is a
`uv`-managed `src/<pkg>/` Python package with no-mocks tests at ≥90% coverage, a
clean `ruff` gate, and `mypy --strict` passing. iTrace had none of that. The
ideal state is the territory: a real, installable, type-checked, algorithmically verified
Python package that actually computes the three target signals — **saccade
direction & dynamics, gaze trajectory, pupil diameter** — with every algorithm
proven against synthetic ground truth by a green no-mocks suite.

## Vision

A researcher runs `uv sync && uv run pytest`, sees a green suite with ≥90%
coverage, then `uv run itrace demo` and watches synthetic gaze get classified
into fixations and saccades with recovered amplitudes and peak velocities that
land on the main sequence, and a pupil trace get deblinked, baseline-corrected,
and phase-labelled — all from a package that imports cleanly on a headless box
with zero webcam and zero MediaPipe. Then they plug in a webcam
(`uv run itrace record`) and the *same* analysis core consumes live frames. The
euphoric surprise: the hardware-fragile part (webcam, MediaPipe) is a thin
optional shell around a pure, fully-tested scientific kernel — so the science is
trustworthy and CI-able precisely because it was decoupled from the camera. It
is not a wrapper that re-exports other people's libraries; the canonical
algorithms (I-VT, I-DT, Engbert–Kliegl microsaccades, main-sequence fit,
rtPupilPhase-style phase detection, n-gram scanpath encoding) are implemented
and tested in-tree.

## Out of Scope

- **Re-surveying the field.** The literature map already exists; this run builds
  the package, not another document.
- **Bundling/forking heavyweight models** (L2CS-Net weights, MediaPipe binaries,
  EyeDentify ResNets). MediaPipe is an *optional* capture backend behind an
  interface; absence must never break import or the test suite.
- **Sub-degree accuracy claims.** Commodity-webcam gaze and pupil precision are
  protocol- and detector-dependent; the package reports only synthetic-recovery
  and capture-smoke evidence until public-dataset or reference-device validation
  exists.
- **A production multi-user web service.** A single-user live dashboard
  (optional `dashboard` extra) is in scope; horizontal scaling is not.
- **Training new ML models.** Estimation here is landmark/geometry + classical
  signal processing, not model training.
- **Touching sibling projects or the `template/` repo.**

## Principles

- **Decouple the fragile from the verifiable.** Pure-NumPy/SciPy analysis core
  (substrate-independent math) is separated from hardware capture (OpenCV /
  MediaPipe). The core is the part that must be correct and is the part that is
  tested.
- **No-mocks validation via known ground truth.** A detector is validated by
  synthesizing a signal whose events are known by construction and asserting the
  detector recovers them within a stated tolerance — real code on real arrays,
  never a mock that asserts the test author's expectation.
- **Import safety is non-negotiable.** `import itrace` and the whole test suite
  succeed with zero optional dependencies installed.
- **Canonical algorithms, cited.** Each algorithm names its source (Salvucci &
  Goldberg 2000; Engbert & Kliegl 2003; Nyström & Holmqvist 2010; Kronemer et
  al. 2024) so claims are checkable.
- **Honest limits over flattering numbers.** Accuracy caveats ship in the docs
  and the API surfaces confidence/quality, not just point estimates.

## Constraints

- Python ≥3.10; `uv`-managed; `src/itrace/` setuptools layout with `py.typed`.
- Core runtime deps: `numpy`, `scipy` only. `pandas` for tabular IO.
- Optional extras: `capture` (opencv-python, mediapipe), `dashboard`
  (streamlit, plotly), `figures` (matplotlib), `web` (FastAPI/uvicorn +
  WebSocket test client support), `all` (capture + dashboard + figures + web),
  and `dev` (pytest, pytest-cov, ruff, mypy, matplotlib, web test support).
- **HARD: zero optional deps required for import or `pytest`.** Capture/dashboard
  modules import their heavy deps lazily inside functions, never at module top.
- `ruff check` clean; `ruff format --check` clean; `mypy --strict src/itrace`
  clean; no-mocks; coverage ≥90% (promotion floor).
- All angles in degrees of visual angle; all times in seconds; all pupil sizes
  carry an explicit unit (`mm` or `relative`). No bare pixel coordinates leak
  past the geometry layer.
- Deterministic: every synthetic generator and any stochastic routine takes an
  explicit seed.

## Goal

Ship `iTrace` as an installable `uv` Python package whose pure analysis core
implements and **verifies against synthetic ground truth** the three target
signal families — gaze geometry, saccade/fixation/microsaccade event detection
with main-sequence dynamics, and pupillometry (deblink, baseline, phase) — plus
a thin optional webcam/MediaPipe capture orchestrator, an optional live
dashboard, a local HTML/WebSocket live orchestrator, a typer CLI, and figure
scripts, with a green no-mocks pytest suite at ≥90% coverage, clean `ruff`, and
clean `mypy --strict`, all evidenced by real captured command output.

## Criteria

### Package skeleton & build
- [x] ISC-1: `pyproject.toml` declares `itrace`, core deps numpy+scipy+pandas+typer, extras capture/dashboard/figures/all/dev, `itrace` console script
- [x] ISC-2: `src/itrace/__init__.py` exports the public API and a `__version__`
- [x] ISC-3: `src/itrace/py.typed` marker file exists (PEP 561)
- [x] ISC-4: `uv sync --extra dev` resolves and installs without error
- [x] ISC-5: `uv run python -c "import itrace"` succeeds with NO optional deps installed
- [x] ISC-6: Anti: importing `itrace` or any core module never imports cv2/mediapipe/streamlit/plotly/matplotlib/fastapi/uvicorn at module top (subprocess import-safety test proves lazy imports)

### Types & data model
- [x] ISC-7: `types.py` defines frozen dataclasses GazeSample, Saccade, Fixation, Microsaccade, PupilSample with units documented
- [x] ISC-8: `EventType` enum covers FIXATION, SACCADE, PSO, SMOOTH_PURSUIT, BLINK
- [x] ISC-9: A GazeStream container holds aligned x/y/t arrays and validates equal length on construction
- [x] ISC-10: Constructing GazeStream with mismatched array lengths raises ValueError

### Gaze geometry
- [x] ISC-11: `geometry.pix2deg()` converts pixel offset to degrees given screen geometry; round-trips with `deg2pix()` within 1e-9
- [x] ISC-12: `geometry.iris_offset_to_gaze_angle()` maps normalized iris offset to yaw/pitch degrees, monotonic in offset
- [x] ISC-13: `geometry.normalize_by_interocular()` removes head-distance scaling; output invariant to a uniform scale factor within 1e-6
- [x] ISC-14: Anti: NaN/inf inputs to geometry functions raise or propagate as NaN, never silently become 0

### Velocity
- [x] ISC-15: `velocity.pos2vel()` (Savitzky–Golay) recovers the analytic velocity of a linear ramp within 1% on the interior
- [x] ISC-16: `velocity.pos2vel()` on a constant signal returns ~0 velocity (|v|<1e-6)
- [x] ISC-17: velocity supports variable timestamps (non-uniform dt) and uniform sampling_rate paths

### Saccade / fixation / microsaccade detection
- [x] ISC-18: `saccades.detect_ivt()` on a synthetic fixation→saccade→fixation recovers exactly 3 events in order
- [x] ISC-19: I-VT recovered saccade amplitude matches the constructed amplitude within 5%
- [x] ISC-20: I-VT recovered saccade peak velocity matches construction within 10%
- [x] ISC-21: `saccades.detect_idt()` (dispersion) labels a held fixation as one fixation, not many
- [x] ISC-22: `saccades.detect_microsaccades()` (Engbert–Kliegl) recovers a known 0.5° microsaccade and ignores fixational noise
- [x] ISC-23: microsaccade lambda threshold uses the median-based velocity-std estimator (not plain std) — verified by code probe
- [x] ISC-24: `saccades.saccade_properties()` returns amplitude, direction (deg), duration, peak velocity per saccade
- [x] ISC-25: direction of a purely rightward saccade is ~0°, upward ~90°, within 1°
- [x] ISC-26: Anti: a pure-noise fixation trace yields zero saccades at the documented default threshold
- [x] ISC-27: detectors agree with an independent second implementation (different formulation, `tests/fixtures/reference_impl.py`) on a shared synthetic trace (numerical oracle). NOTE: intended as Forge cross-vendor; Forge/Codex hit a usage limit (resets 2026-05-30 ~13:48), so this is a same-vendor independent-formulation cross-check; true cross-vendor audit is DEFERRED-VERIFY (follow-up: re-run Forge/Cato after reset).

### Main sequence
- [x] ISC-28: `mainsequence.fit()` recovers V_max and C of a synthetic saturating main sequence within 10%
- [x] ISC-29: power-law fit returns exponent b in the literature range (0.4–0.9) for synthetic main-sequence data
- [x] ISC-30: main-sequence fit on <3 points raises a clear error rather than returning garbage

### Pupillometry
- [x] ISC-31: `pupil.detect_blinks()` finds a constructed blink (zero/NaN run) and returns its index span
- [x] ISC-32: `pupil.interpolate_blinks()` removes the blink and leaves a finite, gap-free trace (no NaNs remain)
- [x] ISC-33: `pupil.baseline_correct()` subtracts a pre-event baseline; corrected baseline window mean ≈ 0
- [x] ISC-34: `pupil.smooth()` (low-pass) reduces high-frequency noise variance by >50% on a noisy constant
- [x] ISC-35: `pupil.mad_reject()` flags injected spike outliers and not clean samples
- [x] ISC-36: Anti: a fully-NaN pupil trace is reported as unusable, not silently interpolated to a flat line

### Real-time pupil phase (rtPupilPhase-style)
- [x] ISC-37: `pupilphase.PhaseDetector` streamed sample-by-sample over a sine pupil signal labels peaks, troughs, dilation, constriction
- [x] ISC-38: detected peak indices align with the analytic sine maxima within a small tolerance window
- [x] ISC-39: PhaseDetector is causal — a label at time t uses only samples ≤ t (verified by construction/probe)

### Scanpath encoding
- [x] ISC-40: `encoding.encode_directions()` maps a saccade sequence to U/D/L/R characters (case = long/short) per the Stuttgart scheme
- [x] ISC-41: `encoding.ngram_counts()` returns correct n-gram frequencies for a known character sequence
- [x] ISC-42: a rightward-then-upward sequence encodes as the expected 2-character string

### Capture orchestrator (thin, optional)
- [x] ISC-43: `capture.iris_landmarks_to_sample()` converts a MediaPipe-shaped landmark array (plain floats) to a GazeSample without importing mediapipe
- [x] ISC-44: `capture.WebcamSource` raises a clear, actionable error when opencv/mediapipe are absent (no bare ImportError/AttributeError)
- [x] ISC-45: capture module exposes timestamped capture samples with gaze, relative pupil proxy, frame index, FPS estimate, and quality flags

### Pipeline, IO, CLI, dashboard, figures
- [x] ISC-46: `pipeline.analyze_session()` runs gaze→events→encoding + pupil→deblink→phase end-to-end on a synthetic recording and returns a structured report
- [x] ISC-47: `io.read_gaze_csv()` / `io.write_gaze_csv()` round-trip a GazeStream (values equal within float tolerance)
- [x] ISC-48: `cli` (typer) exposes `demo`, `analyze`, `stats`, `figures`, `record`, `camera-probe`, `live-html`, `dashboard`; `itrace demo` exits 0 and prints recovered event counts
- [x] ISC-49: `itrace analyze <csv>` writes a JSON report file that parses and contains saccade + pupil sections
- [x] ISC-50: `dashboard.py` imports streamlit lazily and raises a clear message when absent; a render-figure helper is unit-tested without streamlit
- [x] ISC-51: figure scripts and `itrace figures` produce non-empty gallery PNGs/GIFs in `output/figures/`

### Gates
- [x] ISC-52: Antecedent: real `uv run pytest --cov` ≥90%, `ruff check` clean, `ruff format --check` clean, `mypy --strict src/itrace` clean — all evidenced by quoted command output
- [x] ISC-53: Honesty disclosure (added post-Advisor): README + manuscript explicitly relabel the work as *algorithmically verified against synthetic ground truth, not device-validated against a reference eye-tracker*, and name the untested capture path + confounds (light reflex, head pose, 30 fps undersampling)

### 3D eye forward model & closed-loop validation (iteration 2)
- [x] ISC-54: `eyemodel.Eye3D` parameterises a 3D eyeball (centre, radius_mm, gaze yaw/pitch, pupil_radius_mm, iris_radius_mm) with a unit gaze vector
- [x] ISC-55: `eyemodel.gaze_vector()` returns a unit vector; yaw=pitch=0 points along the optical axis, monotonic in yaw/pitch
- [x] ISC-56: `eyemodel.project_pinhole()` maps a 3D point to normalised image coords; a point on the optical axis projects to the principal point
- [x] ISC-57: `eyemodel.eye_to_landmarks()` emits a MediaPipe-shaped (478,3) normalised landmark array with iris + eye-corner indices filled consistently with `capture` constants
- [x] ISC-58: centred gaze (0,0) places the iris at the eye centre → `iris_landmarks_to_sample` recovers ~0° yaw/pitch (within 1°)
- [x] ISC-59: a rightward 3D gaze projects the iris to the right; recovered yaw is positive and increases with true yaw
- [x] ISC-60: projected pupil-to-iris radius ratio increases monotonically with `pupil_radius_mm` (the image pupillometry signal)
- [x] ISC-61: Anti: a blinking frame produces an invalid/NaN landmark frame, never a silently-valid bogus gaze

### Animated scene + full closed loop
- [x] ISC-62: `scene.EyeSceneSpec` declares a gaze trajectory (fixation targets + saccades), pupil dynamics (baseline + dilation events), and blink intervals, deterministic/seeded
- [x] ISC-63: `scene.animate()` returns per-frame true (yaw,pitch,pupil,blink) AND projected landmark arrays for N frames
- [x] ISC-64: `scene.closed_loop()` runs frames → `iris_landmarks_to_sample` → GazeStream → pipeline, returning recovered GazeStream/PupilStream + truth arrays + error metrics
- [x] ISC-65: closed-loop recovered gaze RMS error vs the 3D forward-model truth is < 2.0° for |gaze| ≤ 15° (the geometry/landmark path is now exercised end-to-end)
- [x] ISC-66: closed-loop recovers the correct number of saccades for a scripted 3-saccade trajectory
- [x] ISC-67: closed-loop recovered pupil trace correlates with the true pupil dynamics at r > 0.9 after deblinking
- [x] ISC-68: Anti: closed-loop is NOT a tautology — the forward model (3D sphere + pinhole) and the estimator (arcsine sphere approx) are independent formulations, so a nonzero-but-bounded recovery error is expected and asserted (error > 0)

### Animation rendering (full loop visual)
- [x] ISC-69: `scripts/generate_loop_animation.py` renders a multi-panel GIF (3D eye + recovered gaze + pupil/phase) via matplotlib PillowWriter
- [x] ISC-70: the same script writes a static full-loop summary PNG at 300 dpi to `output/figures/`
- [x] ISC-71: animation generation is unit-tested (skips cleanly if matplotlib absent); GIF file is non-empty

### Quality uplift & gates
- [x] ISC-72: `__version__` bumped to 0.2.0; `eyemodel`, `scene` exported from the package API
- [x] ISC-73: test count materially increased and coverage stays ≥90% with the new modules included
- [x] ISC-74: full gate re-green: `pytest` + `ruff check` + `ruff format --check` + `mypy --strict` all clean (quoted output)

### Documentation & manuscript accuracy audit
- [x] ISC-75: every numeric claim in README/manuscript/ISA (test count, coverage %, recovered amplitude) re-derived from THIS run's actual command output and matches — explicit audit, not assertion
- [x] ISC-76: every code symbol/file referenced in README + AGENTS + manuscript exists in the tree (grep/ls audit); manuscript-referenced figures exist in `output/figures/`
- [x] ISC-77: README + manuscript document the 3D forward model and the closed-loop validation, with the strengthened-but-still-not-device-validation framing
- [x] ISC-78: Assumption ledger (added post-Advisor iter-2): README + manuscript explicitly enumerate the idealizations the forward model and estimator SHARE (pinhole/no lens distortion, no corneal refraction, spherical eye, zero kappa, synthetic≠real-MediaPipe), frame the 0.16° as a LOWER BOUND on real error, and reclassify the pupil r=1.0 as a near-identity consistency check, not recovery accuracy

### Statistical power under webcam observation noise (iteration 3 — Science)
- [x] ISC-79: `power.run_noise_sweep()` returns a `MetricCurve` per signal (gaze RMS, saccade F1, pupil r) with mean + 95% CI over `n_trials` seeded replicates per noise level
- [x] ISC-80: gaze RMS error increases strictly monotonically with landmark noise σ (the falsifiable core — flat would mean a broken loop)
- [x] ISC-81: recovered robustness ordering matches first-principles prediction — saccades most fragile (velocity = noise-amplifying derivative), pupil most robust (noise-suppressing spatial average), gaze intermediate
- [x] ISC-82: 95% CI computed via fixed-seed bootstrap percentile; `ci_low ≤ mean ≤ ci_high` for every point and bounded metrics stay inside their mathematical bounds
- [x] ISC-83: `power.recovery_threshold()` returns the σ where a metric crosses its usability bound (gaze<2°≈σ0.005, saccade F1<0.8≈σ0.0014, pupil r<0.9≈σ0.005) and `None` when never crossed
- [x] ISC-84: the whole sweep is deterministic given seeds (reproducible science)
- [x] ISC-85: saccade precision/recall/F1 scoring is correct on extremes (perfect→1, none→0, no-overlap→0)
- [x] ISC-86: landmark-noise injection in `scene.closed_loop` demonstrably increases gaze error

### Synthetic data coverage & first-principles noise model
- [x] ISC-87: synthetic data explicitly covers all three target signals — gaze direction (`scene.true_yaw/pitch`), saccades (`scene.true_saccades` intervals), pupil diameter (`scene.true_pupil_mm`)
- [x] ISC-88: webcam noise is modelled (FirstPrinciples) as Gaussian landmark-localisation σ in normalised image units — the irreducible aggregate of photon/compression/quantisation/MediaPipe error — applied to landmarks before estimation; pupil-measurement noise added separately as a documented modelling assumption

### Visualisation (Art) & figures
- [x] ISC-89: `scripts/generate_orbs_animation.py` renders eyes as shaded floating 3-D orbs (gaze rotation, dilating pupil, gaze ray) on a dark editorial canvas → `eye_orbs.gif` + `eye_orbs_still.png`
- [x] ISC-90: `scripts/generate_power_figure.py` renders the 3-panel power figure with 95% CI bands and threshold crossings → `output/figures/noise_power.png`
- [x] ISC-91: orb GIF + power figure are non-empty, unit-tested, and skip cleanly without matplotlib
- [x] ISC-92: manuscript gains a noise-sensitivity results section citing the power figure, an idealized noise-model paragraph in methods, and the orb figure; abstract + token note updated
- [x] ISC-93: Honesty reframe (post-Advisor iter-3): (a) "statistical power" → "noise-sensitivity/robustness sweep" (no hypothesis-test power computed); (b) noise model labelled *idealized i.i.d. Gaussian*, NOT optics-derived (ignores spatial/temporal correlation, heteroscedasticity, bias); (c) pupil robustness explicitly marked *conditional / by-construction* under `pupil_noise_scale`, not an emergent finding, while gaze-vs-saccade ordering IS emergent; (d) `power.sigma_to_pixels()` added and the defensible headline stated — saccade timing is marginal under the package's idealized subpixel-to-pixel landmark perturbation regime and must not be reported as a real-device accuracy bound; (e) saccade F1 interval-overlap matching documented

### Modular manuscript expansion & accessible statistics (iteration 4)
- [x] ISC-94: composable statistics: `power.summary_records()` (flat JSON/DataFrame-friendly rows: σ, pixels, per-metric mean/ci/std, n) + `power.format_summary_markdown()` render the sweep as an accessible Markdown table; figure + table share ONE `NoiseSweepResult` so they cannot drift
- [x] ISC-95: accessible figure: `generate_power_figure.py` refactored into a composable per-panel function; each signal encoded by colour AND marker AND line style (greyscale/colour-blind legible), units on every axis, a secondary σ→pixel top axis, gridlines, CI bands, threshold crossings; writes `noise_summary.md` alongside `noise_power.png`
- [x] ISC-96: manuscript modularly expanded from ~6 monolithic files to 21 single-topic composable section modules (02a–02h methods, 03a–03e results, 04 discussion, 05 limitations, 06 conclusion, S01 statistical methods, S02 software architecture); each file has exactly one labelled H1
- [x] ISC-97: manuscript integrity: every `@sec/@fig/@eq/@tbl` cross-reference resolves to a defined label (0 unresolved), all citation keys resolve in references.bib (0 missing), no hand-written section numbers, statistics table carries per-cell ± CI and n
- [x] ISC-98: S01 statistical-methods supplement documents the CI method (eq), seeded-trial independence/determinism, interval-overlap F1, threshold interpolation, and the single-source-of-truth pipeline — the statistics are valid and auditable
- [x] ISC-99: Statistical-validity fix (post-Advisor iter-4): CIs switched from symmetric Student-t to **bootstrap percentile** (B=2000, fixed-seeded) because F1∈[0,1] and r∈[-1,1] make a symmetric interval invalid near 1.0; verified CIs never exceed the metric bounds; the determinism-vs-CI tension reconciled in prose (CI = Monte Carlo SE over the noise realisations the seeds sample, not rerun variability); manuscript table re-synced cell-for-cell to the regenerated bootstrap values

### RedTeam refresh: configurable methods, real capture writes, gallery, manuscript sync (iteration 5)
- [x] ISC-100: `config.py` defines frozen Detection/Pupil/Capture/Analysis/Figure configs; `pipeline.analyze_gaze` accepts `AnalysisConfig` while preserving legacy keyword arguments
- [x] ISC-101: `SessionReport` carries PSO candidates, quality metrics, and config metadata with default-safe fields so existing callers still construct reports
- [x] ISC-102: configurable pipeline routes fixed I-VT and adaptive I-VT, optionally detects typed PSOs, and reports the actual threshold used
- [x] ISC-103: `itrace.stats.bootstrap` and `itrace.stats.events` provide reusable percentile/bootstrap utilities and interval-overlap precision/recall/F1
- [x] ISC-104: `synthetic.synthetic_session` creates seeded multi-saccade, PSO, blink, dropout, and pupil-dilation recordings with truth records for verifier tests
- [x] ISC-105: `capture.CaptureSample` and `itrace record` write real gaze CSV and optional relative-pupil CSV; timestamping uses monotonic time, not a hardcoded 30 Hz counter
- [x] ISC-106: `itrace camera-probe` performs a short optional dependency/camera access smoke test without treating hardware availability as scientific validation
- [x] ISC-107: `all` optional extra installs capture + dashboard + figures + web; optional import smoke tests verify cv2, mediapipe, streamlit, plotly, matplotlib, fastapi, and uvicorn together
- [x] ISC-108: `itrace figures --animations` renders the static gallery plus deterministic synthetic replay GIF; Matplotlib figures are closed after tests so no open-figure warning remains
- [x] ISC-109: README, manuscript, and ISA are synchronized to v0.4.1, current metric-ledger evidence, bootstrap percentile CIs, and the corrected capture/pupil-proxy/live-HTML contract
- [x] ISC-110: manuscript integrity test checks source cross-references, citation keys, stale metric literals, and figure links
- [x] ISC-111: full gate re-green after refresh: `pytest --cov`, `ruff check`, `ruff format --check`, `mypy --strict`, and `uv lock --check`
- [x] ISC-112: Pulse/render artifacts refreshed: figures, animations, Markdown/text/PDF manuscript outputs regenerated from the updated source
- [x] ISC-113: `itrace record --records-out` writes a full capture-record CSV with frame index, monotonic timestamp, gaze, relative pupil proxy, FPS estimate, and quality columns
- [x] ISC-114: `itrace record` and `itrace camera-probe` suppress native OpenCV/MediaPipe stderr diagnostics by default while exposing `--backend-logs` for debugging
- [x] ISC-115: after camera permission was granted, the real webcam path succeeded: 5/5 probe frames detected faces and a 30-frame record wrote 29 detected face samples across gaze, pupil, and capture-record CSVs

### Local HTML orchestrator (iteration 6)
- [x] ISC-116: `web` optional extra declares FastAPI, uvicorn, and current Starlette WebSocket test-client support; `all` includes it without making web deps core requirements
- [x] ISC-117: `WebcamSource.frames()` remains backward-compatible while `WebcamSource.live_frames()` adds frame dimensions, an eye-region bounding box, and a JPEG/base64 eye crop
- [x] ISC-118: eye bounding-box computation uses Face Mesh eye/iris landmarks, pads and clamps to frame bounds, and is unit-tested with synthetic MediaPipe-shaped landmarks
- [x] ISC-119: `itrace.live.create_app()` exposes `GET /`, `GET /api/status`, `WebSocket /ws/live`, and `POST /api/export` with hardware-free WebSocket tests via a synthetic frame source
- [x] ISC-120: `itrace live-html` exposes `--camera`, `--host`, `--port`, `--output-dir`, `--backend-logs`, and `--open-browser`; missing web dependencies raise an actionable `uv sync --extra web` message
- [x] ISC-121: packaged HTML/CSS/JS assets center a large zoomed live eye crop above controls, status, gaze/saccade/pupil diagnostics, and native Canvas/SVG visualizations
- [x] ISC-122: live WebSocket messages serialize `CaptureSample`, rolling `GazeStream`/`PupilStream`, `SessionReport` summaries, event spans, and plot-ready series from the Python package path rather than duplicating analysis in JavaScript
- [x] ISC-123: export is explicit and local-only: no files are written without `--output-dir`, and export writes gaze CSV, pupil CSV, capture-record CSV, and report JSON only on request
- [x] ISC-124: import-safety tests prove `import itrace` and `import itrace.live` do not load optional capture/dashboard/figure/web libraries at module load

### Robust analysis continuation (iteration 7)
- [x] ISC-125: `calibration.AffineCalibration` fits and applies a 2-D affine gaze calibration from known targets, with RMS/mean/median/p95/max error summaries
- [x] ISC-126: `calibration.robust_gaze_quality` reports valid/dropout fraction, median sample interval, timing jitter, longest gap, large-gap count, and nonmonotonic timestamp count; `SessionReport.quality` surfaces those fields
- [x] ISC-127: I-VT exposes `merge_gap_s` to bridge short subthreshold holes inside one saccade before duration filtering; tests prove a split ramp is repaired only when the configured window permits it
- [x] ISC-128: `PupilConfig` controls blink threshold, interpolation padding, Butterworth cutoff, and filter order in the real `pipeline.analyze_pupil` path
- [x] ISC-129: pupil summaries include validity/blink burden plus derivative dynamics (`peak_dilation_velocity`, `peak_constriction_velocity`) alongside cleaned mean/std/min/max and phase counts
- [x] ISC-130: invalid robustness settings fail loudly (`merge_gap_s`, blink padding, smooth cutoff/order, calibration shape/finite/ridge checks) rather than producing silent default behavior
- [x] ISC-131: README, manuscript source, and manuscript README document the calibration, merge-gap, configurable pupil, and current gate-metric evidence
- [x] ISC-132: continuation gates re-green after the robustness additions: targeted tests, full `pytest --cov`, `ruff check`, `ruff format --check`, and `mypy --strict`

### TODO implementation sprint (iteration 8)
- [x] ISC-133: `--config-json` loads `AnalysisConfig`-shaped JSON for `analyze` and `stats`; explicit CLI flags override JSON settings and tests prove precedence
- [x] ISC-134: `itrace validate-recording` writes quality, event plausibility, pupil validity, calibration availability, report-validation, warning, error, and event-CI blocks
- [x] ISC-135: `itrace calibrate` fits affine calibration from `raw_x,raw_y,target_x,target_y`, writes calibration/error JSON, and can apply it to a gaze CSV
- [x] ISC-136: live HTML calibration stores `AffineCalibration` in `LiveState`, streams raw and calibrated gaze, exposes target-range metadata, and exports calibration artifacts
- [x] ISC-137: default-off detection robustness adds min inter-event gap, max saccade duration, edge-event rejection, and provisional smooth-pursuit events
- [x] ISC-138: blink-aware gaze interpolation fills only short finite-bounded invalid runs and refuses long gaps
- [x] ISC-139: pupil response summaries include latency-to-peak, dilation AUC, baseline-relative peak change, velocity summaries, and phase fractions
- [x] ISC-140: synthetic sessions support deterministic timestamp jitter, correlated noise, head-pose drift, and lighting/dropout quality flags
- [x] ISC-141: figure gallery includes quality diagnostics: dropout raster, sampling-interval histogram, calibration residuals, and pupil velocity
- [x] ISC-142: report payload validation and `CaptureBackend` protocol scaffold future integrations without adding runtime dependencies
- [x] ISC-143: README, TODO, protocol docs, manuscript sources, and generated manuscript artifacts are synchronized to the sprint behavior and current gate metrics

### Validation-domain and live interface expansion (iteration 9)
- [x] ISC-144: `validation.synthetic_validation_suite` runs seeded within-domain recovery and cross-domain stability summaries across clean, jitter, head-drift, and low-light/dropout synthetic domains
- [x] ISC-145: synthetic-domain validation reports interval-overlap saccade precision/recall/F1 plus amplitude, direction, peak-velocity, duration, gaze-quality, and pupil-validity summaries with bootstrap intervals where repeated samples permit them
- [x] ISC-146: `validation.live_recording_diagnostics` reports live sampling regularity, finite gaze fraction, pupil-valid fraction, path length, dispersion, quality index, and warnings while explicitly refusing live reference-truth F1
- [x] ISC-147: `itrace synthetic-validation` writes the validation-suite JSON and the live HTML app exposes `/api/validation/synthetic` without adding required or top-level optional imports
- [x] ISC-148: the live HTML page includes a modular validation panel with Python-backed synthetic-domain F1 visualization, summary table, live quality cards, and no browser-side detector/statistics duplication
- [x] ISC-149: adaptive I-VT threshold estimation uses finite velocity samples only, so dropout-heavy synthetic/live traces do not emit NumPy empty-slice warnings

### Verifier-first scholarship and methods expansion (iteration 10)
- [x] ISC-150: `docs/verification_metrics.json` is the single source for version, gate date, test count, coverage, repository URL, and render evidence; the manuscript renderer hydrates metrics from it by default
- [x] ISC-151: project URLs in `pyproject.toml` point to the iTrace repository, and tests reject template-repository URL drift
- [x] ISC-152: public docs test against stale metric literals, including prior test-count and coverage values
- [x] ISC-153: `capture.iris_landmarks_to_binocular_sample` returns additive per-eye diagnostics, vergence, vertical disparity, asymmetry, and an averaged relative pupil proxy without importing MediaPipe
- [x] ISC-154: `pupilseg` segments caller-supplied eye crops in pure NumPy/SciPy and returns explicit `PupilUnit.PIXELS` or pupil/iris-relative samples with confidence and quality fields
- [x] ISC-155: live HTML calibration uses backend-owned start/sample/fit/reset session endpoints, with per-target finite-sample aggregation before fitting `AffineCalibration`
- [x] ISC-156: `benchmark` and `itrace benchmark` compare iTrace or external event outputs against caller-supplied truth/comparator files and include the truth boundary in every report
- [x] ISC-157: README, scholarship audit, research brief, protocol docs, TODO, ISA, manuscript source, and bibliography document the new methods without adding device-validation or millimetre-pupil claims

### PDF and empirical-session expansion (iteration 11)
- [x] ISC-158: manuscript config and standalone render remove the FractAI affiliation, render ORCID `0000-0001-6232-9096`, use 0.42 in margins, and replace Pandoc's default title with a generated cover page containing `output/figures/cover_visual.png`
- [x] ISC-159: `scripts/generate_graphical_abstract.py` writes `cover_visual.png` and `graphical_abstract.png`, and the graphical abstract is inserted before existing result figures as Figure 1
- [x] ISC-160: `experiments.default_eye_video_protocol` defines derived-only fixed-gaze, reading, and center/four-corner target trials with deterministic target schedules
- [x] ISC-161: `experiments.experiment_report` estimates session-specific quality, drift, jitter, held-out target residuals, and target latency from derived `CaptureSample` records while carrying explicit truth/storage boundaries
- [x] ISC-162: live HTML exposes backend-owned experiment session/trial/report/export/reset routes and packaged controls without adding browser-side analysis
- [x] ISC-163: `itrace experiment-report` rebuilds the derived empirical report from `experiment_manifest.json` plus capture-record CSV, so live-session summaries are reproducible without a browser or camera

### Figure overhaul and empirical-pilot hydration (iteration 12)
- [x] ISC-164: `src/itrace/viz/palette.py` centralizes figure palette, font scale, panel styling, and result-callout helpers used by regenerated publication figures
- [x] ISC-165: the graphical abstract uses a symbolic capture→landmarks→core→reports layout plus separate synthetic-truth, closed-loop, gate, and single-pilot evidence badges hydrated from JSON artifacts when present
- [x] ISC-166: `scripts/summarize_empirical_pilot.py` reads only derived `experiment_report.json` exports and writes `docs/empirical_pilot_metrics.json` plus `output/figures/empirical_pilot_summary.png`, with unavailable fields explicit before recording
- [x] ISC-167: the manuscript includes a single-pilot empirical diagnostics section whose values hydrate from `docs/empirical_pilot_metrics.json` and whose limitation sentence preserves the no-reference-device-validation boundary

### Documentation, visualization, and analysis traceability (iteration 13)
- [x] ISC-168: `docs/TRACEABILITY_MATRIX.json` maps public validation claims and figures to docs, tested Python methods or scripts, tests, generated evidence files, and truth-boundary wording
- [x] ISC-169: `docs/figure_manifest.json` records generated publication figure provenance: source script/data, byte count, SHA-256 hash, dimensions, and nonblank pixel variance
- [x] ISC-169: documentation integrity tests verify the traceability matrix paths, pure-core method symbols, public documentation links, and figure references without importing optional visualization dependencies
- [x] ISC-170: `docs/VALIDATION_PROTOCOLS.md`, README, TODO, and ISA state the traceability protocol: Python-computed payloads are the source of truth, browser Canvas/SVG is display-only, and UI screenshots are not device-validation evidence
- [x] ISC-171: `itrace.stats.range_bridge` writes a synthetic-to-empirical bridge payload and publication figure that separates observed N=1 local scale, synthetic/stress evidence, and non-comparable quantities without adding device-validation claims
- [x] ISC-172: `itrace.stats.evidence` writes a statistical interpretation ledger and publication figure that maps each reported statistic to its source artifact, estimand, scholarship basis, and explicit non-claim boundary
- [x] ISC-173: `docs/empirical_sessions_manifest.json` plus `scripts/aggregate_empirical_sessions.py` provide a single-participant/device repeated-session empirical intake gate that validates metadata/provenance, rejects raw-video report paths, writes `docs/empirical_sessions_summary.json`, treats the five-session/two-condition diagnostic pilot as v1-ready, and records the 12-session/backlit/reference-backed validation target as future scope
- [x] ISC-174: the live HTML empirical-session panel exposes condition, participant, device, session-group, and reference-kind controls before session start, sends those fields to the backend, and displays the assigned session ID, manifest path, auto-save state, condition, reference lane, and report path
- [x] ISC-175: non-`none` empirical `reference_kind` rows count toward v1 readiness only when a valid repo-relative `reference_artifact` JSON exists; manual-annotation artifacts validate derived target-window annotations and reject raw-video paths

## Test Strategy

| isc | type | check | threshold | tool |
|-----|------|-------|-----------|------|
| ISC-1..6 | build/import | resolve, import, grep lazy | exit 0; no top-level cv2 | `uv`, `python -c`, `rg` |
| ISC-7..10 | unit | dataclass fields, validation | ValueError on bad input | `pytest` |
| ISC-11..17 | numeric | analytic round-trip / known answer | rel-err < stated tol | `pytest` + numpy.allclose |
| ISC-18..27 | ground-truth recovery | synthesize events, detect, compare | counts exact; amp 5%; vel 10% | `pytest` |
| ISC-27 | cross-vendor | diff vs Forge reference | same event boundaries | `pytest` |
| ISC-28..30 | fit recovery | recover synthetic params | params 10% | `pytest` + scipy |
| ISC-31..39 | pupil ground-truth | constructed blink/sine | no residual NaN; peaks aligned | `pytest` |
| ISC-40..42 | encoding | known sequence → known string/counts | exact equality | `pytest` |
| ISC-43..45 | capture (no hw) | float-array path; absent-dep error | clear error message | `pytest` |
| ISC-46..51 | integration | end-to-end synthetic; file outputs | report shape; PNG exists | `pytest`, `Read`, file stat |
| ISC-52 | gates | full toolchain run | green; cov≥90 | `uv run` quoted output |
| ISC-116..124 | live HTML | optional web deps, WebSocket payloads, export, import safety, UI assets | routes respond; no optional import leak | `pytest --extra web`, browser smoke |
| ISC-125..132 | robustness | calibration, quality, gap merge, pupil config, gates | exact synthetic recovery; green full suite | `pytest`, `ruff`, `mypy` |
| ISC-133..143 | sprint | config JSON, validation CLI, calibration, live HTML calibration, quality figures, docs | stable payloads; green full suite | `pytest`, `ruff`, `mypy`, render |
| ISC-144..149 | validation domains | within/across synthetic suites + live diagnostics | stable JSON; no live truth overclaim; green tests | `pytest`, browser smoke, `ruff`, `mypy` |
| ISC-150..157 | verifier pass | metrics ledger, binocular diagnostics, pupilseg, benchmark, docs | no stale public metrics; explicit truth/unit boundaries | `pytest`, `ruff`, `mypy`, render, route/browser smoke |
| ISC-158..163 | PDF/experiment pass | cover/ORCID/layout, graphical abstract, guided empirical sessions | rendered assets exist; no FractAI; derived-only reports; live routes tested | `pytest`, `node --check`, render, route smoke |
| ISC-164..167 | figure/pilot pass | shared visual grammar, graphical abstract badges, pilot metrics JSON, pilot summary figure | PNG/GIF artifacts exist; no raw video; manuscript tokens resolve | `pytest`, `ruff`, render, browser/live smoke |
| ISC-168..170 | traceability pass | claims/figures mapped to docs, tested methods, tests, evidence, boundaries | matrix paths and pure-core symbols resolve; docs link matrix | `pytest`, crossref audit |
| ISC-173..175 | empirical release gate | repeated sessions, live metadata, manual evidence artifacts | no enum-only reference evidence; auto-save metadata persists | `pytest`, browser smoke, render |

## Features

| name | description | satisfies | depends_on | parallelizable |
|------|-------------|-----------|------------|----------------|
| skeleton | pyproject, __init__, py.typed, version | ISC-1..6 | — | no (first) |
| config | frozen public configs for analysis/capture/figures | ISC-100 | skeleton | yes |
| calibration | affine target calibration + gaze-quality diagnostics | ISC-125..126 | types | yes |
| validation | synthetic-domain recovery + live plausibility diagnostics | ISC-144..149 | synthetic,pipeline,stats | yes |
| types | dataclasses + enum + GazeStream | ISC-7..10 | skeleton | yes |
| geometry | pix2deg/deg2pix, iris→angle, normalize | ISC-11..14 | types | yes |
| velocity | Savitzky–Golay pos2vel | ISC-15..17 | types | yes |
| events | I-VT, I-DT, microsaccades, gap merge, properties | ISC-18..27,127 | velocity,geometry | yes |
| adaptive detection | adaptive I-VT, PSO, ISI, acceleration | ISC-100..103 | events,config | yes |
| mainseq | saturating + power-law fit | ISC-28..30 | events | yes |
| pupil | blink/baseline/smooth/MAD/configurable quality | ISC-31..36,128..129 | types | yes |
| pupilseg | pure eye-crop segmentation, pixels/relative only | ISC-154 | types | yes |
| pupilphase | causal streaming phase detector | ISC-37..39 | pupil | yes |
| encoding | direction chars + n-grams | ISC-40..42 | events | yes |
| capture | webcam/MediaPipe thin orchestrator + binocular diagnostics | ISC-43..45,153 | types,geometry | yes |
| live HTML | FastAPI/WebSocket local orchestrator + packaged HTML assets | ISC-116..124 | capture,pipeline,config | yes |
| live calibration | backend-owned affine calibration sessions streamed to the HTML UI | ISC-136,155 | live HTML,calibration | yes |
| synthetic session | multi-event gaze+pupil truth generator | ISC-104 | types | yes |
| benchmark | external truth/comparator event scoring | ISC-156 | stats.events | yes |
| empirical pilot | derived live-session metrics and manuscript hydration | ISC-160..167 | experiments,live HTML | yes |
| figure system | shared palette, graphical abstract, gallery, animations | ISC-51,89..92,164..165 | synthetic,pipeline,power | yes |
| pipeline+io | end-to-end + CSV/JSON | ISC-46..47,49 | all core | no (integrative) |
| cli | typer commands | ISC-48..49 | pipeline | no |
| dashboard | streamlit/plotly (optional) | ISC-50 | pipeline | yes |
| figures | scripts/CLI → PNG/GIF gallery | ISC-51,108 | events,mainseq,viz | yes |
| verification metrics | single metrics ledger for docs/render/tests | ISC-150..152 | docs,scripts | yes |
| validation protocols | public-dataset, reference-tracker, benchmark, and release scaffolds | ISC-143,157 | docs | yes |
| traceability matrix | claim/figure evidence map tying docs and visualization outputs to tested methods | ISC-168..170 | docs,tests,figures | yes |
| guided experiments | derived live protocols, reports, and exports | ISC-160..163 | capture,calibration,stats,live HTML | yes |
| synthetic-empirical bridge | N=1 local scale compared to synthetic/stress/statistical variables with explicit non-comparability labels | ISC-171 | stats.range_bridge,viz | yes |
| statistical interpretation ledger | generated map from statistics to estimands, sources, scholarship basis, and non-claims | ISC-172 | stats.evidence,viz,docs | yes |
| empirical sessions manifest | repeated-session intake, diagnostic-v1 readiness, future validation scope, and derived-only provenance checks | ISC-173 | empirical,scripts,docs | yes |
| empirical release controls | live metadata controls and validated reference artifacts for future device-validation scope | ISC-174..175 | live HTML,empirical,docs | yes |
| cover + graphical abstract | generated cover visual and Figure 1 graphical abstract | ISC-158..159 | figures,render | yes |
| gates | tests+ruff+mypy+cov | ISC-52 | everything | no (last) |

## Verification (iteration 9 — validation-domain/live UI expansion)

All evidence below is from real commands run on 2026-05-31 in
`working/iTrace`.

- focused validation/web/package tests: `20 passed in 2.03s` from
  `uv run pytest tests/test_validation_domains.py tests/test_live_html.py tests/test_package.py -q`.
- warning-hard regression: `2 passed in 0.31s` from adaptive-threshold and
  synthetic-validation endpoint checks with `-W error`.
- synthetic validation CLI: `wrote output/synthetic_validation.json (domains=4,
  macro_f1=0.673, worst=head_drift)` from
  `uv run itrace synthetic-validation --out output/synthetic_validation.json --repetitions 2`.
- full pytest/coverage: passed the promotion coverage floor in that iteration;
  exact historical gate numerics are superseded by
  `docs/verification_metrics.json`.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: `Success: no issues found in 42 source files` from
  `uv run mypy src/itrace`.
- JavaScript syntax: `node --check src/itrace/web_static/app.js` exited 0.

## Decisions

- 2026-05-29 — ISA authored directly via Write (doctrine v6.2.x sanctions
  model-direct ISA authoring until ISA-skill CLI lands); E5 Interview satisfied
  by an inline self-interview against the in-repo RESEARCH_BRIEF (the ideal
  state is unusually well-specified by the brief, so the open questions are
  architectural, not requirements-elicitation).
- 2026-05-29 — `refined:` ISC floor: E5 soft floor is ≥256; the granularity
  rule yields a natural N=52 atomic probes for a focused scientific package.
  Padding to 256 would manufacture non-atomic noise. Soft floor relaxed with
  this show-your-math; thinking floor (HARD, ≥8) is met with 10.
- 2026-05-29 — Architecture: pure core (numpy/scipy) vs optional hardware shell.
  MediaPipe is absent on this box and platform-fragile (no py3.13 wheel) —
  validating the core against synthetic ground truth is what makes the science
  CI-able. Capture takes already-extracted landmark floats so it is testable
  without a camera.
- 2026-05-29 — Delegation show-your-math: package authored single-coherent
  (tightly-coupled modules) rather than fanned out to parallel writers
  (integration risk). Forge is used for an *independent numerical oracle*
  (ISC-27) and for the E5 cross-vendor VERIFY audit; Cato + Advisor at VERIFY.
- 2026-05-29 — Forge/Codex (GPT-5.4) hit a hard usage limit mid-BUILD (resets
  2026-05-30 ~13:48). It blocks BOTH the Forge oracle and the Cato/Forge
  cross-vendor VERIFY audit (Cato runs the same codex backend). Mitigation:
  (a) ISC-27 satisfied by an in-tree independent second implementation with a
  deliberately different formulation; (b) cross-vendor audit downgraded to
  DEFERRED-VERIFY with follow-up `FOLLOWUP-XVENDOR: re-run Forge+Cato audit
  after 2026-05-30 13:48`; (c) Advisor (PAI Inference.ts, not codex) still runs
  at VERIFY as the E5 commitment-boundary gate. Per Rule 2a truth-table, a
  both-skipped cross-vendor result is escalated to the user, not laundered as
  pass.
- 2026-05-30 — Capture dependency pin: `mediapipe>=0.10.21,<0.10.22` is used
  because later 0.10.x wheels on this Python expose only `mediapipe.tasks` and
  not the legacy `mp.solutions.face_mesh.FaceMesh` API used by the thin webcam
  shell. Optional import smoke tests verify the pinned Face Mesh path.
- 2026-05-30 — Live HTML architecture: the browser is only a local orchestrator.
  The backend streams eye crops and Python-derived analysis over FastAPI
  WebSockets; JavaScript draws diagnostics but does not reimplement detection,
  statistics, pupil processing, or export logic. Persistence is explicit and
  off by default.
- 2026-05-31 — Robustness continuation: calibration stays in the pure core,
  because fitting a target map and reporting calibration error are analysis
  operations rather than webcam operations. Live webcam gaze remains labelled as
  relative/algorithmic unless fitted against known targets or a reference device.
  The saccade merge-gap repair is opt-in and reported in config metadata, so it
  improves robustness without hiding a post-hoc event-editing choice.

## Changelog

- conjectured: a green synthetic-ground-truth suite + clean ruff/mypy is
  sufficient to call a webcam eye-tracking package "real and validated".
  refuted_by: the E5 Advisor (`Inference.ts --mode advisor`) — "validated" in
  measurement science means agreement with a reference measurement on the real
  physical quantity; synthetic-only recovery is *algorithmic verification* and
  is circular w.r.t. device accuracy, and the dominant error sources live in the
  untested cv2/MediaPipe capture path.
  learned: separate the two claims explicitly; ship the algorithmic-verification
  claim (which is true and evidenced) and disclose the device-validation gap
  prominently rather than letting "validated" overclaim.
  criterion_now: README + manuscript carry a "Validation status" disclosure
  relabelling the work as algorithmically verified, not device-validated
  (new ISC-53 below).
- conjectured (iter-2): a 3-D forward-model closed loop with bounded nonzero
  error "validates the geometry/landmark path".
  refuted_by: the iter-2 Advisor — the loop only bounds the discrepancy the
  forward model and estimator do NOT share; every shared idealization (pinhole,
  no corneal refraction, spherical eye, zero kappa, synthetic≠real MediaPipe)
  contributes zero residual by construction, so 0.16° is a LOWER bound on real
  error; and pupil r=1.0 is a near-identity (monotone) consistency check, not
  recovery accuracy.
  learned: a closed loop between two of your own models is internal-consistency
  verification, not validation; disclose the shared-assumption ledger and frame
  the residual as a lower bound. (Reinforces memory verified-is-not-validated.)
  criterion_now: README + manuscript carry the assumption ledger, lower-bound
  framing, and pupil reclassification (new ISC-78).
- conjectured (iter-3): the noise sweep shows an emergent three-way ordering
  "saccades < gaze < pupil in noise tolerance" because a derivative amplifies
  and an average suppresses — a clean first-principles finding.
  refuted_by: the iter-3 Advisor — only saccade<gaze is emergent (both propagate
  the landmark-noise model); the pupil position is set by the free
  `pupil_noise_scale` parameter (no segmentation is modelled), so the "average
  suppresses" mechanism describes a system not implemented and the ordering
  flips if the knob turns. Also: "statistical power" is a misnomer for a
  sensitivity sweep; "first-principles Gaussian" overstates an idealized i.i.d.
  model.
  learned: an ordering that moves under a free parameter is not a finding;
  separate emergent results from assumed ones; translate σ to physical pixels —
  the real headline is that saccade breakdown (≈0.9 px) is below the real
  landmark-localisation floor. Reinforces [[verified-is-not-validated]].
  criterion_now: docs reframed as noise-sensitivity, pupil marked conditional,
  pixel-floor headline added, `sigma_to_pixels` shipped (new ISC-93).
- conjectured (iter-4): a 95% Student-t CI is fine for the sweep's three metrics.
  refuted_by: the iter-4 Advisor — F1∈[0,1] and r∈[-1,1] are bounded, and near
  1.0 a symmetric t-interval can extend past the bound (CI > 1), which a stats
  reviewer rejects on sight; also a CI on a deterministic pipeline needs its
  meaning stated (variability over sampled noise realisations, not reruns).
  learned: use bound-respecting CIs (bootstrap percentile / Fisher-z) for bounded
  metrics; state what a CI means when the pipeline is deterministic; verify
  greyscale/colour-blind legibility by rendering, not asserting.
  criterion_now: bootstrap percentile CIs (bound-verified), reconciled framing in
  S01/02h, accessible composable stats table + figure (new ISC-94..99).
- conjectured (iter-5): optional capture dependencies can simply float to the
  newest MediaPipe wheel because import safety keeps them isolated.
  refuted_by: the iter-5 optional smoke test — MediaPipe 0.10.35 imported but
  did not expose the `mp.solutions.face_mesh.FaceMesh` path used by the webcam
  shell, so capture would fail only after a user installed the extra.
  learned: optional dependency health is still part of the package contract;
  pin the legacy Face Mesh API range, smoke-test the actual symbol path, and
  report camera permission/hardware failure separately from dependency failure.
  criterion_now: `mediapipe>=0.10.21,<0.10.22`, `all` extra smoke test, and
  `itrace camera-probe` environmental failure path (new ISC-106..107).
- conjectured (iter-6): a browser dashboard should own as much live analysis as
  possible to feel responsive.
  refuted_by: the project invariant — validated methods live in Python and the
  browser must stay a thin orchestrator, otherwise the live view and package API
  drift into two implementations.
  learned: stream compact, plot-ready summaries from the tested Python path and
  let Canvas/SVG handle only rendering. This keeps the eye-crop UI responsive
  while preserving the method boundary.
  criterion_now: `itrace live-html`, `capture.live_frames`, `live.create_app`,
  native UI assets, explicit export, and web import-safety tests (new
  ISC-116..124).
- conjectured (iter-7): the configurable pupil and detection objects were
  sufficient once they existed.
  refuted_by: follow-up tests — some knobs were metadata only (`PupilConfig`
  threshold/smoothing choices), and a split-saccade robustness setting needed a
  synthetic trace whose ground truth actually contained a short low-velocity
  hole.
  learned: public configuration must be execution-wired and verifier-visible.
  A knob that appears in a report but does not affect the method path is worse
  than no knob because it invites false auditability.
  criterion_now: affine calibration/error summaries, robust gaze-quality
  metrics, opt-in I-VT gap merge, wired pupil thresholds/padding/filter settings,
  and new tests (ISC-125..132).

## Verification (iteration 10 — verifier-first scholarship and methods expansion)

All evidence below is from real commands run on 2026-06-04 in
`working/iTrace`.

- scholarship: primary/official pupil segmentation and pupillometry sources
  were checked and recorded in `docs/SCHOLARSHIP_AUDIT.md`; the manuscript cites
  PupilEXT, ElSe, and PuRe only as comparison/boundary sources, not as iTrace
  validation evidence.
- focused verifier suite: `57 passed, 2 warnings in 7.06s` from
  `uv run pytest tests/test_manuscript_integrity.py tests/test_capture.py
  tests/test_pupilseg.py tests/test_benchmark.py tests/test_live_html.py
  tests/test_saccades.py tests/test_package.py --tb=short`.
- full pytest/coverage: superseded by the iteration-11 metric ledger and final
  gates; exact prior-run values are intentionally not repeated as public
  evidence.
- ruff format: superseded by the iteration-11 metric ledger and final gates.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: superseded by the iteration-11 metric ledger and final gates.
- JavaScript syntax: `node --check src/itrace/web_static/app.js` exited 0.
- optional extra import smoke: `optional imports ok 0.4.1` from
  `uv run --extra capture --extra dashboard --extra figures --extra web python -c ...`.
- live route smoke: local `itrace live-html --host 127.0.0.1 --port 8765`
  served the index, static JS, `/api/status`, calibration session start/reset,
  and synthetic-validation GET route; the temporary server was stopped afterward.
- lockfile: `Resolved 105 packages in 6ms` from `uv lock --check`.

## Verification (iteration 15 — synthetic-empirical range bridge refresh)

All evidence below is from real commands refreshed on 2026-06-09 in
`working/iTrace`.

- focused empirical/documentation/manuscript tests:
  `57 passed` from
  `uv run pytest tests/test_empirical_sessions.py tests/test_live_html.py tests/test_manuscript_integrity.py tests/test_documentation_traceability.py --tb=short`.
- full pytest/coverage: `528 passed, 2 warnings`; `Total coverage:
  90.23%` from `uv run pytest --cov=itrace`.
- ruff format: `121 files already formatted` from
  `uv run ruff format --check src/ tests/ scripts/`.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: `Success: no issues found in 56 source files` from
  `uv run mypy src/itrace`.
- JavaScript syntax: `node --check src/itrace/web_static/app.js` exited 0.
- lockfile: `Resolved 105 packages in 11ms` from `uv lock --check`.
- runtime optional extra import smoke: `optional imports ok 0.4.1` from
  `uv run --extra capture --extra dashboard --extra figures --extra web python -c ...`.
- live route smoke: `live route smoke ok` from a TestClient check of `/`,
  `/api/status`, experiment session start/status/reset, and synthetic validation.
- live browser smoke: `chrome-devtools-axi` exercised a temporary synthetic
  dashboard, completed two three-trial sessions, confirmed automatic exports,
  manifest upserts, `Start Next Session` allocation without overwriting prior
  exports, and zero browser console errors; the canonical collection dashboard
  was then verified at `http://127.0.0.1:8766` with `output/empirical_pilot` and
  `docs/empirical_sessions_manifest.json`.
- render: `scripts/render_manuscript.py --metrics-json docs/verification_metrics.json
  --empirical-json docs/empirical_pilot_metrics.json --empirical-sessions-json
  docs/empirical_sessions_summary.json` regenerated Markdown, text, LaTeX, and
  PDF; `pdftotext`/`pdfinfo` confirmed the ORCID-only author metadata and no
  cover-missing fallback text.
- generated visual evidence: `scripts/generate_figures.py` refreshed
  `docs/figure_manifest.json` and the figure gallery; the graphical abstract was
  inspected as a rendered PNG and shows the `528 tests / 90.23% coverage` badge.

## Verification (iteration 8 — TODO implementation sprint)

All evidence below is from real commands run on 2026-05-31 in
`working/iTrace`.

- focused sprint tests: `16 passed in 3.19s` from config JSON/CLI precedence,
  validation JSON, calibration CSV application, smooth pursuit, blink-aware gaze
  interpolation, synthetic-noise determinism, live calibration/export, report
  validation, and quality figures.
- full pytest/coverage: passed the promotion coverage floor in that iteration;
  exact historical gate numerics are superseded by
  `docs/verification_metrics.json`.
- ruff format: `88 files already formatted` from
  `uv run ruff format --check src/ tests/ scripts/`.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: `Success: no issues found in 42 source files` from
  `uv run mypy src/itrace`.
- lockfile: `Resolved 105 packages in 3ms` from `uv lock --check`.
- optional imports: `uv run --extra capture --extra dashboard --extra figures --extra web`
  imported `cv2`, `mediapipe`, `streamlit`, `plotly`, `matplotlib`, `fastapi`,
  and `uvicorn`.
- render: all figure scripts, `itrace figures --out-dir output/figures --animations`,
  and Pandoc Markdown/text/LaTeX/PDF manuscript builds completed with regenerated
  artifacts.
- manuscript integrity: passed after updating the rendered manuscript and public
  docs to the then-current metrics ledger.
- camera smoke: `camera=0 read_frames=30 detected_face_samples=0`; the camera
  opened and delivered frames, but no face was detected in the captured frames,
  so this is an environmental face-content/permission/positioning result rather
  than a package failure.
- live HTML route smoke: `http://127.0.0.1:8765` served `/`, `/static/app.js`,
  `/static/app.css`, `/sw.js`, and `/api/status`; status reported configured
  output dir, FastAPI/uvicorn/cv2/MediaPipe availability, and calibration target
  range ±15°.

## Verification (iteration 7 — robustness continuation)

All evidence below is from real commands run on 2026-05-31 in
`working/iTrace`.

- targeted robustness tests: `9 passed in 0.23s` from calibration, merge-gap,
  pupil-summary, and pipeline routing tests; follow-up config/pupil/direct
  validation tests: `7 passed in 0.26s`.
- pytest/coverage: passed the promotion coverage floor in that iteration; exact
  historical gate numerics are superseded by `docs/verification_metrics.json`.
- ruff format: `84 files already formatted` from
  `uv run ruff format --check src/ tests/ scripts/`.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: `Success: no issues found in 39 source files` from
  `uv run mypy src/itrace`.
- optional imports: `uv run --extra capture --extra dashboard --extra figures --extra web`
  imported `cv2`, `mediapipe`, `streamlit`, `plotly`, `matplotlib`, `fastapi`,
  and `uvicorn`, and verified the pinned `mp.solutions.face_mesh.FaceMesh` path.
- implementation scope: added `itrace.calibration`, exported it from the public
  API, wired robust gaze quality into `SessionReport.quality`, wired
  `merge_gap_s` through fixed/adaptive I-VT, and wired `PupilConfig` threshold,
  blink padding, cutoff, and order into `analyze_pupil`.

## Verification (iteration 6 — live HTML orchestrator)

All evidence below is from real commands run on 2026-05-30 in
`working/iTrace`, refreshed by iteration 7 where the gate metrics changed.

- pytest/coverage: passed the promotion coverage floor in that iteration; exact
  historical gate numerics are superseded by `docs/verification_metrics.json`.
- ruff format: `84 files already formatted` from
  `uv run ruff format --check src/ tests/ scripts/`.
- ruff check: `All checks passed!` from `uv run ruff check src/ tests/ scripts/`.
- mypy: `Success: no issues found in 39 source files` from
  `uv run mypy src/itrace`.
- lockfile: `Resolved 105 packages in 4ms` from `uv lock --check`.
- web tests: `4 passed in 0.40s` from
  `uv run --extra web pytest tests/test_live_html.py`.
- optional imports: `uv run --extra capture --extra dashboard --extra figures --extra web`
  imported the then-declared runtime extras and verified `mp.solutions.face_mesh`
  exists under the pinned MediaPipe range. Current runtime extra smoke excludes
  dev-only Starlette/FastAPI TestClient clients such as `httpx2`.
- camera access: `uv run --extra capture --extra web itrace camera-probe --camera 0 --frames 30`
  opened the camera and completed; that particular probe saw 0 face samples, an
  environmental frame-content condition. The browser smoke immediately after
  server start did detect a face and rendered a real eye crop.
- live HTML smoke: `uv run --extra all itrace live-html --camera 0 --host 127.0.0.1 --port 8765 --output-dir output/live-html`
  served `/`, `/static/app.css`, `/static/app.js`, `/api/status`, and the live
  WebSocket; the no-op `/sw.js` returned 200. `chrome-devtools-axi` opened the
  page with no console errors or failed local requests; after pressing Start the
  page rendered a base64 JPEG eye crop, 8 samples, gaze/pupil diagnostics, and
  event counters. Pressing Export wrote `live_gaze.csv`, `live_pupil.csv`,
  `live_capture_records.csv`, and `live_report.json` under `output/live-html`.
- render: `scripts/generate_figures.py`, `scripts/generate_loop_animation.py`,
  `scripts/generate_orbs_animation.py`, `scripts/generate_power_figure.py`,
  `itrace figures --out-dir output/figures --animations`, and Pandoc
  Markdown/text/LaTeX/PDF manuscript builds all completed with regenerated
  artifacts.

## Verification (iteration 2 — 3D loop + improvements)

- gates: passed in that iteration with ruff, format, and mypy clean; exact
  historical gate numerics are superseded by `docs/verification_metrics.json`.
- closed loop (real run): `gaze_rms=0.163 n_sacc=3 pupil_r=1.00` (`scene.closed_loop()`); single-eye recovery `true_yaw=15 → recovered_x=15.287`; pupil ratios `[0.1667, 0.3333, 0.5, 0.6667]` (monotone).
- animation artifacts: `output/figures/closed_loop.gif` + `closed_loop_summary.png` written by `scripts/generate_loop_animation.py` (visually confirmed: both eyes + recovered gaze with saccade shading + pupil trace).
- historical doc-accuracy audit: package version, gate metrics, figure links, and citation keys were cross-checked against the then-current repository state. Current release metrics now live in `docs/verification_metrics.json` so public prose does not preserve stale literal counts.
- Advisor (iter-2) ran (`Inference.ts --mode advisor`): flagged "validates geometry" as overclaim → reworded to internal-consistency verification + lower-bound framing + assumption ledger + pupil-r reclassification (ISC-78).
- **DEFERRED-VERIFY (cross-vendor, Rule 2a):** Forge/Cato re-probed this session → codex still usage-limited (`try again at May 30th, 2026 1:48 PM`). `FOLLOWUP-XVENDOR` stands.

## Verification

All evidence below is a verbatim token from a real command run this session
(R1 artifact-backing; the suite IS the generator, R8).

**Gates (ISC-52, and the antecedent for the whole build):**
- pytest: `84 passed in 1.91s` — quoted from `uv run pytest`.
- coverage: passed the configured coverage floor; exact historical gate
  numerics are superseded by `docs/verification_metrics.json`.
- ruff: `All checks passed!` (`uv run ruff check src/ tests/ scripts/`).
- ruff format: `34 files already formatted` (`ruff format --check`).
- mypy: `Success: no issues found in 16 source files` (`uv run mypy src/itrace`).

**Build / import (ISC-1..6):**
- `uv sync --extra dev` resolved (`+ pytest==9.0.3 ... + ruff==0.15.15 ...`).
- `uv run python -c "import itrace"` → `itrace 0.1.0 imported OK` (no optional deps).
- ISC-6: `test_no_optional_deps_imported_at_module_load` + `test_iris_landmarks_no_mediapipe_imported` pass — `cv2`/`mediapipe`/`streamlit` absent from `sys.modules` after import.

**Ground-truth recovery (ISC-18..30):** `uv run itrace demo --amplitude 12` →
`saccade: amplitude=12.00deg (truth 12.00) direction=-0.0deg peak_vel=432deg/s`
and `fixations=2 saccades=1`, `scanpath='R'`. Test suite asserts amplitude
within 5%, peak velocity within 10%, microsaccade recovery, median-based
threshold, and no-saccades-on-noise — all in the `84 passed` run.

**Cross-implementation oracle (ISC-27):** `test_agreement_with_independent_reference`
passes — core I-VT/microsaccade detectors agree with `reference_impl.py` on
event count and onset (<0.02 s) despite different velocity math.

**Pupillometry (ISC-31..39):** demo →
`pupil: {'mean_size': 2.999..., 'n_peaks': 5.0, 'n_troughs': 5.0, 'n_blinks': 1.0}`;
`test_detector_is_causal` proves online==offline-prefix (no future sample used).

**CLI / IO / report (ISC-46..49):** `uv run itrace analyze ... --out output/report.json`
→ `wrote output/report.json (1 saccades, 2 fixations)`; report.json parses with
`"n_saccades": 1`, `"pupil": {... "n_blinks": 1.0}`.

**Figures (ISC-51):** `uv run python scripts/generate_figures.py` wrote
`output/figures/main_sequence.png` (129096 bytes) and `direction_polar.png`
(257863 bytes) — real 300-dpi PNGs.

**DEFERRED-VERIFY (cross-vendor audit, Rule 2a):** Forge/Codex (GPT-5.4) hit a
hard usage limit (`You've hit your usage limit ... try again at May 30th, 2026
1:48 PM`), which also blocks Cato (same codex backend). The *cross-vendor*
audit is therefore `[DEFERRED-VERIFY]` with follow-up `FOLLOWUP-XVENDOR: re-run
Forge + Cato read-only audit after 2026-05-30 13:48`. The in-tree independent
oracle (ISC-27) stands in for numerical cross-check; the PAI Advisor
(`Inference.ts --mode advisor`, non-codex) was run as the E5 commitment-boundary
gate. Per the Rule 2a truth-table, this no-signal cross-vendor result is
surfaced to the user, never laundered as pass.
