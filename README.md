# iTrace — open-source webcam eye-movement analysis

A real, installable Python toolkit for the three webcam eye-tracking signals —
**gaze trajectory, saccade direction & dynamics, and pupil diameter** — built as
a pure NumPy/SciPy analysis core (algorithmically verified against synthetic
ground truth; see [Validation status](#validation-status--read-this-carefully))
with an optional, thin webcam + MediaPipe capture shell.

> The design principle: the hardware-fragile part (camera, MediaPipe) is a
> thin optional layer around a pure scientific kernel. The kernel is
> algorithmically verified against synthetic ground truth, so the scientific
> methods are regression-protected *and* the whole test suite runs headless with
> zero optional dependencies.

## What it computes

| Signal | Module | Algorithm (cited) |
|--------|--------|-------------------|
| Pixels → degrees of visual angle | `geometry` | `pix2deg`/`deg2pix`, iris-offset → eyeball rotation |
| Velocity | `velocity` | Savitzky–Golay (uniform), central-difference (variable dt) |
| Fixations / saccades | `saccades` | I-VT & I-DT (Salvucci & Goldberg, 2000) |
| Microsaccades | `saccades` | Engbert & Kliegl (2003), median-based threshold |
| Saccade dynamics | `mainsequence` | main sequence: saturating + power-law fit |
| Scanpath | `encoding` | direction-character + n-gram encoding |
| Pupil preprocessing | `pupil` | blink detect/interpolate, MAD reject, low-pass, baseline |
| Real-time pupil phase | `pupilphase` | causal dilation/constriction/peak/trough (rtPupilPhase-style) |
| Calibration + quality | `calibration` | affine screen-space calibration, RMS/p95 error, dropout/gap/jitter quality |
| **3-D eye forward model** | `eyemodel` | parameterised eyeball + pinhole projection → MediaPipe-shaped landmarks |
| **Animated closed loop** | `scene` | 3-D gaze trajectory → projection → estimation → recovery vs truth |
| **Noise power analysis** | `power` | Monte-Carlo recovery accuracy vs webcam noise, with 95% CIs |
| Configurable analysis | `config`, `pipeline` | fixed/adaptive I-VT, merge-gap repair, PSO, pupil/capture/figure configs |
| Synthetic sessions | `synthetic` | seeded gaze + saccade + PSO + blink + pupil truth |
| Synthetic validation | `validation` | within-domain recovery, cross-domain stability, live plausibility diagnostics |
| Live capture (optional) | `capture` | OpenCV + MediaPipe Face Mesh iris landmarks, relative pupil proxy |
| Local HTML orchestrator (optional) | `live` | FastAPI/WebSocket backend + native Canvas/SVG diagnostics |

## Install & run

```bash
uv sync --extra dev            # core + test toolchain
uv run pytest --cov=itrace     # green, no-mocks, ≥90% coverage
uv run itrace demo             # synthesize → analyse → print recovered events
```

Optional backends (not required for the core or the tests):

```bash
uv sync --extra capture        # opencv-python + MediaPipe Face Mesh backend
uv sync --extra dashboard      # streamlit + plotly (live dashboard)
uv sync --extra figures        # matplotlib (publication figures)
uv sync --extra web            # FastAPI + uvicorn local HTML orchestrator
uv sync --extra all            # capture + dashboard + figures + web
```

## CLI

```bash
itrace demo                            # end-to-end synthetic demo
itrace analyze gaze.csv --out report.json --pupil-csv pupil.csv
itrace analyze gaze.csv --config-json analysis.json --velocity-threshold 35
itrace stats gaze.csv --out stats.json       # stats + quality + event bootstrap CIs
itrace validate-recording gaze.csv --pupil-csv pupil.csv --out validation.json
itrace synthetic-validation --out synthetic_validation.json --repetitions 5
itrace calibrate points.csv --out calibration.json --apply-gaze raw_gaze.csv
itrace figures --out-dir output/figures --animations  # render gallery + GIF (figures extra)
itrace record --camera 0 --out gaze.csv --pupil-out pupil.csv --records-out capture.csv
itrace camera-probe --camera 0 --frames 5    # clean dependency/camera probe
itrace camera-probe --camera 0 --frames 5 --backend-logs  # include MediaPipe/OpenCV logs
itrace live-html --camera 0 --output-dir output/live-html  # local HTML app (web + capture)
itrace dashboard                              # needs the dashboard extra
```

## Live HTML orchestrator

`itrace live-html` runs a local, single-user FastAPI app whose browser page is
only a thin orchestrator. The Python backend still owns webcam capture,
MediaPipe landmark extraction, gaze/pupil estimation, event detection, rolling
statistics, and export. The first viewport centers a large zoomed JPEG crop of
the detected eye region with its bounding box, FPS, camera status, and face
status. The lower panels use native browser Canvas/SVG to inspect the rolling
gaze trace, x/y time series, speed with threshold and saccade spans, pupil
proxy, event raster, scanpath, direction polar plot, amplitude histogram, and
main-sequence scatter. A validation panel runs the Python synthetic-domain
suite from the browser, plots per-domain saccade-recovery F1 with bootstrap
intervals, and keeps live webcam diagnostics explicitly labelled as
plausibility checks rather than reference-device validation.

Run it locally:

```bash
uv run --extra capture --extra web itrace live-html --camera 0 --output-dir output/live-html
```

No files are written unless `--output-dir` is supplied and the browser export
button is pressed. Export writes gaze CSV, relative-pupil CSV, capture-record
CSV, a JSON analysis report, and, when live calibration has been fitted,
`calibration.json` plus `live_gaze_calibrated.csv`. Calibration uses a Python
`AffineCalibration` fit against the current target plane (default ±15° visual
angle); the browser only collects/requests the fit and displays raw vs calibrated
coordinates. The live webcam pupil value is a relative algorithmic proxy; the app
is a method-plumbing and diagnostic interface, not a reference-device validation.

## Config, validation, and calibration

`--config-json` accepts an `AnalysisConfig`-shaped JSON object. Values from the
file override defaults, and explicit CLI options override the file:

```json
{
  "detection": {
    "method": "adaptive_ivt",
    "velocity_threshold_deg_s": 35.0,
    "include_pso": true,
    "include_smooth_pursuit": true,
    "reject_edge_events": true
  },
  "pupil": {
    "blink_threshold": 0.02,
    "baseline_window_s": [0.0, 0.5],
    "response_window_s": [0.5, 2.0]
  }
}
```

`itrace validate-recording` runs the real analysis pipeline after safe short-gap
interpolation and writes recording quality, event plausibility, pupil validity,
calibration availability, report-structure validation, warnings, and errors.
`itrace synthetic-validation` runs seeded domains (`clean_desktop`,
`webcam_jitter`, `head_drift`, `low_light_dropout`) and reports within-domain
bootstrap summaries plus cross-domain stability gaps for saccade recovery,
amplitude/direction/peak-velocity error, gaze finite fraction, and pupil
validity. The same suite backs the live HTML validation panel. Live webcam
diagnostics report sampling regularity, finite gaze fraction, pupil-valid
fraction, path length, dispersion, and warnings, but they intentionally do not
compute F1 because no live reference truth exists.
`itrace calibrate` fits an affine mapping from `raw_x,raw_y,target_x,target_y`
CSV columns and can apply that mapping to an existing gaze CSV.

## Library

```python
import itrace
from itrace import pipeline, synthetic
from itrace.calibration import AffineCalibration, calibration_error
from itrace.config import AnalysisConfig, DetectionConfig, PupilConfig

gaze, truth = synthetic.gaze_with_saccade(amplitude_deg=10.0, direction_deg=0.0)
pupil, _ = synthetic.pupil_sine_with_blink()
report = pipeline.analyze_session(gaze, pupil)

print(len(report.saccades), report.saccades[0].amplitude_deg)  # 1, ~10.0
print(report.scanpath)                                         # 'R'
print(report.pupil["n_blinks"])                                # 1.0

# Adaptive I-VT + PSO reporting without breaking the old fixed-threshold API.
cfg = AnalysisConfig(
    detection=DetectionConfig(method="adaptive_ivt", include_pso=True, merge_gap_s=0.012),
    pupil=PupilConfig(blink_threshold=0.02, smooth_cutoff_hz=4.0),
)
session_gaze, session_pupil, session_truth = synthetic.synthetic_session()
session_report = pipeline.analyze_session(session_gaze, session_pupil, config=cfg)
print(session_report.quality["detection_threshold_deg_s"])
print(session_report.quality["dropout_fraction"], session_report.quality["longest_gap_s"])
print(len(session_truth.saccades), len(session_report.psos))

# Optional affine calibration when you have known target points.
cal = AffineCalibration.fit(raw_x=[0, 1, 0], raw_y=[0, 0, 1], target_x=[1, 3, 1], target_y=[-1, -1, 1])
print(calibration_error(cal, [0, 1, 0], [0, 0, 1], [1, 3, 1], [-1, -1, 1])["rms_error_deg"])

# Full closed loop: 3-D eye animation → projection → estimation → recovery
from itrace import scene
result = scene.closed_loop()
print(result.metrics["gaze_rms_deg"])   # ~0.16  (recovered vs 3-D truth)
print(int(result.metrics["n_saccades"]))  # 3

from itrace import validation
suite = validation.synthetic_validation_suite(repetitions=3)
print(suite["cross_domain"]["macro_saccade_f1"])
```

## Accuracy & limits (read before trusting a number)

Consumer-webcam tracking is **not** IR-pupillometer-grade. Real-frame gaze error
depends on the dataset, camera, pose, lighting, model, and calibration protocol;
public webcam/mobile datasets such as GazeCapture and MPIIGaze and appearance
models such as L2CS-Net report their own protocol-specific error measures rather
than a universal webcam constant. Webcam pupil diameter likewise requires a
dedicated segmentation or learned-model path; iTrace's live value is a relative
proxy. These bounds constrain study design; iTrace surfaces them rather than
hiding them. See
[`docs/RESEARCH_BRIEF.md`](docs/RESEARCH_BRIEF.md) for the full ecosystem survey
that motivated the design.

## The 3-D forward model & closed loop

iTrace ships a 3-D eyeball forward model ([`eyemodel`](src/itrace/eyemodel.py))
and an animated closed loop ([`scene`](src/itrace/scene.py)) that exercise the
*whole* system end-to-end:

```
3-D eyeball (true yaw/pitch/pupil)
   → perspective pinhole projection → MediaPipe-shaped landmarks
   → capture.iris_landmarks_to_sample → GazeStream
   → pipeline (saccade detection + pupillometry)
   → recovered signals  ⟶ compared against the 3-D truth
```

Render it: `uv run python scripts/generate_loop_animation.py` writes
[`output/figures/closed_loop.gif`](output/figures/) (3-D eye + recovered gaze +
pupil) and a static `closed_loop_summary.png`.

The forward model (3-D sphere + perspective projection) and the estimator
(an arcsine sphere approximation) are **independent formulations**, so for the
discrepancy they *don't* share the loop is a real test, not a tautology —
recovered gaze RMS vs the 3-D truth is **0.16°** for gaze within ±15° and the
three scripted saccades are recovered exactly.

**What this is — and is not.** This is *internal-consistency / algorithmic
verification*: it proves the estimator correctly inverts the forward model and
catches sign flips, axis swaps, unit errors, and frame mismatches. It is **not**
device validation. The 0.16° is the residual between two of *our own* models, so
it is a **lower bound on real-world error** — any idealization the two paths
*share* contributes exactly zero error by construction and is invisible to the
loop:

| Shared idealization (loop is blind to it) | Real-world reality |
|---|---|
| Pinhole projection, no lens distortion | webcam lens distortion, foreshortening |
| No corneal refraction | apparent pupil shifts ~0.5–1 mm |
| Perfectly spherical eyeball | aspheric cornea |
| Zero kappa angle (optical = visual axis) | ~5° visual/optical-axis offset |
| Synthetic landmarks | real MediaPipe noise, bias, head-pose/occlusion failures |

The recovered **pupil** is a near-identity check, not a recovery: the projected
pupil/iris ratio is a monotone function of the true pupil radius, so r ≈ 1.0 is
expected by construction — treat it as a *consistency check* that deblinking and
the pipeline preserve the signal, not as evidence of pupil-size accuracy.

## Noise-sensitivity analysis

A **sensitivity / robustness sweep** (not a hypothesis-testing power calculation):
how does recovery degrade as observation noise grows? Webcam noise is modelled —
deliberately *idealized* — as i.i.d. Gaussian landmark-localisation jitter of
standard deviation σ (normalised image units); `power.run_noise_sweep()` injects
it and runs a seeded Monte-Carlo sweep (mean ± 95% bootstrap percentile CI,
25 trials/level):

```bash
uv run python scripts/generate_power_figure.py     # -> output/figures/noise_power.png
uv run python scripts/generate_orbs_animation.py   # -> output/figures/eye_orbs.gif (floating orbs)
```

| Signal | Usable until (σ, ≈) | ≈ px @640w | Status |
|--------|--------------------|-----------|--------|
| **Saccade detection (F1≥0.8)** | 0.0014 | **0.9 px** | **emergent** — rests on *velocity* (a derivative), which amplifies noise → degrades first |
| **Gaze direction (RMS<2°)** | 0.0048 | 3.1 px | **emergent** — direct position estimate, intermediate |
| **Pupil correlation (r≥0.9)** | 0.0050 | 3.2 px | **conditional** — robustness is *set by the `pupil_noise_scale` assumption*, not measured; real pupil segmentation is not implemented |

**The defensible headline** (translating σ to pixels): saccade detection breaks
down at σ ≈ 0.9 px, a sub-pixel-to-few-pixel perturbation scale small enough
that real capture effects from landmark jitter, head pose, illumination, lens
distortion, and calibration can dominate. So webcam saccade timing is
theoretically marginal **before** those real-world effects are added, whereas
gaze and coarse fixation tolerate a few pixels.
The gaze-vs-saccade ordering emerges from the noise model; the pupil position does
not — it would move if `pupil_noise_scale` changed, so it is reported as
conditional, not as a finding. Saccade F1 uses interval-overlap matching; with 25
seeds the σ crossings are approximate.

## Validation status — read this carefully

iTrace is **algorithmically verified against synthetic ground truth (now
including a 3-D forward-model closed loop that checks the estimator inverts the
forward model under shared idealizing assumptions). It is not yet validated
against real eye-tracking data or a reference eye-tracker.** These are different
claims and the distinction matters:

- **What is verified.** Each estimation algorithm recovers, within a stated
  tolerance, the parameters of signals generated by a known forward model — a
  minimum-jerk saccade of known amplitude/direction, an embedded microsaccade, a
  sinusoidal pupil trace with a NaN blink, and a full 3-D-eye → projection →
  recovery loop (internal-consistency gaze residual 0.16°, a lower bound).
  Independently-formulated implementations cross-check the detectors
  (`tests/fixtures/reference_impl.py`) and the geometry (`eyemodel`'s perspective
  projection vs the arcsine estimator). No mocks. 429 tests, 91.18% coverage.
- **What is NOT yet established.** No comparison against a reference eye-tracker
  or a public webcam gaze dataset (e.g. MPIIGaze, GazeCapture) on real frames;
  no human-subject data; the physical hardware path (`capture.WebcamSource`'s
  live OpenCV frame grab + MediaPipe landmark inference) is probed only as an
  environment-dependent smoke test, not as device validation. The 3-D loop checks the
  landmark→gaze *mathematics* on synthetic landmarks, not camera frames, and is
  blind to every idealization the forward model and estimator share (see the
  assumption table above) — so real-world error sources (lens distortion, corneal
  refraction, kappa angle, lighting, motion blur, MediaPipe jitter/bias, the
  pupillary light reflex, head-pose/calibration confounds) remain uncharacterised.

The synthetic suite (and the closed loop) prove the NumPy/SciPy mathematics is
correct; they do not, by themselves, establish that iTrace tracks real eyes
accurately. Treat iTrace today as a **well-tested reference implementation of the
estimation algorithms**, and add a reference-device or public-dataset comparison
before reporting device-level accuracy. The ideal state, the criteria, and the
verbatim verification evidence (including this disclosure) live in
[`ISA.md`](ISA.md).

Future public-dataset, reference-device, detector-benchmark, and release
protocol scaffolds are documented in
[`docs/VALIDATION_PROTOCOLS.md`](docs/VALIDATION_PROTOCOLS.md).

## Status

`working/` project (standalone, self-contained), v0.4.0. Gates: 429 tests green
at 91.18% coverage, `ruff` clean, `ruff format` clean, `mypy --strict` clean. MIT.
Upcoming minor, medium, and large improvements are tracked in
[`TODO.md`](TODO.md).
