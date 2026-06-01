# Abstract {#sec:abstract}

Webcam-based eye tracking promises low-cost access to gaze, saccade, and pupil
signals, but the open-source landscape is fragmented across browser gaze
trackers, appearance models, event classifiers, pupillometry packages, and
capture shells whose hardware dependencies make reproducible, headless
validation difficult. We present **iTrace**, a Python toolkit that separates a
pure NumPy/SciPy analysis core — gaze geometry,
Savitzky–Golay velocity, I-VT and I-DT event identification
[@salvucci2000identifying], Engbert–Kliegl microsaccade detection
[@engbert2003microsaccades], saturating and power-law main-sequence fitting
[@bahill1975main], causal real-time pupil-phase detection
[@kronemer2024rtpupilphase], and scanpath direction encoding — from an
optional, thin webcam/MediaPipe capture shell. Because the core never imports a
hardware dependency, the entire test suite runs on a headless machine and
algorithmically verifies each detector against synthetic signals whose ground
truth is known by construction: the I-VT detector recovers a
{{DEMO_AMPLITUDE}}° saccade's
amplitude to within 5% and its peak velocity to within 10%, and a second,
independently-formulated implementation cross-checks the event boundaries. A
3-D eyeball forward model closes the loop — projecting a known gaze through a
pinhole camera to landmarks and back through the estimator — recovering gaze to
an RMS residual of 0.16°, an internal-consistency check that exercises the
geometry path (not just the event algorithms) and bounds, but does not
establish, real-world error. A Monte-Carlo noise-sensitivity sweep then varies
an idealized landmark-localisation noise σ and quantifies how each signal
degrades: gaze stays usable to σ≈0.005 (≈3 px at 640 px width) while saccade
detection — resting on the noise-amplifying velocity — collapses first at
σ≈0.0014 (≈0.9 px), implying that webcam saccade timing is theoretically
marginal before real MediaPipe, lens, illumination, head-pose, and calibration
errors are added. (The pupil
channel's robustness is set by a modelling assumption, not measured, and is
reported as conditional.) The
package ships with a typer command-line interface, a Streamlit dashboard, a
local HTML/WebSocket live orchestrator, and publication-figure scripts, all
under a green test suite with {{TEST_COUNT}} tests at {{COVERAGE_PCT}} coverage.
iTrace demonstrates that webcam eye-movement software becomes more auditable
when the verifiable algorithms are decoupled from fragile hardware.
