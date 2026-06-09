# Abstract {#sec:abstract}

Webcam eye tracking can make gaze, saccade, and pupil analysis broadly
accessible, but commodity-camera software often entangles three concerns that
need different evidence: camera capture, scientific signal processing, and
claims about real-eye accuracy. We present **iTrace**, a Python toolkit that
separates those concerns into (i) a pure NumPy/SciPy core for gaze geometry,
Savitzky-Golay velocity, I-VT and I-DT event identification,
Engbert-Kliegl microsaccade detection, main-sequence fitting, causal
pupil-phase detection, pupil preprocessing, scanpath statistics, and benchmark
scoring; (ii) an optional webcam/MediaPipe capture shell; and (iii) bounded
diagnostic/export interfaces for live sessions and user-supplied truth files.
The core never imports a hardware dependency, so its algorithms are exercised
headlessly against constructed ground truth rather than camera availability.

The resulting evidence is algorithmic, explicit, and reproducible. On synthetic
traces, the I-VT detector recovers a {{DEMO_AMPLITUDE}} deg saccade's amplitude
to within 5% and peak velocity to within 10%, and an independently formulated
reference implementation cross-checks event boundaries. A 3-D eyeball forward
model projects known gaze through a pinhole camera to landmarks and back through
the estimator, recovering gaze to 0.16 deg RMS; this exercises the geometry path
without claiming real-device accuracy. A seeded Monte-Carlo landmark-noise sweep
then shows the expected ordering of fragility: gaze remains below a 2 deg RMS
bound to sigma approximately 0.005 (about 3 px at 640 px width), while
velocity-thresholded saccade recovery falls below F1 = 0.8 near sigma 0.0014
(about 0.9 px). The pupil channel is reported conditionally because its
noise-sweep robustness follows from the modelling assumption used to inject
pupil noise. iTrace also ships a typer CLI, Streamlit dashboard, local
HTML/WebSocket orchestrator, guided derived-record empirical workflow, and
publication-figure scripts, all under {{TEST_COUNT}} tests at {{COVERAGE_PCT}}
coverage. Its contribution is not a claim that webcams are validated eye
trackers; it is a verification-first reference implementation whose evidence
boundaries are visible in the code, figures, reports, and manuscript. iTrace is
MIT-licensed and openly released at
https://github.com/docxology/itrace, with this version archived at DOI
10.5281/zenodo.20614909 (concept DOI 10.5281/zenodo.20614908 resolves to the
latest version).
