# Methods: the capture shell {#sec:capture}

The capture shell is the sole hardware-facing module, and it is deliberately
thin. Its testable heart, `iris_landmarks_to_sample`, converts one frame of
normalised MediaPipe Face Mesh landmarks — supplied as a plain float array — into
a binocular gaze sample, with no MediaPipe import anywhere in the call.
MediaPipe Iris is cited here as a commodity-camera landmark source, not as a
guarantee of calibrated eye-tracker accuracy [@vakunov2020mediapipeiris].
Because the function consumes plain floats, the entire landmark→gaze mathematics
is unit-tested headless, and the same function is the bridge the forward-model
loop ([@sec:forward]) exercises.

`WebcamSource` wraps the genuinely device-bound work: opening the camera with
OpenCV, running MediaPipe inference per frame, and yielding timestamped capture
samples with gaze, a relative pupil/iris proxy, frame index, an FPS estimate, and
quality flags. The live-frame path preserves that existing sample API while also
returning the frame dimensions, a clamped pixel-space eye-region box derived
from the Face Mesh eye and iris landmarks, and a JPEG data URI containing the
zoomed eye crop. The proxy is explicitly labelled relative: MediaPipe Face Mesh
does not expose a calibrated pupil boundary or millimetre diameter, so the live
path records a camera-dependent size signal rather than a physical pupil measure.
`itrace record` writes the detected gaze samples to CSV, can write the relative
pupil proxy to a second CSV, and can write the full capture-record table
(`frame_index`, monotonic timestamp, gaze, pupil proxy, FPS estimate, and quality
flags) via `--records-out`. `itrace camera-probe` exercises dependency and camera
access without turning hardware availability into a scientific validation claim.
Both hardware commands suppress noisy native OpenCV/MediaPipe stderr diagnostics
by default while retaining `--backend-logs` for debugging.
Constructing it without the `capture` extra installed raises a clear, actionable
error naming the missing extra, rather than a bare `ModuleNotFoundError` from
deep in the stack. Only the few lines that actually grab frames are excluded from
coverage; the dependency-absent error path, landmark conversion, capture-sample
shape, split gaze/pupil CSV writers, and full capture-record CSV writer are all
tested headless.

The separate `itrace live-html` command adds a local single-user FastAPI
orchestrator around the same verified Python path. The browser page receives
WebSocket messages containing the eye crop, the latest capture sample, rolling
`pipeline.analyze_session` summaries, event records, and plot-ready series; the
browser itself only draws controls and native Canvas/SVG diagnostics. It is
configured as the `web` extra and remains import-safe: `fastapi`, `uvicorn`,
OpenCV, and MediaPipe are imported only inside server or capture entry points.
Persistence is explicit: without `--output-dir` the app is in-memory, and with
`--output-dir` it writes CSV/JSON artifacts only when export is requested.
