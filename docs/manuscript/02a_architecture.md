# Methods: architecture {#sec:methods}

iTrace is organised as two strata with a one-way dependency: a pure analysis
**core** and an optional hardware **shell**. The split is not cosmetic — it is the
mechanism that makes the scientific claims of this paper checkable.

The **core** depends only on NumPy, SciPy, and pandas. It divides into a
signal-and-event layer (`geometry`, `calibration`, `velocity`, `saccades`,
`mainsequence`, `pupil`, `pupilseg`, `pupilphase`, `encoding`, `eyemodel`,
`scene`, `power`, `benchmark`, `experiments`, `pipeline`, `io`) and a
quantitative `stats` subpackage layered on top of the detected events
(`stats.descriptive`, `stats.distributions`, `stats.scanpath_metrics`,
`stats.similarity`, `stats.timeseries`; [@sec:diststats], [@sec:advdetect]). Every
value the core consumes or produces is a plain array or a typed dataclass; it has
no knowledge of cameras, files beyond CSV, or rendering. The unit and coordinate
conventions are fixed at the boundary — pixels with a top-left origin, degrees of
visual angle with a mathematical-convention direction (`0deg` right, `+90deg`
up), seconds for time, and a pupil unit that travels with every pupil number — so
no ambiguous quantity ever leaks between layers.

Three recent additions keep that contract explicit at the places where a reader
might otherwise overinterpret output. `pupilseg` accepts a supplied eye crop and
returns a dark-component segmentation in image pixels, or a pupil/iris-relative
sample if the caller provides an iris-radius normalizer; it does not infer
millimetres and never claims absolute pupil diameter. `benchmark` accepts
user-supplied truth and comparator event files and reports interval-overlap
recovery, timing error, and amplitude error; every payload records the truth
boundary instead of implying that detector agreement is a reference measurement.
`experiments` adds the analogous file-replayable shape for prompted live
sessions: it defines trial schedules, slices derived capture samples, and reports
session quality plus held-out target residuals without storing raw video or
treating screen prompts as a reference tracker. These modules are additive
interfaces around the same core streams, not special cases that bypass the typed
unit contract.

Calibration is kept in that same core layer because it is an analysis transform,
not a camera driver. The shipped model is deliberately inspectable — a fitted
two-dimensional affine map from raw gaze coordinates to known target coordinates,
with RMS, percentile, and maximum error summaries. That makes screen-space
calibration auditable when target points exist, while leaving live webcam output
honestly labelled as relative/algorithmic when no external target or reference
device is present.

The **shell** (`capture`, `live`, `dashboard`, `cli`, `viz`) is the only code
that touches OpenCV, MediaPipe, Streamlit, Plotly, or matplotlib, and it imports those
dependencies **lazily, inside functions**, never at module load; a missing
optional dependency raises an actionable error rather than breaking `import
itrace`. Two consequences follow, and they are the load-bearing design decisions
of the whole package:

1. `import itrace` and the *entire* test suite succeed on a headless machine with
   none of the optional dependencies installed.
2. The scientific algorithms can therefore be verified continuously, in CI,
   without a webcam — which is precisely what makes their correctness auditable
   ([@sec:reproducibility]).

The `stats`, viz, and method layers obey a second, finer rule that keeps the
headless guarantee honest: matplotlib is treated like any other optional
plotting backend and lives **only** in the visualisation modules, imported at the
top of those modules with the `Agg` backend selected before `pyplot`. A method or
statistics module never imports a plotting library, so a fitted distribution, a
scanpath-similarity matrix, or a windowed event-rate series is computed and
returned as arrays regardless of whether any display is available.

The dependency rule is strict and one-directional: the shell may import the core,
the core may never import the shell. A new capture backend (browser WebRTC, a
deep-learning gaze model, a research IR camera) is added by writing a shell
([@sec:capture]) that emits the same `GazeStream` / `PupilStream` /
capture-record tables the core already analyses. The verified core never changes,
every analysis in [@sec:events] through [@sec:advdetect] applies unaltered to its
output, and the visualization/report layer can be regenerated from the same
stored payloads. The full module map and dependency contract are in
[@sec:software].
