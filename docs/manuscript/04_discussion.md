# Discussion {#sec:discussion}

## What the decoupling buys

By implementing established algorithms in a hardware-free core and verifying
them against constructed ground truth, iTrace makes the scientific layer
continuously testable and reproducible. A regression in the saccade detector, the
microsaccade threshold, or the blink interpolation is caught by a headless test
run, not discovered later on real data. The optional capture shell can then evolve
— new landmark backends, browser capture, deep-learning gaze — without putting the
verified core at risk, because the shell only produces the gaze and pupil streams
the core already analyses. The local HTML orchestrator is an example of this
discipline: the browser receives eye crops and plot-ready summaries, but all
capture, estimation, event detection, statistics, and export remain in the same
Python package path tested elsewhere. This is the central design message of
[@fig:graphical-abstract]: the camera path is an input shell, while the evidence
path is a reproducible sequence of typed records, tested algorithms, and rendered
reports.

## What the expanded analysis layer adds

The same decoupling extends past detection into description. On top of the
estimators sits a quantitative layer (`itrace.stats`) and a visualization layer
(`itrace.viz`), both pure-core in spirit: the statistics modules import only
NumPy/SciPy, and matplotlib is confined to the optional `figures` extra so the
core stays headless. The statistics layer turns the typed events of
[@sec:events] and [@sec:dynamics] into descriptive summaries of fixation and
saccade quantities, maximum-likelihood fits of positive-support distributions
(gamma, log-normal, ex-Gaussian) ranked by AIC/BIC with a Kolmogorov–Smirnov
distance, scanpath spread and organisation metrics (gaze dispersion,
convex-hull area, BCEA, and spatial / direction-transition entropy), and a
bootstrap-percentile confidence interval that attaches sampling uncertainty to
the main-sequence exponent ([@sec:diststats]). The visualization layer renders
the standard velocity, scanpath, distribution, main-sequence, microsaccade, and
pupil figures — and a composite session dashboard — deterministically and
byte-for-byte from a `SessionReport` ([@sec:gallery]).

The honest framing matters here too. Every one of these methods *describes* a
scanpath it is handed; none of them measures real-eye accuracy or repairs the
device validation gap of [@sec:limitations]. A tighter bootstrap interval, a
lower KS distance, or a smaller BCEA is a property of the *detected events*, not
evidence that those events match a real eye. What the layer genuinely adds is
reach and reproducibility: the same descriptive vocabulary other eye-movement
packages expose, computed by tested, seed-pinned functions over the verified
core, applicable unchanged to a synthetic stream with known ground truth, a real
recording pushed through the pipeline, or a user-supplied benchmark file. The
new `pupilseg` and `benchmark` modules are deliberately bounded examples of the
same principle: the former turns a supplied eye crop into pixels or relative
units without claiming millimetres; the latter compares events against supplied
truth without claiming that another detector is truth. The most useful
comparison point is therefore not "is iTrace more accurate than WebGazer,
L2CS-Net, REMoDNaV, pymovements, PupEyes, pypillometry, ElSe, PuRe, or
PupilEXT?" but "does iTrace make the exact algorithmic contract clear enough
that those systems can be benchmarked against it on shared fixtures or public
datasets?" [@papoutsaki2016webgazer; @hempel2022l2cs; @dar2021remodnav;
@krause2023pymovements; @zhang2026pupeyes; @mittner2020pypillometry;
@fuhl2015else; @santini2018pure; @zandi2021pupilext]. The guided empirical
workflow sits in the same category: it makes a local session auditable by writing
derived records and a summary report, while leaving the stronger reference-device
study for future data.

The N=1 pilot in [@sec:empirical-pilot] is useful precisely because it sits
between the two evidence types. It does not make the real-world claims that only
a reference-device study could support, but it prevents the synthetic validation
from floating without an empirical scale. The synthetic sections answer whether
the code recovers known signals and how idealised landmark noise degrades those
signals; the pilot shows what one ordinary live run looked like in sampling
rate, sampling regularity, drift, prompted-target residuals, and target
acquisition latency. Those numbers are therefore best read as context for
expectations and future benchmarks, not as population estimates. In practical
terms, [@tbl:empirical-synthetic-range] and
[@fig:synthetic-empirical-range-bridge] separate which synthetic defaults are
contextualized by a local demonstration from which remain deliberate stress
ranges: clean finite capture and webcam-rate timing look plausible for the
demonstration path, whereas dropout, drift, landmark-noise, and calibration-bias
ranges remain modelling levers that must be tested against future reference data.
The statistical ledger in [@fig:statistical-interpretation-ledger] applies the
same discipline to the rest of the results: relative model weights, residual
diagnostics, bootstrap intervals, spatial descriptors, and scanpath summaries are
reported as finite-record diagnostics with explicit non-claims.

The repeated-session ledger turns that point from prose into a collection rule.
For a first paper release, additional local replicates should be treated as
coverage of the declared single-participant/single-device operating scale:
different lighting, display, posture, and task conditions can reveal whether the
live workflow repeatedly produces finite streams, stable sampling, bounded
drift, and interpretable prompted-target residuals. They should not be used to
rename the study as device validation. The release gate is therefore deliberately
two-part: enough repeated sessions and conditions to support the within-setup
diagnostic claim, and at least one reference-device, public-dataset, or
manual-annotation lane before the manuscript can claim device-level agreement.
That separation keeps the v1 paper publishable as an algorithmic and workflow
paper while preventing a local webcam repeatability series from becoming an
unstated accuracy claim.

## Algorithmic verification, not device validation

A precise claim matters. What [@sec:results] establishes is *algorithmic
verification*: each estimator recovers the parameters of signals produced by a
known forward model. This is not *device validation* — agreement with an
independent reference measurement of the same physical quantity on real eyes. We
have not performed the latter. A passing synthetic suite proves the estimation
mathematics is correct and regression-protected; by itself it is compatible with
poor real-world performance. The hardware capture path is not exercised by the
suite, and it is exactly where the dominant error sources live: head-pose and
calibration confounds in gaze, the pupillary light reflex in pupillometry, and
saccade undersampling at about 30 Hz. Until a head-to-head against a
reference tracker or a public dataset exists, iTrace should be cited as a verified
reference implementation of the algorithms, not a validated tracker.

## The closed loop bounds, but does not establish, real error

The 3-D closed loop ([@sec:closedloop]) exercises the landmark→gaze projection,
but its guarantee is bounded by what the forward model and estimator *share*. Both
assume an ideal pinhole projection with no lens distortion, no corneal refraction
(the apparent pupil differs from the physical pupil by ~0.5–1 mm), a perfectly
spherical eyeball, and a zero kappa angle (~5° in real eyes). Any error these
shared idealizations introduce is identically zero in the loop by construction,
and the synthetic landmarks carry none of MediaPipe's real noise, bias, or
eccentric-gaze foreshortening. The 0.16° residual is thus a *lower bound* on
real-world error, and the $r = 1.0$ pupil result is a consistency check, not
accuracy. The quantitative limits this implies are collected in [@sec:limitations].
