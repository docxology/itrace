# Methods: advanced detection and scanpath comparison {#sec:advdetect}

The detectors of [@sec:events] fix one threshold for a whole recording and reduce
a scanpath to summary scalars. Two further layers relax those choices: an
`itrace.detection` module that adapts its thresholds to the data and reports
movement dynamics the fixed detectors discard, and two `itrace.stats` modules
that compare and resolve scanpaths over time. Like the rest of the core they are
pure NumPy/SciPy, take explicit seeds wherever they resample, and operate on
whatever gaze they are given — synthetic streams with known ground truth
([@sec:forward]) or recorded ones. Nothing here measures real-eye accuracy; that
remains the device-validation gap of [@sec:limitations].

## Adaptive detection and movement dynamics

The fixed-threshold I-VT of [@sec:events] requires the analyst to pick a velocity
cut that suits the recording's noise floor. `detection.detect_ivt_adaptive`
instead follows the data-driven scheme of Nyström & Holmqvist
[@nystrom2010adaptive]: it seeds a peak-velocity threshold, iteratively
re-estimates it from the mean and standard deviation of the samples that fall
*below* the current threshold (the putative fixation/noise population), and stops
when the estimate converges. The converged value, not a hand-set constant, then
classifies saccades, so the same call adapts to a clean high-rate trace and a
noisy webcam one without re-tuning. The converged threshold is also surfaced in
`SessionReport.quality` when the configurable pipeline is run with
`DetectionConfig(method="adaptive_ivt")`, so a downstream report can state which
threshold actually certified the detected events.

The same configurable route also exposes a conservative merge-gap repair. When
`merge_gap_s` is positive, I-VT bridges only subthreshold gaps whose duration is
no longer than that value and whose run is bounded on both sides by saccadic
samples. This is deliberately narrower than smoothing the whole velocity trace:
it repairs one known failure mode — short capture dropouts or plateaus splitting
one saccade — while preserving event boundaries elsewhere and making the chosen
repair window visible in the report configuration.

Fast eye movements are followed by **post-saccadic oscillations** (PSOs) — the
brief wobble of the eye as it settles — which a plain I-VT either swallows into
the saccade or mislabels as a tiny second saccade. `detection.detect_pso` scans
the samples immediately after each detected saccade offset for a short,
lower-amplitude velocity excursion in the settling window and tags it as a PSO
rather than a movement, in the spirit of the glissade/PSO class that REMoDNaV
adds to the saccade/fixation dichotomy [@dar2021remodnav]. iTrace does not yet
claim REMoDNaV's full four-class smooth-pursuit classification — that is named as
future work in [@sec:limitations] — so the PSO tag is reported as a labelled
sub-event of the saccade it follows, not as an independent event class.

Two further quantities describe the *dynamics* the fixed detectors leave on the
floor. `detection.intersaccadic_intervals` returns the gaps between successive
saccade onsets (seconds), the inter-saccadic-interval distribution used
throughout the eye-movement literature to characterise scanning tempo and
fixational dwell [@holmqvist2011eye]; an empty or single-saccade input yields an
empty array rather than raising. `detection.saccade_peak_accelerations`
differentiates the velocity signal a second time and reports the peak
acceleration (deg/s²) within each saccade interval, the second-order companion to
the peak-velocity main sequence of [@sec:dynamics]. Both reuse the velocity
estimators of
[@sec:events] rather than recomputing them, so a recording's acceleration profile
is consistent with the velocities its saccades were detected from.

## Scanpath similarity

Comparing two scanpaths — a recording against a template, or two sessions of the
same task — needs a distance, not a scalar summary. `itrace.stats.similarity`
provides three complementary ones over the cardinal direction strings of
[@sec:dynamics], demonstrated on recovered scanpaths in [@sec:scanresults].

The **Levenshtein scanpath distance** is the minimum number of single-character
insertions, deletions, and substitutions that turn one encoded direction string
into another, computed by the standard dynamic-programming edit-distance recurrence
over the two strings; it captures how much one scan *order* must be rewritten to
match the other and is robust to small local insertions a fixed alignment would
punish. We report both the raw edit count and a length-normalised version
(distance divided by the longer string's length) so paths of unequal length are
comparable, and an empty-versus-empty comparison returns distance zero.

The **n-gram cosine similarity** lifts the comparison from order to habitual
sub-patterns. Each scanpath is turned into a vector of n-gram counts over its
direction string — reusing `encoding.ngram_counts` — and the two vectors are
compared by cosine similarity over their shared vocabulary, giving a value in
$[0, 1]$ that is high when two recordings share the same characteristic short
sequences regardless of where they occur. Either vector being empty (a recording
with fewer than $n$ saccades) yields a defined similarity of zero rather than a
division by zero.

The **transition matrix** is the raw material both similarities are built to
summarise: the first-order count (or row-normalised probability) of moving from
each direction symbol to each next one, the same construction that underlies the
direction-transition entropy of [@sec:diststats] and the generative,
information-theoretic view of scanpaths [@boccignone2011generative]. Exposing the
matrix directly lets an analyst inspect *which* transitions drive a similarity or
entropy value rather than trusting the scalar, and the row-normalised form is
guaranteed to be a proper stochastic matrix or an all-zero row for an unobserved
symbol.

## Windowed event-rate time series

A single recording is rarely stationary: scanning tempo, fixation stability, and
blink rate drift over a task. `itrace.stats.timeseries` slices a recording into
fixed-width time windows (with an optional overlap) and reports, per window, the
fixation rate, the saccade rate, and the blink rate in events per second, each
built from the typed events the core already detected and the blink intervals of
the pupil channel. The result is an aligned set of arrays — window-centre times
and the three rate series — suitable for plotting a session's evolution or for
feeding the spatial-stability metrics of [@sec:diststats] (gaze dispersion, BCEA,
used as a low-vision fixation-stability index [@castet2012quantifying]) a
window at a time. Windows that contain no samples report a rate of zero rather
than `nan`, and a window narrower than the sampling interval is rejected with an
actionable error, so a mis-specified window size fails loudly instead of
producing an empty or misleading series.
