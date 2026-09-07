# Results: scanpath comparison and temporal dynamics {#sec:scanresults}

The descriptive scanpath metrics of [@sec:diststats] and the spatial/timeline
visualisations of [@sec:gallery] turn a single recording into a comparable
summary of *where* and *when* gaze moved. As everywhere in this section, the
inputs are synthetic sessions with known ground truth: the numbers below are
internal-consistency checks on the metrics and detectors, not measurements of a
real eye ([@sec:limitations]).

## Self-similarity is a tautology, used as a control

Comparing a synthetic session against an exact copy of itself returns a
similarity of 1.0, and comparing it against a deterministically time-shifted copy
returns the same value once the paths are aligned. This is true *by
construction* — identical fixation centroids and saccade sequences cannot differ
— so it is reported not as a finding but as the degenerate anchor every
similarity measure must hit. Its value is as a negative control: a measure that
failed to return 1.0 on a session against itself would be broken, and perturbing
one fixation by a known offset drops the similarity below 1.0 in the expected
direction, confirming the measure responds to real differences rather than
always saturating.

## Adaptive and fixed thresholds recover the same scripted events

The scripted multi-saccade sessions are built so that fixed-threshold I-VT
[@salvucci2000identifying] and a data-adaptive threshold derived from the
velocity distribution [@nystrom2010adaptive] should classify the same intervals.
They do: on the synthetic grid both recover the same saccade count with onsets
agreeing to within a sample, and the adaptive threshold settles inside the band
of fixed thresholds that bracket the clean velocity gap. Where the signal carries
injected noise the two diverge first on the smallest events, the same fragility
that the saccade-F1 collapse of [@sec:noiseresults] makes quantitative. The point
is a cross-check between two independent detector designs on a shared,
known-truth signal — the kind of agreement an event-classification reference
implementation is meant to provide [@dar2021remodnav] — not a claim that either
threshold is correct for real webcam data.

## Windowed event rates expose temporal structure

Sliding a fixed-width window across a session and counting events per window
yields a saccade-rate and fixation-rate time series whose envelope tracks the
scripted event schedule: rate peaks line up with the constructed saccade bursts
and fall to zero across the long fixations, and integrating the windowed counts
recovers the total event counts of the whole-session report. The microsaccade
rate over a fixational-jitter session stays near zero except around the single
injected microsaccade, the temporal counterpart to the amplitude separation seen
in [@sec:results]. These windowed rates are descriptive summaries of the detected
event stream; on a synthetic session they recover the schedule we wrote in, and
on a real recording they would inherit exactly the detector fragilities catalogued
in [@sec:noiseresults], with no additional accuracy implied.

Taken together, the spatial spread metrics (dispersion, convex-hull area, BCEA,
and the two scanpath entropies of [@sec:diststats]) and the timeline views give a
consistent, reproducible picture of the synthetic sessions: structured scanpaths
read as low direction-transition entropy and a compact hull, diffuse ones as the
opposite, and the temporal rates localise each event in time. Every number is a
property of the detected events on a known-truth signal — a verification that the
metrics and detectors behave as specified, kept strictly separate from any claim
about real-eye behaviour.
