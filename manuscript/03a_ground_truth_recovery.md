# Results: ground-truth recovery {#sec:results}

The first question for any detector is whether it recovers events it could not
have peeked at. Every trace in this section is synthetic: a generator builds the
signal *and* returns the parameters used to build it, so the recovered numbers
are checked against truth held out by construction. None of these results speak
to real-eye accuracy, which remains the device-validation gap of
[@sec:limitations].

On a synthetic fixation→saccade→fixation trace at \SI{250}{\hertz}, the I-VT
detector [@salvucci2000identifying] recovers exactly one saccade bracketed by two
fixations; the recovered amplitude matches the constructed {{DEMO_AMPLITUDE}}° to
within 5% and peak velocity to within 10% across the test grid. The min-jerk
generator sweeps amplitude and direction, and the recovered direction tracks the
constructed angle around the full circle (the gaze convention of 0° right, +90°
up), so a sign flip or axis swap on either path would surface immediately. A
purely-noisy fixation trace yields zero saccades at the default threshold — the
anti-criterion confirming the detector does not hallucinate events from noise.

The Engbert–Kliegl detector [@engbert2003microsaccades] recovers an embedded 0.5°
microsaccade from fixational jitter while ignoring the surrounding noise; a probe
confirms the threshold uses the median-based robust estimator (it returns a value
below half the plain standard deviation in the presence of velocity outliers),
the property checked directly in [@sec:events]. Recovered microsaccade amplitude
and peak velocity fall on a tight, low-amplitude main-sequence cluster, distinct
from the large-saccade branch — the expected separation of the two scales rather
than a fitted claim. The main-sequence fit recovers the saturating $V_{\max}$ and
$C$ of a synthetic relationship [@bahill1975main] to within 10% and returns a
power-law exponent inside the physiological 0.4–0.9 range ([@fig:mainseq]).

![Main-sequence recovery from a synthetic multi-amplitude recording. Panel A plots detected saccades by amplitude, peak velocity, and direction, with the fitted saturating curve and recovered parameters. Panel B repeats the same detections on log-log axes with the fitted power law, making the exponent and goodness-of-fit explicit.](../output/figures/main_sequence.png){#fig:mainseq width=100%}
