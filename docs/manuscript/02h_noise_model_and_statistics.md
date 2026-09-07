# Methods: noise model and statistical design {#sec:noisemodel}

## Idealized noise model

To ask *how much noise the recovery tolerates*, we identify the quantity that
dominates corruption. Photon shot noise, JPEG compression, pixel quantisation,
and MediaPipe's finite precision all manifest downstream as one thing: **jitter
in the normalised image coordinates of the landmarks**. We model it as an
*idealized* additive **i.i.d. Gaussian** of standard deviation $\sigma$
(normalised image-coordinate units) on every landmark before estimation. This is
explicitly simplified: it ignores spatial correlation between adjacent landmarks,
temporal correlation, heteroscedasticity (error grows at eccentric gaze and under
poor lighting), and systematic bias — and it is *not* derived from sensor optics.
Systematic *biases* (corneal refraction, kappa angle, lens distortion) are
excluded entirely; they belong to the device validation gap ([@sec:limitations]).
Appearance-based gaze work on GazeCapture, MPIIGaze, and L2CS-Net makes the
same practical point from the opposite direction: real-frame accuracy depends on
dataset, camera, illumination, head pose, and calibration conditions, not just on
the downstream event detector [@krafka2016gazecapture; @zhang2017mpiigaze;
@hempel2022l2cs].
Pupil-size measurement, which a real pipeline derives from a separate segmentation
we do not implement, carries its own observation noise scaled with $\sigma$ as a
stated modelling assumption — so the pupil channel's noise behaviour is *assumed*,
not emergent. The model is chosen to be the simplest perturbation that is
falsifiable rather than the most realistic one: it is a lower bound on the kinds
of corruption a real webcam imposes, and the recovery curves it produces are best
read as best-case envelopes.

## Statistical design

The sweep is a **sensitivity / robustness** analysis, not a hypothesis-testing
power calculation. At each noise level we run $n$ independent, seeded closed-loop
recordings ([@sec:forward]) and measure three outcomes — gaze RMS error (deg),
saccade detection F1, and pupil correlation with truth. Per level we report the
mean and a 95% **bootstrap percentile** confidence interval across the $n$ trials.
We use a bootstrap rather than a symmetric Student-$t$ interval because two of the
metrics are bounded (F1 $\in[0,1]$, correlation $\in[-1,1]$): near 1.0 a symmetric
interval can run past the bound, whereas a percentile of resampled means cannot
leave the range of the in-bound observations. We then locate the noise level at
which each outcome crosses a usability bound by linear interpolation between grid
points. The interval describes variability of the mean across the random
landmark-noise realisations sampled by the $n$ seeded trials (a Monte Carlo
standard error over the noise process), not variability from re-running the
deterministic pipeline. Saccade F1 scores a detected event as a true positive when
its sample interval **overlaps** a ground-truth saccade interval (no separate
temporal tolerance); precision and recall are computed from the same overlap match
and combined as the harmonic mean, so a detector that fragments one true saccade
into several events or merges several into one is penalised rather than
rewarded. Every trial seed is deterministic, so the whole sweep — and therefore
every figure and table number — is reproducible. Full statistical detail is in
[@sec:statmethods]. The sweep is falsifiable: were recovery independent of noise,
gaze RMS would be flat across the grid.

The same percentile-bootstrap discipline is reused wherever the package attaches
uncertainty to a point estimate — the main-sequence exponent interval of
[@sec:diststats] and the event-rate summaries of [@sec:advdetect] — for the
identical reason: it assumes no symmetry and stays inside the range the data
support. None of these intervals describe a population of eyes or sessions the
recording was not designed to sample; they describe sampling variability of the
statistic over the realisations actually observed, and they are reported as such.

For package-level validation, `itrace.validation` separates two cases that are
often conflated. Synthetic-domain validation has event truth: seeded clean,
webcam-jitter, head-drift, and low-light/dropout sessions provide saccade
intervals, amplitudes, directions, pupil events, blinks, and quality flags, so
within-domain repeat summaries and cross-domain stability gaps can report
interval-overlap precision, recall, F1, amplitude error, direction error,
peak-velocity error, and pupil-validity statistics. Live webcam diagnostics do
not have that oracle. Following the data-quality reporting emphasis in recent
eye-tracking standards work, the live HTML interface therefore reports sampling
regularity, finite gaze fraction, pupil-valid fraction, path length, dispersion,
and warnings, but deliberately leaves recovery F1 undefined unless a future
reference-device or public-dataset path supplies truth [@jakobi2024quality].
