# Results: noise-sensitivity analysis {#sec:noiseresults}

Sweeping the landmark-noise standard deviation $\sigma$ from 0 to 0.016
(normalised image units), 25 seeded trials per level, gives a clear and
differentiated degradation of the three signals ([@fig:power], [@tbl:noise]).
The sweep is intentionally idealised: it injects independent landmark noise into
the same synthetic scene rather than replaying real MediaPipe failures, so its
value is to rank algorithmic fragilities under controlled perturbation.

**Gaze** RMS error rises monotonically and approximately linearly with $\sigma$,
crossing the 2° usability bound at $\sigma \approx {{NOISE_GAZE_SIGMA}}$. **Saccade** detection
is the most fragile emergent signal: F1 falls below 0.8 already at
$\sigma \approx {{NOISE_SACCADE_SIGMA}}$ under the revised one-to-one overlap scorer. Detection rests on
velocity, the time-derivative of position, which amplifies high-frequency noise.
Both orderings *emerge* from propagating landmark noise through the real
estimator.

The **pupil** result carries a sharp caveat: its robustness (correlation above
0.9 until $\sigma \approx {{NOISE_PUPIL_SIGMA}}$) is **robust by construction** under the assumed
`pupil_noise_scale`, not a measured property, and would move if that parameter
changed. It is reported as a conditional illustration, not a finding; the genuine
emergent result is the gaze-vs-saccade ordering.

The most defensible practical takeaway comes from translating $\sigma$ to pixels.
At a 640 px width, the saccade crossing is approximately {{NOISE_SACCADE_PX}} px,
whereas the gaze crossing is approximately {{NOISE_GAZE_PX}} px in this idealised
sweep. That pixel-scale comparison is not a universal detector specification:
real webcam landmark error is camera-, pose-, lighting-, compression-, and
algorithm-dependent. It characterises overlap-based event recovery under this
perturbation model. It does not establish onset/offset accuracy or a real-camera
noise tolerance; those require independent measurements and boundary-error metrics.

The N=1 empirical target residual in [@sec:empirical-pilot] is much larger than
the residuals in this idealised landmark-noise sweep for the expected reason: it
bundles calibration, screen-prompt, user-compliance, target-eccentricity,
timestamp, and webcam-session effects into one prompted-target diagnostic. That
makes it useful for bounding realistic live-demo and stress-test ranges, but it
should not be back-solved into an iid landmark-noise $\sigma$ or treated as
reference-device gaze accuracy.

![Recovery vs idealised webcam observation noise. Each panel reports the Monte Carlo mean and pointwise approximate 95% percentile interval for one recovered signal. These are not simultaneous confidence bands. The manuscript criterion and interpolated crossing are shown in normalised sigma and approximate pixels at 640 px width. Colour, markers, and line styles distinguish signals. Saccade F1 uses greedy one-to-one positive overlap; it does not measure onset/offset accuracy. Pupil correlation depends on the separate pupil-noise assumption.](../../output/figures/noise_power.png){#fig:power width=100%}

The same numbers, with per-cell uncertainty, are the generated statistics table
([@tbl:noise]); it and the figure are generated from one sweep so they never
drift. (Snapshot at $n=25$; regenerate with `scripts/generate_power_figure.py`.)

: Recovery vs landmark noise sigma: mean [lower, upper] pointwise interval. {#tbl:noise}

{{NOISE_TABLE}}
