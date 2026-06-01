# Results: noise-sensitivity analysis {#sec:noiseresults}

Sweeping the landmark-noise standard deviation $\sigma$ from 0 to 0.016
(normalised image units), 25 seeded trials per level, gives a clear and
differentiated degradation of the three signals ([@fig:power], [@tbl:noise]).

**Gaze** RMS error rises monotonically and approximately linearly with $\sigma$,
crossing the 2° usability bound at $\sigma \approx 0.005$. **Saccade** detection
is the most fragile emergent signal: F1 falls below 0.8 already at
$\sigma \approx 0.0014$ — a direct consequence of detection resting on the
velocity, the time-derivative of position, which amplifies high-frequency noise.
Both orderings *emerge* from propagating landmark noise through the real
estimator.

The **pupil** result carries a sharp caveat: its robustness (correlation above
0.9 until $\sigma \approx 0.005$) is **robust by construction** under the assumed
`pupil_noise_scale`, not a measured property, and would move if that parameter
changed. It is reported as a conditional illustration, not a finding; the genuine
emergent result is the gaze-vs-saccade ordering.

The most defensible practical takeaway comes from translating $\sigma$ to pixels.
At a 640 px width, the saccade breakdown at $\sigma \approx 0.0014$ is
$\approx 0.9$ px, whereas gaze tolerates roughly three pixels in this idealised
sweep. That pixel-scale comparison is not a universal detector specification:
real webcam landmark error is camera-, pose-, lighting-, compression-, and
algorithm-dependent. It does show why webcam saccade *timing* is theoretically
marginal before any real-world bias, temporal correlation, or tracking dropout is
added, while coarse fixation and gaze direction have more headroom.

![Recovery vs idealised webcam observation noise. Each panel reports the Monte-Carlo mean and 95% bootstrap confidence band for one recovered signal, with the manuscript usability bound drawn as a horizontal line and the interpolated bound crossing annotated in both normalised σ and approximate pixels at 640 px width. Colour, marker, and line style all encode the signal so the figure remains legible in greyscale; the pupil panel is robust only under the stated `pupil_noise_scale` assumption.](../output/figures/noise_power.png){#fig:power width=100%}

The same numbers, with per-cell uncertainty, are the canonical statistics table
([@tbl:noise]); it and the figure are generated from one sweep so they never
drift. (Snapshot at $n=25$; regenerate with `scripts/generate_power_figure.py`.)

: Recovery vs landmark noise σ (mean ± 95% half-CI, $n=25$). {#tbl:noise}

| σ (norm) | σ (px@640) | gaze RMS (deg) | saccade F1 | pupil r |
|---|---|---|---|---|
| 0.0000 | 0.00 | 0.163 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 0.0010 | 0.64 | 0.452 ± 0.006 | 0.949 ± 0.029 | 0.996 ± 0.000 |
| 0.0020 | 1.28 | 0.842 ± 0.012 | 0.612 ± 0.028 | 0.982 ± 0.001 |
| 0.0040 | 2.56 | 1.665 ± 0.026 | 0.220 ± 0.007 | 0.934 ± 0.003 |
| 0.0080 | 5.12 | 3.418 ± 0.046 | 0.274 ± 0.017 | 0.800 ± 0.008 |
| 0.0160 | 10.24 | 6.798 ± 0.102 | 0.560 ± 0.043 | 0.546 ± 0.018 |
