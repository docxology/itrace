# Results: figure gallery {#sec:gallery}

The analysis core is paired with a deterministic visualization layer
(`itrace.viz`, matplotlib on the headless Agg backend) that renders the standard
eye-movement figures directly from a `SessionReport`. Each figure is produced
from a synthetic session whose ground truth is known by construction, so the
gallery is reproducible byte-for-byte and never depends on a captured recording.
Categorical series draw from the shared Wong (2011) colour-blind-safe palette;
scalar fields use perceptual sequential or diverging colormaps that match the
quantity being encoded. All panels share the same house style — a readable font
floor, white background, restrained spines, and compact result callouts — so the
published figures match without pretending one palette serves every data type. In
the multi-series comparison figures, signals are separated by marker and line
style as well as colour, so the figures stay legible in greyscale. The figures
below are regenerated for publication by `scripts/generate_figures.py`, which
also writes the graphical abstract, noise sidecars, empirical summary, and
figure manifest. The CLI command
`itrace figures --out-dir output/figures --animations` remains the gallery-only
refresh path.

The gallery spans the three signal families iTrace recovers. For **gaze and
events**, a velocity trace (`output/figures/velocity_trace.png`) overlays the
2-D speed signal with the I-VT threshold and shades each detected saccade, while
a spatial scanpath (`output/figures/scanpath.png`) draws fixations sized by
dwell time and joined by fixation-transition arrows in screen coordinates. For **saccade
dynamics**, an amplitude histogram (`output/figures/amplitude_histogram.png`) carries
its maximum-likelihood distribution overlay, and a main-sequence panel
(`output/figures/main_sequence_diagnostics.png`) plots amplitude against peak velocity on
log-log axes with the fitted power law and its residuals. For **pupillometry**, a
pupil trace (`output/figures/pupil_trace.png`) shows the raw signal, the
blink-interpolated and smoothed overlay, and the causally-detected dilation peaks
and troughs.

A multi-panel session dashboard (`output/figures/session_dashboard.png`)
composes the velocity trace, scanpath, amplitude distribution, main sequence,
saccade-direction polar histogram, and a textual summary into a single
at-a-glance figure. The statistical diagnostics composite in
[@fig:statistical-diagnostics] takes the same `SessionReport` view and makes the
statistics layer inspectable: distribution-family ranking is plotted as
lower-is-better AIC deltas with AIC weights, AICc, BIC/KS annotations, and robust amplitude-shape
fields (median and IQR with seeded bootstrap intervals, Bowley skewness, and
IQR outside values), plus a seeded bootstrap of the lowest-AIC family winner to
show model-ranking stability without treating that frequency as a posterior
probability. The
best-fit amplitude family is checked against empirical quantiles and fitted-CDF
residuals with integrated/tail-weighted distance summaries and a DKW/Massart
reference band, the
main-sequence exponent is shown with a seeded bootstrap interval, fixation
centroids are paired with dispersion/BCEA/hull descriptors, and the scanpath
string is decomposed into a first-order transition matrix. The panel is useful
because each plotted value is computed by the tested Python statistics modules;
it does not turn a synthetic session into population physiology or device
validation evidence. The same refresh writes
`output/figures/statistical_diagnostics.json`, so the PNG is display-only over
an audited statistical payload rather than an untracked matplotlib-only
calculation.

![Statistical diagnostics over a deterministic synthetic session. Panel A ranks amplitude-distribution families by delta AIC while carrying Akaike weights, AICc, BIC, Kolmogorov-Smirnov descriptors, bootstrap lowest-AIC winner stability, robust median/IQR shape summaries with seeded bootstrap intervals, Bowley quartile skewness, and IQR outside-value counts; panel B plots empirical amplitude quantiles against fitted quantiles for the lowest-AIC family, reports QQ residual scale, and adds a P-P/CDF residual inset with integrated and Anderson-Darling-style tail-weighted CDF distances plus a DKW/Massart reference band; panel C shows the seeded bootstrap interval for the main-sequence exponent; panel D overlays fixation centroids with a BCEA-style ellipse and spatial-spread summaries; panel E renders the row-normalised scanpath transition matrix. The figure visualizes tested core descriptors and relative model diagnostics, not real-eye accuracy, population inference, or proof that a fitted family is true.](../output/figures/statistical_diagnostics.png){#fig:statistical-diagnostics width=100%}

The statistical panel is intentionally information-dense, so
[@fig:statistical-interpretation-ledger] records the companion interpretation
ledger generated from `output/figures/statistical_interpretation_ledger.json`.
Each row names the statistic, its reported value, the estimand it actually
summarizes, and the claim it explicitly does not support. This matters for the
scholarship boundary: AIC weights remain relative candidate-family weights, DKW
bands remain finite-sample empirical-CDF reference bands, bootstrap intervals
remain report-level uncertainty summaries, and the N=1 range bridge remains a
single-session scale check rather than device validation.

![Statistical interpretation ledger generated from `output/figures/statistical_interpretation_ledger.json`. The rows map robust shape summaries, relative distribution comparison, CDF residual checks, bootstrap main-sequence uncertainty, spatial descriptors, scanpath transitions, idealized noise sensitivity, and the synthetic-to-empirical range bridge to their source artifacts, estimands, and explicit non-claims. The figure improves statistical readability without adding new evidence; it states what each statistic does not prove, including population physiology, posterior model truth, real-eye accuracy, and device validation.](../output/figures/statistical_interpretation_ledger.png){#fig:statistical-interpretation-ledger width=100%}

The gallery now also includes gaze density, fixation heatmap, AOI dwell, event
raster, pupil PSD, saccade-rate, microsaccade, and deterministic synthetic
replay outputs. Because every panel is a pure function of a `SessionReport`, the
same plots apply unchanged to a real recording analysed through the pipeline,
subject to the same detector and calibration limits stated in [@sec:limitations].
