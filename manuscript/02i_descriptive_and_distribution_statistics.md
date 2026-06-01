# Methods: descriptive and distribution statistics {#sec:diststats}

The `itrace.stats` package turns the typed events of [@sec:events] and
[@sec:dynamics] into summary statistics, fitted distributions, and spread
metrics. Like the rest of the core it is pure NumPy/SciPy, takes explicit seeds
wherever it resamples, and operates on whatever gaze it is given — synthetic
streams with known ground truth or recorded ones. The methods below *describe* a
scanpath; none of them measure real-eye accuracy, which remains the
device-validation gap of [@sec:limitations].

## Descriptive summaries

Given a list of fixations or saccades, `stats.descriptive` returns the count and
the mean, standard deviation, median, and inter-quartile range of the relevant
per-event quantity — fixation duration for fixations; amplitude, duration,
direction, and peak velocity for saccades — reusing the property arrays already
vectorised by `saccades.saccade_properties`. An empty list yields a count of zero
and zero-filled summaries rather than raising, so a recording in which a detector
finds no events of a given type flows through the pipeline without a special case. These
are ordinary sample statistics over the detected events, not estimates of an
underlying population the recording was not designed to sample.

## Distribution fitting and model comparison

Fixation durations and saccade amplitudes are strictly positive and right-skewed,
so iTrace fits three standard positive-support families by maximum likelihood:
the **gamma** and **log-normal** distributions, and the **ex-Gaussian**
(a Gaussian convolved with an exponential), whose right tail has long been used
to model the shape of reaction-time and fixation-duration distributions
[@ratcliff1979group; @luce1986response]. For the two SciPy families the location
parameter is **pinned to zero** (`floc=0`) so the fit respects the positive
support and the scale parameter is not traded off against a spurious shift; the
ex-Gaussian is fit by numerically maximising its log-likelihood over
$(\mu, \sigma, \tau)$ from method-of-moments starting values. Models are ranked by
the **Akaike and Bayesian information criteria**
($\mathrm{AIC} = 2k - 2\ln\hat{L}$, $\mathrm{BIC} = k\ln n - 2\ln\hat{L}$), which
penalise the extra ex-Gaussian parameter against any gain in fit, and a
lower-is-better ordering is reported alongside the raw log-likelihoods. Absolute
goodness of fit is summarised by a one-sample **Kolmogorov–Smirnov** statistic
$D = \sup_x |F_n(x) - \hat{F}(x)|$ between the empirical CDF and each fitted CDF;
we report $D$ as a descriptive distance and treat the companion $p$-value with
caution, since the parameters were estimated from the same sample (the classic
fitted-parameter caveat). Fitting requires a minimum number of finite positive
observations and raises a `ValueError` below it, mirroring the guard in
`mainsequence.fit`.

## Scanpath spread metrics

The spatial extent of a scanpath is summarised by four metrics computed over
fixation centroids (in degrees of visual angle). **Gaze dispersion** is the root
mean square distance of fixations from their centroid. The **bivariate contour
ellipse area (BCEA)** — the area of the ellipse expected to contain a stated
fraction (default 68%) of fixations under a bivariate-normal assumption,
$\mathrm{BCEA} = 2\pi k\,\sigma_x \sigma_y \sqrt{1-\rho^2}$ with
$k = -\ln(1-P)$ — is a standard, units-of-deg$^2$ measure of fixation stability
introduced for fixational eye movements [@steinman1965effect] and widely used as a
low-vision fixation-stability index [@castet2012quantifying]. The **convex-hull
area** of the fixation centroids gives a distribution-free footprint of explored
space via `scipy.spatial.ConvexHull` (returning zero for fewer than three
non-collinear points rather than raising). Both BCEA and convex-hull area are
reported only as conditional descriptors: BCEA assumes approximate bivariate
normality, which a structured scanpath violates, and is flagged as such.

Two **entropy** metrics capture organisation rather than extent. *Stationary
position entropy* discretises fixation centroids onto a coarse spatial grid and
reports the Shannon entropy of the occupancy histogram in bits — higher when gaze
is spread evenly across cells, lower when it concentrates. *Direction-transition
entropy* reuses the cardinal scanpath string of [@sec:dynamics]: it forms the
first-order transition-count matrix between successive direction symbols and
reports the entropy of the transition distribution, a compact index of how
predictable the scan order is, in the spirit of information-theoretic scanpath
analysis [@boccignone2011generative]. Both are normalised against the maximum
entropy of their respective alphabets so values are comparable across recordings
of different length, and both degenerate gracefully (zero entropy) when only one
cell or symbol is observed.

## Bootstrap CI on the main-sequence exponent

The power-law main-sequence fit of [@sec:dynamics] reports a point estimate of the
exponent $b$ (`power_b` from `mainsequence.fit`). To attach uncertainty,
`stats.scanpath_metrics.main_sequence_exponent_ci` resamples the matched amplitude / peak-velocity
pairs with replacement $B$ times (default $B = 2000$, explicit seed), refits the
log-log regression on each resample, and reports a two-sided 95% **bootstrap
percentile** confidence interval on $b$ [@efron1993introduction]. We use the
percentile interval here for the same reason it is used in the noise sweep
([@sec:statmethods]): it makes no symmetry assumption and stays inside the range
supported by the data. The interval describes sampling variability of the exponent
*across the observed saccades*; it is not a claim about a population of eyes the
recording did not sample. Resamples that lose enough unique positive amplitudes
to make the log-log fit ill-conditioned are discarded; if no valid resample
survives, the interval collapses to the point estimate rather than producing a
spurious narrow bound.
