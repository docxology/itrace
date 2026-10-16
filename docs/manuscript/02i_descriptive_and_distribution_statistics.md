# Methods: descriptive and distribution statistics {#sec:diststats}

The `itrace.stats` package turns the typed events of [@sec:events] and
[@sec:dynamics] into summary statistics, fitted distributions, and spread
metrics. Like the rest of the core it is pure NumPy/SciPy, takes explicit seeds
wherever it resamples, and operates on whatever gaze it is given — synthetic
streams with known ground truth or recorded ones. The methods below *describe* a
scanpath; none of them measure real-eye accuracy, which remains the
device validation gap of [@sec:limitations]. The generated interpretation
ledger in [@fig:statistical-interpretation-ledger] is a companion artifact for
these methods: it maps each reported statistic to its source payload, estimand,
scholarship basis, and explicit non-claim boundary.

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

The publication diagnostics sidecar adds a robust amplitude-shape block before
any fitted model is interpreted. It reports the median, inter-quartile range,
coefficient of variation, mean-minus-median gap, Bowley's quartile skewness, and
Tukey-style 1.5 IQR outside-value counts [@bowley1901elements;
@tukey1977exploratory]. The same block adds seeded percentile-bootstrap
intervals for the median and IQR [@efron1993introduction], reported as
finite-sample uncertainty summaries rather than population confidence claims.
These descriptors are intentionally redundant with the parametric fits below:
the quartile summaries show the scale and asymmetry of the detected events even
when the sample is too small, too skewed, or too structured for a fitted family
to deserve much interpretive weight.

## Distribution fitting and model comparison

The publication payload compares five marginal candidates for detected positive
amplitudes: **normal, gamma, log-normal, Weibull, and ex-Gaussian**, fitted by
SciPy maximum-likelihood routines. Gamma, log-normal, and Weibull have positive
support with location **fixed at zero** (`floc=0`), leaving shape and scale free.
Normal and ex-Gaussian have support on the whole real line and can assign mass
to impossible negative amplitudes; they are unconstrained descriptive comparators,
not positive-truncated models. The ex-Gaussian is a Gaussian convolved with an
exponential, parameterised by SciPy as $(K,\mathrm{loc},\mathrm{scale})$ with
$K=\tau/\sigma$ [@ratcliff1979group; @luce1986response]. The free-parameter count
excludes fixed location parameters, including a caller-fixed normal or
ex-Gaussian location. Models are ranked by
the **Akaike and Bayesian information criteria**
($\mathrm{AIC} = 2k - 2\ln\hat{L}$, $\mathrm{BIC} = k\ln n - 2\ln\hat{L}$)
[@akaike1974new; @schwarz1978estimating], which penalise the extra ex-Gaussian
parameter against any gain in fit, and a lower-is-better ordering is reported
alongside the raw log-likelihoods. The diagnostics payload also reports the
small-sample correction
$\mathrm{AICc} = \mathrm{AIC} + 2k(k+1)/(n-k-1)$ when it is defined, plus
normalised Akaike weights $w_i \propto \exp(-\Delta_i/2)$ and evidence ratios
within the same finite candidate set [@burnham2002model]. These weights are
relative support diagnostics for the compared families; they are not posterior
probabilities and do not prove that the selected family generated the data.
Here $n$ counts detected events, not gaze frames. AIC remains the declared
ranking, weighting, and winner-frequency criterion; AICc and BIC are sensitivity
descriptors. AICc is undefined for $n\leq k+1$, and this commonly used correction
is a heuristic for these fitted marginal families, not a finite-sample guarantee
for every density model [@hurvich1989regression]. The marginal product likelihood
also does not model serial dependence between events.
The same sidecar runs a deterministic bootstrap over detected amplitudes,
refits the same candidate family set on each resample, and records the
lowest-AIC winner frequencies as a stability diagnostic, not as posterior model
probabilities [@efron1993introduction; @burnham2002model].
Absolute goodness of fit is summarised by a one-sample **Kolmogorov–Smirnov** statistic
$D = \sup_x |F_n(x) - \hat{F}(x)|$ between the empirical CDF and each fitted CDF
[@massey1951kolmogorov]; we report $D$ as a descriptive distance and treat the
companion fixed-CDF $p$-value as **uncalibrated**, since the parameters were
estimated from the same sample. A separate refit-bootstrap KS diagnostic
simulates from the fitted family, refits each draw, and uses the plus-one
Monte Carlo tail probability. Its family was selected from these same data:
this exploratory diagnostic is not adjusted for model selection, serial
dependence, or multiple comparisons ([@sec:resampling-design]). Fitting requires a minimum
number of finite positive observations and raises a `ValueError` below it,
mirroring the guard in `mainsequence.fit`.

For the publication diagnostics payload, the lowest-AIC family is also rendered
as a quantile--quantile check: empirical order statistics are plotted against
the fitted distribution's quantiles at plotting positions $(i - 0.5)/n$
[@wilk1968probability]. The sidecar reports the QQ residuals, root-mean-square
residual, central-80% residual, maximum absolute residual, fitted-line slope,
intercept, and correlation. The same best-fit family is also checked on the CDF
scale: empirical plotting positions are compared with fitted CDF values at the
observed amplitudes, producing a P-P-style residual series, residual RMSE, mean
absolute residual, maximum residual, and the fitted KS distance
[@massey1951kolmogorov]. The same residual block reports a
Cramer--von-Mises-style integrated squared CDF distance and an
Anderson--Darling-style tail-weighted distance [@anderson1952asymptotic], plus
lower- and upper-tail maximum residuals. These scalar distances help separate a
fit that is wrong everywhere from one that fails mainly in the tails, but they
are still descriptive because the family and parameters were selected from the
same events. For scale, the payload also reports a
DKW/Massart reference width
`epsilon = sqrt(log(2 / alpha) / (2 n))` for the empirical CDF at $\alpha=0.05$
[@massart1990tight], plus whether the maximum fitted-CDF residual exceeds that
width. This band is displayed as a distribution-free visual reference for ECDF
fluctuation for IID observations around a fixed CDF; it does not account for
serial dependence. Because the distribution family and its
parameters were selected from the same sample, it is not a formal acceptance
test. The quantile and CDF residuals are model-adequacy diagnostics for the
detected sample; they are not a proof that the family is true or that another
recording would share the same amplitude distribution.

## Scanpath spread metrics

The spatial extent of a scanpath is summarised by four metrics computed over
fixation centroids (in degrees of visual angle). **Gaze dispersion** is the root
mean square distance of fixations from their centroid. The **bivariate contour
ellipse area (BCEA)** — the area of the ellipse expected to contain a stated
fraction (default 68%) of fixations under a bivariate-normal assumption,
`BCEA = 2 * pi * k * sigma_x * sigma_y * sqrt(1 - rho^2)` with
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
analysis [@boccignone2004gaze]. Both are reported in bits, without maximum-entropy
normalisation. Direction-transition entropy is the count-weighted conditional
entropy $H(\mathrm{next}\mid\mathrm{current})$, not joint transition entropy.
Spatial binning defaults to the supplied record's extent; cross-record comparisons
must specify a common grid, extent, and symbol convention. Unequal event counts
and sparse transition rows introduce estimation uncertainty and finite-sample
bias [@paninski2003estimation]. Zero observed transitions do not establish zero
population probability. Both metrics return zero for degenerate one-cell or
one-symbol inputs.

## Bootstrap CI on the main-sequence exponent

The power-law main-sequence fit of [@sec:dynamics] reports a point estimate of the
exponent $b$ (`power_b` from `mainsequence.fit_power_law`). To attach uncertainty,
`stats.scanpath_metrics.main_sequence_exponent_ci` resamples the matched amplitude / peak-velocity
pairs with replacement $B$ times (default $B = 2000$, explicit seed), refits the
log-log regression on each resample, and reports a two-sided 95% **bootstrap
percentile** confidence interval on $b$ [@efron1993introduction]. We use the
percentile interval here for the same reason it is used in the noise sweep
([@sec:statmethods]): it makes no symmetry assumption and stays inside the range
of the resampled estimates. Coverage is approximate and not established merely
by that property. IID resampling perturbs the observed event pairs under an
exchangeability assumption; moving-block resampling is available with an explicit
block length. Neither estimates a population of eyes from a single recording.
The detailed result reports attempted, successful, and failed draws. Rank-deficient
draws are excluded and the surviving interval is conditional on successful fits;
if no valid draw survives, the interval is unavailable and the API raises instead
of returning zero-width uncertainty [@morris2019simulation].
