# Supplement S3: resampling design and inferential boundaries {#sec:resampling-design}

## Estimand, independent unit, and observed record

The noise sweep estimates the expected recovery metric for a fixed synthetic
scene under a specified perturbation generator. Its independent unit is an entire
generated trial. Event-wise bootstraps instead perturb the events detected in a
supplied recording. Neither design estimates between-person uncertainty from
the single participant represented in the empirical ledgers. Repeated sessions
from one person do not become independent people when resampled
[@bland2007multiple; @morris2019simulation].

`bootstrap_indices` supports IID indices and overlapping moving blocks of a
declared number of consecutive observations. Block starts are sampled uniformly
from the available complete blocks; concatenation is truncated to the original
record length, without wrapping across the record boundary. `bootstrap_statistic`
rejects missing or nonfinite input rather than silently deleting rows and changing
adjacency. Pair resampling uses the same indices for amplitude and velocity.
The default block length of one reproduces IID resampling; longer blocks require
an approximately stationary sequence and a scientifically justified length
[@kunsch1989jackknife]. A block as long as the record has no resampling variation.
Nonstationary task epochs, gaps, and participant clusters need a design tailored
to those units; an arbitrary block length is not a universal correction.
The controlled illustration in [@fig:resampling-design] makes the effect of
changing that resampling unit visible without assuming it validates coverage.

![Resampling sensitivity for one seeded stationary Gaussian AR(1) record in arbitrary units. Panel A shows the ordered observations; panel B displays exact percentile interval endpoints for the mean under IID resampling (L=1) and several moving-block lengths; panel C compares resampled standard errors with the finite-record analytic AR(1) mean standard error. All quantities come from the accompanying JSON, with the seed and design recorded. This example illustrates sensitivity to the resampling unit; it is not eye data, a coverage experiment, or a recommendation for a particular block length.](../../output/figures/resampling_sensitivity.png){#fig:resampling-design width=100%}

For this illustration, innovations have variance one and the initial observation
has stationary variance. With $n$ observations and lag correlation $\rho$, the
known generating variance of the sample mean is

$$V = (n + 2 S) / (n^2 (1 - \rho^2))$$

where $V$ denotes the variance of the sample mean and
`S = sum((n - h) * rho**h for h in range(1, n))`.

The analytic reference is checked against a separately constructed covariance
matrix. Agreement of one interval with its generating mean is not evidence of
nominal interval coverage; that would require repeated independent datasets and
Monte Carlo uncertainty for the coverage estimate [@morris2019simulation].

## Failure accounting and fit calibration

Main-sequence uncertainty uses log-log fits alone, without fitting the unrelated
saturating model on each draw. Draws with insufficient unique amplitudes are
counted as failures. The detailed API and figure expose attempted and successful
counts; an interval based on the remaining draws is conditional on success.
No successful draws produces an unavailable interval, never artificial precision.
Percentile endpoints are displayed directly, even when they do not bracket the
original point estimate [@efron1987better; @morris2019simulation].

The fixed-family KS refit bootstrap generates a sample from the fitted null,
refits all originally estimated parameters, and computes its fitted-CDF distance.
Originally fixed parameters stay fixed. For $r$ null distances at least as large
as observed among $B$ draws, the reported probability is $(r+1)/(B+1)$; the
minimum nonzero probability is $1/(B+1)$, disclosed as Monte Carlo resolution.
Every requested null fit must succeed. The publication diagnostic applies this
procedure to the AIC-selected family without repeating family selection, so it
remains explicitly exploratory. It is not selection-adjusted, does not establish
serial independence, and does not accept a model as true. QQ, P-P, KS, and
AD/CvM-style distances remain useful descriptive checks under these limits.

## What would support stronger claims

Future confirmatory comparisons should prespecify the independent unit, target
metric, candidate models, missingness policy, calibration split, acceptable error,
and multiplicity policy. Reusing calibration observations for evaluation or
selecting a favorable metric after seeing the data changes the question being
answered. A complete evaluation should report failures and exclusions with their
denominators, rather than only statistics from successful records
[@morris2019simulation]. Device agreement needs measurements of comparable
quantities and units; a perfectly correlated scaled signal can still disagree
systematically [@bland1986agreement].
