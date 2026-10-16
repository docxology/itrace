# Supplement S1: statistical methods {#sec:statmethods}

This supplement gives the full statistical detail behind [@sec:noiseresults] so
the analysis is reproducible and auditable.

## Estimator and interval

For each (noise level, metric) we collect $n$ independent trial values
$\{x_i\}$ and report the sample mean $\bar{x}$ with a two-sided 95% **bootstrap
percentile** confidence interval: we draw $B = 2000$ resamples of size $n$ with
replacement, take the mean of each, and read the 2.5th and 97.5th percentiles of
the resampled means.

We deliberately avoid a symmetric Student-$t$ interval
`xbar +/- t_(0.975,n-1) * s / sqrt(n)` here because two of the three metrics are
**bounded** — saccade F1 $\in [0,1]$ and pupil correlation $\in [-1,1]$ — and at
low noise both sit near 1.0, exactly where a symmetric interval can extend past
the bound and report a nonsensical value above 1. A percentile of resampled means
cannot leave the observed bounds. This support property does not establish 95%
coverage: for example, an all-success Bernoulli sample produces the interval
$[1,1]$ even when the generating probability is below one. These are approximate,
pointwise intervals; coverage for the sweep settings has not been established
[@efron1987better; @morris2019simulation].

## What the interval means (and the determinism caveat)

The pipeline is fully deterministic: re-running it reproduces every number
byte-for-byte ([@sec:reproducibility]). The confidence interval therefore does
**not** describe "what happens if I rerun the code." It describes the variability
of the mean **across the random landmark-noise realisations sampled by the $n$
seeded trials** under the declared scene and noise process. This Monte Carlo
uncertainty interval is distinct from the standard error of the mean,
$s n^{-1/2}$ for trial standard deviation $s$, reported separately in the sidecar
[@morris2019simulation]. It estimates neither between-participant variability
nor uncertainty in how well the noise model represents a real camera.

## Trial independence and determinism

Each trial uses a distinct deterministic seed (`base_seed + level_index*1000 +
trial`), and the bootstrap resampler is itself fixed-seeded. Pseudorandom draws
are treated as independent under the generator design; distinct integer seeds
are a reproducibility policy, not a proof of independence. Reproduction is
conditional on the recorded runtime and locked numerical dependencies.

## Saccade detection scoring

Saccade F1 uses **interval-overlap** matching against the ground-truth saccade
intervals using the same greedy one-to-one matcher as external benchmark scoring.
Predictions are processed in onset order and each selects the unmatched truth
interval with the largest strictly positive overlap. Touching endpoints do not
match; no interval can contribute more than one true positive. Fragmentation and
merging are therefore penalised. Precision, recall, and $F_1 = 2PR/(P+R)$ follow
from these counts. This greedy rule is deterministic, not a globally optimal
assignment. Even one-to-one overlap F1 does not establish onset/offset accuracy:
a greatly extended interval can still overlap a true event. Boundary errors
require separate matched-event timing metrics.

This revision replaces the archived release's many-to-many any-overlap sweep
scorer. The regenerated sweep and thresholds belong to the revised scorer;
historical release artifacts retain their original interpretation.

## Threshold crossings

A usability threshold (gaze RMS = 2°, saccade F1 = 0.8, pupil $r$ = 0.9) is
located by linear interpolation between the two grid points that bracket the
crossing; with $n=25$ these crossings are approximate and should be read to ~2
significant figures. The crossing in physical units uses
`sigma_to_pixels(σ, image_width)` ([@sec:noisemodel]).

## One source of truth

The figure ([@fig:power]) and the table ([@tbl:noise]) are both rendered from a
single `NoiseSweepResult` via `power.summary_records` / `format_summary_markdown`,
so prose, plot, and table cannot drift apart — verified by a cell-for-cell match
in the documentation audit.
