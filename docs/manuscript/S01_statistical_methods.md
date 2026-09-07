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
can never leave the range of the in-bound observations, so the interval is always
valid. For the unbounded gaze RMS the two methods agree closely.

## What the interval means (and the determinism caveat)

The pipeline is fully deterministic: re-running it reproduces every number
byte-for-byte ([@sec:reproducibility]). The confidence interval therefore does
**not** describe "what happens if I rerun the code." It describes the variability
of the mean **across the random landmark-noise realisations sampled by the $n$
seeded trials** — a Monte Carlo standard error of the mean over the noise
process. With $n=25$ fixed seeds it quantifies how much the level-mean would move
under a different finite draw of noise realisations.

## Trial independence and determinism

Each trial uses a distinct deterministic seed (`base_seed + level_index*1000 +
trial`), and the bootstrap resampler is itself fixed-seeded, so trials are
independent draws of the landmark-noise process while the whole sweep — including
every CI — is byte-for-byte reproducible.

## Saccade detection scoring

Saccade F1 uses **interval-overlap** matching against the ground-truth saccade
intervals: a true saccade is recalled if any detected interval overlaps it, and a
detected interval is a true positive if it overlaps any true saccade. There is no
separate temporal-tolerance parameter; precision, recall, and
$F_1 = 2PR/(P+R)$ follow directly. This makes the score insensitive to small
onset/offset shifts while still penalising spurious or missed events.

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
