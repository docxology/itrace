# Statistical design and interpretation

This guide describes development beyond the archived v0.5.0 baseline. The
published tag and Zenodo files retain that release's original implementation.
Use an exact Git commit and the current verification ledger for this revision.

## Choose the unit before the interval

| Question | Unit and assumption | Interpretation |
|---|---|---|
| Mean recovery under a synthetic noise process | Whole independently generated trial | Monte Carlo uncertainty conditional on the scene and generator |
| Stability of detected-event summaries | IID detected events, if exchangeable | Perturbation of the observed event list; no between-person inference |
| Sensitivity to local serial dependence | Consecutive moving blocks under approximate stationarity | Depends on a justified block length; examine multiple lengths |
| Between-person or between-session differences | Participants or sessions defined by the study design | Requires those independent units in the data; individual samples do not replace them |

Percentile intervals preserve the range of bootstrap estimates but do not
automatically have nominal coverage. The noise table reports actual lower and
upper endpoints, and the sidecar separately reports trial standard deviations
and Monte Carlo standard errors. A pointwise interval at each noise level is
not a simultaneous band across the curve.

```python
import numpy as np
from itrace.stats import bootstrap_statistic, percentile_interval
from itrace.stats.bootstrap import resampling_sensitivity
from itrace.stats.scanpath_metrics import main_sequence_exponent_bootstrap

observations = np.asarray(resampling_sensitivity(seed=72)["values"])
draws = bootstrap_statistic(
    observations, np.mean, n_boot=2000, seed=72, block_length=4
)
lower, upper = percentile_interval(draws)

amplitude = np.linspace(2.0, 20.0, 30)
velocity = 40.0 * amplitude**0.65
result = main_sequence_exponent_bootstrap(
    amplitude, velocity, n_boot=500, seed=72, block_length=4
)
print(result.n_successful, result.n_failed, result.ci_low, result.ci_high)
```

The example uses a synthetic stationary AR(1) record, not eye measurements or
a block-length recommendation. Real missing rows, task boundaries, and clusters need an
explicit preprocessing and resampling policy. The API rejects nonfinite data
rather than changing adjacency silently. Main-sequence draws retain amplitude/
velocity pairing and report rank-deficient failures. No successful draws raises
an error instead of returning a point-width interval.

## Distribution comparisons and calibration

The diagnostics compare five candidate marginals by AIC and use that same
criterion for deltas, weights, winner frequencies, and the selected QQ/CDF fit.
AICc and BIC are sensitivity diagnostics. The AICc formula is a small-sample
heuristic for these families, not a guarantee; it is undefined for `n <= k + 1`.
The sample size counts events, and fixed locations do not count as estimated
parameters. Normal and ex-Gaussian candidates have real-line support; filtering
to positive amplitudes does not make them truncated positive distributions.

`FitResult.ks_pvalue` remains available for compatibility but is an uncalibrated
fixed-CDF probability after fitting. For a prespecified IID family, use
`parametric_ks_test(data, "normal", n_boot=999, seed=72)`: each generated sample
is refitted, fixed parameters remain fixed, and every draw must succeed. Its
plus-one tail probability has resolution `1 / (n_boot + 1)`. The publication
figure applies this to a data-selected family and labels the result exploratory;
selection and serial dependence are not adjusted away by refitting parameters.

## Event scores, agreement, and denominators

Closed-loop, noise-sweep, and benchmark scores share onset-ordered, greedy
one-to-one matching using the largest strictly positive overlap with unmatched
truth. Merging and fragmentation are penalized. Touching endpoints do not match.
The scorer is deterministic, not globally optimal. Overlap F1 does not measure
boundary accuracy; inspect onset, offset, and duration errors separately.

Correlation describes co-variation, not measurement agreement. Compare pupil
measurements only after declaring their units and calibration target. Reference
agreement studies require prespecified acceptable errors and a repeated-
measurement design. A relative live proxy versus millimetres is not made
comparable by drawing a difference plot.

Transition rows disclose observed counts and remain blank when no outgoing
transition was observed. Entropies are in bits; comparison needs a common grid,
screen extent, and symbol convention, plus attention to finite-sample bias.

The [scholarship review](SCHOLARSHIP_REVIEW_20261015.md) records primary sources,
metadata, and access limits. The manuscript's statistical-design supplement and
generated `resampling_sensitivity.json`/PNG provide the mathematical illustration.
