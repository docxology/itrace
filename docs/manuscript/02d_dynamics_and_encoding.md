# Methods: saccade dynamics and scanpath encoding {#sec:dynamics}

## Main sequence

For each saccade iTrace reports amplitude, direction, duration, and peak
velocity. The amplitude–peak-velocity relationship — the *main sequence* — is fit
to two standard parameterisations: a saturating exponential

$$ V = V_{\max}\,\big(1 - e^{-A/C}\big) $$ {#eq:mainseq}

([@eq:mainseq]; @bahill1975main) by non-linear least squares, and a power law
$V = aA^b$ by linear regression in log-log space, reporting the coefficient of
determination. A physiological oculomotor system gives an exponent $b$ of roughly
0.4–0.9; deviation is a recognised marker, so the fit doubles as a sanity probe.

## Scanpath encoding

Saccade sequences are encoded as direction characters as an implementation-level
summary: `R`/`L`/`U`/`D` for the nearest cardinal, upper-case for long saccades
and lower-case for short ones relative to a length threshold. N-gram statistics
over the resulting string give a compact order-sensitive descriptor for package
reports and tests without claiming biometric identification or anomaly-detection
performance.
