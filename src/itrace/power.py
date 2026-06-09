"""Noise-sensitivity analysis of parameter recovery under webcam observation noise.

This is a **sensitivity / robustness sweep** — it characterises how each
recovered signal degrades as observation noise grows. It is *not* a
hypothesis-testing "statistical power" calculation (no null hypothesis, effect
size, or P(detect) is computed); we keep the colloquial sense the request used
but state the operational meaning here.

Idealized noise model
---------------------
The dominant quantity that corrupts gaze/saccade/pupil recovery from a webcam is
**landmark-localisation error**: photon shot noise, JPEG compression, pixel
quantisation, and MediaPipe's finite precision all manifest downstream as jitter
in the normalised image coordinates of the landmarks. We model this with an
*idealized* additive **i.i.d. Gaussian** of standard deviation ``sigma`` (in
normalised image-coordinate units) on every landmark. This is deliberately
simplified: it ignores spatial correlation between adjacent landmarks, temporal
correlation, heteroscedasticity (error grows at eccentric gaze and under poor
lighting), and systematic bias. It is not derived from sensor optics.

What is — and is not — emergent
-------------------------------
* **gaze** RMS error and **saccade** F1 both *emerge* from propagating the
  landmark noise through the real estimator. Their ordering (saccades degrade
  before gaze) is a genuine consequence of saccade detection resting on velocity
  — a derivative, which amplifies high-frequency noise.
* **pupil** correlation robustness is **NOT emergent**: the pupil/iris ratio is
  read from a separate modelled measurement whose noise is set by
  ``pupil_noise_scale`` (the sweep does not run a calibrated image-segmentation
  or millimetre pupil model). Pupil position in the ordering is therefore an
  *assumption*, not a finding, and would move if ``pupil_noise_scale`` changed.
  Report it as conditional.

Saccade F1 uses **interval-overlap** matching: a true saccade counts as recalled
if any detected interval overlaps it (no separate temporal tolerance). Means use
a **bootstrap percentile** 95% CI (bound-respecting for F1/correlation); with ~25
seeds, threshold crossings are approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import EyeSceneSpec, closed_loop
from .types import FloatArray

# Metrics tracked across the sweep (key in ClosedLoopResult.metrics, "higher is
# better" flag used for threshold-direction).
METRICS: dict[str, bool] = {
    "gaze_rms_deg": False,  # lower is better
    "saccade_f1": True,  # higher is better
    "pupil_corr": True,  # higher is better
}


@dataclass(frozen=True, slots=True)
class MetricCurve:
    """Mean and 95% CI of one metric across the noise sweep."""

    name: str
    higher_is_better: bool
    noise_levels: list[float]
    mean: list[float]
    ci_low: list[float]
    ci_high: list[float]
    std: list[float]


@dataclass(frozen=True, slots=True)
class NoiseSweepResult:
    """Full sweep: one :class:`MetricCurve` per tracked metric."""

    noise_levels: list[float]
    n_trials: int
    curves: dict[str, MetricCurve]

    def curve(self, metric: str) -> MetricCurve:
        return self.curves[metric]


def _mean_ci(
    values: FloatArray, confidence: float = 0.95, n_boot: int = 2000, seed: int = 12345
) -> tuple[float, float, float, float]:
    """Return (mean, ci_low, ci_high, std) via a **bootstrap percentile** interval.

    A symmetric Student-t interval is invalid for the bounded metrics here
    (F1 in [0, 1], correlation in [-1, 1]): near 1.0 it can run past the bound.
    A percentile of resampled means can never leave the range of the observed
    (in-bound) values. The resampling rng is fixed-seeded, so the CI is
    deterministic and the whole sweep stays reproducible.

    Interpretation: the interval describes variability of the mean across the
    random landmark-noise realisations sampled by the n seeded trials (a Monte
    Carlo standard error over the noise process), not rerun variability of the
    deterministic pipeline.
    """
    n = values.size
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n < 2 or std == 0.0:
        return mean, mean, mean, std
    rng = np.random.default_rng(seed)
    boot_means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (1.0 - confidence) / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 + confidence) / 2.0))
    return mean, lo, hi, std


def run_noise_sweep(
    noise_levels: list[float] | None = None,
    n_trials: int = 20,
    spec: EyeSceneSpec | None = None,
    *,
    base_seed: int = 1000,
) -> NoiseSweepResult:
    """Monte-Carlo sweep of recovery accuracy vs landmark noise.

    Each (noise level, trial) runs one closed loop with a distinct deterministic
    seed, so the whole sweep is reproducible.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016]
    spec = spec or EyeSceneSpec()

    samples: dict[str, list[FloatArray]] = {m: [] for m in METRICS}
    for li, sigma in enumerate(noise_levels):
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for trial in range(n_trials):
            seed = base_seed + li * 1000 + trial
            res = closed_loop(spec, landmark_noise_sd=sigma, seed=seed)
            for m in METRICS:
                per_metric[m].append(res.metrics[m])
        for m in METRICS:
            samples[m].append(np.asarray(per_metric[m], dtype=np.float64))

    curves: dict[str, MetricCurve] = {}
    for m, higher in METRICS.items():
        means, los, his, stds = [], [], [], []
        for arr in samples[m]:
            mean, lo, hi, std = _mean_ci(arr)
            means.append(mean)
            los.append(lo)
            his.append(hi)
            stds.append(std)
        curves[m] = MetricCurve(
            name=m,
            higher_is_better=higher,
            noise_levels=list(noise_levels),
            mean=means,
            ci_low=los,
            ci_high=his,
            std=stds,
        )
    return NoiseSweepResult(noise_levels=list(noise_levels), n_trials=n_trials, curves=curves)


def sigma_to_pixels(sigma_normalised: float, image_width_px: float = 640.0) -> float:
    """Translate landmark noise σ (normalised units) to pixels for a given webcam.

    Lets the idealised sweep be compared with the one-pixel-to-few-pixel scale
    that users recognise in 640 px webcam frames. It is a physical scale
    translation, not a claim that every detector, camera, face pose, or lighting
    condition shares one universal localisation floor.
    """
    return sigma_normalised * image_width_px


def recovery_threshold(curve: MetricCurve, bound: float) -> float | None:
    """Noise level at which the mean metric first crosses ``bound``.

    For a "lower is better" metric (gaze RMS) this is where the mean first
    *exceeds* the bound; for "higher is better" (F1, correlation) where it first
    *drops below* it. Linear interpolation gives a sub-grid crossing. Returns
    ``None`` if the metric never crosses the bound across the swept range.
    """
    levels, means = curve.noise_levels, curve.mean
    for i in range(1, len(levels)):
        prev, cur = means[i - 1], means[i]
        crossed = (cur > bound >= prev) if not curve.higher_is_better else (cur < bound <= prev)
        if crossed:
            span = cur - prev
            frac = 0.0 if span == 0 else (bound - prev) / span
            return float(levels[i - 1] + frac * (levels[i] - levels[i - 1]))
    return None


def summary_records(
    result: NoiseSweepResult, image_width_px: float = 640.0
) -> list[dict[str, float]]:
    """One composable record per noise level: σ, pixels, and per-metric mean ± CI.

    A flat, JSON/DataFrame-friendly shape so the same numbers drive the figure,
    the manuscript table, and any downstream analysis — one source of truth for
    the statistics.
    """
    rows: list[dict[str, float]] = []
    for i, sigma in enumerate(result.noise_levels):
        row: dict[str, float] = {
            "noise_sigma": float(sigma),
            "noise_px": sigma_to_pixels(sigma, image_width_px),
            "n_trials": float(result.n_trials),
        }
        for metric, curve in result.curves.items():
            row[f"{metric}_mean"] = curve.mean[i]
            row[f"{metric}_ci_low"] = curve.ci_low[i]
            row[f"{metric}_ci_high"] = curve.ci_high[i]
            row[f"{metric}_std"] = curve.std[i]
        rows.append(row)
    return rows


_TABLE_COLS: tuple[tuple[str, str], ...] = (
    ("gaze_rms_deg", "gaze RMS (deg)"),
    ("saccade_f1", "saccade F1"),
    ("pupil_corr", "pupil r"),
)


def format_summary_markdown(result: NoiseSweepResult, image_width_px: float = 640.0) -> str:
    """Render the sweep as an accessible Markdown table (mean ± half-CI per cell).

    Accessible by construction: every cell carries its own uncertainty, units are
    in the header, and a pixel column grounds σ physically — the statistics are
    legible without reading any figure or relying on colour.
    """
    rows = summary_records(result, image_width_px)
    header = [
        "sigma (norm)",
        f"sigma (px@{int(image_width_px)})",
        *(lbl for _k, lbl in _TABLE_COLS),
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [f"{r['noise_sigma']:.4f}", f"{r['noise_px']:.2f}"]
        for key, _label in _TABLE_COLS:
            half = (r[f"{key}_ci_high"] - r[f"{key}_ci_low"]) / 2.0
            cells.append(f"{r[f'{key}_mean']:.3f} +/- {half:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    caption = (
        f"\n*Recovery vs landmark noise sigma (n={result.n_trials} seeded trials/level; "
        f"mean +/- 95% bootstrap-percentile half-CI, bound-respecting for F1/r). sigma in "
        f"normalised units and pixels at {int(image_width_px)} px width.*"
    )
    return "\n".join(lines) + "\n" + caption
