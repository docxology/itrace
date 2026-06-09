"""Pure statistical diagnostics payloads for reports and figures.

The plotting layer should display values computed by the tested Python core,
not quietly recompute a different payload inside matplotlib. This module
collects the statistics used by the publication diagnostics figure into a
JSON-friendly dictionary: distribution-model comparison, main-sequence
uncertainty, spatial spread, and scanpath-transition structure.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .. import saccades
from ..types import FloatArray, SessionReport
from . import distributions, scanpath_metrics, similarity

DEFAULT_DIAGNOSTIC_FAMILIES: tuple[str, ...] = (
    "normal",
    "gamma",
    "lognormal",
    "weibull",
    "exgaussian",
)
"""Distribution families compared by the standard diagnostics payload."""


def _positive_sample(values: FloatArray | list[float]) -> FloatArray:
    """Return finite positive values as a one-dimensional array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return cast(FloatArray, arr[np.isfinite(arr) & (arr > 0.0)])


def _positive_pairs(
    x: FloatArray | list[float],
    y: FloatArray | list[float],
) -> tuple[FloatArray, FloatArray]:
    """Return aligned finite positive pairs, truncating to the shorter input."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    return cast(FloatArray, a[mask]), cast(FloatArray, b[mask])


def _model_comparison_payload(
    sample: FloatArray,
    *,
    families: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """JSON-friendly AIC/BIC/KS model-comparison payload."""
    payload: dict[str, object] = {
        "sample": "saccade_amplitude_deg",
        "n": int(sample.size),
        "families": [],
        "model_selection_bootstrap": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "criterion": "aic",
            "available": False,
        },
        "available": False,
    }
    if sample.size < 3:
        payload["reason"] = "need >=3 positive events"
        payload["model_selection_bootstrap"] = {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "criterion": "aic",
            "available": False,
            "reason": "need >=3 positive events",
        }
        return payload

    results = distributions.compare_distributions(sample, families=families)
    if not results:
        payload["reason"] = "no stable distribution fits"
        payload["model_selection_bootstrap"] = {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "criterion": "aic",
            "available": False,
            "reason": "no stable distribution fits",
        }
        return payload

    best_aic = min(result.aic for result in results)
    finite_aicc = [result.aicc for result in results if np.isfinite(result.aicc)]
    best_aicc = min(finite_aicc) if finite_aicc else None
    aic_weights = distributions.information_weights(results, criterion="aic")
    aicc_weights = distributions.information_weights(results, criterion="aicc")
    best_weight = max(aic_weights.values()) if aic_weights else 0.0
    payload["available"] = True
    payload["best_family"] = results[0].family
    payload["weight_criterion"] = "aic"
    payload["aicc_weight_criterion"] = "aicc"
    payload["model_selection_bootstrap"] = _model_selection_bootstrap(
        sample,
        families=families,
        n_boot=n_boot,
        seed=seed,
    )
    payload["families"] = [
        {
            "family": result.family,
            "n": int(result.n),
            "params": {key: float(value) for key, value in result.params.items()},
            "loglik": float(result.loglik),
            "aic": float(result.aic),
            "delta_aic": float(result.aic - best_aic),
            "aicc": float(result.aicc) if np.isfinite(result.aicc) else None,
            "delta_aicc": (
                float(result.aicc - best_aicc)
                if best_aicc is not None and np.isfinite(result.aicc)
                else None
            ),
            "akaike_weight": float(aic_weights.get(result.family, 0.0)),
            "aicc_weight": float(aicc_weights.get(result.family, 0.0)),
            "evidence_ratio": (
                float(best_weight / aic_weights[result.family])
                if aic_weights.get(result.family, 0.0) > 0.0
                else float("inf")
            ),
            "bic": float(result.bic),
            "ks_statistic": float(result.ks_statistic),
            "ks_pvalue": float(result.ks_pvalue),
        }
        for result in results
    ]
    return payload


def _model_selection_bootstrap(
    sample: FloatArray,
    *,
    families: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Return deterministic bootstrap winner stability for model selection."""
    payload: dict[str, object] = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "criterion": "aic",
        "available": False,
    }
    if sample.size < 3:
        payload["reason"] = "need >=3 positive events"
        return payload
    if n_boot <= 0:
        payload["reason"] = "n_boot must be positive"
        return payload

    rng = np.random.default_rng(seed)
    counts = dict.fromkeys(families, 0)
    successful = 0
    for _ in range(int(n_boot)):
        draw = sample[rng.integers(0, sample.size, size=sample.size)]
        results = distributions.compare_distributions(draw, families=families)
        if not results:
            continue
        counts[results[0].family] = counts.get(results[0].family, 0) + 1
        successful += 1

    if successful == 0:
        payload["reason"] = "no bootstrap distribution fits"
        return payload

    frequencies = {
        family: float(count / successful) for family, count in counts.items() if count > 0
    }
    top_family = max(frequencies, key=lambda family: frequencies[family])
    payload.update(
        {
            "available": True,
            "successful_bootstraps": int(successful),
            "winner_counts": {family: int(count) for family, count in counts.items() if count > 0},
            "winner_frequencies": frequencies,
            "top_family": top_family,
            "top_frequency": float(frequencies[top_family]),
            "unique_winner_count": len(frequencies),
            "method_note": (
                "Bootstrap resamples refit the same candidate set and record the "
                "lowest-AIC winner; frequencies describe ranking stability, not "
                "posterior model probabilities."
            ),
        }
    )
    return payload


def _amplitude_shape_payload(
    sample: FloatArray,
    *,
    n_boot: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    """JSON-friendly robust shape summary for positive saccade amplitudes."""
    payload: dict[str, object] = {
        "sample": "saccade_amplitude_deg",
        "n": int(sample.size),
        "bootstrap_n": int(n_boot),
        "bootstrap_seed": int(seed),
        "confidence": float(confidence),
        "available": False,
    }
    if sample.size == 0:
        payload["reason"] = "need >=1 positive event"
        return payload

    mean = float(np.mean(sample))
    median = float(np.median(sample))
    q25, q75 = np.percentile(sample, [25.0, 75.0])
    iqr = float(q75 - q25)
    std = float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0
    lower = float(q25 - 1.5 * iqr)
    upper = float(q75 + 1.5 * iqr)
    outliers = (sample < lower) | (sample > upper)
    bowley = float((q75 + q25 - 2.0 * median) / iqr) if iqr > 0.0 else 0.0
    interval = _amplitude_shape_bootstrap(sample, n_boot=n_boot, seed=seed, confidence=confidence)

    payload.update(
        {
            "available": True,
            "mean": mean,
            "median": median,
            "std": std,
            "min": float(np.min(sample)),
            "max": float(np.max(sample)),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": iqr,
            "coefficient_of_variation": float(std / mean) if mean > 0.0 else 0.0,
            "mean_median_gap": float(mean - median),
            "bowley_skewness": bowley,
            "iqr_outlier_lower": lower,
            "iqr_outlier_upper": upper,
            "iqr_outlier_count": int(np.count_nonzero(outliers)),
            "iqr_outlier_fraction": float(np.count_nonzero(outliers) / sample.size),
            **interval,
            "method_note": (
                "Quartile/IQR descriptors summarize amplitude shape without "
                "assuming a fitted distribution; percentile bootstrap intervals "
                "are descriptive finite-sample uncertainty summaries."
            ),
        }
    )
    return payload


def _amplitude_shape_bootstrap(
    sample: FloatArray,
    *,
    n_boot: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    """Return deterministic percentile-bootstrap intervals for median and IQR."""
    if sample.size == 0 or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = sample[rng.integers(0, sample.size, size=(int(n_boot), sample.size))]
    medians = np.median(draws, axis=1)
    quartiles = np.percentile(draws, [25.0, 75.0], axis=1)
    iqrs = quartiles[1] - quartiles[0]
    alpha = 1.0 - confidence
    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    median_lo, median_hi = np.percentile(medians, [lo_pct, hi_pct])
    iqr_lo, iqr_hi = np.percentile(iqrs, [lo_pct, hi_pct])
    return {
        "median_ci_low": float(median_lo),
        "median_ci_high": float(median_hi),
        "iqr_ci_low": float(iqr_lo),
        "iqr_ci_high": float(iqr_hi),
    }


def _amplitude_quantile_payload(
    sample: FloatArray,
    *,
    families: tuple[str, ...],
) -> dict[str, object]:
    """JSON-friendly QQ/probability-plot diagnostics for the best amplitude fit."""
    payload: dict[str, object] = {
        "sample": "saccade_amplitude_deg",
        "n": int(sample.size),
        "available": False,
        "plotting_position": "(i - 0.5) / n",
    }
    if sample.size < 3:
        payload["reason"] = "need >=3 positive events"
        return payload

    results = distributions.compare_distributions(sample, families=families)
    if not results:
        payload["reason"] = "no stable distribution fits"
        return payload

    best = results[0]
    frozen = distributions.frozen_from_result(best)
    probabilities = (np.arange(sample.size, dtype=np.float64) + 0.5) / float(sample.size)
    empirical = np.sort(sample)
    fitted = np.asarray(frozen.ppf(probabilities), dtype=np.float64)
    mask = np.isfinite(empirical) & np.isfinite(fitted)
    if (
        int(np.count_nonzero(mask)) < 3
        or np.unique(empirical[mask]).size < 2
        or np.unique(fitted[mask]).size < 2
    ):
        payload["reason"] = "best-fit quantiles are degenerate"
        return payload

    probabilities = probabilities[mask]
    empirical = empirical[mask]
    fitted = fitted[mask]
    residual = empirical - fitted
    slope, intercept = np.polyfit(fitted, empirical, deg=1)
    central_mask = (probabilities >= 0.10) & (probabilities <= 0.90)
    central_residual = residual[central_mask] if np.any(central_mask) else residual
    corr = float(np.corrcoef(fitted, empirical)[0, 1])

    payload.update(
        {
            "available": True,
            "family": best.family,
            "n": int(empirical.size),
            "probabilities": [float(value) for value in probabilities],
            "empirical_quantiles_deg": [float(value) for value in empirical],
            "fitted_quantiles_deg": [float(value) for value in fitted],
            "residuals_deg": [float(value) for value in residual],
            "residual_rmse_deg": float(np.sqrt(np.mean(residual**2))),
            "central_80_rmse_deg": float(np.sqrt(np.mean(central_residual**2))),
            "median_abs_residual_deg": float(np.median(np.abs(residual))),
            "max_abs_residual_deg": float(np.max(np.abs(residual))),
            "qq_slope": float(slope),
            "qq_intercept": float(intercept),
            "qq_correlation": corr,
            "method_note": (
                "Empirical order statistics are compared with fitted quantiles "
                "from the lowest-AIC family; this is graphical model adequacy, "
                "not a population goodness proof."
            ),
        }
    )
    return payload


def _amplitude_cdf_payload(
    sample: FloatArray,
    *,
    families: tuple[str, ...],
    alpha: float = 0.05,
) -> dict[str, object]:
    """JSON-friendly empirical-CDF residual diagnostics for the best amplitude fit."""
    payload: dict[str, object] = {
        "sample": "saccade_amplitude_deg",
        "n": int(sample.size),
        "available": False,
        "empirical_cdf_position": "(i - 0.5) / n",
        "dkw_alpha": float(alpha),
    }
    if not 0.0 < alpha < 1.0:
        payload["reason"] = "DKW alpha must be between 0 and 1"
        return payload
    if sample.size < 3:
        payload["reason"] = "need >=3 positive events"
        return payload

    results = distributions.compare_distributions(sample, families=families)
    if not results:
        payload["reason"] = "no stable distribution fits"
        return payload

    best = results[0]
    frozen = distributions.frozen_from_result(best)
    empirical = np.sort(sample)
    empirical_cdf = (np.arange(sample.size, dtype=np.float64) + 0.5) / float(sample.size)
    fitted_cdf = np.asarray(frozen.cdf(empirical), dtype=np.float64)
    mask = np.isfinite(empirical) & np.isfinite(empirical_cdf) & np.isfinite(fitted_cdf)
    if int(np.count_nonzero(mask)) < 3:
        payload["reason"] = "best-fit CDF residuals are degenerate"
        return payload

    empirical = empirical[mask]
    empirical_cdf = empirical_cdf[mask]
    fitted_cdf = fitted_cdf[mask]
    residual = empirical_cdf - fitted_cdf
    abs_residual = np.abs(residual)
    dkw_epsilon = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * empirical.size)))
    max_abs = float(np.max(abs_residual))
    tail_mask_low = empirical_cdf <= 0.20
    tail_mask_high = empirical_cdf >= 0.80
    lower_tail_max = float(np.max(abs_residual[tail_mask_low])) if np.any(tail_mask_low) else 0.0
    upper_tail_max = float(np.max(abs_residual[tail_mask_high])) if np.any(tail_mask_high) else 0.0
    cramer_von_mises = float(
        (1.0 / (12.0 * empirical.size)) + np.sum((fitted_cdf - empirical_cdf) ** 2)
    )
    clipped_fitted = np.clip(fitted_cdf, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    indices = np.arange(1, empirical.size + 1, dtype=np.float64)
    anderson_darling = float(
        -empirical.size
        - np.mean(
            (2.0 * indices - 1.0) * (np.log(clipped_fitted) + np.log1p(-clipped_fitted[::-1]))
        )
    )

    payload.update(
        {
            "available": True,
            "family": best.family,
            "n": int(empirical.size),
            "amplitude_deg": [float(value) for value in empirical],
            "empirical_cdf": [float(value) for value in empirical_cdf],
            "fitted_cdf": [float(value) for value in fitted_cdf],
            "cdf_residuals": [float(value) for value in residual],
            "cdf_rmse": float(np.sqrt(np.mean(residual**2))),
            "mean_abs_cdf_residual": float(np.mean(abs_residual)),
            "max_abs_cdf_residual": max_abs,
            "lower_tail_max_abs_cdf_residual": lower_tail_max,
            "upper_tail_max_abs_cdf_residual": upper_tail_max,
            "tail_max_abs_cdf_residual": max(lower_tail_max, upper_tail_max),
            "cramer_von_mises_statistic": cramer_von_mises,
            "anderson_darling_statistic": anderson_darling,
            "fit_ks_statistic": float(best.ks_statistic),
            "dkw_epsilon": dkw_epsilon,
            "dkw_confidence": float(1.0 - alpha),
            "exceeds_dkw_band": bool(max_abs > dkw_epsilon),
            "method_note": (
                "Empirical-CDF plotting positions are compared with fitted CDF "
                "values from the lowest-AIC family; the integrated and "
                "tail-weighted residual statistics add descriptive fit diagnostics, "
                "and the DKW/Massart band is a distribution-free reference width "
                "for an ECDF around a fixed CDF. Because the family parameters "
                "were fitted on the same sample, these fields are descriptive "
                "rather than formal acceptance tests."
            ),
        }
    )
    return payload


def _main_sequence_payload(
    amplitude: FloatArray,
    velocity: FloatArray,
    *,
    n_boot: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    """JSON-friendly bootstrap interval payload for the main-sequence exponent."""
    payload: dict[str, object] = {
        "n": int(amplitude.size),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "confidence": float(confidence),
        "available": False,
    }
    if amplitude.size < 3 or np.unique(amplitude).size < 3:
        payload["reason"] = "need >=3 unique saccades"
        return payload
    try:
        estimate, lo, hi = scanpath_metrics.main_sequence_exponent_ci(
            amplitude,
            velocity,
            n_boot=n_boot,
            seed=seed,
            confidence=confidence,
        )
    except ValueError as exc:
        payload["reason"] = str(exc)
        return payload

    payload.update(
        {
            "available": True,
            "estimate": float(estimate),
            "ci_low": float(lo),
            "ci_high": float(hi),
        }
    )
    return payload


def session_statistical_diagnostics(
    report: SessionReport,
    *,
    n_boot: int = 500,
    model_boot: int = 40,
    seed: int = 2401,
    confidence: float = 0.95,
    cdf_band_alpha: float = 0.05,
    families: tuple[str, ...] = DEFAULT_DIAGNOSTIC_FAMILIES,
) -> dict[str, Any]:
    """Return the audited statistical diagnostics payload for a session report.

    Parameters
    ----------
    report
        Analysed session whose detected events are summarised.
    n_boot
        Bootstrap resamples for the main-sequence exponent interval.
    model_boot
        Bootstrap resamples for distribution-family winner stability.
    seed
        Deterministic bootstrap seed.
    confidence
        Two-sided confidence level for the exponent interval.
    cdf_band_alpha
        Tail probability for the distribution-free DKW/Massart ECDF reference
        band reported in ``amplitude_cdf_diagnostics``.
    families
        Distribution families compared on the positive amplitude sample.

    Returns
    -------
    dict
        JSON-friendly payload consumed by the publication statistics figure and
        sidecar artifact.
    """
    props = saccades.saccade_properties(report.saccades)
    amplitude = _positive_sample(props["amplitude_deg"])
    amp_pair, vel_pair = _positive_pairs(props["amplitude_deg"], props["peak_velocity_deg_s"])
    symbols, matrix = similarity.transition_matrix(report.saccades)

    return {
        "kind": "itrace_statistical_diagnostics",
        "truth_boundary": (
            "Descriptive statistics over a supplied SessionReport; not population physiology, "
            "reference-device validation, or biological ground truth."
        ),
        "n_samples": int(report.n_samples),
        "duration_s": float(report.duration_s),
        "n_fixations": len(report.fixations),
        "n_saccades": len(report.saccades),
        "amplitude_model_comparison": _model_comparison_payload(
            amplitude,
            families=families,
            n_boot=model_boot,
            seed=seed + 202,
        ),
        "amplitude_shape_summary": _amplitude_shape_payload(
            amplitude,
            n_boot=n_boot,
            seed=seed + 101,
            confidence=confidence,
        ),
        "amplitude_quantile_diagnostics": _amplitude_quantile_payload(
            amplitude,
            families=families,
        ),
        "amplitude_cdf_diagnostics": _amplitude_cdf_payload(
            amplitude,
            families=families,
            alpha=cdf_band_alpha,
        ),
        "main_sequence_exponent": _main_sequence_payload(
            amp_pair,
            vel_pair,
            n_boot=n_boot,
            seed=seed,
            confidence=confidence,
        ),
        "spatial_stability": scanpath_metrics.scanpath_summary(report),
        "transition_matrix": {
            "symbols": symbols,
            "matrix": matrix.tolist(),
            "scanpath": report.scanpath,
        },
    }
