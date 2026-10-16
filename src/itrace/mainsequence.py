"""Saccadic main-sequence fitting.

The *main sequence* is the lawful relationship between saccade amplitude and
peak velocity. Two standard parameterisations are fitted:

* saturating exponential ``V = V_max * (1 - exp(-A / C))`` (Bahill et al., 1975);
* power law ``V = a * A^b`` (linear in log-log space).

The exponent depends on the task, amplitude range, sampling, and fit convention;
this descriptive fit is not a clinical classification.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .types import FloatArray


def _saturating(amplitude: FloatArray, v_max: float, c: float) -> FloatArray:
    # clip the exponent to avoid overflow when the optimiser probes extreme c
    expo = np.clip(-amplitude / c, -700.0, 700.0)
    return v_max * (1.0 - np.exp(expo))


def fit(amplitudes_deg: FloatArray, peak_velocities_deg_s: FloatArray) -> dict[str, float]:
    """Fit both main-sequence models and return their parameters.

    Returns a dict with ``v_max``, ``C`` (saturating model), ``power_a``,
    ``power_b`` and ``r_squared_power``.

    Raises
    ------
    ValueError
        If fewer than three valid (positive) points are supplied.
    """
    amp, vel = _positive_pairs(amplitudes_deg, peak_velocities_deg_s)
    power = fit_power_law(amp, vel)

    # Saturating exponential.
    p0 = (float(np.max(vel)), float(np.median(amp)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(_saturating, amp, vel, p0=p0, bounds=(0.0, np.inf), maxfev=10000)
        v_max, c = float(popt[0]), float(popt[1])
    except (OptimizeWarning, RuntimeError, ValueError) as exc:
        raise ValueError("saturating main-sequence fit did not converge") from exc
    if not np.all(np.isfinite(popt)):
        raise ValueError("saturating main-sequence fit returned nonfinite parameters")
    return {"v_max": v_max, "C": c, **power}


def _positive_pairs(amp: FloatArray, vel: FloatArray) -> tuple[FloatArray, FloatArray]:
    a = np.asarray(amp, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    if a.ndim != 1 or v.ndim != 1 or a.shape != v.shape:
        raise ValueError("amplitude and peak-velocity arrays must match and be one-dimensional")
    valid = np.isfinite(a) & np.isfinite(v) & (a > 0) & (v > 0)
    a, v = a[valid], v[valid]
    if a.size < 3:
        raise ValueError(f"main-sequence fit needs >=3 positive points; got {a.size}")
    if np.unique(a).size < 2:
        raise ValueError("power-law fit requires distinct amplitudes")
    return a, v


def fit_power_law(
    amplitudes_deg: FloatArray, peak_velocities_deg_s: FloatArray
) -> dict[str, float]:
    """Fit only log velocity on log amplitude, without a nonlinear surrogate.

    Returns ``power_a``, ``power_b``, and log-space ``r_squared_power``. At least
    three finite positive pairs and two distinct amplitudes are required.
    """
    amp, vel = _positive_pairs(amplitudes_deg, peak_velocities_deg_s)

    # Power law via log-log linear regression.
    log_a, log_v = np.log(amp), np.log(vel)
    b, log_intercept = np.polyfit(log_a, log_v, 1)
    pred = log_intercept + b * log_a
    ss_res = float(np.sum((log_v - pred) ** 2))
    ss_tot = float(np.sum((log_v - np.mean(log_v)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "power_a": float(np.exp(log_intercept)),
        "power_b": float(b),
        "r_squared_power": r_squared,
    }
