"""Saccadic main-sequence fitting.

The *main sequence* is the lawful relationship between saccade amplitude and
peak velocity. Two standard parameterisations are fitted:

* saturating exponential ``V = V_max * (1 - exp(-A / C))`` (Bahill et al., 1975);
* power law ``V = a * A^b`` (linear in log-log space).

A healthy oculomotor system gives a power-law exponent ``b`` of roughly
0.4-0.9. Deviations are a recognised clinical/behavioural marker.
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
    amp = np.asarray(amplitudes_deg, dtype=np.float64)
    vel = np.asarray(peak_velocities_deg_s, dtype=np.float64)
    if amp.shape != vel.shape:
        msg = "amplitude and peak-velocity arrays must match"
        raise ValueError(msg)
    valid = np.isfinite(amp) & np.isfinite(vel) & (amp > 0) & (vel > 0)
    amp, vel = amp[valid], vel[valid]
    if amp.size < 3:
        msg = f"main-sequence fit needs >=3 positive points; got {amp.size}"
        raise ValueError(msg)

    # Saturating exponential.
    p0 = (float(np.max(vel)), float(np.median(amp)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(_saturating, amp, vel, p0=p0, maxfev=10000)
        v_max, c = float(popt[0]), float(abs(popt[1]))
    except (OptimizeWarning, RuntimeError, ValueError):  # pragma: no cover - rare non-convergence
        v_max, c = float(np.max(vel)), float(np.median(amp))

    # Power law via log-log linear regression.
    log_a, log_v = np.log(amp), np.log(vel)
    b, log_intercept = np.polyfit(log_a, log_v, 1)
    pred = log_intercept + b * log_a
    ss_res = float(np.sum((log_v - pred) ** 2))
    ss_tot = float(np.sum((log_v - np.mean(log_v)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "v_max": v_max,
        "C": c,
        "power_a": float(np.exp(log_intercept)),
        "power_b": float(b),
        "r_squared_power": r_squared,
    }
