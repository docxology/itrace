"""Velocity estimation from position time-series.

Two paths, matching the two ways gaze data arrives:

* uniform sampling -> Savitzky-Golay differentiation (pymovements ``pos2vel``
  default; smooths while differentiating, robust to per-sample noise);
* non-uniform timestamps -> central-difference gradient on the actual ``t``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import savgol_filter

from .types import FloatArray


def _odd_window(window: int, n: int) -> int:
    """Largest valid odd Savitzky-Golay window <= ``window`` and < ``n``."""
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def pos2vel_savgol(
    position: FloatArray,
    sampling_rate_hz: float,
    window: int = 7,
    polyorder: int = 2,
) -> FloatArray:
    """Differentiate a uniformly-sampled position signal (deg) -> velocity (deg/s).

    The Savitzky-Golay filter fits a local polynomial and returns its analytic
    first derivative, which both smooths and differentiates in one pass.
    """
    pos = np.asarray(position, dtype=np.float64)
    if pos.ndim != 1:
        msg = "position must be 1-D"
        raise ValueError(msg)
    if sampling_rate_hz <= 0:
        msg = "sampling_rate_hz must be positive"
        raise ValueError(msg)
    n = pos.shape[0]
    if n < 3:
        return np.zeros_like(pos)
    win = _odd_window(window, n)
    po = min(polyorder, win - 1)
    dt = 1.0 / sampling_rate_hz
    result: FloatArray = savgol_filter(pos, window_length=win, polyorder=po, deriv=1, delta=dt)
    return result


def pos2vel_gradient(position: FloatArray, t: FloatArray) -> FloatArray:
    """Central-difference velocity for non-uniform timestamps (deg/s)."""
    pos = np.asarray(position, dtype=np.float64)
    ts = np.asarray(t, dtype=np.float64)
    if pos.shape != ts.shape:
        msg = "position and t must have the same shape"
        raise ValueError(msg)
    if pos.shape[0] < 2:
        return np.zeros_like(pos)
    return cast(FloatArray, np.gradient(pos, ts))


def speed_2d(vx: FloatArray, vy: FloatArray) -> FloatArray:
    """Euclidean speed from x/y velocity components."""
    return np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
