"""Bootstrap utilities shared by statistics and recovery analyses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..types import FloatArray


def percentile_interval(
    values: FloatArray | list[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided percentile interval over bootstrap estimates."""
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1); got {confidence}"
        raise ValueError(msg)
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        msg = "percentile_interval needs at least one finite value"
        raise ValueError(msg)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(finite, 100.0 * alpha))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha)))
    return lo, hi


def bootstrap_statistic(
    values: FloatArray | list[float],
    statistic: Callable[[FloatArray], float],
    *,
    n_boot: int = 2000,
    seed: int = 12345,
) -> FloatArray:
    """Resample a one-dimensional array and evaluate ``statistic``."""
    if n_boot < 1:
        msg = f"n_boot must be >= 1; got {n_boot}"
        raise ValueError(msg)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        msg = "bootstrap_statistic needs at least one value"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        out[i] = float(statistic(arr[idx]))
    return out
