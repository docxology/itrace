"""Bootstrap utilities shared by statistics and recovery analyses."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from ..types import FloatArray


def bootstrap_indices(
    n: int, *, n_boot: int = 2000, seed: int = 12345, block_length: int = 1
) -> Iterator[NDArray[np.int64]]:
    """Yield IID indices or overlapping moving blocks, truncated to ``n``.

    Blocks never wrap across record boundaries. A block length greater than one
    assumes ordered, approximately stationary observations; it does not resample
    participants or repair gaps. Choose the length from the scientific design
    and examine sensitivity. Length ``n`` yields no resampling variability.
    """
    for name, value in (("n", n), ("n_boot", n_boot), ("block_length", block_length)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}")
    if block_length > n:
        raise ValueError("block_length must not exceed the observation count")
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_length, dtype=np.int64)
    n_blocks = (n + block_length - 1) // block_length
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks, dtype=np.int64)
        yield (starts[:, None] + offsets).ravel()[:n]


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
    block_length: int = 1,
) -> FloatArray:
    """Evaluate a finite scalar statistic on IID or moving-block resamples.

    Data must be finite and one-dimensional. Missing rows must be handled
    explicitly before resampling; dropping them here would change adjacency.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError("bootstrap_statistic needs finite one-dimensional data")
    if arr.size == 0:
        msg = "bootstrap_statistic needs at least one value"
        raise ValueError(msg)
    estimates = []
    for idx in bootstrap_indices(arr.size, n_boot=n_boot, seed=seed, block_length=block_length):
        estimate = float(statistic(arr[idx]))
        if not np.isfinite(estimate):
            raise ValueError("bootstrap statistic returned a nonfinite estimate")
        estimates.append(estimate)
    return np.asarray(estimates, dtype=np.float64)


def resampling_sensitivity(
    *, n: int = 256, rho: float = 0.85, n_boot: int = 1000, seed: int = 2401
) -> dict[str, object]:
    """Illustrate block-length sensitivity on one stationary Gaussian AR(1) record.

    Innovations have unit variance; the initial value has stationary variance.
    The analytic mean SE uses the finite-record covariance sum. This is not a
    coverage experiment or a block-length selector; units are arbitrary.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 32:
        raise ValueError("n must be an integer >= 32")
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("rho must be finite and in (-1, 1)")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError("n_boot must be an integer >= 2")
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    values[0] = rng.normal(scale=1.0 / np.sqrt(1.0 - rho**2))
    for i, innovation in enumerate(rng.normal(size=n - 1), start=1):
        values[i] = rho * values[i - 1] + innovation
    lags = np.arange(1, n, dtype=np.float64)
    variance = (n + 2.0 * np.sum((n - lags) * rho**lags)) / (n**2 * (1.0 - rho**2))
    rows = []
    for length in (1, 4, 16, 32):
        draws = bootstrap_statistic(
            values,
            lambda arr: float(np.mean(arr)),
            n_boot=n_boot,
            seed=seed + length,
            block_length=length,
        )
        lo, hi = percentile_interval(draws)
        rows.append(
            {
                "block_length": length,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_se": float(np.std(draws, ddof=1)),
            }
        )
    return {
        "kind": "itrace_resampling_sensitivity",
        "seed": seed,
        "n": n,
        "n_boot": n_boot,
        "rho": rho,
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "generating_mean": 0.0,
        "analytic_mean_se": float(np.sqrt(variance)),
        "intervals": rows,
        "confidence": 0.95,
        "units": "arbitrary",
        "boundary": "One synthetic stationary AR(1) record; sensitivity illustration, "
        "not empirical eye data, interval coverage validation, or a block-length recommendation.",
    }
