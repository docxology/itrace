"""Spatial-spread and scanpath metrics.

This module collects descriptive statistics that summarise *where* and *how*
gaze moved during a recording, complementing the event-level detection in
:mod:`itrace.saccades`:

* **Spatial spread** -- :func:`gaze_dispersion` (RMS distance from the
  centroid), :func:`convex_hull_area` (area of the explored region) and
  :func:`bcea` (the bivariate contour ellipse area, a standard fixation-
  stability measure).
* **Distributional structure** -- :func:`shannon_entropy` and the two derived
  entropies :func:`fixation_position_entropy` (spatial spread of fixation
  centroids over a grid) and :func:`direction_transition_entropy` (predictability
  of the next saccade direction given the current one).
* **Uncertainty** -- :func:`main_sequence_exponent_ci`, a bootstrap-percentile
  confidence interval for the main-sequence power-law exponent.

Everything is pure NumPy/SciPy and deterministic: the only randomness lives in
:func:`main_sequence_exponent_ci`, which takes an explicit ``seed``.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .. import mainsequence
from ..encoding import encode_directions
from ..types import Fixation, FloatArray, Saccade, SessionReport
from .bootstrap import percentile_interval


def _finite_xy(
    xs: FloatArray | list[float],
    ys: FloatArray | list[float],
) -> tuple[FloatArray, FloatArray]:
    """Return aligned finite x/y samples, preserving original order."""
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.shape != y.shape:
        msg = f"xs and ys must have equal shape; got {x.shape} vs {y.shape}"
        raise ValueError(msg)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def shannon_entropy(probabilities: FloatArray | list[float], base: float = 2.0) -> float:
    """Shannon entropy of a (possibly unnormalised) distribution.

    Zero-probability entries contribute zero (``0 * log 0 := 0``). The input is
    normalised defensively, so raw counts may be passed directly.

    Parameters
    ----------
    probabilities:
        Non-negative weights or probabilities. May be empty.
    base:
        Logarithm base; ``2.0`` (the default) yields entropy in bits.

    Returns
    -------
    float
        The entropy ``-sum p_i log_base(p_i)``. ``0.0`` for an empty input or a
        distribution with no positive mass.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    total = float(np.sum(p))
    if total <= 0.0:
        return 0.0
    p = p / total
    nz = p[p > 0.0]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * (np.log(nz) / np.log(base))))


def gaze_dispersion(xs: FloatArray | list[float], ys: FloatArray | list[float]) -> float:
    """Root-mean-square distance of points from their centroid.

    Parameters
    ----------
    xs, ys:
        Equal-length coordinate arrays. May be empty.

    Returns
    -------
    float
        ``sqrt(mean((x - xbar)^2 + (y - ybar)^2))``. ``0.0`` for an empty input.
    """
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0
    dx = x - float(np.mean(x))
    dy = y - float(np.mean(y))
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def gaze_path_length(xs: FloatArray | list[float], ys: FloatArray | list[float]) -> float:
    """Total finite point-to-point gaze-path length in degrees.

    Non-finite samples are dropped before consecutive distances are computed.
    This gives a descriptive path length over observed gaze, not an imputation
    through missing data.
    """
    x, y = _finite_xy(xs, ys)
    if x.size < 2:
        return 0.0
    steps = np.hypot(np.diff(x), np.diff(y))
    return float(np.sum(steps))


def gaze_path_efficiency(xs: FloatArray | list[float], ys: FloatArray | list[float]) -> float:
    """Straight-line displacement divided by total finite gaze-path length.

    The result is clipped to ``[0, 1]`` for numerical robustness. It is ``0.0``
    when there are fewer than two finite samples or the path length is zero.
    """
    x, y = _finite_xy(xs, ys)
    if x.size < 2:
        return 0.0
    path = gaze_path_length(x, y)
    if path <= 0.0:
        return 0.0
    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    return float(np.clip(displacement / path, 0.0, 1.0))


def convex_hull_area(xs: FloatArray | list[float], ys: FloatArray | list[float]) -> float:
    """Area of the convex hull of a 2-D point cloud.

    Uses :class:`scipy.spatial.ConvexHull`, whose ``volume`` attribute is the
    enclosed area in two dimensions.

    Parameters
    ----------
    xs, ys:
        Equal-length coordinate arrays.

    Returns
    -------
    float
        The hull area, or ``0.0`` when fewer than three points are supplied or
        the points are degenerate/collinear (so no 2-D hull exists).
    """
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size < 3 or y.size < 3:
        return 0.0
    points = np.column_stack((x, y))
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return 0.0
    return float(hull.volume)


def bcea(
    xs: FloatArray | list[float],
    ys: FloatArray | list[float],
    probability: float = 0.68,
) -> float:
    """Bivariate contour ellipse area (BCEA).

    The BCEA is the area of the ellipse that contains a given probability mass
    of a bivariate-normal fit to the point cloud, a standard fixation-stability
    measure::

        BCEA = 2 * k * pi * sx * sy * sqrt(1 - rho^2),   k = -ln(1 - probability)

    where ``sx``/``sy`` are the per-axis standard deviations (``ddof=1``) and
    ``rho`` the Pearson correlation, clipped to ``[-1, 1]``.

    Parameters
    ----------
    xs, ys:
        Equal-length coordinate arrays.
    probability:
        Enclosed probability mass, strictly within ``(0, 1)``.

    Returns
    -------
    float
        The contour-ellipse area, or ``0.0`` when fewer than two points are
        supplied or either axis has zero variance.

    Raises
    ------
    ValueError
        If ``probability`` is not in the open interval ``(0, 1)``.
    """
    if not 0.0 < probability < 1.0:
        msg = f"probability must be in (0, 1); got {probability}"
        raise ValueError(msg)
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    rho = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
    k = -np.log(1.0 - probability)
    return float(2.0 * k * np.pi * sx * sy * np.sqrt(1.0 - rho**2))


def fixation_position_entropy(
    fixations: list[Fixation],
    grid: tuple[int, int] = (4, 4),
    extent: tuple[float, float, float, float] | None = None,
    base: float = 2.0,
) -> float:
    """Shannon entropy of fixation centroids binned over a spatial grid.

    Fixation centroids are histogrammed into a ``grid`` of cells spanning
    ``[xmin, xmax] x [ymin, ymax]`` (taken from ``extent`` when given, else from
    the data range) and the entropy of the resulting cell-count distribution is
    returned. High entropy means gaze was spread evenly over the scene; low
    entropy means it concentrated in a few regions.

    Parameters
    ----------
    fixations:
        Detected fixations whose centroids are binned.
    grid:
        ``(n_x, n_y)`` number of cells along each axis.
    extent:
        Optional ``(xmin, xmax, ymin, ymax)`` bounds; inferred from the data
        when ``None``.
    base:
        Logarithm base for the entropy (``2.0`` -> bits).

    Returns
    -------
    float
        Entropy of the cell-count distribution; ``0.0`` when there are no
        fixations.
    """
    n_x, n_y = grid
    if n_x < 1 or n_y < 1:
        msg = f"grid cells must be >= 1 along each axis; got {grid}"
        raise ValueError(msg)
    if not fixations:
        return 0.0
    xs = np.array([f.centroid_x for f in fixations], dtype=np.float64)
    ys = np.array([f.centroid_y for f in fixations], dtype=np.float64)
    if extent is None:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
    else:
        xmin, xmax, ymin, ymax = extent
    # Degenerate range along an axis: pad so histogram2d gets a valid interval.
    if xmax <= xmin:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymax <= ymin:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    counts, _, _ = np.histogram2d(xs, ys, bins=(n_x, n_y), range=((xmin, xmax), (ymin, ymax)))
    return shannon_entropy(counts.ravel(), base=base)


def direction_transition_entropy(
    saccades: list[Saccade],
    long_threshold_deg: float = 5.0,
    base: float = 2.0,
) -> float:
    """Conditional entropy ``H(next | current)`` of saccade-direction symbols.

    The saccade list is encoded to a direction string via
    :func:`itrace.encoding.encode_directions`; the empirical first-order
    transition matrix over its characters yields the conditional entropy. Low
    values mean the next saccade direction is highly predictable from the
    current one.

    Parameters
    ----------
    saccades:
        Saccades to encode and analyse.
    long_threshold_deg:
        Amplitude threshold passed to the encoder (long vs. short casing).
    base:
        Logarithm base for the entropy (``2.0`` -> bits).

    Returns
    -------
    float
        Weighted-average conditional entropy ``sum_c p(c) H(next | c)``;
        ``0.0`` when there are fewer than two saccades.
    """
    if len(saccades) < 2:
        return 0.0
    seq = encode_directions(saccades, long_threshold_deg=long_threshold_deg)
    if len(seq) < 2:
        return 0.0
    symbols = sorted(set(seq))
    index = {ch: i for i, ch in enumerate(symbols)}
    k = len(symbols)
    matrix = np.zeros((k, k), dtype=np.float64)
    for current, nxt in pairwise(seq):
        matrix[index[current], index[nxt]] += 1.0
    row_totals = matrix.sum(axis=1)
    grand_total = float(row_totals.sum())
    if grand_total <= 0.0:
        return 0.0
    conditional = 0.0
    for i in range(k):
        if row_totals[i] <= 0.0:
            continue
        weight = float(row_totals[i]) / grand_total
        conditional += weight * shannon_entropy(matrix[i], base=base)
    return conditional


def main_sequence_exponent_ci(
    amp: FloatArray,
    vel: FloatArray,
    *,
    n_boot: int = 2000,
    seed: int = 12345,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap-percentile confidence interval for the main-sequence exponent.

    The point estimate is the power-law exponent ``power_b`` from
    :func:`itrace.mainsequence.fit`. ``n_boot`` resamples of the (amplitude,
    velocity) index set are refit, and the percentile interval of the resampled
    exponents at ``confidence`` is returned.

    Parameters
    ----------
    amp, vel:
        Equal-length saccade amplitude (deg) and peak-velocity (deg/s) arrays.
    n_boot:
        Number of bootstrap resamples (must be >= 1).
    seed:
        Seed for :func:`numpy.random.default_rng`.
    confidence:
        Two-sided confidence level, strictly within ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(b, lo, hi)`` -- the point estimate and the lower/upper percentile
        bounds. When no resample yields a valid fit, ``lo`` and ``hi`` both
        collapse to ``b``.

    Raises
    ------
    ValueError
        If ``n_boot < 1`` or ``confidence`` is not in ``(0, 1)``.
    """
    if n_boot < 1:
        msg = f"n_boot must be >= 1; got {n_boot}"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1); got {confidence}"
        raise ValueError(msg)
    amp_arr = np.asarray(amp, dtype=np.float64).ravel()
    vel_arr = np.asarray(vel, dtype=np.float64).ravel()
    b = float(mainsequence.fit(amp_arr, vel_arr)["power_b"])

    rng = np.random.default_rng(seed)
    n = amp_arr.shape[0]
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(amp_arr[idx]).size < 3:
            continue
        try:
            fit_b = float(mainsequence.fit(amp_arr[idx], vel_arr[idx])["power_b"])
        except ValueError:
            continue
        samples.append(fit_b)
    if not samples:
        return b, b, b
    boot = np.asarray(samples, dtype=np.float64)
    lo, hi = percentile_interval(boot, confidence=confidence)
    return b, lo, hi


def scanpath_summary(
    report_or_fixations: SessionReport | list[Fixation],
    saccades: list[Saccade] | None = None,
) -> dict[str, float]:
    """Bundle the scanpath metrics into a single summary dict.

    Accepts either a :class:`~itrace.types.SessionReport` (in which case its
    ``fixations`` and ``saccades`` are used) or an explicit fixation list plus a
    ``saccades`` keyword/positional argument.

    Parameters
    ----------
    report_or_fixations:
        A :class:`SessionReport`, or a list of :class:`Fixation` when
        ``saccades`` is supplied explicitly.
    saccades:
        Saccade list, required when the first argument is a fixation list and
        ignored when it is a :class:`SessionReport`.

    Returns
    -------
    dict of str to float
        ``gaze_dispersion``, ``convex_hull_area``, ``bcea``,
        ``fixation_position_entropy`` and ``direction_transition_entropy``.

    Raises
    ------
    ValueError
        If a fixation list is passed without a ``saccades`` argument.
    """
    if isinstance(report_or_fixations, SessionReport):
        fixations = report_or_fixations.fixations
        sacc = report_or_fixations.saccades
    else:
        fixations = report_or_fixations
        if saccades is None:
            msg = "saccades must be provided when passing a fixation list"
            raise ValueError(msg)
        sacc = saccades

    xs = [f.centroid_x for f in fixations]
    ys = [f.centroid_y for f in fixations]
    return {
        "gaze_dispersion": gaze_dispersion(xs, ys),
        "gaze_path_length": gaze_path_length(xs, ys),
        "gaze_path_efficiency": gaze_path_efficiency(xs, ys),
        "convex_hull_area": convex_hull_area(xs, ys),
        "bcea": bcea(xs, ys),
        "fixation_position_entropy": fixation_position_entropy(fixations),
        "direction_transition_entropy": direction_transition_entropy(sacc),
    }


def raw_gaze_spatial_summary(
    xs: FloatArray | list[float],
    ys: FloatArray | list[float],
) -> dict[str, float]:
    """Summarise raw finite gaze coordinates for live/session reporting.

    Returns sample count, path length, path efficiency, dispersion, hull area,
    and BCEA in one JSON-friendly block. Non-finite x/y pairs are dropped
    consistently across all metrics.
    """
    x, y = _finite_xy(xs, ys)
    return {
        "finite_sample_count": float(x.size),
        "path_length_deg": gaze_path_length(x, y),
        "path_efficiency": gaze_path_efficiency(x, y),
        "dispersion_deg": gaze_dispersion(x, y),
        "convex_hull_area_deg2": convex_hull_area(x, y),
        "bcea_deg2": bcea(x, y),
    }
