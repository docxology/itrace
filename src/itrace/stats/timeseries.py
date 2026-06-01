"""Temporal-rate metrics over oculomotor events.

This module turns event onset times into *rates over time*, complementing the
spatial/distributional summaries in :mod:`itrace.stats.scanpath_metrics`:

* **Binned rates** -- :func:`event_rate` histograms onset times into fixed-width
  bins and divides by the bin width, giving an events-per-second series.
  :func:`fixation_rate_series` and :func:`saccade_rate_series` apply it to the
  ``onset_t`` of detected fixations / saccades.
* **Scalar rates** -- :func:`blink_rate_hz` (blinks per second over the pupil
  span) and :func:`microsaccade_rate_hz` (microsaccades per second over the
  recording).
* **Sliding windows** -- :func:`sliding_window_stat` computes a running
  ``mean``/``std``/``count`` of a value series over overlapping time windows.

Everything is pure NumPy and deterministic (no randomness, no matplotlib).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import pupil as pupil_module
from ..types import Fixation, FloatArray, Microsaccade, PupilStream, Saccade

__all__ = [
    "blink_rate_hz",
    "event_rate",
    "fixation_rate_series",
    "microsaccade_rate_hz",
    "saccade_rate_series",
    "sliding_window_stat",
]


def event_rate(
    onsets_s: Sequence[float] | FloatArray,
    duration_s: float,
    *,
    bin_s: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Binned event rate (events per second) over ``[0, duration_s]``.

    Onset times are histogrammed into contiguous bins of width ``bin_s`` and the
    per-bin counts are divided by ``bin_s`` to give a rate in Hz. The final bin
    is the partial remainder when ``duration_s`` is not a whole multiple of
    ``bin_s``; its rate still uses ``bin_s`` as the denominator. Onsets outside
    ``[0, duration_s]`` are ignored.

    Parameters
    ----------
    onsets_s:
        Event onset times in seconds. May be empty.
    duration_s:
        Total span over which to bin; must be strictly positive.
    bin_s:
        Bin width in seconds; must be strictly positive.

    Returns
    -------
    tuple of FloatArray
        ``(bin_centers, rate)`` -- the centre time of each bin and the
        corresponding rate in events per second. Both have length
        ``ceil(duration_s / bin_s)``.

    Raises
    ------
    ValueError
        If ``duration_s`` or ``bin_s`` is not strictly positive.
    """
    if duration_s <= 0.0:
        msg = f"duration_s must be > 0; got {duration_s}"
        raise ValueError(msg)
    if bin_s <= 0.0:
        msg = f"bin_s must be > 0; got {bin_s}"
        raise ValueError(msg)

    onsets = np.asarray(onsets_s, dtype=np.float64).ravel()
    n_bins = int(np.ceil(duration_s / bin_s))
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_s
    if onsets.size == 0:
        counts = np.zeros(n_bins, dtype=np.float64)
    else:
        in_range = onsets[(onsets >= 0.0) & (onsets <= duration_s)]
        counts = np.histogram(in_range, bins=edges)[0].astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, counts / bin_s


def fixation_rate_series(
    fixations: Sequence[Fixation],
    duration_s: float,
    *,
    bin_s: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Binned fixation rate (fixations per second) over the recording.

    Thin wrapper over :func:`event_rate` applied to fixation ``onset_t``.

    Parameters
    ----------
    fixations:
        Detected fixations.
    duration_s:
        Total span; must be strictly positive.
    bin_s:
        Bin width in seconds; must be strictly positive.

    Returns
    -------
    tuple of FloatArray
        ``(bin_centers, rate)`` as returned by :func:`event_rate`.
    """
    onsets = [f.onset_t for f in fixations]
    return event_rate(onsets, duration_s, bin_s=bin_s)


def saccade_rate_series(
    saccades: Sequence[Saccade],
    duration_s: float,
    *,
    bin_s: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Binned saccade rate (saccades per second) over the recording.

    Thin wrapper over :func:`event_rate` applied to saccade ``onset_t``.

    Parameters
    ----------
    saccades:
        Detected saccades.
    duration_s:
        Total span; must be strictly positive.
    bin_s:
        Bin width in seconds; must be strictly positive.

    Returns
    -------
    tuple of FloatArray
        ``(bin_centers, rate)`` as returned by :func:`event_rate`.
    """
    onsets = [s.onset_t for s in saccades]
    return event_rate(onsets, duration_s, bin_s=bin_s)


def blink_rate_hz(pupil_stream: PupilStream) -> float:
    """Blink rate in blinks per second over the pupil-stream span.

    Blinks are detected with :func:`itrace.pupil.detect_blinks`; the count is
    divided by the recording span ``t[-1] - t[0]``.

    Parameters
    ----------
    pupil_stream:
        Pupil signal to analyse.

    Returns
    -------
    float
        Blinks per second, or ``0.0`` when the span is non-positive (fewer than
        two samples, or a degenerate time axis).
    """
    if len(pupil_stream) < 2:
        return 0.0
    span = float(pupil_stream.t[-1] - pupil_stream.t[0])
    if span <= 0.0:
        return 0.0
    n_blinks = len(pupil_module.detect_blinks(pupil_stream))
    return n_blinks / span


def microsaccade_rate_hz(
    microsaccades: Sequence[Microsaccade],
    duration_s: float,
) -> float:
    """Microsaccade rate in microsaccades per second.

    Parameters
    ----------
    microsaccades:
        Detected microsaccades.
    duration_s:
        Recording span in seconds; must be strictly positive.

    Returns
    -------
    float
        ``len(microsaccades) / duration_s``.

    Raises
    ------
    ValueError
        If ``duration_s`` is not strictly positive.
    """
    if duration_s <= 0.0:
        msg = f"duration_s must be > 0; got {duration_s}"
        raise ValueError(msg)
    return len(microsaccades) / duration_s


def sliding_window_stat(
    values: Sequence[float] | FloatArray,
    times: Sequence[float] | FloatArray,
    *,
    window_s: float,
    step_s: float,
    stat: str = "mean",
) -> tuple[FloatArray, FloatArray]:
    """Running statistic of a value series over overlapping time windows.

    Windows of width ``window_s`` are stepped by ``step_s`` from the first to
    the last timestamp; for each window the chosen ``stat`` is computed over the
    values whose timestamps fall in ``[start, start + window_s)``. The window
    centre time is reported alongside each statistic.

    Parameters
    ----------
    values:
        Value series aligned with ``times``.
    times:
        Sample times in seconds, the same length as ``values``.
    window_s:
        Window width in seconds; must be strictly positive.
    step_s:
        Step between successive window starts in seconds; must be strictly
        positive.
    stat:
        One of ``"mean"``, ``"std"`` (population, ``ddof=0``) or ``"count"``.
        Empty windows yield ``0.0`` for every statistic.

    Returns
    -------
    tuple of FloatArray
        ``(window_centers, stat_values)`` -- the centre time of each window and
        the statistic over it. Both are empty when ``times`` is empty.

    Raises
    ------
    ValueError
        If ``window_s`` or ``step_s`` is not strictly positive, if ``values``
        and ``times`` differ in length, or if ``stat`` is unknown.
    """
    if window_s <= 0.0:
        msg = f"window_s must be > 0; got {window_s}"
        raise ValueError(msg)
    if step_s <= 0.0:
        msg = f"step_s must be > 0; got {step_s}"
        raise ValueError(msg)
    if stat not in {"mean", "std", "count"}:
        msg = f"stat must be one of 'mean', 'std', 'count'; got {stat!r}"
        raise ValueError(msg)

    vals = np.asarray(values, dtype=np.float64).ravel()
    ts = np.asarray(times, dtype=np.float64).ravel()
    if vals.shape != ts.shape:
        msg = f"values and times must be equal length; got {vals.shape} vs {ts.shape}"
        raise ValueError(msg)
    if ts.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty

    t0 = float(ts[0])
    t_end = float(ts[-1])
    starts: list[float] = []
    start = t0
    while start <= t_end:
        starts.append(start)
        start += step_s
    if not starts:
        starts = [t0]

    centers = np.empty(len(starts), dtype=np.float64)
    out = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        in_win = vals[(ts >= s) & (ts < s + window_s)]
        centers[i] = s + window_s / 2.0
        if in_win.size == 0:
            out[i] = 0.0
        elif stat == "mean":
            out[i] = float(np.mean(in_win))
        elif stat == "std":
            out[i] = float(np.std(in_win))
        else:  # count
            out[i] = float(in_win.size)
    return centers, out
