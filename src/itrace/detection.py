"""Advanced oculomotor detection beyond fixed-threshold I-VT.

This module layers data-driven and post-hoc analyses on top of the canonical
detectors in :mod:`itrace.saccades`:

* **Adaptive velocity threshold** (Nystrom & Holmqvist, 2010
  [@nystrom2010adaptive]). Rather than fixing the saccade velocity cut-off at a
  literature constant, the threshold is estimated from the recording itself by
  the fixed-point iteration
  ``thr = mean(v | v < thr) + lambda * std(v | v < thr)`` until it converges.
  The converged value separates the noisy intersaccadic velocity floor from the
  saccadic peaks, adapting to each recording's noise level.
* **Post-saccadic oscillation (PSO) flagging.** Immediately after a saccade the
  eye frequently overshoots and rings; this module scans a short window after
  each saccade offset for a secondary velocity peak.
* **Intersaccadic intervals** and **per-saccade peak accelerations**, two common
  derived dynamics used in main-sequence and fatigue analyses.

Everything operates on a :class:`~itrace.types.GazeStream` and reuses
:func:`itrace.saccades.velocities` for the speed signal, so the velocity maths
stays in one place. This is a pure NumPy/SciPy core method module: it never
imports matplotlib and is importable without the plotting extras.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from . import saccades
from .geometry import direction_deg
from .types import PSO, Fixation, FloatArray, GazeStream, Saccade, SmoothPursuit


def adaptive_ivt_threshold(
    stream: GazeStream,
    *,
    lambda_factor: float = 6.0,
    max_iter: int = 20,
    tol: float = 1e-3,
) -> float:
    """Data-driven I-VT velocity threshold (Nystrom & Holmqvist, 2010).

    The threshold is found by fixed-point iteration on the *sub-threshold*
    velocity samples (those that currently look like fixational noise)::

        thr <- mean(v | v < thr) + lambda_factor * std(v | v < thr)

    The iteration is seeded from the mean observed speed, which sits within the
    basin of attraction that excludes the saccadic velocity peaks: subsequent
    sub-threshold statistics are then dominated by the intersaccadic noise floor,
    and the estimate converges onto that floor plus ``lambda_factor`` standard
    deviations of it [@nystrom2010adaptive]. The converged threshold separates
    the noise floor from the saccadic peaks without any hand-tuned constant.

    Parameters
    ----------
    stream:
        The gaze stream to analyse.
    lambda_factor:
        Multiplier on the sub-threshold velocity standard deviation. The classic
        value is ``6.0``.
    max_iter:
        Maximum number of fixed-point iterations.
    tol:
        Absolute convergence tolerance on the threshold (deg/s).

    Returns
    -------
    float
        The converged velocity threshold in deg/s. For streams too short to
        differentiate (``< 3`` samples) the threshold is ``0.0``. For a
        (near-)constant-speed signal -- which has no separable noise floor -- the
        maximum observed speed is returned, so downstream I-VT flags no spurious
        saccades.
    """
    if len(stream) < 3:
        return 0.0
    _vx, _vy, speed = saccades.velocities(stream)
    speed = speed[np.isfinite(speed)]
    if speed.size == 0:
        return 0.0
    max_speed = float(np.max(speed))
    # A (near-)constant speed signal has no separable noise floor: any threshold
    # below the constant flags everything, above it nothing. Return the peak so
    # downstream I-VT produces no spurious saccades.
    if max_speed <= 0.0 or float(np.std(speed)) <= tol:
        return max_speed

    thr = float(np.mean(speed))
    for _ in range(max_iter):
        # ``<=`` always retains at least the minimum sample, so the sub-threshold
        # set is never empty and the statistics below are always well-defined.
        below = speed[speed <= thr]
        new_thr = float(np.mean(below)) + lambda_factor * float(np.std(below))
        if abs(new_thr - thr) < tol:
            thr = new_thr
            break
        thr = new_thr
    return thr


def detect_ivt_adaptive(
    stream: GazeStream,
    *,
    lambda_factor: float = 6.0,
    min_saccade_duration_s: float = 0.006,
    merge_gap_s: float = 0.0,
    min_inter_event_gap_s: float = 0.0,
    max_saccade_duration_s: float | None = None,
    reject_edge_events: bool = False,
) -> tuple[list[Fixation], list[Saccade]]:
    """I-VT with a per-recording adaptive velocity threshold.

    Computes :func:`adaptive_ivt_threshold` and feeds it to
    :func:`itrace.saccades.detect_ivt`. This removes the need to hand-tune the
    velocity cut-off for recordings with different noise levels.

    Parameters
    ----------
    stream:
        The gaze stream to segment.
    lambda_factor:
        Passed through to :func:`adaptive_ivt_threshold`.
    min_saccade_duration_s:
        Minimum saccade duration, forwarded to
        :func:`itrace.saccades.detect_ivt` (suppresses single-sample spikes).
    merge_gap_s:
        Maximum subthreshold gap to bridge inside one saccade candidate.

    Returns
    -------
    tuple of (list[Fixation], list[Saccade])
        Same shape as :func:`itrace.saccades.detect_ivt`.
    """
    thr = adaptive_ivt_threshold(stream, lambda_factor=lambda_factor)
    return saccades.detect_ivt(
        stream,
        velocity_threshold_deg_s=thr,
        min_saccade_duration_s=min_saccade_duration_s,
        merge_gap_s=merge_gap_s,
        min_inter_event_gap_s=min_inter_event_gap_s,
        max_saccade_duration_s=max_saccade_duration_s,
        reject_edge_events=reject_edge_events,
    )


def detect_smooth_pursuit(
    stream: GazeStream,
    *,
    min_velocity_deg_s: float = 2.0,
    max_velocity_deg_s: float = 30.0,
    min_duration_s: float = 0.1,
) -> list[SmoothPursuit]:
    """Detect provisional smooth-pursuit intervals from sustained moderate speed."""
    if min_velocity_deg_s < 0.0:
        msg = "min_velocity_deg_s must be non-negative"
        raise ValueError(msg)
    if max_velocity_deg_s <= min_velocity_deg_s:
        msg = "max_velocity_deg_s must exceed min_velocity_deg_s"
        raise ValueError(msg)
    if min_duration_s < 0.0:
        msg = "min_duration_s must be non-negative"
        raise ValueError(msg)
    if len(stream) < 3:
        return []
    _vx, _vy, speed = saccades.velocities(stream)
    mask = (speed >= min_velocity_deg_s) & (speed <= max_velocity_deg_s)
    pursuits: list[SmoothPursuit] = []
    for onset, offset in saccades._runs(mask):
        duration = float(stream.t[offset] - stream.t[onset])
        if duration < min_duration_s:
            continue
        dx = float(stream.x[offset] - stream.x[onset])
        dy = float(stream.y[offset] - stream.y[onset])
        pursuits.append(
            SmoothPursuit(
                onset_idx=onset,
                offset_idx=offset,
                onset_t=float(stream.t[onset]),
                offset_t=float(stream.t[offset]),
                mean_velocity_deg_s=float(np.mean(speed[onset : offset + 1])),
                direction_deg=direction_deg(dx, dy),
            )
        )
    return pursuits


def detect_pso(
    stream: GazeStream,
    saccades_list: list[Saccade],
    *,
    window_s: float = 0.04,
    peak_fraction: float = 0.2,
) -> list[dict[str, float]]:
    """Flag post-saccadic oscillations after each saccade.

    Within ``window_s`` seconds following a saccade's offset, the speed signal is
    searched for a secondary local peak exceeding ``peak_fraction`` of that
    saccade's peak velocity. Such a peak is reported as a candidate PSO (glissade
    / dynamic overshoot).

    Parameters
    ----------
    stream:
        The gaze stream the saccades were detected on.
    saccades_list:
        Detected saccades to scan after.
    window_s:
        Length of the post-offset search window (seconds).
    peak_fraction:
        Fraction of the saccade peak velocity a secondary peak must exceed to be
        flagged.

    Returns
    -------
    list of dict
        One record per detected PSO with keys ``onset_t``, ``offset_t`` and
        ``peak_velocity_deg_s``. Empty when no oscillation is found (or for an
        empty saccade list / too-short stream).
    """
    n = len(stream)
    if n < 3 or not saccades_list:
        return []
    _vx, _vy, speed = saccades.velocities(stream)
    out: list[dict[str, float]] = []
    for sac in saccades_list:
        start = sac.offset_idx + 1
        if start >= n:
            continue
        # Window end index: last sample within window_s of the offset.
        end = start
        t_limit = float(stream.t[sac.offset_idx]) + window_s
        while end + 1 < n and float(stream.t[end + 1]) <= t_limit:
            end += 1
        if end <= start:
            continue
        seg = speed[start : end + 1]
        rel = int(np.argmax(seg))
        local_peak = float(seg[rel])
        if local_peak <= peak_fraction * sac.peak_velocity_deg_s:
            continue
        peak_idx = start + rel
        out.append(
            {
                "onset_t": float(stream.t[start]),
                "offset_t": float(stream.t[end]),
                "peak_velocity_deg_s": float(speed[peak_idx]),
            }
        )
    return out


def detect_pso_events(
    stream: GazeStream,
    saccades_list: list[Saccade],
    *,
    window_s: float = 0.04,
    peak_fraction: float = 0.2,
) -> list[PSO]:
    """Return typed PSO candidates for report integration.

    This preserves :func:`detect_pso`'s historical ``list[dict]`` return shape
    while giving the pipeline stable indices and parent-saccade links.
    """
    n = len(stream)
    if n < 3 or not saccades_list:
        return []
    _vx, _vy, speed = saccades.velocities(stream)
    out: list[PSO] = []
    for parent_idx, sac in enumerate(saccades_list):
        start = sac.offset_idx + 1
        if start >= n:
            continue
        end = start
        t_limit = float(stream.t[sac.offset_idx]) + window_s
        while end + 1 < n and float(stream.t[end + 1]) <= t_limit:
            end += 1
        if end <= start:
            continue
        seg = speed[start : end + 1]
        rel = int(np.argmax(seg))
        local_peak = float(seg[rel])
        if local_peak <= peak_fraction * sac.peak_velocity_deg_s:
            continue
        peak_idx = start + rel
        out.append(
            PSO(
                onset_idx=start,
                offset_idx=end,
                onset_t=float(stream.t[start]),
                offset_t=float(stream.t[end]),
                peak_velocity_deg_s=float(speed[peak_idx]),
                parent_saccade_idx=parent_idx,
            )
        )
    return out


def intersaccadic_intervals(saccades_list: list[Saccade]) -> FloatArray:
    """Gaps (seconds) between consecutive saccades.

    Each interval is ``next.onset_t - current.offset_t`` for saccades taken in
    list order.

    Parameters
    ----------
    saccades_list:
        Detected saccades (assumed time-ordered, as returned by the detectors).

    Returns
    -------
    FloatArray
        Length ``len(saccades_list) - 1``; empty when fewer than two saccades.
    """
    if len(saccades_list) < 2:
        return np.empty(0, dtype=np.float64)
    return np.array(
        [nxt.onset_t - cur.offset_t for cur, nxt in pairwise(saccades_list)],
        dtype=np.float64,
    )


def saccade_peak_accelerations(
    stream: GazeStream,
    saccades_list: list[Saccade],
) -> FloatArray:
    """Peak absolute acceleration (deg/s^2) within each saccade.

    The speed signal is differentiated by central finite differences with respect
    to the stream timestamps; for each saccade the maximum ``|d(speed)/dt|`` over
    its index span is returned.

    Parameters
    ----------
    stream:
        The gaze stream the saccades were detected on.
    saccades_list:
        Detected saccades.

    Returns
    -------
    FloatArray
        One non-negative, finite peak acceleration per saccade (deg/s^2); empty
        when there are no saccades or the stream is too short to differentiate.
    """
    n = len(stream)
    if not saccades_list or n < 2:
        return np.empty(0, dtype=np.float64)
    _vx, _vy, speed = saccades.velocities(stream)
    accel = np.gradient(speed, stream.t)
    abs_accel = np.abs(accel)
    out = np.empty(len(saccades_list), dtype=np.float64)
    for i, sac in enumerate(saccades_list):
        lo = max(sac.onset_idx, 0)
        hi = min(sac.offset_idx, n - 1)
        span = abs_accel[lo : hi + 1]
        finite = span[np.isfinite(span)]
        out[i] = float(np.max(finite)) if finite.size else 0.0
    return out
