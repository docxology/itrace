"""Saccade, fixation and microsaccade detection.

Implements three canonical, cited algorithms on degree-of-visual-angle gaze:

* **I-VT** velocity-threshold identification (Salvucci & Goldberg, 2000).
* **I-DT** dispersion-threshold identification (Salvucci & Goldberg, 2000).
* **Engbert & Kliegl (2003)** microsaccade detection with the median-based
  velocity-std estimator.

All operate on a :class:`~itrace.types.GazeStream` and return typed events.
"""

from __future__ import annotations

import numpy as np

from .geometry import direction_deg
from .types import BoolArray, Fixation, FloatArray, GazeStream, Microsaccade, Saccade
from .velocity import pos2vel_gradient, pos2vel_savgol, speed_2d


def _runs(mask: BoolArray) -> list[tuple[int, int]]:
    """Return [onset, offset] index pairs (inclusive) for each True run."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def velocities(
    stream: GazeStream,
    *,
    uniform: bool = True,
    window: int = 7,
    polyorder: int = 2,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return per-axis velocity and 2-D speed (deg/s) for a gaze stream."""
    if uniform and len(stream) >= 3:
        sr = stream.sampling_rate_hz
        vx = pos2vel_savgol(stream.x, sr, window=window, polyorder=polyorder)
        vy = pos2vel_savgol(stream.y, sr, window=window, polyorder=polyorder)
    else:
        vx = pos2vel_gradient(stream.x, stream.t)
        vy = pos2vel_gradient(stream.y, stream.t)
    return vx, vy, speed_2d(vx, vy)


def _build_saccade(stream: GazeStream, onset: int, offset: int, speed: FloatArray) -> Saccade:
    dx = float(stream.x[offset] - stream.x[onset])
    dy = float(stream.y[offset] - stream.y[onset])
    amplitude = float(np.hypot(dx, dy))
    peak_v = float(np.max(speed[onset : offset + 1]))
    return Saccade(
        onset_idx=onset,
        offset_idx=offset,
        onset_t=float(stream.t[onset]),
        offset_t=float(stream.t[offset]),
        amplitude_deg=amplitude,
        direction_deg=direction_deg(dx, dy),
        peak_velocity_deg_s=peak_v,
    )


def _build_fixation(stream: GazeStream, onset: int, offset: int) -> Fixation:
    sl = slice(onset, offset + 1)
    return Fixation(
        onset_idx=onset,
        offset_idx=offset,
        onset_t=float(stream.t[onset]),
        offset_t=float(stream.t[offset]),
        centroid_x=float(np.mean(stream.x[sl])),
        centroid_y=float(np.mean(stream.y[sl])),
    )


def detect_ivt(
    stream: GazeStream,
    velocity_threshold_deg_s: float = 30.0,
    *,
    min_saccade_duration_s: float = 0.006,
    merge_gap_s: float = 0.0,
    min_inter_event_gap_s: float = 0.0,
    max_saccade_duration_s: float | None = None,
    reject_edge_events: bool = False,
    uniform: bool = True,
) -> tuple[list[Fixation], list[Saccade]]:
    """I-VT: classify samples by speed into fixations and saccades.

    Samples whose 2-D speed exceeds ``velocity_threshold_deg_s`` are saccadic;
    contiguous runs are collapsed into events. Saccade runs shorter than
    ``min_saccade_duration_s`` are absorbed back into the surrounding fixation
    (suppresses single-sample velocity spikes). ``merge_gap_s`` bridges short
    subthreshold gaps inside a candidate saccade before the duration filter is
    applied.
    """
    if len(stream) < 2:
        return [], []
    if velocity_threshold_deg_s < 0.0:
        msg = "velocity_threshold_deg_s must be non-negative"
        raise ValueError(msg)
    if min_saccade_duration_s < 0.0:
        msg = "min_saccade_duration_s must be non-negative"
        raise ValueError(msg)
    if merge_gap_s < 0.0:
        msg = "merge_gap_s must be non-negative"
        raise ValueError(msg)
    if min_inter_event_gap_s < 0.0:
        msg = "min_inter_event_gap_s must be non-negative"
        raise ValueError(msg)
    if max_saccade_duration_s is not None and max_saccade_duration_s <= 0.0:
        msg = "max_saccade_duration_s must be positive when provided"
        raise ValueError(msg)
    _vx, _vy, speed = velocities(stream, uniform=uniform)
    sacc_mask = speed > velocity_threshold_deg_s

    gap_bridge_s = max(merge_gap_s, min_inter_event_gap_s)
    if gap_bridge_s > 0.0:
        for onset, offset in _runs(~sacc_mask):
            if onset == 0 or offset == len(stream) - 1:
                continue
            gap_duration = float(stream.t[offset] - stream.t[onset])
            if gap_duration <= gap_bridge_s:
                sacc_mask[onset : offset + 1] = True

    # Drop too-short saccade runs.
    for onset, offset in _runs(sacc_mask):
        dur = float(stream.t[offset] - stream.t[onset])
        amplitude = float(
            np.hypot(stream.x[offset] - stream.x[onset], stream.y[offset] - stream.y[onset])
        )
        should_drop = (
            offset <= onset
            or amplitude <= 0.0
            or dur < min_saccade_duration_s
            or (max_saccade_duration_s is not None and dur > max_saccade_duration_s)
            or (reject_edge_events and (onset == 0 or offset == len(stream) - 1))
        )
        if should_drop:
            sacc_mask[onset : offset + 1] = False

    saccades = [_build_saccade(stream, o, e, speed) for o, e in _runs(sacc_mask)]
    fixations = [_build_fixation(stream, o, e) for o, e in _runs(~sacc_mask)]
    return fixations, saccades


def detect_idt(
    stream: GazeStream,
    dispersion_threshold_deg: float = 1.0,
    min_duration_s: float = 0.1,
) -> list[Fixation]:
    """I-DT: dispersion-threshold fixation identification.

    A window of consecutive samples whose spatial dispersion
    ``(max(x)-min(x)) + (max(y)-min(y))`` stays below the threshold for at least
    ``min_duration_s`` is one fixation; the window is greedily extended until
    dispersion would exceed the threshold.
    """
    n = len(stream)
    if n < 2:
        return []
    x, y, t = stream.x, stream.y, stream.t
    fixations: list[Fixation] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n:
            sl = slice(start, end + 2)
            disp = (float(np.ptp(x[sl]))) + (float(np.ptp(y[sl])))
            if disp > dispersion_threshold_deg:
                break
            end += 1
        if float(t[end] - t[start]) >= min_duration_s and end > start:
            fixations.append(_build_fixation(stream, start, end))
            start = end + 1
        else:
            start += 1
    return fixations


def _ek_velocity(coord: FloatArray, dt: float) -> FloatArray:
    """Engbert & Kliegl (2003) 5-point moving-average velocity (eq. 1)."""
    n = coord.shape[0]
    v = np.zeros(n, dtype=np.float64)
    if n < 5:
        return v
    v[2:-2] = (coord[4:] + coord[3:-1] - coord[1:-3] - coord[:-4]) / (6.0 * dt)
    return v


def _ek_threshold(v: FloatArray, lam: float) -> float:
    """Median-based velocity-std threshold ``eta = lambda * sigma`` (eq. 2-3).

    ``sigma^2 = median(v^2) - median(v)^2``; the subtraction can go slightly
    negative for short/degenerate signals, so it is clamped at 0.
    """
    sigma_sq = float(np.median(v**2) - np.median(v) ** 2)
    sigma = float(np.sqrt(max(sigma_sq, 0.0)))
    return lam * sigma


def detect_microsaccades(
    stream: GazeStream,
    lambda_threshold: float = 6.0,
    min_duration_samples: int = 3,
) -> list[Microsaccade]:
    """Engbert & Kliegl (2003) microsaccade detection.

    Velocities use the 5-point estimator; a per-axis elliptic threshold
    ``(vx/eta_x)^2 + (vy/eta_y)^2 > 1`` held for at least
    ``min_duration_samples`` consecutive samples marks a microsaccade.
    """
    n = len(stream)
    if n < 5:
        return []
    dt = 1.0 / stream.sampling_rate_hz
    vx = _ek_velocity(stream.x, dt)
    vy = _ek_velocity(stream.y, dt)
    eta_x = _ek_threshold(vx, lambda_threshold)
    eta_y = _ek_threshold(vy, lambda_threshold)
    if eta_x <= 0 or eta_y <= 0:
        return []
    test = (vx / eta_x) ** 2 + (vy / eta_y) ** 2
    mask = test > 1.0
    out: list[Microsaccade] = []
    for onset, offset in _runs(mask):
        if offset - onset + 1 < min_duration_samples:
            continue
        dx = float(stream.x[offset] - stream.x[onset])
        dy = float(stream.y[offset] - stream.y[onset])
        peak_v = float(np.max(np.hypot(vx[onset : offset + 1], vy[onset : offset + 1])))
        out.append(
            Microsaccade(
                onset_idx=onset,
                offset_idx=offset,
                amplitude_deg=float(np.hypot(dx, dy)),
                peak_velocity_deg_s=peak_v,
                direction_deg=direction_deg(dx, dy),
            )
        )
    return out


def saccade_properties(saccades: list[Saccade]) -> dict[str, FloatArray]:
    """Vectorise a list of saccades into property arrays for analysis/plotting."""
    return {
        "amplitude_deg": np.array([s.amplitude_deg for s in saccades], dtype=np.float64),
        "direction_deg": np.array([s.direction_deg for s in saccades], dtype=np.float64),
        "duration_s": np.array([s.duration_s for s in saccades], dtype=np.float64),
        "peak_velocity_deg_s": np.array(
            [s.peak_velocity_deg_s for s in saccades], dtype=np.float64
        ),
    }
