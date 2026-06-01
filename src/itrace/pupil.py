"""Pupillometry preprocessing.

A principled, transparent pipeline matching the open-source pupillometry
consensus (PupEyes / pypillometry): detect blinks, interpolate across them,
reject MAD outliers, low-pass smooth, and baseline-correct. Every step is a
pure function over a :class:`~itrace.types.PupilStream`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .pupilphase import PhaseDetector
from .types import BoolArray, FloatArray, PupilStream


def detect_blinks(
    stream: PupilStream,
    min_valid: float = 1e-6,
) -> list[tuple[int, int]]:
    """Return [onset, offset] index pairs (inclusive) of blink/invalid runs.

    A sample is invalid if it is NaN, non-positive, or below ``min_valid``
    (eyelid closure collapses the measured pupil toward zero).
    """
    size = stream.size
    invalid = ~np.isfinite(size) | (size <= min_valid)
    if size.size == 0:
        return []
    padded = np.concatenate(([False], invalid, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def interpolate_blinks(
    stream: PupilStream,
    pad_samples: int = 1,
    *,
    min_valid: float = 1e-6,
) -> PupilStream:
    """Linearly interpolate across blink/invalid runs.

    Each invalid run is widened by ``pad_samples`` on both sides (to remove the
    partial-occlusion ramp), then filled by linear interpolation between the
    nearest surrounding valid samples. Edge runs are filled with the nearest
    valid value.

    Raises
    ------
    ValueError
        If the trace has no valid samples at all (cannot interpolate from
        nothing -- the trace is unusable, not silently flattened).
    """
    if pad_samples < 0:
        msg = "pad_samples must be non-negative"
        raise ValueError(msg)
    if min_valid < 0.0:
        msg = "min_valid must be non-negative"
        raise ValueError(msg)
    size = stream.size.copy()
    valid = np.isfinite(size) & (size > min_valid)
    if not np.any(valid):
        msg = "pupil trace has no valid samples; cannot interpolate (unusable)"
        raise ValueError(msg)

    for onset, offset in detect_blinks(stream, min_valid=min_valid):
        lo = max(onset - pad_samples, 0)
        hi = min(offset + pad_samples, size.size - 1)
        valid[lo : hi + 1] = False

    idx = np.arange(size.size)
    size[~valid] = np.interp(idx[~valid], idx[valid], size[valid])
    return PupilStream(t=stream.t, size=size, unit=stream.unit)


def mad_reject(stream: PupilStream, n_mad: float = 5.0) -> BoolArray:
    """Boolean mask of samples flagged as outliers by median-absolute-deviation.

    ``True`` marks an outlier. Uses the robust scaled MAD
    (``1.4826 * MAD``) so the threshold is in approximate standard deviations.
    """
    size = stream.size
    finite = np.isfinite(size)
    flags = np.zeros(size.size, dtype=bool)
    if not np.any(finite):
        return ~finite
    med = float(np.median(size[finite]))
    mad = float(np.median(np.abs(size[finite] - med)))
    if mad <= 0:
        flags[~finite] = True
        return flags
    scaled = 1.4826 * mad
    deviation = np.abs(size - med) / scaled
    return cast(BoolArray, ~finite | (deviation > n_mad))


def smooth(stream: PupilStream, cutoff_hz: float = 4.0, order: int = 2) -> PupilStream:
    """Zero-phase Butterworth low-pass filter (falls back to a moving average).

    For very short traces (where the filter padding is invalid) a centred
    moving average is used instead so the function never raises on small input.
    """
    size = stream.size.astype(np.float64)
    n = size.size
    if n < 2:
        return stream
    if not np.all(np.isfinite(size)):
        msg = "smooth() requires a gap-free trace; interpolate blinks first"
        raise ValueError(msg)
    sr = _sampling_rate(stream.t)
    nyq = sr / 2.0
    if n >= 3 * (order + 1) and 0 < cutoff_hz < nyq:
        sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
        out: FloatArray = sosfiltfilt(sos, size)
    else:  # pragma: no cover - degenerate short-trace fallback
        kernel = np.ones(3) / 3.0
        out = np.convolve(size, kernel, mode="same")
    return PupilStream(t=stream.t, size=out, unit=stream.unit)


def baseline_correct(
    stream: PupilStream,
    baseline_window_s: tuple[float, float],
    mode: str = "subtractive",
) -> PupilStream:
    """Subtract (or divide by) the mean pupil size in a baseline time window.

    ``baseline_window_s`` is ``(start_t, end_t)`` relative to the same clock as
    ``stream.t``. ``mode`` is ``"subtractive"`` (default) or ``"divisive"``.
    """
    t0, t1 = baseline_window_s
    if t1 <= t0:
        msg = "baseline window end must exceed its start"
        raise ValueError(msg)
    mask = (stream.t >= t0) & (stream.t <= t1)
    if not np.any(mask):
        msg = "baseline window contains no samples"
        raise ValueError(msg)
    base = float(np.mean(stream.size[mask]))
    if mode == "subtractive":
        corrected = stream.size - base
    elif mode == "divisive":
        if base == 0:
            msg = "cannot divisively baseline-correct against a zero baseline"
            raise ValueError(msg)
        corrected = stream.size / base
    else:
        msg = f"unknown baseline mode {mode!r}"
        raise ValueError(msg)
    return PupilStream(t=stream.t, size=corrected, unit=stream.unit)


def quality_summary(stream: PupilStream, *, min_valid: float = 1e-6) -> dict[str, float]:
    """Summarise pupil validity, blink burden and first-derivative dynamics."""
    if min_valid < 0.0:
        msg = "min_valid must be non-negative"
        raise ValueError(msg)
    n = len(stream)
    finite_positive = np.isfinite(stream.size) & (stream.size > min_valid)
    blinks = detect_blinks(stream, min_valid=min_valid)
    invalid = ~finite_positive
    finite_t = stream.t[np.isfinite(stream.t)]
    if finite_t.size >= 2:
        dt = np.diff(finite_t)
        positive_dt = dt[dt > 0.0]
    else:
        positive_dt = np.empty(0, dtype=np.float64)
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.0

    peak_dilation = 0.0
    peak_constriction = 0.0
    if n >= 2 and np.any(finite_positive):
        try:
            clean = interpolate_blinks(stream, min_valid=min_valid)
            velocity = np.gradient(clean.size, clean.t)
            finite_velocity = velocity[np.isfinite(velocity)]
            if finite_velocity.size:
                peak_dilation = float(np.max(finite_velocity))
                peak_constriction = float(np.min(finite_velocity))
        except ValueError:
            pass

    return {
        "n_samples": float(n),
        "valid_sample_fraction": float(np.mean(finite_positive)) if n else 0.0,
        "invalid_sample_fraction": float(np.mean(invalid)) if n else 0.0,
        "blink_fraction": float(np.mean(invalid)) if n else 0.0,
        "n_blinks": float(len(blinks)),
        "median_dt_s": median_dt,
        "peak_dilation_velocity": peak_dilation,
        "peak_constriction_velocity": peak_constriction,
    }


def response_features(
    stream: PupilStream,
    *,
    baseline_window_s: tuple[float, float],
    response_window_s: tuple[float, float],
) -> dict[str, float]:
    """Summarise baseline-relative pupil response features."""
    b0, b1 = baseline_window_s
    r0, r1 = response_window_s
    if b1 <= b0:
        msg = "baseline_window_s end must exceed start"
        raise ValueError(msg)
    if r1 <= r0:
        msg = "response_window_s end must exceed start"
        raise ValueError(msg)
    finite = np.isfinite(stream.size)
    baseline_mask = finite & (stream.t >= b0) & (stream.t <= b1)
    response_mask = finite & (stream.t >= r0) & (stream.t <= r1)
    if not np.any(baseline_mask):
        msg = "baseline window contains no finite samples"
        raise ValueError(msg)
    if not np.any(response_mask):
        msg = "response window contains no finite samples"
        raise ValueError(msg)
    baseline = float(np.mean(stream.size[baseline_mask]))
    rt = stream.t[response_mask]
    rs = stream.size[response_mask]
    peak_idx = int(np.argmax(rs))
    peak = float(rs[peak_idx])
    baseline_relative = (peak - baseline) / baseline if baseline != 0.0 else 0.0
    centered = rs - baseline
    positive = np.maximum(centered, 0.0)
    auc = float(np.sum((positive[1:] + positive[:-1]) * np.diff(rt) * 0.5)) if rt.size >= 2 else 0.0

    velocity = (
        np.gradient(stream.size, stream.t) if len(stream) >= 2 else np.zeros_like(stream.size)
    )
    v_resp = velocity[response_mask]
    dil = v_resp[v_resp > 0.0]
    con = v_resp[v_resp < 0.0]
    phases = PhaseDetector().run(stream.size.tolist())
    phase_names = [phase.value for phase in phases]
    response_indices = np.flatnonzero(response_mask)
    denom = float(response_indices.size) if response_indices.size else 1.0
    out = {
        "baseline_size": baseline,
        "response_peak_size": peak,
        "latency_to_peak_s": float(rt[peak_idx] - r0),
        "dilation_auc": auc,
        "baseline_relative_peak_change": float(baseline_relative),
        "mean_dilation_velocity": float(np.mean(dil)) if dil.size else 0.0,
        "mean_constriction_velocity": float(np.mean(con)) if con.size else 0.0,
    }
    for label in ("dilation", "constriction", "peak", "trough", "unknown"):
        count = sum(1 for idx in response_indices if phase_names[int(idx)] == label)
        out[f"phase_fraction_{label}"] = float(count / denom)
    return out


def _sampling_rate(t: FloatArray) -> float:
    if t.size < 2:
        return 1.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else 1.0
