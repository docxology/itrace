"""Calibration and recording-quality helpers for gaze streams.

The calibration model is intentionally small and inspectable: a two-dimensional
affine map from raw gaze coordinates to target coordinates. It is suitable for
screen-space webcam calibration and for synthetic verifier fixtures; it makes no
claim about device-level accuracy unless fitted against an external reference.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .types import FloatArray, GazeStream


def _as_float_vector(name: str, values: Iterable[float]) -> FloatArray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        msg = f"{name} must be a 1-D vector"
        raise ValueError(msg)
    return arr


def _check_equal_lengths(*arrays: FloatArray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        msg = "calibration arrays must be equal-length"
        raise ValueError(msg)


def _coef3(values: FloatArray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


@dataclass(frozen=True, slots=True)
class AffineCalibration:
    """Two-dimensional affine gaze calibration.

    The transform is:

    ``target_x = ax*x + bx*y + cx``
    ``target_y = ay*x + by*y + cy``
    """

    x_coefficients: tuple[float, float, float]
    y_coefficients: tuple[float, float, float]
    n_points: int
    rms_error_deg: float

    @classmethod
    def fit(
        cls,
        raw_x: Iterable[float],
        raw_y: Iterable[float],
        target_x: Iterable[float],
        target_y: Iterable[float],
        *,
        ridge: float = 0.0,
    ) -> AffineCalibration:
        """Fit an affine calibration from raw gaze to known targets."""
        rx = _as_float_vector("raw_x", raw_x)
        ry = _as_float_vector("raw_y", raw_y)
        tx = _as_float_vector("target_x", target_x)
        ty = _as_float_vector("target_y", target_y)
        _check_equal_lengths(rx, ry, tx, ty)
        if rx.size < 3:
            msg = "affine calibration needs at least 3 points"
            raise ValueError(msg)
        if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(ry))):
            msg = "raw calibration coordinates must be finite"
            raise ValueError(msg)
        if not (np.all(np.isfinite(tx)) and np.all(np.isfinite(ty))):
            msg = "target calibration coordinates must be finite"
            raise ValueError(msg)
        if ridge < 0.0:
            msg = "ridge must be non-negative"
            raise ValueError(msg)

        design = np.column_stack([rx, ry, np.ones_like(rx)])
        if ridge == 0.0:
            x_coef, *_ = np.linalg.lstsq(design, tx, rcond=None)
            y_coef, *_ = np.linalg.lstsq(design, ty, rcond=None)
        else:
            penalty = ridge * np.eye(3)
            lhs = design.T @ design + penalty
            x_coef = np.linalg.solve(lhs, design.T @ tx)
            y_coef = np.linalg.solve(lhs, design.T @ ty)
        px = design @ x_coef
        py = design @ y_coef
        err = np.hypot(px - tx, py - ty)
        return cls(
            x_coefficients=_coef3(x_coef),
            y_coefficients=_coef3(y_coef),
            n_points=int(rx.size),
            rms_error_deg=float(np.sqrt(np.mean(err**2))),
        )

    def apply(self, x: Iterable[float], y: Iterable[float]) -> tuple[FloatArray, FloatArray]:
        """Apply the fitted calibration to coordinate vectors."""
        rx = _as_float_vector("x", x)
        ry = _as_float_vector("y", y)
        _check_equal_lengths(rx, ry)
        design = np.column_stack([rx, ry, np.ones_like(rx)])
        xc = np.array(self.x_coefficients, dtype=np.float64)
        yc = np.array(self.y_coefficients, dtype=np.float64)
        return design @ xc, design @ yc

    def apply_stream(self, stream: GazeStream) -> GazeStream:
        """Return a calibrated copy of a :class:`GazeStream`."""
        x, y = self.apply(stream.x, stream.y)
        return GazeStream(t=stream.t, x=x, y=y)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly calibration record."""
        return {
            "x_coefficients": list(self.x_coefficients),
            "y_coefficients": list(self.y_coefficients),
            "n_points": self.n_points,
            "rms_error_deg": self.rms_error_deg,
        }


def calibration_error(
    calibration: AffineCalibration,
    raw_x: Iterable[float],
    raw_y: Iterable[float],
    target_x: Iterable[float],
    target_y: Iterable[float],
) -> dict[str, float]:
    """Return pointwise calibration-error summary in degrees."""
    px, py = calibration.apply(raw_x, raw_y)
    tx = _as_float_vector("target_x", target_x)
    ty = _as_float_vector("target_y", target_y)
    _check_equal_lengths(px, py, tx, ty)
    err = np.hypot(px - tx, py - ty)
    return {
        "n_points": float(err.size),
        "mean_error_deg": float(np.mean(err)) if err.size else 0.0,
        "median_error_deg": float(np.median(err)) if err.size else 0.0,
        "rms_error_deg": float(np.sqrt(np.mean(err**2))) if err.size else 0.0,
        "p95_error_deg": float(np.percentile(err, 95)) if err.size else 0.0,
        "max_error_deg": float(np.max(err)) if err.size else 0.0,
    }


def robust_gaze_quality(stream: GazeStream, *, gap_factor: float = 3.0) -> dict[str, float]:
    """Summarise finite-sample, timing-jitter, and gap quality for a gaze stream."""
    n = len(stream)
    finite = np.isfinite(stream.t) & np.isfinite(stream.x) & np.isfinite(stream.y)
    valid_fraction = float(np.mean(finite)) if n else 0.0
    finite_t = stream.t[np.isfinite(stream.t)]
    if finite_t.size >= 2:
        dt = np.diff(finite_t)
        positive_dt = dt[dt > 0.0]
    else:
        positive_dt = np.empty(0, dtype=np.float64)
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.0
    jitter = float(np.median(np.abs(positive_dt - median_dt))) if positive_dt.size else 0.0
    longest_gap = float(np.max(positive_dt)) if positive_dt.size else 0.0
    gap_threshold = median_dt * gap_factor if median_dt > 0.0 else np.inf
    large_gap_count = float(np.sum(positive_dt > gap_threshold)) if positive_dt.size else 0.0
    return {
        "n_samples": float(n),
        "valid_sample_fraction": valid_fraction,
        "dropout_fraction": 1.0 - valid_fraction,
        "median_dt_s": median_dt,
        "sampling_jitter_s": jitter,
        "longest_gap_s": longest_gap,
        "large_gap_count": large_gap_count,
        "nonmonotonic_timestamp_count": float(np.sum(np.diff(stream.t) <= 0.0)) if n >= 2 else 0.0,
    }


def interpolate_gaze_gaps(stream: GazeStream, *, max_gap_s: float) -> GazeStream:
    """Linearly interpolate short bounded invalid gaze runs.

    Invalid samples are those with non-finite x or y. Only runs with a finite
    sample on both sides and a duration no greater than ``max_gap_s`` are filled;
    edge gaps remain invalid, and long bounded gaps raise so callers do not
    accidentally invent long stretches of gaze.
    """
    if max_gap_s < 0.0:
        msg = "max_gap_s must be non-negative"
        raise ValueError(msg)
    valid = np.isfinite(stream.x) & np.isfinite(stream.y) & np.isfinite(stream.t)
    if np.all(valid):
        return stream
    x = stream.x.copy()
    y = stream.y.copy()
    padded = np.concatenate(([False], ~valid, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    for start, end in zip(starts, ends, strict=True):
        if start == 0 or end == len(stream) - 1:
            continue
        duration = float(stream.t[end] - stream.t[start])
        if duration > max_gap_s:
            msg = f"bounded gaze gap {duration:.6g}s exceeds max_gap_s={max_gap_s:.6g}s"
            raise ValueError(msg)
        left = start - 1
        right = end + 1
        idx = np.arange(start, end + 1)
        x[idx] = np.interp(stream.t[idx], [stream.t[left], stream.t[right]], [x[left], x[right]])
        y[idx] = np.interp(stream.t[idx], [stream.t[left], stream.t[right]], [y[left], y[right]])
    return GazeStream(t=stream.t, x=x, y=y)
