"""Live HTML analysis payloads and browser message assembly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from .. import pipeline, saccades, validation
from ..calibration import AffineCalibration
from ..capture import CaptureSample, LiveFrameSample, samples_to_streams
from ..config import AnalysisConfig, DetectionConfig
from ..reporting import (
    empty_session_report_dict,
    error_session_report_dict,
    partial_session_report_dict,
    session_report_dict,
)
from ..stats.descriptive import session_statistics
from ..types import GazeStream, PupilStream, PupilUnit
from .state import LiveState

DetectionMethodName = Literal["ivt", "adaptive_ivt"]


def _json_float(value: float | int | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _json_float_list(values: Iterable[float | np.floating[Any]]) -> list[float | None]:
    return [_json_float(value) for value in values]


def _finite_gaze_fraction(gaze: GazeStream) -> float:
    """Return the fraction of samples with finite binocular gaze coordinates."""
    if len(gaze) == 0:
        return 0.0
    finite = np.isfinite(gaze.x) & np.isfinite(gaze.y)
    return float(np.mean(finite))


def method_name(method: str) -> DetectionMethodName:
    if method == "adaptive_ivt":
        return "adaptive_ivt"
    if method != "ivt":
        msg = f"method must be 'ivt' or 'adaptive_ivt'; got {method!r}"
        raise ValueError(msg)
    return "ivt"


def _empty_analysis_payload() -> dict[str, object]:
    empty_gaze = GazeStream(
        t=np.zeros(0, dtype=np.float64),
        x=np.zeros(0, dtype=np.float64),
        y=np.zeros(0, dtype=np.float64),
    )
    empty_pupil = PupilStream(
        t=np.zeros(0, dtype=np.float64),
        size=np.zeros(0, dtype=np.float64),
        unit=PupilUnit.RELATIVE,
    )
    report_dict = empty_session_report_dict()
    return {
        "report": report_dict,
        "series": {"t": [], "x": [], "y": [], "pupil": [], "speed": []},
        "statistics": session_statistics(empty_gaze, empty_pupil),
        "diagnostics": validation.live_recording_diagnostics(
            empty_gaze,
            empty_pupil,
            report_dict,
        ),
    }


def analysis_payload(
    samples: list[CaptureSample],
    *,
    method: DetectionMethodName,
    velocity_threshold_deg_s: float,
    include_pso: bool,
    calibration: AffineCalibration | None = None,
) -> dict[str, object]:
    """Build rolling analysis payload for live HTML panels."""
    if not samples:
        return _empty_analysis_payload()

    gaze, pupil = samples_to_streams(samples)
    speed: list[float | None]
    if len(samples) >= 2 and np.all(np.isfinite(gaze.x)) and np.all(np.isfinite(gaze.y)):
        _vx, _vy, speed_arr = saccades.velocities(gaze)
        speed = _json_float_list(speed_arr)
    else:
        speed = [None for _ in samples]

    quality = {"finite_sample_fraction": _finite_gaze_fraction(gaze)}
    duration = float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0
    if len(samples) >= 3:
        cfg = AnalysisConfig(
            detection=DetectionConfig(
                method=method,
                velocity_threshold_deg_s=velocity_threshold_deg_s,
                include_pso=include_pso,
            )
        )
        try:
            report = pipeline.analyze_session(gaze, pupil, config=cfg)
            report_dict = session_report_dict(report)
            statistics_payload = session_statistics(gaze, pupil, report)
        except ValueError as exc:
            report_dict = error_session_report_dict(
                n_samples=len(samples),
                duration_s=duration,
                quality=quality,
                error=str(exc),
            )
            statistics_payload = session_statistics(gaze, pupil)
    else:
        report_dict = partial_session_report_dict(
            n_samples=len(samples),
            duration_s=duration,
            quality=quality,
        )
        statistics_payload = session_statistics(gaze, pupil)

    series: dict[str, object] = {
        "t": _json_float_list(gaze.t),
        "x": _json_float_list(gaze.x),
        "y": _json_float_list(gaze.y),
        "pupil": _json_float_list(pupil.size),
        "speed": speed,
    }
    if calibration is not None:
        calibrated = calibration.apply_stream(gaze)
        series["calibrated_x"] = _json_float_list(calibrated.x)
        series["calibrated_y"] = _json_float_list(calibrated.y)
    return {
        "report": report_dict,
        "series": series,
        "statistics": statistics_payload,
        "diagnostics": validation.live_recording_diagnostics(gaze, pupil, report_dict),
    }


def live_message_from_frame(
    frame: LiveFrameSample,
    state: LiveState,
    *,
    method: str = "ivt",
    velocity_threshold_deg_s: float = 30.0,
    include_pso: bool = False,
    rolling_window_s: float = 10.0,
) -> dict[str, object]:
    """Build the browser message for one live frame."""
    method_name_value = method_name(method)
    samples = state.recent(rolling_window_s)
    analysis = analysis_payload(
        samples,
        method=method_name_value,
        velocity_threshold_deg_s=velocity_threshold_deg_s,
        include_pso=include_pso,
        calibration=state.calibration,
    )
    capture = frame.capture
    calibrated_gaze = None
    if state.calibration is not None:
        cx, cy = state.calibration.apply([capture.gaze.x], [capture.gaze.y])
        calibrated_gaze = {"x": _json_float(cx[0]), "y": _json_float(cy[0])}
    return {
        "type": "sample",
        "capture": {
            "frame_index": capture.frame_index,
            "timestamp_s": _json_float(capture.timestamp_s),
            "gaze": {"x": _json_float(capture.gaze.x), "y": _json_float(capture.gaze.y)},
            "calibrated_gaze": calibrated_gaze,
            "pupil": {
                "size": _json_float(capture.pupil.size) if capture.pupil is not None else None,
                "unit": capture.pupil.unit.value if capture.pupil is not None else None,
            },
            "fps_estimate_hz": _json_float(capture.fps_estimate_hz),
            "quality": capture.quality,
        },
        "frame": {
            "width": frame.frame_width,
            "height": frame.frame_height,
            "eye_box": frame.eye_box.to_dict(),
            "eye_crop_jpeg": frame.eye_crop_jpeg,
        },
        "analysis": analysis["report"],
        "series": analysis["series"],
        "statistics": analysis["statistics"],
        "diagnostics": analysis["diagnostics"],
        "method": {
            "name": method_name_value,
            "velocity_threshold_deg_s": velocity_threshold_deg_s,
            "include_pso": include_pso,
            "rolling_window_s": rolling_window_s,
        },
        "calibration": {
            "active": state.calibration is not None,
            "target_range_deg": state.calibration_target_range_deg,
            "session_points": len(state.calibration_points),
        },
        "experiment": state.experiment_status(),
    }
