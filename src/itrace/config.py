"""Configuration objects for iTrace analysis, capture and figures.

The defaults preserve the pre-0.4 behaviour: fixed-threshold I-VT, 30 deg/s,
microsaccades enabled, PSO detection disabled, and optional capture/dashboard
features kept behind their extras.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, cast, get_type_hints

DetectionMethod = Literal["ivt", "adaptive_ivt"]


def _validate_scalar_fields(
    config: DetectionConfig | PupilConfig | CaptureConfig | CalibrationConfig | FigureConfig,
) -> None:
    for name, annotation in get_type_hints(type(config)).items():
        value = getattr(config, name)
        if annotation is bool and not isinstance(value, bool):
            msg = f"{name} must be a boolean"
            raise ValueError(msg)
        if annotation is int and (not isinstance(value, int) or isinstance(value, bool)):
            msg = f"{name} must be an integer"
            raise ValueError(msg)
        if annotation is float or annotation == float | None:
            if annotation == float | None and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                msg = f"{name} must be a finite number"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Event-detection settings for gaze analysis."""

    method: DetectionMethod = "ivt"
    velocity_threshold_deg_s: float = 30.0
    adaptive_lambda_factor: float = 6.0
    min_saccade_duration_s: float = 0.006
    merge_gap_s: float = 0.0
    min_inter_event_gap_s: float = 0.0
    max_saccade_duration_s: float | None = None
    reject_edge_events: bool = False
    long_saccade_deg: float = 5.0
    include_microsaccades: bool = True
    include_smooth_pursuit: bool = False
    smooth_pursuit_min_velocity_deg_s: float = 2.0
    smooth_pursuit_max_velocity_deg_s: float = 30.0
    smooth_pursuit_min_duration_s: float = 0.1
    include_pso: bool = False
    pso_window_s: float = 0.04
    pso_peak_fraction: float = 0.2

    def __post_init__(self) -> None:
        _validate_scalar_fields(self)
        if self.method not in {"ivt", "adaptive_ivt"}:
            msg = f"method must be 'ivt' or 'adaptive_ivt'; got {self.method!r}"
            raise ValueError(msg)
        if self.velocity_threshold_deg_s < 0.0:
            msg = "velocity_threshold_deg_s must be non-negative"
            raise ValueError(msg)
        if self.adaptive_lambda_factor <= 0.0:
            msg = "adaptive_lambda_factor must be positive"
            raise ValueError(msg)
        if self.min_saccade_duration_s < 0.0:
            msg = "min_saccade_duration_s must be non-negative"
            raise ValueError(msg)
        if self.merge_gap_s < 0.0:
            msg = "merge_gap_s must be non-negative"
            raise ValueError(msg)
        if self.min_inter_event_gap_s < 0.0:
            msg = "min_inter_event_gap_s must be non-negative"
            raise ValueError(msg)
        if self.max_saccade_duration_s is not None and self.max_saccade_duration_s <= 0.0:
            msg = "max_saccade_duration_s must be positive when provided"
            raise ValueError(msg)
        if self.long_saccade_deg < 0.0:
            msg = "long_saccade_deg must be non-negative"
            raise ValueError(msg)
        if self.smooth_pursuit_min_velocity_deg_s < 0.0:
            msg = "smooth_pursuit_min_velocity_deg_s must be non-negative"
            raise ValueError(msg)
        if self.smooth_pursuit_max_velocity_deg_s <= self.smooth_pursuit_min_velocity_deg_s:
            msg = "smooth_pursuit_max_velocity_deg_s must exceed the minimum"
            raise ValueError(msg)
        if self.smooth_pursuit_min_duration_s < 0.0:
            msg = "smooth_pursuit_min_duration_s must be non-negative"
            raise ValueError(msg)
        if self.pso_window_s <= 0.0:
            msg = "pso_window_s must be positive"
            raise ValueError(msg)
        if self.pso_peak_fraction <= 0.0:
            msg = "pso_peak_fraction must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PupilConfig:
    """Pupil cleaning and summary settings."""

    enabled: bool = True
    smooth_window: int = 5
    blink_threshold: float = 0.0
    blink_pad_samples: int = 1
    smooth_cutoff_hz: float = 4.0
    smooth_order: int = 2
    baseline_window_s: tuple[float, float] | None = None
    response_window_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _validate_scalar_fields(self)
        if self.smooth_window < 1:
            msg = "smooth_window must be >= 1"
            raise ValueError(msg)
        if self.blink_threshold < 0.0:
            msg = "blink_threshold must be non-negative"
            raise ValueError(msg)
        if self.blink_pad_samples < 0:
            msg = "blink_pad_samples must be non-negative"
            raise ValueError(msg)
        if self.smooth_cutoff_hz <= 0.0:
            msg = "smooth_cutoff_hz must be positive"
            raise ValueError(msg)
        if self.smooth_order < 1:
            msg = "smooth_order must be >= 1"
            raise ValueError(msg)
        for name, window in (
            ("baseline_window_s", self.baseline_window_s),
            ("response_window_s", self.response_window_s),
        ):
            if window is not None:
                if (
                    not isinstance(window, tuple | list)
                    or len(window) != 2
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, int | float)
                        or not math.isfinite(v)
                        for v in window
                    )
                ):
                    msg = f"{name} must contain two finite numbers"
                    raise ValueError(msg)
                if window[1] <= window[0]:
                    msg = f"{name} end must exceed start"
                    raise ValueError(msg)
                object.__setattr__(self, name, tuple(window))


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Webcam capture settings."""

    camera_index: int = 0
    max_angle_deg: float = 25.0
    max_frames: int = 300
    write_pupil: bool = False

    def __post_init__(self) -> None:
        _validate_scalar_fields(self)
        if self.camera_index < 0:
            msg = "camera_index must be non-negative"
            raise ValueError(msg)
        if self.max_angle_deg <= 0.0:
            msg = "max_angle_deg must be positive"
            raise ValueError(msg)
        if self.max_frames < 1:
            msg = "max_frames must be >= 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Full analysis configuration passed into :func:`itrace.pipeline.analyze_gaze`."""

    detection: DetectionConfig = DetectionConfig()
    pupil: PupilConfig = PupilConfig()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the configuration."""
        return {
            "detection": asdict(self.detection),
            "pupil": asdict(self.pupil),
        }


def _known_field_subset(cls: type[Any], values: object, section: str) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        msg = f"{section} config must be an object"
        raise ValueError(msg)
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        msg = f"unknown {section} config keys: {', '.join(unknown)}"
        raise ValueError(msg)
    return dict(values)


def analysis_config_from_mapping(values: dict[str, object]) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from a JSON-like mapping.

    Only ``detection`` and ``pupil`` sections are accepted. Unknown keys fail
    loudly so a misspelled method setting cannot silently fall back to defaults.
    """
    if not isinstance(values, dict):
        msg = "analysis config mapping must be a dict"
        raise ValueError(msg)
    allowed = {"detection", "pupil"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        msg = f"unknown analysis config sections: {', '.join(unknown)}"
        raise ValueError(msg)
    detection_values = _known_field_subset(DetectionConfig, values.get("detection"), "detection")
    pupil_values = _known_field_subset(PupilConfig, values.get("pupil"), "pupil")
    return AnalysisConfig(
        detection=DetectionConfig(**cast(Any, detection_values)),
        pupil=PupilConfig(**cast(Any, pupil_values)),
    )


def analysis_config_from_json(path: str | Path) -> AnalysisConfig:
    """Load an :class:`AnalysisConfig` from a JSON file."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        msg = "analysis config JSON must contain an object"
        raise ValueError(msg)
    return analysis_config_from_mapping(payload)


@dataclass(frozen=True, slots=True)
class AOIConfig:
    """Area of Interest list configuration for spatial gaze analysis."""

    aois: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"aois": self.aois}


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Gaze calibration settings."""

    method: Literal["affine", "polynomial"] = "affine"
    polynomial_degree: int = 2
    target_range_deg: float = 15.0

    def __post_init__(self) -> None:
        _validate_scalar_fields(self)
        if self.method not in {"affine", "polynomial"}:
            msg = f"method must be 'affine' or 'polynomial'; got {self.method!r}"
            raise ValueError(msg)
        if self.polynomial_degree not in {2, 3}:
            msg = "polynomial_degree must be 2 or 3"
            raise ValueError(msg)
        if self.target_range_deg <= 0.0:
            msg = "target_range_deg must be positive"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FigureConfig:
    """Figure/animation rendering settings."""

    out_dir: str = "output/figures"
    seed: int = 0
    dpi: int = 300
    animations: bool = False

    def __post_init__(self) -> None:
        _validate_scalar_fields(self)
        if self.dpi < 72:
            msg = "dpi must be >= 72"
            raise ValueError(msg)
