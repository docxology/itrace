from __future__ import annotations

import numpy as np
import pytest

from itrace import pupil, velocity
from itrace.config import DetectionConfig, PupilConfig, analysis_config_from_mapping
from itrace.types import GazeStream, PupilStream


@pytest.mark.parametrize(
    "field",
    [
        "velocity_threshold_deg_s",
        "adaptive_lambda_factor",
        "min_saccade_duration_s",
        "merge_gap_s",
        "min_inter_event_gap_s",
        "max_saccade_duration_s",
        "long_saccade_deg",
        "smooth_pursuit_min_velocity_deg_s",
        "smooth_pursuit_max_velocity_deg_s",
        "smooth_pursuit_min_duration_s",
        "pso_window_s",
        "pso_peak_fraction",
    ],
)
def test_detection_rejects_nonfinite_numeric_settings(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        analysis_config_from_mapping({"detection": {field: float("nan")}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("velocity_threshold_deg_s", -1),
        ("min_saccade_duration_s", -1),
        ("long_saccade_deg", -1),
        ("smooth_pursuit_min_velocity_deg_s", -1),
        ("smooth_pursuit_max_velocity_deg_s", 1),
        ("pso_window_s", 0),
        ("pso_peak_fraction", 0),
    ],
)
def test_detection_rejects_invalid_finite_ranges(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        analysis_config_from_mapping({"detection": {field: value}})


def test_config_windows_are_copied_and_unknown_fields_fail() -> None:
    window = [0.0, 1.0]
    config = analysis_config_from_mapping({"pupil": {"baseline_window_s": window}})
    window[1] = 100.0
    assert config.pupil.baseline_window_s == (0.0, 1.0)
    with pytest.raises(ValueError, match="end must exceed"):
        PupilConfig(baseline_window_s=(1.0, 0.0))
    with pytest.raises(ValueError, match="unknown analysis"):
        analysis_config_from_mapping({"typo": {}})
    with pytest.raises(ValueError, match="unknown detection"):
        analysis_config_from_mapping({"detection": {"typo": 1}})
    with pytest.raises(ValueError, match="must be an object"):
        analysis_config_from_mapping({"pupil": 1})
    assert analysis_config_from_mapping({"detection": None}).detection == DetectionConfig()


@pytest.mark.parametrize(
    ("length", "order"), [(n, order) for n in range(17) for order in (1, 2, 3)]
)
def test_filter_padding_boundaries_preserve_constant_stream(length: int, order: int) -> None:
    stream = PupilStream(t=np.arange(length, dtype=np.float64) / 60.0, size=np.full(length, 3.0))
    result = pupil.smooth(stream, order=order)
    assert len(result) == length
    np.testing.assert_allclose(result.size, 3.0, atol=1e-10)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("cutoff_hz", 0.0),
        ("cutoff_hz", float("inf")),
        ("order", 0),
        ("order", 1.5),
        ("fallback_window", 0),
    ],
)
def test_smoothing_rejects_invalid_settings(keyword: str, value: float) -> None:
    stream = PupilStream(t=np.arange(5, dtype=np.float64), size=np.ones(5))
    with pytest.raises(ValueError, match=keyword):
        pupil.smooth(stream, **{keyword: value})


def test_preprocessing_degenerate_cases_are_explicit() -> None:
    empty = PupilStream(t=np.zeros(0), size=np.zeros(0))
    assert pupil.detect_blinks(empty) == []
    stream = PupilStream(t=np.arange(3, dtype=np.float64), size=np.ones(3))
    with pytest.raises(ValueError, match="pad_samples"):
        pupil.interpolate_blinks(stream, pad_samples=-1)
    with pytest.raises(ValueError, match="min_valid"):
        pupil.interpolate_blinks(stream, min_valid=-1)
    with pytest.raises(ValueError, match="n_mad"):
        pupil.mad_reject(stream, n_mad=0)
    with pytest.raises(ValueError, match="PupilUnit"):
        PupilStream(t=stream.t, size=stream.size, unit="mm")
    with pytest.raises(ValueError, match=">=2"):
        _ = GazeStream(t=np.zeros(1), x=np.zeros(1), y=np.zeros(1)).sampling_rate_hz
    for kwargs in ({"window": 0}, {"polyorder": 0}):
        with pytest.raises(ValueError, match="integer"):
            velocity.pos2vel_savgol(np.zeros(10), 60.0, **kwargs)
