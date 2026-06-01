"""Configuration object tests."""

from __future__ import annotations

import pytest

from itrace.config import AnalysisConfig, CaptureConfig, DetectionConfig, FigureConfig, PupilConfig


def test_analysis_config_defaults_preserve_fixed_ivt() -> None:
    cfg = AnalysisConfig()
    assert cfg.detection.method == "ivt"
    assert cfg.detection.velocity_threshold_deg_s == 30.0
    assert cfg.detection.merge_gap_s == 0.0
    assert cfg.detection.include_microsaccades is True
    assert cfg.detection.include_pso is False
    assert cfg.to_dict()["detection"]["method"] == "ivt"  # type: ignore[index]


def test_detection_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="method"):
        DetectionConfig(method="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="adaptive_lambda_factor"):
        DetectionConfig(adaptive_lambda_factor=0.0)
    with pytest.raises(ValueError, match="merge_gap_s"):
        DetectionConfig(merge_gap_s=-0.001)
    with pytest.raises(ValueError, match="min_inter_event_gap_s"):
        DetectionConfig(min_inter_event_gap_s=-0.001)
    with pytest.raises(ValueError, match="max_saccade_duration_s"):
        DetectionConfig(max_saccade_duration_s=0.0)
    with pytest.raises(ValueError, match="smooth_pursuit_min_duration_s"):
        DetectionConfig(smooth_pursuit_min_duration_s=-0.001)


def test_other_config_validation() -> None:
    with pytest.raises(ValueError, match="smooth_window"):
        PupilConfig(smooth_window=0)
    with pytest.raises(ValueError, match="blink_pad_samples"):
        PupilConfig(blink_pad_samples=-1)
    with pytest.raises(ValueError, match="smooth_cutoff_hz"):
        PupilConfig(smooth_cutoff_hz=0.0)
    with pytest.raises(ValueError, match="smooth_order"):
        PupilConfig(smooth_order=0)
    with pytest.raises(ValueError, match="camera_index"):
        CaptureConfig(camera_index=-1)
    with pytest.raises(ValueError, match="dpi"):
        FigureConfig(dpi=10)
