"""Composable synthetic session tests."""

from __future__ import annotations

import numpy as np

from itrace import pipeline, synthetic
from itrace.config import AnalysisConfig, DetectionConfig
from itrace.synthetic import SyntheticSessionSpec, synthetic_session


def test_synthetic_session_is_seed_deterministic() -> None:
    spec = SyntheticSessionSpec(seed=4, n_saccades=4)
    g1, p1, t1 = synthetic_session(spec)
    g2, p2, t2 = synthetic_session(spec)
    assert np.allclose(g1.x, g2.x, equal_nan=True)
    assert np.allclose(g1.y, g2.y, equal_nan=True)
    assert np.allclose(p1.size, p2.size, equal_nan=True)
    assert t1.saccades == t2.saccades


def test_synthetic_session_returns_truth_and_recoverable_events() -> None:
    gaze, pupil, truth = synthetic_session(
        SyntheticSessionSpec(seed=6, n_saccades=5, noise_deg=0.0, pupil_noise=0.0)
    )
    cfg = AnalysisConfig(detection=DetectionConfig(method="adaptive_ivt", include_pso=True))
    report = pipeline.analyze_session(gaze, pupil, config=cfg)
    assert len(truth.saccades) == 5
    assert len(report.saccades) >= 3
    assert report.pupil["n_blinks"] >= 1.0
    assert report.quality["finite_sample_fraction"] == 1.0


def test_synthetic_session_dropout_marks_quality_fraction() -> None:
    gaze, _pupil, _truth = synthetic_session(SyntheticSessionSpec(dropout_fraction=0.1, seed=1))
    report = pipeline.analyze_gaze(gaze)
    assert report.quality["finite_sample_fraction"] < 1.0


def test_synthetic_session_supports_timestamp_jitter_correlated_noise_and_quality_flags() -> None:
    spec = synthetic.SyntheticSessionSpec(
        seed=11,
        timestamp_jitter_s=0.0005,
        correlated_noise_deg=0.04,
        head_pose_drift_deg=1.5,
        lighting_dropouts_s=((1.0, 1.2),),
    )

    gaze1, _pupil1, truth1 = synthetic.synthetic_session(spec)
    gaze2, _pupil2, truth2 = synthetic.synthetic_session(spec)

    assert np.allclose(gaze1.t, gaze2.t)
    assert np.allclose(gaze1.x, gaze2.x, equal_nan=True)
    assert not np.allclose(np.diff(gaze1.t), np.median(np.diff(gaze1.t)))
    assert truth1.quality_flags["timestamp_jitter_s"] == 0.0005
    assert truth1.quality_flags == truth2.quality_flags
    assert truth1.lighting_dropouts_s == ((1.0, 1.2),)
