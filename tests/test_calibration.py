"""Gaze calibration and quality-summary tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from itrace.calibration import AffineCalibration, calibration_error, robust_gaze_quality
from itrace.types import GazeStream

FIXTURES = Path(__file__).parent / "fixtures" / "calibration"


def test_affine_calibration_recovers_known_transform() -> None:
    raw_x = np.array([-2.0, 0.0, 2.0, -2.0, 2.0], dtype=np.float64)
    raw_y = np.array([-1.0, 0.0, 1.0, 2.0, -2.0], dtype=np.float64)
    target_x = 1.5 * raw_x - 0.25 * raw_y + 3.0
    target_y = 0.5 * raw_x + 2.0 * raw_y - 1.0

    cal = AffineCalibration.fit(raw_x, raw_y, target_x, target_y)
    corrected = cal.apply(raw_x, raw_y)

    assert np.allclose(corrected[0], target_x)
    assert np.allclose(corrected[1], target_y)
    assert cal.n_points == raw_x.size
    assert cal.rms_error_deg < 1e-9


def test_affine_calibration_applies_to_gaze_stream() -> None:
    raw = GazeStream(
        t=np.array([0.0, 0.1, 0.2]),
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([2.0, 1.0, 0.0]),
    )
    cal = AffineCalibration.fit(raw.x, raw.y, raw.x + 10.0, raw.y - 5.0)

    corrected = cal.apply_stream(raw)

    assert isinstance(corrected, GazeStream)
    assert np.allclose(corrected.t, raw.t)
    assert np.allclose(corrected.x, raw.x + 10.0)
    assert np.allclose(corrected.y, raw.y - 5.0)


def test_affine_calibration_rejects_underdetermined_or_bad_inputs() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        AffineCalibration.fit([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="equal-length"):
        AffineCalibration.fit([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        AffineCalibration.fit([0.0, 1.0, np.nan], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])


def test_calibration_error_reports_rms_and_percentiles() -> None:
    raw_x = np.array([0.0, 1.0, 2.0])
    raw_y = np.array([0.0, 0.0, 0.0])
    target_x = np.array([0.0, 2.0, 4.0])
    target_y = np.array([0.0, 0.0, 0.0])
    cal = AffineCalibration.fit(raw_x, raw_y, target_x, target_y)

    err = calibration_error(cal, raw_x, raw_y, target_x + 1.0, target_y)

    assert err["n_points"] == 3.0
    assert err["rms_error_deg"] == pytest.approx(1.0)
    assert err["p95_error_deg"] == pytest.approx(1.0)


def test_calibrated_vs_uncalibrated_fixture_matches_expected_json() -> None:
    rows = list(csv.DictReader((FIXTURES / "calibrated_vs_uncalibrated_points.csv").open()))
    raw_x = [float(row["raw_x"]) for row in rows]
    raw_y = [float(row["raw_y"]) for row in rows]
    target_x = [float(row["target_x"]) for row in rows]
    target_y = [float(row["target_y"]) for row in rows]

    calibration = AffineCalibration.fit(raw_x, raw_y, target_x, target_y)
    error = calibration_error(calibration, raw_x, raw_y, target_x, target_y)
    uncalibrated_error = np.hypot(
        np.asarray(raw_x, dtype=np.float64) - np.asarray(target_x, dtype=np.float64),
        np.asarray(raw_y, dtype=np.float64) - np.asarray(target_y, dtype=np.float64),
    )
    expected = json.loads((FIXTURES / "expected_calibration_error.json").read_text())

    assert float(np.sqrt(np.mean(uncalibrated_error**2))) > 1.5
    for key, value in expected.items():
        assert error[key] == pytest.approx(value)


def test_robust_gaze_quality_summarizes_dropouts_jitter_and_gaps() -> None:
    stream = GazeStream(
        t=np.array([0.0, 0.01, 0.02, 0.08, 0.09, 0.10]),
        x=np.array([0.0, 0.1, np.nan, 0.4, 0.5, 0.6]),
        y=np.array([0.0, 0.1, 0.2, 0.4, np.inf, 0.6]),
    )

    summary = robust_gaze_quality(stream)

    assert summary["n_samples"] == 6.0
    assert summary["valid_sample_fraction"] == pytest.approx(4 / 6)
    assert summary["dropout_fraction"] == pytest.approx(2 / 6)
    assert summary["median_dt_s"] == pytest.approx(0.01)
    assert summary["longest_gap_s"] == pytest.approx(0.06)
    assert summary["large_gap_count"] == 1.0
    assert summary["sampling_jitter_s"] > 0.0


def test_interpolate_gaze_gaps_fills_short_bounded_invalid_runs_only() -> None:
    from itrace.calibration import interpolate_gaze_gaps

    stream = GazeStream(
        t=np.array([0.0, 0.01, 0.02, 0.03, 0.20]),
        x=np.array([0.0, np.nan, 2.0, 3.0, np.nan]),
        y=np.array([0.0, np.nan, 2.0, 3.0, 4.0]),
    )

    filled = interpolate_gaze_gaps(stream, max_gap_s=0.02)

    assert np.isfinite(filled.x[1])
    assert filled.x[1] == pytest.approx(1.0)
    assert np.isnan(filled.x[-1])


def test_interpolate_gaze_gaps_rejects_long_bounded_gap() -> None:
    from itrace.calibration import interpolate_gaze_gaps

    stream = GazeStream(
        t=np.array([0.0, 0.01, 0.05, 0.10]),
        x=np.array([0.0, np.nan, np.nan, 10.0]),
        y=np.array([0.0, np.nan, np.nan, 0.0]),
    )

    with pytest.raises(ValueError, match="exceeds max_gap_s"):
        interpolate_gaze_gaps(stream, max_gap_s=0.02)
