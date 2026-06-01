"""Tests for the TODO-roadmap CLI additions."""

from __future__ import annotations

import csv
import json

import numpy as np

from itrace import cli, io
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink


def test_analyze_config_json_and_cli_precedence(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "detection": {
                    "method": "ivt",
                    "velocity_threshold_deg_s": 9999.0,
                    "include_microsaccades": False,
                }
            }
        )
    )

    config_only = tmp_path / "config_only.json"
    cli.analyze(gaze_csv=gaze_csv, out=config_only, config_json=config_json)
    data = json.loads(config_only.read_text())
    assert data["n_saccades"] == 0
    assert data["config"]["detection"]["velocity_threshold_deg_s"] == 9999.0

    explicit = tmp_path / "explicit.json"
    cli.analyze(gaze_csv=gaze_csv, out=explicit, velocity_threshold=30.0, config_json=config_json)
    data = json.loads(explicit.read_text())
    assert data["n_saccades"] == 1
    assert data["config"]["detection"]["velocity_threshold_deg_s"] == 30.0
    assert data["config"]["detection"]["include_microsaccades"] is False


def test_stats_command_includes_quality_pupil_and_ci(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0, noise_deg=0.01)
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    pupil_stream, _ = pupil_sine_with_blink()
    pupil_csv = io.write_pupil_csv(pupil_stream, tmp_path / "pupil.csv")
    out = tmp_path / "stats.json"

    cli.stats(gaze_csv=gaze_csv, out=out, pupil_csv=pupil_csv)

    data = json.loads(out.read_text())
    assert "quality" in data
    assert "valid_sample_fraction" in data["quality"]
    assert "pupil" in data
    assert "valid_sample_fraction" in data["pupil"]
    assert "event_ci" in data
    assert "saccade_duration_mean_ci_s" in data["event_ci"]


def test_validate_recording_writes_warnings_and_quality(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)
    x = gaze.x.copy()
    x[2] = np.nan
    gaze = type(gaze)(t=gaze.t, x=x, y=gaze.y)
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    pupil_stream, _ = pupil_sine_with_blink(blink_window_s=(0.2, 0.4))
    pupil_csv = io.write_pupil_csv(pupil_stream, tmp_path / "pupil.csv")
    out = tmp_path / "validation.json"

    cli.validate_recording(gaze_csv=gaze_csv, pupil_csv=pupil_csv, out=out)

    data = json.loads(out.read_text())
    assert data["n_samples"] == len(gaze)
    assert data["quality"]["dropout_fraction"] > 0.0
    assert data["pupil"]["blink_fraction"] > 0.0
    assert data["calibration"]["available"] is False
    assert data["warnings"]
    assert data["errors"] == []


def test_calibrate_command_writes_model_and_calibrated_gaze(tmp_path) -> None:
    points = tmp_path / "points.csv"
    with points.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["raw_x", "raw_y", "target_x", "target_y"])
        writer.writeheader()
        for raw_x, raw_y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            writer.writerow(
                {
                    "raw_x": raw_x,
                    "raw_y": raw_y,
                    "target_x": 2.0 * raw_x + 1.0,
                    "target_y": 3.0 * raw_y - 1.0,
                }
            )
    gaze, _ = gaze_with_saccade(amplitude_deg=1.0, fixation_s=0.02)
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "raw_gaze.csv")
    calibrated_csv = tmp_path / "calibrated_gaze.csv"
    out = tmp_path / "calibration.json"

    cli.calibrate(points_csv=points, out=out, apply_gaze=gaze_csv, calibrated_out=calibrated_csv)

    payload = json.loads(out.read_text())
    assert payload["n_points"] == 4
    assert payload["rms_error_deg"] < 1e-9
    corrected = io.read_gaze_csv(calibrated_csv)
    assert np.allclose(corrected.x, 2.0 * gaze.x + 1.0)
    assert np.allclose(corrected.y, 3.0 * gaze.y - 1.0)
