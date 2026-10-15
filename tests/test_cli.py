"""CLI tests via typer's CliRunner (ISC-48..49)."""

from __future__ import annotations

import json
import os
import re

import pytest
from typer.testing import CliRunner

from itrace import cli, io
from itrace.cli import app
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink

runner = CliRunner()


def _plain_help(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_demo_exits_zero_and_reports_events() -> None:  # ISC-48
    result = runner.invoke(app, ["demo", "--amplitude", "12"])
    assert result.exit_code == 0, result.output
    assert "saccades=1" in result.output
    assert "scanpath=" in result.output


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip().count(".") >= 2


def test_commands_registered() -> None:  # ISC-48
    result = runner.invoke(app, ["--help"])
    for cmd in (
        "demo",
        "analyze",
        "stats",
        "calibrate",
        "validate-recording",
        "record",
        "dashboard",
        "camera-probe",
        "live-html",
        "experiment-report",
    ):
        assert cmd in result.output


@pytest.mark.parametrize("color", [False, True])
def test_capture_cli_options_registered(color: bool) -> None:
    environment = {
        "TERM": "xterm-256color",
        "NO_COLOR": None if color else "1",
        "FORCE_COLOR": "1" if color else None,
    }
    record = runner.invoke(app, ["record", "--help"], color=color, env=environment)
    probe = runner.invoke(app, ["camera-probe", "--help"], color=color, env=environment)
    live = runner.invoke(app, ["live-html", "--help"], color=color, env=environment)
    analyze = runner.invoke(app, ["analyze", "--help"], color=color, env=environment)
    stats = runner.invoke(app, ["stats", "--help"], color=color, env=environment)
    experiment = runner.invoke(app, ["experiment-report", "--help"], color=color, env=environment)
    if color:
        assert record.output != _plain_help(record.output)
    assert record.exit_code == 0
    assert probe.exit_code == 0
    assert live.exit_code == 0
    assert analyze.exit_code == 0
    assert stats.exit_code == 0
    assert experiment.exit_code == 0
    assert "records-out" in _plain_help(record.output)
    assert "backend-logs" in _plain_help(record.output)
    assert "backend-logs" in _plain_help(probe.output)
    for option in ("camera", "host", "port", "output-dir", "backend-logs", "open-browser"):
        assert option in _plain_help(live.output)
    assert "config-json" in _plain_help(analyze.output)
    assert "config-json" in _plain_help(stats.output)
    assert "capture_records_csv" in _plain_help(experiment.output)


def test_native_stderr_suppression_modes(capfd) -> None:
    with cli._native_stderr(backend_logs=True):
        os.write(2, b"visible-native-log")
    assert "visible-native-log" in capfd.readouterr().err

    with cli._native_stderr(backend_logs=False):
        os.write(2, b"hidden-native-log")
    assert "hidden-native-log" not in capfd.readouterr().err


def test_cli_validate_recording_invokes_refactored_core(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)
    gaze_csv = tmp_path / "gaze.csv"
    out = tmp_path / "val.json"
    io.write_gaze_csv(gaze, gaze_csv)

    result = runner.invoke(app, ["validate-recording", str(gaze_csv), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_samples"] > 0
    assert "quality" in payload


def test_analyze_writes_json_report(tmp_path) -> None:  # ISC-49
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)
    pstream, _ = pupil_sine_with_blink()
    gaze_csv = tmp_path / "gaze.csv"
    pupil_csv = tmp_path / "pupil.csv"
    out = tmp_path / "report.json"
    io.write_gaze_csv(gaze, gaze_csv)
    io.write_pupil_csv(pstream, pupil_csv)

    result = runner.invoke(
        app,
        ["analyze", str(gaze_csv), "--out", str(out), "--pupil-csv", str(pupil_csv)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["n_saccades"] >= 1
    assert "saccades" in report
    assert report["pupil"]  # pupil section present
