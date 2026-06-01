"""Tests for the ``itrace stats`` and ``itrace figures`` CLI commands.

No mocks: commands are exercised end-to-end over real synthetic recordings
written to real temp CSVs, and their JSON / PNG outputs are read back.
"""

from __future__ import annotations

import json

import numpy as np

from itrace import cli, io
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink
from itrace.types import GazeStream


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    rng = np.random.default_rng(seed)
    xs, ys, ts = [], [], []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def test_stats_command_writes_full_summary(tmp_path) -> None:
    gaze_csv = io.write_gaze_csv(_multi_saccade_stream(), tmp_path / "gaze.csv")
    pstream, _ = pupil_sine_with_blink()
    pupil_csv = io.write_pupil_csv(pstream, tmp_path / "pupil.csv")
    out = tmp_path / "stats.json"

    cli.stats(gaze_csv=gaze_csv, out=out, pupil_csv=pupil_csv)

    data = json.loads(out.read_text())
    assert "descriptive" in data
    assert "scanpath" in data
    assert "amplitude_fit" in data  # >= 3 saccades present
    assert data["amplitude_fit"]["family"] == "gamma"
    assert data["descriptive"]["saccades"]["count"] >= 3.0


def test_stats_command_skips_fit_with_few_saccades(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)  # single saccade
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    out = tmp_path / "stats.json"

    cli.stats(gaze_csv=gaze_csv, out=out)

    data = json.loads(out.read_text())
    assert "amplitude_fit" not in data
    assert "descriptive" in data


def test_figures_command_writes_dashboard_png(tmp_path) -> None:
    cli.figures(out_dir=tmp_path, seed=1)
    png = tmp_path / "session_dashboard.png"
    assert png.exists()
    assert png.stat().st_size > 0
    for name in (
        "scanpath.png",
        "pupil_trace.png",
        "event_raster.png",
        "main_sequence_diagnostics.png",
        "dropout_raster.png",
        "sampling_intervals.png",
        "calibration_residuals.png",
        "pupil_velocity.png",
    ):
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_figures_command_writes_animation_when_requested(tmp_path) -> None:
    cli.figures(out_dir=tmp_path, seed=1, animations=True)
    gif = tmp_path / "synthetic_replay.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0
