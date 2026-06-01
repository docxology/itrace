"""Tests for the multi-panel session dashboard (no mocks, real synthetic data)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from itrace import pipeline
from itrace.synthetic import gaze_fixation_noise, gaze_with_saccade, pupil_sine_with_blink
from itrace.types import GazeStream
from itrace.viz import dashboard


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


def test_session_dashboard_returns_six_panel_figure() -> None:
    gaze = _multi_saccade_stream()
    pstream, _ = pupil_sine_with_blink()
    report = pipeline.analyze_session(gaze, pstream)
    assert len(report.saccades) > 0

    fig = dashboard.session_dashboard(report, gaze, pstream)
    try:
        assert len(fig.axes) >= 6
        assert fig._suptitle is not None
    finally:
        plt.close(fig)


def test_session_dashboard_handles_zero_saccades() -> None:
    gaze = gaze_fixation_noise(duration_s=0.5, noise_deg=0.02)
    report = pipeline.analyze_session(gaze)
    assert len(report.saccades) == 0

    # Must not raise even though the fit / amplitude panels have no data.
    fig = dashboard.session_dashboard(report, gaze)
    try:
        assert len(fig.axes) >= 6
    finally:
        plt.close(fig)


def test_render_dashboard_writes_png(tmp_path) -> None:
    out = dashboard.render_dashboard(tmp_path / "dash.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_dashboard_is_deterministic(tmp_path) -> None:
    a = dashboard.render_dashboard(tmp_path / "a.png", seed=3)
    b = dashboard.render_dashboard(tmp_path / "b.png", seed=3)
    assert a.read_bytes() == b.read_bytes()
