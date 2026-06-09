"""Tests for the statistical-diagnostics composite figure."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)


def _multi_saccade_report(seed: int = 4):
    from itrace import pipeline
    from itrace.synthetic import gaze_with_saccade
    from itrace.types import GazeStream

    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 18):
        direction = float(rng.uniform(-180.0, 180.0))
        gaze, _truth = gaze_with_saccade(
            amplitude_deg=float(amp),
            direction_deg=direction,
            fixation_s=0.06,
        )
        xs.append(gaze.x)
        ys.append(gaze.y)
        ts.append(gaze.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    stream = GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))
    return pipeline.analyze_session(stream)


def test_figure_statistical_diagnostics_renders_synthetic_report(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz.statistics import figure_statistical_diagnostics

    report = _multi_saccade_report()
    fig = figure_statistical_diagnostics(report)
    try:
        assert isinstance(fig, Figure)
        titles = {ax.get_title() for ax in fig.axes}
        assert "Model comparison" in titles
        assert "Best-fit QQ" in titles
        assert "Main-sequence exponent" in titles
        assert "Spatial stability" in titles
        assert "Scanpath transitions" in titles
        labels = "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)
        assert "Bowley skew" in labels
        assert "w(AIC)" in labels
        assert "boot top=" in labels
        assert "median" in labels
        assert "CI" in labels
        assert "IQR outliers" in labels
        assert "QQ RMSE" in labels
        assert "P-P max |res|" in labels
        assert "CvM" in labels
        assert "AD" in labels
        assert "DKW 95%" in labels
        out = tmp_path / "statistical_diagnostics.png"
        fig.savefig(out)
        assert out.exists()
        assert out.stat().st_size > 1000
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_figure_statistical_diagnostics_handles_empty_report(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.types import SessionReport
    from itrace.viz.statistics import figure_statistical_diagnostics

    report = SessionReport(n_samples=0, duration_s=0.0, fixations=[], saccades=[])
    fig = figure_statistical_diagnostics(report)
    try:
        assert isinstance(fig, Figure)
        labels = "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)
        assert "need >=3 positive events" in labels
        assert "no fixations" in labels
        assert "no encoded saccades" in labels
        out = tmp_path / "empty_statistics.png"
        fig.savefig(out)
        assert out.stat().st_size > 1000
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
