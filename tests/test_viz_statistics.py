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


def test_percentile_plot_keeps_endpoints_when_estimate_is_outside_interval() -> None:
    import matplotlib.pyplot as plt

    from itrace.viz.statistics import _plot_main_sequence_ci

    fig, ax = plt.subplots()
    try:
        _plot_main_sequence_ci(
            ax,
            {
                "available": True,
                "estimate": 0.7,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "n_boot": 20,
                "n_successful": 17,
                "block_length": 3,
            },
        )
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.4, 0.6])
        assert "valid=17" in " ".join(text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def test_resampling_figure_uses_core_intervals_and_writes_real_png(tmp_path) -> None:
    import matplotlib.pyplot as plt

    from itrace.stats.bootstrap import resampling_sensitivity
    from itrace.viz.resampling import figure_resampling_sensitivity

    payload = resampling_sensitivity(n_boot=30, seed=3)
    fig = figure_resampling_sensitivity(payload)
    try:
        assert len(fig.axes) == 3
        rows = payload["intervals"]
        np.testing.assert_array_equal(
            fig.axes[1].lines[0].get_xdata(), [rows[0]["ci_low"], rows[0]["ci_high"]]
        )
        path = tmp_path / "resampling.png"
        fig.savefig(path)
        assert path.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_transition_plot_marks_unobserved_rows_as_missing() -> None:
    import matplotlib.pyplot as plt

    from itrace.viz.statistics import _plot_transition_matrix

    fig, ax = plt.subplots()
    try:
        _plot_transition_matrix(
            ax,
            {
                "symbols": ["L", "R"],
                "matrix": [[0.0, 1.0], [0.0, 0.0]],
                "row_counts": [1, 0],
                "scanpath": "LR",
            },
        )
        masked = ax.images[0].get_array()
        assert np.all(masked.mask[1])
        assert not np.any(masked.mask[0])
        assert "R (n=0)" in [tick.get_text() for tick in ax.get_yticklabels()]
    finally:
        plt.close(fig)
