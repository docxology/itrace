"""Tests for :mod:`itrace.viz.distributions` (no mocks; real data and figures).

Each test builds a real sample (deterministic ``default_rng`` gamma draws for
the histograms, an analytic power-law amplitude/velocity set for the main
sequence), exercises the plotting helper, and asserts the matplotlib object
type, the presence of legend text, and that a saved PNG is non-empty. The
too-few-samples and empty-input branches are covered explicitly so module line
coverage stays above the 90% project floor.

Following the repository convention (``test_orbs_power_figures.py``), the whole
module is skipped when matplotlib is absent and the matplotlib-dependent imports
are made inside each test function so the module imports cleanly without the
optional dependency.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)


def _gamma_sample(
    n: int = 200, *, shape: float = 2.0, scale: float = 0.15, seed: int = 7
) -> np.ndarray:
    """Deterministic positive sample resembling fixation durations (seconds)."""
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=shape, scale=scale, size=n)


def _power_law_main_sequence(n: int = 40, *, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Analytic main sequence ``V = a * A^b`` with small multiplicative noise."""
    rng = np.random.default_rng(seed)
    amp = np.linspace(1.0, 25.0, n)
    vel = 40.0 * amp**0.55
    vel = vel * (1.0 + rng.normal(0.0, 0.02, n))
    return amp, vel


def _png_nonempty(fig, tmp_path) -> None:
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Duration histogram                                                          #
# --------------------------------------------------------------------------- #
def test_plot_duration_histogram_returns_axes_with_legend() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    ax = viz.plot_duration_histogram(_gamma_sample(), family="gamma", bins=15)
    assert isinstance(ax, Axes)
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("observed" in lbl for lbl in labels)
    # The fitted-PDF overlay annotates the family in the legend.
    assert any("gamma" in lbl for lbl in labels)


def test_figure_duration_histogram_saves_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import distributions as viz

    fig = viz.figure_duration_histogram(_gamma_sample(), bins=20)
    assert isinstance(fig, Figure)
    _png_nonempty(fig, tmp_path)


def test_duration_histogram_too_few_samples_skips_overlay() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    # Two points: histogram is drawn, the PDF overlay is skipped.
    ax = viz.plot_duration_histogram([0.2, 0.3], bins=5)
    assert isinstance(ax, Axes)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("observed" in lbl for lbl in labels)
    assert not any("gamma" in lbl for lbl in labels)


def test_duration_histogram_empty_input_draws_placeholder() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    ax = viz.plot_duration_histogram([])
    assert isinstance(ax, Axes)
    # No legend is drawn for the no-data placeholder.
    assert ax.get_legend() is None


def test_duration_histogram_drops_nonpositive_and_nonfinite() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    raw = [0.1, -0.2, 0.0, np.nan, np.inf, 0.25, 0.3, 0.18]
    ax = viz.plot_duration_histogram(raw, bins=4)
    assert isinstance(ax, Axes)
    # Five valid positive points -> overlay present.
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("gamma" in lbl for lbl in labels)


# --------------------------------------------------------------------------- #
# Amplitude histogram                                                         #
# --------------------------------------------------------------------------- #
def test_plot_amplitude_histogram_returns_axes() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    amp, _vel = _power_law_main_sequence()
    ax = viz.plot_amplitude_histogram(amp, family="gamma", bins=10)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "amplitude (deg)"
    assert ax.get_legend() is not None


def test_figure_amplitude_histogram_saves_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import distributions as viz

    amp, _vel = _power_law_main_sequence()
    fig = viz.figure_amplitude_histogram(amp, bins=12)
    assert isinstance(fig, Figure)
    _png_nonempty(fig, tmp_path)


def test_amplitude_histogram_empty_input() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    ax = viz.plot_amplitude_histogram(np.array([], dtype=np.float64))
    assert isinstance(ax, Axes)
    assert ax.get_legend() is None


# --------------------------------------------------------------------------- #
# Main sequence                                                               #
# --------------------------------------------------------------------------- #
def test_plot_main_sequence_returns_axes_with_fit_annotation() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    amp, vel = _power_law_main_sequence()
    ax = viz.plot_main_sequence(amp, vel)
    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # The fitted exponent and log-log R^2 are reported in the legend.
    fit_label = next(lbl for lbl in labels if "b=" in lbl)
    assert "R^2" in fit_label


def test_plot_main_sequence_recovers_exponent() -> None:
    from itrace import mainsequence

    amp, vel = _power_law_main_sequence(n=60)
    fit = mainsequence.fit(amp, vel)
    assert abs(fit["power_b"] - 0.55) < 0.05


def test_plot_main_sequence_too_few_points_no_fit_line() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    # Two points: scatter only, no power-law overlay.
    ax = viz.plot_main_sequence([3.0, 9.0], [120.0, 230.0])
    assert isinstance(ax, Axes)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("saccades" in lbl for lbl in labels)
    assert not any("b=" in lbl for lbl in labels)


def test_plot_main_sequence_empty_input_placeholder() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    ax = viz.plot_main_sequence([], [])
    assert isinstance(ax, Axes)
    assert ax.get_legend() is None


def test_plot_main_sequence_residuals_returns_axes() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    amp, vel = _power_law_main_sequence()
    ax = viz.plot_main_sequence_residuals(amp, vel)
    assert isinstance(ax, Axes)
    assert ax.get_xscale() == "log"
    assert ax.get_ylabel() == "log-velocity residual"
    assert ax.get_legend() is not None


def test_plot_main_sequence_residuals_too_few_points() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import distributions as viz

    ax = viz.plot_main_sequence_residuals([3.0, 9.0], [120.0, 230.0])
    assert isinstance(ax, Axes)
    # Placeholder branch: no legend.
    assert ax.get_legend() is None


def test_figure_main_sequence_two_panels_and_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import distributions as viz

    amp, vel = _power_law_main_sequence()
    fig = viz.figure_main_sequence(amp, vel)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    _png_nonempty(fig, tmp_path)
