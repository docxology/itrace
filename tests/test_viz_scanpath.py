"""Tests for itrace.viz.scanpath (no mocks; real synthetic data + temp files)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from itrace import saccades
from itrace.synthetic import gaze_with_microsaccade, gaze_with_saccade
from itrace.types import (
    Fixation,
    GazeStream,
    Microsaccade,
    SessionReport,
)
from itrace.viz import scanpath


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    """Concatenate several gaze_with_saccade outputs into one multi-saccade stream."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(4.0, 18.0, 6):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.08)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = ts[-1][-1] + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def _detected_microsaccades() -> list[Microsaccade]:
    """High-sampling microsaccade stream yielding >=1 detected microsaccade."""
    stream, _onset = gaze_with_microsaccade(sampling_rate_hz=500.0)
    micros = saccades.detect_microsaccades(stream)
    assert len(micros) >= 1
    return micros


def _png_is_nonempty(fig: Figure, path) -> None:
    fig.savefig(path)
    assert path.exists()
    assert path.stat().st_size > 0


# --- scanpath -------------------------------------------------------------


def test_plot_scanpath_returns_axes_and_inverts_y(tmp_path) -> None:
    stream = _multi_saccade_stream()
    fixations, saccs = saccades.detect_ivt(stream)
    assert len(fixations) >= 2

    ax = scanpath.plot_scanpath(fixations, saccs)
    assert isinstance(ax, Axes)
    # Screen coords: y must be inverted (top value > bottom value).
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""

    _png_is_nonempty(ax.figure, tmp_path / "scanpath.png")


def test_plot_scanpath_empty_fixations(tmp_path) -> None:
    ax = scanpath.plot_scanpath([], [])
    assert isinstance(ax, Axes)
    # Empty branch still inverts y and stays drawable.
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    _png_is_nonempty(ax.figure, tmp_path / "empty_scanpath.png")


def test_plot_scanpath_single_fixation_no_arrows() -> None:
    fix = Fixation(
        onset_idx=0,
        offset_idx=10,
        onset_t=0.0,
        offset_t=0.2,
        centroid_x=1.0,
        centroid_y=-2.0,
    )
    ax = scanpath.plot_scanpath([fix], [])
    assert isinstance(ax, Axes)
    # No connecting line for a single fixation (only the scatter collection).
    assert len(ax.lines) == 0


def test_figure_scanpath_from_report(tmp_path) -> None:
    stream = _multi_saccade_stream()
    fixations, saccs = saccades.detect_ivt(stream)
    report = SessionReport(
        n_samples=len(stream),
        duration_s=float(stream.t[-1] - stream.t[0]),
        fixations=fixations,
        saccades=saccs,
    )
    fig = scanpath.figure_scanpath(report)
    assert isinstance(fig, Figure)
    _png_is_nonempty(fig, tmp_path / "report_scanpath.png")


def test_figure_scanpath_accepts_figsize_opt() -> None:
    report = SessionReport(n_samples=0, duration_s=0.0, fixations=[], saccades=[])
    fig = scanpath.figure_scanpath(report, figsize=(4, 4))
    assert isinstance(fig, Figure)
    assert tuple(fig.get_size_inches()) == (4.0, 4.0)


# --- microsaccades --------------------------------------------------------


def test_plot_microsaccade_polar_is_polar(tmp_path) -> None:
    micros = _detected_microsaccades()
    ax = scanpath.plot_microsaccade_polar(micros)
    assert isinstance(ax, Axes)
    assert ax.name == "polar"
    # A polar histogram of N microsaccades produces bar patches.
    assert len(ax.patches) > 0
    _png_is_nonempty(ax.figure, tmp_path / "polar.png")


def test_plot_microsaccade_polar_empty() -> None:
    ax = scanpath.plot_microsaccade_polar([])
    assert isinstance(ax, Axes)
    assert ax.name == "polar"
    assert len(ax.patches) == 0


def test_plot_microsaccade_main_sequence(tmp_path) -> None:
    micros = _detected_microsaccades()
    ax = scanpath.plot_microsaccade_main_sequence(micros)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    # Scatter created a path collection.
    assert len(ax.collections) > 0
    _png_is_nonempty(ax.figure, tmp_path / "micro_seq.png")


def test_plot_microsaccade_main_sequence_empty() -> None:
    ax = scanpath.plot_microsaccade_main_sequence([])
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 0


def test_figure_microsaccades_two_panels(tmp_path) -> None:
    micros = _detected_microsaccades()
    fig = scanpath.figure_microsaccades(micros)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    # First panel is polar, second is rectilinear.
    assert fig.axes[0].name == "polar"
    assert fig.axes[1].name == "rectilinear"
    _png_is_nonempty(fig, tmp_path / "micro_fig.png")


def test_figure_microsaccades_empty_input() -> None:
    fig = scanpath.figure_microsaccades([])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_provided_ax_is_reused() -> None:
    # When an ax is passed, the function must draw on it, not create a new one.
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    returned = scanpath.plot_microsaccade_main_sequence([], ax=ax)
    assert returned is ax


@pytest.mark.parametrize("bins", [8, 24])
def test_polar_bins_parameter(bins: int) -> None:
    micros = _detected_microsaccades()
    ax = scanpath.plot_microsaccade_polar(micros, bins=bins)
    assert len(ax.patches) == bins
