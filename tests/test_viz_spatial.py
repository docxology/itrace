"""Tests for itrace.viz.spatial (no mocks; real synthetic data + temp files)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from itrace import saccades
from itrace.synthetic import gaze_with_saccade
from itrace.types import Fixation, GazeStream
from itrace.viz import spatial


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


def _detected_fixations() -> list[Fixation]:
    stream = _multi_saccade_stream()
    fixations, _saccs = saccades.detect_ivt(stream)
    assert len(fixations) >= 2
    return fixations


def _grid_aois() -> list[dict]:
    """A 2x2 AOI grid covering the centre of the (screen) plane."""
    return [
        {"name": "top_left", "x": -10.0, "y": -10.0, "w": 10.0, "h": 10.0},
        {"name": "top_right", "x": 0.0, "y": -10.0, "w": 10.0, "h": 10.0},
        {"name": "bottom_left", "x": -10.0, "y": 0.0, "w": 10.0, "h": 10.0},
        {"name": "bottom_right", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0},
    ]


def _png_is_nonempty(fig: Figure, path) -> None:
    fig.savefig(path)
    assert path.exists()
    assert path.stat().st_size > 0


# --- assign_aoi -----------------------------------------------------------


def test_assign_aoi_membership() -> None:
    aois = _grid_aois()
    assert spatial.assign_aoi(-5.0, -5.0, aois) == "top_left"
    assert spatial.assign_aoi(5.0, -5.0, aois) == "top_right"
    assert spatial.assign_aoi(-5.0, 5.0, aois) == "bottom_left"
    assert spatial.assign_aoi(5.0, 5.0, aois) == "bottom_right"


def test_assign_aoi_outside_returns_none() -> None:
    aois = _grid_aois()
    assert spatial.assign_aoi(100.0, 100.0, aois) is None
    assert spatial.assign_aoi(-50.0, 0.0, aois) is None


def test_assign_aoi_empty_list_is_none() -> None:
    assert spatial.assign_aoi(0.0, 0.0, []) is None


def test_assign_aoi_inclusive_boundary() -> None:
    aois = [{"name": "box", "x": 0.0, "y": 0.0, "w": 4.0, "h": 4.0}]
    assert spatial.assign_aoi(0.0, 0.0, aois) == "box"
    assert spatial.assign_aoi(4.0, 4.0, aois) == "box"
    assert spatial.assign_aoi(4.001, 4.0, aois) is None


def test_assign_aoi_negative_size_normalised() -> None:
    # Rectangle given by an upper corner with negative w/h must still match.
    aois = [{"name": "neg", "x": 4.0, "y": 4.0, "w": -4.0, "h": -4.0}]
    assert spatial.assign_aoi(2.0, 2.0, aois) == "neg"


def test_assign_aoi_first_match_wins() -> None:
    aois = [
        {"name": "a", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0},
        {"name": "b", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0},
    ]
    assert spatial.assign_aoi(5.0, 5.0, aois) == "a"


# --- fixation_heatmap -----------------------------------------------------


def test_fixation_heatmap_returns_axes_and_png(tmp_path) -> None:
    fixations = _detected_fixations()
    ax = spatial.fixation_heatmap(fixations)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    # imshow produced an image.
    assert len(ax.images) == 1
    _png_is_nonempty(ax.figure, tmp_path / "heatmap.png")


def test_fixation_heatmap_respects_extent() -> None:
    fixations = _detected_fixations()
    ax = spatial.fixation_heatmap(fixations, bins=10, extent=(-20.0, 20.0, -20.0, 20.0))
    assert len(ax.images) == 1
    img = ax.images[0]
    left, right, bottom, top = img.get_extent()
    assert left == pytest.approx(-20.0)
    assert right == pytest.approx(20.0)
    # Screen convention: top edge value < bottom edge value.
    assert top < bottom


def test_fixation_heatmap_empty() -> None:
    ax = spatial.fixation_heatmap([])
    assert isinstance(ax, Axes)
    assert len(ax.images) == 0
    assert ax.get_xlabel() != ""


def test_figure_fixation_heatmap(tmp_path) -> None:
    fixations = _detected_fixations()
    fig = spatial.figure_fixation_heatmap(fixations, bins=20)
    assert isinstance(fig, Figure)
    _png_is_nonempty(fig, tmp_path / "heatmap_fig.png")


def test_fixation_heatmap_provided_ax_reused() -> None:
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    returned = spatial.fixation_heatmap([], ax=ax)
    assert returned is ax


# --- gaze_density ---------------------------------------------------------


def test_gaze_density_returns_axes_and_png(tmp_path) -> None:
    stream = _multi_saccade_stream()
    ax = spatial.gaze_density(stream.x, stream.y)
    assert isinstance(ax, Axes)
    # hexbin creates a PolyCollection.
    assert len(ax.collections) > 0
    # Screen convention: y inverted.
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    _png_is_nonempty(ax.figure, tmp_path / "density.png")


def test_gaze_density_empty() -> None:
    ax = spatial.gaze_density([], [])
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 0
    assert ax.get_xlabel() != ""


def test_gaze_density_mismatched_lengths_truncate() -> None:
    ax = spatial.gaze_density([0.0, 1.0, 2.0], [0.0, 1.0])
    # Two valid paired points still draw without error.
    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0


def test_gaze_density_drops_nonfinite() -> None:
    ax = spatial.gaze_density([np.nan, np.inf], [0.0, 1.0])
    # All points dropped -> empty branch.
    assert len(ax.collections) == 0


def test_figure_gaze_density(tmp_path) -> None:
    stream = _multi_saccade_stream()
    fig = spatial.figure_gaze_density(stream.x, stream.y, gridsize=20)
    assert isinstance(fig, Figure)
    _png_is_nonempty(fig, tmp_path / "density_fig.png")


# --- AOI dwell ------------------------------------------------------------


def test_plot_aoi_dwell_sums_partition_total(tmp_path) -> None:
    fixations = _detected_fixations()
    aois = _grid_aois()
    ax = spatial.plot_aoi_dwell(fixations, aois)
    assert isinstance(ax, Axes)
    # One bar per AOI plus the outside bucket.
    assert len(ax.patches) == len(aois) + 1
    bar_total = float(sum(p.get_height() for p in ax.patches))
    fixation_total = float(sum(f.duration_s for f in fixations))
    # Every fixation is attributed to exactly one bucket: the partition is exact.
    assert bar_total == pytest.approx(fixation_total)
    _png_is_nonempty(ax.figure, tmp_path / "dwell.png")


def test_plot_aoi_dwell_all_outside() -> None:
    fixations = _detected_fixations()
    # AOI far from any centroid -> all dwell falls into the outside bucket.
    aois = [{"name": "far", "x": 1000.0, "y": 1000.0, "w": 1.0, "h": 1.0}]
    labels, dwell = spatial._aoi_dwell_times(fixations, aois)
    assert labels[-1] == spatial.OUTSIDE_LABEL
    assert dwell[0] == pytest.approx(0.0)
    assert dwell[-1] == pytest.approx(float(sum(f.duration_s for f in fixations)))


def test_plot_aoi_dwell_no_aois_has_outside_bar() -> None:
    fixations = _detected_fixations()
    ax = spatial.plot_aoi_dwell(fixations, [])
    # Only the outside bucket is present.
    assert len(ax.patches) == 1


def test_plot_aoi_dwell_provided_ax_reused() -> None:
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    returned = spatial.plot_aoi_dwell([], [], ax=ax)
    assert returned is ax


def test_figure_aoi_two_panels(tmp_path) -> None:
    fixations = _detected_fixations()
    aois = _grid_aois()
    fig = spatial.figure_aoi(fixations, aois)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    # Left panel carries the AOI rectangle patches.
    assert len(fig.axes[0].patches) == len(aois)
    # Right panel carries the dwell bars.
    assert len(fig.axes[1].patches) == len(aois) + 1
    _png_is_nonempty(fig, tmp_path / "aoi_fig.png")


def test_figure_aoi_empty_inputs() -> None:
    fig = spatial.figure_aoi([], [])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
