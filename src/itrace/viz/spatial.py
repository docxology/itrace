"""Spatial density and area-of-interest (AOI) visualisations.

This is a *viz* module: :mod:`matplotlib` is a hard import here (it is an
optional dependency of the package as a whole) and the non-interactive ``Agg``
backend is selected at import time so the plots render headless in CI and in the
figure pipeline. Stats/analysis modules must never import this module.

Three families of plots live here:

* **Fixation heatmap** -- a 2-D histogram of fixation centroids weighted by
  dwell time, shown with :func:`~matplotlib.axes.Axes.imshow` in screen
  coordinates (origin upper, ``y`` increasing downward) with a colourbar.
* **Gaze density** -- a hexbin 2-D density of raw gaze samples.
* **AOI dwell** -- total dwell time per rectangular area of interest (plus an
  ``"outside"`` bucket), with a companion two-panel figure that overlays the AOI
  rectangles on a scanpath alongside the dwell bars.

Every public plotting function accepts an optional ``ax`` so panels can be
composed into larger figures, and degrades gracefully on empty input. The
``figure_*`` wrappers build a standalone :class:`~matplotlib.figure.Figure`.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..types import Fixation, FloatArray
from .scanpath import plot_scanpath

from .palette import WONG  # single-source Wong (2011) colour-blind-safe palette

# Bucket label for fixations that fall in no AOI rectangle.
OUTSIDE_LABEL = "outside"


def _new_axes(ax: Axes | None, *, figsize: tuple[float, float] = (6, 5)) -> Axes:
    """Return ``ax`` if given, else a fresh single-axes figure's Axes.

    Parameters
    ----------
    ax
        An existing Axes to draw onto, or ``None`` to create a new figure.
    figsize
        Figure size in inches used only when a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes to draw onto.
    """
    if ax is not None:
        return ax
    _fig, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def assign_aoi(centroid_x: float, centroid_y: float, aois: list[dict[str, Any]]) -> str | None:
    """Return the name of the first AOI rectangle containing a centroid.

    Each AOI is a mapping ``{"name", "x", "y", "w", "h"}`` describing an
    axis-aligned rectangle with lower corner ``(x, y)`` and size ``(w, h)`` in
    the same (screen, degrees) coordinates as the fixation centroids. Negative
    widths/heights are normalised so a rectangle is never silently empty. Bounds
    are inclusive on both edges. The first matching AOI (in list order) wins, so
    callers control precedence for overlapping regions.

    Parameters
    ----------
    centroid_x
        Horizontal centroid coordinate.
    centroid_y
        Vertical (screen, downward-positive) centroid coordinate.
    aois
        Ordered list of AOI dictionaries.

    Returns
    -------
    str or None
        The ``name`` of the containing AOI, or ``None`` if the point lies
        outside every rectangle.
    """
    for aoi in aois:
        x = float(aoi["x"])
        y = float(aoi["y"])
        w = float(aoi["w"])
        h = float(aoi["h"])
        x0, x1 = (x, x + w) if w >= 0 else (x + w, x)
        y0, y1 = (y, y + h) if h >= 0 else (y + h, y)
        if x0 <= centroid_x <= x1 and y0 <= centroid_y <= y1:
            return str(aoi["name"])
    return None


def _fixation_centroids(fixations: list[Fixation]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return centroid-x, centroid-y and dwell-time arrays for a fixation list."""
    xs = np.array([f.centroid_x for f in fixations], dtype=np.float64)
    ys = np.array([f.centroid_y for f in fixations], dtype=np.float64)
    durs = np.array([f.duration_s for f in fixations], dtype=np.float64)
    return xs, ys, durs


def _annotate_empty(ax: Axes, message: str, *, xlabel: str, ylabel: str, title: str) -> Axes:
    """Label and annotate an Axes that has no data to draw."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=WONG[3])
    return ax


def fixation_heatmap(
    fixations: list[Fixation],
    *,
    bins: int = 40,
    extent: tuple[float, float, float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Dwell-weighted 2-D histogram of fixation centroids.

    Centroids are binned into a ``bins x bins`` grid weighted by each fixation's
    ``duration_s`` and displayed with :func:`~matplotlib.axes.Axes.imshow` using
    ``origin="upper"`` (screen coordinates, ``y`` increasing downward) and a
    colourbar reporting accumulated dwell time per cell.

    Parameters
    ----------
    fixations
        Fixations whose ``centroid_x``/``centroid_y`` give positions and
        ``duration_s`` the per-fixation weight.
    bins
        Number of bins along each axis.
    extent
        Optional ``(xmin, xmax, ymin, ymax)`` histogram range. When ``None`` the
        range is taken from the data span.
    ax
        Target Axes. A new figure/Axes is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the heatmap. With no fixations a labelled,
        annotated, image-free Axes is returned.
    """
    ax = _new_axes(ax)
    xlabel, ylabel = "horizontal gaze (deg)", "vertical gaze (deg, screen)"
    title = "Fixation dwell heatmap"

    if not fixations:
        return _annotate_empty(ax, "no fixations", xlabel=xlabel, ylabel=ylabel, title=title)

    xs, ys, durs = _fixation_centroids(fixations)
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        hist_range: list[list[float]] | None = [[xmin, xmax], [ymin, ymax]]
    else:
        hist_range = None

    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=bins, range=hist_range, weights=durs)
    # histogram2d returns hist[x_index, y_index]; transpose so rows map to y for
    # imshow, then use origin="upper" to keep the screen-down convention.
    img = ax.imshow(
        hist.T,
        origin="upper",
        aspect="auto",
        extent=(
            float(x_edges[0]),
            float(x_edges[-1]),
            float(y_edges[-1]),
            float(y_edges[0]),
        ),
        cmap="magma",
    )
    ax.figure.colorbar(img, ax=ax, label="dwell time (s)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def figure_fixation_heatmap(
    fixations: list[Fixation],
    *,
    bins: int = 40,
    extent: tuple[float, float, float, float] | None = None,
) -> Figure:
    """Standalone figure wrapping :func:`fixation_heatmap`.

    Parameters
    ----------
    fixations
        Fixations to bin.
    bins
        Number of bins along each axis.
    extent
        Optional ``(xmin, xmax, ymin, ymax)`` histogram range.

    Returns
    -------
    matplotlib.figure.Figure
        A tight-laid-out figure with one heatmap Axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fixation_heatmap(fixations, bins=bins, extent=extent, ax=ax)
    fig.tight_layout()
    return fig


def gaze_density(
    xs: FloatArray | list[float],
    ys: FloatArray | list[float],
    *,
    gridsize: int = 40,
    ax: Axes | None = None,
) -> Axes:
    """Hexbin 2-D density of raw gaze points in screen coordinates.

    Parameters
    ----------
    xs
        Horizontal gaze samples (degrees).
    ys
        Vertical (screen, downward-positive) gaze samples (degrees).
    gridsize
        Number of hexagons in the x-direction (forwarded to
        :func:`~matplotlib.axes.Axes.hexbin`).
    ax
        Target Axes. A new figure/Axes is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the density. The y-axis is inverted for the screen
        convention. Empty or mismatched-length input yields a labelled,
        annotated, data-free Axes.
    """
    ax = _new_axes(ax)
    xlabel, ylabel = "horizontal gaze (deg)", "vertical gaze (deg, screen)"
    title = "Gaze density"

    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if x.size == 0:
        return _annotate_empty(ax, "no gaze", xlabel=xlabel, ylabel=ylabel, title=title)

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
    ax.figure.colorbar(hb, ax=ax, label="samples")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    return ax


def figure_gaze_density(
    xs: FloatArray | list[float],
    ys: FloatArray | list[float],
    *,
    gridsize: int = 40,
) -> Figure:
    """Standalone figure wrapping :func:`gaze_density`.

    Parameters
    ----------
    xs
        Horizontal gaze samples (degrees).
    ys
        Vertical gaze samples (degrees, screen).
    gridsize
        Hexbin grid size.

    Returns
    -------
    matplotlib.figure.Figure
        A tight-laid-out figure with one density Axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    gaze_density(xs, ys, gridsize=gridsize, ax=ax)
    fig.tight_layout()
    return fig


def _aoi_dwell_times(
    fixations: list[Fixation],
    aois: list[dict[str, Any]],
) -> tuple[list[str], FloatArray]:
    """Total dwell time per AOI name plus an ``"outside"`` bucket.

    Parameters
    ----------
    fixations
        Fixations to attribute to AOIs by their centroids.
    aois
        Ordered AOI rectangles.

    Returns
    -------
    tuple of (list[str], FloatArray)
        The bucket labels (AOI names in input order, then ``"outside"``) and the
        matching summed dwell times in seconds.
    """
    labels = [str(aoi["name"]) for aoi in aois]
    labels.append(OUTSIDE_LABEL)
    totals = dict.fromkeys(labels, 0.0)
    for fix in fixations:
        name = assign_aoi(fix.centroid_x, fix.centroid_y, aois)
        totals[name if name is not None else OUTSIDE_LABEL] += fix.duration_s
    dwell = np.array([totals[label] for label in labels], dtype=np.float64)
    return labels, dwell


def plot_aoi_dwell(
    fixations: list[Fixation],
    aois: list[dict[str, Any]],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of total dwell time per AOI (with an ``"outside"`` bucket).

    Each fixation is attributed to the AOI whose rectangle contains its centroid
    (via :func:`assign_aoi`); fixations in no AOI accumulate into the
    ``"outside"`` bucket. Bar heights are the summed ``duration_s`` per bucket.

    Parameters
    ----------
    fixations
        Fixations to attribute.
    aois
        Ordered AOI rectangles ``{"name", "x", "y", "w", "h"}``.
    ax
        Target Axes. A new figure/Axes is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the dwell bars. An ``"outside"`` bar is always
        present, so the Axes is never data-free even with no AOIs.
    """
    ax = _new_axes(ax, figsize=(6, 4))
    labels, dwell = _aoi_dwell_times(fixations, aois)
    positions = np.arange(len(labels), dtype=np.float64)
    colours = [WONG[i % len(WONG)] for i in range(len(labels))]
    ax.bar(positions, dwell, color=colours, edgecolor="white")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("dwell time (s)")
    ax.set_title("AOI dwell time")
    return ax


def _draw_aoi_rectangles(ax: Axes, aois: list[dict[str, Any]]) -> None:
    """Overlay each AOI as a labelled outlined rectangle on ``ax``."""
    from matplotlib.patches import Rectangle

    for i, aoi in enumerate(aois):
        colour = WONG[i % len(WONG)]
        rect = Rectangle(
            (float(aoi["x"]), float(aoi["y"])),
            float(aoi["w"]),
            float(aoi["h"]),
            fill=False,
            edgecolor=colour,
            lw=1.5,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.text(
            float(aoi["x"]),
            float(aoi["y"]),
            str(aoi["name"]),
            color=colour,
            fontsize=8,
            va="bottom",
            zorder=5,
        )


def figure_aoi(fixations: list[Fixation], aois: list[dict[str, Any]]) -> Figure:
    """Two-panel AOI figure: scanpath with AOI overlays and a dwell bar chart.

    The left panel draws the scanpath (via :func:`itrace.viz.scanpath.plot_scanpath`)
    with each AOI rectangle overlaid in screen coordinates; the right panel is
    the per-AOI dwell bar chart from :func:`plot_aoi_dwell`.

    Parameters
    ----------
    fixations
        Fixations shared by both panels.
    aois
        Ordered AOI rectangles ``{"name", "x", "y", "w", "h"}``.

    Returns
    -------
    matplotlib.figure.Figure
        A 1x2 figure (scanpath + AOI overlays, dwell bars).
    """
    fig, (ax_scan, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    plot_scanpath(fixations, [], ax=ax_scan)
    _draw_aoi_rectangles(ax_scan, aois)
    ax_scan.set_title("Scanpath with AOIs")
    plot_aoi_dwell(fixations, aois, ax=ax_bar)
    fig.tight_layout()
    return fig
