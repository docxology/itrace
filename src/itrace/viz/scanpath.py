"""Spatial scanpath and microsaccade visualisations.

This is a *viz* module: matplotlib is a hard import here (it is an optional
dependency of the package as a whole) and the non-interactive ``Agg`` backend is
selected at import time so the plots render headless in CI and in the figure
pipeline. Stats/analysis modules must never import this module.

Two families of plots live here:

* **Scanpath** -- fixation centroids drawn in screen coordinates (y down),
  sized by dwell time and joined in temporal order, with saccades overlaid as
  arrows from one fixation to the next.
* **Microsaccades** -- a polar histogram of microsaccade directions and an
  amplitude-vs-peak-velocity (main-sequence) scatter.

Every public plotting function accepts an optional ``ax`` so panels can be
composed into larger figures, and degrades gracefully on empty input.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..types import Fixation, FloatArray, Microsaccade, Saccade, SessionReport

# Wong 2011 colour-blind-safe palette.
WONG: list[str] = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

# Marker-area scaling for fixation dwell time (points^2 per second).
_FIXATION_AREA_PER_S = 1200.0
_FIXATION_MIN_AREA = 30.0


def _new_axes(ax: Axes | None, *, polar: bool = False) -> Axes:
    """Return ``ax`` if given, else create a fresh (optionally polar) Axes."""
    if ax is not None:
        return ax
    subplot_kw = {"projection": "polar"} if polar else {}
    _fig, new_ax = plt.subplots(figsize=(5, 5), subplot_kw=subplot_kw)
    return new_ax


def _fixation_centroids(fixations: list[Fixation]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Centroid x, centroid y and dwell-time arrays for a fixation list."""
    xs = np.array([f.centroid_x for f in fixations], dtype=np.float64)
    ys = np.array([f.centroid_y for f in fixations], dtype=np.float64)
    durs = np.array([f.duration_s for f in fixations], dtype=np.float64)
    return xs, ys, durs


def plot_scanpath(
    fixations: list[Fixation],
    saccades: list[Saccade],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Draw a spatial scanpath: dwell-sized fixations joined in temporal order.

    Fixation centroids are scattered with marker area proportional to dwell
    time, connected by a line in temporal order, and each consecutive transition
    is overlaid with an arrow (a visual saccade). The y-axis is inverted because
    the centroids live in screen coordinates (y increases downward).

    Parameters
    ----------
    fixations:
        Fixations in temporal order; their ``centroid_x``/``centroid_y`` give the
        plotted positions and ``duration_s`` the marker size.
    saccades:
        Detected saccades. They are not required for geometry (arrows connect
        consecutive fixations) but their count is reported in the legend; passing
        an empty list is fine.
    ax:
        Target Axes. A new figure/Axes is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes the scanpath was drawn on. The y-axis is inverted and both
        axes are labelled in degrees.

    Notes
    -----
    Empty and single-fixation inputs are handled without error: an empty input
    yields an annotated, inverted, labelled (but data-free) Axes, and a single
    fixation is scattered with no connecting line or arrows.
    """
    ax = _new_axes(ax)
    ax.set_xlabel("horizontal gaze (deg)")
    ax.set_ylabel("vertical gaze (deg, screen)")
    ax.set_title("Scanpath")

    if not fixations:
        ax.text(
            0.5,
            0.5,
            "no fixations",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=WONG[3],
        )
        ax.invert_yaxis()
        return ax

    xs, ys, durs = _fixation_centroids(fixations)
    sizes = np.maximum(durs * _FIXATION_AREA_PER_S, _FIXATION_MIN_AREA)

    if xs.size > 1:
        ax.plot(xs, ys, color=WONG[5], lw=1.0, alpha=0.6, zorder=1)
        # One arrow per consecutive fixation transition.
        for i in range(xs.size - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops={"arrowstyle": "->", "color": WONG[3], "lw": 1.2},
                zorder=2,
            )

    ax.scatter(
        xs,
        ys,
        s=sizes,
        color=WONG[0],
        edgecolor="white",
        zorder=3,
        label=f"{xs.size} fixations / {len(saccades)} saccades",
    )
    ax.legend(loc="best")
    ax.invert_yaxis()
    return ax


def figure_scanpath(
    report: SessionReport,
    *,
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """Render a one-panel scanpath figure from a :class:`SessionReport`.

    Parameters
    ----------
    report:
        Session whose ``fixations`` and ``saccades`` are plotted.
    figsize:
        Figure size in inches, forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scanpath Axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_scanpath(report.fixations, report.saccades, ax=ax)
    fig.tight_layout()
    return fig


def plot_microsaccade_polar(
    microsaccades: list[Microsaccade],
    *,
    ax: Axes | None = None,
    bins: int = 16,
) -> Axes:
    """Polar histogram of microsaccade directions.

    Parameters
    ----------
    microsaccades:
        Detected microsaccades; their ``direction_deg`` (gaze convention) is
        binned over the full circle.
    ax:
        A *polar* Axes. When ``None`` a new polar Axes is created
        (``subplot_kw={"projection": "polar"}``).
    bins:
        Number of angular bins spanning ``[-pi, pi]``.

    Returns
    -------
    matplotlib.axes.Axes
        The polar Axes. With no microsaccades the Axes is returned annotated and
        empty (no bars).
    """
    ax = _new_axes(ax, polar=True)
    ax.set_title("Microsaccade directions")

    if not microsaccades:
        ax.text(
            0.5,
            0.5,
            "no microsaccades",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=WONG[3],
        )
        return ax

    angles = np.radians([m.direction_deg for m in microsaccades])
    counts, edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(
        centers,
        counts,
        width=np.diff(edges),
        color=WONG[2],
        edgecolor="white",
        align="center",
    )
    return ax


def plot_microsaccade_main_sequence(
    microsaccades: list[Microsaccade],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Scatter microsaccade amplitude against peak velocity (main sequence).

    Parameters
    ----------
    microsaccades:
        Detected microsaccades; ``amplitude_deg`` is plotted on x and
        ``peak_velocity_deg_s`` on y.
    ax:
        Target Axes. A new figure/Axes is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes with the scatter. Empty input yields a labelled, annotated,
        data-free Axes.
    """
    ax = _new_axes(ax)
    ax.set_xlabel("amplitude (deg)")
    ax.set_ylabel("peak velocity (deg/s)")
    ax.set_title("Microsaccade main sequence")

    if not microsaccades:
        ax.text(
            0.5,
            0.5,
            "no microsaccades",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=WONG[3],
        )
        return ax

    amp = np.array([m.amplitude_deg for m in microsaccades], dtype=np.float64)
    vel = np.array([m.peak_velocity_deg_s for m in microsaccades], dtype=np.float64)
    ax.scatter(amp, vel, color=WONG[1], s=28, edgecolor="white")
    return ax


def figure_microsaccades(microsaccades: list[Microsaccade]) -> Figure:
    """Two-panel microsaccade figure: direction polar + main-sequence scatter.

    Parameters
    ----------
    microsaccades:
        Detected microsaccades shared by both panels.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with a polar direction histogram (left) and an amplitude/peak-
        velocity scatter (right).
    """
    fig = plt.figure(figsize=(11, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_seq = fig.add_subplot(1, 2, 2)
    plot_microsaccade_polar(microsaccades, ax=ax_polar)
    plot_microsaccade_main_sequence(microsaccades, ax=ax_seq)
    fig.tight_layout()
    return fig
