"""Quality diagnostic figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from itrace.types import GazeStream, PupilStream

from .palette import FONT_FLOOR, WONG


def figure_dropout_raster(gaze: GazeStream) -> Figure:
    """Raster of finite vs invalid gaze samples."""
    valid = np.isfinite(gaze.x) & np.isfinite(gaze.y) & np.isfinite(gaze.t)
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.imshow((~valid)[None, :], aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("sample")
    ax.set_title("gaze dropout raster")
    finite_pct = 100.0 * float(np.mean(valid)) if valid.size else 0.0
    invalid_count = int(np.sum(~valid))
    ax.text(
        0.01,
        1.06,
        f"white = finite, red = invalid | finite {finite_pct:.1f}% | invalid n={invalid_count}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_FLOOR,
        color="0.2",
    )
    fig.tight_layout()
    return fig


def figure_sampling_intervals(gaze: GazeStream) -> Figure:
    """Histogram of gaze sample intervals."""
    dt = np.diff(gaze.t) if len(gaze) >= 2 else np.zeros(0, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 3))
    if dt.size:
        ax.hist(dt[np.isfinite(dt)], bins=20, color=WONG[0], edgecolor="white")
    ax.set_xlabel("sample interval (s)")
    ax.set_ylabel("count")
    ax.set_title("sampling intervals")
    fig.tight_layout()
    return fig


def figure_calibration_residuals(residuals_deg: npt.ArrayLike) -> Figure:
    """Bar plot of calibration residual magnitudes."""
    arr = np.asarray(residuals_deg, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(np.arange(arr.size), arr, color=WONG[1])
    ax.set_xlabel("target")
    ax.set_ylabel("residual (deg)")
    ax.set_title("calibration residuals")
    fig.tight_layout()
    return fig


def figure_pupil_velocity(pupil: PupilStream) -> Figure:
    """Pupil size derivative over time."""
    velocity = np.gradient(pupil.size, pupil.t) if len(pupil) >= 2 else np.zeros_like(pupil.size)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(pupil.t, velocity, color=WONG[2], lw=1.2)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"pupil velocity ({pupil.unit.value}/s)")
    ax.set_title("pupil velocity")
    fig.tight_layout()
    return fig
