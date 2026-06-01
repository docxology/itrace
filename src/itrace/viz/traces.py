"""Time-series plots for gaze velocity and pupil signals.

This is the only iTrace module besides its siblings under :mod:`itrace.viz`
that imports :mod:`matplotlib`; the ``Agg`` backend is selected before
:mod:`matplotlib.pyplot` is imported so the functions run headless. Every
plotting function is pure: it draws onto a supplied (or freshly created)
:class:`~matplotlib.axes.Axes`, returns it, and never calls ``plt.show``. The
``figure_*`` wrappers return a titled :class:`~matplotlib.figure.Figure`.

Colours come from the Wong (2011) colour-blind-safe palette.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .. import pupil as pupil_mod
from .. import saccades as saccades_mod
from ..pupilphase import Phase, PhaseDetector
from ..types import GazeStream, PupilStream

# Wong (2011) colour-blind-safe palette.
WONG: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]


def _ensure_ax(ax: Axes | None) -> Axes:
    """Return ``ax`` or a freshly created single-axes figure's Axes.

    Parameters
    ----------
    ax
        An existing Axes to draw onto, or ``None`` to create a new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes to draw onto.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 3.5))
    return ax


def plot_velocity_trace(
    stream: GazeStream,
    *,
    velocity_threshold_deg_s: float = 30.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot 2-D gaze speed over time with the I-VT threshold and saccade spans.

    The 2-D speed (deg/s) from :func:`itrace.saccades.velocities` is drawn
    against time. A dashed horizontal line marks ``velocity_threshold_deg_s``
    and each saccade interval detected by :func:`itrace.saccades.detect_ivt`
    (using its ``onset_t``/``offset_t``) is shaded.

    Parameters
    ----------
    stream
        Gaze stream in degrees of visual angle.
    velocity_threshold_deg_s
        I-VT velocity threshold (deg/s) used both for the dashed line and for
        the saccade detection that drives the shaded spans.
    ax
        Axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the velocity trace.
    """
    ax = _ensure_ax(ax)
    _vx, _vy, speed = saccades_mod.velocities(stream)
    ax.plot(stream.t, speed, color=WONG[0], lw=1.2, label="2-D speed")
    ax.axhline(
        velocity_threshold_deg_s,
        color=WONG[3],
        ls="--",
        lw=1.0,
        label=f"threshold ({velocity_threshold_deg_s:.0f} deg/s)",
    )

    _fixations, saccs = saccades_mod.detect_ivt(
        stream, velocity_threshold_deg_s=velocity_threshold_deg_s
    )
    for i, sac in enumerate(saccs):
        ax.axvspan(
            sac.onset_t,
            sac.offset_t,
            color=WONG[1],
            alpha=0.25,
            label="saccade" if i == 0 else None,
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (deg/s)")
    ax.legend(loc="upper right", fontsize="small")
    return ax


def figure_velocity_trace(stream: GazeStream, **opts: object) -> Figure:
    """Render :func:`plot_velocity_trace` as a standalone titled figure.

    Parameters
    ----------
    stream
        Gaze stream in degrees of visual angle.
    **opts
        Forwarded to :func:`plot_velocity_trace` (e.g.
        ``velocity_threshold_deg_s``).

    Returns
    -------
    matplotlib.figure.Figure
        A one-axes figure titled with the velocity trace.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_velocity_trace(stream, ax=ax, **opts)  # type: ignore[arg-type]
    ax.set_title("Gaze velocity trace (I-VT)")
    fig.tight_layout()
    return fig


def plot_pupil_trace(stream: PupilStream, *, ax: Axes | None = None) -> Axes:
    """Plot a raw pupil trace with the cleaned overlay, blink spans and extrema.

    The raw size is drawn with NaN gaps left visible (matplotlib breaks the
    line at NaNs). The blink-interpolated
    (:func:`itrace.pupil.interpolate_blinks`) and low-pass-smoothed
    (:func:`itrace.pupil.smooth`) trace is overlaid. Blink intervals from
    :func:`itrace.pupil.detect_blinks` are shaded, and the
    :class:`~itrace.pupilphase.PhaseDetector` run on the smoothed signal marks
    peaks (dilation maxima) and troughs (constriction minima).

    A trace with no blinks is handled gracefully (no spans are drawn and the
    raw and cleaned curves coincide except for smoothing).

    Parameters
    ----------
    stream
        Pupil stream; NaNs denote blink/invalid samples.
    ax
        Axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the pupil trace.
    """
    ax = _ensure_ax(ax)
    t = stream.t
    ax.plot(t, stream.size, color=WONG[5], lw=0.8, alpha=0.8, label="raw")

    cleaned = pupil_mod.smooth(pupil_mod.interpolate_blinks(stream))
    ax.plot(t, cleaned.size, color=WONG[0], lw=1.5, label="interpolated + smoothed")

    blinks = pupil_mod.detect_blinks(stream)
    for i, (onset, offset) in enumerate(blinks):
        ax.axvspan(
            float(t[onset]),
            float(t[offset]),
            color=WONG[3],
            alpha=0.2,
            label="blink" if i == 0 else None,
        )

    phases = PhaseDetector().run([float(v) for v in cleaned.size])
    peak_t = [float(t[i]) for i, ph in enumerate(phases) if ph is Phase.PEAK]
    peak_y = [float(cleaned.size[i]) for i, ph in enumerate(phases) if ph is Phase.PEAK]
    trough_t = [float(t[i]) for i, ph in enumerate(phases) if ph is Phase.TROUGH]
    trough_y = [float(cleaned.size[i]) for i, ph in enumerate(phases) if ph is Phase.TROUGH]
    if peak_t:
        ax.scatter(peak_t, peak_y, color=WONG[1], s=28, marker="^", zorder=5, label="peak")
    if trough_t:
        ax.scatter(trough_t, trough_y, color=WONG[2], s=28, marker="v", zorder=5, label="trough")

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"pupil size ({stream.unit.value})")
    ax.legend(loc="upper right", fontsize="small")
    return ax


def figure_pupil_trace(stream: PupilStream, **opts: object) -> Figure:
    """Render :func:`plot_pupil_trace` as a standalone titled figure.

    Parameters
    ----------
    stream
        Pupil stream; NaNs denote blink/invalid samples.
    **opts
        Forwarded to :func:`plot_pupil_trace`.

    Returns
    -------
    matplotlib.figure.Figure
        A one-axes figure titled with the pupil trace.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_pupil_trace(stream, ax=ax, **opts)  # type: ignore[arg-type]
    ax.set_title("Pupil trace (cleaned overlay)")
    fig.tight_layout()
    return fig
