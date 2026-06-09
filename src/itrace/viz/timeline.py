"""Temporal-event and spectral diagnostic plots.

This is a *viz* module: matplotlib is an optional dependency and is imported
here at module top with the headless ``Agg`` backend selected *before*
:mod:`matplotlib.pyplot`, so importing :mod:`itrace.viz.timeline` is safe in any
environment that has matplotlib installed. Every plotting helper is pure: it
draws onto a supplied (or freshly created) :class:`~matplotlib.axes.Axes`,
returns it, and never calls ``plt.show``. The ``figure_*`` wrappers return a
titled :class:`~matplotlib.figure.Figure`.

Three diagnostics live here:

* **Event raster** -- a two-row timeline of a :class:`~itrace.types.SessionReport`
  with fixations drawn as horizontal bars on one row and saccades on a second,
  sharing a common time axis.
* **Rate trace** -- a step plot of a pre-computed rate series (event counts per
  unit time), pairing with the time-series helpers in :mod:`itrace.stats`.
* **Pupil PSD** -- the Welch power spectral density of the
  blink-interpolated-and-smoothed pupil signal, on log-log axes labelled in Hz.

Colours come from the Wong (2011) colour-blind-safe palette.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .. import pupil as pupil_mod
from ..types import FloatArray, PupilStream, SessionReport

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .palette import FONT_FLOOR, WONG  # single-source Wong (2011) colour-blind-safe palette

# Welch segment ceiling; the effective ``nperseg`` is clamped to the signal
# length so short traces still yield a (coarse) spectrum.
_PSD_MAX_NPERSEG = 256

# Vertical placement of the two raster rows and the half-height of each bar.
_ROW_FIXATION = 1.0
_ROW_SACCADE = 0.0
_BAR_HALF_HEIGHT = 0.35


def _ensure_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return ``ax`` or a freshly created single-axes figure's Axes.

    Parameters
    ----------
    ax
        An existing Axes to draw onto, or ``None`` to create a new figure.
    figsize
        Size (inches) of the figure created when ``ax`` is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes to draw onto.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)
    return ax


def plot_event_raster(report: SessionReport, *, ax: Axes | None = None) -> Axes:
    """Draw a two-row timeline of a session's fixations and saccades.

    Each fixation is a horizontal bar spanning ``onset_t``..``offset_t`` on the
    upper row; each saccade is a bar on the lower row. Time runs along the x
    axis and the two rows are labelled so the oculomotor sequence reads at a
    glance. A session with no events of a given kind simply contributes no bars
    on that row (the row label and axis are still drawn).

    Parameters
    ----------
    report
        The analysed :class:`~itrace.types.SessionReport`.
    ax
        Axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the event raster.
    """
    ax = _ensure_ax(ax, (8.4, 3.1))

    fix_spans = [(f.onset_t, max(f.offset_t - f.onset_t, 0.0)) for f in report.fixations]
    sac_spans = [(s.onset_t, max(s.offset_t - s.onset_t, 0.0)) for s in report.saccades]

    if fix_spans:
        ax.broken_barh(
            fix_spans,
            (_ROW_FIXATION - _BAR_HALF_HEIGHT, 2 * _BAR_HALF_HEIGHT),
            facecolors=WONG[0],
            edgecolor="white",
            label="fixation",
        )
    if sac_spans:
        ax.broken_barh(
            sac_spans,
            (_ROW_SACCADE - _BAR_HALF_HEIGHT, 2 * _BAR_HALF_HEIGHT),
            facecolors=WONG[1],
            edgecolor="white",
            label="saccade",
        )

    ax.set_yticks([_ROW_SACCADE, _ROW_FIXATION])
    ax.set_yticklabels(["saccades", "fixations"])
    ax.set_ylim(_ROW_SACCADE - 0.6, _ROW_FIXATION + 0.6)
    ax.set_xlabel("time (s)")
    ax.set_title("Oculomotor event raster")
    if fix_spans or sac_spans:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=2,
            frameon=False,
            fontsize=FONT_FLOOR,
        )
    ax.text(
        0.01,
        1.04,
        f"fixations n={len(report.fixations)} | saccades n={len(report.saccades)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_FLOOR,
        color="0.25",
    )
    return ax


def figure_event_raster(report: SessionReport) -> Figure:
    """Render :func:`plot_event_raster` as a standalone titled figure.

    Parameters
    ----------
    report
        The analysed :class:`~itrace.types.SessionReport`.

    Returns
    -------
    matplotlib.figure.Figure
        A one-axes figure containing the event raster.
    """
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    plot_event_raster(report, ax=ax)
    fig.tight_layout()
    return fig


def plot_rate(
    times: FloatArray | list[float],
    rates: FloatArray | list[float],
    *,
    label: str = "rate",
    ax: Axes | None = None,
) -> Axes:
    """Step plot of a rate series (e.g. event counts per unit time).

    The series is drawn as a post-step line so that each ``rate`` value is held
    over the interval beginning at its corresponding ``time`` -- the natural
    reading for a binned count. Pairs with the time-series helpers in
    :mod:`itrace.stats`.

    Parameters
    ----------
    times
        Bin times (seconds), one per rate value.
    rates
        Rate values (events per unit time) aligned with ``times``.
    label
        Legend label for the series.
    ax
        Axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the rate trace.

    Raises
    ------
    ValueError
        If ``times`` and ``rates`` have different lengths.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    r = np.asarray(rates, dtype=np.float64).ravel()
    if t.shape != r.shape:
        msg = f"times and rates must have equal length; got {t.shape} vs {r.shape}"
        raise ValueError(msg)

    ax = _ensure_ax(ax, (7, 3.0))
    if t.size > 0:
        ax.step(t, r, where="post", color=WONG[2], lw=1.4, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rate (1/s)")
    ax.set_title("Event rate")
    if t.size > 0:
        ax.legend(loc="upper right", fontsize=FONT_FLOOR)
    return ax


def figure_rate(
    times: FloatArray | list[float],
    rates: FloatArray | list[float],
    *,
    label: str = "rate",
) -> Figure:
    """Render :func:`plot_rate` as a standalone titled figure.

    Parameters
    ----------
    times
        Bin times (seconds), one per rate value.
    rates
        Rate values aligned with ``times``.
    label
        Legend label for the series.

    Returns
    -------
    matplotlib.figure.Figure
        A one-axes figure containing the rate trace.
    """
    fig, ax = plt.subplots(figsize=(7, 3.0))
    plot_rate(times, rates, label=label, ax=ax)
    fig.tight_layout()
    return fig


def plot_pupil_psd(stream: PupilStream, *, ax: Axes | None = None) -> Axes:
    """Welch power spectral density of the cleaned pupil signal (log-log).

    The pupil trace is blink-interpolated
    (:func:`itrace.pupil.interpolate_blinks`) and low-pass smoothed
    (:func:`itrace.pupil.smooth`), then its one-sided PSD is estimated with
    :func:`scipy.signal.welch` at the inferred sampling rate. Only strictly
    positive frequency and power bins are plotted (the DC bin and any zero-power
    bins are dropped) so both axes can be log-scaled and labelled in Hz.

    Parameters
    ----------
    stream
        Pupil stream; NaNs denote blink/invalid samples.
    ax
        Axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the log-log PSD.

    Raises
    ------
    ValueError
        If the trace has no valid samples (propagated from
        :func:`itrace.pupil.interpolate_blinks`).
    """
    ax = _ensure_ax(ax, (6, 4))
    n = len(stream)

    # A Welch estimate needs at least a few samples; fewer cannot yield a
    # sampling rate (and the smoothing/interpolation steps are ill-defined), so
    # annotate "no data" rather than raise.
    if n < 3:
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (size^2/Hz)")
        ax.set_title("Pupil power spectral density")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ax

    cleaned = pupil_mod.smooth(pupil_mod.interpolate_blinks(stream))
    sr = 1.0 / float(np.median(np.diff(stream.t)))

    nperseg = int(min(_PSD_MAX_NPERSEG, n))
    freqs, psd = welch(cleaned.size, fs=sr, nperseg=nperseg)
    # Only strictly positive frequency and power bins survive (the DC bin and
    # any non-positive bins are dropped) so both axes can be log-scaled.
    keep = (freqs > 0.0) & (psd > 0.0)

    ax.plot(freqs[keep], psd[keep], color=WONG[0], lw=1.4, label="Welch PSD")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (size^2/Hz)")
    ax.set_title("Pupil power spectral density")
    ax.legend(loc="upper right", fontsize=FONT_FLOOR)
    return ax


def figure_pupil_psd(stream: PupilStream) -> Figure:
    """Render :func:`plot_pupil_psd` as a standalone titled figure.

    Parameters
    ----------
    stream
        Pupil stream; NaNs denote blink/invalid samples.

    Returns
    -------
    matplotlib.figure.Figure
        A one-axes figure containing the pupil PSD.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_pupil_psd(stream, ax=ax)
    fig.tight_layout()
    return fig
