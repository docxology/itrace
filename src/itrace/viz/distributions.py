"""Distribution and main-sequence diagnostic plots.

This is a *viz* module: matplotlib is an optional dependency and is imported
here at module top with the headless ``Agg`` backend selected *before* pyplot is
imported, so importing :mod:`itrace.viz.distributions` is safe in any
environment that has matplotlib installed.

Two families of diagnostics live here:

* **Histogram-with-fit** -- a density histogram of a sample (fixation/saccade
  durations, saccade amplitudes) with the maximum-likelihood PDF from
  :mod:`itrace.stats.distributions` overlaid and the goodness-of-fit (AIC, KS)
  reported in the legend.
* **Main sequence** -- the lawful amplitude/peak-velocity relationship plotted
  on log-log axes with the fitted power law (:func:`itrace.mainsequence.fit`)
  and its residuals.

Every plotting helper accepts an optional ``ax`` so panels can be composed; the
``figure_*`` wrappers build a standalone :class:`~matplotlib.figure.Figure`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from itrace import mainsequence
from itrace.stats import distributions as dist
from itrace.types import FloatArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from .palette import WONG  # single-source Wong (2011) colour-blind-safe palette

# Minimum sample size below which a maximum-likelihood fit is meaningless; the
# histogram is still drawn but the PDF overlay is skipped.
_MIN_FIT_SAMPLES = 3


def _as_float_array(values: FloatArray | list[float]) -> FloatArray:
    """Return a 1-D float64 array of the finite, positive entries of ``values``.

    Distributions such as gamma and lognormal have support on ``(0, inf)``; a
    non-positive or non-finite sample is dropped so the fit and overlay are well
    defined.

    Parameters
    ----------
    values
        Raw sample (e.g. durations in seconds or amplitudes in degrees).

    Returns
    -------
    FloatArray
        The cleaned 1-D sample.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    return cast(FloatArray, finite)


def _fit_annotation(result: Any) -> str:
    """Build a compact ``AIC=.. KS=..`` legend label from a fit result.

    The :class:`itrace.stats.distributions.FitResult` carries goodness-of-fit
    statistics; their exact attribute names may evolve, so the common spellings
    are probed defensively and any that are missing are simply omitted.

    Parameters
    ----------
    result
        A ``FitResult`` returned by :func:`itrace.stats.distributions.fit_distribution`.

    Returns
    -------
    str
        Legend label such as ``"gamma  AIC=12.3  KS=0.08"``.
    """
    family = getattr(result, "family", "fit")
    parts: list[str] = [str(family)]

    aic = getattr(result, "aic", None)
    if isinstance(aic, (int, float)) and np.isfinite(float(aic)):
        parts.append(f"AIC={float(aic):.1f}")

    ks = None
    for name in ("ks_statistic", "ks_stat", "ks", "ks_d"):
        candidate = getattr(result, name, None)
        if isinstance(candidate, (int, float)) and np.isfinite(float(candidate)):
            ks = float(candidate)
            break
    if ks is not None:
        parts.append(f"KS={ks:.3f}")

    return "  ".join(parts)


def _overlay_pdf(ax: Axes, sample: FloatArray, family: str, colour: str) -> bool:
    """Fit ``family`` to ``sample`` and overlay its PDF on ``ax``.

    Parameters
    ----------
    ax
        Target axes (a density histogram is assumed already drawn).
    sample
        Cleaned 1-D positive sample.
    family
        Distribution family name passed to
        :func:`itrace.stats.distributions.fit_distribution`.
    colour
        Line colour for the PDF curve.

    Returns
    -------
    bool
        ``True`` if a PDF was overlaid, ``False`` if the sample was too small.
    """
    if sample.size < _MIN_FIT_SAMPLES:
        return False
    result = dist.fit_distribution(sample, family=family)
    frozen = dist.frozen_from_result(result)
    grid = np.linspace(float(sample.min()), float(sample.max()), 200)
    ax.plot(grid, frozen.pdf(grid), color=colour, lw=2, label=_fit_annotation(result))
    return True


def _histogram_with_fit(
    values: FloatArray | list[float],
    *,
    family: str,
    bins: int,
    ax: Axes | None,
    xlabel: str,
    title: str,
) -> Axes:
    """Shared implementation of the duration/amplitude histogram-with-fit plot."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    sample = _as_float_array(values)

    if sample.size == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ax

    ax.hist(
        sample,
        bins=bins,
        density=True,
        color=WONG[0],
        edgecolor="white",
        alpha=0.75,
        label="observed",
    )
    _overlay_pdf(ax, sample, family, WONG[3])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_duration_histogram(
    values: FloatArray | list[float],
    *,
    family: str = "gamma",
    bins: int = 20,
    ax: Axes | None = None,
) -> Axes:
    """Density histogram of event durations with a fitted PDF overlay.

    Parameters
    ----------
    values
        Durations in seconds (e.g. ``saccade_properties(...)["duration_s"]``).
    family
        Distribution family fitted via
        :func:`itrace.stats.distributions.fit_distribution` (default ``"gamma"``).
    bins
        Number of histogram bins.
    ax
        Optional axes to draw on; a new figure/axes is created when omitted.

    Returns
    -------
    Axes
        The axes containing the histogram (and the PDF overlay when there are at
        least three positive samples).
    """
    return _histogram_with_fit(
        values,
        family=family,
        bins=bins,
        ax=ax,
        xlabel="duration (s)",
        title="Duration distribution",
    )


def figure_duration_histogram(
    values: FloatArray | list[float],
    **opts: Any,
) -> Figure:
    """Standalone figure wrapping :func:`plot_duration_histogram`.

    Parameters
    ----------
    values
        Durations in seconds.
    **opts
        Forwarded to :func:`plot_duration_histogram` (``family``, ``bins``).

    Returns
    -------
    Figure
        A tight-laid-out figure with one duration-histogram axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_duration_histogram(values, ax=ax, **opts)
    fig.tight_layout()
    return fig


def plot_amplitude_histogram(
    amplitudes: FloatArray | list[float],
    *,
    family: str = "gamma",
    bins: int = 20,
    ax: Axes | None = None,
) -> Axes:
    """Density histogram of saccade amplitudes with a fitted PDF overlay.

    Parameters
    ----------
    amplitudes
        Saccade amplitudes in degrees of visual angle.
    family
        Distribution family fitted via
        :func:`itrace.stats.distributions.fit_distribution` (default ``"gamma"``).
    bins
        Number of histogram bins.
    ax
        Optional axes to draw on; a new figure/axes is created when omitted.

    Returns
    -------
    Axes
        The axes containing the histogram (and the PDF overlay when there are at
        least three positive samples).
    """
    return _histogram_with_fit(
        amplitudes,
        family=family,
        bins=bins,
        ax=ax,
        xlabel="amplitude (deg)",
        title="Amplitude distribution",
    )


def figure_amplitude_histogram(
    amplitudes: FloatArray | list[float],
    **opts: Any,
) -> Figure:
    """Standalone figure wrapping :func:`plot_amplitude_histogram`.

    Parameters
    ----------
    amplitudes
        Saccade amplitudes in degrees.
    **opts
        Forwarded to :func:`plot_amplitude_histogram` (``family``, ``bins``).

    Returns
    -------
    Figure
        A tight-laid-out figure with one amplitude-histogram axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_amplitude_histogram(amplitudes, ax=ax, **opts)
    fig.tight_layout()
    return fig


def _clean_main_sequence(
    amp: FloatArray | list[float],
    vel: FloatArray | list[float],
) -> tuple[FloatArray, FloatArray]:
    """Return paired positive, finite (amplitude, peak-velocity) arrays.

    Parameters
    ----------
    amp
        Saccade amplitudes (degrees).
    vel
        Saccade peak velocities (degrees / second).

    Returns
    -------
    tuple of FloatArray
        The masked ``(amp, vel)`` pair, suitable for log-log work.
    """
    a = np.asarray(amp, dtype=np.float64).ravel()
    v = np.asarray(vel, dtype=np.float64).ravel()
    n = min(a.size, v.size)
    a, v = a[:n], v[:n]
    valid = np.isfinite(a) & np.isfinite(v) & (a > 0.0) & (v > 0.0)
    return a[valid], v[valid]


def plot_main_sequence(
    amp: FloatArray | list[float],
    vel: FloatArray | list[float],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Log-log scatter of amplitude vs peak velocity with the power-law fit.

    The power law ``V = a * A^b`` is fitted by :func:`itrace.mainsequence.fit`
    and overlaid; its exponent ``b`` and the log-log :math:`R^2` are annotated.

    Parameters
    ----------
    amp
        Saccade amplitudes (degrees).
    vel
        Saccade peak velocities (degrees / second).
    ax
        Optional axes; created when omitted.

    Returns
    -------
    Axes
        The configured log-log axes.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    a, v = _clean_main_sequence(amp, vel)

    if a.size == 0:
        ax.set_xlabel("amplitude (deg)")
        ax.set_ylabel("peak velocity (deg/s)")
        ax.set_title("Saccadic main sequence")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return ax

    ax.scatter(a, v, color=WONG[0], s=28, label="saccades")

    if a.size >= 3:
        fit = mainsequence.fit(a, v)
        grid = np.linspace(float(a.min()), float(a.max()), 200)
        ax.plot(
            grid,
            fit["power_a"] * grid ** fit["power_b"],
            color=WONG[3],
            lw=2,
            label=f"power: b={fit['power_b']:.2f}  R^2={fit['r_squared_power']:.3f}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("amplitude (deg)")
    ax.set_ylabel("peak velocity (deg/s)")
    ax.set_title("Saccadic main sequence")
    ax.legend()
    return ax


def plot_main_sequence_residuals(
    amp: FloatArray | list[float],
    vel: FloatArray | list[float],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Residuals of ``log(vel)`` about the fitted power law.

    For each saccade the residual ``log(v) - [log(a_coef) + b*log(amp)]`` is
    plotted against amplitude; a horizontal zero line aids reading systematic
    departures from the power law.

    Parameters
    ----------
    amp
        Saccade amplitudes (degrees).
    vel
        Saccade peak velocities (degrees / second).
    ax
        Optional axes; created when omitted.

    Returns
    -------
    Axes
        The residual axes (semilog-x).
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    a, v = _clean_main_sequence(amp, vel)

    if a.size < 3:
        ax.set_xlabel("amplitude (deg)")
        ax.set_ylabel("log-velocity residual")
        ax.set_title("Main-sequence residuals")
        ax.text(0.5, 0.5, "too few points", ha="center", va="center", transform=ax.transAxes)
        return ax

    fit = mainsequence.fit(a, v)
    predicted = np.log(fit["power_a"]) + fit["power_b"] * np.log(a)
    residual = np.log(v) - predicted

    ax.axhline(0.0, color=WONG[1], lw=1.0, ls="--", label="zero")
    ax.scatter(a, residual, color=WONG[2], s=28, label="residual")
    ax.set_xscale("log")
    ax.set_xlabel("amplitude (deg)")
    ax.set_ylabel("log-velocity residual")
    ax.set_title("Main-sequence residuals")
    ax.legend()
    return ax


def figure_main_sequence(
    amp: FloatArray | list[float],
    vel: FloatArray | list[float],
) -> Figure:
    """Two-panel main-sequence figure: power-law fit and its residuals.

    Parameters
    ----------
    amp
        Saccade amplitudes (degrees).
    vel
        Saccade peak velocities (degrees / second).

    Returns
    -------
    Figure
        A figure whose left panel is the fitted log-log main sequence and whose
        right panel is the log-velocity residual scatter.
    """
    fig, (ax_fit, ax_res) = plt.subplots(1, 2, figsize=(11, 4))
    plot_main_sequence(amp, vel, ax=ax_fit)
    plot_main_sequence_residuals(amp, vel, ax=ax_res)
    fig.tight_layout()
    return fig
