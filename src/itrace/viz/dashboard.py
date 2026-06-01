"""Multi-panel session dashboard.

Assembles a single publication figure summarising one analysed recording by
composing the panel functions from the sibling :mod:`itrace.viz` modules with a
couple of locally-drawn panels (saccade-direction polar histogram and a textual
summary). The figure is robust to degenerate sessions: a recording with no
saccades still renders (the fit/histogram panels annotate "no data" instead of
raising).

``render_dashboard`` is the thin entry point used by the figure pipeline: it
synthesises a deterministic multi-saccade session, runs the analysis pipeline,
builds the dashboard, and writes a 300-dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .. import pipeline, saccades
from ..synthetic import gaze_with_saccade, pupil_sine_with_blink
from ..types import GazeStream, PupilStream, Saccade, SessionReport
from .distributions import plot_amplitude_histogram, plot_main_sequence
from .scanpath import plot_scanpath
from .traces import plot_velocity_trace

# Wong (2011) colour-blind-safe palette.
WONG: list[str] = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def _plot_direction_polar(saccs: list[Saccade], ax: Axes) -> None:
    """Draw a polar histogram of saccade directions onto a polar Axes."""
    ax.set_title("Saccade directions")
    if not saccs:
        ax.text(0.5, 0.5, "no saccades", ha="center", va="center", transform=ax.transAxes)
        return
    angles = np.radians([s.direction_deg for s in saccs])
    counts, edges = np.histogram(angles, bins=12, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(centers, counts, width=np.diff(edges), color=WONG[2], edgecolor="white")


def _plot_summary_text(report: SessionReport, ax: Axes) -> None:
    """Render a text-only panel listing the headline session statistics."""
    ax.axis("off")
    power_b = report.main_sequence.get("power_b")
    power_b_str = f"{power_b:.2f}" if isinstance(power_b, float) else "n/a"
    scanpath = report.scanpath if report.scanpath else "(none)"
    if len(scanpath) > 32:
        scanpath = scanpath[:29] + "..."
    lines = [
        "Session summary",
        "",
        f"samples:        {report.n_samples}",
        f"duration:       {report.duration_s:.2f} s",
        f"fixations:      {len(report.fixations)}",
        f"saccades:       {len(report.saccades)}",
        f"microsaccades:  {len(report.microsaccades)}",
        f"main-seq b:     {power_b_str}",
        f"scanpath:       {scanpath}",
    ]
    ax.text(  # type: ignore[attr-defined]
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )


def session_dashboard(
    report: SessionReport,
    gaze: GazeStream,
    pupil: PupilStream | None = None,
) -> Figure:
    """Build a six-panel summary figure for one analysed session.

    Panels: (1) velocity trace with the I-VT threshold and shaded saccades,
    (2) spatial scanpath (dwell-sized fixations + saccade arrows), (3) saccade
    amplitude histogram with a fitted distribution, (4) log-log main sequence
    with the power-law fit, (5) saccade-direction polar histogram, and (6) a
    textual summary.

    Parameters
    ----------
    report:
        The analysed :class:`~itrace.types.SessionReport`.
    gaze:
        The gaze stream the report was computed from (drives the velocity panel).
    pupil:
        Unused placeholder for symmetry / future pupil panel; accepted so callers
        can pass a full session. May be ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        A 2x3 dashboard figure. Never raises on a 0-saccade session.
    """
    del pupil  # reserved for a future pupil panel; intentionally unused
    fig = plt.figure(figsize=(15, 8))
    ax_vel = fig.add_subplot(2, 3, 1)
    ax_scan = fig.add_subplot(2, 3, 2)
    ax_amp = fig.add_subplot(2, 3, 3)
    ax_seq = fig.add_subplot(2, 3, 4)
    ax_dir = fig.add_subplot(2, 3, 5, projection="polar")
    ax_txt = fig.add_subplot(2, 3, 6)

    plot_velocity_trace(gaze, ax=ax_vel)
    plot_scanpath(report.fixations, report.saccades, ax=ax_scan)

    props = saccades.saccade_properties(report.saccades)
    plot_amplitude_histogram(props["amplitude_deg"], ax=ax_amp)
    plot_main_sequence(props["amplitude_deg"], props["peak_velocity_deg_s"], ax=ax_seq)
    _plot_direction_polar(report.saccades, ax_dir)
    _plot_summary_text(report, ax_txt)

    fig.suptitle("iTrace session dashboard", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    """Concatenate a spread of saccades into one deterministic gaze stream."""
    rng = np.random.default_rng(seed)
    xs, ys, ts = [], [], []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def render_dashboard(out_path: str | Path, *, seed: int = 0) -> Path:
    """Synthesise a session, analyse it, and write the dashboard PNG.

    Parameters
    ----------
    out_path:
        Destination PNG path.
    seed:
        Seed for the synthetic multi-saccade stream (kept for reproducibility).

    Returns
    -------
    pathlib.Path
        The written path.
    """
    gaze = _multi_saccade_stream(seed=seed)
    pstream, _ = pupil_sine_with_blink(seed=seed)
    report = pipeline.analyze_session(gaze, pstream)
    fig = session_dashboard(report, gaze, pstream)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
