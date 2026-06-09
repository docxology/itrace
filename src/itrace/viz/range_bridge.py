"""Publication figure for the synthetic/empirical range bridge."""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .palette import FONT_FLOOR, GRID, INK, MUTED, PANEL_FACE, WONG, panel_label


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _wrap(text: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _metric_display(row: dict[str, Any]) -> str:
    empirical = _mapping(row.get("empirical"))
    if empirical.get("available"):
        return str(empirical.get("display", "available"))
    return "unavailable"


def _comparison_display(row: dict[str, Any]) -> str:
    for key in ("synthetic", "noise_model", "statistical_model"):
        payload = _mapping(row.get(key))
        if payload:
            display = str(payload.get("display", "unavailable"))
            replacements = {
                "synthetic sessions can be generated at chosen rates": (
                    "configured synthetic\nsampling rates"
                ),
                "timestamp-jitter stress domain, if configured": "timestamp-jitter\nstress domain",
                "head-drift domain is a stress test": "head-drift\nstress domain",
                "idealized landmark-noise RMS 0.16-6.80 deg": (
                    "idealized landmark-noise\nRMS 0.16-6.80 deg"
                ),
                "not a synthetic saccade-latency oracle": "not a synthetic\nlatency oracle",
            }
            return replacements.get(display, display)
    return "unavailable"


def _status_colour(status: str) -> str:
    if status == "observed local scale":
        return WONG[2]
    if status == "stress-domain only":
        return WONG[1]
    if status == "not directly comparable":
        return WONG[3]
    return WONG[0]


def _badge(ax: Any, x: float, y: float, text: str, *, color: str) -> None:
    wrapped = {
        "observed local scale": "observed\nlocal scale",
        "stress-domain only": "stress-domain\nonly",
        "not directly comparable": "not directly\ncomparable",
        "descriptive model check": "descriptive\nmodel check",
    }.get(text, text)
    width = 0.18
    height = 0.065 if "\n" in wrapped else 0.046
    patch = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=ax.transAxes,
        fc=color,
        ec=color,
        alpha=0.13,
        lw=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y,
        wrapped,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.6,
        color=INK,
        fontweight="bold",
        linespacing=0.9,
    )


def _draw_boundary(ax: Any, payload: dict[str, Any]) -> None:
    ax.axis("off")
    ax.set_facecolor(PANEL_FACE)
    panel_label(ax, "A")
    title = "What the local pilot contributes"
    ax.text(0.03, 0.90, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK)
    lines = [
        (0.70, "Observed local scale: derived session\nvalues from one clean pilot."),
        (0.47, "Synthetic truth: known-label oracle\nfor algorithmic correctness."),
        (0.24, "Noise/stress domains: idealized\nperturbations, not reference-device validation."),
    ]
    for idx, (y, line) in enumerate(lines):
        ax.text(
            0.05,
            y,
            line,
            transform=ax.transAxes,
            fontsize=10.6,
            color=INK,
            va="center",
            linespacing=1.05,
        )
        ax.scatter(
            [0.025],
            [y],
            transform=ax.transAxes,
            s=80,
            color=[WONG[2], WONG[0], WONG[1]][idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.text(
        0.03,
        0.075,
        "Boundary: local scale, not device validation",
        transform=ax.transAxes,
        fontsize=FONT_FLOOR,
        fontweight="bold",
        color=WONG[3],
    )


def _draw_metric_rows(ax: Any, metrics: list[dict[str, Any]]) -> None:
    ax.axis("off")
    panel_label(ax, "B")
    ax.text(0.02, 0.95, "metric", transform=ax.transAxes, fontsize=12, color=MUTED)
    ax.text(0.30, 0.95, "N=1 value", transform=ax.transAxes, fontsize=12, color=MUTED)
    ax.text(
        0.48, 0.95, "synthetic/model evidence", transform=ax.transAxes, fontsize=12, color=MUTED
    )
    ax.text(0.82, 0.95, "status", transform=ax.transAxes, fontsize=12, color=MUTED)

    selected = [
        "finite_gaze_fraction",
        "sampling_rate_hz",
        "sampling_interval_cv",
        "max_drift_deg_s",
        "heldout_target_rms_deg",
        "target_acquisition_latency_s",
        "saccade_detection_stress",
        "model_selection_stability",
    ]
    rows = {str(row.get("id")): row for row in metrics}
    for idx, row_id in enumerate(selected):
        row = rows.get(row_id, {})
        y = 0.855 - idx * 0.098
        color = _status_colour(str(row.get("status", "")))
        ax.axhline(y - 0.047, xmin=0.015, xmax=0.985, color=GRID, lw=0.8)
        ax.text(
            0.02,
            y,
            _wrap(row.get("label", row_id), 25),
            transform=ax.transAxes,
            fontsize=FONT_FLOOR,
            color=INK,
            va="center",
            linespacing=0.95,
            gid=f"bridge-metric-{idx}",
        )
        ax.text(
            0.30,
            y,
            _metric_display(row),
            transform=ax.transAxes,
            fontsize=FONT_FLOOR,
            color=INK,
            va="center",
            fontweight="bold",
            gid=f"bridge-value-{idx}",
        )
        ax.text(
            0.48,
            y,
            _comparison_display(row),
            transform=ax.transAxes,
            fontsize=FONT_FLOOR,
            color=MUTED,
            va="center",
            linespacing=0.96,
            gid=f"bridge-evidence-{idx}",
        )
        _badge(ax, 0.81, y, str(row.get("status", "unknown")), color=color)


def _draw_trial_spread(ax: Any, trial_spread: dict[str, Any]) -> None:
    panel_label(ax, "C")
    ax.set_title("Per-trial spread in the local pilot")
    ax.axis("off")
    if not trial_spread.get("available"):
        ax.text(
            0.5,
            0.5,
            str(trial_spread.get("reason", "trial spread unavailable")),
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
        )
        return

    metrics = _mapping(trial_spread.get("metrics"))
    fields = [
        ("sampling_rate_hz", "Hz"),
        ("sampling_interval_cv", "CV"),
        ("gaze_dispersion_deg", "dispersion"),
        ("drift_slope_deg_s", "drift"),
    ]
    annotations = []
    for field, label in fields:
        row = _mapping(metrics.get(field))
        if not row:
            continue
        lo = float(row["min"])
        hi = float(row["max"])
        med = float(row["median"])
        annotations.append((label, lo, med, hi))

    if not annotations:
        ax.text(0.5, 0.5, "numeric trial spread unavailable", transform=ax.transAxes)
        return

    ax.text(0.08, 0.82, "metric", transform=ax.transAxes, fontsize=12, color=MUTED)
    ax.text(0.50, 0.82, "min", transform=ax.transAxes, fontsize=12, color=MUTED, ha="right")
    ax.text(0.69, 0.82, "median", transform=ax.transAxes, fontsize=12, color=MUTED, ha="right")
    ax.text(0.91, 0.82, "max", transform=ax.transAxes, fontsize=12, color=MUTED, ha="right")
    for idx, (label, lo, med, hi) in enumerate(annotations):
        y = 0.68 - idx * 0.15
        ax.axhline(y - 0.07, xmin=0.06, xmax=0.96, color=GRID, lw=0.8)
        ax.scatter([0.045], [y], transform=ax.transAxes, s=60, color=WONG[idx % len(WONG)])
        ax.text(0.08, y, label, transform=ax.transAxes, va="center", fontsize=FONT_FLOOR, color=INK)
        ax.text(
            0.50,
            y,
            f"{lo:.3g}",
            transform=ax.transAxes,
            va="center",
            fontsize=FONT_FLOOR,
            color=INK,
            ha="right",
        )
        ax.text(
            0.69,
            y,
            f"{med:.3g}",
            transform=ax.transAxes,
            va="center",
            fontsize=FONT_FLOOR,
            color=INK,
            fontweight="bold",
            ha="right",
        )
        ax.text(
            0.91,
            y,
            f"{hi:.3g}",
            transform=ax.transAxes,
            va="center",
            fontsize=FONT_FLOOR,
            color=INK,
            ha="right",
        )
    ax.text(
        0.02,
        0.04,
        "Spread is within one local derived session, not population uncertainty.",
        transform=ax.transAxes,
        fontsize=FONT_FLOOR,
        color=MUTED,
    )


def figure_synthetic_empirical_range_bridge(payload: dict[str, Any]) -> Any:
    """Render a publication figure from a precomputed bridge payload."""
    fig = plt.figure(figsize=(15.6, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.6], width_ratios=[1.05, 1.45, 1.05])
    ax_boundary = fig.add_subplot(gs[0, 0])
    ax_rows = fig.add_subplot(gs[:, 1:])
    ax_spread = fig.add_subplot(gs[1, 0])

    _draw_boundary(ax_boundary, payload)
    metrics = payload.get("metrics", [])
    _draw_metric_rows(ax_rows, [row for row in metrics if isinstance(row, dict)])
    _draw_trial_spread(ax_spread, _mapping(payload.get("trial_spread")))
    fig.suptitle(
        "Range bridge: local scale, not device validation",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.08, wspace=0.30, hspace=0.34)
    return fig
