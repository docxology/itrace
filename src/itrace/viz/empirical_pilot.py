"""Empirical-pilot summary figure and output writer.

The ``scripts/summarize_empirical_pilot.py`` orchestrator supplies repo-root
defaults and delegates here; metrics extraction lives in :mod:`itrace.empirical`
(no figure code there, so it stays importable without matplotlib).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from itrace.empirical import load_or_create_metrics, pilot_tokens, write_json_atomic
from itrace.experiments import TRUTH_BOUNDARY
from itrace.viz.palette import CARD_FACE, INK, MUTED, PANEL_FACE, WONG, apply_house_style


def _card(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    value: str,
    *,
    color: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=CARD_FACE,
            edgecolor=color,
            linewidth=1.4,
        )
    )
    ax.text(x + 0.03, y + height * 0.66, label, ha="left", va="center", color=MUTED, fontsize=10.5)
    ax.text(
        x + 0.03,
        y + height * 0.36,
        value,
        ha="left",
        va="center",
        color=INK,
        fontsize=14,
        weight="bold",
    )


def generate_empirical_pilot_figure(metrics: Mapping[str, object], out: Path) -> Path:
    """Render the empirical pilot summary figure."""
    apply_house_style()
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = metrics.get("manuscript_tokens", {})
    if not isinstance(tokens, Mapping):
        tokens = pilot_tokens(metrics)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.88,
            boxstyle="round,pad=0.02",
            facecolor=PANEL_FACE,
            edgecolor="#d7e0e8",
        )
    )
    ax.text(0.06, 0.86, "N=1 empirical session diagnostic", fontsize=18, weight="bold", color=INK)
    ax.text(
        0.06,
        0.79,
        str(tokens.get("EMPIRICAL_PILOT_STATUS", "pending local recording")),
        fontsize=12,
        color=MUTED,
    )

    values = [
        ("finite gaze", str(tokens.get("EMPIRICAL_PILOT_FINITE_GAZE", "unavailable")), WONG[0]),
        ("sample rate", str(tokens.get("EMPIRICAL_PILOT_SAMPLING_HZ", "unavailable")), WONG[2]),
        ("sampling CV", str(tokens.get("EMPIRICAL_PILOT_SAMPLING_CV", "unavailable")), WONG[5]),
        ("max drift", str(tokens.get("EMPIRICAL_PILOT_DRIFT", "unavailable")), WONG[3]),
        ("held-out RMS", str(tokens.get("EMPIRICAL_PILOT_HELDOUT_RMS", "unavailable")), WONG[1]),
        ("target latency", str(tokens.get("EMPIRICAL_PILOT_LATENCY", "unavailable")), WONG[4]),
    ]
    for index, (label, value, color) in enumerate(values):
        row = index // 3
        col = index % 3
        _card(ax, (0.06 + col * 0.30, 0.51 - row * 0.21), 0.25, 0.15, label, value, color=color)

    boundary = str(metrics.get("truth_boundary", TRUTH_BOUNDARY))
    boundary = boundary.replace(
        "no reference-device validation or universal webcam accuracy is claimed",
        "no reference-device validation\nor universal webcam accuracy is claimed",
    )
    ax.text(
        0.06,
        0.14,
        "Boundary: order-of-magnitude package demo; " + boundary,
        fontsize=10.5,
        color=INK,
        weight="bold",
        wrap=True,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_empirical_pilot_outputs(
    *,
    report_path: Path,
    metrics_out: Path,
    figure_out: Path,
    pilot_id: str = "local_pilot_001",
    repo_root: Path,
) -> dict[str, Path]:
    """Write metrics JSON and summary figure for the empirical pilot."""
    metrics = load_or_create_metrics(report_path, repo_root=repo_root, pilot_id=pilot_id)
    return {
        "metrics": write_json_atomic(metrics_out, metrics),
        "figure": generate_empirical_pilot_figure(metrics, figure_out),
    }
