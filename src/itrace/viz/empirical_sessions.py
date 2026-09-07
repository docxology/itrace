"""Empirical-session aggregate figure and output writer.

The ``scripts/aggregate_empirical_sessions.py`` orchestrator supplies repo-root
defaults and delegates here; aggregation/validation lives in
:mod:`itrace.empirical` (no figure code there, so it stays importable without
matplotlib).
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from itrace.empirical import (
    aggregate_empirical_sessions as core_aggregate_empirical_sessions,
)
from itrace.empirical import (
    write_json_atomic,
)
from itrace.viz.palette import CARD_FACE, INK, MUTED, PANEL_FACE, WONG, apply_house_style

DEFAULT_SESSIONS_MANIFEST = Path("docs/empirical_sessions_manifest.json")
DEFAULT_SESSIONS_SUMMARY = Path("docs/empirical_sessions_summary.json")
DEFAULT_SESSIONS_FIGURE = Path("output/figures/empirical_sessions_summary.png")
"""Repo-relative defaults for the empirical-session orchestrator."""


def _fmt_pct(value: object) -> str:
    if not isinstance(value, int | float):
        return "unavailable"
    return f"{100.0 * float(value):.1f}%"


def _fmt_num(value: object, *, suffix: str = "", digits: int = 2) -> str:
    if not isinstance(value, int | float):
        return "unavailable"
    return f"{float(value):.{digits}f}{suffix}"


def _card(
    ax: Any,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    value: str,
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
    ax.text(x + 0.03, y + height * 0.66, label, ha="left", va="center", color=MUTED, fontsize=10)
    ax.text(
        x + 0.03,
        y + height * 0.36,
        value,
        ha="left",
        va="center",
        color=INK,
        fontsize=13.5,
        weight="bold",
    )


def generate_empirical_sessions_figure(summary: Mapping[str, object], out: Path) -> Path:
    """Render a compact v1-readiness figure for empirical-session intake."""
    apply_house_style()
    out.parent.mkdir(parents=True, exist_ok=True)
    readiness = summary.get("v1_readiness", {})
    if not isinstance(readiness, Mapping):
        readiness = {}
    blockers = readiness.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []

    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
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
    future_scope = summary.get("future_validation_scope", {})
    if not isinstance(future_scope, Mapping):
        future_scope = {}
    status = "diagnostic v1 ready" if readiness.get("ready") else "not diagnostic-v1 ready yet"
    ax.text(0.06, 0.86, "Empirical data intake summary", fontsize=18, weight="bold", color=INK)
    ax.text(0.06, 0.80, status, fontsize=12.5, color=WONG[2] if readiness.get("ready") else WONG[3])

    colors = [WONG[index % len(WONG)] for index in range(8)]
    values = [
        ("available sessions", str(summary.get("available_session_count", 0)), colors[0]),
        ("replicates", str(summary.get("replicate_count", 0)), colors[2]),
        ("conditions", str(summary.get("condition_count", 0)), colors[1]),
        (
            "validated reference",
            f"{summary.get('reference_evidence_count', 0)} (future)",
            colors[4],
        ),
        (
            "participant/device",
            f"{summary.get('participant_count', 0)} / {summary.get('device_count', 0)}",
            colors[5],
        ),
        (
            "held-out RMS median",
            _fmt_num(summary.get("heldout_rms_error_deg_median"), suffix=" deg"),
            colors[3],
        ),
        ("finite gaze", _fmt_pct(summary.get("finite_gaze_fraction_weighted")), colors[6]),
        ("samples", str(summary.get("total_sample_count", 0)), colors[7]),
    ]
    for index, (label, value, color) in enumerate(values):
        row = index // 4
        col = index % 4
        _card(ax, (0.06 + col * 0.225, 0.54 - row * 0.18), 0.19, 0.13, label, value, color)

    ax.text(
        0.06,
        0.275,
        "Boundary: diagnostic v1 is ready without reference evidence; reference-backed "
        "agreement is future validation scope.",
        fontsize=10.6,
        color=WONG[3],
        weight="bold",
        wrap=True,
    )
    blocker_text = "; ".join(str(item) for item in blockers[:4]) if blockers else "none"
    if len(blockers) > 4:
        blocker_text += f"; +{len(blockers) - 4} more"
    ax.text(
        0.06,
        0.19,
        "Blocking criteria: " + blocker_text,
        fontsize=10.8,
        color=INK,
        weight="bold",
        wrap=True,
    )
    missing = future_scope.get("missing_conditions", [])
    if not isinstance(missing, list):
        missing = []
    missing_text = ", ".join(str(item) for item in missing) or "none"
    future_line = (
        "Future validation: "
        f"{future_scope.get('available_sessions_remaining', 'unavailable')} more sessions, "
        f"missing conditions: {missing_text}, "
        "and validated reference evidence before population, cross-device, or "
        "device-accuracy claims."
    )
    ax.text(0.06, 0.12, textwrap.fill(future_line, width=116), fontsize=10.2, color=MUTED)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def write_empirical_session_outputs(
    *,
    manifest_path: Path,
    summary_out: Path,
    figure_out: Path,
    repo_root: Path,
) -> dict[str, Path]:
    """Write aggregate empirical-session JSON and figure outputs."""
    try:
        summary = core_aggregate_empirical_sessions(
            manifest_path=manifest_path, repo_root=repo_root
        )
    except ValueError:
        # Fall back to checked-in summary if raw report directories are absent in fresh clone
        if summary_out.exists():
            summary = json.loads(summary_out.read_text(encoding="utf-8"))
        else:
            raise
    return {
        "summary": write_json_atomic(summary_out, summary),
        "figure": generate_empirical_sessions_figure(summary, figure_out),
    }
