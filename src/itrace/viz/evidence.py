"""Visualization for statistical interpretation ledgers."""

from __future__ import annotations

import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .palette import FONT_FLOOR, GRID, INK, MUTED, WONG, panel_label


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _wrap(text: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def _line_count(text: str) -> int:
    return max(1, text.count("\n") + 1)


def _ledger_cells(row: dict[str, Any]) -> dict[str, str]:
    return {
        "statistic": _wrap(row.get("label", ""), 24),
        "value": _wrap(row.get("reported_value", ""), 26),
        "role": _wrap(row.get("interpretation", ""), 46),
        "boundary": _wrap(row.get("does_not_prove", ""), 42),
    }


def _row_height(cells: dict[str, str]) -> float:
    line_count = max(_line_count(text) for text in cells.values())
    return max(0.078, 0.034 + line_count * 0.018)


def _class_colour(evidence_class: str) -> str:
    if evidence_class in {"descriptive_statistic", "scanpath_descriptor"}:
        return WONG[0]
    if evidence_class in {"relative_model_diagnostic", "model_residual_diagnostic"}:
        return WONG[1]
    if evidence_class in {"bootstrap_uncertainty", "synthetic_stress_diagnostic"}:
        return WONG[2]
    return WONG[3]


def _badge(ax: Any, x: float, y: float, text: str, *, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y - 0.018),
        0.115,
        0.036,
        boxstyle="round,pad=0.006,rounding_size=0.011",
        transform=ax.transAxes,
        fc=color,
        ec=color,
        alpha=0.13,
        lw=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.057,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
        fontweight="bold",
    )


def figure_statistical_interpretation_ledger(payload: dict[str, Any]) -> Any:
    """Render a publication figure from a precomputed interpretation ledger."""
    rows = _rows(payload)
    display_rows = rows[:8]
    row_cells = [_ledger_cells(row) for row in display_rows]
    fig, ax = plt.subplots(figsize=(16.4, 9.8))
    ax.axis("off")
    panel_label(ax, "A", x=0.01, y=0.98)
    ax.text(
        0.055,
        0.965,
        "Statistical interpretation ledger",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.055,
        0.918,
        "Each row links a generated statistic to its estimand and boundary; "
        "visualization does not create new validation evidence.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_FLOOR,
        color=MUTED,
    )

    columns = [
        ("statistic", 0.045, "statistic", INK, "bold"),
        ("reported value", 0.248, "value", INK, "normal"),
        ("evidence role", 0.414, "role", MUTED, "normal"),
        ("does not prove", 0.665, "boundary", INK, "normal"),
    ]
    for label, x, _key, _color, _weight in columns:
        ax.text(x, 0.855, label, transform=ax.transAxes, fontsize=12, color=MUTED)

    y_top = 0.825
    for idx, (row, cells) in enumerate(zip(display_rows, row_cells, strict=True)):
        height = _row_height(cells)
        y_text = y_top - 0.018
        color = _class_colour(str(row.get("evidence_class", "")))
        y_mid = y_top - height / 2
        ax.axhline(y_top - height, xmin=0.04, xmax=0.965, color=GRID, lw=0.9)
        ax.scatter([0.035], [y_mid], transform=ax.transAxes, s=64, color=color, zorder=3)
        for _label, x, key, text_color, weight in columns:
            ax.text(
                x,
                y_text,
                cells[key],
                transform=ax.transAxes,
                va="top",
                fontsize=FONT_FLOOR,
                color=text_color,
                fontweight=weight,
                linespacing=1.02,
                gid=f"ledger-{key}-{idx}",
            )
        if not row.get("available", False):
            _badge(ax, 0.865, y_mid, "unavailable", color=WONG[3])
        y_top -= height

    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        summary_text = (
            f"{summary.get('row_count', '--')} rows | "
            f"{summary.get('available_count', '--')} available | "
            "not population physiology or device validation"
        )
    else:
        summary_text = "not population physiology or device validation"
    ax.text(
        0.05,
        0.055,
        summary_text,
        transform=ax.transAxes,
        fontsize=FONT_FLOOR,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.05,
        0.022,
        str(payload.get("truth_boundary", "")),
        transform=ax.transAxes,
        fontsize=10.4,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.08)
    return fig
