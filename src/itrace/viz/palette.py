"""Shared visualization style for every iTrace figure.

Single source of truth for the Wong (2011) colour-blind-safe qualitative
palette and the house matplotlib style (readable font floor, white background,
clean spines). The CLI gallery and the manuscript figure scripts both import
from here so the published figures are visually consistent and accessible.

This follows the template convention of *one* palette and *one* accessibility
floor declared centrally, rather than a copy of the hex list in every module.
The Wong palette is the eight-colour set from Wong B. (2011), "Points of view:
Color blindness", Nature Methods 8:441; the six qualitative hues below are the
subset iTrace uses for line/scatter series.
"""

from __future__ import annotations

from typing import Any

# Wong (2011) colour-blind-safe qualitative palette.
WONG: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]

INK = "#17202f"
MUTED = "#627386"
GRID = "#d8e1ea"
PANEL_FACE = "#f8fafc"
CARD_FACE = "#ffffff"

# Readable font floor + clean print defaults. Titles reach the template's 16pt
# accessibility target; the base/tick sizes stay just below it so dense
# multi-panel scientific figures remain legible without label collisions.
# Smallest auto-rendered text (ticks) is 11pt — well above the prior 8pt.
HOUSE_RC: dict[str, object] = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": CARD_FACE,
    "axes.edgecolor": "#ccd6e0",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.45,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
}

# Minimum point size for any explicit ``fontsize=`` annotation in a figure.
FONT_FLOOR = 11


def apply_house_style() -> None:
    """Apply the iTrace house matplotlib style in place (idempotent).

    Call once at a figure entry point (gallery render, manuscript script) before
    building axes. Safe to call repeatedly; only mutates ``plt.rcParams``.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(HOUSE_RC)


def panel_label(ax: Any, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    """Place a consistent bold panel label in axes coordinates."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )


def result_box(
    ax: Any,
    text: str,
    *,
    xy: tuple[float, float] = (0.04, 0.96),
    color: str = INK,
    align: str = "left",
) -> None:
    """Draw one compact result callout with the shared publication style."""
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=align,
        va="top",
        fontsize=FONT_FLOOR,
        color=INK,
        linespacing=1.08,
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.12",
            "fc": CARD_FACE,
            "ec": color,
            "lw": 1.0,
            "alpha": 0.96,
        },
    )
