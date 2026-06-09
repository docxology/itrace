"""Generate the iTrace cover visual and graphical abstract figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from itrace import scene
from itrace.viz.palette import CARD_FACE, FONT_FLOOR, INK, MUTED, WONG, apply_house_style

ROOT = Path(__file__).resolve().parents[1]
AVAILABILITY_FOOTER = "MIT License | github.com/docxology/itrace | DOI 10.5281/zenodo.20614909"
NO_REFERENCE_DEVICE_BOUNDARY = "no reference-device claim"


def _metric_payload() -> dict[str, object]:
    path = ROOT / "docs" / "verification_metrics.json"
    if not path.exists():
        return {"test_count": "--", "coverage_pct": "--"}
    return json.loads(path.read_text(encoding="utf-8"))


def _macro_f1() -> str:
    path = ROOT / "output" / "synthetic_validation.json"
    if not path.exists():
        return "--"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cross = payload.get("cross_domain", {})
    value = cross.get("macro_saccade_f1")
    return f"{float(value):.3f}" if isinstance(value, int | float) else "--"


def _closed_loop_rms() -> str:
    result = scene.closed_loop()
    value = result.metrics.get("gaze_rms_deg")
    return f"{float(value):.2f} deg" if isinstance(value, int | float) else "--"


def _saccade_edge_px() -> str:
    path = ROOT / "output" / "figures" / "noise_metrics.json"
    if not path.exists():
        return "--"
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", {})
    if not isinstance(thresholds, dict):
        return "--"
    saccade = thresholds.get("saccade_f1_0_8", {})
    if not isinstance(saccade, dict):
        return "--"
    value = saccade.get("pixels_at_640")
    if isinstance(value, int | float):
        return f"{float(value):.1f} px"
    return "--"


def _pilot_payload() -> dict[str, object]:
    path = ROOT / "docs" / "empirical_pilot_metrics.json"
    if not path.exists():
        return {"available": False, "manuscript_tokens": {"EMPIRICAL_PILOT_STATUS": "pending"}}
    return json.loads(path.read_text(encoding="utf-8"))


def _session_payload() -> dict[str, object]:
    path = ROOT / "docs" / "empirical_sessions_summary.json"
    if not path.exists():
        return {"available_session_count": 0, "condition_count": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _range_bridge_status() -> str:
    path = ROOT / "output" / "figures" / "synthetic_empirical_range_bridge.json"
    if not path.exists():
        return "range bridge pending"
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return "range bridge unavailable"
    metric_count = summary.get("metric_count", "--")
    stress = summary.get("stress_domain_count", "--")
    non_comparable = summary.get("not_directly_comparable_count", "--")
    return f"range bridge: {metric_count} rows; {stress} stress; {non_comparable} non-comparable"


def _pilot_token(payload: dict[str, object], key: str, default: str = "unavailable") -> str:
    tokens = payload.get("manuscript_tokens", {})
    if isinstance(tokens, dict):
        value = tokens.get(key, default)
        return str(value)
    return default


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    color: str,
    subtitle: str = "",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=color,
        facecolor=CARD_FACE,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.60,
        label,
        ha="center",
        va="center",
        weight="bold",
        color=INK,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=FONT_FLOOR,
        )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color=color,
            shrinkA=8,
            shrinkB=8,
        )
    )


def _badge(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    width: float = 0.18,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.12,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0,
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + 0.06,
        text,
        color="white",
        ha="center",
        va="center",
        weight="bold",
        fontsize=11.5,
        linespacing=0.95,
    )


def _pill(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    width: float,
    height: float = 0.058,
    fontsize: float = 11.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.028",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=color,
        weight="bold",
        fontsize=fontsize,
    )


def _callout_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    value: str,
    *,
    color: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.028",
        linewidth=0,
        facecolor=color,
        alpha=0.94,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.024,
        y + height * 0.68,
        label,
        ha="left",
        va="center",
        color="white",
        fontsize=FONT_FLOOR,
        weight="bold",
    )
    ax.text(
        x + 0.024,
        y + height * 0.36,
        value,
        ha="left",
        va="center",
        color="white",
        fontsize=15.0,
        weight="bold",
    )


def _band(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    color: str,
    face: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=face,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.98,
        )
    )
    ax.text(
        x + 0.024,
        y + height - 0.042,
        label,
        ha="left",
        va="center",
        color=color,
        fontsize=FONT_FLOOR,
        weight="bold",
    )


def _camera_symbol(ax: plt.Axes, center: tuple[float, float], color: str) -> None:
    cx, cy = center
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.064, cy - 0.038),
            0.128,
            0.076,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            linewidth=1.8,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.add_patch(Rectangle((cx - 0.040, cy + 0.038), 0.045, 0.018, facecolor=color))
    ax.add_patch(Circle((cx, cy), 0.022, facecolor=color, edgecolor="none", alpha=0.92))
    ax.add_patch(Circle((cx + 0.048, cy + 0.023), 0.007, facecolor=color, edgecolor="none"))
    ax.text(cx, cy - 0.070, "camera shell", ha="center", va="center", color=INK, weight="bold")
    ax.text(
        cx,
        cy - 0.100,
        "lazy cv2 / MediaPipe",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )


def _eye_symbol(ax: plt.Axes, center: tuple[float, float], color: str) -> None:
    cx, cy = center
    ax.add_patch(Ellipse((cx, cy), 0.19, 0.095, facecolor="white", edgecolor=INK, linewidth=1.4))
    ax.add_patch(Circle((cx, cy), 0.035, facecolor=color, edgecolor="white", linewidth=1.3))
    ax.add_patch(Circle((cx, cy), 0.014, facecolor=INK, edgecolor="none"))
    for px, py in [
        (cx - 0.072, cy),
        (cx - 0.046, cy + 0.023),
        (cx - 0.046, cy - 0.023),
        (cx + 0.046, cy + 0.023),
        (cx + 0.046, cy - 0.023),
        (cx + 0.072, cy),
    ]:
        ax.add_patch(Circle((px, py), 0.007, facecolor=WONG[2], edgecolor="white", linewidth=0.7))
    _arrow(ax, (cx + 0.038, cy + 0.002), (cx + 0.13, cy + 0.052), color)
    ax.text(cx, cy - 0.073, "landmarks", ha="center", va="center", color=INK, weight="bold")
    ax.text(
        cx,
        cy - 0.103,
        "per-eye diagnostics",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )


def _signal_hub(ax: plt.Axes, center: tuple[float, float]) -> None:
    cx, cy = center
    ax.add_patch(Circle((cx, cy), 0.108, facecolor="white", edgecolor=WONG[3], linewidth=2.2))
    ax.add_patch(Circle((cx, cy), 0.063, facecolor="#fff7ed", edgecolor=WONG[3], linewidth=1.4))
    ax.text(cx, cy + 0.012, "pure", ha="center", va="center", color=INK, fontsize=15, weight="bold")
    ax.text(cx, cy - 0.026, "core", ha="center", va="center", color=INK, fontsize=15, weight="bold")
    ax.text(
        cx,
        cy - 0.090,
        "NumPy / SciPy",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )

    # Three compact signal glyphs orbit the core: gaze vector, saccade burst, pupil trace.
    _arrow(ax, (cx - 0.156, cy + 0.030), (cx - 0.095, cy + 0.070), WONG[0])
    ax.text(
        cx - 0.125,
        cy + 0.082,
        "gaze",
        ha="center",
        va="center",
        color=WONG[0],
        fontsize=FONT_FLOOR,
    )

    xs = [cx - 0.035, cx - 0.008, cx + 0.016, cx + 0.040]
    ys = [cy + 0.152, cy + 0.120, cy + 0.163, cy + 0.124]
    ax.plot(xs, ys, color=WONG[3], linewidth=2.5)
    ax.scatter(xs, ys, s=18, color=WONG[3], zorder=3)
    ax.text(
        cx + 0.002,
        cy + 0.185,
        "saccade",
        ha="center",
        va="center",
        color=WONG[3],
        fontsize=FONT_FLOOR,
    )

    trace_x = [cx + 0.088, cx + 0.112, cx + 0.136, cx + 0.160]
    trace_y = [cy - 0.055, cy - 0.030, cy - 0.070, cy - 0.045]
    ax.plot(trace_x, trace_y, color=WONG[2], linewidth=2.3)
    ax.text(
        cx + 0.150,
        cy - 0.092,
        "pupil",
        ha="center",
        va="center",
        color=WONG[2],
        fontsize=FONT_FLOOR,
    )


def _report_symbol(ax: plt.Axes, center: tuple[float, float], color: str) -> None:
    cx, cy = center
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.062, cy - 0.065),
            0.124,
            0.130,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
    )
    for i, width in enumerate((0.076, 0.092, 0.060)):
        y = cy + 0.032 - i * 0.034
        ax.plot([cx - 0.040, cx - 0.040 + width], [y, y], color=color, linewidth=2.0)
    ax.add_patch(Rectangle((cx + 0.018, cy - 0.036), 0.026, 0.026, facecolor=color, alpha=0.86))
    ax.text(cx, cy - 0.078, "reports + UI", ha="center", va="center", color=INK, weight="bold")
    ax.text(
        cx,
        cy - 0.108,
        "display-only browser",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )


def _evidence_symbol(
    ax: plt.Axes, xy: tuple[float, float], label: str, subtitle: str, color: str
) -> None:
    x, y = xy
    ax.add_patch(
        Circle((x + 0.036, y + 0.052), 0.033, facecolor=color, edgecolor="white", linewidth=1.0)
    )
    ax.add_patch(
        Polygon(
            [(x + 0.020, y + 0.052), (x + 0.032, y + 0.036), (x + 0.055, y + 0.070)],
            closed=False,
            fill=False,
            edgecolor="white",
            linewidth=2.3,
            joinstyle="round",
        )
    )
    ax.text(
        x + 0.084, y + 0.066, label, ha="left", va="center", color=INK, fontsize=11.3, weight="bold"
    )
    ax.text(
        x + 0.084, y + 0.034, subtitle, ha="left", va="center", color=MUTED, fontsize=FONT_FLOOR
    )


def _stage_label(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=FONT_FLOOR,
        color=color,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.14",
            "fc": "white",
            "ec": color,
            "lw": 1.0,
        },
    )


def generate_cover_visual(out_dir: Path) -> Path:
    """Render the visual inserted on the PDF cover page."""
    apply_house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.1), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))

    eye_center = (0.34, 0.56)
    ax.add_patch(Circle(eye_center, 0.19, facecolor="white", edgecolor="#18212f", linewidth=2.0))
    ax.add_patch(Circle(eye_center, 0.085, facecolor="#0072b2", edgecolor="white", linewidth=2.0))
    ax.add_patch(Circle(eye_center, 0.038, facecolor="#18212f", edgecolor="none"))
    for end in ((0.68, 0.78), (0.72, 0.56), (0.68, 0.34)):
        _arrow(ax, (eye_center[0] + 0.11, eye_center[1]), end, "#0072b2")
    for i, y in enumerate((0.78, 0.56, 0.34)):
        _box(
            ax,
            (0.70, y - 0.06),
            0.22,
            0.12,
            ["gaze", "saccades", "pupil"][i],
            color=WONG[i],
        )
    ax.text(0.07, 0.15, "pure core", fontsize=13, weight="bold", color="#18212f")
    ax.text(0.07, 0.10, "optional capture shell", fontsize=11, color="#607080")
    ax.text(0.72, 0.15, "verified by construction", fontsize=13, weight="bold", color="#18212f")
    ax.text(0.72, 0.10, "bounded live diagnostics", fontsize=11, color="#607080")
    path = out_dir / "cover_visual.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def generate_graphical_abstract(out_dir: Path) -> Path:
    """Render the manuscript graphical abstract."""
    apply_house_style()
    metrics = _metric_payload()
    sessions = _session_payload()
    test_gate = (
        f"{metrics.get('test_count', '--')} tests | {metrics.get('coverage_pct', '--')} coverage"
    )
    session_gate = (
        f"{sessions.get('available_session_count', '--')} sessions | "
        f"{sessions.get('condition_count', '--')} conditions"
    )
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc", edgecolor="none"))

    ax.text(
        0.045,
        0.925,
        "iTrace: eye-to-code evidence pipeline",
        fontsize=24,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.045,
        0.875,
        "Optional webcam capture feeds typed samples; the tested Python core produces "
        "reports, figures, and bounded diagnostic evidence.",
        color=MUTED,
        fontsize=13,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.045, 0.255),
            0.910,
            0.560,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor="white",
            edgecolor="#d7dee8",
            linewidth=1.2,
        )
    )

    # Observed eye: the visual source of typed samples.
    eye_center = (0.215, 0.570)
    ax.add_patch(
        Ellipse(eye_center, 0.300, 0.155, facecolor="#fff7ed", edgecolor=INK, linewidth=2.0)
    )
    ax.add_patch(Circle(eye_center, 0.058, facecolor=WONG[0], edgecolor="white", linewidth=2.0))
    ax.add_patch(Circle(eye_center, 0.023, facecolor=INK, edgecolor="none"))
    for px, py in [
        (0.095, 0.570),
        (0.132, 0.620),
        (0.132, 0.520),
        (0.298, 0.620),
        (0.298, 0.520),
        (0.335, 0.570),
    ]:
        ax.add_patch(Circle((px, py), 0.0085, facecolor=WONG[2], edgecolor="white", linewidth=0.8))
    ax.text(0.215, 0.465, "observed eye", ha="center", va="center", color=INK, weight="bold")
    ax.text(
        0.215,
        0.428,
        "landmarks -> typed samples",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )

    _arrow(ax, (0.365, 0.570), (0.425, 0.570), WONG[0])

    # Tested Python core: code-shaped middle panel, not a generic dashboard.
    ax.add_patch(
        FancyBboxPatch(
            (0.430, 0.405),
            0.230,
            0.330,
            boxstyle="round,pad=0.018,rounding_size=0.028",
            facecolor="#0f172a",
            edgecolor=WONG[3],
            linewidth=2.0,
        )
    )
    ax.text(0.455, 0.690, "tested Python core", color="white", weight="bold", fontsize=14)
    for i, line in enumerate(
        [
            "geometry -> gaze degrees",
            "saccades -> events",
            "pupil -> phase + quality",
            "pipeline -> reports",
        ]
    ):
        y = 0.640 - i * 0.054
        ax.add_patch(
            Rectangle((0.455, y - 0.016), 0.012, 0.012, facecolor=WONG[i], edgecolor="none")
        )
        ax.text(0.478, y - 0.010, line, color="#e2e8f0", fontsize=11.2, va="center")
    ax.text(
        0.545,
        0.438,
        "zero hardware imports in core",
        color="#cbd5e1",
        ha="center",
        va="center",
        fontsize=FONT_FLOOR,
    )

    _arrow(ax, (0.675, 0.570), (0.730, 0.570), WONG[4])

    # Generated reports and live displays.
    ax.add_patch(
        FancyBboxPatch(
            (0.745, 0.445),
            0.150,
            0.210,
            boxstyle="round,pad=0.014,rounding_size=0.020",
            facecolor="#f4fbff",
            edgecolor=WONG[4],
            linewidth=1.6,
        )
    )
    for y, width in [(0.610, 0.095), (0.575, 0.118), (0.540, 0.072)]:
        ax.plot([0.770, 0.770 + width], [y, y], color=WONG[4], linewidth=2.4)
    ax.plot(
        [0.770, 0.800, 0.832, 0.868], [0.500, 0.525, 0.492, 0.518], color=WONG[2], linewidth=2.5
    )
    ax.scatter([0.770, 0.800, 0.832, 0.868], [0.500, 0.525, 0.492, 0.518], s=18, color=WONG[2])
    ax.text(0.820, 0.405, "reports + UI", ha="center", va="center", color=INK, weight="bold")
    ax.text(
        0.820,
        0.368,
        "CSV / JSON / figures",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=FONT_FLOOR,
    )

    # Evidence lane beneath the main visual.
    _evidence_symbol(
        ax,
        (0.095, 0.272),
        "algorithmic evidence",
        "synthetic truth + independent 3-D loop",
        WONG[1],
    )
    _evidence_symbol(
        ax, (0.430, 0.272), "diagnostic evidence", "derived session records; no raw video", WONG[0]
    )
    _evidence_symbol(
        ax, (0.730, 0.272), "claim boundary", "not reference-device validation", WONG[5]
    )

    _callout_card(ax, (0.055, 0.120), 0.275, 0.095, "test/coverage gate", test_gate, color=WONG[4])
    _callout_card(
        ax,
        (0.365, 0.120),
        0.270,
        0.095,
        "five-session diagnostic v1",
        session_gate,
        color=WONG[0],
    )
    _callout_card(
        ax,
        (0.670, 0.120),
        0.275,
        0.095,
        "evidence boundary",
        NO_REFERENCE_DEVICE_BOUNDARY,
        color=WONG[5],
    )

    ax.text(
        0.500,
        0.057,
        AVAILABILITY_FOOTER,
        color=INK,
        fontsize=12.2,
        weight="bold",
        ha="center",
        va="center",
    )

    path = out_dir / "graphical_abstract.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
    paths = [generate_cover_visual(out_dir), generate_graphical_abstract(out_dir)]
    for path in paths:
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
