"""Summarize one derived live empirical pilot export.

The script reads the derived ``experiment_report.json`` written by the live HTML
experiment export and writes two downstream artifacts:

* ``docs/empirical_pilot_metrics.json`` for manuscript token hydration.
* ``output/figures/empirical_pilot_summary.png`` for README/manuscript figures.

No raw eye video is read or written; absent pilot data produces an explicit
``available=false`` metrics file so public prose stays honest before recording.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from itrace.empirical import empirical_metrics_from_report, write_json_atomic
from itrace.experiments import STORAGE_BOUNDARY, TRUTH_BOUNDARY
from itrace.viz.palette import CARD_FACE, INK, MUTED, PANEL_FACE, WONG, apply_house_style

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_DIR = ROOT / "output" / "empirical_pilot" / "local_pilot_001"
DEFAULT_REPORT = DEFAULT_PILOT_DIR / "experiment" / "experiment_report.json"
DEFAULT_METRICS = ROOT / "docs" / "empirical_pilot_metrics.json"
DEFAULT_FIGURE = ROOT / "output" / "figures" / "empirical_pilot_summary.png"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _weighted_mean(items: Sequence[tuple[float | None, float]]) -> float | None:
    valid = [(value, weight) for value, weight in items if value is not None and weight > 0.0]
    total = sum(weight for _value, weight in valid)
    if total <= 0.0:
        return None
    return sum(float(value) * weight for value, weight in valid if value is not None) / total


def _trial_summaries(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    trials = report.get("trials", {})
    if not isinstance(trials, Mapping):
        return []
    return [value for value in trials.values() if isinstance(value, Mapping)]


def _fmt(value: float | None, *, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}{suffix}"


def _pct(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{100.0 * value:.1f}%"


def _repo_relative_or_resolved(path: Path) -> str:
    """Return a repo-relative provenance path when the report is inside ROOT."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _tokens(payload: Mapping[str, object]) -> dict[str, str]:
    available = bool(payload.get("available"))
    sample_count = int(payload.get("sample_count", 0))
    completed = int(payload.get("completed_trial_count", 0))
    finite_gaze = _number(payload.get("finite_gaze_fraction"))
    sampling_hz = _number(payload.get("sampling_rate_hz"))
    sampling_cv = _number(payload.get("sampling_interval_cv"))
    drift = _number(payload.get("max_drift_deg_s"))
    heldout = payload.get("heldout_target_error", {})
    latency = payload.get("target_acquisition_latency_s", {})
    heldout_rms = _number(heldout.get("rms_error_deg")) if isinstance(heldout, Mapping) else None
    median_latency = (
        _number(latency.get("median_latency_s")) if isinstance(latency, Mapping) else None
    )
    return {
        "EMPIRICAL_PILOT_STATUS": (
            f"{completed} completed trials, {sample_count} derived samples"
            if available
            else "pending local recording"
        ),
        "EMPIRICAL_PILOT_FINITE_GAZE": _pct(finite_gaze),
        "EMPIRICAL_PILOT_SAMPLING_HZ": _fmt(sampling_hz, suffix=" Hz", digits=1),
        "EMPIRICAL_PILOT_SAMPLING_CV": _fmt(sampling_cv, digits=3),
        "EMPIRICAL_PILOT_DRIFT": _fmt(drift, suffix=" deg/s", digits=3),
        "EMPIRICAL_PILOT_HELDOUT_RMS": _fmt(heldout_rms, suffix=" deg", digits=2),
        "EMPIRICAL_PILOT_LATENCY": _fmt(median_latency, suffix=" s median", digits=2),
    }


def unavailable_metrics(
    *,
    pilot_id: str = "local_pilot_001",
    reason: str = "pilot report not recorded yet",
) -> dict[str, object]:
    """Return an explicit unavailable metrics payload."""
    payload: dict[str, object] = {
        "kind": "empirical_pilot_metrics",
        "pilot_id": pilot_id,
        "available": False,
        "reason": reason,
        "source_report": None,
        "sample_count": 0,
        "completed_trial_count": 0,
        "finite_gaze_fraction": None,
        "sampling_rate_hz": None,
        "sampling_interval_cv": None,
        "max_drift_deg_s": None,
        "pupil_valid_fraction": None,
        "heldout_target_error": {"available": False, "reason": reason},
        "target_acquisition_latency_s": {"available": False, "reason": reason},
        "truth_boundary": TRUTH_BOUNDARY,
        "storage_boundary": STORAGE_BOUNDARY,
    }
    payload["manuscript_tokens"] = _tokens(payload)
    return payload


def metrics_from_report(
    report: Mapping[str, object],
    *,
    source_report: Path,
    pilot_id: str = "local_pilot_001",
) -> dict[str, object]:
    """Extract manuscript-ready metrics from a derived experiment report."""
    payload = empirical_metrics_from_report(
        report,
        source_report=source_report,
        repo_root=ROOT,
        pilot_id=pilot_id,
    )
    payload["manuscript_tokens"] = _tokens(payload)
    return payload


def load_or_create_metrics(
    report_path: Path,
    *,
    pilot_id: str = "local_pilot_001",
) -> dict[str, object]:
    """Load a report if present; otherwise return an unavailable payload."""
    if not report_path.exists():
        return unavailable_metrics(pilot_id=pilot_id)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return unavailable_metrics(pilot_id=pilot_id, reason="pilot report is not a JSON object")
    return metrics_from_report(payload, source_report=report_path, pilot_id=pilot_id)


def write_metrics(metrics: Mapping[str, object], out: Path) -> Path:
    """Write empirical metrics JSON."""
    return write_json_atomic(out, metrics)


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
        tokens = _tokens(metrics)

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
    report_path: Path = DEFAULT_REPORT,
    metrics_out: Path = DEFAULT_METRICS,
    figure_out: Path = DEFAULT_FIGURE,
    *,
    pilot_id: str = "local_pilot_001",
) -> dict[str, Path]:
    """Write metrics JSON and summary figure for the empirical pilot."""
    metrics = load_or_create_metrics(report_path, pilot_id=pilot_id)
    return {
        "metrics": write_metrics(metrics, metrics_out),
        "figure": generate_empirical_pilot_figure(metrics, figure_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the iTrace empirical pilot export.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--pilot-id", default="local_pilot_001")
    args = parser.parse_args()

    paths = write_empirical_pilot_outputs(
        report_path=args.report,
        metrics_out=args.metrics_out,
        figure_out=args.figure_out,
        pilot_id=args.pilot_id,
    )
    for key, path in paths.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
