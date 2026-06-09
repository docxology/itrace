"""Noise-sensitivity figure + statistics table: recovery vs webcam noise.

Renders an accessible, composable 3-panel figure (one panel per signal) with 95%
CI bands, threshold crossings, and a secondary pixel axis, from a real
Monte-Carlo sweep (:mod:`itrace.power`). Also writes the same numbers as a
Markdown statistics table so prose, figure, and table share one source of truth.

Accessibility: every series uses a distinct **marker and line style** (not colour
alone), units are on every axis, σ is shown in both normalised and pixel units,
and gridlines + larger fonts aid low-vision reading.

    uv run python scripts/generate_power_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itrace import power
from itrace.power import MetricCurve, NoiseSweepResult
from itrace.viz.palette import WONG, apply_house_style, result_box

# Single-source palette + house style (readable font floor, white bg, clean
# spines) shared with every iTrace figure.
apply_house_style()

# Per-metric visual encoding — colour PLUS marker PLUS linestyle (redundant, so
# the figure reads in greyscale and for colour-blind viewers).
STYLE: dict[str, dict[str, str]] = {
    "gaze_rms_deg": {"color": WONG[0], "marker": "o", "ls": "-", "label": "gaze RMS error"},
    "saccade_f1": {"color": WONG[3], "marker": "s", "ls": "--", "label": "saccade detection F1"},
    "pupil_corr": {"color": WONG[2], "marker": "^", "ls": ":", "label": "pupil correlation"},
}
BOUND = {"gaze_rms_deg": 2.0, "saccade_f1": 0.8, "pupil_corr": 0.9}
Y_LABEL = {
    "gaze_rms_deg": "RMS gaze error (deg; lower is better)",
    "saccade_f1": "interval-overlap F1 (higher is better)",
    "pupil_corr": "Pearson r (higher is better)",
}
IMAGE_WIDTH_PX = 640.0
THRESHOLD_BOUNDS = {
    "gaze_rms_2deg": ("gaze_rms_deg", BOUND["gaze_rms_deg"]),
    "saccade_f1_0_8": ("saccade_f1", BOUND["saccade_f1"]),
    "pupil_corr_0_9": ("pupil_corr", BOUND["pupil_corr"]),
}


def _panel(ax, curve: MetricCurve, style: dict[str, str], bound: float, panel_label: str) -> None:
    """Draw one signal's mean ± CI band, threshold, and a secondary pixel axis."""
    x = curve.noise_levels
    ax.plot(
        x,
        curve.mean,
        marker=style["marker"],
        linestyle=style["ls"],
        color=style["color"],
        lw=2,
        ms=6,
        label="mean",
    )
    ax.fill_between(x, curve.ci_low, curve.ci_high, color=style["color"], alpha=0.2, label="95% CI")
    ax.axhline(bound, color="0.35", ls="-.", lw=1.2, label=f"bound = {bound:g}")
    thr = power.recovery_threshold(curve, bound)
    title = f"{panel_label}. {style['label']}"
    if thr is not None:
        ax.axvline(thr, color="0.35", ls=":", lw=1.2)
        px = power.sigma_to_pixels(thr, IMAGE_WIDTH_PX)
        result_box(
            ax,
            f"bound crossing\nsigma={thr:.4f}\n~{px:.1f} px",
            xy=(0.54, 0.78),
            color=str(style["color"]),
        )
        ax.annotate(
            "",
            xy=(thr, bound),
            xycoords="data",
            xytext=(0.54, 0.78),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0},
        )
    ax.set_title(title)
    ax.set_xlabel("landmark noise σ (normalised image units)")
    ax.set_ylabel(Y_LABEL[curve.name])
    if curve.higher_is_better:
        ax.set_ylim(0.0, 1.06)
    else:
        ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    # secondary top axis in pixels (physical grounding)
    sec = ax.secondary_xaxis(
        "top",
        functions=(
            lambda s: power.sigma_to_pixels(s, IMAGE_WIDTH_PX),
            lambda p: p / IMAGE_WIDTH_PX,
        ),
    )
    sec.set_xlabel(f"≈ pixels @ {int(IMAGE_WIDTH_PX)} px width")


def generate_power_figure(
    out_dir: Path, res: NoiseSweepResult | None = None, n_trials: int = 25
) -> Path:
    """Render the accessible 3-panel noise-sensitivity figure."""
    res = res or power.run_noise_sweep(n_trials=n_trials)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 11.0), constrained_layout=True)
    for ax, metric, label in zip(axes, STYLE, ("A", "B", "C"), strict=True):
        _panel(ax, res.curve(metric), STYLE[metric], BOUND[metric], label)
    fig.suptitle(
        f"Recovery vs idealised landmark noise (n={res.n_trials} seeded trials/level; "
        "mean and 95% bootstrap CI)",
        y=1.03,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "noise_power.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_table(
    out_dir: Path, res: NoiseSweepResult | None = None, n_trials: int = 25
) -> Path:
    """Write the statistics as an accessible Markdown table (shared source of truth)."""
    res = res or power.run_noise_sweep(n_trials=n_trials)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "noise_summary.md"
    path.write_text(power.format_summary_markdown(res, IMAGE_WIDTH_PX) + "\n")
    return path


def write_noise_metrics(
    out_dir: Path, res: NoiseSweepResult | None = None, n_trials: int = 25
) -> Path:
    """Write structured noise-sweep sidecar metrics used by figures and prose."""
    res = res or power.run_noise_sweep(n_trials=n_trials)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds: dict[str, dict[str, float | None]] = {}
    for key, (metric, bound) in THRESHOLD_BOUNDS.items():
        crossing = power.recovery_threshold(res.curve(metric), bound)
        thresholds[key] = {
            "metric": metric,
            "bound": float(bound),
            "sigma_normalised": crossing,
            "pixels_at_640": None
            if crossing is None
            else power.sigma_to_pixels(crossing, IMAGE_WIDTH_PX),
        }
    payload = {
        "kind": "itrace_noise_sweep_metrics",
        "image_width_px": IMAGE_WIDTH_PX,
        "n_trials": res.n_trials,
        "noise_levels": res.noise_levels,
        "thresholds": thresholds,
        "records": power.summary_records(res, IMAGE_WIDTH_PX),
    }
    path = out_dir / "noise_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> list[Path]:
    out_dir = Path(__file__).resolve().parent.parent / "output" / "figures"
    res = power.run_noise_sweep(n_trials=25)
    paths = [
        generate_power_figure(out_dir, res),
        write_summary_table(out_dir, res),
        write_noise_metrics(out_dir, res),
    ]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
