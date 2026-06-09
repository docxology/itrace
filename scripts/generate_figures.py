"""Generate publication figures from synthetic recordings.

Thin orchestrator (the template's figures-from-``generate_*`` convention):
all computation lives in :mod:`itrace`; this script only synthesises a session,
runs the pipeline, and renders 300-dpi colour-blind-safe PNGs into
``output/figures/``.

    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from aggregate_empirical_sessions import (
    DEFAULT_FIGURE as DEFAULT_EMPIRICAL_SESSIONS_FIGURE,
)
from aggregate_empirical_sessions import (
    DEFAULT_MANIFEST as DEFAULT_EMPIRICAL_SESSIONS_MANIFEST,
)
from aggregate_empirical_sessions import (
    DEFAULT_SUMMARY as DEFAULT_EMPIRICAL_SESSIONS_SUMMARY,
)
from aggregate_empirical_sessions import (
    write_empirical_session_outputs,
)
from generate_graphical_abstract import generate_cover_visual, generate_graphical_abstract
from generate_loop_animation import generate_loop_gif, generate_loop_summary
from generate_orbs_animation import generate_orbs_gif, generate_orbs_still
from generate_power_figure import generate_power_figure, write_noise_metrics, write_summary_table
from itrace import mainsequence, power, saccades, scene
from itrace.synthetic import gaze_with_saccade
from itrace.types import GazeStream
from itrace.viz.gallery import render_gallery
from itrace.viz.palette import WONG, apply_house_style
from summarize_empirical_pilot import (
    DEFAULT_FIGURE as DEFAULT_EMPIRICAL_FIGURE,
)
from summarize_empirical_pilot import (
    DEFAULT_METRICS as DEFAULT_EMPIRICAL_METRICS,
)
from summarize_empirical_pilot import (
    DEFAULT_REPORT as DEFAULT_EMPIRICAL_REPORT,
)
from summarize_empirical_pilot import (
    write_empirical_pilot_outputs,
)

# Single-source palette + house style (readable font floor, white bg, clean
# spines) shared with the CLI gallery so every manuscript figure matches.
apply_house_style()

ROOT = Path(__file__).resolve().parent.parent
FIGURE_MANIFEST = ROOT / "docs" / "figure_manifest.json"

SOURCE_MAP: dict[str, dict[str, object]] = {
    "cover_visual.png": {
        "script": "scripts/generate_graphical_abstract.py",
        "source_data": ["docs/verification_metrics.json"],
    },
    "graphical_abstract.png": {
        "script": "scripts/generate_graphical_abstract.py",
        "source_data": [
            "docs/verification_metrics.json",
            "docs/empirical_sessions_summary.json",
            "LICENSE",
            "CITATION.cff",
        ],
    },
    "empirical_pilot_summary.png": {
        "script": "scripts/summarize_empirical_pilot.py",
        "source_data": ["docs/empirical_pilot_metrics.json"],
    },
    "empirical_sessions_summary.png": {
        "script": "scripts/aggregate_empirical_sessions.py",
        "source_data": [
            "docs/empirical_sessions_manifest.json",
            "docs/empirical_sessions_summary.json",
        ],
    },
    "main_sequence.png": {
        "script": "scripts/generate_figures.py",
        "source_data": ["src/itrace/synthetic.py"],
    },
    "direction_polar.png": {
        "script": "scripts/generate_figures.py",
        "source_data": ["src/itrace/synthetic.py"],
    },
    "closed_loop_summary.png": {
        "script": "scripts/generate_loop_animation.py",
        "source_data": ["src/itrace/scene.py"],
    },
    "closed_loop.gif": {
        "script": "scripts/generate_loop_animation.py",
        "source_data": ["src/itrace/scene.py"],
    },
    "eye_orbs_still.png": {
        "script": "scripts/generate_orbs_animation.py",
        "source_data": ["src/itrace/scene.py"],
    },
    "eye_orbs.gif": {
        "script": "scripts/generate_orbs_animation.py",
        "source_data": ["src/itrace/scene.py"],
    },
    "noise_power.png": {
        "script": "scripts/generate_power_figure.py",
        "source_data": ["output/figures/noise_metrics.json"],
    },
    "noise_summary.md": {
        "script": "scripts/generate_power_figure.py",
        "source_data": ["output/figures/noise_metrics.json"],
    },
    "noise_metrics.json": {
        "script": "scripts/generate_power_figure.py",
        "source_data": ["src/itrace/power.py", "src/itrace/scene.py"],
    },
    "statistical_diagnostics.png": {
        "script": "scripts/generate_figures.py",
        "source_data": ["output/figures/statistical_diagnostics.json"],
    },
    "statistical_diagnostics.json": {
        "script": "scripts/generate_figures.py",
        "source_data": [
            "src/itrace/synthetic.py",
            "src/itrace/stats/diagnostics.py",
            "src/itrace/stats/distributions.py",
            "src/itrace/stats/scanpath_metrics.py",
            "src/itrace/stats/similarity.py",
        ],
    },
    "synthetic_empirical_range_bridge.png": {
        "script": "scripts/generate_figures.py",
        "source_data": ["output/figures/synthetic_empirical_range_bridge.json"],
    },
    "synthetic_empirical_range_bridge.json": {
        "script": "scripts/generate_figures.py",
        "source_data": [
            "docs/empirical_pilot_metrics.json",
            "output/synthetic_validation.json",
            "output/figures/noise_metrics.json",
            "output/figures/statistical_diagnostics.json",
            "src/itrace/stats/range_bridge.py",
        ],
    },
    "statistical_interpretation_ledger.png": {
        "script": "scripts/generate_figures.py",
        "source_data": ["output/figures/statistical_interpretation_ledger.json"],
    },
    "statistical_interpretation_ledger.json": {
        "script": "scripts/generate_figures.py",
        "source_data": [
            "docs/empirical_pilot_metrics.json",
            "output/figures/noise_metrics.json",
            "output/figures/statistical_diagnostics.json",
            "output/figures/synthetic_empirical_range_bridge.json",
            "src/itrace/stats/evidence.py",
        ],
    },
}


def _source_metadata(path_name: str) -> dict[str, object]:
    """Return manifest source metadata, including dynamic empirical provenance."""
    source = dict(
        SOURCE_MAP.get(
            path_name,
            {
                "script": "scripts/generate_figures.py",
                "source_data": ["src/itrace/synthetic.py"],
            },
        )
    )
    if (
        path_name
        in {
            "empirical_pilot_summary.png",
            "synthetic_empirical_range_bridge.json",
            "statistical_interpretation_ledger.json",
        }
        and DEFAULT_EMPIRICAL_METRICS.exists()
    ):
        payload = json.loads(DEFAULT_EMPIRICAL_METRICS.read_text(encoding="utf-8"))
        source_report = payload.get("source_report")
        if payload.get("available") and isinstance(source_report, str):
            source_data = list(source["source_data"])
            if source_report not in source_data:
                source_data.append(source_report)
            source["source_data"] = source_data
    if (
        path_name == "empirical_sessions_summary.png"
        and DEFAULT_EMPIRICAL_SESSIONS_SUMMARY.exists()
    ):
        payload = json.loads(DEFAULT_EMPIRICAL_SESSIONS_SUMMARY.read_text(encoding="utf-8"))
        source_data = list(source["source_data"])
        sessions = payload.get("sessions", [])
        if isinstance(sessions, list):
            for row in sessions:
                if not isinstance(row, dict):
                    continue
                source_report = row.get("source_report") or row.get("report")
                if isinstance(source_report, str) and source_report not in source_data:
                    source_data.append(source_report)
        source["source_data"] = source_data
    return source


def _current_empirical_report() -> tuple[Path, str]:
    """Return the configured current pilot report, falling back to the default."""
    if not DEFAULT_EMPIRICAL_METRICS.exists():
        return DEFAULT_EMPIRICAL_REPORT, "local_pilot_001"
    try:
        payload = json.loads(DEFAULT_EMPIRICAL_METRICS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_EMPIRICAL_REPORT, "local_pilot_001"
    source_report = payload.get("source_report")
    pilot_id = payload.get("pilot_id")
    if not isinstance(source_report, str) or not source_report:
        return DEFAULT_EMPIRICAL_REPORT, str(pilot_id or "local_pilot_001")
    report_path = Path(source_report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    if not report_path.exists():
        return DEFAULT_EMPIRICAL_REPORT, str(pilot_id or "local_pilot_001")
    return report_path, str(pilot_id or report_path.parents[1].name)


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    rng = np.random.default_rng(seed)
    xs, ys, ts = [], [], []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = ts[-1][-1] + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def generate_main_sequence(out_dir: Path) -> Path:
    """Main-sequence scatter with fitted saturating and power-law models."""
    stream = _multi_saccade_stream()
    _f, saccs = saccades.detect_ivt(stream)
    props = saccades.saccade_properties(saccs)
    amp, vel = props["amplitude_deg"], props["peak_velocity_deg_s"]
    direction = props["direction_deg"]
    fit = mainsequence.fit(amp, vel)

    grid = np.linspace(amp.min(), amp.max(), 200)
    power_curve = fit["power_a"] * grid ** fit["power_b"]
    sat_curve = fit["v_max"] * (1.0 - np.exp(-grid / fit["C"]))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    scatter = ax_lin.scatter(
        amp,
        vel,
        c=direction,
        cmap="twilight_shifted",
        vmin=-180,
        vmax=180,
        s=42,
        edgecolor="white",
        linewidth=0.6,
        label="detected saccades",
        zorder=3,
    )
    ax_lin.plot(
        grid,
        sat_curve,
        color=WONG[3],
        lw=2.4,
        label="saturating fit",
    )
    ax_lin.set_xlabel("amplitude (deg)")
    ax_lin.set_ylabel("peak velocity (deg/s)")
    ax_lin.set_title("A. recovered main sequence")
    ax_lin.grid(True, alpha=0.28)
    ax_lin.legend(loc="lower right", frameon=False)
    ax_lin.text(
        0.04,
        0.95,
        f"n={amp.size}\nVmax={fit['v_max']:.0f} deg/s\nC={fit['C']:.1f} deg",
        transform=ax_lin.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )
    cbar = fig.colorbar(scatter, ax=ax_lin, pad=0.01, fraction=0.05)
    cbar.set_label("saccade direction (deg)")

    ax_log.scatter(
        amp,
        vel,
        color=WONG[0],
        s=34,
        edgecolor="white",
        linewidth=0.5,
        label="detected saccades",
        zorder=3,
    )
    ax_log.plot(grid, power_curve, color=WONG[4], lw=2.2, label="power-law fit")
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("amplitude (deg, log scale)")
    ax_log.set_ylabel("peak velocity (deg/s, log scale)")
    ax_log.set_title("B. log-log model check")
    ax_log.grid(True, which="both", alpha=0.25)
    ax_log.legend(loc="lower right", frameon=False)
    ax_log.text(
        0.04,
        0.95,
        f"V = a A^b\nb={fit['power_b']:.2f}\nlog-log R2={fit['r_squared_power']:.3f}",
        transform=ax_log.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )
    path = out_dir / "main_sequence.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def generate_direction_polar(out_dir: Path) -> Path:
    """Polar histogram of saccade directions."""
    stream = _multi_saccade_stream()
    _f, saccs = saccades.detect_ivt(stream)
    directions = np.radians([s.direction_deg for s in saccs])

    fig, ax = plt.subplots(figsize=(6.2, 5.4), subplot_kw={"projection": "polar"})
    counts, edges = np.histogram(directions, bins=16, range=(-np.pi, np.pi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    cmap = plt.colormaps["viridis"]
    max_count = max(int(counts.max()), 1)
    colors = [cmap(0.25 + 0.65 * (c / max_count)) for c in counts]
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.bar(
        centers,
        counts,
        width=np.diff(edges) * 0.92,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
    )
    resultant = np.array([np.cos(directions).mean(), np.sin(directions).mean()])
    resultant_len = float(np.hypot(*resultant))
    resultant_ang = float(np.arctan2(resultant[1], resultant[0]))
    ax.annotate(
        "",
        xy=(resultant_ang, max_count * resultant_len),
        xytext=(resultant_ang, 0),
        arrowprops={"arrowstyle": "-|>", "color": WONG[3], "lw": 2.2},
    )
    ax.set_title("Saccade direction distribution\n0 deg = right, +90 deg = up", pad=18)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    fig.text(
        0.5,
        0.06,
        f"n={directions.size} | 16 bins | mean-vector length={resultant_len:.2f}",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )
    fig.subplots_adjust(bottom=0.2, top=0.82)
    path = out_dir / "direction_polar.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _figure_metadata(path: Path) -> dict[str, object]:
    """Return provenance + image metadata for one generated figure artifact."""
    source = _source_metadata(path.name)
    rel = path.resolve().relative_to(ROOT)
    suffix = path.suffix.lower()
    artifact_kind = {
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "animation",
        ".md": "table",
        ".json": "data",
    }.get(suffix, "artifact")
    entry: dict[str, object] = {
        "path": str(rel),
        "artifact_kind": artifact_kind,
        "script": source["script"],
        "source_data": source["source_data"],
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if suffix in {".png", ".jpg", ".jpeg"}:
        image = mpimg.imread(path)
        entry["width_px"] = int(image.shape[1])
        entry["height_px"] = int(image.shape[0])
        entry["pixel_std"] = float(np.std(image[..., :3]))
    elif suffix == ".gif":
        try:
            from PIL import Image
        except ImportError:
            entry["width_px"] = 0
            entry["height_px"] = 0
            entry["pixel_std"] = 0.0
        else:
            with Image.open(path) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
                entry["width_px"] = int(image.width)
                entry["height_px"] = int(image.height)
                entry["pixel_std"] = float(np.std(frame))
                entry["frame_count"] = int(getattr(image, "n_frames", 1))
    else:
        entry["width_px"] = 0
        entry["height_px"] = 0
        entry["pixel_std"] = 0.0
    return entry


def write_figure_manifest(paths: list[Path], out: Path = FIGURE_MANIFEST) -> Path:
    """Write a provenance manifest for generated publication figure artifacts."""
    metrics_path = ROOT / "docs" / "verification_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    unique_paths = sorted({path.resolve() for path in paths if path.exists()})
    payload = {
        "kind": "itrace_publication_figure_manifest",
        "version": metrics.get("version", "0.4.1"),
        "updated": metrics.get("gate_date", "2026-06-05"),
        "truth_boundary": (
            "Figures visualize Python-computed outputs and generated evidence; "
            "browser/canvas rendering is display-only and does not create validation evidence."
        ),
        "figures": [_figure_metadata(path) for path in unique_paths],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    power_res = power.run_noise_sweep(n_trials=25)
    loop_res = scene.closed_loop()
    power_paths = [
        generate_power_figure(out_dir, power_res),
        write_summary_table(out_dir, power_res),
        write_noise_metrics(out_dir, power_res),
    ]
    empirical_report, pilot_id = _current_empirical_report()
    empirical_paths = write_empirical_pilot_outputs(
        report_path=empirical_report,
        metrics_out=DEFAULT_EMPIRICAL_METRICS,
        figure_out=DEFAULT_EMPIRICAL_FIGURE,
        pilot_id=pilot_id,
    )
    empirical_session_paths = write_empirical_session_outputs(
        manifest_path=DEFAULT_EMPIRICAL_SESSIONS_MANIFEST,
        summary_out=DEFAULT_EMPIRICAL_SESSIONS_SUMMARY,
        figure_out=DEFAULT_EMPIRICAL_SESSIONS_FIGURE,
    )
    gallery_paths = render_gallery(out_dir, seed=0, animations=True)
    paths = [
        *power_paths,
        generate_loop_summary(out_dir, loop_res),
        generate_loop_gif(out_dir, res=loop_res),
        generate_orbs_still(out_dir, loop_res),
        generate_orbs_gif(out_dir, res=loop_res),
        empirical_paths["figure"],
        empirical_session_paths["figure"],
        *gallery_paths,
        generate_cover_visual(out_dir),
        generate_graphical_abstract(out_dir),
        generate_main_sequence(out_dir),
        generate_direction_polar(out_dir),
    ]
    paths.append(write_figure_manifest(paths))
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
