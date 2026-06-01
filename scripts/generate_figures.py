"""Generate publication figures from synthetic recordings.

Thin orchestrator (the template's figures-from-``generate_*`` convention):
all computation lives in :mod:`itrace`; this script only synthesises a session,
runs the pipeline, and renders 300-dpi colour-blind-safe PNGs into
``output/figures/``.

    uv run python scripts/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from itrace import mainsequence, saccades
from itrace.synthetic import gaze_with_saccade
from itrace.types import GazeStream
from itrace.viz.gallery import render_gallery
from itrace.viz.palette import WONG, apply_house_style

# Single-source palette + house style (readable font floor, white bg, clean
# spines) shared with the CLI gallery so every manuscript figure matches.
apply_house_style()


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
    ax.text(
        0.5,
        0.5,
        f"n={directions.size}\n16 bins\nmean-vector length={resultant_len:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )
    fig.tight_layout()
    path = out_dir / "direction_polar.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> list[Path]:
    out_dir = Path(__file__).resolve().parent.parent / "output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        generate_main_sequence(out_dir),
        generate_direction_polar(out_dir),
        *render_gallery(out_dir, seed=0, animations=False),
    ]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
