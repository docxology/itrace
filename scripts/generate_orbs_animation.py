"""Floating-orb eye animation: synthetic eyes as shaded 3-D orbs.

Art-informed data visualisation (not a generative image): two eyeball *orbs*
float in 3-D space against a dark editorial canvas, rotating to the true gaze,
with a coloured iris disc, a dilating black pupil, and a gaze ray — paired with
the recovered gaze and pupil traces. Renders a GIF and a static still.

    uv run python scripts/generate_orbs_animation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from itrace import scene
from itrace.eyemodel import Camera, Eye3D
from itrace.scene import ClosedLoopResult

# Art direction: dark editorial canvas, Wong colour-blind-safe accents.
BG = "#0d0d12"
IRIS = "#0072B2"
PUPIL = "#05050a"
RAY = "#E69F00"
GAZE_C = "#56B4E9"
PUPIL_C = "#009E73"
SCLERA = "#dfe3ea"

_R = 12.0  # eyeball radius (mm), matches eyemodel default

plt.rcParams.update(
    {
        "axes.titleweight": "bold",
        "figure.dpi": 120,
        "font.size": 10,
    }
)


def _sphere(cx: float, cy: float, cz: float, r: float, n: int = 18):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _iris_point(center: np.ndarray, yaw: float, pitch: float, depth: float) -> np.ndarray:
    g = Eye3D(yaw_deg=yaw, pitch_deg=pitch).gaze_vector()
    return center + depth * g


def _style_3d(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0, 0, 0, 0))
    ax.grid(False)


def _mask_spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(float(t[start]), float(t[end])) for start, end in zip(starts, ends, strict=True)]


def _norm01(x: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _shade_events(ax, res: ClosedLoopResult) -> None:
    for onset, offset in res.scene.true_saccades:
        ax.axvspan(res.scene.t[onset], res.scene.t[offset], color=RAY, alpha=0.13, lw=0)
    for start, end in _mask_spans(res.scene.t, res.scene.blink):
        ax.axvspan(start, end, color="white", alpha=0.09, lw=0)


def _layout(res: ClosedLoopResult):
    cam = Camera()
    centers = {
        "right": np.array([cam.interocular_mm / 2, 0.0, cam.distance_mm]),
        "left": np.array([-cam.interocular_mm / 2, 0.0, cam.distance_mm]),
    }
    fig = plt.figure(figsize=(13, 5.2), facecolor=BG)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_title("A. true 3-D eye state", color="w")
    _style_3d(ax3d)
    for c in centers.values():
        xs, ys, zs = _sphere(*c, _R)
        ax3d.plot_surface(xs, ys, zs, color=SCLERA, alpha=0.18, linewidth=0, shade=True)
    ax3d.set_xlim(-_R * 4, _R * 4)
    ax3d.set_ylim(-_R * 3, _R * 3)
    ax3d.set_zlim(cam.distance_mm - _R * 3, cam.distance_mm + _R * 3)
    ax3d.view_init(elev=8, azim=-90)
    ax3d.text2D(
        0.03,
        0.02,
        "orange rays = true gaze\niris/pupil size = scripted state",
        transform=ax3d.transAxes,
        color="0.86",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#17171f", "ec": "#3a3a46", "alpha": 0.92},
    )

    # right time-series panels
    ax_g = fig.add_subplot(2, 2, 2)
    ax_p = fig.add_subplot(2, 2, 4)
    for ax in (ax_g, ax_p):
        ax.set_facecolor(BG)
        ax.tick_params(colors="w", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#555")
    g = res.recovered_gaze
    _shade_events(ax_g, res)
    ax_g.plot(
        res.scene.t, res.scene.true_yaw, color=GAZE_C, lw=0.9, ls="--", alpha=0.75, label="true yaw"
    )
    ax_g.plot(
        res.scene.t,
        res.scene.true_pitch,
        color=RAY,
        lw=0.9,
        ls="--",
        alpha=0.75,
        label="true pitch",
    )
    ax_g.plot(g.t, g.x, color=GAZE_C, lw=1.4, label="recovered yaw")
    ax_g.plot(g.t, g.y, color=RAY, lw=1.4, label="recovered pitch")
    ax_g.set_title("B. recovered gaze (deg)", color="w", fontsize=10)
    ax_g.set_ylabel("deg", color="w")
    ax_g.grid(True, color="white", alpha=0.12)
    ax_g.legend(fontsize=6.5, facecolor=BG, labelcolor="w", edgecolor="#555", ncol=2)
    ax_g.text(
        0.72,
        0.08,
        f"RMS {res.metrics['gaze_rms_deg']:.2f} deg",
        transform=ax_g.transAxes,
        color="w",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#17171f", "ec": "#3a3a46", "alpha": 0.92},
    )
    cur_g = ax_g.axvline(g.t[0], color="w", lw=1)
    p = res.recovered_pupil
    _shade_events(ax_p, res)
    ax_p.plot(
        res.scene.t, _norm01(res.scene.true_pupil_mm), color="0.78", lw=0.9, ls="--", label="true"
    )
    ax_p.plot(p.t, _norm01(p.size), color=PUPIL_C, lw=1.5, label="recovered")
    ax_p.set_title("C. recovered pupil (normalised)", color="w", fontsize=10)
    ax_p.set_xlabel("time (s)", color="w")
    ax_p.set_ylabel("0-1", color="w")
    ax_p.grid(True, color="white", alpha=0.12)
    ax_p.legend(fontsize=7, facecolor=BG, labelcolor="w", edgecolor="#555")
    ax_p.text(
        0.02,
        0.08,
        f"r {res.metrics['pupil_corr']:.2f}",
        transform=ax_p.transAxes,
        color="w",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#17171f", "ec": "#3a3a46", "alpha": 0.92},
    )
    cur_p = ax_p.axvline(p.t[0], color="w", lw=1)
    fig.suptitle("Synthetic truth and recovered signals in the same closed loop", color="w", y=0.99)
    fig.tight_layout()
    return fig, ax3d, centers, {"cur_g": cur_g, "cur_p": cur_p}


def _draw_eyes(ax3d, centers, scn, i: int):
    arts = []
    if scn.blink[i]:
        return arts
    pupil_scale = float(scn.true_pupil_mm[i]) / 2.0
    for c in centers.values():
        ic = _iris_point(c, float(scn.true_yaw[i]), float(scn.true_pitch[i]), _R)
        gtip = _iris_point(c, float(scn.true_yaw[i]), float(scn.true_pitch[i]), _R * 2.2)
        arts.append(ax3d.scatter(*ic, color=IRIS, s=140, depthshade=False))
        arts.append(ax3d.scatter(*ic, color=PUPIL, s=60 * pupil_scale, depthshade=False))
        arts.append(
            ax3d.plot([c[0], gtip[0]], [c[1], gtip[1]], [c[2], gtip[2]], color=RAY, lw=1.5)[0]
        )
    return arts


def generate_orbs_still(out_dir: Path, res: ClosedLoopResult | None = None) -> Path:
    res = res or scene.closed_loop()
    fig, ax3d, centers, cur = _layout(res)
    scn = res.scene
    rep = int(np.flatnonzero(~scn.blink)[scn.t.size // 3])
    _draw_eyes(ax3d, centers, scn, rep)
    cur["cur_g"].set_xdata([scn.t[rep]] * 2)
    cur["cur_p"].set_xdata([scn.t[rep]] * 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eye_orbs_still.png"
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_orbs_gif(
    out_dir: Path, n_frames: int = 50, res: ClosedLoopResult | None = None
) -> Path:
    res = res or scene.closed_loop()
    fig, ax3d, centers, cur = _layout(res)
    scn = res.scene
    step = max(scn.t.size // n_frames, 1)
    idx = list(range(0, scn.t.size, step))
    dynamic: list = []

    def update(i: int):
        nonlocal dynamic
        for a in dynamic:
            a.remove()
        dynamic = _draw_eyes(ax3d, centers, scn, i)
        ax3d.view_init(elev=8, azim=-90 + 12 * np.sin(2 * np.pi * i / scn.t.size))
        cur["cur_g"].set_xdata([scn.t[i]] * 2)
        cur["cur_p"].set_xdata([scn.t[i]] * 2)
        return dynamic

    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eye_orbs.gif"
    anim.save(path, writer=PillowWriter(fps=15))
    plt.close(fig)
    return path


def main() -> list[Path]:
    out_dir = Path(__file__).resolve().parent.parent / "output" / "figures"
    res = scene.closed_loop()
    paths = [generate_orbs_still(out_dir, res), generate_orbs_gif(out_dir, res=res)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
