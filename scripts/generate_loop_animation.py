"""Render the full iTrace loop as an animation: 3-D eye -> projection ->
recovered gaze + pupil.

Thin orchestrator: all computation is in :mod:`itrace.scene`; this script only
lays out matplotlib panels and writes a GIF (via PillowWriter) plus a static
300-dpi summary PNG into ``output/figures/``.

    uv run python scripts/generate_loop_animation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from itrace import capture, saccades, scene
from itrace.scene import ClosedLoopResult

from itrace.viz.palette import WONG, apply_house_style

# Single-source palette + house style (readable font floor, white bg, clean
# spines) shared with every iTrace figure.
apply_house_style()


def _iris_center(landmarks: np.ndarray, idx: tuple[int, ...]) -> tuple[float, float]:
    pts = landmarks[list(idx)]
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _both_eyes(landmarks: np.ndarray) -> tuple[list[float], list[float], np.ndarray]:
    """Iris-centre x/y for both eyes plus the 4 corner points (for plotting)."""
    rx, ry = _iris_center(landmarks, capture.RIGHT_IRIS)
    lx, ly = _iris_center(landmarks, capture.LEFT_IRIS)
    corners = landmarks[list(capture.RIGHT_EYE_CORNERS) + list(capture.LEFT_EYE_CORNERS)]
    return [rx, lx], [ry, ly], corners


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


def _draw_event_spans(ax: plt.Axes, res: ClosedLoopResult) -> None:
    for onset, offset in res.scene.true_saccades:
        ax.axvspan(res.scene.t[onset], res.scene.t[offset], color=WONG[4], alpha=0.10, lw=0)
    for start, end in _mask_spans(res.scene.t, res.scene.blink):
        ax.axvspan(start, end, color="0.2", alpha=0.12, lw=0)


def _layout(res: ClosedLoopResult) -> tuple[plt.Figure, dict[str, object]]:
    fig, (ax_eye, ax_gaze, ax_pupil) = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.45, 1.1]},
        constrained_layout=True,
    )

    # Panel 1 — projected eye (image plane), iris dot moves with gaze.
    ax_eye.set_title("A. projected landmarks")
    ax_eye.set_xlim(0.10, 0.90)
    ax_eye.set_ylim(0.57, 0.44)  # image y grows downward
    ax_eye.set_aspect("equal")
    ax_eye.set_xlabel("normalised image x")
    ax_eye.set_ylabel("normalised image y")
    ax_eye.grid(True, alpha=0.2)
    valid = np.flatnonzero(~res.scene.blink)
    if valid.size:
        right_x, right_y, left_x, left_y = [], [], [], []
        for idx in valid:
            xs, ys, _corners = _both_eyes(res.scene.landmarks[int(idx)])
            right_x.append(xs[0])
            right_y.append(ys[0])
            left_x.append(xs[1])
            left_y.append(ys[1])
        ax_eye.plot(right_x, right_y, color=WONG[0], lw=1.1, alpha=0.6, label="right iris path")
        ax_eye.plot(left_x, left_y, color=WONG[4], lw=1.1, alpha=0.6, label="left iris path")
    (iris_dot,) = ax_eye.plot([], [], "o", color=WONG[0], ms=11, mec="white", mew=0.8)
    corners = ax_eye.plot([], [], "x", color="0.35", ms=7, mew=1.2)[0]

    # Panel 2 — recovered gaze with detected saccades.
    g = res.recovered_gaze
    ax_gaze.set_title("B. gaze recovery and events")
    _draw_event_spans(ax_gaze, res)
    ax_gaze.plot(res.scene.t, res.scene.true_yaw, color=WONG[0], lw=1.0, ls="--", label="true yaw")
    ax_gaze.plot(
        res.scene.t, res.scene.true_pitch, color=WONG[1], lw=1.0, ls="--", label="true pitch"
    )
    ax_gaze.plot(g.t, g.x, color=WONG[0], lw=1.6, label="recovered yaw")
    ax_gaze.plot(g.t, g.y, color=WONG[1], lw=1.6, label="recovered pitch")
    _f, saccs = saccades.detect_ivt(g)
    for s in saccs:
        ax_gaze.axvspan(s.onset_t, s.offset_t, color=WONG[3], alpha=0.16)
    ax_gaze.set_xlabel("time (s)")
    ax_gaze.set_ylabel("degrees of visual angle")
    ax_gaze.grid(True, alpha=0.25)
    ax_gaze.legend(loc="upper right", ncol=2, frameon=False)
    ax_gaze.text(
        0.02,
        0.04,
        f"RMS={res.metrics['gaze_rms_deg']:.2f} deg\nF1={res.metrics['saccade_f1']:.2f}",
        transform=ax_gaze.transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.82", "alpha": 0.9},
    )
    cursor_g = ax_gaze.axvline(g.t[0], color="0.3", lw=1)

    # Panel 3 — recovered (deblinked) pupil.
    p = res.recovered_pupil
    ax_pupil.set_title("C. pupil recovery")
    _draw_event_spans(ax_pupil, res)
    ax_pupil.plot(
        res.scene.t, _norm01(res.scene.true_pupil_mm), color="0.25", lw=1.1, ls="--", label="true"
    )
    ax_pupil.plot(p.t, _norm01(p.size), color=WONG[2], lw=1.7, label="recovered")
    ax_pupil.set_xlabel("time (s)")
    ax_pupil.set_ylabel("normalised pupil signal")
    ax_pupil.grid(True, alpha=0.25)
    ax_pupil.legend(loc="upper right", frameon=False)
    ax_pupil.text(
        0.03,
        0.04,
        f"pupil r={res.metrics['pupil_corr']:.2f}",
        transform=ax_pupil.transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.82", "alpha": 0.9},
    )
    cursor_p = ax_pupil.axvline(p.t[0], color="0.3", lw=1)

    artists = {
        "iris_dot": iris_dot,
        "corners": corners,
        "cursor_g": cursor_g,
        "cursor_p": cursor_p,
    }
    return fig, artists


def generate_loop_summary(out_dir: Path, res: ClosedLoopResult | None = None) -> Path:
    """Write a static 300-dpi PNG of the full loop (ISC-70)."""
    res = res or scene.closed_loop()
    fig, art = _layout(res)
    # populate the eye panel with a representative (mid-trajectory) frame
    scn = res.scene
    rep = int(np.flatnonzero(~scn.blink)[scn.t.size // 3])
    xs, ys, cps = _both_eyes(scn.landmarks[rep])
    art["iris_dot"].set_data(xs, ys)
    art["corners"].set_data(cps[:, 0], cps[:, 1])
    m = res.metrics
    fig.suptitle(
        f"iTrace closed loop — gaze RMS {m['gaze_rms_deg']:.2f}°, "
        f"{int(m['n_saccades'])} saccades, pupil r={m['pupil_corr']:.2f}",
        y=1.06,
        fontsize=11,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "closed_loop_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_loop_gif(
    out_dir: Path, n_frames: int = 60, res: ClosedLoopResult | None = None
) -> Path:
    """Write an animated GIF of the full loop (ISC-69)."""
    res = res or scene.closed_loop()
    fig, art = _layout(res)
    scn = res.scene
    step = max(scn.t.size // n_frames, 1)
    frame_idx = list(range(0, scn.t.size, step))

    def update(i: int) -> tuple[object, ...]:
        lm = scn.landmarks[i]
        if np.all(np.isfinite(lm[list(capture.RIGHT_IRIS)])):
            xs, ys, cps = _both_eyes(lm)
            art["iris_dot"].set_data(xs, ys)
            art["corners"].set_data(cps[:, 0], cps[:, 1])
        else:  # blink frame
            art["iris_dot"].set_data([], [])
        art["cursor_g"].set_xdata([scn.t[i], scn.t[i]])
        art["cursor_p"].set_xdata([scn.t[i], scn.t[i]])
        return tuple(art.values())

    anim = FuncAnimation(fig, update, frames=frame_idx, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "closed_loop.gif"
    anim.save(path, writer=PillowWriter(fps=15))
    plt.close(fig)
    return path


def main() -> list[Path]:
    out_dir = Path(__file__).resolve().parent.parent / "output" / "figures"
    res = scene.closed_loop()
    paths = [generate_loop_summary(out_dir, res), generate_loop_gif(out_dir, res=res)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
