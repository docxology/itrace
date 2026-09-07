"""Render the full iTrace loop as an animation: 3-D eye -> projection ->
recovered gaze + pupil.

Thin orchestrator: all computation is in :mod:`itrace.scene`; panel layout and
GIF/PNG rendering live in :mod:`itrace.viz.loop_animation`.

    uv run python scripts/generate_loop_animation.py
"""

from pathlib import Path

from itrace import scene
from itrace.viz.loop_animation import generate_loop_gif, generate_loop_summary

ROOT = Path(__file__).resolve().parents[1]


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
    res = scene.closed_loop()
    paths = [generate_loop_summary(out_dir, res), generate_loop_gif(out_dir, res=res)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
