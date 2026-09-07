"""Floating-orb eye animation: synthetic eyes as shaded 3-D orbs.

Art-informed data visualisation (not a generative image): two eyeball *orbs*
float in 3-D space against a dark editorial canvas, rotating to the true gaze,
with a coloured iris disc, a dilating black pupil, and a gaze ray — paired with
the recovered gaze and pupil traces. Renders a GIF and a static still.

    uv run python scripts/generate_orbs_animation.py
"""

from pathlib import Path

from itrace import scene
from itrace.viz.orbs_animation import generate_orbs_gif, generate_orbs_still

ROOT = Path(__file__).resolve().parents[1]


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
    res = scene.closed_loop()
    paths = [generate_orbs_still(out_dir, res), generate_orbs_gif(out_dir, res=res)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
