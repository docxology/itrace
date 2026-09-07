"""Noise-sensitivity figure + statistics table: recovery vs webcam noise.

Renders an accessible, composable 3-panel figure (one panel per signal) with 95%
CI bands, threshold crossings, and a secondary pixel axis, from a real
Monte-Carlo sweep (:mod:`itrace.power`). Also writes the same numbers as a
Markdown statistics table so prose, figure, and table share one source of truth.

Accessibility: every series uses a distinct **marker and line style** (not colour
alone), units are on every axis, σ is shown in both normalised and pixel units,
and gridlines + larger fonts aid low-vision reading.
"""

from __future__ import annotations

from pathlib import Path

from itrace import power
from itrace.viz.power_figure import generate_power_figure, write_noise_metrics, write_summary_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
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
