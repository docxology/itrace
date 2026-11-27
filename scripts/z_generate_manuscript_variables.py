#!/usr/bin/env python3
"""Prepare a synthetic-only Template review from the canonical manuscript."""

from __future__ import annotations

import json
import re
import shutil
from inspect import signature
from pathlib import Path
from statistics import NormalDist

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from itrace.manuscript import (
    PILOT_TOKENS,
    SESSION_TOKENS,
    audit_crossrefs,
    load_metric_tokens,
    load_noise_tokens,
    source_files,
    substitute_tokens,
)
from itrace.scene import EyeSceneSpec
from itrace.stats.scanpath_metrics import bcea
from itrace.synthetic import SyntheticSessionSpec
from itrace.validation import SYNTHETIC_MAX_GAP_S

ROOT = Path(__file__).resolve().parents[1]
FIGURE_RE = re.compile(
    r"!\[(.*?)\]\(\.\./\.\./output/figures/([\w.-]+)\)\{#(fig:[\w-]+)([^}]*?)\}",
    re.DOTALL,
)
WITHHELD = {
    "empirical_pilot_summary.png",
    "empirical_sessions_summary.png",
    "synthetic_empirical_range_bridge.png",
    "statistical_interpretation_ledger.png",
}
CAPTION = "Historical participant-derived figure withheld from this synthetic-only review."
REVIEW_NOTE = (
    "> This review omits participant-derived values and figures. "
    "The [published manuscript](https://doi.org/10.5281/zenodo.23419306) "
    "provides the archived study context.\n\n"
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def draw_notice(path: Path) -> None:
    figure = Figure(figsize=(8, 1.4), facecolor="#eef3f7")
    FigureCanvasAgg(figure)
    figure.text(0.5, 0.62, "Historical figure withheld", ha="center", weight="bold")
    figure.text(
        0.5,
        0.35,
        "Participant-derived values are omitted from this synthetic review.",
        ha="center",
    )
    figure.savefig(path, dpi=128)


def main() -> int:
    manuscript = ROOT / "docs" / "manuscript"
    if audit_crossrefs(manuscript):
        return 1

    runtime = ROOT / "output"
    hydrated = runtime / "manuscript"
    figures = runtime / "figures"
    data = runtime / "data"
    hydrated.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    values = load_metric_tokens(ROOT / "docs" / "verification_metrics.json")
    values.update(load_noise_tokens(figures / "noise_metrics.json"))
    values.update(dict.fromkeys((*PILOT_TOKENS, *SESSION_TOKENS), "withheld in this review"))
    write_json(data / "manuscript_variables.json", values)
    write_json(
        data / "synthetic_review_parameters.json",
        {
            "synthetic_gaze_sample_rate_hz": SyntheticSessionSpec().sampling_rate_hz,
            "closed_loop_sample_rate_hz": EyeSceneSpec().sampling_rate_hz,
            "bounded_gap_limit_s": SYNTHETIC_MAX_GAP_S,
            "gaussian_mad_scale": round(1.0 / NormalDist().inv_cdf(0.75), 4),
            "bcea_default_containment_percent": round(
                100 * signature(bcea).parameters["probability"].default
            ),
        },
    )
    registry: dict[str, dict[str, str]] = {}

    def project_figure(match: re.Match[str]) -> str:
        caption, filename, label, attributes = match.groups()
        caption = " ".join(caption.split())
        if filename in WITHHELD:
            filename = f"{Path(filename).stem}-withheld.png"
            draw_notice(figures / filename)
            caption = CAPTION
        if not (figures / filename).is_file():
            raise FileNotFoundError(f"Required synthetic paper figure: {filename}")
        registry[label] = {"filename": filename, "caption": caption, "alt_text": caption}
        return f"![{caption}](../figures/{filename}){{#{label}{attributes}}}"

    for path in source_files(manuscript):
        text = substitute_tokens(path.read_text(encoding="utf-8"), values)
        text = FIGURE_RE.sub(project_figure, text)
        if path.name == "03ez_empirical_pilot.md":
            heading, newline, body = text.partition("\n")
            text = heading + newline + "\n" + REVIEW_NOTE + body.lstrip("\n")
        (hydrated / path.name).write_text(text, encoding="utf-8")
    for filename in ("config.yaml", "preamble.md", "references.bib"):
        shutil.copyfile(manuscript / filename, hydrated / filename)
    write_json(figures / "figure_registry.json", registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
