"""Generate the iTrace cover visual and graphical abstract figures.

Thin orchestrator over :mod:`itrace.viz.graphical_abstract`: path bootstrap and
repo-root plumbing only; all figure construction lives in the module.

    uv run python scripts/generate_graphical_abstract.py
"""

from __future__ import annotations

from pathlib import Path

from itrace.viz.graphical_abstract import (
    AVAILABILITY_FOOTER as AVAILABILITY_FOOTER,
)
from itrace.viz.graphical_abstract import (
    NO_REFERENCE_DEVICE_BOUNDARY as NO_REFERENCE_DEVICE_BOUNDARY,
)
from itrace.viz.graphical_abstract import (
    generate_cover_visual as _generate_cover_visual,
)
from itrace.viz.graphical_abstract import (
    generate_graphical_abstract as _generate_graphical_abstract,
)

ROOT = Path(__file__).resolve().parents[1]


def generate_cover_visual(out_dir: Path) -> Path:
    """Render the visual inserted on the PDF cover page."""
    return _generate_cover_visual(out_dir)


def generate_graphical_abstract(out_dir: Path, repo_root: Path = ROOT) -> Path:
    """Render the manuscript graphical abstract."""
    return _generate_graphical_abstract(out_dir, repo_root)


def main() -> list[Path]:
    out_dir = ROOT / "output" / "figures"
    paths = [generate_cover_visual(out_dir), generate_graphical_abstract(out_dir)]
    for p in paths:
        print(f"wrote {p}")
    return paths


if __name__ == "__main__":
    main()
