"""Full-loop animation rendering tests (ISC-69..71). Skips without matplotlib."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.mark.timeout(120)
def test_loop_summary_png(tmp_path) -> None:  # ISC-70
    import generate_loop_animation as gla

    path = gla.generate_loop_summary(tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 1000


@pytest.mark.timeout(120)
def test_loop_gif(tmp_path) -> None:  # ISC-69/ISC-71
    import generate_loop_animation as gla

    path = gla.generate_loop_gif(tmp_path, n_frames=12)
    assert path.exists()
    assert path.suffix == ".gif"
    assert path.stat().st_size > 1000  # a real, non-empty GIF
