"""Figure-generation test (ISC-51). Skips cleanly if matplotlib is absent."""

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


def test_generate_main_sequence_png(tmp_path) -> None:  # ISC-51
    import generate_figures

    path = generate_figures.generate_main_sequence(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 1000  # a real PNG, not empty


def test_generate_direction_polar_png(tmp_path) -> None:  # ISC-51
    import generate_figures

    path = generate_figures.generate_direction_polar(tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
