"""Tests for the synthetic-to-empirical range bridge figure."""

from __future__ import annotations

import importlib.util
import re
from itertools import pairwise

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)


def _assert_text_columns_do_not_overlap(fig, prefixes: tuple[str, ...]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for prefix in prefixes:
        boxes = [
            text.get_window_extent(renderer)
            for ax in fig.axes
            for text in ax.texts
            if str(text.get_gid()).startswith(prefix)
        ]
        boxes = sorted(boxes, key=lambda box: box.y0)
        for lower, upper in pairwise(boxes):
            assert lower.y1 <= upper.y0 + 1.5, prefix


def test_range_bridge_figure_renders_payload(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.stats.range_bridge import load_synthetic_empirical_range_bridge
    from itrace.viz.range_bridge import figure_synthetic_empirical_range_bridge

    payload = load_synthetic_empirical_range_bridge()
    fig = figure_synthetic_empirical_range_bridge(payload)
    try:
        assert isinstance(fig, Figure)
        labels = "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)
        normalized = re.sub(r"\s+", " ", labels)
        assert "Observed local scale" in labels
        assert "not reference-device validation" in labels
        assert "finite gaze fraction" in labels
        assert "held-out target RMS" in labels
        assert "not directly comparable" in normalized
        assert "Boundary: local scale, not device validation" in labels
        _assert_text_columns_do_not_overlap(
            fig,
            (
                "bridge-metric-",
                "bridge-value-",
                "bridge-evidence-",
            ),
        )
        out = tmp_path / "synthetic_empirical_range_bridge.png"
        fig.savefig(out, dpi=160)
        assert out.exists()
        assert out.stat().st_size > 1000
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
