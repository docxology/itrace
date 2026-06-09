"""Tests for statistical evidence ledger visualizations."""

from __future__ import annotations

from itertools import pairwise

import matplotlib.image as mpimg

from itrace.stats.evidence import load_statistical_interpretation_ledger
from itrace.viz.evidence import figure_statistical_interpretation_ledger


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


def test_statistical_interpretation_ledger_figure_is_nonblank(tmp_path) -> None:
    payload = load_statistical_interpretation_ledger()
    fig = figure_statistical_interpretation_ledger(payload)
    path = tmp_path / "statistical_interpretation_ledger.png"
    fig.savefig(path, dpi=160)

    assert path.exists()
    assert path.stat().st_size > 10_000
    image = mpimg.imread(path)
    assert image.shape[1] >= 1200
    assert image.shape[0] >= 650
    assert image.std() > 0.005
    encoded = str(payload).lower()
    assert "does_not_prove" in encoded
    assert "device validation" in encoded
    assert "population physiology" in encoded
    _assert_text_columns_do_not_overlap(
        fig,
        (
            "ledger-statistic-",
            "ledger-value-",
            "ledger-role-",
            "ledger-boundary-",
        ),
    )
