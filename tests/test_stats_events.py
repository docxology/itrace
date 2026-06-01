"""Event-level recovery metrics."""

from __future__ import annotations

import pytest

from itrace.stats.events import event_prf, interval_overlap_s
from itrace.types import Saccade


def _sacc(onset: float, offset: float) -> Saccade:
    return Saccade(0, 1, onset, offset, 5.0, 0.0, 100.0)


def test_interval_overlap_known_value() -> None:
    assert interval_overlap_s(0.0, 1.0, 0.5, 1.5) == pytest.approx(0.5)
    assert interval_overlap_s(0.0, 0.2, 0.3, 0.5) == 0.0


def test_event_prf_matches_overlapping_events() -> None:
    truth = [_sacc(0.0, 0.1), _sacc(1.0, 1.1)]
    pred = [_sacc(0.02, 0.12), _sacc(2.0, 2.1)]
    out = event_prf(truth, pred)
    assert out["true_positive"] == 1.0
    assert out["false_positive"] == 1.0
    assert out["false_negative"] == 1.0
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(0.5)


def test_event_prf_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="min_overlap_s"):
        event_prf([], [], min_overlap_s=-0.1)
