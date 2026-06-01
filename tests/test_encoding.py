"""Scanpath-encoding tests (ISC-40..42)."""

from __future__ import annotations

from itrace import encoding
from itrace.types import Saccade


def _sacc(direction: float, amplitude: float) -> Saccade:
    return Saccade(0, 1, 0.0, 0.02, amplitude, direction, 300.0)


def test_encode_directions_chars_and_case() -> None:  # ISC-40
    saccs = [
        _sacc(0.0, 8.0),  # long right -> R
        _sacc(90.0, 2.0),  # short up   -> u
        _sacc(180.0, 9.0),  # long left  -> L
        _sacc(-90.0, 1.0),  # short down -> d
    ]
    assert encoding.encode_directions(saccs, long_threshold_deg=5.0) == "RuLd"


def test_rightward_then_upward() -> None:  # ISC-42
    saccs = [_sacc(2.0, 10.0), _sacc(88.0, 10.0)]
    assert encoding.encode_directions(saccs, long_threshold_deg=5.0) == "RU"


def test_ngram_counts() -> None:  # ISC-41
    counts = encoding.ngram_counts("RuLdRu", n=2)
    assert counts["Ru"] == 2
    assert counts["uL"] == 1
    assert sum(counts.values()) == 5  # 6 chars -> 5 bigrams


def test_ngram_edge_cases() -> None:
    assert encoding.ngram_counts("R", n=2) == {}
    assert encoding.ngram_counts("", n=1) == {}
    import pytest

    with pytest.raises(ValueError, match="n must be"):
        encoding.ngram_counts("RU", n=0)


def test_direction_char_nearest_sector() -> None:
    assert encoding.direction_char(44.0, 10.0, 5.0) == "R"  # closer to right
    assert encoding.direction_char(46.0, 10.0, 5.0) == "U"  # closer to up
