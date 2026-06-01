"""Scanpath direction encoding and n-gram analysis.

Encodes a saccade sequence as a string of direction characters following the
Stuttgart scheme (Bulling et al. / eye-movements-as-biometrics line of work):
``R``/``L``/``U``/``D`` for right/left/up/down, **upper-case for long** saccades
and **lower-case for short** ones (relative to a length threshold). N-gram
statistics over the resulting string characterise habitual scan patterns.
"""

from __future__ import annotations

from collections import Counter

from .types import Saccade

# Direction sectors are 90deg-wide bins centred on the cardinal axes, in the
# gaze convention (0deg = right, 90deg = up).
_CARDINALS: tuple[tuple[float, str], ...] = (
    (0.0, "r"),
    (90.0, "u"),
    (180.0, "l"),
    (-90.0, "d"),
)


def _angular_distance(a: float, b: float) -> float:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def direction_char(direction_deg: float, amplitude_deg: float, long_threshold_deg: float) -> str:
    """Map one saccade's (direction, amplitude) to a single encoded character."""
    nearest = min(_CARDINALS, key=lambda c: _angular_distance(direction_deg, c[0]))
    ch = nearest[1]
    return ch.upper() if amplitude_deg >= long_threshold_deg else ch


def encode_directions(saccades: list[Saccade], long_threshold_deg: float = 5.0) -> str:
    """Encode a saccade list as a direction string (one char per saccade)."""
    return "".join(
        direction_char(s.direction_deg, s.amplitude_deg, long_threshold_deg) for s in saccades
    )


def ngram_counts(sequence: str, n: int = 2) -> dict[str, int]:
    """Count contiguous n-grams in an encoded direction string."""
    if n < 1:
        msg = "n must be >= 1"
        raise ValueError(msg)
    if len(sequence) < n:
        return {}
    grams = (sequence[i : i + n] for i in range(len(sequence) - n + 1))
    return dict(Counter(grams))
