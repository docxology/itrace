"""Real-time pupil-phase detection (rtPupilPhase-style, causal).

Detects, sample by sample, whether the pupil is dilating or constricting and
flags local peaks (dilation maxima) and troughs (constriction minima) using
only past samples -- so the detector is suitable for closed-loop, online use
(Kronemer et al., 2024). No future sample is ever consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    """Instantaneous pupil-phase label."""

    DILATION = "dilation"
    CONSTRICTION = "constriction"
    PEAK = "peak"
    TROUGH = "trough"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PhaseDetector:
    """Causal streaming pupil-phase classifier.

    Feed samples in order with :meth:`update`; each call returns the
    :class:`Phase` for that sample based solely on the current and the two most
    recent prior samples. A peak is the apex between a rising and a falling
    sample; a trough is the nadir between a falling and a rising sample.

    ``min_delta`` suppresses labels from sub-threshold jitter (samples whose
    change is below it are treated as continuing the previous trend).
    """

    min_delta: float = 0.0
    _prev: float | None = None
    _prev_dir: int = 0  # +1 rising, -1 falling, 0 flat/unknown

    def reset(self) -> None:
        self._prev = None
        self._prev_dir = 0

    def update(self, value: float) -> Phase:
        if self._prev is None:
            self._prev = value
            return Phase.UNKNOWN

        delta = value - self._prev
        if abs(delta) <= self.min_delta:  # noqa: SIM108 - clarity over nested ternary
            direction = self._prev_dir
        else:
            direction = 1 if delta > 0 else -1

        phase = Phase.UNKNOWN
        if direction == 1:
            phase = Phase.TROUGH if self._prev_dir == -1 else Phase.DILATION
        elif direction == -1:
            phase = Phase.PEAK if self._prev_dir == 1 else Phase.CONSTRICTION

        self._prev = value
        if direction != 0:
            self._prev_dir = direction
        return phase

    def run(self, values: list[float] | tuple[float, ...]) -> list[Phase]:
        """Convenience: stream a whole sequence, returning one label per sample."""
        self.reset()
        return [self.update(float(v)) for v in values]
