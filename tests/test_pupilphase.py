"""Real-time pupil-phase detector tests (ISC-37..39)."""

from __future__ import annotations

import numpy as np

from itrace.pupilphase import Phase, PhaseDetector


def _sine(n: int = 240, period: int = 60) -> np.ndarray:
    t = np.arange(n)
    return 3.0 + 0.5 * np.sin(2 * np.pi * t / period)


def test_labels_cover_all_phases() -> None:  # ISC-37
    phases = PhaseDetector().run(_sine().tolist())
    seen = set(phases)
    assert Phase.PEAK in seen
    assert Phase.TROUGH in seen
    assert Phase.DILATION in seen
    assert Phase.CONSTRICTION in seen


def test_peaks_align_with_sine_maxima() -> None:  # ISC-38
    period = 60
    signal = _sine(period=period)
    phases = PhaseDetector().run(signal.tolist())
    peak_idx = [i for i, p in enumerate(phases) if p is Phase.PEAK]
    analytic = period / 4  # first maximum of sin at phase pi/2
    # the causal detector flags a peak one sample after the maximum
    assert any(abs(i - analytic) <= 2 for i in peak_idx)


def test_detector_is_causal() -> None:  # ISC-39
    signal = _sine().tolist()
    full = PhaseDetector().run(signal)
    # online == offline prefix: feeding only the first k samples reproduces the
    # first k labels exactly -> no future sample influenced a past label.
    det = PhaseDetector()
    det.reset()
    online = [det.update(v) for v in signal[:50]]
    assert online == full[:50]


def test_min_delta_suppresses_jitter() -> None:
    det = PhaseDetector(min_delta=1.0)
    phases = det.run([0.0, 0.1, 0.2, 0.1, 0.0])  # all sub-threshold
    assert all(p in (Phase.UNKNOWN,) for p in phases)


def test_first_sample_unknown() -> None:
    det = PhaseDetector()
    assert det.update(1.0) is Phase.UNKNOWN
