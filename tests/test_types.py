"""Data-model tests (ISC-7..10)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace.types import (
    PSO,
    EventType,
    Fixation,
    GazeSample,
    GazeStream,
    Microsaccade,
    PupilSample,
    PupilStream,
    PupilUnit,
    Saccade,
)


def test_dataclasses_carry_expected_fields() -> None:  # ISC-7
    g = GazeSample(t=0.0, x=1.0, y=2.0)
    assert (g.t, g.x, g.y) == (0.0, 1.0, 2.0)
    p = PupilSample(t=0.0, size=3.0, unit=PupilUnit.MM)
    assert p.unit is PupilUnit.MM
    s = Saccade(0, 5, 0.0, 0.02, 10.0, 0.0, 400.0)
    assert s.duration_s == pytest.approx(0.02)
    m = Microsaccade(0, 3, 0.5, 30.0, 0.0)
    assert m.amplitude_deg == 0.5
    pso = PSO(4, 7, 0.2, 0.23, 80.0, 0)
    assert pso.duration_s == pytest.approx(0.03)
    f = Fixation(0, 9, 0.0, 0.1, 0.0, 0.0)
    assert f.duration_s == pytest.approx(0.1)


def test_event_type_enum_complete() -> None:  # ISC-8
    values = {e.value for e in EventType}
    assert {"fixation", "saccade", "pso", "smooth_pursuit", "blink"} <= values


def test_gazestream_validates_and_infers_rate() -> None:  # ISC-9
    t = np.linspace(0.0, 1.0, 251)
    s = GazeStream(t=t, x=np.zeros(251), y=np.zeros(251))
    assert len(s) == 251
    assert s.sampling_rate_hz == pytest.approx(250.0, rel=1e-3)


def test_gazestream_rejects_mismatched_lengths() -> None:  # ISC-10
    with pytest.raises(ValueError, match="equal-length"):
        GazeStream(t=np.zeros(3), x=np.zeros(4), y=np.zeros(3))


def test_pupilstream_rejects_mismatched_lengths() -> None:  # ISC-10
    with pytest.raises(ValueError, match="equal-length"):
        PupilStream(t=np.zeros(3), size=np.zeros(4))


def test_streams_reject_2d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        GazeStream(t=np.zeros((2, 2)), x=np.zeros((2, 2)), y=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="1-D"):
        PupilStream(t=np.zeros((2, 2)), size=np.zeros((2, 2)))
