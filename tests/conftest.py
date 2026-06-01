"""Shared pytest fixtures and path setup for the iTrace suite.

No mocks anywhere: every test runs real package code over real arrays whose
ground truth is known by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make tests/fixtures/ importable for the independent reference oracle.
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from itrace.synthetic import (
    gaze_with_microsaccade,
    gaze_with_saccade,
    pupil_sine_with_blink,
)
from itrace.types import GazeStream


@pytest.fixture
def saccade_stream() -> GazeStream:
    stream, _ = gaze_with_saccade(amplitude_deg=10.0, direction_deg=0.0)
    return stream


@pytest.fixture
def saccade_stream_and_truth():
    return gaze_with_saccade(amplitude_deg=10.0, direction_deg=0.0)


@pytest.fixture
def microsaccade_stream_and_onset():
    return gaze_with_microsaccade()


@pytest.fixture
def pupil_stream_and_peaks():
    return pupil_sine_with_blink()


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Prevent figure leakage across visualization tests."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    plt.close("all")
