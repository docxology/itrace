"""Dashboard figure-builder tests (ISC-50) - run without streamlit/plotly."""

from __future__ import annotations

import sys
from importlib.util import find_spec

import pytest

from itrace import dashboard, pipeline
from itrace.synthetic import gaze_with_saccade


def test_timeseries_figure_spec_shape() -> None:  # ISC-50
    gaze, _ = gaze_with_saccade()
    spec = dashboard.build_timeseries_figure(gaze)
    assert set(spec) == {"data", "layout"}
    assert len(spec["data"]) == 2
    assert len(spec["data"][0]["x"]) == len(gaze)


def test_direction_histogram_spec() -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=12.0)
    report = pipeline.analyze_gaze(gaze)
    spec = dashboard.build_direction_histogram(report, n_bins=8)
    assert spec["data"][0]["type"] == "barpolar"
    assert len(spec["data"][0]["r"]) == 8


def test_builders_do_not_import_streamlit() -> None:  # ISC-50
    gaze, _ = gaze_with_saccade()
    dashboard.build_timeseries_figure(gaze)
    assert "streamlit" not in sys.modules


def test_require_streamlit_raises_clear_error() -> None:  # ISC-50
    if find_spec("streamlit") is not None:
        assert dashboard._require_streamlit() is not None
    else:
        with pytest.raises(RuntimeError, match="uv sync --extra dashboard"):
            dashboard._require_streamlit()
