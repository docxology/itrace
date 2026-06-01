"""Live dashboard (optional ``dashboard`` extra: streamlit + plotly).

The Streamlit entry point is launched via ``itrace dashboard``. Streamlit and
Plotly are imported lazily so importing this module (and unit-testing the
figure builders) needs neither. :func:`build_timeseries_figure` returns a
Plotly figure spec as a plain dict when Plotly is absent, so the data-shaping
logic is testable headless.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import GazeStream, SessionReport


def build_timeseries_figure(stream: GazeStream) -> dict[str, object]:
    """Build a gaze x/y time-series figure spec (plain dict; no plotly needed).

    Returns a Plotly-compatible ``{"data": [...], "layout": {...}}`` dict that
    ``plotly.graph_objects.Figure(**spec)`` can consume directly.
    """
    return {
        "data": [
            {"x": stream.t.tolist(), "y": stream.x.tolist(), "name": "gaze x (deg)"},
            {"x": stream.t.tolist(), "y": stream.y.tolist(), "name": "gaze y (deg)"},
        ],
        "layout": {
            "title": "Gaze position over time",
            "xaxis": {"title": "time (s)"},
            "yaxis": {"title": "position (deg)"},
        },
    }


def build_direction_histogram(report: SessionReport, n_bins: int = 8) -> dict[str, object]:
    """Polar histogram spec of saccade directions (plain dict)."""
    directions = np.array([s.direction_deg for s in report.saccades], dtype=np.float64)
    edges = np.linspace(-180.0, 180.0, n_bins + 1)
    counts, _ = np.histogram(directions, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "data": [{"r": counts.tolist(), "theta": centers.tolist(), "type": "barpolar"}],
        "layout": {"title": "Saccade direction distribution"},
    }


def run_app() -> None:  # pragma: no cover - launches the interactive server
    """Streamlit entry point (requires the ``dashboard`` extra)."""
    st = _require_streamlit()
    st.title("iTrace - live eye-movement dashboard")
    st.write(
        "Install the capture extra and connect a webcam, or load a CSV to "
        "analyse a recorded session."
    )


def _require_streamlit() -> Any:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        msg = (
            "The live dashboard needs the 'dashboard' extra. Install it with:\n"
            "    uv sync --extra dashboard\n"
            "(provides streamlit and plotly)."
        )
        raise RuntimeError(msg) from exc
    return st
