"""Local HTML orchestrator for live iTrace webcam analysis.

This package intentionally imports no web or hardware dependencies at module
load. FastAPI/uvicorn and OpenCV/MediaPipe are imported only inside server
entry points so ``import itrace.live`` remains safe on headless machines.
"""

from __future__ import annotations

from .analysis import analysis_payload, live_message_from_frame, method_name
from .server import create_app, serve_live_html
from .state import CalibrationSessionPoint, LiveState

__all__ = [
    "CalibrationSessionPoint",
    "LiveState",
    "analysis_payload",
    "create_app",
    "live_message_from_frame",
    "method_name",
    "serve_live_html",
]
