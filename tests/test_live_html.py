"""Live HTML orchestrator tests without requiring webcam hardware."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from itrace.capture import CaptureSample, EyeBox, LiveFrameSample
from itrace.live import LiveState, create_app, live_message_from_frame
from itrace.types import GazeSample, PupilSample, PupilUnit


def _capture_sample(index: int, *, t: float | None = None, x: float = 0.0) -> CaptureSample:
    timestamp = index / 60.0 if t is None else t
    return CaptureSample(
        frame_index=index,
        timestamp_s=timestamp,
        gaze=GazeSample(t=timestamp, x=x, y=np.sin(timestamp) * 0.5),
        pupil=PupilSample(t=timestamp, size=0.22 + index * 0.001, unit=PupilUnit.RELATIVE),
        fps_estimate_hz=60.0,
        quality={"face_detected": 1.0, "pupil_proxy_relative": 1.0},
    )


def _live_frame(index: int, *, x: float = 0.0) -> LiveFrameSample:
    return LiveFrameSample(
        capture=_capture_sample(index, x=x),
        frame_width=640,
        frame_height=480,
        eye_box=EyeBox(x=160, y=120, width=320, height=120),
        eye_crop_jpeg="data:image/jpeg;base64,"
        + base64.b64encode(f"frame-{index}".encode()).decode("ascii"),
    )


def _no_face_frame(index: int) -> LiveFrameSample:
    timestamp = index / 30.0
    return LiveFrameSample(
        capture=CaptureSample(
            frame_index=index,
            timestamp_s=timestamp,
            gaze=GazeSample(t=timestamp, x=float("nan"), y=float("nan")),
            pupil=PupilSample(t=timestamp, size=float("nan"), unit=PupilUnit.RELATIVE),
            fps_estimate_hz=30.0,
            quality={"face_detected": 0.0},
        ),
        frame_width=640,
        frame_height=480,
        eye_box=EyeBox(x=90, y=96, width=460, height=164),
        eye_crop_jpeg="data:image/jpeg;base64,"
        + base64.b64encode(f"no-face-{index}".encode()).decode("ascii"),
    )


def _frame_source(_camera: int, max_frames: int | None) -> Iterable[LiveFrameSample]:
    xs = [0.0, 0.2, 8.0, 8.2]
    limit = len(xs) if max_frames is None else min(max_frames, len(xs))
    for index, x in enumerate(xs[:limit]):
        yield _live_frame(index, x=x)


CALIBRATION_TARGETS = [
    {"x": 10.0, "y": 0.0},
    {"x": 20.0, "y": 0.0},
    {"x": 10.0, "y": 1.0},
    {"x": 20.0, "y": 1.0},
]


def test_live_message_serializes_capture_frame_and_analysis() -> None:
    state = LiveState()
    for frame in _frame_source(0, None):
        state.append(frame.capture)
    message = live_message_from_frame(
        _live_frame(4, x=8.0),
        state,
        method="adaptive_ivt",
        velocity_threshold_deg_s=25.0,
        include_pso=True,
        rolling_window_s=5.0,
    )

    assert message["type"] == "sample"
    assert message["frame"]["eye_box"] == {"x": 160, "y": 120, "width": 320, "height": 120}
    assert str(message["frame"]["eye_crop_jpeg"]).startswith("data:image/jpeg;base64,")
    assert message["capture"]["quality"]["face_detected"] == 1.0
    assert message["method"]["name"] == "adaptive_ivt"
    assert message["analysis"]["n_samples"] == 4
    assert message["diagnostics"]["truth_boundary"] == "no live reference truth"
    assert message["diagnostics"]["sample_count"] == 4
    assert message["statistics"]["recording"]["sample_count"] == 4.0
    assert message["statistics"]["gaze"]["path_length_deg"] > 0.0
    assert message["statistics"]["events"]["saccade_rate_per_min"] >= 0.0
    assert len(message["series"]["t"]) == 4
    assert len(message["series"]["speed"]) == 4


def test_live_message_serializes_no_face_frames_without_analysis_crash() -> None:
    state = LiveState()
    for index in range(4):
        state.append(_no_face_frame(index).capture)

    message = live_message_from_frame(_no_face_frame(4), state)

    assert message["type"] == "sample"
    assert message["capture"]["quality"]["face_detected"] == 0.0
    assert message["analysis"]["n_samples"] == 4
    assert message["analysis"]["quality"]["finite_sample_fraction"] == 0.0
    assert "analysis_error" in message["analysis"]
    assert message["series"]["speed"] == [None, None, None, None]
    assert "low finite gaze fraction" in message["diagnostics"]["warnings"]


def test_live_state_clear_preserves_calibration_and_reset_clears_all() -> None:
    state = LiveState()
    for frame in _frame_source(0, None):
        state.append(frame.capture)
    calibration = state.fit_calibration(CALIBRATION_TARGETS)

    state.clear_samples()

    assert state.snapshot() == []
    assert state.calibration is calibration

    for frame in _frame_source(0, None):
        state.append(frame.capture)

    state.reset_all()

    assert state.snapshot() == []
    assert state.calibration is None


def test_live_app_serves_index_status_and_websocket_samples(tmp_path: Path) -> None:
    app = create_app(
        camera_index=2,
        output_dir=tmp_path,
        frame_source_factory=_frame_source,
    )
    client = TestClient(app)

    index = client.get("/")
    assert index.status_code == 200
    assert "iTrace Live" in index.text
    for element_id in (
        "clearSessionButton",
        "resetAllButton",
        "sampleStatus",
        "qualityStatus",
        "runSyntheticValidationButton",
        "domainValidationPlot",
        "eyeOverlay",
        "statistics-panel",
        "sampleRate",
        "pathEfficiency",
    ):
        assert f'id="{element_id}"' in index.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/sw.js").status_code == 200

    status = client.get("/api/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["camera_index"] == 2
    assert payload["persistence"] == "configured"
    assert payload["config"]["default_camera_index"] == 2
    assert payload["dependencies"]["fastapi"] is True
    assert "pipeline.analyze_session" in payload["methods"]
    assert "stats.descriptive.session_statistics" in payload["methods"]
    assert "validation.synthetic_validation_suite" in payload["methods"]

    empty_stats = client.get("/api/statistics/live")
    assert empty_stats.status_code == 200
    assert empty_stats.json()["statistics"]["recording"]["sample_count"] == 0.0

    with client.websocket_connect("/ws/live?camera=2&max_frames=1") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "sample"
        assert message["capture"]["frame_index"] == 0
        assert message["frame"]["eye_box"]["width"] == 320
        assert message["frame"]["eye_crop_jpeg"].startswith("data:image/jpeg;base64,")
        assert len(message["series"]["x"]) == 1
        assert "diagnostics" in message
        assert "statistics" in message

    live_stats = client.get("/api/statistics/live")
    assert live_stats.status_code == 200
    assert live_stats.json()["statistics"]["recording"]["sample_count"] == 1.0

    bad_stats = client.get("/api/statistics/live?window_s=0")
    assert bad_stats.status_code == 400

    synthetic_validation = client.get("/api/validation/synthetic?repetitions=1")
    assert synthetic_validation.status_code == 200
    assert synthetic_validation.json()["domain_count"] >= 4


def test_live_export_requires_configured_output_dir() -> None:
    app = create_app(frame_source_factory=_frame_source)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?max_frames=1") as websocket:
        assert websocket.receive_json()["type"] == "sample"

    response = client.post("/api/export")

    assert response.status_code == 400
    assert "No output directory" in response.json()["detail"]


def test_live_export_writes_csv_and_report(tmp_path: Path) -> None:
    app = create_app(output_dir=tmp_path, frame_source_factory=_frame_source)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?max_frames=4") as websocket:
        for _ in range(4):
            assert websocket.receive_json()["type"] == "sample"

    response = client.post("/api/export")

    assert response.status_code == 200
    paths = response.json()["paths"]
    for key in ("gaze_csv", "pupil_csv", "capture_records_csv", "report_json"):
        assert Path(paths[key]).exists()
    assert "timestamp_s" in Path(paths["capture_records_csv"]).read_text()


def test_live_export_short_capture_keeps_csv_and_fallback_report(tmp_path: Path) -> None:
    def short_frames(_camera: int, max_frames: int | None) -> Iterable[LiveFrameSample]:
        limit = min(max_frames or 8, 8)
        for index in range(limit):
            frame = _live_frame(index, x=float(index) * 0.1)
            yield LiveFrameSample(
                capture=CaptureSample(
                    frame_index=frame.capture.frame_index,
                    timestamp_s=frame.capture.timestamp_s,
                    gaze=frame.capture.gaze,
                    pupil=PupilSample(
                        t=frame.capture.timestamp_s,
                        size=float("nan"),
                        unit=PupilUnit.RELATIVE,
                    ),
                    fps_estimate_hz=frame.capture.fps_estimate_hz,
                    quality=frame.capture.quality,
                ),
                frame_width=frame.frame_width,
                frame_height=frame.frame_height,
                eye_box=frame.eye_box,
                eye_crop_jpeg=frame.eye_crop_jpeg,
            )

    app = create_app(output_dir=tmp_path, frame_source_factory=short_frames)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?max_frames=8") as websocket:
        for _ in range(8):
            message = websocket.receive_json()
            assert message["type"] == "sample"
    assert "analysis_error" in message["analysis"]

    response = client.post("/api/export")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert Path(paths["gaze_csv"]).exists()
    report = json.loads(Path(paths["report_json"]).read_text())
    assert report["n_samples"] == 8
    assert "analysis_error" in report


def test_live_clear_and_reset_endpoints_drive_export_and_calibration_state(
    tmp_path: Path,
) -> None:
    app = create_app(output_dir=tmp_path, frame_source_factory=_frame_source)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?max_frames=4") as websocket:
        for _ in range(4):
            assert websocket.receive_json()["type"] == "sample"

    fit = client.post("/api/calibration/fit", json={"targets": CALIBRATION_TARGETS})
    assert fit.status_code == 200

    clear = client.post("/api/session/clear")

    assert clear.status_code == 200
    assert clear.json() == {"ok": True, "sample_count": 0, "calibration_active": True}
    assert client.get("/api/status").json()["calibration"]["active"] is True

    empty_export = client.post("/api/export")
    assert empty_export.status_code == 409
    assert "No live samples" in empty_export.json()["detail"]

    with client.websocket_connect("/ws/live?max_frames=4") as websocket:
        for _ in range(4):
            assert websocket.receive_json()["type"] == "sample"

    refit = client.post("/api/calibration/fit", json={"targets": CALIBRATION_TARGETS})
    assert refit.status_code == 200

    reset = client.post("/api/session/reset")

    assert reset.status_code == 200
    assert reset.json() == {"ok": True, "sample_count": 0, "calibration_active": False}
    assert client.get("/api/status").json()["calibration"]["active"] is False

    with client.websocket_connect("/ws/live?max_frames=1") as websocket:
        message = websocket.receive_json()
    assert message["capture"]["calibrated_gaze"] is None
    assert message["calibration"]["active"] is False


def test_live_calibration_fit_streams_calibrated_gaze_and_exports(tmp_path: Path) -> None:
    def frames(_camera: int, max_frames: int | None) -> Iterable[LiveFrameSample]:
        xs = [0.0, 1.0, 0.0, 1.0]
        for index, x in enumerate(xs[: max_frames or len(xs)]):
            yield _live_frame(index, x=x)

    app = create_app(output_dir=tmp_path, frame_source_factory=frames, calibration_target_deg=15.0)
    client = TestClient(app)
    with client.websocket_connect("/ws/live?max_frames=4") as websocket:
        for _ in range(4):
            assert websocket.receive_json()["type"] == "sample"

    response = client.post(
        "/api/calibration/fit",
        json={"targets": CALIBRATION_TARGETS},
    )
    assert response.status_code == 200
    assert response.json()["calibration"]["n_points"] == 4

    with client.websocket_connect("/ws/live?max_frames=1") as websocket:
        message = websocket.receive_json()
    assert "calibrated_gaze" in message["capture"]
    assert message["calibration"]["active"] is True

    status = client.get("/api/status").json()
    assert status["calibration"]["active"] is True
    assert status["calibration"]["target_range_deg"] == 15.0

    export = client.post("/api/export").json()["paths"]
    assert Path(export["calibration_json"]).exists()
    assert Path(export["calibrated_gaze_csv"]).exists()
