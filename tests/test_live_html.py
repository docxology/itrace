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


def _record_complete_experiment(
    client: TestClient,
    *,
    start_payload: dict[str, object] | None = None,
) -> dict[str, str]:
    payload = {"trial_duration_s": 5.0, "target_range_deg": 10.0}
    if start_payload:
        payload.update(start_payload)
    start = client.post("/api/experiment/session/start", json=payload)
    assert start.status_code == 200
    for trial_id in ("fixed_center", "reading", "corner_saccades"):
        with client.websocket_connect("/ws/live?max_frames=2") as websocket:
            for _ in range(2):
                assert websocket.receive_json()["type"] == "sample"
        trial_start = client.post(
            "/api/experiment/trial/start",
            json={"trial_id": trial_id},
        )
        assert trial_start.status_code == 200
        with client.websocket_connect("/ws/live?max_frames=4") as websocket:
            for _ in range(4):
                assert websocket.receive_json()["type"] == "sample"
        trial_finish = client.post("/api/experiment/trial/finish")
        assert trial_finish.status_code == 200
    auto_export = trial_finish.json().get("auto_export")
    assert auto_export is not None
    assert auto_export["ok"] is True
    return auto_export["paths"]


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
    assert message["experiment"]["active"] is False
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
        "startExperimentButton",
        "experimentSampleCount",
        "experimentCompletedNames",
        "experimentExportReady",
        "experimentSessionId",
        "experimentManifestPath",
        "experimentConditionInput",
        "experimentParticipantInput",
        "experimentDeviceInput",
        "experimentSessionGroupInput",
        "experimentReferenceKindInput",
        "experimentReportPath",
        "experimentTrialSelect",
        "experimentTrialTable",
        "experimentStage",
        "experimentStepper",
        "experimentTimerValue",
        "experimentTimerBar",
        "experimentNextActionButton",
        "exportExperimentButton",
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
    assert "experiments.experiment_report" in payload["methods"]
    assert payload["experiment"]["active"] is False
    assert payload["experiment"]["next_session_id"] == "local_pilot_001"
    assert payload["experiment"]["next_replicate_id"] == "R001"
    assert payload["experiment"]["metadata"]["condition"] == "indoor_office_daylight"

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


def test_live_calibration_session_samples_targets_and_fits(tmp_path: Path) -> None:
    raw_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    cursor = {"index": 0}

    def frame(index: int, x: float, y: float) -> LiveFrameSample:
        timestamp = index / 30.0
        return LiveFrameSample(
            capture=CaptureSample(
                frame_index=index,
                timestamp_s=timestamp,
                gaze=GazeSample(t=timestamp, x=x, y=y),
                pupil=PupilSample(t=timestamp, size=0.25, unit=PupilUnit.RELATIVE),
                fps_estimate_hz=30.0,
                quality={"face_detected": 1.0},
            ),
            frame_width=640,
            frame_height=480,
            eye_box=EyeBox(x=160, y=120, width=320, height=120),
            eye_crop_jpeg="data:image/jpeg;base64,"
            + base64.b64encode(f"cal-{index}".encode()).decode("ascii"),
        )

    def frames(_camera: int, max_frames: int | None) -> Iterable[LiveFrameSample]:
        limit = max_frames or 1
        for _ in range(limit):
            index = cursor["index"]
            cursor["index"] += 1
            x, y = raw_points[index % len(raw_points)]
            yield frame(index, x, y)

    app = create_app(output_dir=tmp_path, frame_source_factory=frames)
    client = TestClient(app)

    start = client.post("/api/calibration/session/start", json={"target_range_deg": 12.0})
    assert start.status_code == 200
    assert start.json()["target_range_deg"] == 12.0

    for target in CALIBRATION_TARGETS:
        with client.websocket_connect("/ws/live?max_frames=1") as websocket:
            assert websocket.receive_json()["type"] == "sample"
        sample = client.post(
            "/api/calibration/session/sample",
            json={"target": target, "window_s": 0.001, "min_samples": 1},
        )
        assert sample.status_code == 200
        assert sample.json()["session_points"] >= 1

    fit = client.post("/api/calibration/session/fit")
    assert fit.status_code == 200
    assert fit.json()["calibration"]["n_points"] == 4

    status = client.get("/api/status").json()
    assert status["calibration"]["active"] is True
    assert status["calibration"]["session_points"] == 4

    reset_session = client.post("/api/calibration/session/reset")
    assert reset_session.status_code == 200
    assert reset_session.json()["session_points"] == 0
    assert client.get("/api/status").json()["calibration"]["active"] is True

    reset_cal = client.post("/api/calibration/reset")
    assert reset_cal.status_code == 200
    status = client.get("/api/status").json()
    assert status["calibration"]["active"] is False
    assert status["calibration"]["session_points"] == 0


def test_live_experiment_session_reports_and_exports(tmp_path: Path) -> None:
    app = create_app(output_dir=tmp_path, frame_source_factory=_frame_source)
    client = TestClient(app)

    start = client.post(
        "/api/experiment/session/start",
        json={"trial_duration_s": 5.0, "target_range_deg": 10.0},
    )
    assert start.status_code == 200
    assert start.json()["experiment"]["active"] is True
    assert start.json()["protocol"]["trials"][0]["trial_id"] == "fixed_center"
    start_status = start.json()["experiment"]
    assert start_status["required_trial_count"] == 3
    assert start_status["all_trials_completed"] is False
    assert start_status["next_trial_id"] == "fixed_center"
    assert start_status["missing_trial_ids"] == [
        "fixed_center",
        "reading",
        "corner_saccades",
    ]
    assert start_status["export_ready"] is False
    assert [trial["status"] for trial in start_status["trial_statuses"]] == [
        "pending",
        "pending",
        "pending",
    ]

    with client.websocket_connect("/ws/live?max_frames=2") as websocket:
        for _ in range(2):
            assert websocket.receive_json()["type"] == "sample"

    trial_start = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "fixed_center"},
    )
    assert trial_start.status_code == 200
    active_status = trial_start.json()["experiment"]
    assert active_status["active_trial"]["trial_id"] == "fixed_center"
    assert active_status["active_trial_id"] == "fixed_center"
    assert active_status["trial_duration_s"] == 5.0
    assert active_status["trial_elapsed_s"] == 0.0
    assert active_status["trial_remaining_s"] == 5.0
    assert active_status["trial_progress"] == 0.0
    assert active_status["current_target"]["label"] == "center"

    with client.websocket_connect("/ws/live?max_frames=4") as websocket:
        for _ in range(4):
            message = websocket.receive_json()
            assert message["type"] == "sample"
    live_experiment = message["experiment"]
    assert live_experiment["active_trial_id"] == "fixed_center"
    assert live_experiment["trial_elapsed_s"] > 0.0
    assert live_experiment["trial_remaining_s"] < 5.0
    assert 0.0 < live_experiment["trial_progress"] < 1.0

    trial_finish = client.post("/api/experiment/trial/finish")
    assert trial_finish.status_code == 200
    finish_status = trial_finish.json()["experiment"]
    assert finish_status["completed_trial_count"] == 1
    assert finish_status["completed_trial_ids"] == ["fixed_center"]
    assert finish_status["trial_statuses"][0]["status"] == "complete"
    assert finish_status["trial_statuses"][0]["sample_count"] >= 4
    assert finish_status["trial_statuses"][0]["finite_gaze_fraction"] == 1.0
    assert finish_status["export_ready"] is False
    assert finish_status["all_trials_completed"] is False
    assert finish_status["missing_trial_ids"] == ["reading", "corner_saccades"]
    assert "complete: reading, corner_saccades" in finish_status["export_blockers"]

    report = client.post("/api/experiment/session/report")
    assert report.status_code == 200
    payload = report.json()["report"]
    assert payload["kind"] == "derived_eye_video_experiment"
    assert payload["completed_trial_count"] == 1
    assert payload["storage_boundary"].startswith("derived gaze")

    incomplete_export = client.post("/api/experiment/session/export")
    assert incomplete_export.status_code == 409
    assert "complete all experiment trials" in incomplete_export.json()["detail"]

    for trial_id in ("reading", "corner_saccades"):
        trial_start = client.post(
            "/api/experiment/trial/start",
            json={"trial_id": trial_id},
        )
        assert trial_start.status_code == 200
        with client.websocket_connect("/ws/live?max_frames=4") as websocket:
            for _ in range(4):
                assert websocket.receive_json()["type"] == "sample"
        trial_finish = client.post("/api/experiment/trial/finish")
        assert trial_finish.status_code == 200

    complete_status = client.get("/api/experiment/session/status").json()["experiment"]
    assert complete_status["all_trials_completed"] is True
    assert complete_status["export_ready"] is True
    assert complete_status["missing_trial_ids"] == []
    auto_export = trial_finish.json()["auto_export"]
    assert auto_export["ok"] is True
    assert trial_finish.json()["report"]["kind"] == "derived_eye_video_experiment"
    assert trial_finish.json()["report"]["completed_trial_count"] == 3
    auto_paths = auto_export["paths"]
    assert Path(auto_paths["manifest_json"]).exists()
    assert Path(auto_paths["report_json"]).exists()

    export = client.post("/api/experiment/session/export")
    assert export.status_code == 200
    assert export.json()["report"]["completed_trial_count"] == 3
    paths = export.json()["paths"]
    assert Path(paths["manifest_json"]).exists()
    assert Path(paths["report_json"]).exists()

    duplicate_start = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "fixed_center"},
    )
    assert duplicate_start.status_code == 409
    assert "already been completed" in duplicate_start.json()["detail"]

    reset = client.post("/api/experiment/session/reset")
    assert reset.status_code == 200
    assert reset.json()["experiment"]["active"] is False


def test_live_experiment_exports_sequential_session_dirs_and_manifest_entries(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output" / "empirical_pilot"
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    app = create_app(
        output_dir=output_root,
        empirical_manifest_path=manifest_path,
        frame_source_factory=_frame_source,
    )
    client = TestClient(app)

    first = _record_complete_experiment(
        client,
        start_payload={
            "condition": "indoor_office_daylight",
            "participant_id": "P007",
            "device_id": "webcam_a",
            "session_group": "P007_webcam_a",
            "reference_kind": "manual_annotation",
        },
    )
    assert first["session_id"] == "local_pilot_001"
    assert first["replicate_id"] == "R001"
    assert Path(first["report_json"]).parent == output_root / "local_pilot_001" / "experiment"
    assert Path(first["report_json"]).exists()
    assert Path(first["empirical_summary_json"]).exists()

    reset = client.post("/api/experiment/session/reset")
    assert reset.status_code == 200
    second = _record_complete_experiment(
        client,
        start_payload={"condition": "indoor_office_dim", "reference_kind": "none"},
    )
    assert second["session_id"] == "local_pilot_002"
    assert second["replicate_id"] == "R002"
    assert Path(second["report_json"]).parent == output_root / "local_pilot_002" / "experiment"
    assert Path(second["report_json"]).exists()
    assert Path(second["empirical_summary_json"]).exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = manifest["sessions"]
    assert [session["session_id"] for session in sessions] == [
        "local_pilot_001",
        "local_pilot_002",
    ]
    assert [session["replicate_id"] for session in sessions] == ["R001", "R002"]
    assert [session["condition"] for session in sessions] == [
        "indoor_office_daylight",
        "indoor_office_dim",
    ]
    assert sessions[0]["participant_id"] == "P007"
    assert sessions[0]["device_id"] == "webcam_a"
    assert sessions[0]["session_group"] == "P007_webcam_a"
    assert sessions[0]["reference_kind"] == "manual_annotation"
    assert sessions[1]["reference_kind"] == "none"
    assert sessions[0]["report"] == (
        "output/empirical_pilot/local_pilot_001/experiment/experiment_report.json"
    )
    assert sessions[1]["report"] == (
        "output/empirical_pilot/local_pilot_002/experiment/experiment_report.json"
    )
    summary = json.loads(Path(second["empirical_summary_json"]).read_text(encoding="utf-8"))
    assert summary["available_session_count"] == 2
    assert summary["replicate_count"] == 2
    assert summary["condition_count"] == 2
    assert summary["reference_candidate_count"] == 1
    assert summary["reference_evidence_count"] == 0
    assert summary["reference_evidence_issues"][0]["session_id"] == "local_pilot_001"
    assert (
        summary["v1_readiness"]["replicate_plan"][
            "minimum_additional_sessions_to_meet_count_criteria"
        ]
        == 3
    )


def test_live_experiment_numbered_leaf_output_dir_allocates_next_sibling(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output" / "empirical_pilot"
    existing = output_root / "local_pilot_002" / "experiment"
    existing.mkdir(parents=True)
    (existing / "experiment_report.json").write_text("{}", encoding="utf-8")
    app = create_app(
        output_dir=output_root / "local_pilot_002",
        frame_source_factory=_frame_source,
    )
    client = TestClient(app)

    paths = _record_complete_experiment(client)

    assert paths["session_id"] == "local_pilot_003"
    assert paths["replicate_id"] == "R003"
    assert Path(paths["report_json"]).parent == output_root / "local_pilot_003" / "experiment"
    assert (existing / "experiment_report.json").read_text(encoding="utf-8") == "{}"


def test_live_experiment_preview_uses_v1_condition_schedule(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "empirical_pilot"
    for session_id in ("local_pilot_001", "local_pilot_002"):
        existing = output_root / session_id / "experiment"
        existing.mkdir(parents=True)
        (existing / "experiment_report.json").write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    app = create_app(
        output_dir=output_root,
        empirical_manifest_path=manifest_path,
        frame_source_factory=_frame_source,
    )
    client = TestClient(app)

    status = client.get("/api/status")
    assert status.status_code == 200
    experiment = status.json()["experiment"]
    assert experiment["next_session_id"] == "local_pilot_003"
    assert experiment["next_replicate_id"] == "R003"
    assert experiment["metadata"]["condition"] == "indoor_office_dim"

    paths = _record_complete_experiment(client)

    assert paths["session_id"] == "local_pilot_003"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["condition"] == "indoor_office_dim"


def test_live_experiment_rejects_invalid_ordering() -> None:
    app = create_app(frame_source_factory=_frame_source)
    client = TestClient(app)

    early_trial = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "fixed_center"},
    )
    assert early_trial.status_code == 409
    assert "not been started" in early_trial.json()["detail"]

    start = client.post("/api/experiment/session/start")
    assert start.status_code == 200

    out_of_order = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "reading"},
    )
    assert out_of_order.status_code == 409
    assert "start next pending experiment trial first" in out_of_order.json()["detail"]

    early_report = client.post("/api/experiment/session/report")
    assert early_report.status_code == 200
    assert early_report.json()["report"]["completed_trial_count"] == 0


def test_live_experiment_rejects_bad_payloads_active_trial_and_export(
    tmp_path: Path,
) -> None:
    app = create_app(output_dir=tmp_path, frame_source_factory=_frame_source)
    client = TestClient(app)

    bad_duration = client.post(
        "/api/experiment/session/start",
        json={"trial_duration_s": True},
    )
    assert bad_duration.status_code == 400
    assert "trial_duration_s" in bad_duration.json()["detail"]

    bad_range = client.post(
        "/api/experiment/session/start",
        json={"target_range_deg": True},
    )
    assert bad_range.status_code == 400
    assert "target_range_deg" in bad_range.json()["detail"]

    bad_reference = client.post(
        "/api/experiment/session/start",
        json={"reference_kind": "spreadsheet_guess"},
    )
    assert bad_reference.status_code == 400
    assert "reference_kind" in bad_reference.json()["detail"]

    not_started_export = client.post("/api/experiment/session/export")
    assert not_started_export.status_code == 409
    assert "not been started" in not_started_export.json()["detail"]

    start = client.post(
        "/api/experiment/session/start",
        json={"trial_duration_s": 5.0, "target_range_deg": 10.0},
    )
    assert start.status_code == 200

    empty_export = client.post("/api/experiment/session/export")
    assert empty_export.status_code == 409
    assert "no experiment trials" in empty_export.json()["detail"]

    missing_trial_id = client.post("/api/experiment/trial/start", json={})
    assert missing_trial_id.status_code == 400

    with client.websocket_connect("/ws/live?max_frames=1") as websocket:
        assert websocket.receive_json()["type"] == "sample"

    trial = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "fixed_center"},
    )
    assert trial.status_code == 200

    duplicate_trial = client.post(
        "/api/experiment/trial/start",
        json={"trial_id": "reading"},
    )
    assert duplicate_trial.status_code == 409
    assert "already active" in duplicate_trial.json()["detail"]

    bad_finish_payload = client.post("/api/experiment/trial/finish", json={"trial_id": 1})
    assert bad_finish_payload.status_code == 400

    wrong_finish_trial = client.post(
        "/api/experiment/trial/finish",
        json={"trial_id": "reading"},
    )
    assert wrong_finish_trial.status_code == 409
    assert "does not match" in wrong_finish_trial.json()["detail"]

    active_report = client.post("/api/experiment/session/report")
    assert active_report.status_code == 409
    assert "active experiment trial" in active_report.json()["detail"]

    active_export = client.post("/api/experiment/session/export")
    assert active_export.status_code == 409
    assert "active experiment trial" in active_export.json()["detail"]

    no_output_app = create_app(frame_source_factory=_frame_source)
    no_output_client = TestClient(no_output_app)
    assert no_output_client.post("/api/experiment/session/start").status_code == 200
    no_output_export = no_output_client.post("/api/experiment/session/export")
    assert no_output_export.status_code == 400
    assert "No output directory" in no_output_export.json()["detail"]


def test_live_experiment_samples_survive_rolling_sample_limit() -> None:
    state = LiveState(max_samples=2)
    state.start_experiment_session(trial_duration_s=1.0, target_range_deg=10.0)
    state.append(_capture_sample(0, t=0.0))
    state.start_experiment_trial("fixed_center")
    for index in range(1, 8):
        state.append(_capture_sample(index, t=float(index) * 0.1))
    state.finish_experiment_trial()

    status = state.experiment_status()

    assert len(state.snapshot()) == 2
    assert status["experiment_sample_count"] == 8
    assert status["trial_statuses"][0]["sample_count"] == 8
    assert status["trial_statuses"][0]["finite_gaze_fraction"] == 1.0
