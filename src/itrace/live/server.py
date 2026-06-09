"""FastAPI server for the local HTML orchestrator."""

import asyncio
import webbrowser
from collections.abc import Callable, Iterable
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from .. import validation
from ..capture import LiveFrameSample, WebcamSource, native_stderr
from .analysis import analysis_payload, live_message_from_frame, method_name
from .state import LiveState

if TYPE_CHECKING:
    from fastapi import FastAPI

FrameSourceFactory = Callable[[int, int | None], Iterable[LiveFrameSample]]


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "web_static"


def _real_frame_source(
    camera_index: int,
    max_frames: int | None,
    *,
    backend_logs: bool,
) -> Iterable[LiveFrameSample]:
    with native_stderr(backend_logs):
        source = WebcamSource(camera_index=camera_index)
        yield from source.live_frames(max_frames=max_frames)


def create_app(
    *,
    camera_index: int = 0,
    output_dir: str | Path | None = None,
    empirical_manifest_path: str | Path | None = None,
    backend_logs: bool = False,
    frame_source_factory: FrameSourceFactory | None = None,
    calibration_target_deg: float = 15.0,
) -> "FastAPI":
    """Create the local FastAPI app for the HTML orchestrator."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        msg = (
            "The live HTML orchestrator needs the 'web' extra. Install it with:\n"
            "    uv sync --extra web\n"
            "(provides fastapi and uvicorn)."
        )
        raise RuntimeError(msg) from exc

    assets_dir = _assets_dir()
    state = LiveState(
        output_dir=Path(output_dir) if output_dir is not None else None,
        empirical_manifest_path=Path(empirical_manifest_path)
        if empirical_manifest_path is not None
        else None,
        calibration_target_range_deg=calibration_target_deg,
    )
    app = FastAPI(title="iTrace Live HTML", version="0.4.1")
    app.state.live_state = state
    app.mount("/static", StaticFiles(directory=assets_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(assets_dir / "index.html")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker() -> Response:
        return Response(
            "// iTrace Live intentionally registers no service worker.\n",
            media_type="text/javascript",
        )

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return {
            "ok": True,
            "camera_index": camera_index,
            "output_dir": str(state.output_dir) if state.output_dir is not None else None,
            "empirical_manifest_path": str(state.empirical_manifest_path)
            if state.empirical_manifest_path is not None
            else None,
            "persistence": "configured" if state.output_dir is not None else "memory",
            "sample_count": len(state.snapshot()),
            "config": {
                "default_camera_index": camera_index,
                "max_samples": state.max_samples,
                "backend_logs": backend_logs,
            },
            "calibration": {
                "active": state.calibration is not None,
                "target_range_deg": state.calibration_target_range_deg,
                "session_points": len(state.calibration_points),
            },
            "experiment": state.experiment_status(),
            "dependencies": {
                "fastapi": True,
                "uvicorn": find_spec("uvicorn") is not None,
                "cv2": find_spec("cv2") is not None,
                "mediapipe": find_spec("mediapipe") is not None,
            },
            "methods": [
                "capture.WebcamSource.live_frames",
                "capture.iris_landmarks_to_capture_sample",
                "pipeline.analyze_session",
                "saccades.detect_ivt",
                "detection.adaptive_ivt_threshold",
                "pupilphase.PhaseDetector",
                "stats.descriptive.session_statistics",
                "stats.scanpath_metrics.raw_gaze_spatial_summary",
                "validation.live_recording_diagnostics",
                "validation.synthetic_validation_suite",
                "experiments.default_eye_video_protocol",
                "experiments.experiment_report",
            ],
        }

    @app.get("/api/statistics/live")
    def live_statistics(
        method: str = "ivt",
        velocity_threshold: float = 30.0,
        include_pso: bool = False,
        window_s: float = 10.0,
    ) -> dict[str, object]:
        if window_s <= 0.0:
            raise HTTPException(status_code=400, detail="window_s must be positive")
        analysis = analysis_payload(
            state.recent(window_s),
            method=method_name(method),
            velocity_threshold_deg_s=velocity_threshold,
            include_pso=include_pso,
            calibration=state.calibration,
        )
        return {
            "ok": True,
            "sample_count": len(state.recent(window_s)),
            "analysis": analysis["report"],
            "statistics": analysis["statistics"],
            "diagnostics": analysis["diagnostics"],
        }

    @app.post("/api/export")
    def export() -> dict[str, object]:
        try:
            paths = state.export()
        except RuntimeError as exc:
            status_code = 400 if state.output_dir is None else 409
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"ok": True, "paths": paths}

    @app.post("/api/session/clear")
    def clear_session() -> dict[str, object]:
        state.clear_samples()
        return {
            "ok": True,
            "sample_count": len(state.snapshot()),
            "calibration_active": state.calibration is not None,
        }

    @app.post("/api/session/reset")
    def reset_session() -> dict[str, object]:
        state.reset_all()
        return {
            "ok": True,
            "sample_count": len(state.snapshot()),
            "calibration_active": state.calibration is not None,
        }

    @app.get("/api/validation/synthetic")
    def synthetic_validation(repetitions: int = 5, first_seed: int = 0) -> dict[str, object]:
        if repetitions < 1 or repetitions > 25:
            raise HTTPException(status_code=400, detail="repetitions must be between 1 and 25")
        return validation.synthetic_validation_suite(
            repetitions=repetitions,
            first_seed=first_seed,
        )

    @app.post("/api/calibration/fit")
    def fit_calibration(payload: dict[str, object]) -> dict[str, object]:
        raw_targets = payload.get("targets")
        if not isinstance(raw_targets, list):
            raise HTTPException(status_code=400, detail="targets must be a list")
        targets: list[dict[str, float]] = []
        for item in raw_targets:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                raise HTTPException(status_code=400, detail="each target needs x and y")
            targets.append({"x": float(item["x"]), "y": float(item["y"])})
        try:
            cal = state.fit_calibration(targets)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "calibration": cal.to_dict()}

    @app.post("/api/calibration/session/start")
    def start_calibration_session(payload: dict[str, object] | None = None) -> dict[str, object]:
        target_range = None
        if payload and payload.get("target_range_deg") is not None:
            raw_target_range = payload["target_range_deg"]
            if not isinstance(raw_target_range, str | int | float) or isinstance(
                raw_target_range, bool
            ):
                raise HTTPException(status_code=400, detail="target_range_deg must be numeric")
            target_range = float(raw_target_range)
        try:
            state.start_calibration_session(target_range)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "session_points": len(state.calibration_points),
            "target_range_deg": state.calibration_target_range_deg,
        }

    @app.post("/api/calibration/session/sample")
    def sample_calibration_target(payload: dict[str, object]) -> dict[str, object]:
        target_payload = payload.get("target")
        if isinstance(target_payload, dict):
            raw_x = target_payload.get("x")
            raw_y = target_payload.get("y")
        else:
            raw_x = payload.get("x")
            raw_y = payload.get("y")
        if raw_x is None or raw_y is None:
            raise HTTPException(status_code=400, detail="target needs x and y")
        if not isinstance(raw_x, str | int | float) or isinstance(raw_x, bool):
            raise HTTPException(status_code=400, detail="target x must be numeric")
        if not isinstance(raw_y, str | int | float) or isinstance(raw_y, bool):
            raise HTTPException(status_code=400, detail="target y must be numeric")
        raw_window = payload.get("window_s", 0.35)
        raw_min_samples = payload.get("min_samples", 1)
        if not isinstance(raw_window, str | int | float) or isinstance(raw_window, bool):
            raise HTTPException(status_code=400, detail="window_s must be numeric")
        if not isinstance(raw_min_samples, str | int | float) or isinstance(raw_min_samples, bool):
            raise HTTPException(status_code=400, detail="min_samples must be numeric")
        try:
            point = state.sample_calibration_target(
                target_x=float(raw_x),
                target_y=float(raw_y),
                window_s=float(raw_window),
                min_samples=int(raw_min_samples),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "point": point.to_dict(),
            "session_points": len(state.calibration_points),
        }

    @app.post("/api/calibration/session/fit")
    def fit_calibration_session() -> dict[str, object]:
        try:
            cal = state.fit_calibration_session()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "calibration": cal.to_dict(),
            "session_points": len(state.calibration_points),
        }

    @app.post("/api/calibration/session/reset")
    def reset_calibration_session() -> dict[str, object]:
        state.reset_calibration_session()
        return {"ok": True, "session_points": len(state.calibration_points)}

    @app.post("/api/calibration/reset")
    def reset_calibration() -> dict[str, object]:
        state.reset_calibration()
        return {"ok": True}

    @app.post("/api/experiment/session/start")
    def start_experiment_session(payload: dict[str, object] | None = None) -> dict[str, object]:
        trial_duration_s = 30.0
        target_range_deg = None
        metadata: dict[str, str] = {}
        if payload:
            if payload.get("trial_duration_s") is not None:
                raw_duration = payload["trial_duration_s"]
                if not isinstance(raw_duration, str | int | float) or isinstance(
                    raw_duration, bool
                ):
                    raise HTTPException(status_code=400, detail="trial_duration_s must be numeric")
                trial_duration_s = float(raw_duration)
            if payload.get("target_range_deg") is not None:
                raw_range = payload["target_range_deg"]
                if not isinstance(raw_range, str | int | float) or isinstance(raw_range, bool):
                    raise HTTPException(status_code=400, detail="target_range_deg must be numeric")
                target_range_deg = float(raw_range)
            for key in (
                "condition",
                "participant_id",
                "device_id",
                "session_group",
                "consent_scope",
                "reference_kind",
            ):
                raw_value = payload.get(key)
                if raw_value is None:
                    continue
                if not isinstance(raw_value, str) or not raw_value.strip():
                    raise HTTPException(status_code=400, detail=f"{key} must be a nonempty string")
                metadata[key] = raw_value.strip()
        try:
            protocol = state.start_experiment_session(
                trial_duration_s=trial_duration_s,
                target_range_deg=target_range_deg,
                **metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "protocol": protocol.to_dict(), "experiment": state.experiment_status()}

    @app.get("/api/experiment/session/status")
    def experiment_session_status() -> dict[str, object]:
        return {"ok": True, "experiment": state.experiment_status()}

    @app.post("/api/experiment/trial/start")
    def start_experiment_trial(payload: dict[str, object]) -> dict[str, object]:
        raw_trial_id = payload.get("trial_id")
        if not isinstance(raw_trial_id, str) or not raw_trial_id:
            raise HTTPException(status_code=400, detail="trial_id is required")
        try:
            trial = state.start_experiment_trial(raw_trial_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "trial": trial.to_dict(), "experiment": state.experiment_status()}

    @app.post("/api/experiment/trial/finish")
    def finish_experiment_trial(payload: dict[str, object] | None = None) -> dict[str, object]:
        raw_trial_id = payload.get("trial_id") if payload else None
        if raw_trial_id is not None and not isinstance(raw_trial_id, str):
            raise HTTPException(status_code=400, detail="trial_id must be a string")
        try:
            trial = state.finish_experiment_trial(raw_trial_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        result: dict[str, object] = {
            "ok": True,
            "trial": trial.to_dict(),
            "experiment": state.experiment_status(),
        }
        experiment = result["experiment"]
        if isinstance(experiment, dict) and experiment.get("export_ready"):
            try:
                report = state.experiment_report()
                paths = state.export_experiment()
            except RuntimeError as exc:
                result["auto_export"] = {"ok": False, "detail": str(exc)}
            else:
                result["report"] = report
                result["auto_export"] = {"ok": True, "paths": paths}
                result["experiment"] = state.experiment_status()
        return result

    @app.post("/api/experiment/session/report")
    def experiment_session_report() -> dict[str, object]:
        try:
            report = state.experiment_report()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "report": report}

    @app.post("/api/experiment/session/export")
    def export_experiment_session() -> dict[str, object]:
        try:
            report = state.experiment_report()
            paths = state.export_experiment()
        except RuntimeError as exc:
            status_code = 400 if state.output_dir is None else 409
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return {"ok": True, "paths": paths, "report": report}

    @app.post("/api/experiment/session/reset")
    def reset_experiment_session() -> dict[str, object]:
        state.reset_experiment()
        return {"ok": True, "experiment": state.experiment_status()}

    @app.websocket("/ws/live")
    async def live_socket(
        websocket: WebSocket,
        camera: int | None = None,
        method: str = "ivt",
        velocity_threshold: float = 30.0,
        include_pso: bool = False,
        window_s: float = 10.0,
        max_frames: int | None = None,
    ) -> None:
        await websocket.accept()
        selected_camera = camera_index if camera is None else camera
        factory = frame_source_factory or (
            lambda cam, limit: _real_frame_source(cam, limit, backend_logs=backend_logs)
        )
        try:
            for frame in factory(selected_camera, max_frames):
                state.append(frame.capture)
                await websocket.send_json(
                    live_message_from_frame(
                        frame,
                        state,
                        method=method,
                        velocity_threshold_deg_s=velocity_threshold,
                        include_pso=include_pso,
                        rolling_window_s=window_s,
                    )
                )
                await asyncio.sleep(0)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # pragma: no cover - live hardware path
            await websocket.send_json({"type": "error", "message": str(exc)})

    return app


def serve_live_html(
    *,
    camera_index: int = 0,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: str | Path | None = None,
    empirical_manifest_path: str | Path | None = None,
    backend_logs: bool = False,
    open_browser: bool = False,
) -> None:
    """Run the local live HTML server."""
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        msg = (
            "The live HTML orchestrator needs the 'web' extra. Install it with:\n"
            "    uv sync --extra web\n"
            "(provides fastapi and uvicorn)."
        )
        raise RuntimeError(msg) from exc
    app = create_app(
        camera_index=camera_index,
        output_dir=output_dir,
        empirical_manifest_path=empirical_manifest_path,
        backend_logs=backend_logs,
    )
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")
