"""Local HTML orchestrator for live iTrace webcam analysis.

This module intentionally imports no web or hardware dependencies at module
load. FastAPI/uvicorn and OpenCV/MediaPipe are imported only inside the server
entry points so ``import itrace`` remains safe on headless machines.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import webbrowser
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import numpy as np

from . import io, pipeline, saccades, validation
from .calibration import AffineCalibration
from .capture import CaptureSample, LiveFrameSample, WebcamSource
from .config import AnalysisConfig, DetectionConfig
from .stats.descriptive import session_statistics
from .types import GazeStream, PupilStream, PupilUnit

DetectionMethodName = Literal["ivt", "adaptive_ivt"]
FrameSourceFactory = Callable[[int, int | None], Iterable[LiveFrameSample]]


@contextmanager
def _native_stderr(backend_logs: bool) -> Iterator[None]:
    """Optionally silence noisy native OpenCV/MediaPipe stderr diagnostics."""
    if backend_logs:
        yield
        return
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


@dataclass(slots=True)
class LiveState:
    """In-memory rolling capture state for one local HTML session."""

    output_dir: Path | None = None
    max_samples: int = 5000
    samples: list[CaptureSample] = field(default_factory=list)
    calibration: AffineCalibration | None = None
    calibration_target_range_deg: float = 15.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def append(self, sample: CaptureSample) -> None:
        """Append a capture sample, retaining only the latest ``max_samples``."""
        with self._lock:
            if self.samples and sample.timestamp_s <= self.samples[-1].timestamp_s:
                timestamp_s = self.samples[-1].timestamp_s + 1e-6
                sample = replace(
                    sample,
                    timestamp_s=timestamp_s,
                    gaze=replace(sample.gaze, t=timestamp_s),
                    pupil=replace(sample.pupil, t=timestamp_s)
                    if sample.pupil is not None
                    else None,
                )
            self.samples.append(sample)
            if len(self.samples) > self.max_samples:
                del self.samples[: len(self.samples) - self.max_samples]

    def snapshot(self) -> list[CaptureSample]:
        """Return a stable copy of the capture samples."""
        with self._lock:
            return list(self.samples)

    def recent(self, window_s: float) -> list[CaptureSample]:
        """Return samples inside the latest rolling time window."""
        samples = self.snapshot()
        if not samples or window_s <= 0.0:
            return samples
        cutoff = samples[-1].timestamp_s - window_s
        return [sample for sample in samples if sample.timestamp_s >= cutoff]

    def clear_samples(self) -> None:
        """Clear captured samples while preserving calibration and configuration."""
        with self._lock:
            self.samples.clear()

    def reset_all(self) -> None:
        """Clear captured samples and calibration for a fresh live session."""
        with self._lock:
            self.samples.clear()
            self.calibration = None

    def export(self) -> dict[str, str]:
        """Write CSV/JSON artifacts for the current session."""
        if self.output_dir is None:
            msg = "No output directory configured for this live session"
            raise RuntimeError(msg)
        samples = self.snapshot()
        if not samples:
            msg = "No live samples have been captured yet"
            raise RuntimeError(msg)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        gaze, pupil = _samples_to_streams(samples)

        gaze_path = self.output_dir / "live_gaze.csv"
        pupil_path = self.output_dir / "live_pupil.csv"
        records_path = self.output_dir / "live_capture_records.csv"
        report_path = self.output_dir / "live_report.json"

        io.write_gaze_csv(gaze, gaze_path)
        io.write_pupil_csv(pupil, pupil_path)
        _write_capture_records_csv(samples, records_path)
        report_payload: dict[str, object]
        if len(samples) >= 3:
            try:
                report_payload = pipeline.analyze_session(gaze, pupil).to_dict()
            except ValueError as exc:
                report_payload = {
                    "n_samples": len(samples),
                    "duration_s": float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0,
                    "analysis_error": str(exc),
                }
        else:
            report_payload = {"n_samples": len(samples)}
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        paths = {
            "gaze_csv": str(gaze_path),
            "pupil_csv": str(pupil_path),
            "capture_records_csv": str(records_path),
            "report_json": str(report_path),
        }
        if self.calibration is not None:
            calibrated_path = self.output_dir / "live_gaze_calibrated.csv"
            calibration_path = self.output_dir / "calibration.json"
            io.write_gaze_csv(self.calibration.apply_stream(gaze), calibrated_path)
            calibration_path.write_text(json.dumps(self.calibration.to_dict(), indent=2))
            paths["calibrated_gaze_csv"] = str(calibrated_path)
            paths["calibration_json"] = str(calibration_path)
        return paths

    def fit_calibration(self, targets: list[dict[str, float]]) -> AffineCalibration:
        """Fit calibration from recent raw gaze samples to supplied targets."""
        samples = self.snapshot()
        if len(targets) < 3:
            msg = "calibration needs at least 3 target points"
            raise RuntimeError(msg)
        if len(samples) < len(targets):
            msg = "not enough live samples to fit calibration"
            raise RuntimeError(msg)
        recent = samples[-len(targets) :]
        cal = AffineCalibration.fit(
            [sample.gaze.x for sample in recent],
            [sample.gaze.y for sample in recent],
            [target["x"] for target in targets],
            [target["y"] for target in targets],
        )
        with self._lock:
            self.calibration = cal
        return cal

    def reset_calibration(self) -> None:
        with self._lock:
            self.calibration = None


def _json_float(value: float | int | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _json_float_list(values: Iterable[float | np.floating[Any]]) -> list[float | None]:
    return [_json_float(value) for value in values]


def _samples_to_streams(samples: list[CaptureSample]) -> tuple[GazeStream, PupilStream]:
    gaze = GazeStream(
        t=np.array([sample.gaze.t for sample in samples], dtype=np.float64),
        x=np.array([sample.gaze.x for sample in samples], dtype=np.float64),
        y=np.array([sample.gaze.y for sample in samples], dtype=np.float64),
    )
    pupil_samples = [sample.pupil for sample in samples if sample.pupil is not None]
    pupil = PupilStream(
        t=np.array([sample.t for sample in pupil_samples], dtype=np.float64),
        size=np.array([sample.size for sample in pupil_samples], dtype=np.float64),
        unit=pupil_samples[0].unit if pupil_samples else PupilUnit.RELATIVE,
    )
    return gaze, pupil


def _finite_gaze_fraction(gaze: GazeStream) -> float:
    """Return the fraction of samples with finite binocular gaze coordinates."""
    if len(gaze) == 0:
        return 0.0
    finite = np.isfinite(gaze.x) & np.isfinite(gaze.y)
    return float(np.mean(finite))


def _write_capture_records_csv(samples: list[CaptureSample], out: Path) -> Path:
    quality_keys = sorted({key for sample in samples for key in sample.quality})
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "gaze_x_deg",
        "gaze_y_deg",
        "pupil_size",
        "pupil_unit",
        "fps_estimate_hz",
        *[f"quality_{key}" for key in quality_keys],
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row: dict[str, object] = {
                "frame_index": sample.frame_index,
                "timestamp_s": sample.timestamp_s,
                "gaze_x_deg": sample.gaze.x,
                "gaze_y_deg": sample.gaze.y,
                "pupil_size": sample.pupil.size if sample.pupil is not None else "",
                "pupil_unit": sample.pupil.unit.value if sample.pupil is not None else "",
                "fps_estimate_hz": sample.fps_estimate_hz,
            }
            row.update({f"quality_{key}": sample.quality.get(key, "") for key in quality_keys})
            writer.writerow(row)
    return out


def _method_name(method: str) -> DetectionMethodName:
    return "adaptive_ivt" if method == "adaptive_ivt" else "ivt"


def _analysis_payload(
    samples: list[CaptureSample],
    *,
    method: DetectionMethodName,
    velocity_threshold_deg_s: float,
    include_pso: bool,
    calibration: AffineCalibration | None = None,
) -> dict[str, object]:
    if not samples:
        empty_gaze = GazeStream(
            t=np.zeros(0, dtype=np.float64),
            x=np.zeros(0, dtype=np.float64),
            y=np.zeros(0, dtype=np.float64),
        )
        empty_pupil = PupilStream(
            t=np.zeros(0, dtype=np.float64),
            size=np.zeros(0, dtype=np.float64),
            unit=PupilUnit.RELATIVE,
        )
        return {
            "report": {"n_samples": 0, "duration_s": 0.0},
            "series": {"t": [], "x": [], "y": [], "pupil": [], "speed": []},
            "statistics": session_statistics(empty_gaze, empty_pupil),
            "diagnostics": validation.live_recording_diagnostics(
                empty_gaze,
                empty_pupil,
                {"n_samples": 0, "duration_s": 0.0, "quality": {}},
            ),
        }
    gaze, pupil = _samples_to_streams(samples)
    speed: list[float | None]
    if len(samples) >= 2 and np.all(np.isfinite(gaze.x)) and np.all(np.isfinite(gaze.y)):
        _vx, _vy, speed_arr = saccades.velocities(gaze)
        speed = _json_float_list(speed_arr)
    else:
        speed = [None for _ in samples]

    report_dict: dict[str, object]
    statistics_payload: dict[str, object]
    if len(samples) >= 3:
        cfg = AnalysisConfig(
            detection=DetectionConfig(
                method=method,
                velocity_threshold_deg_s=velocity_threshold_deg_s,
                include_pso=include_pso,
            )
        )
        try:
            report = pipeline.analyze_session(gaze, pupil, config=cfg)
            report_dict = report.to_dict()
            statistics_payload = session_statistics(gaze, pupil, report)
        except ValueError as exc:
            report_dict = {
                "n_samples": len(samples),
                "duration_s": float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0,
                "n_fixations": 0,
                "n_saccades": 0,
                "n_microsaccades": 0,
                "n_psos": 0,
                "scanpath": "",
                "quality": {"finite_sample_fraction": _finite_gaze_fraction(gaze)},
                "saccades": [],
                "fixations": [],
                "analysis_error": str(exc),
            }
            statistics_payload = session_statistics(gaze, pupil)
    else:
        duration = float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0
        report_dict = {
            "n_samples": len(samples),
            "duration_s": duration,
            "n_fixations": 0,
            "n_saccades": 0,
            "n_microsaccades": 0,
            "n_psos": 0,
            "scanpath": "",
            "quality": {"finite_sample_fraction": _finite_gaze_fraction(gaze)},
            "saccades": [],
            "fixations": [],
        }
        statistics_payload = session_statistics(gaze, pupil)
    series: dict[str, object] = {
        "t": _json_float_list(gaze.t),
        "x": _json_float_list(gaze.x),
        "y": _json_float_list(gaze.y),
        "pupil": _json_float_list(pupil.size),
        "speed": speed,
    }
    if calibration is not None:
        calibrated = calibration.apply_stream(gaze)
        series["calibrated_x"] = _json_float_list(calibrated.x)
        series["calibrated_y"] = _json_float_list(calibrated.y)
    return {
        "report": report_dict,
        "series": series,
        "statistics": statistics_payload,
        "diagnostics": validation.live_recording_diagnostics(gaze, pupil, report_dict),
    }


def live_message_from_frame(
    frame: LiveFrameSample,
    state: LiveState,
    *,
    method: str = "ivt",
    velocity_threshold_deg_s: float = 30.0,
    include_pso: bool = False,
    rolling_window_s: float = 10.0,
) -> dict[str, object]:
    """Build the browser message for one live frame."""
    method_name = _method_name(method)
    samples = state.recent(rolling_window_s)
    analysis = _analysis_payload(
        samples,
        method=method_name,
        velocity_threshold_deg_s=velocity_threshold_deg_s,
        include_pso=include_pso,
        calibration=state.calibration,
    )
    capture = frame.capture
    calibrated_gaze = None
    if state.calibration is not None:
        cx, cy = state.calibration.apply([capture.gaze.x], [capture.gaze.y])
        calibrated_gaze = {"x": _json_float(cx[0]), "y": _json_float(cy[0])}
    return {
        "type": "sample",
        "capture": {
            "frame_index": capture.frame_index,
            "timestamp_s": _json_float(capture.timestamp_s),
            "gaze": {"x": _json_float(capture.gaze.x), "y": _json_float(capture.gaze.y)},
            "calibrated_gaze": calibrated_gaze,
            "pupil": {
                "size": _json_float(capture.pupil.size) if capture.pupil is not None else None,
                "unit": capture.pupil.unit.value if capture.pupil is not None else None,
            },
            "fps_estimate_hz": _json_float(capture.fps_estimate_hz),
            "quality": capture.quality,
        },
        "frame": {
            "width": frame.frame_width,
            "height": frame.frame_height,
            "eye_box": frame.eye_box.to_dict(),
            "eye_crop_jpeg": frame.eye_crop_jpeg,
        },
        "analysis": analysis["report"],
        "series": analysis["series"],
        "statistics": analysis["statistics"],
        "diagnostics": analysis["diagnostics"],
        "method": {
            "name": method_name,
            "velocity_threshold_deg_s": velocity_threshold_deg_s,
            "include_pso": include_pso,
            "rolling_window_s": rolling_window_s,
        },
        "calibration": {
            "active": state.calibration is not None,
            "target_range_deg": state.calibration_target_range_deg,
        },
    }


def _real_frame_source(
    camera_index: int,
    max_frames: int | None,
    *,
    backend_logs: bool,
) -> Iterable[LiveFrameSample]:
    with _native_stderr(backend_logs):
        source = WebcamSource(camera_index=camera_index)
        yield from source.live_frames(max_frames=max_frames)


def create_app(
    *,
    camera_index: int = 0,
    output_dir: str | Path | None = None,
    backend_logs: bool = False,
    frame_source_factory: FrameSourceFactory | None = None,
    calibration_target_deg: float = 15.0,
) -> Any:
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
    # FastAPI resolves postponed annotations against module globals. Keep the
    # actual classes out of module import time, but make them visible after the
    # lazy import so the WebSocket parameter is not treated as a query field.
    globals()["WebSocket"] = WebSocket
    globals()["WebSocketDisconnect"] = WebSocketDisconnect

    assets_dir = Path(__file__).with_name("web_static")
    state = LiveState(
        output_dir=Path(output_dir) if output_dir is not None else None,
        calibration_target_range_deg=calibration_target_deg,
    )
    app = FastAPI(title="iTrace Live HTML", version="0.4.0")
    app.state.live_state = state
    app.mount("/static", StaticFiles(directory=assets_dir), name="static")

    @app.get("/")
    def index() -> Any:
        return FileResponse(assets_dir / "index.html")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker() -> Any:
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
            },
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
        analysis = _analysis_payload(
            state.recent(window_s),
            method=_method_name(method),
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

    @app.post("/api/calibration/reset")
    def reset_calibration() -> dict[str, object]:
        state.reset_calibration()
        return {"ok": True}

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
        backend_logs=backend_logs,
    )
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="info")
