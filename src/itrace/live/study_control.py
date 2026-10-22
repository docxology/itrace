from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import NoReturn

from ..study import JsonValue, StudyError, StudyManifest, StudySession
from .capture_service import CaptureService
from .study_connection import BrowserConnection


class StudyAction(Enum):
    PREPARE = "prepare"
    SYNC = "sync"
    CONFIRM_SYNC = "confirm-sync"
    TRIAL = "trial"
    EVENT = "event"
    ABORT = "abort"
    EXPORT = "export"
    CAPTURE_START = "capture-start"
    CAPTURE_STOP = "capture-stop"
    HEARTBEAT = "heartbeat"
    SAVE = "save"
    NEW_SESSION = "new-session"


def _unreachable(value: NoReturn) -> NoReturn:
    raise StudyError("action", f"unreachable study action: {value}")


@dataclass(frozen=True, slots=True)
class StudyController:
    manifest: StudyManifest
    session: StudySession
    capture: CaptureService
    token: str
    camera_index: int
    connection: BrowserConnection = field(default_factory=BrowserConnection)

    def expire_connection(self) -> None:
        if self.capture.status()["running"] and self.connection.expired():
            self.capture.stop()

    def context(self) -> dict[str, JsonValue]:
        return {"token": self.token, "manifest": self.manifest.public_payload(), **self.status()}

    def status(self) -> dict[str, JsonValue]:
        camera = self.capture.status()
        public_camera: dict[str, JsonValue] = {
            "active": camera["active"],
            "running": camera["running"],
            "epoch": camera["epoch"],
            "camera_index": self.camera_index,
            "max_frames": camera["max_frames"],
            "reason": camera["reason"],
            "error": camera["error"],
            "frames_received": camera["frames_received"],
            "frames_recorded": camera["frames_recorded"],
            "preview_dropped": camera["preview_dropped"],
        }
        return {"status": self.session.status(), "camera": public_camera}

    def authorized(self, supplied: str) -> bool:
        return secrets.compare_digest(self.token, supplied)

    def preview(self) -> dict[str, JsonValue]:
        frame = self.capture.latest()
        if frame is None:
            return {"available": False}
        sample = frame.capture
        return {
            "available": True,
            "eye_crop": frame.eye_crop_jpeg,
            "face_detected": sample.quality.get("face_detected", 0.0) == 1.0,
            "gaze_valid": isfinite(sample.gaze.x) and isfinite(sample.gaze.y),
            "frame_index": sample.frame_index,
            "timestamp_s": sample.timestamp_s if isfinite(sample.timestamp_s) else None,
        }

    def dispatch(
        self, action: StudyAction, payload: bytes, session_id: str | None = None
    ) -> dict[str, JsonValue]:
        from ..study import parse_object

        body = parse_object(payload)
        match action:
            case StudyAction.PREPARE:
                result = self.session.prepare(body)
                self.connection.renew()
            case StudyAction.HEARTBEAT:
                epoch = self.session.status()["client_epoch"]
                if set(body) != {"client_epoch"} or epoch is None or body["client_epoch"] != epoch:
                    raise StudyError("client_epoch", "heartbeat must identify the prepared page")
                self.connection.renew()
                result = {}
            case StudyAction.SYNC:
                result = self.session.sync(body)
            case StudyAction.CONFIRM_SYNC:
                result = self.session.confirm_sync(body)
            case StudyAction.TRIAL:
                result = self.session.start_trial(body)
            case StudyAction.EVENT:
                result = self.session.event(body)
            case StudyAction.ABORT:
                result = self.session.abort(body)
                self.capture.stop()
            case StudyAction.EXPORT:
                self._empty(body)
                result = self.session.export()
            case StudyAction.CAPTURE_START:
                self._empty(body)
                if self.session.status()["state"] not in {"calibration", "validation", "ready"}:
                    raise StudyError("state", "prepare a new session before starting capture")
                self.connection.renew()
                result = {"epoch": self.capture.start(self.camera_index)}
            case StudyAction.CAPTURE_STOP:
                self._empty(body)
                self.capture.stop()
                result = {}
            case StudyAction.SAVE | StudyAction.NEW_SESSION:
                raise StudyError("action", "session recovery requires the workspace controller")
            case unreachable:
                _unreachable(unreachable)
        return {"ok": True, "result": result, **self.status()}

    @staticmethod
    def _empty(body: Mapping[str, JsonValue]) -> None:
        if body:
            raise StudyError("request", "this action takes an empty JSON object")
