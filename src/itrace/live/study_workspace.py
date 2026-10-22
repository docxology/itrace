from collections.abc import Callable
from threading import RLock

from ..study import JsonValue, StudyError, StudyManifest
from ..study.jsonio import encoded, fields, mapping, parse_object, text
from .study_control import StudyAction, StudyController, _unreachable


class StudyWorkspace:
    def __init__(self, build: Callable[[], StudyController]) -> None:
        self._build = build
        self._controller = build()
        self._lock = RLock()
        self._saved: dict[str, JsonValue] | None = None
        self._prepared: bytes | None = None
        self._restart: tuple[str, str, dict[str, JsonValue]] | None = None

    @property
    def manifest(self) -> StudyManifest:
        return self._controller.manifest

    def context(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._controller.context()

    def status(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._controller.status()

    def preview(self) -> dict[str, JsonValue]:
        with self._lock:
            return self._controller.preview()

    def authorized(self, supplied: str) -> bool:
        return self._controller.authorized(supplied)

    def expire_connection(self) -> None:
        with self._lock:
            self._controller.expire_connection()

    def stop(self) -> None:
        with self._lock:
            self._controller.capture.stop()

    def dispatch(
        self, action: StudyAction, payload: bytes, session_id: str | None = None
    ) -> dict[str, JsonValue]:
        with self._lock:
            body = parse_object(payload)
            if action is StudyAction.NEW_SESSION:
                fields(body, "request_id")
                key = text(body["request_id"], "request_id", identifier=True)
                if self._restart is not None and (session_id, key) == self._restart[:2]:
                    return self._restart[2]
            current = self._controller.session.status()
            if session_id != current["session_id"]:
                raise StudyError("session_id", "control belongs to a different session; reload")
            terminal = current["state"] in {"complete", "aborted", "interrupted"}
            match action:
                case StudyAction.NEW_SESSION:
                    if not terminal:
                        raise StudyError(
                            "state", "stop the session before creating a terminal replacement"
                        )
                    self._controller.capture.stop()
                    saved = self._save()
                    fresh = self._build()
                    previous = text(current["session_id"], "session_id")
                    self._controller = fresh
                    self._saved, self._prepared = None, None
                    response: dict[str, JsonValue] = {
                        "ok": True,
                        "result": {
                            "previous_session_id": previous,
                            "saved_export_dir": saved["export_dir"],
                        },
                        **self.status(),
                    }
                    self._restart = (previous, key, response)
                    return response
                case StudyAction.SAVE:
                    fields(body, "")
                    if terminal:
                        self._controller.capture.stop()
                    result = self._save() if terminal else self._controller.session.export()
                    return {"ok": True, "result": result, **self.status()}
                case StudyAction.PREPARE:
                    canonical = encoded(body)
                    if self._prepared == canonical and current["state"] == "calibration":
                        return {"ok": True, "result": current, **self.status()}
                    answer = self._controller.dispatch(action, payload)
                    self._prepared = canonical
                    return answer
                case StudyAction.ABORT:
                    if terminal:
                        fields(body, "reason")
                        text(body["reason"], "reason")
                        self._controller.capture.stop()
                        return {"ok": True, "result": current, **self.status()}
                    return self._controller.dispatch(action, payload)
                case (
                    StudyAction.SYNC
                    | StudyAction.CONFIRM_SYNC
                    | StudyAction.TRIAL
                    | StudyAction.EVENT
                    | StudyAction.EXPORT
                    | StudyAction.CAPTURE_START
                    | StudyAction.CAPTURE_STOP
                    | StudyAction.HEARTBEAT
                ):
                    return self._controller.dispatch(action, payload)
                case unreachable:
                    _unreachable(unreachable)

    def _save(self) -> dict[str, JsonValue]:
        if self._saved is None:
            result = self._controller.session.export()
            self._saved = mapping(result, "export")
        return self._saved
