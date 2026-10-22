"""Single camera owner; foreign callbacks run only on the acquisition thread."""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Final, Protocol, TypedDict, runtime_checkable

from ..capture import LiveFrameSample

_LOGGER: Final = logging.getLogger(__name__)


class CaptureParameterError(ValueError):
    def __init__(self, parameter: str) -> None:
        self.parameter = parameter
        super().__init__(f"invalid {parameter}")


class CaptureBusyError(RuntimeError):
    def __init__(self, epoch: str | None) -> None:
        self.epoch = epoch
        super().__init__(f"capture owner still running: {epoch}")


class CaptureStopTimeout(TimeoutError):
    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(f"capture owner did not stop within {timeout_s} seconds")


class CaptureStatus(TypedDict):
    active: bool
    running: bool
    epoch: str | None
    camera_index: int | None
    max_frames: int | None
    reason: str
    error: str | None
    frames_received: int
    frames_recorded: int
    preview_dropped: int


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    stage: str
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class CaptureEvent:
    active: bool = False
    epoch: str | None = None
    camera_index: int | None = None
    max_frames: int | None = None
    reason: str = "idle"
    failures: tuple[CaptureFailure, ...] = ()
    frames_received: int = 0
    frames_recorded: int = 0
    preview_dropped: int = 0


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


class CaptureService:
    """Mutable owner with immutable state snapshots and a single-frame preview.

    Recording callbacks must persist synchronously. A blocked foreign read cannot
    be killed safely: stop raises on timeout and retains ownership until exit.
    """

    def __init__(
        self,
        factory: Callable[[int, int | None], Iterable[LiveFrameSample]],
        on_frame: Callable[[LiveFrameSample, str, float], None],
        on_state: Callable[[bool, str, str | None], None],
    ) -> None:
        self._factory, self._on_frame, self._on_state = factory, on_frame, on_state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event = CaptureEvent()
        self._latest: LiveFrameSample | None = None
        self._preview_pending = False

    def start(self, camera_index: int, max_frames: int | None = None) -> str:
        """Claim the camera and return its epoch without opening hardware here."""
        if type(camera_index) is not int or not 0 <= camera_index <= 2**31 - 1:
            raise CaptureParameterError("camera_index")
        if max_frames is not None and (
            type(max_frames) is not int or not 1 <= max_frames <= sys.maxsize
        ):
            raise CaptureParameterError("max_frames")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise CaptureBusyError(self._event.epoch)
            epoch = uuid.uuid4().hex
            self._event = CaptureEvent(
                epoch=epoch, camera_index=camera_index, max_frames=max_frames, reason="starting"
            )
            self._stop = threading.Event()
            self._latest, self._preview_pending = None, False
            self._thread = threading.Thread(
                target=self._run,
                args=(camera_index, max_frames, epoch),
                name=f"itrace-capture-{epoch}",
                daemon=True,
            )
            try:
                self._thread.start()
            except RuntimeError as exc:
                failure = CaptureFailure("thread_start", type(exc).__name__, str(exc))
                self._event = replace(self._event, reason="failed", failures=(failure,))
                raise
            return epoch

    def stop(self, timeout_s: float = 5.0) -> None:
        """Request shutdown; never close an iterator from outside its owner."""
        try:
            valid = (
                type(timeout_s) in (int, float)
                and math.isfinite(timeout_s)
                and 0 <= timeout_s <= threading.TIMEOUT_MAX
            )
        except OverflowError as exc:
            raise CaptureParameterError("timeout_s") from exc
        if not valid:
            raise CaptureParameterError("timeout_s")
        with self._lock:
            thread = self._thread
            self._stop.set()
            if thread is None or not thread.is_alive():
                return
            if self._event.active:
                self._event = replace(self._event, reason="stopping")
        if thread is threading.current_thread():
            return
        thread.join(timeout_s)
        with self._lock:
            if self._thread is not thread or not thread.is_alive():
                return
            error = CaptureStopTimeout(float(timeout_s))
            failure = CaptureFailure("stop", type(error).__name__, str(error))
            self._event = replace(
                self._event, reason="stop_timeout", failures=(*self._event.failures, failure)
            )
            raise error

    def status(self) -> CaptureStatus:
        """Return a detached JSON snapshot; running includes cleanup/callbacks."""
        with self._lock:
            event = self._event
            return CaptureStatus(
                active=event.active,
                running=self._thread is not None and self._thread.is_alive(),
                epoch=event.epoch,
                camera_index=event.camera_index,
                max_frames=event.max_frames,
                reason=event.reason,
                error=self._error(),
                frames_received=event.frames_received,
                frames_recorded=event.frames_recorded,
                preview_dropped=event.preview_dropped,
            )

    def latest(self) -> LiveFrameSample | None:
        """Observe the one-frame preview without advancing the source."""
        with self._lock:
            self._preview_pending = False
            return self._latest

    def _error(self) -> str | None:
        return "; ".join(f"{f.stage}: {f.kind}: {f.message}" for f in self._event.failures) or None

    def _failure(self, stage: str, exc: BaseException) -> None:
        with self._lock:
            failure = CaptureFailure(stage, type(exc).__name__, str(exc))
            self._event = replace(self._event, failures=(*self._event.failures, failure))
            epoch = self._event.epoch
        _LOGGER.error("capture.worker_failed", extra={"epoch": epoch, "stage": stage}, exc_info=exc)

    def _close(self, resource: _Closable) -> None:
        try:
            resource.close()
        except BaseException as exc:
            # Foreign cleanup must not replace the original failure or prevent other closes.
            self._failure("cleanup", exc)

    def _consume(self, frames: Iterator[LiveFrameSample], epoch: str, limit: int | None) -> None:
        stage = "read"
        received = 0
        try:
            while not self._stop.is_set() and (limit is None or received < limit):
                stage = "read"
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                receipt = time.monotonic()
                received += 1
                with self._lock:
                    self._event = replace(self._event, frames_received=received)
                stage = "on_frame"
                self._on_frame(frame, epoch, receipt)
                with self._lock:
                    self._event = replace(
                        self._event,
                        frames_recorded=self._event.frames_recorded + 1,
                        preview_dropped=self._event.preview_dropped + int(self._preview_pending),
                    )
                    self._latest, self._preview_pending = frame, True
        except BaseException as exc:
            # Foreign iterators/callbacks can raise arbitrary errors; retain them before cleanup.
            self._failure(stage, exc)

    def _run(self, camera: int, limit: int | None, epoch: str) -> None:
        stage = "on_state_active"
        try:
            if not self._stop.is_set():
                with self._lock:
                    self._event = replace(self._event, active=True, reason="active")
                self._on_state(True, "active", None)
                stage = "factory"
                with ExitStack() as cleanup:
                    source = self._factory(camera, limit)
                    if isinstance(source, _Closable):
                        cleanup.callback(self._close, source)
                    stage = "iterator"
                    frames = iter(source)
                    if frames is not source and isinstance(frames, _Closable):
                        cleanup.callback(self._close, frames)
                    self._consume(frames, epoch, limit)
                    stage = "cleanup"
        except BaseException as exc:
            # This is the thread boundary: foreign construction/close failures become status.
            self._failure(stage, exc)
        with self._lock:
            reason = (
                "failed"
                if self._event.failures
                else ("stopped" if self._stop.is_set() else "completed")
            )
            self._event = replace(self._event, active=False, reason=reason)
            error = self._error()
        try:
            self._on_state(False, reason, error)
        except BaseException as exc:
            # State observers are also foreign; failed notification must remain visible.
            self._failure("on_state_terminal", exc)
            with self._lock:
                self._event = replace(self._event, reason="failed")
