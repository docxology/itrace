from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import RLock

from ..capture import CaptureSample
from . import clocks
from .calibration import Calibration, Frame
from .events import transition, validate_shape
from .jsonio import JsonValue, StudyError, fields, number, parse_object, text
from .model import StudyManifest
from .readiness import capture_quality
from .runtime import Runtime
from .setup import parse_setup
from .storage import Journal


class StudySession:
    def __init__(
        self,
        manifest: StudyManifest,
        output_root: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = RLock()
        self._clock = clock
        self._last_clock = -1.0
        self._runtime = Runtime(manifest)
        self._journal = Journal(manifest, output_root)
        self._frames: deque[Frame] = deque(maxlen=30000)
        self._last_native: tuple[float, int] | None = None
        self._captures = 0
        self._exports = 0
        self._journal.append("created", {"manifest_digest": manifest.digest}, self._now())

    def _now(self) -> float:
        now = number(self._clock(), "host_clock")
        if now < self._last_clock:
            self._runtime.invalidate("host clock moved backwards")
            raise StudyError("host_clock", "host clock moved backwards")
        self._last_clock = now
        return now

    def status(self) -> dict[str, JsonValue]:
        with self._lock:
            if self._journal.failed:
                self._runtime.invalidate("durable storage failed")
            now = self._now()
            result = self._runtime.payload(now)
            result.update(
                {
                    "session_id": self._journal.session_id,
                    "output_dir": str(self._journal.path),
                    "storage_failed": self._journal.failed,
                    "capture_quality": capture_quality(
                        list(self._frames),
                        now,
                        self._runtime.manifest.limits,
                        active=self._runtime.capture_active,
                    ),
                }
            )
            return result

    def prepare(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            if self._runtime.state != "setup":
                raise StudyError("state", "prepare is exclusive and cannot be replayed")
            setup = parse_setup(payload)
            self._journal.append("prepare", parse_object(setup.payload), self._now())
            self._runtime.setup = setup
            self._runtime.calibration = Calibration(setup, self._runtime.manifest.limits)
            self._runtime.state = "calibration"
            return self.status()

    def sync(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            setup = self._runtime.require_setup()
            if len(self._runtime.exchanges) >= 4096:
                raise StudyError("sync", "exchange count exceeds 4096")
            exchange = clocks.begin(payload, setup.client_epoch, self._now(), self._now())
            self._journal.append("sync", exchange.payload(), exchange.server_received_s)
            self._runtime.exchanges[exchange.sync_id] = exchange
            return exchange.payload()

    def confirm_sync(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            sync_id = text(payload.get("sync_id"), "sync_id", identifier=True)
            previous = self._runtime.exchanges.get(sync_id)
            if previous is None:
                raise StudyError("sync_id", "unknown exchange")
            now = self._now()
            if now - previous.server_sent_s > self._runtime.manifest.limits.max_event_lag_s:
                raise StudyError("sync_id", "confirmation too late")
            exchange = clocks.confirm(payload, previous, self._runtime.manifest.limits)
            self._journal.append("confirm_sync", exchange.payload(), now)
            self._runtime.exchanges[sync_id] = exchange
            return exchange.payload()

    def start_trial(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            fields(payload, "trial_id")
            now = self._now()
            runtime = self._runtime
            if (
                runtime.state == "rest"
                and runtime.rest_until_s is not None
                and now >= runtime.rest_until_s
            ):
                runtime.state = "ready"
            if runtime.state != "ready" or not runtime.ready_sync(now):
                raise StudyError("state", "validated calibration and current sync required")
            runtime.require_capture()
            trial = runtime.trial()
            if payload["trial_id"] != trial.trial_id:
                raise StudyError("trial_id", "must start next frozen manifest trial")
            self._journal.append("start_trial", dict(payload), now)
            runtime.state = "awaiting_onset"
            return {"trial": trial.public_payload(), "status": self.status()}

    def event(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            now = self._now()
            runtime = self._runtime
            try:
                kind = validate_shape(payload)
                if payload["event_id"] in runtime.event_ids:
                    raise StudyError("event_id", "duplicate event identity")
                if runtime.state in {"aborted", "complete", "interrupted"}:
                    raise StudyError("state", "session cannot accept further events")
                exchange = runtime.exchanges.get(text(payload["sync_id"], "sync_id"))
                if exchange is None:
                    raise StudyError("sync_id", "unknown exchange")
                estimate, error = clocks.event_time(payload, exchange, now, runtime.manifest.limits)
                result = transition(
                    runtime, self._journal, payload, kind, estimate, error, now, list(self._frames)
                )
            except StudyError as exc:
                self._journal.append(
                    "rejected",
                    {"event": dict(payload), "field": exc.field, "reason": exc.reason},
                    now,
                )
                if exc.field in {"client_epoch", "browser_ms", "sync_id", "validation"}:
                    runtime.invalidate(exc.reason)
                if exc.field == "calibration":
                    calibration = runtime.require_calibration()
                    calibration.onset = None
                    if calibration.index == 9 and calibration.coefficients is None:
                        runtime.invalidate(exc.reason)
                raise
            return {
                "event_id": payload["event_id"],
                "sequence": self._journal.sequence,
                **result,
                "status": self.status(),
            }

    def abort(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        with self._lock:
            fields(payload, "reason")
            text(payload["reason"], "reason")
            self._journal.append("abort", dict(payload), self._now())
            self._runtime.invalidate(text(payload["reason"], "reason"), aborted=True)
            return self.status()

    def capture_state(self, active: bool, epoch_id: str, reason: str | None = None) -> None:
        with self._lock:
            text(epoch_id, "epoch_id", identifier=True)
            now = self._now()
            runtime = self._runtime
            changed = runtime.capture_epoch is not None and runtime.capture_epoch != epoch_id
            self._journal.append(
                "capture_state", {"active": active, "epoch_id": epoch_id, "reason": reason}, now
            )
            if changed or (not active and runtime.capture_active):
                runtime.invalidate(reason or "capture stopped/restarted")
            if changed:
                self._last_native = None
            runtime.capture_active, runtime.capture_epoch = active, epoch_id

    def capture(self, sample: CaptureSample, epoch_id: str, host_received_s: float) -> None:
        with self._lock:
            text(epoch_id, "epoch_id", identifier=True)
            receipt = number(host_received_s, "host_received_s")
            now = self._now()
            if self._captures >= 1000000:
                raise StudyError("capture", "capture count allocation bound reached")
            runtime = self._runtime
            previous = self._last_native
            valid_clock = (
                math.isfinite(sample.timestamp_s)
                and type(sample.timestamp_s) is not bool
                and type(sample.frame_index) is int
                and sample.frame_index >= 0
                and 0 <= now - receipt <= 5
                and 0 <= receipt - sample.timestamp_s <= 5
                and epoch_id == runtime.capture_epoch
                and runtime.capture_active
                and (
                    previous is None
                    or (sample.timestamp_s > previous[0] and sample.frame_index > previous[1])
                )
            )
            reason = None if valid_clock else "native clock/frame discontinuity or inactive epoch"
            self._journal.capture(sample, epoch_id, receipt, valid_clock, reason, now)
            self._captures += 1
            if not valid_clock or self._captures > 1000000:
                self._journal.append(
                    "capture_fault", {"reason": reason or "capture count exceeded"}, receipt
                )
                runtime.invalidate(reason or "capture count exceeded")
                return
            self._last_native = sample.timestamp_s, sample.frame_index
            valid = (
                math.isfinite(sample.gaze.x)
                and math.isfinite(sample.gaze.y)
                and abs(sample.gaze.x) <= 1e6
                and abs(sample.gaze.y) <= 1e6
                and sample.quality.get("face_detected", 1) > 0
            )
            self._frames.append(
                Frame(sample.timestamp_s, epoch_id, sample.gaze.x, sample.gaze.y, valid)
            )

    def export(self) -> dict[str, JsonValue]:
        from .journal_read import read_journal
        from .replay import reconstruct

        with self._lock:
            if self._exports >= 32:
                raise StudyError("export", "export count exceeds 32")
            report = reconstruct(self._journal.path)
            records, _ = read_journal(self._journal.path)
            target = self._journal.export_files(report, records)
            self._exports += 1
            return {"export_dir": str(target), "report": report}
