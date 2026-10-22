from __future__ import annotations

import math
from collections.abc import Mapping

from .calibration import Calibration, Dwell
from .jsonio import JsonValue, StudyError, encoded, fields, integer, number, text
from .model import StudyManifest
from .setup import parse_setup, rectangle


class ReplayTransitions:
    """Mutable journal reducer; saved dwells must follow the producer's atomic write order."""

    def __init__(self, manifest: StudyManifest) -> None:
        self.manifest = manifest
        self.state = "setup"
        self.created = False
        self.calibration: Calibration | None = None
        self.capture_epoch: str | None = None
        self.capture_active = False
        self.index = 0
        self.rest_until_s = 0.0
        self.expected: str | None = None
        self.failure: StudyError | None = None
        self.collect_event: Mapping[str, JsonValue] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"aborted", "interrupted", "complete"}

    @property
    def report_state(self) -> str:
        return self.state if self.terminal or self.state == "post_validation" else "incomplete"

    def invalidate(self, *, aborted: bool = False) -> None:
        if self.state not in {"aborted", "complete"}:
            self.state = "aborted" if aborted else "interrupted"

    def require_calibration(self) -> Calibration:
        if self.calibration is None:
            raise StudyError("prepare", "preparation required before transition")
        return self.calibration

    def require_capture(self) -> str:
        if not self.capture_active or self.capture_epoch is None:
            raise StudyError("capture", "active capture epoch required")
        return self.capture_epoch

    def record(self, kind: str, payload: Mapping[str, JsonValue], received: float) -> None:
        if self.expected is not None and kind != self.expected:
            raise StudyError("journal", f"expected {self.expected} after preceding transition")
        if not self.created and kind != "created":
            raise StudyError("journal", "created must be first")
        if self.terminal and kind in {
            "created",
            "prepare",
            "events",
            "start_trial",
            "calibration",
            "validation",
            "post_validation",
        }:
            raise StudyError("state", "transition after terminal session state")
        match kind:
            case "created":
                if self.created:
                    raise StudyError("journal", "duplicate created record")
                self.created = True
            case "prepare":
                if self.state != "setup" or self.calibration is not None:
                    raise StudyError("prepare", "preparation is exclusive")
                self.calibration = Calibration(parse_setup(payload), self.manifest.limits)
                self.state = "calibration"
            case "sync" | "confirm_sync":
                self.require_calibration()
            case "start_trial":
                fields(payload, "trial_id")
                if self.state == "rest" and received >= self.rest_until_s:
                    self.state = "ready"
                if self.state != "ready" or self.index >= len(self.manifest.trials):
                    raise StudyError("state", "validated calibration and completed rest required")
                self.require_capture()
                if payload["trial_id"] != self.manifest.trials[self.index].trial_id:
                    raise StudyError("trial_id", "invalid frozen trial order")
                self.state = "awaiting_onset"
            case "capture_state":
                active = payload.get("active")
                if type(active) is not bool:
                    raise StudyError("capture", "capture active must be boolean")
                epoch = text(payload.get("epoch_id"), "epoch_id", identifier=True)
                if (self.capture_epoch is not None and epoch != self.capture_epoch) or (
                    not active and self.capture_active
                ):
                    self.invalidate()
                self.capture_epoch, self.capture_active = epoch, active
            case "abort" | "capture_fault":
                text(payload.get("reason"), "reason")
                self.invalidate(aborted=kind == "abort")
            case "rejected":
                field = text(payload.get("field"), "field")
                text(payload.get("reason"), "reason")
                if self.failure is not None:
                    if (field, payload["reason"], payload.get("event")) != (
                        self.failure.field,
                        self.failure.reason,
                        self.collect_event,
                    ):
                        raise StudyError("rejected", "rejection does not match failed collection")
                    self.failure = None
                    self.expected = None
                if field in {"client_epoch", "browser_ms", "sync_id", "validation"}:
                    self.invalidate()
                if field == "calibration":
                    calibration = self.require_calibration()
                    calibration.onset = None
                    if calibration.index == 9 and calibration.coefficients is None:
                        self.invalidate()
            case "calibration" | "validation" | "post_validation":
                self.dwell(kind, payload)
            case "calibration_diagnostics":
                if self.expected != kind or payload != self.require_calibration().diagnostics:
                    raise StudyError("calibration", "unbound or inconsistent diagnostics")
                self.expected = None

    def event(self, event: Mapping[str, JsonValue], estimate: float, uncertainty: float) -> None:
        kind = event["kind"]
        match kind:
            case "interrupt":
                text(event.get("reason"), "reason")
                self.invalidate()
            case "target_onset" | "target_collect":
                if self.state not in {"calibration", "validation", "post_validation"}:
                    raise StudyError("state", "target event outside calibration/validation")
                calibration = self.require_calibration()
                epoch = self.require_capture()
                match kind:
                    case "target_onset":
                        calibration.onset = calibration.begin(event, estimate, uncertainty, epoch)
                    case "target_collect":
                        onset, target = calibration.onset, calibration.target()
                        if (
                            onset is None
                            or target is None
                            or event["target_id"] != target["target_id"]
                        ):
                            raise StudyError("target_id", "target onset required")
                        end = (
                            onset.estimate_s
                            + onset.uncertainty_s
                            + self.manifest.limits.settle_s
                            + self.manifest.limits.collect_s
                        )
                        if onset.epoch != epoch or estimate - uncertainty < end:
                            raise StudyError(
                                "target_collect", "epoch changed or collection too early"
                            )
                        self.expected = self.state
                        self.collect_event = event
            case "trial_onset":
                if self.state != "awaiting_onset" or event["presentation"] != "raf":
                    raise StudyError("trial_onset", "start_trial and rAF onset required")
                self.require_capture()
                rect = rectangle(event["rect"], self.require_calibration().setup)
                trial = self.manifest.trials[self.index]
                ratio = number(rect["width"], "width") / number(rect["height"], "height")
                if abs(ratio / (trial.stimulus.width / trial.stimulus.height) - 1) > 0.02:
                    raise StudyError("rect", "stimulus aspect ratio changed")
                self.state = "trial"
            case "response" | "timeout":
                if self.state != "trial":
                    raise StudyError("response", "acknowledged trial onset required")
                self.require_capture()

    def finish_trial(self, received: float) -> None:
        trial = self.manifest.trials[self.index]
        self.index += 1
        self.rest_until_s = received + trial.rest_s
        self.state = (
            "post_validation"
            if self.index == len(self.manifest.trials)
            else "rest"
            if trial.rest_s
            else "ready"
        )

    def dwell(self, kind: str, payload: Mapping[str, JsonValue]) -> None:
        calibration = self.require_calibration()
        onset, target = calibration.onset, calibration.target()
        if self.expected != kind or onset is None or target is None:
            raise StudyError("calibration", "saved dwell requires acknowledged collection")
        if payload.get("target_id") != target["target_id"] or payload.get("phase") != kind:
            raise StudyError("calibration", "saved dwell target/phase mismatch")
        start = onset.estimate_s + onset.uncertainty_s + self.manifest.limits.settle_s
        for key, expected in (
            ("window_start_s", start),
            ("window_end_s", start + self.manifest.limits.collect_s),
            ("target_x_css_px", onset.x),
            ("target_y_css_px", onset.y),
        ):
            if not math.isclose(number(payload.get(key), key), expected, rel_tol=0, abs_tol=1e-9):
                raise StudyError("calibration", f"saved {key} differs from target event")
        if payload.get("epoch_id") != onset.epoch:
            raise StudyError("capture", "saved dwell epoch differs from target event")
        total = integer(payload.get("total_samples"), "total_samples", 0, 30000)
        valid = integer(payload.get("valid_samples"), "valid_samples", 0, total)
        fraction = valid / total if total else 0.0
        adequate = valid >= self.manifest.limits.min_samples and (
            fraction >= self.manifest.limits.min_valid_fraction
        )
        if payload.get("quality_adequate") is not adequate or (
            not adequate and calibration.index < 14
        ):
            raise StudyError("calibration", "saved dwell quality inconsistent with sample counts")
        for key, expected in (
            ("valid_fraction", fraction),
            ("dropout_fraction", 1 - fraction),
            ("excluded_samples", total - valid),
        ):
            if number(payload.get(key), key) != expected:
                raise StudyError("calibration", "inconsistent saved sample accounting")
        gx, gy = payload.get("gaze_x_deg"), payload.get("gaze_y_deg")
        if not adequate and (gx is not None or gy is not None):
            raise StudyError("calibration", "inadequate dwell must retain missing means")
        dwell = Dwell(
            number(gx, "gaze_x_deg", -1e6, 1e6) if adequate else math.nan,
            number(gy, "gaze_y_deg", -1e6, 1e6) if adequate else math.nan,
            onset.x,
            onset.y,
            encoded(payload),
        )
        try:
            calibration.accept(dwell)
        except StudyError as exc:
            self.failure = exc
            self.expected = "rejected"
            return
        self.expected = "calibration_diagnostics"
        match calibration.index:
            case 9:
                self.state = "validation"
            case 14:
                self.state = "ready"
            case 19:
                self.state = "complete"
