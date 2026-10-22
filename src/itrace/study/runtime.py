from __future__ import annotations

from dataclasses import dataclass, replace

from .calibration import Calibration
from .clocks import Exchange
from .jsonio import JsonValue, StudyError
from .model import StudyManifest, Trial
from .setup import Setup


@dataclass(frozen=True, slots=True)
class Outcome:
    trial_id: str
    role: str
    outcome: str = "unattempted"
    key: str | None = None
    onset_s: float | None = None
    uncertainty_s: float | None = None
    response_s: float | None = None
    correct: bool | None = None

    def payload(self, *, private: bool = False) -> dict[str, JsonValue]:
        return {
            "trial_id": self.trial_id,
            "role": self.role,
            "outcome": self.outcome,
            "key": self.key,
            "onset_s": self.onset_s,
            "uncertainty_s": self.uncertainty_s,
            "response_s": self.response_s,
            "rt_s": self.response_s - self.onset_s
            if self.response_s is not None and self.onset_s is not None
            else None,
            "correct": self.correct if private or self.role == "practice" else None,
        }


class Runtime:
    def __init__(self, manifest: StudyManifest) -> None:
        self.manifest = manifest
        self.state = "setup"
        self.setup: Setup | None = None
        self.calibration: Calibration | None = None
        self.capture_active = False
        self.capture_epoch: str | None = None
        self.exchanges: dict[str, Exchange] = {}
        self.event_ids: set[str] = set()
        self.last_event_s = -1.0
        self.index = 0
        self.rest_until_s: float | None = None
        self.outcomes = [Outcome(trial.trial_id, trial.role) for trial in manifest.trials]

    def trial(self) -> Trial:
        if self.index >= len(self.manifest.trials):
            raise StudyError("trial_id", "all trials finished")
        return self.manifest.trials[self.index]

    def require_capture(self) -> str:
        if not self.capture_active or self.capture_epoch is None:
            raise StudyError("capture", "active capture epoch required")
        return self.capture_epoch

    def require_setup(self) -> Setup:
        if self.setup is None:
            raise StudyError("setup", "prepare required")
        return self.setup

    def require_calibration(self) -> Calibration:
        if self.calibration is None:
            raise StudyError("calibration", "prepare required")
        return self.calibration

    def invalidate(self, reason: str, *, aborted: bool = False) -> None:
        if self.state in {"complete", "aborted"}:
            return
        if self.index < len(self.outcomes) and self.state in {"trial", "awaiting_onset"}:
            self.outcomes[self.index] = replace(
                self.outcomes[self.index],
                outcome="aborted" if aborted else "interrupted",
                correct=None,
            )
        self.state = "aborted" if aborted else "interrupted"

    def ready_sync(self, now: float) -> bool:
        return any(
            exchange.offset_s is not None
            and now - exchange.server_sent_s <= self.manifest.limits.sync_max_age_s
            for exchange in self.exchanges.values()
        )

    def payload(self, now: float) -> dict[str, JsonValue]:
        calibration = self.calibration
        return {
            "state": self.state,
            "manifest_digest": self.manifest.digest,
            "capture_active": self.capture_active,
            "capture_epoch": self.capture_epoch,
            "client_epoch": self.setup.client_epoch if self.setup is not None else None,
            "next_trial_id": self.manifest.trials[self.index].trial_id
            if self.index < len(self.outcomes)
            else None,
            "trial_index": self.index,
            "trial_count": len(self.outcomes),
            "target": calibration.target()
            if calibration is not None
            and self.state in {"calibration", "validation", "post_validation"}
            else None,
            "calibration": dict(calibration.diagnostics) if calibration is not None else None,
            "rest_until_s": self.rest_until_s,
            "ready": self.state == "ready" and self.capture_active and self.ready_sync(now),
            "outcomes": [outcome.payload() for outcome in self.outcomes],
        }
