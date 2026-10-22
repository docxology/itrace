"""Immutable bound manifest values, deliberately excluding keys from public data."""

from __future__ import annotations

from dataclasses import dataclass

from .jsonio import JsonValue, StudyError


@dataclass(frozen=True, slots=True)
class Limits:
    max_sync_error_s: float
    sync_max_age_s: float
    max_event_lag_s: float
    max_validation_error_fraction: float
    settle_s: float
    collect_s: float
    min_samples: int
    min_valid_fraction: float


@dataclass(frozen=True, slots=True)
class Stimulus:
    path: str
    sha256: str
    width: int
    height: int
    data: bytes
    decoded_bytes: int


@dataclass(frozen=True, slots=True)
class Option:
    key: str
    label: str


@dataclass(frozen=True, slots=True)
class Trial:
    trial_id: str
    role: str
    block_id: str
    task_id: str
    family_id: str
    stimulus: Stimulus
    query: str
    options: tuple[Option, ...]
    correct_key: str
    deadline_s: float
    rest_s: float

    def public_payload(self) -> dict[str, JsonValue]:
        return {
            "trial_id": self.trial_id,
            "role": self.role,
            "block_id": self.block_id,
            "task_id": self.task_id,
            "family_id": self.family_id,
            "query": self.query,
            "options": [{"key": item.key, "label": item.label} for item in self.options],
            "stimulus": {
                "trial_id": self.trial_id,
                "sha256": self.stimulus.sha256,
                "width": self.stimulus.width,
                "height": self.stimulus.height,
            },
            "deadline_s": self.deadline_s,
            "rest_s": self.rest_s,
        }


@dataclass(frozen=True, slots=True)
class StudyManifest:
    study_id: str
    title: str
    trials: tuple[Trial, ...]
    limits: Limits
    digest: str
    original_bytes: bytes

    def public_payload(self) -> dict[str, JsonValue]:
        return {
            "schema": "itrace.private-study",
            "version": 1,
            "study_id": self.study_id,
            "title": self.title,
            "digest": self.digest,
            "trials": [trial.public_payload() for trial in self.trials],
            "scene_counts": {
                role: {
                    "trials": sum(trial.role == role for trial in self.trials),
                    "unique_stimulus_hashes": len(
                        {trial.stimulus.sha256 for trial in self.trials if trial.role == role}
                    ),
                    "unique_families": len(
                        {trial.family_id for trial in self.trials if trial.role == role}
                    ),
                }
                for role in ("practice", "scored")
            },
            "limits": {
                "max_sync_error_s": self.limits.max_sync_error_s,
                "sync_max_age_s": self.limits.sync_max_age_s,
                "max_event_lag_s": self.limits.max_event_lag_s,
                "max_validation_error_fraction": self.limits.max_validation_error_fraction,
                "settle_s": self.limits.settle_s,
                "collect_s": self.limits.collect_s,
                "min_samples": self.limits.min_samples,
                "min_valid_fraction": self.limits.min_valid_fraction,
            },
        }

    def stimulus_bytes(self, trial_id: str) -> bytes:
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial.stimulus.data
        raise StudyError("trial_id", "unknown trial")
