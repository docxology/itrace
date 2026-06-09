"""In-memory rolling state for one local HTML session."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock

import numpy as np

from .. import empirical, experiments, io, pipeline
from ..calibration import AffineCalibration
from ..capture import CaptureSample, samples_to_streams, write_capture_records_csv
from ..reporting import error_session_report_dict, partial_session_report_dict, session_report_dict

SESSION_PREFIX = "local_pilot"
SESSION_ID_RE = re.compile(r"^(?P<prefix>[A-Za-z][A-Za-z0-9_-]*)_(?P<number>\d+)$")
MANIFEST_KIND = "itrace_empirical_sessions_manifest"
DEFAULT_PARTICIPANT_ID = "P001"
DEFAULT_DEVICE_ID = "local_webcam_001"
DEFAULT_CONDITION = "indoor_office_daylight"
DEFAULT_CONSENT_SCOPE = "derived_records_only"
DEFAULT_REFERENCE_KIND = "none"
V1_CONDITION_SCHEDULE = (
    (1, 2, "indoor_office_daylight"),
    (3, 5, "indoor_office_dim"),
    (6, 8, "indoor_office_backlit"),
    (9, 12, "indoor_office_daylight"),
)


def _scheduled_condition_for_session_number(number: int) -> str:
    """Return the v1 collection condition planned for a local session number."""
    for start, end, condition in V1_CONDITION_SCHEDULE:
        if start <= number <= end:
            return condition
    return DEFAULT_CONDITION


@dataclass(frozen=True, slots=True)
class CalibrationSessionPoint:
    """One backend-aggregated calibration target sample."""

    target_x: float
    target_y: float
    raw_x: float
    raw_y: float
    timestamp_s: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """JSON-friendly calibration point record."""
        return {
            "target_x": self.target_x,
            "target_y": self.target_y,
            "raw_x": self.raw_x,
            "raw_y": self.raw_y,
            "timestamp_s": self.timestamp_s,
            "n_samples": float(self.n_samples),
        }


@dataclass(slots=True)
class LiveState:
    """In-memory rolling capture state for one local HTML session."""

    output_dir: Path | None = None
    empirical_manifest_path: Path | None = None
    max_samples: int = 5000
    samples: list[CaptureSample] = field(default_factory=list)
    calibration_points: list[CalibrationSessionPoint] = field(default_factory=list)
    calibration: AffineCalibration | None = None
    calibration_target_range_deg: float = 15.0
    experiment_protocol: experiments.ExperimentProtocol | None = None
    experiment_trials: list[experiments.RecordedTrial] = field(default_factory=list)
    experiment_active_trial: experiments.RecordedTrial | None = None
    experiment_samples: list[CaptureSample] = field(default_factory=list)
    experiment_session_id: str | None = None
    experiment_replicate_id: str | None = None
    experiment_condition: str = DEFAULT_CONDITION
    experiment_participant_id: str = DEFAULT_PARTICIPANT_ID
    experiment_device_id: str = DEFAULT_DEVICE_ID
    experiment_session_group: str | None = None
    experiment_consent_scope: str = DEFAULT_CONSENT_SCOPE
    experiment_reference_kind: str = DEFAULT_REFERENCE_KIND
    experiment_output_dir: Path | None = None
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
            if self.experiment_protocol is not None:
                self.experiment_samples.append(sample)
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
            self.calibration_points.clear()
            self.calibration = None
            self.experiment_protocol = None
            self.experiment_trials.clear()
            self.experiment_active_trial = None
            self.experiment_samples.clear()
            self.experiment_session_id = None
            self.experiment_replicate_id = None
            self.experiment_output_dir = None

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
        gaze, pupil = samples_to_streams(samples)

        gaze_path = self.output_dir / "live_gaze.csv"
        pupil_path = self.output_dir / "live_pupil.csv"
        records_path = self.output_dir / "live_capture_records.csv"
        report_path = self.output_dir / "live_report.json"

        io.write_gaze_csv(gaze, gaze_path)
        io.write_pupil_csv(pupil, pupil_path)
        write_capture_records_csv(samples, records_path)
        if len(samples) >= 3:
            try:
                report_payload = session_report_dict(pipeline.analyze_session(gaze, pupil))
            except ValueError as exc:
                report_payload = error_session_report_dict(
                    n_samples=len(samples),
                    duration_s=float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0,
                    quality={},
                    error=str(exc),
                )
        else:
            report_payload = partial_session_report_dict(
                n_samples=len(samples),
                duration_s=float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0,
                quality={},
            )
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

    def start_calibration_session(self, target_range_deg: float | None = None) -> None:
        """Begin a backend-owned calibration sampling session."""
        with self._lock:
            self.calibration_points.clear()
            if target_range_deg is not None:
                if target_range_deg <= 0.0 or not np.isfinite(target_range_deg):
                    msg = "target_range_deg must be positive and finite"
                    raise RuntimeError(msg)
                self.calibration_target_range_deg = float(target_range_deg)

    def sample_calibration_target(
        self,
        *,
        target_x: float,
        target_y: float,
        window_s: float = 0.35,
        min_samples: int = 1,
    ) -> CalibrationSessionPoint:
        """Aggregate recent finite gaze samples for one calibration target."""
        if window_s <= 0.0 or not np.isfinite(window_s):
            msg = "window_s must be positive and finite"
            raise RuntimeError(msg)
        if min_samples < 1:
            msg = "min_samples must be >= 1"
            raise RuntimeError(msg)
        samples = self.snapshot()
        if not samples:
            msg = "no live samples available for calibration target"
            raise RuntimeError(msg)
        cutoff = samples[-1].timestamp_s - window_s
        recent = [
            sample
            for sample in samples
            if sample.timestamp_s >= cutoff
            and np.isfinite(sample.gaze.x)
            and np.isfinite(sample.gaze.y)
        ]
        if len(recent) < min_samples:
            msg = "not enough finite live samples for calibration target"
            raise RuntimeError(msg)
        point = CalibrationSessionPoint(
            target_x=float(target_x),
            target_y=float(target_y),
            raw_x=float(np.median([sample.gaze.x for sample in recent])),
            raw_y=float(np.median([sample.gaze.y for sample in recent])),
            timestamp_s=float(np.median([sample.timestamp_s for sample in recent])),
            n_samples=len(recent),
        )
        with self._lock:
            self.calibration_points.append(point)
        return point

    def fit_calibration_session(self) -> AffineCalibration:
        """Fit calibration from backend-aggregated session points."""
        with self._lock:
            points = list(self.calibration_points)
        if len(points) < 3:
            msg = "calibration session needs at least 3 sampled targets"
            raise RuntimeError(msg)
        cal = AffineCalibration.fit(
            [point.raw_x for point in points],
            [point.raw_y for point in points],
            [point.target_x for point in points],
            [point.target_y for point in points],
        )
        with self._lock:
            self.calibration = cal
        return cal

    def reset_calibration_session(self) -> None:
        """Clear pending calibration target samples without removing a fitted model."""
        with self._lock:
            self.calibration_points.clear()

    def reset_calibration(self) -> None:
        with self._lock:
            self.calibration_points.clear()
            self.calibration = None

    def _latest_timestamp(self) -> float:
        with self._lock:
            return self.samples[-1].timestamp_s if self.samples else 0.0

    def start_experiment_session(
        self,
        *,
        trial_duration_s: float = 30.0,
        target_range_deg: float | None = None,
        condition: str | None = None,
        participant_id: str | None = None,
        device_id: str | None = None,
        session_group: str | None = None,
        consent_scope: str | None = None,
        reference_kind: str | None = None,
    ) -> experiments.ExperimentProtocol:
        """Start a backend-owned guided eye-video experiment session."""
        target_range = (
            self.calibration_target_range_deg
            if target_range_deg is None
            else float(target_range_deg)
        )
        protocol = experiments.default_eye_video_protocol(
            trial_duration_s=float(trial_duration_s),
            target_range_deg=target_range,
        )
        defaults = self._manifest_metadata_defaults()
        resolved_participant = (
            participant_id or defaults["participant_id"] or DEFAULT_PARTICIPANT_ID
        )
        resolved_device = device_id or defaults["device_id"] or DEFAULT_DEVICE_ID
        resolved_group = session_group or defaults["session_group"]
        if resolved_group is None:
            resolved_group = f"{resolved_participant}_{resolved_device}"
        resolved_condition = condition or defaults["condition"] or DEFAULT_CONDITION
        resolved_consent_scope = consent_scope or defaults["consent_scope"] or DEFAULT_CONSENT_SCOPE
        resolved_reference_kind = (
            reference_kind or defaults["reference_kind"] or DEFAULT_REFERENCE_KIND
        )
        if resolved_reference_kind not in empirical.REFERENCE_KINDS:
            msg = f"reference_kind must be one of {sorted(empirical.REFERENCE_KINDS)}"
            raise ValueError(msg)
        session_id: str | None = None
        replicate_id: str | None = None
        session_dir: Path | None = None
        if self.output_dir is not None:
            session_id, replicate_id, session_dir = self._allocate_experiment_session()
        with self._lock:
            self.experiment_protocol = protocol
            self.experiment_trials.clear()
            self.experiment_active_trial = None
            self.experiment_samples.clear()
            self.experiment_session_id = session_id
            self.experiment_replicate_id = replicate_id
            self.experiment_condition = resolved_condition
            self.experiment_participant_id = resolved_participant
            self.experiment_device_id = resolved_device
            self.experiment_session_group = resolved_group
            self.experiment_consent_scope = resolved_consent_scope
            self.experiment_reference_kind = resolved_reference_kind
            self.experiment_output_dir = session_dir
        return protocol

    def experiment_status(self) -> dict[str, object]:
        """Return JSON-friendly experiment session state."""
        with self._lock:
            protocol = self.experiment_protocol
            trials = list(self.experiment_trials)
            active = self.experiment_active_trial
            session_id = self.experiment_session_id
            replicate_id = self.experiment_replicate_id
            samples = list(self.samples)
            experiment_samples = list(self.experiment_samples)
            session_output_dir = self.experiment_output_dir
            condition = self.experiment_condition
            participant_id = self.experiment_participant_id
            device_id = self.experiment_device_id
            session_group = self.experiment_session_group
            consent_scope = self.experiment_consent_scope
            reference_kind = self.experiment_reference_kind
        next_session_id: str | None = None
        next_replicate_id: str | None = None
        if protocol is None and self.output_dir is not None:
            next_session_id, next_replicate_id, _ = self._preview_experiment_session()
            defaults = self._manifest_metadata_defaults()
            participant_id = defaults["participant_id"] or DEFAULT_PARTICIPANT_ID
            device_id = defaults["device_id"] or DEFAULT_DEVICE_ID
            session_group = defaults["session_group"]
            condition = defaults["condition"] or DEFAULT_CONDITION
            consent_scope = defaults["consent_scope"] or DEFAULT_CONSENT_SCOPE
            reference_kind = defaults["reference_kind"] or DEFAULT_REFERENCE_KIND
        sample_count = len(samples)
        experiment_sample_count = len(experiment_samples)
        latest_timestamp = samples[-1].timestamp_s if samples else 0.0
        trial_statuses: list[dict[str, object]] = []
        completed_ids = {trial.trial_id for trial in trials if trial.ended_at_s is not None}
        active_spec = protocol.trial(active.trial_id) if protocol is not None and active else None
        active_elapsed_s: float | None = None
        active_remaining_s: float | None = None
        active_progress: float | None = None
        current_target: dict[str, object] | None = None
        current_target_remaining_s: float | None = None
        current_target_progress: float | None = None
        if active is not None and active_spec is not None:
            active_elapsed_s = max(0.0, float(latest_timestamp - active.started_at_s))
            active_remaining_s = max(0.0, float(active_spec.duration_s - active_elapsed_s))
            active_progress = min(1.0, active_elapsed_s / active_spec.duration_s)
            for cue in active_spec.target_schedule:
                if cue.start_s <= active_elapsed_s <= cue.end_s:
                    cue_duration = max(1e-9, cue.end_s - cue.start_s)
                    cue_elapsed = min(cue_duration, max(0.0, active_elapsed_s - cue.start_s))
                    current_target = cue.to_dict()
                    current_target["elapsed_s"] = cue_elapsed
                    current_target_remaining_s = max(0.0, cue.end_s - active_elapsed_s)
                    current_target["remaining_s"] = current_target_remaining_s
                    current_target_progress = min(1.0, cue_elapsed / cue_duration)
                    current_target["progress"] = current_target_progress
                    break
        if protocol is not None:
            for spec in protocol.trials:
                finished = next(
                    (
                        trial
                        for trial in reversed(trials)
                        if trial.trial_id == spec.trial_id and trial.ended_at_s is not None
                    ),
                    None,
                )
                is_active = active is not None and active.trial_id == spec.trial_id
                window_start = active.started_at_s if is_active and active else None
                window_end = latest_timestamp if is_active else None
                if finished is not None:
                    window_start = finished.started_at_s
                    window_end = finished.ended_at_s
                window_source = experiment_samples if experiment_samples else samples
                window = []
                if window_start is not None and window_end is not None:
                    window = [
                        sample
                        for sample in window_source
                        if window_start <= sample.timestamp_s <= window_end
                    ]
                finite_count = sum(
                    1
                    for sample in window
                    if np.isfinite(sample.gaze.x) and np.isfinite(sample.gaze.y)
                )
                trial_statuses.append(
                    {
                        "trial_id": spec.trial_id,
                        "kind": spec.kind,
                        "status": "active"
                        if is_active
                        else "complete"
                        if finished is not None
                        else "pending",
                        "expected_duration_s": spec.duration_s,
                        "observed_duration_s": max(0.0, float(window_end - window_start))
                        if window_start is not None and window_end is not None
                        else 0.0,
                        "progress": min(
                            1.0,
                            max(0.0, float(window_end - window_start)) / spec.duration_s,
                        )
                        if window_start is not None and window_end is not None
                        else 0.0,
                        "remaining_s": max(
                            0.0,
                            spec.duration_s - max(0.0, float(window_end - window_start)),
                        )
                        if window_start is not None and window_end is not None
                        else spec.duration_s,
                        "sample_count": len(window),
                        "finite_gaze_count": finite_count,
                        "finite_gaze_fraction": finite_count / len(window) if window else None,
                    }
                )
        required_ids = [trial.trial_id for trial in protocol.trials] if protocol is not None else []
        all_trials_completed = bool(required_ids) and all(
            trial_id in completed_ids for trial_id in required_ids
        )
        missing_trial_ids = [trial_id for trial_id in required_ids if trial_id not in completed_ids]
        next_trial_id = missing_trial_ids[0] if missing_trial_ids and active is None else None
        export_blockers: list[str] = []
        if protocol is None:
            export_blockers.append("start an experiment session")
        if active is not None:
            export_blockers.append("finish the active trial")
        if missing_trial_ids:
            export_blockers.append("complete: " + ", ".join(missing_trial_ids))
        if self.output_dir is None:
            export_blockers.append("configure --output-dir")
        export_ready = (
            protocol is not None
            and active is None
            and all_trials_completed
            and self.output_dir is not None
        )
        return {
            "active": protocol is not None,
            "session_id": session_id,
            "replicate_id": replicate_id,
            "next_session_id": next_session_id,
            "next_replicate_id": next_replicate_id,
            "session_output_dir": str(session_output_dir)
            if session_output_dir is not None
            else None,
            "metadata": {
                "participant_id": participant_id,
                "device_id": device_id,
                "session_group": session_group,
                "condition": condition,
                "consent_scope": consent_scope,
                "reference_kind": reference_kind,
            },
            "protocol": protocol.to_dict() if protocol is not None else None,
            "recorded_trials": [trial.to_dict() for trial in trials],
            "active_trial": active.to_dict() if active is not None else None,
            "active_trial_id": active.trial_id if active is not None else None,
            "trial_started_at_s": active.started_at_s if active is not None else None,
            "trial_duration_s": active_spec.duration_s if active_spec is not None else None,
            "trial_elapsed_s": active_elapsed_s,
            "trial_remaining_s": active_remaining_s,
            "trial_progress": active_progress,
            "current_target": current_target,
            "current_target_remaining_s": current_target_remaining_s,
            "current_target_progress": current_target_progress,
            "completed_trial_count": len(
                [trial for trial in trials if trial.ended_at_s is not None]
            ),
            "sample_count": sample_count,
            "experiment_sample_count": experiment_sample_count,
            "required_trial_count": len(required_ids),
            "completed_trial_ids": [
                trial_id for trial_id in required_ids if trial_id in completed_ids
            ],
            "missing_trial_ids": missing_trial_ids,
            "next_trial_id": next_trial_id,
            "all_trials_completed": all_trials_completed,
            "trial_statuses": trial_statuses,
            "export_ready": export_ready,
            "export_blockers": export_blockers,
        }

    def start_experiment_trial(self, trial_id: str) -> experiments.RecordedTrial:
        """Mark one guided trial as active at the latest sample timestamp."""
        with self._lock:
            protocol = self.experiment_protocol
            active = self.experiment_active_trial
            trials = list(self.experiment_trials)
        if protocol is None:
            msg = "experiment session has not been started"
            raise RuntimeError(msg)
        protocol.trial(trial_id)
        if active is not None:
            msg = "another experiment trial is already active"
            raise RuntimeError(msg)
        completed_ids = {trial.trial_id for trial in trials if trial.ended_at_s is not None}
        if trial_id in completed_ids:
            msg = "experiment trial has already been completed"
            raise RuntimeError(msg)
        next_pending = next(
            (spec.trial_id for spec in protocol.trials if spec.trial_id not in completed_ids),
            None,
        )
        if next_pending is not None and trial_id != next_pending:
            msg = f"start next pending experiment trial first: {next_pending}"
            raise RuntimeError(msg)
        trial = experiments.RecordedTrial(
            trial_id=trial_id,
            started_at_s=self._latest_timestamp(),
            ended_at_s=None,
        )
        with self._lock:
            self.experiment_active_trial = trial
        return trial

    def finish_experiment_trial(self, trial_id: str | None = None) -> experiments.RecordedTrial:
        """Finish the active guided trial at the latest sample timestamp."""
        with self._lock:
            active = self.experiment_active_trial
        if active is None:
            msg = "no experiment trial is active"
            raise RuntimeError(msg)
        if trial_id is not None and trial_id != active.trial_id:
            msg = "active experiment trial does not match requested trial_id"
            raise RuntimeError(msg)
        ended_at = self._latest_timestamp()
        if ended_at <= active.started_at_s:
            ended_at = active.started_at_s + 1e-6
        finished = experiments.RecordedTrial(
            trial_id=active.trial_id,
            started_at_s=active.started_at_s,
            ended_at_s=ended_at,
        )
        with self._lock:
            self.experiment_trials.append(finished)
            self.experiment_active_trial = None
        return finished

    def experiment_report(self) -> dict[str, object]:
        """Build the current derived experiment report."""
        with self._lock:
            protocol = self.experiment_protocol
            trials = list(self.experiment_trials)
            active = self.experiment_active_trial
        if protocol is None:
            msg = "experiment session has not been started"
            raise RuntimeError(msg)
        if active is not None:
            msg = "finish the active experiment trial before reporting"
            raise RuntimeError(msg)
        with self._lock:
            samples = (
                list(self.experiment_samples) if self.experiment_samples else list(self.samples)
            )
        return experiments.experiment_report(samples, protocol, trials)

    def export_experiment(self) -> dict[str, str]:
        """Write derived experiment artifacts for completed trials."""
        if self.output_dir is None:
            msg = "No output directory configured for this live session"
            raise RuntimeError(msg)
        with self._lock:
            protocol = self.experiment_protocol
            trials = list(self.experiment_trials)
            active = self.experiment_active_trial
            session_dir = self.experiment_output_dir
            session_id = self.experiment_session_id
            replicate_id = self.experiment_replicate_id
        if protocol is None:
            msg = "experiment session has not been started"
            raise RuntimeError(msg)
        if active is not None:
            msg = "finish the active experiment trial before export"
            raise RuntimeError(msg)
        if not trials:
            msg = "no experiment trials have been completed"
            raise RuntimeError(msg)
        required_ids = [trial.trial_id for trial in protocol.trials]
        completed_ids = {trial.trial_id for trial in trials if trial.ended_at_s is not None}
        missing = [trial_id for trial_id in required_ids if trial_id not in completed_ids]
        if missing:
            msg = "complete all experiment trials before export: " + ", ".join(missing)
            raise RuntimeError(msg)
        with self._lock:
            samples = (
                list(self.experiment_samples) if self.experiment_samples else list(self.samples)
            )
        if session_dir is None or session_id is None or replicate_id is None:
            session_id, replicate_id, session_dir = self._allocate_experiment_session()
            with self._lock:
                self.experiment_session_id = session_id
                self.experiment_replicate_id = replicate_id
                self.experiment_output_dir = session_dir
        paths = experiments.write_experiment_bundle(
            samples,
            protocol,
            trials,
            session_dir / "experiment",
        )
        paths["session_id"] = session_id
        paths["replicate_id"] = replicate_id
        paths["session_dir"] = str(session_dir)
        self._upsert_manifest_entry(paths["report_json"], protocol.protocol_id)
        summary_path = self._refresh_empirical_summary()
        if summary_path is not None:
            paths["empirical_summary_json"] = str(summary_path)
        return paths

    def reset_experiment(self) -> None:
        """Clear experiment protocol and trial state without clearing samples."""
        with self._lock:
            self.experiment_protocol = None
            self.experiment_trials.clear()
            self.experiment_active_trial = None
            self.experiment_samples.clear()
            self.experiment_session_id = None
            self.experiment_replicate_id = None
            self.experiment_output_dir = None

    def _manifest_metadata_defaults(self) -> dict[str, str | None]:
        defaults: dict[str, str | None] = {
            "participant_id": DEFAULT_PARTICIPANT_ID,
            "device_id": DEFAULT_DEVICE_ID,
            "session_group": None,
            "condition": DEFAULT_CONDITION,
            "consent_scope": DEFAULT_CONSENT_SCOPE,
            "reference_kind": DEFAULT_REFERENCE_KIND,
        }
        payload = self._read_manifest_payload()
        sessions = payload.get("sessions", [])
        if not isinstance(sessions, list):
            return defaults
        latest = next((row for row in reversed(sessions) if isinstance(row, dict)), None)
        if latest is not None:
            for key in defaults:
                value = latest.get(key)
                if isinstance(value, str) and value:
                    defaults[key] = value
        if self.output_dir is not None:
            _session_id, _replicate_id, session_dir = self._preview_experiment_session()
            match = SESSION_ID_RE.match(session_dir.name)
            if match is not None:
                defaults["condition"] = _scheduled_condition_for_session_number(
                    int(match.group("number"))
                )
            if defaults["session_group"] is None:
                defaults["session_group"] = f"{defaults['participant_id']}_{defaults['device_id']}"
        return defaults

    def _read_manifest_payload(self) -> dict[str, object]:
        path = self.empirical_manifest_path
        if path is None or not path.exists():
            return {
                "kind": MANIFEST_KIND,
                "version": 1,
                "truth_boundary": experiments.TRUTH_BOUNDARY,
                "storage_boundary": experiments.STORAGE_BOUNDARY,
                "design_scope": "single_participant_single_device_repeated_sessions",
                "scope_boundary": (
                    "Repeated prompted sessions estimate within-participant, within-device "
                    "operating scale. They do not establish population generality, "
                    "cross-device generality, or reference-device accuracy without "
                    "independent reference-backed evidence."
                ),
                "v1_readiness_criteria": dict(empirical.DEFAULT_V1_CRITERIA),
                "future_validation_scope": dict(empirical.DEFAULT_FUTURE_VALIDATION_SCOPE),
                "sessions": [],
            }
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"kind": MANIFEST_KIND, "version": 1, "sessions": []}
        if not isinstance(payload.get("sessions"), list):
            payload["sessions"] = []
        return payload

    def _session_root_and_base(self) -> tuple[Path, str, int | None, int]:
        if self.output_dir is None:
            msg = "No output directory configured for this live session"
            raise RuntimeError(msg)
        match = SESSION_ID_RE.match(self.output_dir.name)
        if match is None:
            return self.output_dir, SESSION_PREFIX, None, 3
        return (
            self.output_dir.parent,
            match.group("prefix"),
            int(match.group("number")),
            len(match.group("number")),
        )

    def _used_session_numbers(self, root: Path, prefix: str) -> set[int]:
        used: set[int] = set()
        if root.exists():
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                match = SESSION_ID_RE.match(child.name)
                if match is not None and match.group("prefix") == prefix:
                    used.add(int(match.group("number")))
        payload = self._read_manifest_payload()
        sessions = payload.get("sessions", [])
        if isinstance(sessions, list):
            for row in sessions:
                if not isinstance(row, dict):
                    continue
                session_id = row.get("session_id")
                if not isinstance(session_id, str):
                    continue
                match = SESSION_ID_RE.match(session_id)
                if match is not None and match.group("prefix") == prefix:
                    used.add(int(match.group("number")))
        return used

    def _preview_experiment_session(self) -> tuple[str, str, Path]:
        root, prefix, requested_number, width = self._session_root_and_base()
        used = self._used_session_numbers(root, prefix)
        if requested_number is not None and requested_number not in used:
            number = requested_number
        else:
            number = max(used, default=0) + 1
        session_id = f"{prefix}_{number:0{width}d}"
        return session_id, f"R{number:0{width}d}", root / session_id

    def _allocate_experiment_session(self) -> tuple[str, str, Path]:
        return self._preview_experiment_session()

    def _repo_relative_path(self, path: str | Path) -> str:
        raw = Path(path)
        resolved = raw.resolve()
        if self.empirical_manifest_path is not None:
            try:
                root = self.empirical_manifest_path.resolve().parent.parent
                return str(resolved.relative_to(root))
            except ValueError:
                pass
        return str(raw)

    def _upsert_manifest_entry(self, report_path: str, protocol_id: str) -> None:
        path = self.empirical_manifest_path
        if (
            path is None
            or self.experiment_session_id is None
            or self.experiment_replicate_id is None
        ):
            return
        payload = self._read_manifest_payload()
        sessions = payload.setdefault("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
            payload["sessions"] = sessions
        entry = {
            "session_id": self.experiment_session_id,
            "status": "available",
            "participant_id": self.experiment_participant_id,
            "device_id": self.experiment_device_id,
            "session_group": self.experiment_session_group
            or f"{self.experiment_participant_id}_{self.experiment_device_id}",
            "replicate_id": self.experiment_replicate_id,
            "condition": self.experiment_condition,
            "protocol_id": protocol_id,
            "consent_scope": self.experiment_consent_scope,
            "reference_kind": self.experiment_reference_kind,
            "report": self._repo_relative_path(report_path),
            "notes": (
                "Auto-recorded prompted diagnostic session; no reference eye-tracker, "
                "public dataset, or manual annotation unless reference_kind says otherwise."
            ),
        }
        replaced = False
        for index, row in enumerate(sessions):
            if isinstance(row, dict) and row.get("session_id") == self.experiment_session_id:
                sessions[index] = entry
                replaced = True
                break
        if not replaced:
            sessions.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        empirical.write_json_atomic(path, payload)

    def _refresh_empirical_summary(self) -> Path | None:
        manifest_path = self.empirical_manifest_path
        if manifest_path is None:
            return None
        repo_root = manifest_path.resolve().parent.parent
        summary_path = manifest_path.with_name("empirical_sessions_summary.json")
        return empirical.write_empirical_sessions_summary(
            manifest_path=manifest_path,
            summary_out=summary_path,
            repo_root=repo_root,
        )
