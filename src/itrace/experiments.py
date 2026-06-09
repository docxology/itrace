"""Guided eye-video experiment protocols and derived-session reports.

The module is deliberately pure: it operates on already-derived
``CaptureSample`` records and never stores raw camera frames. The reports are
session-specific empirical estimates from prompted webcam recordings, not
reference-device validation.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from . import io, pipeline, validation
from .calibration import AffineCalibration, calibration_error
from .capture import CaptureSample, samples_to_streams, write_capture_records_csv
from .reporting import error_session_report_dict, partial_session_report_dict, session_report_dict
from .stats.descriptive import session_statistics
from .types import GazeSample, PupilSample, PupilUnit

TRUTH_BOUNDARY = (
    "screen target prompts provide session-specific estimates only; no reference-device "
    "validation or universal webcam accuracy is claimed"
)
STORAGE_BOUNDARY = (
    "derived gaze/pupil/capture records only; raw eye video and persisted eye-crop images "
    "are not written by the default workflow"
)


@dataclass(frozen=True, slots=True)
class TargetCue:
    """One screen target or prompt interval inside a trial."""

    start_s: float
    end_s: float
    x_deg: float
    y_deg: float
    label: str
    use_for_fit: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly target cue."""
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "x_deg": self.x_deg,
            "y_deg": self.y_deg,
            "label": self.label,
            "use_for_fit": self.use_for_fit,
        }


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """A single guided recording condition."""

    trial_id: str
    kind: str
    duration_s: float
    prompt: str
    target_schedule: tuple[TargetCue, ...] = ()
    display_text: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly trial specification."""
        return {
            "trial_id": self.trial_id,
            "kind": self.kind,
            "duration_s": self.duration_s,
            "prompt": self.prompt,
            "display_text": self.display_text,
            "target_schedule": [cue.to_dict() for cue in self.target_schedule],
        }


@dataclass(frozen=True, slots=True)
class ExperimentProtocol:
    """A full guided eye-video experiment protocol."""

    protocol_id: str
    target_range_deg: float
    trials: tuple[TrialSpec, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly protocol specification."""
        return {
            "protocol_id": self.protocol_id,
            "target_range_deg": self.target_range_deg,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    def trial(self, trial_id: str) -> TrialSpec:
        """Return one trial by identifier."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        msg = f"unknown trial_id: {trial_id}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RecordedTrial:
    """Absolute sample-time window for one recorded trial."""

    trial_id: str
    started_at_s: float
    ended_at_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly recorded trial window."""
        return {
            "trial_id": self.trial_id,
            "started_at_s": self.started_at_s,
            "ended_at_s": self.ended_at_s,
        }


def _target_grid(target_range_deg: float) -> list[tuple[str, float, float]]:
    r = float(target_range_deg)
    return [
        ("center", 0.0, 0.0),
        ("upper left", -r, r),
        ("upper right", r, r),
        ("lower right", r, -r),
        ("lower left", -r, -r),
    ]


def _grid_schedule(
    *,
    duration_s: float,
    target_range_deg: float,
    hold_s: float,
) -> tuple[TargetCue, ...]:
    if hold_s <= 0.0:
        msg = "hold_s must be positive"
        raise ValueError(msg)
    cues: list[TargetCue] = []
    targets = _target_grid(target_range_deg)
    t = 0.0
    index = 0
    while t < duration_s - 1e-9:
        label, x, y = targets[index % len(targets)]
        end_s = min(duration_s, t + hold_s)
        cues.append(
            TargetCue(
                start_s=t,
                end_s=end_s,
                x_deg=x,
                y_deg=y,
                label=label,
                use_for_fit=t < duration_s / 2.0,
            )
        )
        t = end_s
        index += 1
    return tuple(cues)


def default_eye_video_protocol(
    *,
    trial_duration_s: float = 30.0,
    target_range_deg: float = 15.0,
    saccade_hold_s: float = 1.5,
) -> ExperimentProtocol:
    """Return the default fixed-gaze, reading, and corner-saccade protocol."""
    if trial_duration_s <= 0.0 or not np.isfinite(trial_duration_s):
        msg = "trial_duration_s must be positive and finite"
        raise ValueError(msg)
    if target_range_deg <= 0.0 or not np.isfinite(target_range_deg):
        msg = "target_range_deg must be positive and finite"
        raise ValueError(msg)
    fixation = TrialSpec(
        trial_id="fixed_center",
        kind="fixation",
        duration_s=float(trial_duration_s),
        prompt="Fixate the center target without intentionally moving your eyes.",
        target_schedule=(
            TargetCue(
                start_s=0.0,
                end_s=float(trial_duration_s),
                x_deg=0.0,
                y_deg=0.0,
                label="center",
            ),
        ),
    )
    reading = TrialSpec(
        trial_id="reading",
        kind="reading",
        duration_s=float(trial_duration_s),
        prompt="Read the paragraph at a natural pace while keeping your head still.",
        display_text=(
            "The verified core records gaze, saccades, and pupil proxies as derived signals. "
            "This live experiment estimates the stability of the current webcam session."
        ),
    )
    saccade = TrialSpec(
        trial_id="corner_saccades",
        kind="saccade_grid",
        duration_s=float(trial_duration_s),
        prompt="Move your eyes to each highlighted center or corner target as it appears.",
        target_schedule=_grid_schedule(
            duration_s=float(trial_duration_s),
            target_range_deg=target_range_deg,
            hold_s=saccade_hold_s,
        ),
    )
    return ExperimentProtocol(
        protocol_id="derived_eye_video_v1",
        target_range_deg=float(target_range_deg),
        trials=(fixation, reading, saccade),
    )


def protocol_from_dict(payload: Mapping[str, object]) -> ExperimentProtocol:
    """Parse a protocol dictionary written by :func:`ExperimentProtocol.to_dict`."""
    trials: list[TrialSpec] = []
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        msg = "protocol trials must be a list"
        raise ValueError(msg)
    for raw_item in raw_trials:
        if not isinstance(raw_item, Mapping):
            msg = "trial entries must be mappings"
            raise ValueError(msg)
        item = cast(Mapping[str, object], raw_item)
        cues: list[TargetCue] = []
        raw_schedule = item.get("target_schedule", [])
        if not isinstance(raw_schedule, list):
            msg = "target_schedule must be a list"
            raise ValueError(msg)
        for raw_cue_payload in raw_schedule:
            if not isinstance(raw_cue_payload, Mapping):
                msg = "target schedule entries must be mappings"
                raise ValueError(msg)
            cue_payload = cast(Mapping[str, object], raw_cue_payload)
            cues.append(
                TargetCue(
                    start_s=_float_field(cue_payload, "start_s"),
                    end_s=_float_field(cue_payload, "end_s"),
                    x_deg=_float_field(cue_payload, "x_deg"),
                    y_deg=_float_field(cue_payload, "y_deg"),
                    label=str(cue_payload["label"]),
                    use_for_fit=bool(cue_payload.get("use_for_fit", False)),
                )
            )
        trials.append(
            TrialSpec(
                trial_id=str(item["trial_id"]),
                kind=str(item["kind"]),
                duration_s=_float_field(item, "duration_s"),
                prompt=str(item["prompt"]),
                target_schedule=tuple(cues),
                display_text=str(item.get("display_text", "")),
            )
        )
    if not trials:
        msg = "protocol must contain at least one trial"
        raise ValueError(msg)
    return ExperimentProtocol(
        protocol_id=str(payload.get("protocol_id", "derived_eye_video_v1")),
        target_range_deg=_float_field(payload, "target_range_deg"),
        trials=tuple(trials),
    )


def _float_field(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        msg = f"{key} must be numeric"
        raise ValueError(msg)
    return float(value)


def recorded_trials_from_dicts(items: Sequence[Mapping[str, object]]) -> tuple[RecordedTrial, ...]:
    """Parse recorded trial windows from manifest dictionaries."""
    return tuple(
        RecordedTrial(
            trial_id=str(item["trial_id"]),
            started_at_s=_float_field(item, "started_at_s"),
            ended_at_s=_float_field(item, "ended_at_s")
            if item.get("ended_at_s") is not None
            else None,
        )
        for item in items
    )


def read_capture_records_csv(path: Path) -> list[CaptureSample]:
    """Read ``write_capture_records_csv`` output back into capture samples."""
    rows = list(csv.DictReader(path.open(newline="")))
    required = {"frame_index", "timestamp_s", "gaze_x_deg", "gaze_y_deg", "fps_estimate_hz"}
    if not rows:
        return []
    missing = sorted(required - set(rows[0]))
    if missing:
        msg = f"capture records CSV missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    samples: list[CaptureSample] = []
    for row in rows:
        timestamp = float(row["timestamp_s"])
        pupil: PupilSample | None = None
        if row.get("pupil_size") not in {None, ""}:
            unit = PupilUnit(row.get("pupil_unit") or PupilUnit.RELATIVE.value)
            pupil = PupilSample(t=timestamp, size=float(row["pupil_size"]), unit=unit)
        quality = {
            key.removeprefix("quality_"): float(value)
            for key, value in row.items()
            if key.startswith("quality_") and value not in {None, ""}
        }
        samples.append(
            CaptureSample(
                frame_index=int(row["frame_index"]),
                timestamp_s=timestamp,
                gaze=GazeSample(
                    t=timestamp, x=float(row["gaze_x_deg"]), y=float(row["gaze_y_deg"])
                ),
                pupil=pupil,
                fps_estimate_hz=float(row["fps_estimate_hz"]),
                quality=quality,
            )
        )
    return samples


def _finite_capture_samples(samples: Sequence[CaptureSample]) -> list[CaptureSample]:
    return [
        sample
        for sample in samples
        if np.isfinite(sample.timestamp_s)
        and np.isfinite(sample.gaze.x)
        and np.isfinite(sample.gaze.y)
    ]


def slice_trial_samples(
    samples: Sequence[CaptureSample],
    trial: RecordedTrial,
) -> list[CaptureSample]:
    """Return samples inside one completed recorded trial window."""
    if trial.ended_at_s is None:
        msg = "trial must have ended_at_s before slicing"
        raise ValueError(msg)
    if trial.ended_at_s < trial.started_at_s:
        msg = "trial ended before it started"
        raise ValueError(msg)
    return [
        sample for sample in samples if trial.started_at_s <= sample.timestamp_s <= trial.ended_at_s
    ]


def _drift_slope_deg_s(samples: Sequence[CaptureSample]) -> float:
    finite = _finite_capture_samples(samples)
    if len(finite) < 3:
        return 0.0
    t = np.array([sample.timestamp_s for sample in finite], dtype=np.float64)
    t = t - float(t[0])
    if float(np.ptp(t)) <= 0.0:
        return 0.0
    x = np.array([sample.gaze.x for sample in finite], dtype=np.float64)
    y = np.array([sample.gaze.y for sample in finite], dtype=np.float64)
    sx = float(np.polyfit(t, x, deg=1)[0])
    sy = float(np.polyfit(t, y, deg=1)[0])
    return float(np.hypot(sx, sy))


def _trial_summary(samples: Sequence[CaptureSample]) -> dict[str, object]:
    gaze, pupil = samples_to_streams(samples)
    duration = float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0
    if len(samples) >= 3:
        try:
            report = pipeline.analyze_session(gaze, pupil)
            report_payload = session_report_dict(report)
            statistics = session_statistics(gaze, pupil, report)
        except ValueError as exc:
            report_payload = error_session_report_dict(
                n_samples=len(samples),
                duration_s=duration,
                quality={},
                error=str(exc),
            )
            statistics = session_statistics(gaze, pupil)
    else:
        report_payload = partial_session_report_dict(
            n_samples=len(samples),
            duration_s=duration,
            quality={},
        )
        statistics = session_statistics(gaze, pupil)
    diagnostics = validation.live_recording_diagnostics(gaze, pupil, report_payload)
    return {
        "sample_count": len(samples),
        "duration_s": duration,
        "finite_gaze_fraction": diagnostics["finite_gaze_fraction"],
        "sampling_rate_hz": diagnostics["sampling_rate_hz"],
        "sampling_interval_cv": diagnostics["sampling_interval_cv"],
        "gaze_dispersion_deg": diagnostics["gaze_dispersion_deg"],
        "drift_slope_deg_s": _drift_slope_deg_s(samples),
        "pupil_valid_fraction": diagnostics["pupil_valid_fraction"],
        "pupil_dynamic_range": diagnostics["pupil_dynamic_range"],
        "n_saccades": report_payload.get("n_saccades", 0),
        "n_fixations": report_payload.get("n_fixations", 0),
        "statistics": statistics,
        "warnings": diagnostics["warnings"],
    }


def _target_observations(
    samples: Sequence[CaptureSample],
    protocol: ExperimentProtocol,
    recorded_trials: Sequence[RecordedTrial],
    *,
    settle_s: float,
    min_samples: int,
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    by_trial = {trial.trial_id: trial for trial in protocol.trials}
    for recorded in recorded_trials:
        if recorded.ended_at_s is None or recorded.trial_id not in by_trial:
            continue
        spec = by_trial[recorded.trial_id]
        for cue_index, cue in enumerate(spec.target_schedule):
            start = recorded.started_at_s + cue.start_s + settle_s
            end = min(recorded.started_at_s + cue.end_s, recorded.ended_at_s)
            window = [
                sample
                for sample in samples
                if start <= sample.timestamp_s <= end
                and np.isfinite(sample.gaze.x)
                and np.isfinite(sample.gaze.y)
            ]
            if len(window) < min_samples:
                continue
            observations.append(
                {
                    "trial_id": recorded.trial_id,
                    "cue_index": cue_index,
                    "label": cue.label,
                    "target_x_deg": cue.x_deg,
                    "target_y_deg": cue.y_deg,
                    "raw_x_deg": float(np.median([sample.gaze.x for sample in window])),
                    "raw_y_deg": float(np.median([sample.gaze.y for sample in window])),
                    "started_at_s": start,
                    "ended_at_s": end,
                    "n_samples": len(window),
                    "use_for_fit": cue.use_for_fit,
                }
            )
    return observations


def _fit_calibration_from_observations(
    observations: Sequence[Mapping[str, object]],
) -> AffineCalibration | None:
    fit_points = [obs for obs in observations if obs.get("use_for_fit")]
    if len(fit_points) < 3:
        return None
    try:
        return AffineCalibration.fit(
            [_float_field(obs, "raw_x_deg") for obs in fit_points],
            [_float_field(obs, "raw_y_deg") for obs in fit_points],
            [_float_field(obs, "target_x_deg") for obs in fit_points],
            [_float_field(obs, "target_y_deg") for obs in fit_points],
        )
    except ValueError:
        return None


def _heldout_error(
    calibration: AffineCalibration | None,
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    heldout = [obs for obs in observations if not obs.get("use_for_fit")]
    if calibration is None:
        return {
            "available": False,
            "reason": "not enough non-degenerate fit targets",
            "n_points": len(heldout),
        }
    if not heldout:
        return {"available": False, "reason": "no held-out target observations", "n_points": 0}
    err = calibration_error(
        calibration,
        [_float_field(obs, "raw_x_deg") for obs in heldout],
        [_float_field(obs, "raw_y_deg") for obs in heldout],
        [_float_field(obs, "target_x_deg") for obs in heldout],
        [_float_field(obs, "target_y_deg") for obs in heldout],
    )
    return {"available": True, **err}


def _target_latency(
    samples: Sequence[CaptureSample],
    protocol: ExperimentProtocol,
    recorded_trials: Sequence[RecordedTrial],
    calibration: AffineCalibration | None,
    *,
    threshold_deg: float = 4.0,
    settle_s: float = 0.1,
) -> dict[str, object]:
    if calibration is None:
        return {"available": False, "reason": "calibration unavailable"}
    latencies: list[float] = []
    by_trial = {trial.trial_id: trial for trial in protocol.trials}
    for recorded in recorded_trials:
        if recorded.ended_at_s is None or recorded.trial_id not in by_trial:
            continue
        spec = by_trial[recorded.trial_id]
        for cue in spec.target_schedule:
            start = recorded.started_at_s + cue.start_s + settle_s
            end = min(recorded.started_at_s + cue.end_s, recorded.ended_at_s)
            window = [
                sample
                for sample in samples
                if start <= sample.timestamp_s <= end
                and np.isfinite(sample.gaze.x)
                and np.isfinite(sample.gaze.y)
            ]
            if not window:
                continue
            cx, cy = calibration.apply(
                [sample.gaze.x for sample in window],
                [sample.gaze.y for sample in window],
            )
            distance = np.hypot(cx - cue.x_deg, cy - cue.y_deg)
            inside = np.flatnonzero(distance <= threshold_deg)
            if inside.size:
                latencies.append(float(window[int(inside[0])].timestamp_s - start))
    if not latencies:
        return {"available": False, "reason": "no target acquisitions within threshold"}
    return {
        "available": True,
        "threshold_deg": threshold_deg,
        "n_targets": len(latencies),
        "median_latency_s": float(np.median(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
    }


def _default_recorded_trials(
    samples: Sequence[CaptureSample],
    protocol: ExperimentProtocol,
) -> tuple[RecordedTrial, ...]:
    if not samples:
        return ()
    start = float(samples[0].timestamp_s)
    trials: list[RecordedTrial] = []
    cursor = start
    for trial in protocol.trials:
        end = cursor + trial.duration_s
        trials.append(RecordedTrial(trial_id=trial.trial_id, started_at_s=cursor, ended_at_s=end))
        cursor = end
    return tuple(trials)


def experiment_report(
    samples: Sequence[CaptureSample],
    protocol: ExperimentProtocol | None = None,
    recorded_trials: Sequence[RecordedTrial] | None = None,
    *,
    settle_s: float = 0.25,
    min_target_samples: int = 3,
) -> dict[str, object]:
    """Build a derived empirical report from guided eye-video samples."""
    protocol = protocol or default_eye_video_protocol()
    recorded = (
        tuple(recorded_trials)
        if recorded_trials is not None
        else _default_recorded_trials(samples, protocol)
    )
    completed = [trial for trial in recorded if trial.ended_at_s is not None]
    trial_payload: dict[str, object] = {}
    for trial in completed:
        trial_payload[trial.trial_id] = _trial_summary(slice_trial_samples(samples, trial))
    observations = _target_observations(
        samples,
        protocol,
        completed,
        settle_s=settle_s,
        min_samples=min_target_samples,
    )
    calibration = _fit_calibration_from_observations(observations)
    calibration_payload = calibration.to_dict() if calibration is not None else None
    return {
        "kind": "derived_eye_video_experiment",
        "truth_boundary": TRUTH_BOUNDARY,
        "storage_boundary": STORAGE_BOUNDARY,
        "protocol": protocol.to_dict(),
        "recorded_trials": [trial.to_dict() for trial in recorded],
        "sample_count": len(samples),
        "completed_trial_count": len(completed),
        "trials": trial_payload,
        "target_observations": observations,
        "calibration": calibration_payload,
        "heldout_target_error": _heldout_error(calibration, observations),
        "target_acquisition_latency_s": _target_latency(
            samples,
            protocol,
            completed,
            calibration,
        ),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "trial"


def write_target_schedule_csv(protocol: ExperimentProtocol, out: Path) -> Path:
    """Write the protocol target schedule as a CSV file."""
    fields = [
        "trial_id",
        "cue_index",
        "start_s",
        "end_s",
        "x_deg",
        "y_deg",
        "label",
        "use_for_fit",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for trial in protocol.trials:
            for cue_index, cue in enumerate(trial.target_schedule):
                writer.writerow(
                    {
                        "trial_id": trial.trial_id,
                        "cue_index": cue_index,
                        "start_s": cue.start_s,
                        "end_s": cue.end_s,
                        "x_deg": cue.x_deg,
                        "y_deg": cue.y_deg,
                        "label": cue.label,
                        "use_for_fit": cue.use_for_fit,
                    }
                )
    return out


def write_experiment_bundle(
    samples: Sequence[CaptureSample],
    protocol: ExperimentProtocol,
    recorded_trials: Sequence[RecordedTrial],
    output_dir: Path,
) -> dict[str, str]:
    """Write derived experiment CSV/JSON artifacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "derived_eye_video_experiment_manifest",
        "truth_boundary": TRUTH_BOUNDARY,
        "storage_boundary": STORAGE_BOUNDARY,
        "protocol": protocol.to_dict(),
        "recorded_trials": [trial.to_dict() for trial in recorded_trials],
    }
    manifest_path = output_dir / "experiment_manifest.json"
    report_path = output_dir / "experiment_report.json"
    schedule_path = output_dir / "target_schedule.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(experiment_report(samples, protocol, recorded_trials), indent=2),
        encoding="utf-8",
    )
    write_target_schedule_csv(protocol, schedule_path)
    paths = {
        "manifest_json": str(manifest_path),
        "report_json": str(report_path),
        "target_schedule_csv": str(schedule_path),
    }
    for trial in recorded_trials:
        if trial.ended_at_s is None:
            continue
        trial_samples = slice_trial_samples(samples, trial)
        gaze, pupil = samples_to_streams(trial_samples)
        stem = f"trial_{_slug(trial.trial_id)}"
        gaze_path = output_dir / f"{stem}_gaze.csv"
        pupil_path = output_dir / f"{stem}_pupil.csv"
        records_path = output_dir / f"{stem}_capture_records.csv"
        io.write_gaze_csv(gaze, gaze_path)
        io.write_pupil_csv(pupil, pupil_path)
        write_capture_records_csv(trial_samples, records_path)
        paths[f"{trial.trial_id}_gaze_csv"] = str(gaze_path)
        paths[f"{trial.trial_id}_pupil_csv"] = str(pupil_path)
        paths[f"{trial.trial_id}_capture_records_csv"] = str(records_path)
    return paths


def load_experiment_manifest(path: Path) -> tuple[ExperimentProtocol, tuple[RecordedTrial, ...]]:
    """Load protocol and trial windows from an experiment manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        msg = "experiment manifest must be a JSON object"
        raise ValueError(msg)
    protocol_payload = payload.get("protocol")
    if not isinstance(protocol_payload, Mapping):
        msg = "experiment manifest missing protocol object"
        raise ValueError(msg)
    trial_payload = payload.get("recorded_trials", [])
    if not isinstance(trial_payload, list):
        msg = "recorded_trials must be a list"
        raise ValueError(msg)
    return protocol_from_dict(protocol_payload), recorded_trials_from_dicts(trial_payload)
