"""Empirical-session manifest validation and release-readiness summaries.

The functions here intentionally operate on derived report artifacts and
metadata only. They do not read raw video, and they keep prompt-only empirical
sessions separate from reference-device or public-dataset validation evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from .experiments import STORAGE_BOUNDARY, TRUTH_BOUNDARY

MANIFEST_KIND = "itrace_empirical_sessions_manifest"
SUMMARY_KIND = "itrace_empirical_sessions_summary"
SESSION_STATUSES = {"planned", "available", "excluded"}
REFERENCE_KINDS = {"none", "reference_device", "public_dataset", "manual_annotation", "mixed"}
MANUAL_ANNOTATION_ARTIFACT_KIND = "itrace_manual_annotation_evidence"
MANUAL_ANNOTATION_QUALITIES = {"usable", "exclude", "uncertain"}
MANUAL_ANNOTATION_TARGET_HITS = {"yes", "no", "uncertain"}
RAW_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}

DEFAULT_V1_CRITERIA: dict[str, object] = {
    "min_available_sessions": 5,
    "min_replicates": 5,
    "min_participants": 1,
    "min_devices": 1,
    "min_conditions": 2,
    "requires_reference_evidence": False,
}

DEFAULT_FUTURE_VALIDATION_SCOPE: dict[str, object] = {
    "purpose": (
        "Future validation expansion beyond the five-session diagnostic v1; not "
        "required for current diagnostic readiness."
    ),
    "target_available_sessions": 12,
    "target_replicates": 12,
    "target_conditions": 3,
    "condition_targets": [
        "indoor_office_daylight",
        "indoor_office_dim",
        "indoor_office_backlit",
    ],
    "requires_reference_evidence": True,
    "reference_lanes": ["manual_annotation", "public_dataset", "reference_device"],
}

DEFAULT_DESIGN_SCOPE = "single_participant_single_device_repeated_sessions"
DEFAULT_SCOPE_BOUNDARY = (
    "Repeated prompted sessions estimate within-participant, within-device operating "
    "scale. They do not establish population generality, cross-device generality, or "
    "reference-device accuracy without independent reference-backed evidence."
)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> Path:
    """Write a JSON object via same-directory replace to avoid torn ledgers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _criteria_value(criteria: Mapping[str, object], key: str) -> int:
    value = criteria.get(key, DEFAULT_V1_CRITERIA[key])
    number = _number(value)
    if number is not None:
        return max(int(number), 0)
    default_number = _number(DEFAULT_V1_CRITERIA[key])
    if default_number is None:
        raise KeyError(f"default v1 criterion is not numeric: {key}")
    return int(default_number)


def _criteria_flag(criteria: Mapping[str, object], key: str) -> bool:
    value = criteria.get(key, DEFAULT_V1_CRITERIA[key])
    return bool(value)


def _repo_relative_or_resolved(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _trial_summaries(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    trials = report.get("trials", {})
    if not isinstance(trials, Mapping):
        return []
    return [value for value in trials.values() if isinstance(value, Mapping)]


def repo_relative_path(value: str, *, repo_root: Path, field: str) -> Path:
    """Return a safe repo-relative path or raise ``ValueError``.

    Empirical manifests should point to derived JSON/CSV artifacts in this repo.
    Absolute paths, parent traversal, and raw video suffixes are rejected so a
    summary cannot quietly certify private raw capture material.
    """
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be repo-relative, not absolute: {value}")
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain parent traversal: {value}")
    if path.suffix.lower() in RAW_VIDEO_SUFFIXES:
        raise ValueError(f"{field} points to raw video, not a derived artifact: {value}")
    root = repo_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} must resolve inside the repository: {value}")
    return resolved


def validate_reference_artifact(
    session: Mapping[str, object],
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Validate the derived artifact that makes a reference lane count.

    A non-``none`` reference kind is an intent marker until this artifact is
    present and valid. The validator accepts derived JSON artifacts only and
    explicitly rejects raw-video paths so privacy-sensitive media cannot become
    the readiness oracle by accident.
    """
    errors: list[str] = []
    warnings: list[str] = []
    reference_kind = _nonempty_string(session.get("reference_kind")) or "none"
    if reference_kind == "none":
        return {
            "required": False,
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }
    if reference_kind not in REFERENCE_KINDS:
        errors.append(f"reference_kind must be one of {sorted(REFERENCE_KINDS)}")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}

    artifact_value = _nonempty_string(session.get("reference_artifact"))
    if artifact_value is None:
        errors.append("reference_artifact is required before reference evidence counts")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}
    try:
        artifact_path = repo_relative_path(
            artifact_value,
            repo_root=repo_root,
            field="reference_artifact",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}
    if artifact_path.suffix.lower() != ".json":
        errors.append(f"reference_artifact must point to derived JSON evidence: {artifact_value}")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}
    if not artifact_path.exists():
        errors.append(f"reference_artifact does not exist: {artifact_value}")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"reference_artifact is invalid JSON: {exc}")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}
    if not isinstance(artifact, Mapping):
        errors.append("reference_artifact must contain a JSON object")
        return {"required": True, "valid": False, "errors": errors, "warnings": warnings}

    artifact_kind = _nonempty_string(artifact.get("kind"))
    if artifact_kind is None:
        errors.append("reference_artifact.kind must be a nonempty string")
    if _number(artifact.get("version")) is None:
        errors.append("reference_artifact.version must be numeric")
    session_id = _nonempty_string(session.get("session_id"))
    artifact_session_id = _nonempty_string(artifact.get("session_id"))
    if artifact_session_id is None:
        errors.append("reference_artifact.session_id must be a nonempty string")
    elif session_id is not None and artifact_session_id != session_id:
        errors.append(
            f"reference_artifact.session_id must match the manifest session_id {session_id!r}"
        )

    if reference_kind == "manual_annotation":
        if artifact_kind != MANUAL_ANNOTATION_ARTIFACT_KIND:
            errors.append(
                "manual_annotation reference_artifact.kind must be "
                f"{MANUAL_ANNOTATION_ARTIFACT_KIND!r}"
            )
        for field in (
            "source_report",
            "source_records",
            "annotation_scope",
            "annotator_id",
            "created_at",
        ):
            if _nonempty_string(artifact.get(field)) is None:
                errors.append(f"reference_artifact.{field} must be a nonempty string")

        report_value = _nonempty_string(session.get("report"))
        source_report = _nonempty_string(artifact.get("source_report"))
        if source_report is not None:
            try:
                source_report_path = repo_relative_path(
                    source_report,
                    repo_root=repo_root,
                    field="reference_artifact.source_report",
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if report_value is not None and source_report != report_value:
                    errors.append("reference_artifact.source_report must match session.report")
                if not source_report_path.exists():
                    errors.append(
                        f"reference_artifact.source_report does not exist: {source_report}"
                    )
        source_records = _nonempty_string(artifact.get("source_records"))
        if source_records is not None:
            try:
                source_records_path = repo_relative_path(
                    source_records,
                    repo_root=repo_root,
                    field="reference_artifact.source_records",
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if not source_records_path.exists():
                    errors.append(
                        f"reference_artifact.source_records does not exist: {source_records}"
                    )

        annotations = artifact.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            errors.append("reference_artifact.annotations must be a nonempty list")
        elif not errors:
            for index, item in enumerate(annotations):
                if not isinstance(item, Mapping):
                    errors.append(f"reference_artifact.annotations[{index}] must be an object")
                    continue
                for field in ("trial_id", "target_label", "quality", "target_hit"):
                    if _nonempty_string(item.get(field)) is None:
                        errors.append(
                            f"reference_artifact.annotations[{index}].{field} "
                            "must be a nonempty string"
                        )
                quality = _nonempty_string(item.get("quality"))
                if quality is not None and quality not in MANUAL_ANNOTATION_QUALITIES:
                    errors.append(
                        f"reference_artifact.annotations[{index}].quality must be one of "
                        f"{sorted(MANUAL_ANNOTATION_QUALITIES)}"
                    )
                target_hit = _nonempty_string(item.get("target_hit"))
                if target_hit is not None and target_hit not in MANUAL_ANNOTATION_TARGET_HITS:
                    errors.append(
                        f"reference_artifact.annotations[{index}].target_hit must be one of "
                        f"{sorted(MANUAL_ANNOTATION_TARGET_HITS)}"
                    )
                start_s = _number(item.get("start_s"))
                end_s = _number(item.get("end_s"))
                if start_s is None:
                    errors.append(
                        f"reference_artifact.annotations[{index}].start_s must be numeric"
                    )
                if end_s is None:
                    errors.append(f"reference_artifact.annotations[{index}].end_s must be numeric")
                if start_s is not None and end_s is not None and end_s <= start_s:
                    errors.append(
                        f"reference_artifact.annotations[{index}].end_s must exceed start_s"
                    )

    return {"required": True, "valid": not errors, "errors": errors, "warnings": warnings}


def validate_empirical_manifest(
    payload: Mapping[str, object],
    *,
    repo_root: Path,
    require_existing_reports: bool = True,
) -> dict[str, object]:
    """Validate an empirical-session manifest and return errors/warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("kind") != MANIFEST_KIND:
        errors.append(f"kind must be {MANIFEST_KIND!r}")
    if _number(payload.get("version")) is None:
        errors.append("version must be numeric")

    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        errors.append("sessions must be a list")
        sessions = []

    seen_ids: set[str] = set()
    seen_replicates: set[tuple[str, str]] = set()
    for index, session in enumerate(sessions):
        prefix = f"sessions[{index}]"
        if not isinstance(session, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        session_id = _nonempty_string(session.get("session_id"))
        if session_id is None:
            errors.append(f"{prefix}.session_id must be a nonempty string")
        elif session_id in seen_ids:
            errors.append(f"{prefix}.session_id duplicates {session_id!r}")
        else:
            seen_ids.add(session_id)

        status = _nonempty_string(session.get("status")) or "available"
        if status not in SESSION_STATUSES:
            errors.append(f"{prefix}.status must be one of {sorted(SESSION_STATUSES)}")

        for field in (
            "participant_id",
            "device_id",
            "session_group",
            "replicate_id",
            "condition",
            "protocol_id",
            "consent_scope",
            "reference_kind",
        ):
            if _nonempty_string(session.get(field)) is None:
                errors.append(f"{prefix}.{field} must be a nonempty string")

        session_group = _nonempty_string(session.get("session_group"))
        replicate_id = _nonempty_string(session.get("replicate_id"))
        if session_group is not None and replicate_id is not None:
            replicate_key = (session_group, replicate_id)
            if replicate_key in seen_replicates:
                errors.append(
                    f"{prefix}.replicate_id duplicates {replicate_id!r} "
                    f"within session_group {session_group!r}"
                )
            else:
                seen_replicates.add(replicate_key)

        reference_kind = _nonempty_string(session.get("reference_kind"))
        if reference_kind is not None and reference_kind not in REFERENCE_KINDS:
            errors.append(f"{prefix}.reference_kind must be one of {sorted(REFERENCE_KINDS)}")
        elif reference_kind is not None and reference_kind != "none":
            reference_artifact = _nonempty_string(session.get("reference_artifact"))
            if reference_artifact is None:
                warnings.append(
                    f"{prefix}.reference_artifact is required before "
                    f"{reference_kind!r} evidence counts toward readiness"
                )
            else:
                artifact_status = validate_reference_artifact(session, repo_root=repo_root)
                if not artifact_status["valid"]:
                    artifact_errors = artifact_status.get("errors", [])
                    error_items = (
                        artifact_errors if isinstance(artifact_errors, list) else [artifact_errors]
                    )
                    errors.extend(f"{prefix}.{error}" for error in error_items)

        report = _nonempty_string(session.get("report"))
        if status == "available" and report is None:
            errors.append(f"{prefix}.report is required for available sessions")
        if report is None:
            continue

        try:
            report_path = repo_relative_path(report, repo_root=repo_root, field=f"{prefix}.report")
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if status == "available" and report_path.suffix.lower() != ".json":
            errors.append(f"{prefix}.report must point to a derived JSON report: {report}")
            continue
        if require_existing_reports and status == "available" and not report_path.exists():
            errors.append(f"{prefix}.report does not exist: {report}")
            continue
        if report_path.exists() and report_path.suffix.lower() == ".json":
            try:
                report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{prefix}.report is invalid JSON: {exc}")
                continue
            if not isinstance(report_payload, Mapping):
                errors.append(f"{prefix}.report must contain a JSON object")
                continue
            storage_boundary = str(report_payload.get("storage_boundary", ""))
            if "raw eye video" not in storage_boundary:
                warnings.append(f"{prefix}.report lacks explicit raw-video storage boundary")
            if "reference-device validation" not in str(report_payload.get("truth_boundary", "")):
                warnings.append(f"{prefix}.report lacks explicit reference-device truth boundary")

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def _weighted_mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        value = _number(row.get(key))
        weight = _number(row.get("sample_count"))
        if value is not None and weight is not None and weight > 0.0:
            pairs.append((value, weight))
    total = sum(weight for _value, weight in pairs)
    if total <= 0.0:
        return None
    return sum(value * weight for value, weight in pairs) / total


def _weighted_pair_mean(items: Sequence[tuple[float | None, float]]) -> float | None:
    pairs = [(value, weight) for value, weight in items if value is not None and weight > 0.0]
    total = sum(weight for _value, weight in pairs)
    if total <= 0.0:
        return None
    return sum(float(value) * weight for value, weight in pairs if value is not None) / total


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _session_replicate_sort_key(value: object) -> tuple[int, str]:
    text = str(value or "")
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 0, text)


def empirical_metrics_from_report(
    report: Mapping[str, object],
    *,
    source_report: Path,
    repo_root: Path,
    pilot_id: str = "local_pilot_001",
) -> dict[str, object]:
    """Extract derived-session metrics without importing figure/rendering code."""
    trials = _trial_summaries(report)
    weights = [
        float(_number(trial.get("sample_count")) or 0.0)
        for trial in trials
        if isinstance(trial, Mapping)
    ]
    sample_count = int(_number(report.get("sample_count")) or sum(weights))
    completed = int(_number(report.get("completed_trial_count")) or 0)
    finite_gaze = _weighted_pair_mean(
        [
            (_number(trial.get("finite_gaze_fraction")), weight)
            for trial, weight in zip(trials, weights, strict=False)
        ]
    )
    sampling_hz = _weighted_pair_mean(
        [
            (_number(trial.get("sampling_rate_hz")), weight)
            for trial, weight in zip(trials, weights, strict=False)
        ]
    )
    sampling_cv = _weighted_pair_mean(
        [
            (_number(trial.get("sampling_interval_cv")), weight)
            for trial, weight in zip(trials, weights, strict=False)
        ]
    )
    pupil_valid = _weighted_pair_mean(
        [
            (_number(trial.get("pupil_valid_fraction")), weight)
            for trial, weight in zip(trials, weights, strict=False)
        ]
    )
    drift_values = [_number(trial.get("drift_slope_deg_s")) for trial in trials]
    drift = max((value for value in drift_values if value is not None), default=None)

    heldout = report.get("heldout_target_error", {})
    if not isinstance(heldout, Mapping):
        heldout = {"available": False, "reason": "held-out target block missing"}
    latency = report.get("target_acquisition_latency_s", {})
    if not isinstance(latency, Mapping):
        latency = {"available": False, "reason": "latency block missing"}

    return {
        "kind": "empirical_pilot_metrics",
        "pilot_id": pilot_id,
        "available": sample_count > 0 and completed > 0,
        "source_report": _repo_relative_or_resolved(source_report, repo_root),
        "sample_count": sample_count,
        "completed_trial_count": completed,
        "finite_gaze_fraction": finite_gaze,
        "sampling_rate_hz": sampling_hz,
        "sampling_interval_cv": sampling_cv,
        "max_drift_deg_s": drift,
        "pupil_valid_fraction": pupil_valid,
        "heldout_target_error": dict(heldout),
        "target_acquisition_latency_s": dict(latency),
        "truth_boundary": str(report.get("truth_boundary", TRUTH_BOUNDARY)),
        "storage_boundary": str(report.get("storage_boundary", STORAGE_BOUNDARY)),
    }


def empirical_session_row(
    session: Mapping[str, object],
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Return one manifest row augmented with derived report metrics."""
    status = str(session.get("status", "available"))
    row: dict[str, object] = {
        "session_id": str(session.get("session_id", "")),
        "status": status,
        "participant_id": str(session.get("participant_id", "")),
        "device_id": str(session.get("device_id", "")),
        "session_group": str(session.get("session_group", "")),
        "replicate_id": str(session.get("replicate_id", "")),
        "condition": str(session.get("condition", "")),
        "protocol_id": str(session.get("protocol_id", "")),
        "consent_scope": str(session.get("consent_scope", "")),
        "reference_kind": str(session.get("reference_kind", "none")),
        "report": session.get("report"),
        "reference_artifact": session.get("reference_artifact"),
        "available": False,
        "notes": str(session.get("notes", "")),
    }
    reference_status = validate_reference_artifact(session, repo_root=repo_root)
    row["reference_evidence_valid"] = bool(reference_status.get("valid"))
    row["reference_evidence_errors"] = reference_status.get("errors", [])
    row["reference_evidence_warnings"] = reference_status.get("warnings", [])
    if status != "available":
        return row

    report_value = session.get("report")
    if not isinstance(report_value, str):
        return row
    report_path = repo_relative_path(report_value, repo_root=repo_root, field="report")
    report = load_json_object(report_path)
    metrics = empirical_metrics_from_report(
        report,
        source_report=report_path,
        repo_root=repo_root,
        pilot_id=str(session.get("session_id", "session")),
    )
    heldout = metrics.get("heldout_target_error", {})
    latency = metrics.get("target_acquisition_latency_s", {})
    row.update(
        {
            "available": bool(metrics.get("available")),
            "source_report": metrics.get("source_report"),
            "sample_count": metrics.get("sample_count"),
            "completed_trial_count": metrics.get("completed_trial_count"),
            "finite_gaze_fraction": metrics.get("finite_gaze_fraction"),
            "sampling_rate_hz": metrics.get("sampling_rate_hz"),
            "sampling_interval_cv": metrics.get("sampling_interval_cv"),
            "pupil_valid_fraction": metrics.get("pupil_valid_fraction"),
            "max_drift_deg_s": metrics.get("max_drift_deg_s"),
            "heldout_rms_error_deg": (
                heldout.get("rms_error_deg") if isinstance(heldout, Mapping) else None
            ),
            "target_latency_median_s": (
                latency.get("median_latency_s") if isinstance(latency, Mapping) else None
            ),
        }
    )
    return row


def _remaining_plan(
    *,
    available_count: int,
    replicate_count: int,
    participant_count: int,
    device_count: int,
    condition_count: int,
    reference_evidence_count: int,
    criteria: Mapping[str, object],
) -> dict[str, object]:
    required_sessions = _criteria_value(criteria, "min_available_sessions")
    required_replicates = _criteria_value(criteria, "min_replicates")
    required_participants = _criteria_value(criteria, "min_participants")
    required_devices = _criteria_value(criteria, "min_devices")
    required_conditions = _criteria_value(criteria, "min_conditions")
    requires_reference = _criteria_flag(criteria, "requires_reference_evidence")
    remaining_sessions = max(required_sessions - available_count, 0)
    remaining_replicates = max(required_replicates - replicate_count, 0)
    remaining_participants = max(required_participants - participant_count, 0)
    remaining_devices = max(required_devices - device_count, 0)
    remaining_conditions = max(required_conditions - condition_count, 0)
    remaining_reference = 1 if requires_reference and reference_evidence_count == 0 else 0
    count_only = max(
        remaining_sessions,
        remaining_replicates,
        remaining_participants,
        remaining_devices,
        remaining_conditions,
    )
    all_criteria = max(count_only, remaining_reference)
    return {
        "available_sessions_remaining": remaining_sessions,
        "replicates_remaining": remaining_replicates,
        "participants_remaining": remaining_participants,
        "devices_remaining": remaining_devices,
        "conditions_remaining": remaining_conditions,
        "reference_evidence_remaining": remaining_reference,
        "minimum_additional_sessions_to_meet_count_criteria": count_only,
        "minimum_additional_sessions_to_meet_all_criteria_if_reference_backed": all_criteria,
        "additional_prompt_only_sessions_still_leave_reference_blocker": (remaining_reference > 0),
        "next_session_requirements": (
            "At least one added session must be reference-device, public-dataset, "
            "or manual-annotation backed."
            if remaining_reference
            else "No additional reference-backed row is required by the current criteria."
        ),
        "condition_guidance": (
            f"Use at least {remaining_conditions} new condition(s) among the next sessions."
            if remaining_conditions
            else "Condition coverage meets the current criterion."
        ),
    }


def _future_scope_int(scope: Mapping[str, object], key: str) -> int:
    default = _number(DEFAULT_FUTURE_VALIDATION_SCOPE[key])
    value = _number(scope.get(key))
    if value is None:
        value = default
    if value is None:
        raise KeyError(f"future validation scope field is not numeric: {key}")
    return max(int(value), 0)


def _future_scope_flag(scope: Mapping[str, object], key: str) -> bool:
    return bool(scope.get(key, DEFAULT_FUTURE_VALIDATION_SCOPE[key]))


def _future_scope_list(scope: Mapping[str, object], key: str) -> list[str]:
    value = scope.get(key, DEFAULT_FUTURE_VALIDATION_SCOPE.get(key, []))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _future_validation_scope_summary(
    *,
    manifest: Mapping[str, object],
    available_count: int,
    replicate_count: int,
    condition_names: set[str],
    reference_evidence_count: int,
) -> dict[str, object]:
    """Return future validation targets without making them current v1 blockers."""
    scope_obj = manifest.get("future_validation_scope", {})
    scope = dict(DEFAULT_FUTURE_VALIDATION_SCOPE)
    if isinstance(scope_obj, Mapping):
        scope.update(scope_obj)

    target_sessions = _future_scope_int(scope, "target_available_sessions")
    target_replicates = _future_scope_int(scope, "target_replicates")
    target_conditions = _future_scope_int(scope, "target_conditions")
    reference_required = _future_scope_flag(scope, "requires_reference_evidence")
    condition_targets = _future_scope_list(scope, "condition_targets")
    missing_conditions = [
        condition for condition in condition_targets if condition not in condition_names
    ]
    remaining_conditions = max(target_conditions - len(condition_names), len(missing_conditions))
    return {
        **scope,
        "target_available_sessions": target_sessions,
        "target_replicates": target_replicates,
        "target_conditions": target_conditions,
        "requires_reference_evidence": reference_required,
        "condition_targets": condition_targets,
        "reference_lanes": _future_scope_list(scope, "reference_lanes"),
        "available_sessions_remaining": max(target_sessions - available_count, 0),
        "replicates_remaining": max(target_replicates - replicate_count, 0),
        "conditions_remaining": max(remaining_conditions, 0),
        "missing_conditions": missing_conditions,
        "reference_evidence_remaining": (
            1 if reference_required and reference_evidence_count == 0 else 0
        ),
        "status": "future_scope",
    }


def _readiness_interpretation(
    *,
    blockers: Sequence[str],
    criteria: Mapping[str, object],
) -> str:
    if blockers:
        return "diagnostic-only until the declared v1 blockers are closed"
    if _criteria_flag(criteria, "requires_reference_evidence"):
        return "reference-backed evidence meets the declared empirical scope"
    return (
        "five-session diagnostic v1 criteria met; reference-backed device validation "
        "remains future scope"
    )


def _summary_tokens(summary: Mapping[str, object]) -> dict[str, str]:
    readiness = summary.get("v1_readiness", {})
    plan = readiness.get("replicate_plan", {}) if isinstance(readiness, Mapping) else {}
    future_scope = summary.get("future_validation_scope", {})
    if not isinstance(future_scope, Mapping):
        future_scope = {}
    all_min = plan.get("minimum_additional_sessions_to_meet_all_criteria_if_reference_backed")
    count_min = plan.get("minimum_additional_sessions_to_meet_count_criteria")
    remaining_conditions = plan.get("conditions_remaining")
    reference_remaining = plan.get("reference_evidence_remaining")
    reference_text = (
        "at least one reference-backed or manual-annotation row still required"
        if reference_remaining
        else "reference-evidence criterion currently satisfied or disabled"
    )
    available = int(_number(summary.get("available_session_count")) or 0)
    replicates = int(_number(summary.get("replicate_count")) or 0)
    conditions = int(_number(summary.get("condition_count")) or 0)
    reference_rows = int(_number(summary.get("reference_evidence_count")) or 0)
    session_label = "session" if available == 1 else "sessions"
    replicate_label = "replicate ID" if replicates == 1 else "replicate IDs"
    condition_label = "condition" if conditions == 1 else "conditions"
    reference_label = "reference-backed row" if reference_rows == 1 else "reference-backed rows"
    missing_conditions = future_scope.get("missing_conditions", [])
    if not isinstance(missing_conditions, Sequence) or isinstance(
        missing_conditions,
        (str, bytes),
    ):
        missing_conditions = []
    missing_condition_text = ", ".join(str(item) for item in missing_conditions) or "none"
    future_conditions = int(_number(future_scope.get("conditions_remaining")) or 0)
    future_condition_label = "condition" if future_conditions == 1 else "conditions"
    future_reference = int(_number(future_scope.get("reference_evidence_remaining")) or 0)
    future_reference_text = (
        "validated reference evidence still required"
        if future_reference
        else "validated reference evidence already present or not required"
    )
    return {
        "EMPIRICAL_SESSIONS_READINESS": (
            "ready for diagnostic v1"
            if isinstance(readiness, Mapping) and readiness.get("ready")
            else "not ready for diagnostic v1"
        ),
        "EMPIRICAL_SESSIONS_STATUS": (
            f"{available} available {session_label}, "
            f"{replicates} {replicate_label}, "
            f"{conditions} {condition_label}, "
            f"{reference_rows} {reference_label}"
        ),
        "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL": str(all_min),
        "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_COUNT_ONLY": str(count_min),
        "EMPIRICAL_SESSIONS_CONDITIONS_REMAINING": str(remaining_conditions),
        "EMPIRICAL_SESSIONS_REFERENCE_REQUIREMENT": reference_text,
        "EMPIRICAL_SESSIONS_FUTURE_SCOPE": (
            f"{future_scope.get('available_sessions_remaining', 'unavailable')} future "
            f"session exports, {future_conditions} future {future_condition_label} "
            f"(missing: {missing_condition_text}), and {future_reference_text}"
        ),
    }


def build_empirical_sessions_summary(
    *,
    manifest: Mapping[str, object],
    session_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the aggregate summary used by docs and figures."""
    criteria_obj = manifest.get("v1_readiness_criteria", {})
    criteria = criteria_obj if isinstance(criteria_obj, Mapping) else {}
    rows = [dict(row) for row in session_rows]
    available = [row for row in rows if row.get("status") == "available" and row.get("available")]

    participants = {
        str(row.get("participant_id")) for row in available if row.get("participant_id")
    }
    devices = {str(row.get("device_id")) for row in available if row.get("device_id")}
    session_groups = {
        str(row.get("session_group")) for row in available if row.get("session_group")
    }
    replicates = {
        f"{row.get('session_group', '')}:{row.get('replicate_id', row.get('session_id', ''))}"
        for row in available
        if row.get("replicate_id") or row.get("session_id")
    }
    conditions = {str(row.get("condition")) for row in available if row.get("condition")}
    reference_candidates = [
        row for row in available if str(row.get("reference_kind", "none")) != "none"
    ]
    reference_rows = [
        row for row in reference_candidates if row.get("reference_evidence_valid") is True
    ]
    reference_evidence_issues = [
        {
            "session_id": row.get("session_id"),
            "reference_kind": row.get("reference_kind"),
            "reference_artifact": row.get("reference_artifact"),
            "errors": row.get("reference_evidence_errors", []),
            "warnings": row.get("reference_evidence_warnings", []),
        }
        for row in reference_candidates
        if row.get("reference_evidence_valid") is not True
    ]

    blockers: list[str] = []
    required_sessions = _criteria_value(criteria, "min_available_sessions")
    required_replicates = _criteria_value(criteria, "min_replicates")
    required_participants = _criteria_value(criteria, "min_participants")
    required_devices = _criteria_value(criteria, "min_devices")
    required_conditions = _criteria_value(criteria, "min_conditions")
    if len(available) < required_sessions:
        blockers.append(f"available sessions {len(available)}/{required_sessions}")
    if len(replicates) < required_replicates:
        blockers.append(f"replicates {len(replicates)}/{required_replicates}")
    if len(participants) < required_participants:
        blockers.append(f"participants {len(participants)}/{required_participants}")
    if len(devices) < required_devices:
        blockers.append(f"devices {len(devices)}/{required_devices}")
    if len(conditions) < required_conditions:
        blockers.append(f"conditions {len(conditions)}/{required_conditions}")
    if _criteria_flag(criteria, "requires_reference_evidence") and not reference_rows:
        blockers.append("no reference-device, public-dataset, or manual-annotation evidence")

    future_validation_scope = _future_validation_scope_summary(
        manifest=manifest,
        available_count=len(available),
        replicate_count=len(replicates),
        condition_names=conditions,
        reference_evidence_count=len(reference_rows),
    )
    heldout_values = [
        value
        for value in (_number(row.get("heldout_rms_error_deg")) for row in available)
        if value is not None
    ]
    summary = {
        "kind": SUMMARY_KIND,
        "version": 1,
        "source_manifest": str(
            manifest.get("source_manifest", "docs/empirical_sessions_manifest.json")
        ),
        "truth_boundary": str(manifest.get("truth_boundary", TRUTH_BOUNDARY)),
        "storage_boundary": str(manifest.get("storage_boundary", STORAGE_BOUNDARY)),
        "design_scope": str(manifest.get("design_scope", DEFAULT_DESIGN_SCOPE)),
        "scope_boundary": str(manifest.get("scope_boundary", DEFAULT_SCOPE_BOUNDARY)),
        "available_session_count": len(available),
        "planned_session_count": sum(1 for row in rows if row.get("status") == "planned"),
        "excluded_session_count": sum(1 for row in rows if row.get("status") == "excluded"),
        "participant_count": len(participants),
        "device_count": len(devices),
        "session_group_count": len(session_groups),
        "replicate_count": len(replicates),
        "condition_count": len(conditions),
        "reference_evidence_count": len(reference_rows),
        "reference_candidate_count": len(reference_candidates),
        "reference_evidence_issues": reference_evidence_issues,
        "future_validation_scope": future_validation_scope,
        "total_sample_count": int(
            sum(int(_number(row.get("sample_count")) or 0) for row in available)
        ),
        "completed_trial_count": int(
            sum(int(_number(row.get("completed_trial_count")) or 0) for row in available)
        ),
        "finite_gaze_fraction_weighted": _weighted_mean(available, "finite_gaze_fraction"),
        "sampling_rate_hz_weighted": _weighted_mean(available, "sampling_rate_hz"),
        "pupil_valid_fraction_weighted": _weighted_mean(available, "pupil_valid_fraction"),
        "heldout_rms_error_deg_median": _median(heldout_values),
        "heldout_rms_error_deg_min": min(heldout_values) if heldout_values else None,
        "heldout_rms_error_deg_max": max(heldout_values) if heldout_values else None,
        "v1_readiness": {
            "ready": not blockers,
            "criteria": {**DEFAULT_V1_CRITERIA, **dict(criteria)},
            "blockers": blockers,
            "replicate_plan": _remaining_plan(
                available_count=len(available),
                replicate_count=len(replicates),
                participant_count=len(participants),
                device_count=len(devices),
                condition_count=len(conditions),
                reference_evidence_count=len(reference_rows),
                criteria=criteria,
            ),
            "interpretation": _readiness_interpretation(
                blockers=blockers,
                criteria=criteria,
            ),
        },
        "sessions": rows,
    }
    summary["manuscript_tokens"] = _summary_tokens(summary)
    return summary


def aggregate_empirical_sessions(
    *,
    manifest_path: Path,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Return a validated empirical-session aggregate summary from a manifest."""
    root = repo_root if repo_root is not None else manifest_path.resolve().parent.parent
    manifest = load_json_object(manifest_path)
    manifest["source_manifest"] = _repo_relative_or_resolved(manifest_path, root)
    validation = validate_empirical_manifest(manifest, repo_root=root)
    if not validation["valid"]:
        errors = validation.get("errors", [])
        error_items = errors if isinstance(errors, list) else [errors]
        message = "; ".join(str(error) for error in error_items)
        raise ValueError(f"invalid empirical-session manifest: {message}")
    sessions = manifest.get("sessions", [])
    if not isinstance(sessions, list):
        raise ValueError(f"{manifest_path} sessions must be a list")
    rows = [
        empirical_session_row(session, repo_root=root)
        for session in sessions
        if isinstance(session, Mapping)
    ]
    rows.sort(key=lambda row: _session_replicate_sort_key(row.get("replicate_id")))
    summary = build_empirical_sessions_summary(manifest=manifest, session_rows=rows)
    summary["manifest_validation"] = validation
    return summary


def write_empirical_sessions_summary(
    *,
    manifest_path: Path,
    summary_out: Path,
    repo_root: Path | None = None,
) -> Path:
    """Validate, aggregate, and atomically write an empirical-session summary."""
    summary = aggregate_empirical_sessions(manifest_path=manifest_path, repo_root=repo_root)
    return write_json_atomic(summary_out, summary)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path`` with a clear error on shape mismatch."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload
