"""Tests for empirical-session manifests and v1-readiness aggregation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from itrace import empirical, experiments

ROOT = Path(__file__).resolve().parent.parent


def _aggregator_module():
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("aggregate_empirical_sessions")


def _minimal_report(sample_count: int = 30) -> dict[str, object]:
    trial = {
        "sample_count": sample_count / 3,
        "finite_gaze_fraction": 1.0,
        "sampling_rate_hz": 30.0,
        "sampling_interval_cv": 0.02,
        "pupil_valid_fraction": 0.98,
        "drift_slope_deg_s": 0.01,
    }
    return {
        "kind": "derived_eye_video_experiment",
        "truth_boundary": experiments.TRUTH_BOUNDARY,
        "storage_boundary": experiments.STORAGE_BOUNDARY,
        "sample_count": sample_count,
        "completed_trial_count": 3,
        "trials": {
            "fixed_center": dict(trial),
            "reading": dict(trial),
            "corner_saccades": dict(trial),
        },
        "heldout_target_error": {"available": True, "rms_error_deg": 2.25},
        "target_acquisition_latency_s": {"available": True, "median_latency_s": 0.35},
    }


def _manifest(report: str = "output/session_001/experiment_report.json") -> dict[str, object]:
    return {
        "kind": empirical.MANIFEST_KIND,
        "version": 1,
        "truth_boundary": experiments.TRUTH_BOUNDARY,
        "storage_boundary": experiments.STORAGE_BOUNDARY,
        "v1_readiness_criteria": {
            "min_available_sessions": 2,
            "min_replicates": 2,
            "min_participants": 1,
            "min_devices": 1,
            "min_conditions": 2,
            "requires_reference_evidence": True,
        },
        "sessions": [
            {
                "session_id": "session_001",
                "status": "available",
                "participant_id": "P001",
                "device_id": "device_a",
                "session_group": "P001_device_a",
                "replicate_id": "R001",
                "condition": "office_daylight",
                "protocol_id": "derived_eye_video_v1",
                "consent_scope": "derived_records_only",
                "reference_kind": "none",
                "report": report,
            }
        ],
    }


def _write_report_and_records(tmp_path: Path) -> tuple[str, str]:
    report = tmp_path / "output" / "session_001" / "experiment_report.json"
    records = tmp_path / "output" / "session_001" / "live_capture_records.csv"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(_minimal_report()), encoding="utf-8")
    records.write_text(
        "frame_index,timestamp_s,gaze_x_deg,gaze_y_deg,pupil_size,pupil_unit\n"
        "0,0.0,0.0,0.0,0.2,relative\n",
        encoding="utf-8",
    )
    return (
        "output/session_001/experiment_report.json",
        "output/session_001/live_capture_records.csv",
    )


def _manual_annotation_artifact(
    *,
    session_id: str = "session_001",
    source_report: str = "output/session_001/experiment_report.json",
    source_records: str = "output/session_001/live_capture_records.csv",
) -> dict[str, object]:
    return {
        "kind": empirical.MANUAL_ANNOTATION_ARTIFACT_KIND,
        "version": 1,
        "session_id": session_id,
        "source_report": source_report,
        "source_records": source_records,
        "annotation_scope": "prompted_target_windows",
        "annotator_id": "A001",
        "created_at": "2026-06-08T12:00:00Z",
        "annotations": [
            {
                "trial_id": "corner_saccades",
                "target_label": "center",
                "start_s": 0.0,
                "end_s": 1.5,
                "quality": "usable",
                "target_hit": "yes",
            }
        ],
    }


def test_empirical_manifest_validation_rejects_absolute_and_raw_video_paths(
    tmp_path: Path,
) -> None:
    payload = _manifest(report="/tmp/raw_eye_video.mp4")
    result = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)
    assert result["valid"] is False
    assert any("repo-relative" in error for error in result["errors"])

    payload = _manifest(report="output/session_001/raw_eye_video.mp4")
    result = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)
    assert result["valid"] is False
    assert any("raw video" in error for error in result["errors"])


def test_empirical_manifest_rejects_available_non_json_report(tmp_path: Path) -> None:
    report = tmp_path / "output" / "session_001" / "derived_records.csv"
    report.parent.mkdir(parents=True)
    report.write_text("timestamp_s,gaze_x_deg,gaze_y_deg\n0,0,0\n", encoding="utf-8")
    payload = _manifest(report="output/session_001/derived_records.csv")

    result = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)

    assert result["valid"] is False
    assert any("derived JSON report" in error for error in result["errors"])


def test_empirical_manifest_rejects_duplicate_replicates(tmp_path: Path) -> None:
    payload = _manifest()
    duplicate = dict(payload["sessions"][0])
    duplicate["session_id"] = "session_002"
    payload["sessions"].append(duplicate)

    result = empirical.validate_empirical_manifest(
        payload,
        repo_root=tmp_path,
        require_existing_reports=False,
    )

    assert result["valid"] is False
    assert any("replicate_id duplicates" in error for error in result["errors"])


def test_aggregate_empirical_sessions_reports_v1_blockers(tmp_path: Path) -> None:
    report = tmp_path / "output" / "session_001" / "experiment_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(_minimal_report()), encoding="utf-8")
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    aggregator = _aggregator_module()

    summary = aggregator.aggregate_empirical_sessions(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert summary["available_session_count"] == 1
    assert summary["participant_count"] == 1
    assert summary["device_count"] == 1
    assert summary["session_group_count"] == 1
    assert summary["replicate_count"] == 1
    assert summary["condition_count"] == 1
    assert summary["reference_evidence_count"] == 0
    assert summary["finite_gaze_fraction_weighted"] == pytest.approx(1.0)
    assert summary["heldout_rms_error_deg_median"] == pytest.approx(2.25)
    readiness = summary["v1_readiness"]
    assert readiness["ready"] is False
    assert any("replicates" in blocker for blocker in readiness["blockers"])
    assert not any("participants" in blocker for blocker in readiness["blockers"])
    assert not any("devices" in blocker for blocker in readiness["blockers"])
    assert any("reference" in blocker for blocker in readiness["blockers"])
    plan = readiness["replicate_plan"]
    assert plan["minimum_additional_sessions_to_meet_count_criteria"] == 1
    assert plan["minimum_additional_sessions_to_meet_all_criteria_if_reference_backed"] == 1
    assert plan["conditions_remaining"] == 1
    assert plan["reference_evidence_remaining"] == 1
    assert plan["additional_prompt_only_sessions_still_leave_reference_blocker"] is True
    tokens = summary["manuscript_tokens"]
    assert "1 available session" in tokens["EMPIRICAL_SESSIONS_STATUS"]
    assert tokens["EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL"] == "1"


def test_reference_kind_without_artifact_remains_diagnostic_only(tmp_path: Path) -> None:
    report, _records = _write_report_and_records(tmp_path)
    payload = _manifest(report=report)
    session = payload["sessions"][0]
    session["reference_kind"] = "manual_annotation"
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)
    summary = empirical.aggregate_empirical_sessions(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert validation["valid"] is True
    assert any("reference_artifact is required" in warning for warning in validation["warnings"])
    assert summary["reference_candidate_count"] == 1
    assert summary["reference_evidence_count"] == 0
    assert summary["reference_evidence_issues"][0]["reference_kind"] == "manual_annotation"
    assert "reference" in " ".join(summary["v1_readiness"]["blockers"])


def test_manual_annotation_reference_artifact_counts_when_valid(tmp_path: Path) -> None:
    report, records = _write_report_and_records(tmp_path)
    artifact_path = tmp_path / "output" / "session_001" / "manual_annotation_evidence.json"
    artifact_path.write_text(
        json.dumps(_manual_annotation_artifact(source_report=report, source_records=records)),
        encoding="utf-8",
    )
    payload = _manifest(report=report)
    session = payload["sessions"][0]
    session["reference_kind"] = "manual_annotation"
    session["reference_artifact"] = "output/session_001/manual_annotation_evidence.json"
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)
    artifact_validation = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    summary = empirical.aggregate_empirical_sessions(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert validation["valid"] is True
    assert artifact_validation["valid"] is True
    assert summary["reference_candidate_count"] == 1
    assert summary["reference_evidence_count"] == 1
    assert not any("reference" in blocker for blocker in summary["v1_readiness"]["blockers"])


def test_manual_annotation_artifact_rejects_raw_video_source(tmp_path: Path) -> None:
    report, _records = _write_report_and_records(tmp_path)
    artifact_path = tmp_path / "output" / "session_001" / "manual_annotation_evidence.json"
    artifact_path.write_text(
        json.dumps(
            _manual_annotation_artifact(
                source_report=report,
                source_records="output/session_001/raw_eye_video.mp4",
            )
        ),
        encoding="utf-8",
    )
    payload = _manifest(report=report)
    session = payload["sessions"][0]
    session["reference_kind"] = "manual_annotation"
    session["reference_artifact"] = "output/session_001/manual_annotation_evidence.json"

    artifact_validation = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    manifest_validation = empirical.validate_empirical_manifest(payload, repo_root=tmp_path)

    assert artifact_validation["valid"] is False
    assert any("raw video" in error for error in artifact_validation["errors"])
    assert manifest_validation["valid"] is False
    assert any("raw video" in error for error in manifest_validation["errors"])


def test_reference_artifact_validator_rejects_bad_artifact_files(tmp_path: Path) -> None:
    report, _records = _write_report_and_records(tmp_path)
    session = dict(_manifest(report=report)["sessions"][0])
    session["reference_kind"] = "manual_annotation"

    session["reference_artifact"] = "output/session_001/missing.json"
    missing = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    assert missing["valid"] is False
    assert any("does not exist" in error for error in missing["errors"])

    csv_artifact = tmp_path / "output" / "session_001" / "manual_annotation_evidence.csv"
    csv_artifact.write_text("kind,version\nmanual,1\n", encoding="utf-8")
    session["reference_artifact"] = "output/session_001/manual_annotation_evidence.csv"
    non_json = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    assert non_json["valid"] is False
    assert any("derived JSON evidence" in error for error in non_json["errors"])

    bad_json = tmp_path / "output" / "session_001" / "manual_annotation_bad.json"
    bad_json.write_text("{", encoding="utf-8")
    session["reference_artifact"] = "output/session_001/manual_annotation_bad.json"
    invalid_json = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    assert invalid_json["valid"] is False
    assert any("invalid JSON" in error for error in invalid_json["errors"])

    list_json = tmp_path / "output" / "session_001" / "manual_annotation_list.json"
    list_json.write_text("[]", encoding="utf-8")
    session["reference_artifact"] = "output/session_001/manual_annotation_list.json"
    wrong_shape = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    assert wrong_shape["valid"] is False
    assert any("JSON object" in error for error in wrong_shape["errors"])


def test_reference_artifact_validator_reports_schema_errors(tmp_path: Path) -> None:
    report, records = _write_report_and_records(tmp_path)
    artifact_path = tmp_path / "output" / "session_001" / "manual_annotation_schema.json"
    artifact_path.write_text(
        json.dumps(
            {
                "kind": "wrong_kind",
                "version": "one",
                "session_id": "other_session",
                "source_report": "output/session_001/other_report.json",
                "source_records": records,
                "annotation_scope": "",
                "annotator_id": "",
                "created_at": "",
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    session = dict(_manifest(report=report)["sessions"][0])
    session["reference_kind"] = "manual_annotation"
    session["reference_artifact"] = "output/session_001/manual_annotation_schema.json"

    result = empirical.validate_reference_artifact(session, repo_root=tmp_path)

    errors = " ".join(result["errors"])
    assert result["valid"] is False
    assert "version must be numeric" in errors
    assert "session_id must match" in errors
    assert "manual_annotation reference_artifact.kind" in errors
    assert "annotation_scope" in errors
    assert "annotator_id" in errors
    assert "created_at" in errors
    assert "source_report must match" in errors
    assert "source_report does not exist" in errors
    assert "annotations must be a nonempty list" in errors


def test_reference_artifact_validator_reports_bad_annotation_rows(tmp_path: Path) -> None:
    report, records = _write_report_and_records(tmp_path)
    artifact_path = tmp_path / "output" / "session_001" / "manual_annotation_bad_rows.json"
    artifact_path.write_text(
        json.dumps(
            _manual_annotation_artifact(source_report=report, source_records=records)
            | {
                "annotations": [
                    "not an object",
                    {
                        "trial_id": "corner_saccades",
                        "target_label": "center",
                        "start_s": 2.0,
                        "end_s": 1.0,
                        "quality": "great",
                        "target_hit": "maybe",
                    },
                    {
                        "trial_id": "",
                        "target_label": "",
                        "start_s": "early",
                        "end_s": "late",
                        "quality": "",
                        "target_hit": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    session = dict(_manifest(report=report)["sessions"][0])
    session["reference_kind"] = "manual_annotation"
    session["reference_artifact"] = "output/session_001/manual_annotation_bad_rows.json"

    result = empirical.validate_reference_artifact(session, repo_root=tmp_path)

    errors = " ".join(result["errors"])
    assert result["valid"] is False
    assert "annotations[0] must be an object" in errors
    assert "quality must be one of" in errors
    assert "target_hit must be one of" in errors
    assert "end_s must exceed start_s" in errors
    assert "trial_id must be a nonempty string" in errors
    assert "start_s must be numeric" in errors


def test_generic_reference_artifact_schema_counts_for_public_dataset(
    tmp_path: Path,
) -> None:
    report, _records = _write_report_and_records(tmp_path)
    artifact_path = tmp_path / "output" / "session_001" / "public_dataset_reference.json"
    artifact_path.write_text(
        json.dumps(
            {
                "kind": "itrace_public_dataset_reference",
                "version": 1,
                "session_id": "session_001",
            }
        ),
        encoding="utf-8",
    )
    payload = _manifest(report=report)
    session = payload["sessions"][0]
    session["reference_kind"] = "public_dataset"
    session["reference_artifact"] = "output/session_001/public_dataset_reference.json"
    manifest_path = tmp_path / "docs" / "empirical_sessions_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = empirical.validate_reference_artifact(session, repo_root=tmp_path)
    summary = empirical.aggregate_empirical_sessions(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert result["valid"] is True
    assert summary["reference_candidate_count"] == 1
    assert summary["reference_evidence_count"] == 1


def test_empirical_sessions_summary_figure_renders(tmp_path: Path) -> None:
    aggregator = _aggregator_module()
    summary = empirical.build_empirical_sessions_summary(
        manifest={"v1_readiness_criteria": {"requires_reference_evidence": True}},
        session_rows=[
            {
                "session_id": "session_001",
                "status": "available",
                "available": True,
                "participant_id": "P001",
                "device_id": "device_a",
                "session_group": "P001_device_a",
                "replicate_id": "R001",
                "condition": "office_daylight",
                "reference_kind": "none",
                "sample_count": 30,
                "completed_trial_count": 3,
                "finite_gaze_fraction": 1.0,
                "heldout_rms_error_deg": 2.25,
            }
        ],
    )

    figure = aggregator.generate_empirical_sessions_figure(
        summary,
        tmp_path / "empirical_sessions_summary.png",
    )

    assert figure.exists()
    assert figure.stat().st_size > 1000


def test_checked_in_empirical_sessions_summary_matches_manifest() -> None:
    aggregator = _aggregator_module()
    manifest_path = ROOT / "docs" / "empirical_sessions_manifest.json"
    summary_path = ROOT / "docs" / "empirical_sessions_summary.json"
    checked = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = aggregator.aggregate_empirical_sessions(manifest_path=manifest_path)

    assert checked == expected
    assert checked["available_session_count"] == 5
    assert checked["replicate_count"] == 5
    assert checked["condition_count"] == 2
    assert checked["reference_evidence_count"] == 0
    assert checked["v1_readiness"]["ready"] is True
    assert checked["v1_readiness"]["blockers"] == []
    assert checked["v1_readiness"]["criteria"]["min_available_sessions"] == 5
    assert checked["v1_readiness"]["criteria"]["min_replicates"] == 5
    assert checked["v1_readiness"]["criteria"]["min_conditions"] == 2
    assert checked["v1_readiness"]["criteria"]["requires_reference_evidence"] is False
    assert "participants" not in " ".join(checked["v1_readiness"]["blockers"])
    assert "devices" not in " ".join(checked["v1_readiness"]["blockers"])
    plan = checked["v1_readiness"]["replicate_plan"]
    assert plan["minimum_additional_sessions_to_meet_count_criteria"] == 0
    assert plan["minimum_additional_sessions_to_meet_all_criteria_if_reference_backed"] == 0
    assert plan["conditions_remaining"] == 0
    assert plan["reference_evidence_remaining"] == 0
    future = checked["future_validation_scope"]
    assert future["target_available_sessions"] == 12
    assert future["target_replicates"] == 12
    assert future["target_conditions"] == 3
    assert future["requires_reference_evidence"] is True
    assert future["available_sessions_remaining"] == 7
    assert future["replicates_remaining"] == 7
    assert future["conditions_remaining"] == 1
    assert future["missing_conditions"] == ["indoor_office_backlit"]
    assert future["reference_evidence_remaining"] == 1
