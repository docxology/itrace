"""Tests for guided derived eye-video experiment reports."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from itrace import cli, experiments
from itrace.capture import CaptureSample, write_capture_records_csv
from itrace.types import GazeSample, PupilSample, PupilUnit


def _summarizer_module():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("summarize_empirical_pilot")


def _samples_for_protocol(
    protocol: experiments.ExperimentProtocol,
) -> tuple[list[CaptureSample], tuple[experiments.RecordedTrial, ...]]:
    samples: list[CaptureSample] = []
    recorded: list[experiments.RecordedTrial] = []
    trial_start = 0.0
    frame_index = 0
    for trial in protocol.trials:
        trial_end = trial_start + trial.duration_s
        recorded.append(
            experiments.RecordedTrial(
                trial_id=trial.trial_id,
                started_at_s=trial_start,
                ended_at_s=trial_end,
            )
        )
        times = np.arange(trial_start, trial_end, 0.1)
        for t in times:
            rel_t = float(t - trial_start)
            cue = next(
                (item for item in trial.target_schedule if item.start_s <= rel_t <= item.end_s),
                None,
            )
            if cue is not None:
                x = cue.x_deg + 0.04 * np.sin(t)
                y = cue.y_deg + 0.04 * np.cos(t)
            else:
                x = 3.0 * np.sin(t)
                y = 1.0 * np.cos(t * 0.7)
            samples.append(
                CaptureSample(
                    frame_index=frame_index,
                    timestamp_s=float(t),
                    gaze=GazeSample(t=float(t), x=float(x), y=float(y)),
                    pupil=PupilSample(
                        t=float(t), size=0.25 + 0.01 * np.sin(t), unit=PupilUnit.RELATIVE
                    ),
                    fps_estimate_hz=10.0,
                    quality={"face_detected": 1.0},
                )
            )
            frame_index += 1
        trial_start = trial_end
    return samples, tuple(recorded)


def test_default_eye_video_protocol_has_expected_trials_and_targets() -> None:
    protocol = experiments.default_eye_video_protocol(trial_duration_s=10.0, target_range_deg=12.0)

    assert [trial.trial_id for trial in protocol.trials] == [
        "fixed_center",
        "reading",
        "corner_saccades",
    ]
    saccade_trial = protocol.trial("corner_saccades")
    labels = {cue.label for cue in saccade_trial.target_schedule}
    assert {"center", "upper left", "upper right", "lower right", "lower left"} <= labels
    assert any(cue.use_for_fit for cue in saccade_trial.target_schedule)
    assert any(not cue.use_for_fit for cue in saccade_trial.target_schedule)


def test_protocol_parsers_reject_invalid_shapes_and_numeric_fields() -> None:
    protocol = experiments.default_eye_video_protocol(trial_duration_s=10.0, target_range_deg=12.0)
    with pytest.raises(ValueError, match="unknown trial_id"):
        protocol.trial("missing")
    with pytest.raises(ValueError, match="trial_duration_s"):
        experiments.default_eye_video_protocol(trial_duration_s=0.0)
    with pytest.raises(ValueError, match="target_range_deg"):
        experiments.default_eye_video_protocol(target_range_deg=float("nan"))
    with pytest.raises(ValueError, match="hold_s"):
        experiments.default_eye_video_protocol(saccade_hold_s=0.0)

    valid_trial = {
        "trial_id": "fixed_center",
        "kind": "fixation",
        "duration_s": 1.0,
        "prompt": "hold center",
        "target_schedule": [],
    }
    with pytest.raises(ValueError, match="protocol trials"):
        experiments.protocol_from_dict({"target_range_deg": 10.0, "trials": "bad"})
    with pytest.raises(ValueError, match="trial entries"):
        experiments.protocol_from_dict({"target_range_deg": 10.0, "trials": ["bad"]})
    with pytest.raises(ValueError, match="target_schedule"):
        experiments.protocol_from_dict(
            {"target_range_deg": 10.0, "trials": [{**valid_trial, "target_schedule": "bad"}]}
        )
    with pytest.raises(ValueError, match="target schedule entries"):
        experiments.protocol_from_dict(
            {"target_range_deg": 10.0, "trials": [{**valid_trial, "target_schedule": ["bad"]}]}
        )
    with pytest.raises(ValueError, match="duration_s must be numeric"):
        experiments.protocol_from_dict(
            {"target_range_deg": 10.0, "trials": [{**valid_trial, "duration_s": True}]}
        )
    with pytest.raises(ValueError, match="protocol must contain"):
        experiments.protocol_from_dict({"target_range_deg": 10.0, "trials": []})

    active_trial = experiments.recorded_trials_from_dicts(
        [{"trial_id": "fixed_center", "started_at_s": "0.0", "ended_at_s": None}]
    )
    assert active_trial[0].ended_at_s is None
    with pytest.raises(ValueError, match="started_at_s must be numeric"):
        experiments.recorded_trials_from_dicts(
            [{"trial_id": "fixed_center", "started_at_s": False}]
        )


def test_experiment_report_estimates_quality_and_heldout_targets(tmp_path: Path) -> None:
    protocol = experiments.default_eye_video_protocol(trial_duration_s=10.0, target_range_deg=10.0)
    samples, recorded = _samples_for_protocol(protocol)

    report = experiments.experiment_report(samples, protocol, recorded)

    assert report["truth_boundary"] == experiments.TRUTH_BOUNDARY
    assert report["storage_boundary"] == experiments.STORAGE_BOUNDARY
    assert report["completed_trial_count"] == 3
    assert report["sample_count"] == len(samples)
    assert set(report["trials"]) == {"fixed_center", "reading", "corner_saccades"}
    assert report["calibration"] is not None
    assert report["heldout_target_error"]["available"] is True
    assert report["heldout_target_error"]["rms_error_deg"] < 0.25

    paths = experiments.write_experiment_bundle(samples, protocol, recorded, tmp_path)
    assert Path(paths["manifest_json"]).exists()
    assert Path(paths["report_json"]).exists()
    assert Path(paths["target_schedule_csv"]).exists()
    assert Path(paths["corner_saccades_capture_records_csv"]).exists()


def test_sparse_and_incomplete_experiment_reports_keep_boundaries(tmp_path: Path) -> None:
    protocol = experiments.default_eye_video_protocol(
        trial_duration_s=1.0,
        target_range_deg=5.0,
        saccade_hold_s=0.5,
    )
    active = experiments.RecordedTrial("fixed_center", started_at_s=0.0, ended_at_s=None)
    with pytest.raises(ValueError, match="ended_at_s"):
        experiments.slice_trial_samples([], active)
    with pytest.raises(ValueError, match="ended before"):
        experiments.slice_trial_samples(
            [],
            experiments.RecordedTrial("fixed_center", started_at_s=1.0, ended_at_s=0.5),
        )

    empty_report = experiments.experiment_report([], protocol, [active])
    assert empty_report["completed_trial_count"] == 0
    assert empty_report["heldout_target_error"]["available"] is False
    assert empty_report["target_acquisition_latency_s"]["available"] is False

    short_samples = [
        CaptureSample(
            frame_index=index,
            timestamp_s=index * 0.1,
            gaze=GazeSample(t=index * 0.1, x=0.0, y=0.0),
            pupil=None,
            fps_estimate_hz=10.0,
            quality={},
        )
        for index in range(2)
    ]
    short_report = experiments.experiment_report(
        short_samples,
        protocol,
        [experiments.RecordedTrial("fixed_center", started_at_s=0.0, ended_at_s=0.1)],
    )
    assert short_report["trials"]["fixed_center"]["sample_count"] == 2
    assert short_report["trials"]["fixed_center"]["drift_slope_deg_s"] == 0.0

    paths = experiments.write_experiment_bundle([], protocol, [active], tmp_path / "active")
    assert Path(paths["manifest_json"]).exists()
    assert not any(key.endswith("_capture_records_csv") for key in paths)


def test_capture_records_roundtrip_and_cli_experiment_report(tmp_path: Path) -> None:
    protocol = experiments.default_eye_video_protocol(trial_duration_s=10.0, target_range_deg=10.0)
    samples, recorded = _samples_for_protocol(protocol)
    manifest_paths = experiments.write_experiment_bundle(
        samples, protocol, recorded, tmp_path / "bundle"
    )
    records = write_capture_records_csv(samples, tmp_path / "capture_records.csv")
    out = tmp_path / "report.json"

    loaded = experiments.read_capture_records_csv(records)
    assert len(loaded) == len(samples)

    cli.experiment_report_command(
        manifest_json=Path(manifest_paths["manifest_json"]),
        capture_records_csv=records,
        out=out,
    )
    payload = json.loads(out.read_text())
    assert payload["kind"] == "derived_eye_video_experiment"
    assert payload["heldout_target_error"]["available"] is True


def test_empirical_pilot_summarizer_reads_derived_export_without_video(
    tmp_path: Path,
) -> None:
    protocol = experiments.default_eye_video_protocol(trial_duration_s=10.0, target_range_deg=10.0)
    samples, recorded = _samples_for_protocol(protocol)
    bundle = tmp_path / "bundle"
    paths = experiments.write_experiment_bundle(samples, protocol, recorded, bundle)
    summarizer = _summarizer_module()

    outputs = summarizer.write_empirical_pilot_outputs(
        report_path=Path(paths["report_json"]),
        metrics_out=tmp_path / "docs" / "empirical_pilot_metrics.json",
        figure_out=tmp_path / "figures" / "empirical_pilot_summary.png",
        pilot_id="synthetic_pilot",
    )

    metrics = json.loads(outputs["metrics"].read_text())
    assert metrics["available"] is True
    assert metrics["sample_count"] == len(samples)
    assert metrics["completed_trial_count"] == 3
    assert metrics["finite_gaze_fraction"] == pytest.approx(1.0)
    assert metrics["heldout_target_error"]["available"] is True
    assert "reference-device validation" in metrics["truth_boundary"]
    assert "raw eye video" in metrics["storage_boundary"]
    assert "persisted eye-crop images" in metrics["storage_boundary"]
    tokens = metrics["manuscript_tokens"]
    assert tokens["EMPIRICAL_PILOT_FINITE_GAZE"] == "100.0%"
    assert tokens["EMPIRICAL_PILOT_HELDOUT_RMS"].endswith(" deg")
    assert outputs["figure"].exists()
    assert outputs["figure"].stat().st_size > 1000
    raw_video_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
    assert not [path for path in tmp_path.rglob("*") if path.suffix.lower() in raw_video_exts]


def test_empirical_pilot_summarizer_marks_missing_report_unavailable(tmp_path: Path) -> None:
    summarizer = _summarizer_module()

    outputs = summarizer.write_empirical_pilot_outputs(
        report_path=tmp_path / "missing" / "experiment_report.json",
        metrics_out=tmp_path / "docs" / "empirical_pilot_metrics.json",
        figure_out=tmp_path / "figures" / "empirical_pilot_summary.png",
    )

    metrics = json.loads(outputs["metrics"].read_text())
    assert metrics["available"] is False
    assert metrics["source_report"] is None
    assert metrics["manuscript_tokens"]["EMPIRICAL_PILOT_STATUS"] == "pending local recording"
    assert metrics["manuscript_tokens"]["EMPIRICAL_PILOT_FINITE_GAZE"] == "unavailable"
    assert outputs["figure"].exists()
    assert outputs["figure"].stat().st_size > 1000


def test_checked_in_empirical_pilot_metrics_match_source_report() -> None:
    root = Path(__file__).resolve().parent.parent
    metrics_path = root / "docs" / "empirical_pilot_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not payload.get("available"):
        pytest.skip("local empirical pilot has not been exported")

    source_report = Path(str(payload["source_report"]))
    assert not source_report.is_absolute()
    source = root / source_report
    assert source.resolve().is_relative_to(root.resolve())
    report = json.loads(source.read_text(encoding="utf-8"))
    summarizer = _summarizer_module()
    expected = summarizer.metrics_from_report(
        report,
        source_report=source,
        pilot_id=str(payload["pilot_id"]),
    )
    assert payload == expected


def test_experiment_csv_and_manifest_reject_malformed_files(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    assert experiments.read_capture_records_csv(empty_csv) == []

    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("frame_index,timestamp_s\n0,0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        experiments.read_capture_records_csv(missing_columns)

    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        experiments.load_experiment_manifest(manifest)

    manifest.write_text(json.dumps({"protocol": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol object"):
        experiments.load_experiment_manifest(manifest)

    protocol = experiments.default_eye_video_protocol(trial_duration_s=1.0).to_dict()
    manifest.write_text(
        json.dumps({"protocol": protocol, "recorded_trials": {"trial_id": "fixed_center"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recorded_trials"):
        experiments.load_experiment_manifest(manifest)
