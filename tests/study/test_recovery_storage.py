from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from itrace.study import StudyError, StudySession, reconstruct
from itrace.study.jsonio import encoded, mapping, parse_object, sequence
from itrace.study.storage import Journal

from .helpers import Harness
from .recording_helpers import copy_session, rewrite, rows


def test_recovery_retains_valid_prefix_when_process_terminates_midwrite(harness: Harness) -> None:
    journal = harness.directory / "journal.jsonl"
    before = journal.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sys; f=open(sys.argv[1],'ab',buffering=0); "
            "f.write(b'{\"sequence\":'); os.fsync(f.fileno()); os._exit(7)",
            str(journal),
        ],
        check=False,
    )
    report = reconstruct(harness.directory)
    assert result.returncode == 7
    assert report["incomplete_final_line"] is True
    assert journal.read_bytes().startswith(before)


@pytest.mark.parametrize(
    "change", ["invalid_json", "schema", "sequence", "session", "unknown_kind", "payload"]
)
def test_recovery_rejects_when_interior_record_corrupted(
    harness: Harness, tmp_path: Path, change: str
) -> None:
    target = copy_session(harness, tmp_path)
    journal = target / "journal.jsonl"
    data = rows(journal)
    match change:
        case "invalid_json":
            journal.write_bytes(b"{broken}\n" + journal.read_bytes())
        case "schema":
            data[0]["schema"] = "other"
        case "sequence":
            data[1]["sequence"] = 1
        case "session":
            data[0]["session_id"] = "other"
        case "unknown_kind":
            data[0]["kind"] = "unknown"
        case "payload":
            data[0]["payload"] = []
    if change != "invalid_json":
        rewrite(journal, data)
    with pytest.raises(StudyError):
        reconstruct(target)


@pytest.mark.parametrize(
    "file", ["manifest.json", "metadata.json", "image0.png", "capture.csv", "journal.jsonl"]
)
def test_recovery_rejects_when_binding_or_storage_corrupted(
    harness: Harness, tmp_path: Path, file: str
) -> None:
    target = copy_session(harness, tmp_path)
    match file:
        case "metadata.json":
            metadata = dict(parse_object((target / file).read_bytes()))
            metadata["manifest_digest"] = "f" * 64
            (target / file).write_bytes(encoded(metadata))
        case "journal.jsonl":
            (target / file).rename(target / "retained-journal")
        case _:
            (target / file).write_bytes(b"malformed")
    with pytest.raises(StudyError):
        reconstruct(target)


def test_recovery_recomputes_when_derived_flags_and_estimates_tampered(
    ready: Harness, tmp_path: Path
) -> None:
    ready.start()
    ready.onset()
    ready.answer()
    ready.start(1)
    ready.onset(1)
    ready.answer(1)
    target = copy_session(ready, tmp_path)
    journal = target / "journal.jsonl"
    data = rows(journal)
    for row in data:
        if row["kind"] == "events":
            payload = mapping(row["payload"], "payload")
            payload["estimate_s"] = -999
            payload["correct"] = False
    rewrite(journal, data)
    report = reconstruct(target)
    assert report["scored_correct"] == 1
    assert mapping(sequence(report["trials"], "trials")[1], "trial")["rt_s"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_id",
        "bad_option",
        "late_response",
        "missing_onset",
        "wrong_trial",
        "duplicate_sync",
        "confirm_without_sync",
        "duplicate_prepare",
        "digest",
    ],
)
def test_recovery_rejects_when_raw_trace_conflicts(
    ready: Harness, tmp_path: Path, change: str
) -> None:
    ready.start()
    ready.onset()
    ready.answer()
    target = copy_session(ready, tmp_path)
    journal = target / "journal.jsonl"
    data = rows(journal)
    events = [mapping(row["payload"], "payload") for row in data if row["kind"] == "events"]
    onset, response = (mapping(item["event"], "event") for item in events[-2:])
    match change:
        case "duplicate_id":
            response["event_id"] = onset["event_id"]
        case "bad_option":
            response["key"] = "z"
        case "late_response":
            response["browser_ms"] = 999999
        case "missing_onset":
            start = next(row for row in data if row["kind"] == "start_trial")
            start["kind"] = "calibration_diagnostics"
        case "wrong_trial":
            response["trial_id"] = "trial1"
        case "duplicate_sync":
            original = next(row for row in data if row["kind"] == "sync")
            confirm_row = next(row for row in data if row["kind"] == "confirm_sync")
            confirm_row["kind"], confirm_row["payload"] = "sync", original["payload"]
        case "confirm_without_sync":
            next(row for row in data if row["kind"] == "sync")["kind"] = "calibration_diagnostics"
        case "duplicate_prepare":
            data[-1]["kind"], data[-1]["payload"] = "prepare", {"client_epoch": "browser1"}
        case "digest":
            mapping(data[0]["payload"], "payload")["manifest_digest"] = "other"
    rewrite(journal, data)
    with pytest.raises(StudyError):
        reconstruct(target)


def test_export_checksums_bind_exact_bytes_when_partial(harness: Harness) -> None:
    result = harness.session.export()
    export_dir = Path(str(result["export_dir"]))
    hashes = parse_object((export_dir / "checksums.json").read_bytes())
    for relative, digest in hashes.items():
        assert hashlib.sha256((harness.directory / relative).read_bytes()).hexdigest() == digest
    assert (export_dir / "trials.csv").read_text().count("unattempted") == 3


def test_storage_error_is_explicit_when_real_append_path_blocked(harness: Harness) -> None:
    journal = harness.directory / "journal.jsonl"
    journal.rename(harness.directory / "original-journal.jsonl")
    journal.mkdir()
    with pytest.raises(StudyError, match="durable append failed"):
        harness.session.abort({"reason": "escape"})
    assert harness.session.status()["storage_failed"] is True
    with pytest.raises(StudyError, match="previous storage failure"):
        harness.session.abort({"reason": "again"})


def test_output_refuses_symlink_when_explicit_root_alias(harness: Harness, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(harness.directory.parent, target_is_directory=True)
    with pytest.raises(StudyError, match="symlink"):
        StudySession(harness.manifest, alias, harness.clock)


def test_output_refuses_public_source_when_root_inside_package(harness: Harness) -> None:
    with pytest.raises(StudyError, match="outside public source"):
        StudySession(
            harness.manifest, Path(__file__).resolve().parents[2] / "src/private", harness.clock
        )


def test_storage_never_overwrites_when_destination_exists(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        Journal.write_new(target, b"replacement")
    assert target.read_bytes() == b"original"


def test_constructor_retains_typed_failure_when_parent_is_file(
    harness: Harness, tmp_path: Path
) -> None:
    target = tmp_path / "occupied"
    target.write_text("original")
    with pytest.raises(StudyError, match="cannot allocate"):
        StudySession(harness.manifest, target, harness.clock)


def test_capture_recovery_retains_prefix_when_final_csv_row_torn(harness: Harness) -> None:
    harness.sample(1, 2)
    with (harness.directory / "capture.csv").open("ab") as stream:
        stream.write(b"2,101.0,")
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["native_rows"] == 1
    assert capture["incomplete_native_final_line"] is True


def test_capture_recovery_rejects_when_interior_csv_row_malformed(harness: Harness) -> None:
    with (harness.directory / "capture.csv").open("ab") as stream:
        stream.write(b"malformed,row\n")
    with pytest.raises(StudyError, match="row shape"):
        reconstruct(harness.directory)


def test_manifest_rejects_when_root_directory_symlinked(harness: Harness, tmp_path: Path) -> None:
    from itrace.study import load_manifest

    alias = tmp_path / "bound-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(StudyError, match="ancestors"):
        load_manifest(alias / "study.json")
