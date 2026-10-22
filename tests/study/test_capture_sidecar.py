from __future__ import annotations

import csv
from pathlib import Path

import pytest

from itrace.capture import CaptureSample
from itrace.study import StudyError, reconstruct
from itrace.study.jsonio import JsonValue, encoded, mapping
from itrace.types import GazeSample

from .helpers import Harness
from .recording_helpers import finish_behavior, rewrite, rows


def test_complete_study_rejects_empty_sidecar_identity(ready: Harness) -> None:
    finish_behavior(ready)
    for _ in range(5):
        ready.target()
    before = reconstruct(ready.directory)
    assert before["state"] == "complete"
    assert mapping(before["capture"], "capture")["native_rows"] == 380
    path = ready.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0] = {}
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(ready.directory)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extra", 1),
        ("session_id", "forged"),
        ("frame_index", None),
        ("frame_index", True),
        ("frame_index", 1.0),
        ("frame_index", 1.5),
        ("frame_index", -1),
        ("frame_index", 10**100),
        ("frame_index", 2),
        ("native_timestamp_s", True),
        ("native_timestamp_s", None),
        ("native_timestamp_s", 99),
        ("native_timestamp_s", 10**100),
        ("host_received_s", True),
        ("host_received_s", -1),
        ("host_received_s", 10**100),
        ("host_received_s", 106),
        ("epoch_id", "unregistered"),
        ("epoch_id", []),
        ("epoch_id", ""),
        ("clock_valid", 1),
        ("clock_valid", "true"),
        ("clock_valid", False),
        ("reason", "forged"),
    ],
)
def test_sidecar_rejects_forged_fields(harness: Harness, field: str, value: JsonValue) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0][field] = value
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize(
    "payload",
    [
        b"{}\n",
        b'{"clock_valid":false}\n',
        b'{"native_timestamp_s":NaN}\n',
        b'{"native_timestamp_s":1e999}\n',
        b"[]\n",
        b'{"clock_valid":true,"clock_valid":false}\n',
    ],
)
def test_sidecar_rejects_malformed_complete_line(harness: Harness, payload: bytes) -> None:
    harness.sample(1, 2)
    (harness.directory / "capture_epochs.jsonl").write_bytes(payload)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def change_native(directory: Path, field: str, value: str) -> None:
    path = directory / "capture.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header, data = reader.fieldnames, list(reader)
    assert header is not None
    data[0][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_index", "2"),
        ("frame_index", "True"),
        ("frame_index", "1.5"),
        ("frame_index", "9" * 100),
        ("timestamp_s", "99"),
        ("timestamp_s", "1e999"),
        ("timestamp_s", "nan"),
        ("fps_estimate_hz", "nan"),
        ("fps_estimate_hz", "1e999"),
        ("fps_estimate_hz", "-1"),
        ("gaze_x_deg", "1e999"),
        ("pupil_size", "1e999"),
    ],
)
def test_native_rows_cannot_disagree_or_overflow(harness: Harness, field: str, value: str) -> None:
    harness.sample(1, 2)
    change_native(harness.directory, field, value)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_sidecar_order_bound_to_native_order(harness: Harness) -> None:
    harness.sample(1, 2)
    harness.clock.value += 1
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    rewrite(path, list(reversed(rows(path))))
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_false_flag_cannot_be_backed_by_unrelated_fault(harness: Harness) -> None:
    harness.sample(1, 2, timestamp=101)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0]["host_received_s"] = 99
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_clock_fault_cannot_be_relabelled_valid(harness: Harness) -> None:
    harness.sample(1, 2, timestamp=101)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0].update(clock_valid=True, reason=None)
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize("timestamp", [101.0, 1.0, float("nan"), float("inf")])
def test_honest_fault_retained_as_integrity_defect(harness: Harness, timestamp: float) -> None:
    harness.sample(float("nan"), float("nan"), timestamp=timestamp, face=0)
    report = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert report["clock_fault_rows"] == 1
    assert report["finite_gaze_rows"] == 0
    assert report["integrity_ok"] is False


def test_missing_face_nan_is_diagnostic_not_malformed(harness: Harness) -> None:
    harness.sample(float("nan"), float("nan"), face=0)
    report = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert report["clock_fault_rows"] == 0
    assert report["finite_gaze_rows"] == 0
    assert report["missing_gaze_rows"] == 1


@pytest.mark.parametrize("tail", ["native", "sidecar", "both", "unpaired_native", "unpaired_epoch"])
def test_partial_capture_is_explicit_and_bytes_preserved(harness: Harness, tail: str) -> None:
    harness.sample(1, 2)
    native = harness.directory / "capture.csv"
    sidecar = harness.directory / "capture_epochs.jsonl"
    if tail in {"native", "both"}:
        with native.open("ab") as stream:
            stream.write(b"2,101,")
    if tail in {"sidecar", "both"}:
        with sidecar.open("ab") as stream:
            stream.write(b'{"epoch_id":')
    if tail == "unpaired_native":
        sidecar.write_bytes(b"")
    if tail == "unpaired_epoch":
        native.write_bytes(native.read_bytes().splitlines(keepends=True)[0])
    before = native.read_bytes(), sidecar.read_bytes()
    report = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert report["integrity_ok"] is False
    assert report["integrity_defects"]
    assert (native.read_bytes(), sidecar.read_bytes()) == before


def test_metadata_session_identity_is_checked(harness: Harness) -> None:
    path = harness.directory / "metadata.json"
    metadata = rows(path)[0]
    metadata["session_id"] = "forged"
    path.write_bytes(encoded(metadata))
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize(
    "field",
    ["epoch_id", "frame_index", "host_received_s", "native_timestamp_s", "clock_valid", "reason"],
)
def test_each_sidecar_identity_field_required(harness: Harness, field: str) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    del data[0][field]
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize("field", ["frame_index", "timestamp_s"])
def test_paired_native_discontinuity_cannot_claim_valid(harness: Harness, field: str) -> None:
    harness.sample(1, 2)
    harness.clock.value += 1
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    identity = "frame_index" if field == "frame_index" else "native_timestamp_s"
    data[0][identity] = data[1][identity]
    rewrite(path, data)
    change_native(harness.directory, field, str(data[1][identity]))
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_active_epoch_restart_preserves_real_reset(harness: Harness) -> None:
    harness.sample(1, 2)
    harness.clock.value += 10
    harness.session.capture_state(True, "capture2")
    harness.frame_index = 0
    harness.sample(1, 2, epoch="capture2")
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["clock_fault_rows"] == 0
    assert capture["native_rows"] == capture["epoch_rows"] == 2


def test_stale_receipt_fault_is_exactly_verifiable(harness: Harness) -> None:
    harness.clock.value += 10
    harness.session.capture(
        CaptureSample(1, 100, GazeSample(100, 1, 2)),
        "capture1",
        100,
    )
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["integrity_ok"] is False
    assert capture["unverifiable_clock_rows"] == 0
    assert capture["clock_fault_rows"] == 1


def test_valid_flag_cannot_hide_durable_stale_receipt_fault(harness: Harness) -> None:
    harness.clock.value += 10
    harness.session.capture(CaptureSample(1, 100, GazeSample(100, 1, 2)), "capture1", 100)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0].update(clock_valid=True, reason=None)
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_false_flag_with_producer_reason_requires_journal_fault(harness: Harness) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0].update(clock_valid=False, reason="native clock/frame discontinuity or inactive epoch")
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_empty_capture_does_not_claim_integrity(harness: Harness) -> None:
    assert mapping(reconstruct(harness.directory)["capture"], "capture")["integrity_ok"] is False


def test_unpaired_epoch_retains_clock_fault_count(harness: Harness) -> None:
    harness.sample(1, 2, timestamp=101)
    path = harness.directory / "capture.csv"
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["clock_fault_rows"] == 1
    assert capture["integrity_ok"] is False
