from __future__ import annotations

import pytest

from itrace.capture import CaptureSample
from itrace.study import StudyError, reconstruct
from itrace.study.jsonio import JsonValue, mapping
from itrace.types import GazeSample

from .helpers import Harness
from .recording_helpers import finish_behavior, rewrite, rows


def test_complete_stream_with_nearby_stop_has_exact_integrity(ready: Harness) -> None:
    finish_behavior(ready)
    for _ in range(5):
        ready.target()
    ready.session.capture_state(False, "capture1")
    report = reconstruct(ready.directory)
    capture = mapping(report["capture"], "capture")
    assert report["state"] == "complete"
    assert capture["native_rows"] == capture["epoch_rows"] == 380
    assert capture["ambiguous_clock_rows"] == capture["unverifiable_clock_rows"] == 0
    assert capture["integrity_ok"] is True


def test_invocation_binds_start_even_when_receipt_precedes_it(harness: Harness) -> None:
    harness.session.capture_state(False, "capture1")
    harness.clock.value = 102
    harness.session.capture_state(True, "capture2")
    sequence = rows(harness.directory / "journal.jsonl")[-1]["sequence"]
    harness.session.capture(CaptureSample(1, 101, GazeSample(101, 1, 2)), "capture2", 101)
    harness.session.capture_state(False, "capture2")
    row = rows(harness.directory / "capture_epochs.jsonl")[0]
    assert row["host_recorded_s"] == 102
    assert row["journal_sequence"] == sequence
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["integrity_ok"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host_recorded_s", None),
        ("host_recorded_s", True),
        ("host_recorded_s", "100"),
        ("host_recorded_s", -1),
        ("host_recorded_s", 10**100),
        ("host_recorded_s", 99),
        ("host_recorded_s", 105.001),
        ("journal_sequence", None),
        ("journal_sequence", True),
        ("journal_sequence", "5"),
        ("journal_sequence", 5.0),
        ("journal_sequence", 5.5),
        ("journal_sequence", 0),
        ("journal_sequence", -1),
        ("journal_sequence", 200001),
        ("journal_sequence", 6),
        ("journal_sequence", 4),
    ],
)
def test_recording_binding_rejects_malformed_or_forged_values(
    harness: Harness, field: str, value: JsonValue
) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0][field] = value
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize("field", ["host_recorded_s", "journal_sequence", "both"])
def test_undeclared_old_sidecar_epoch_is_rejected(harness: Harness, field: str) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    for key in ("host_recorded_s", "journal_sequence") if field == "both" else (field,):
        data[0].pop(key, None)
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_saved_sequence_cannot_authorize_inactive_state_at_equal_time(harness: Harness) -> None:
    harness.sample(1, 2)
    harness.session.capture_state(False, "capture1")
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0]["journal_sequence"] = rows(harness.directory / "journal.jsonl")[-1]["sequence"]
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize("bound", ["preceding", "following", "row_order"])
def test_recorded_time_respects_serialized_journal_and_capture_order(
    harness: Harness, bound: str
) -> None:
    harness.clock.value = 101
    harness.sync()
    harness.clock.value = 102
    harness.session.capture(CaptureSample(1, 100, GazeSample(100, 1, 2)), "capture1", 100)
    harness.clock.value = 103
    harness.sync()
    harness.clock.value = 104
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    match bound:
        case "preceding":
            data[0]["host_recorded_s"] = 100
        case "following":
            data[0]["host_recorded_s"] = 104
        case "row_order":
            data[1]["journal_sequence"] = data[0]["journal_sequence"]
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_false_flag_with_matching_fault_cannot_override_valid_clock(harness: Harness) -> None:
    harness.sample(1, 2)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0].update(clock_valid=False, reason="native clock/frame discontinuity or inactive epoch")
    rewrite(path, data)
    journal = harness.directory / "journal.jsonl"
    records = rows(journal)
    records.append(
        {
            **records[-1],
            "sequence": len(records) + 1,
            "kind": "capture_fault",
            "payload": {"reason": data[0]["reason"]},
        }
    )
    rewrite(journal, records)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize("timestamp", [99, 101])
def test_reversed_journal_clock_is_rejected(harness: Harness, timestamp: int) -> None:
    harness.sample(1, 2)
    harness.clock.value = 102
    harness.sync()
    journal = harness.directory / "journal.jsonl"
    records = rows(journal)
    records[-1]["server_received_s"] = timestamp
    rewrite(journal, records)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


def test_recording_times_cannot_reverse_within_one_journal_prefix(harness: Harness) -> None:
    harness.clock.value = 102
    harness.session.capture(CaptureSample(1, 100, GazeSample(100, 1, 2)), "capture1", 100)
    harness.clock.value = 103
    harness.session.capture(CaptureSample(2, 101, GazeSample(101, 1, 2)), "capture1", 101)
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[0]["host_recorded_s"] = 104
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)


@pytest.mark.parametrize(("native_age", "delay"), [(0, 0), (5, 0), (0, 5), (5, 5)])
def test_exact_five_second_clock_boundaries_are_inclusive(
    harness: Harness, native_age: int, delay: int
) -> None:
    receipt = 105
    harness.clock.value = receipt + delay
    timestamp = receipt - native_age
    harness.session.capture(
        CaptureSample(1, timestamp, GazeSample(timestamp, 1, 2)), "capture1", receipt
    )
    assert mapping(reconstruct(harness.directory)["capture"], "capture")["integrity_ok"] is True


@pytest.mark.parametrize("receipt", [99, 106])
def test_honest_invalid_receipt_is_verifiable_fault(harness: Harness, receipt: int) -> None:
    harness.clock.value = 105
    harness.session.capture(
        CaptureSample(1, receipt, GazeSample(receipt, 1, 2)), "capture1", receipt
    )
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["clock_fault_rows"] == 1
    assert capture["unverifiable_clock_rows"] == 0
    assert capture["integrity_ok"] is False


def test_same_receipt_faults_require_distinct_acknowledged_prefixes(harness: Harness) -> None:
    harness.sample(1, 2, timestamp=101)
    harness.sample(1, 2, timestamp=101)
    capture = mapping(reconstruct(harness.directory)["capture"], "capture")
    assert capture["clock_fault_rows"] == 2
    path = harness.directory / "capture_epochs.jsonl"
    data = rows(path)
    data[1]["journal_sequence"] = data[0]["journal_sequence"]
    rewrite(path, data)
    with pytest.raises(StudyError):
        reconstruct(harness.directory)
