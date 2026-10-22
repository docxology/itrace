from __future__ import annotations

from pathlib import Path

import pytest

from itrace.study import StudyError, reconstruct
from itrace.study.jsonio import JsonValue, mapping, number

from .helpers import Harness
from .recording_helpers import capture_prefix, copy_session, finish_behavior, rewrite, rows


def completed(ready: Harness) -> list[dict[str, JsonValue]]:
    finish_behavior(ready)
    for _ in range(5):
        ready.target()
    assert reconstruct(ready.directory)["correctness_denominator"] == 2
    return rows(ready.directory / "journal.jsonl")


def save(target: Path, records: list[dict[str, JsonValue]]) -> None:
    for index, record in enumerate(records, 1):
        record["sequence"] = index
    rewrite(target / "journal.jsonl", records)


@pytest.mark.parametrize("terminal", ["abort", "capture_fault", "interrupt", "clock_rejection"])
def test_replay_rejects_when_events_follow_terminal(
    ready: Harness, tmp_path: Path, terminal: str
) -> None:
    # Given a completed real synthetic session with a terminal row before trial one.
    records = completed(ready)
    target = copy_session(ready, tmp_path)
    at = next(i for i, row in enumerate(records) if row["kind"] == "start_trial")
    inserted = dict(records[at])
    match terminal:
        case "abort" | "capture_fault":
            inserted.update(kind=terminal, payload={"reason": "terminal before first trial"})
        case "interrupt":
            original = next(
                mapping(mapping(row["payload"], "payload")["event"], "event")
                for row in records[at:]
                if row["kind"] == "events"
            )
            event = {key: original[key] for key in ("client_epoch", "sync_id", "browser_ms")}
            event.update(event_id="terminal-interrupt", kind="interrupt", reason="disconnect")
            inserted.update(kind="events", payload={"event": event})
        case "clock_rejection":
            inserted.update(
                kind="rejected",
                payload={"event": {}, "field": "sync_id", "reason": "unbound clock"},
            )
    records.insert(at, inserted)
    save(target, records)
    # When replay consumes the structurally valid, renumbered journal.
    # Then it rejects instead of counting the two post-terminal scored responses.
    with pytest.raises(StudyError, match="state: transition after terminal"):
        reconstruct(target)


@pytest.mark.parametrize(
    "change",
    [
        "target_onset",
        "target_collect",
        "trial_onset",
        "response",
        "capture_state",
        "diagnostics",
        "dwell_target",
        "dwell_phase",
        "dwell_window",
        "dwell_epoch",
        "dwell_position",
        "dwell_quality",
        "dwell_samples",
        "duplicate_prepare",
    ],
)
def test_replay_rejects_when_completed_transition_chain_corrupted(
    ready: Harness, tmp_path: Path, change: str
) -> None:
    # Given a completed synthetic trace with one broken transition or dwell binding.
    records = completed(ready)
    target = copy_session(ready, tmp_path)
    match change:
        case "target_onset" | "target_collect" | "trial_onset" | "response":
            at = next(
                i
                for i, row in enumerate(records)
                if row["kind"] == "events"
                and mapping(mapping(row["payload"], "payload")["event"], "event")["kind"] == change
            )
            records.pop(at)
        case "capture_state":
            records = [row for row in records if row["kind"] != change]
        case "diagnostics":
            at = next(
                i for i, row in enumerate(records) if row["kind"] == "calibration_diagnostics"
            )
            records[at]["payload"] = {"fit_rank": 3}
        case "duplicate_prepare":
            at = next(i for i, row in enumerate(records) if row["kind"] == "prepare")
            records.insert(at, dict(records[at]))
        case _:
            row = next(row for row in records if row["kind"] == "calibration")
            payload = dict(mapping(row["payload"], "payload"))
            changes: dict[str, tuple[str, JsonValue]] = {
                "dwell_target": ("target_id", "target-2"),
                "dwell_phase": ("phase", "post_validation"),
                "dwell_window": ("window_start_s", 0),
                "dwell_epoch": ("epoch_id", "other"),
                "dwell_position": ("target_x_css_px", 1),
                "dwell_quality": ("quality_adequate", False),
                "dwell_samples": ("valid_samples", 0),
            }
            key, value = changes[change]
            payload[key] = value
            row["payload"] = payload
    save(target, records)
    # When the altered journal is reconstructed, then it fails closed.
    with pytest.raises(StudyError):
        reconstruct(target)


def test_replay_retains_each_valid_write_prefix_when_collection_interrupted(
    ready: Harness, tmp_path: Path
) -> None:
    # Given every durable write boundary from a completed synthetic session.
    records = completed(ready)
    target = copy_session(ready, tmp_path)
    native_path = target / "capture.csv"
    epoch_path = target / "capture_epochs.jsonl"
    native, epochs = native_path.read_bytes(), epoch_path.read_bytes()
    reports = []
    # When reconstruction reads each prefix, then it accepts only observed progress.
    for end in range(1, len(records) + 1):
        save(target, records[:end])
        native_prefix, epoch_prefix = capture_prefix(native, epochs, end)
        native_path.write_bytes(native_prefix)
        epoch_path.write_bytes(epoch_prefix)
        reports.append(reconstruct(target))
    first_complete = next(
        i
        for i, row in enumerate(records)
        if row["kind"] == "post_validation"
        and mapping(row["payload"], "payload")["target_id"] == "target-19"
    )
    assert all(report["state"] != "complete" for report in reports[:first_complete])
    assert reports[first_complete]["state"] == reports[-1]["state"] == "complete"


def test_replay_retains_rejection_and_retry_when_dwell_samples_insufficient(
    harness: Harness,
) -> None:
    # Given an honest rejected collection followed by successful recalibration.
    with pytest.raises(StudyError, match="insufficient"):
        harness.target(count=5)
    harness.clock.value += 0.01
    harness.ready()
    # When replay reconstructs the real journal, then the rejection remains diagnostic.
    report = reconstruct(harness.directory)
    assert report["rejected_events"] == 1
    assert (
        mapping(report["calibration_diagnostics"], "diagnostics")["initial_validation_adequate"]
        is True
    )


def test_replay_retains_failed_fit_when_rank_deficient(harness: Harness) -> None:
    # Given nine real dwells with constant horizontal gaze and a rejected affine fit.
    for _ in range(8):
        target = mapping(harness.session.status()["target"], "target")
        x = number(target["x_fraction"], "x") * 1000
        harness.target(offset=-(x - 100) / 20)
    target = mapping(harness.session.status()["target"], "target")
    x = number(target["x_fraction"], "x") * 1000
    with pytest.raises(StudyError, match="rank-deficient"):
        harness.target(offset=-(x - 100) / 20)
    # When replay reads the producer's partial failure writes, then it preserves interruption.
    report = reconstruct(harness.directory)
    assert report["state"] == "interrupted"
    assert report["rejected_events"] == 1
    assert report["correctness_denominator"] == 0


@pytest.mark.parametrize("state", ["complete", "aborted", "interrupted"])
def test_replay_retains_terminal_diagnostics_when_producer_allows_them(
    ready: Harness, state: str
) -> None:
    # Given legitimate terminal sessions with subsequent diagnostics and export operations.
    match state:
        case "complete":
            completed(ready)
        case "aborted":
            ready.session.abort({"reason": "operator stop"})
        case "interrupted":
            ready.event("interrupt", reason="disconnect")
    ready.session.capture_state(False, "capture1", "stop after terminal")
    ready.sync()
    with pytest.raises(StudyError, match="cannot accept further events"):
        ready.event("interrupt", reason="duplicate stop")
    # When export reconstructs the journal, then terminal status cannot be overwritten.
    report = mapping(ready.session.export()["report"], "report")
    assert report["state"] == state
    assert report["rejected_events"] == 1


def test_replay_retains_stop_race_when_sync_confirmation_and_onset_arrive_after_abort(
    harness: Harness,
) -> None:
    # Given calibration with an outstanding real synchronization exchange at abort.
    exchange = harness.session.sync(
        {
            "client_epoch": "browser1",
            "client_send_ms": harness.clock.value * 1000,
        }
    )
    harness.session.abort({"reason": "stop during calibration"})
    harness.session.confirm_sync(
        {
            "sync_id": exchange["sync_id"],
            "client_epoch": "browser1",
            "client_received_ms": harness.clock.value * 1000,
        }
    )
    harness.sync()
    with pytest.raises(StudyError, match="cannot accept further events"):
        harness.event(
            "target_onset",
            target_id="target-1",
            position={"x": 150, "y": 120},
            rect={"x": 145, "y": 115, "width": 10, "height": 10},
        )
    harness.session.capture_state(False, "capture1", "stopped")
    # When partial export reconstructs the abort race, then rejection remains diagnostic.
    report = mapping(harness.session.export()["report"], "report")
    assert report["state"] == "aborted"
    assert report["correctness_denominator"] == 0
    assert report["rejected_events"] == 1


@pytest.mark.parametrize("change", ["rest", "stale_sync", "late_confirmation"])
def test_replay_rejects_when_producer_clock_or_rest_gate_bypassed(
    tmp_path: Path, change: str
) -> None:
    # Given real records whose receipt times bypass one producer readiness gate.
    harness = Harness(tmp_path, rest=2)
    harness.ready()
    harness.start()
    harness.onset()
    harness.answer()
    harness.clock.value += 2
    harness.start(1)
    records = rows(harness.directory / "journal.jsonl")
    target = copy_session(harness, tmp_path)
    match change:
        case "rest":
            records[-1]["server_received_s"] = (
                number(records[-1]["server_received_s"], "receipt") - 1
            )
        case "stale_sync":
            records[-1]["server_received_s"] = 1000
        case "late_confirmation":
            row = next(row for row in records if row["kind"] == "confirm_sync")
            row["server_received_s"] = 200
    save(target, records)
    # When replay reads the bypassed gate, then it rejects the transition.
    with pytest.raises(StudyError):
        reconstruct(target)


@pytest.mark.parametrize("removed", ["prepare", "calibration", "validation", "start_trial"])
def test_replay_rejects_when_required_stage_removed(
    ready: Harness, tmp_path: Path, removed: str
) -> None:
    # Given real completed records with one required stage removed.
    records = completed(ready)
    target = copy_session(ready, tmp_path)
    save(target, [row for row in records if row["kind"] != removed])
    # When replay reads the incomplete transition chain, then completion is rejected.
    with pytest.raises(StudyError):
        reconstruct(target)


def test_replay_rejects_when_only_five_post_dwells_claim_completion(
    ready: Harness, tmp_path: Path
) -> None:
    # Given five valid dwell payloads without their onset/collection/trial history.
    records = completed(ready)
    target = copy_session(ready, tmp_path)
    save(
        target, [row for row in records if row["kind"] in {"created", "prepare", "post_validation"}]
    )
    # When replay reads journal kinds alone, then they cannot establish completion.
    with pytest.raises(StudyError):
        reconstruct(target)
