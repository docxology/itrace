from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from itrace.capture import CaptureSample
from itrace.study import JsonValue, StudyError, StudySession, reconstruct
from itrace.study.jsonio import mapping, sequence
from itrace.types import GazeSample

from .helpers import Clock, Harness, setup_payload


def test_complete_trace_reconstructs_when_explicit_responses_and_timeout(ready: Harness) -> None:
    ready.start()
    ready.onset()
    ready.answer()
    ready.start(1)
    ready.onset(1)
    ready.answer(1)
    ready.start(2)
    ready.onset(2)
    ready.clock.value += 3
    ready.event("timeout", trial_id="trial2")
    for _ in range(5):
        ready.target()
    report = reconstruct(ready.directory)
    assert report["state"] == "complete"
    assert (report["correctness_denominator"], report["scored_correct"]) == (1, 1)
    trials = sequence(report["trials"], "trials")
    assert [mapping(row, "trial")["outcome"] for row in trials] == [
        "answered",
        "answered",
        "timeout",
    ]
    assert mapping(trials[1], "trial")["rt_s"] == pytest.approx(0.5)


def test_export_is_unique_and_preserves_native_records_when_partial(harness: Harness) -> None:
    harness.sample(float("nan"), 4)
    first = harness.session.export()
    second = harness.session.export()
    assert first["export_dir"] != second["export_dir"]
    with (harness.directory / "capture.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["timestamp_s"] == "100.0"
    assert row["gaze_x_deg"] == "nan"
    assert json.loads(row["quality_json"]) == {"face_detected": 1}
    assert sequence(mapping(first["report"], "report")["trials"], "trials")[0] is not None
    assert (harness.directory / "manifest.json").read_bytes() == harness.manifest.original_bytes
    assert (harness.directory / "image0.png").read_bytes() == harness.manifest.stimulus_bytes(
        "trial0"
    )


@pytest.mark.parametrize(
    "phase", ["calibration", "validation", "ready", "awaiting_onset", "trial", "rest"]
)
def test_abort_retains_partial_export_when_any_phase(tmp_path: Path, phase: str) -> None:
    harness = Harness(tmp_path, rest=1)
    if phase == "validation":
        for _ in range(9):
            harness.target()
    elif phase in {"ready", "awaiting_onset", "trial", "rest"}:
        harness.ready()
        if phase != "ready":
            harness.start()
        if phase in {"trial", "rest"}:
            harness.onset()
        if phase == "rest":
            harness.answer()
    harness.session.abort({"reason": "escape"})
    report = mapping(harness.session.export()["report"], "report")
    assert report["state"] == "aborted"
    assert report["correctness_denominator"] == 0


def test_abort_setup_retains_unattempted_when_not_prepared(
    harness: Harness, tmp_path: Path
) -> None:
    session = StudySession(harness.manifest, tmp_path / "new-private", harness.clock)
    session.abort({"reason": "escape"})
    assert mapping(session.export()["report"], "report")["state"] == "aborted"


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("response", {"trial_id": "trial0", "key": "a"}),
        ("timeout", {"trial_id": "trial0"}),
        (
            "trial_onset",
            {
                "trial_id": "trial0",
                "presentation": "raf",
                "rect": {"x": 0, "y": 0, "width": 500, "height": 500},
            },
        ),
        ("unknown", {}),
    ],
)
def test_event_rejects_when_outside_required_stage(
    harness: Harness, kind: str, extra: dict[str, JsonValue]
) -> None:
    with pytest.raises(StudyError):
        harness.event(kind, **extra)


@pytest.mark.parametrize(
    "change", ["early", "late", "invalid", "stale", "timeout_early", "duplicate"]
)
def test_response_rejects_when_not_eligible(ready: Harness, change: str) -> None:
    ready.start()
    if change != "early":
        ready.onset()
    ready.clock.value += 3 if change == "late" else 0.5
    payload = ready.event_payload("response", trial_id="trial0", key="a")
    match change:
        case "invalid":
            payload["key"] = "z"
        case "stale":
            payload["trial_id"] = "trial1"
        case "timeout_early":
            payload = ready.event_payload("timeout", trial_id="trial0")
        case "duplicate":
            ready.session.event(payload)
    with pytest.raises(StudyError):
        ready.session.event(payload)
    assert reconstruct(ready.directory)["correctness_denominator"] == 0


def test_double_choice_does_not_replace_when_first_accepted(ready: Harness) -> None:
    ready.start()
    ready.onset()
    ready.answer()
    with pytest.raises(StudyError):
        ready.event("response", trial_id="trial0", key="b")
    trials = sequence(reconstruct(ready.directory)["trials"], "trials")
    assert mapping(trials[0], "trial")["key"] == "a"


def test_scored_status_withholds_key_correctness_when_answered(ready: Harness) -> None:
    ready.start()
    ready.onset()
    feedback = ready.event_payload("response", trial_id="trial0", key="a")
    ready.clock.value += 0.5
    feedback["browser_ms"] = ready.clock.value * 1000
    assert ready.session.event(feedback)["practice_correct"] is True
    ready.start(1)
    ready.onset(1)
    ready.answer(1)
    outcomes = sequence(ready.session.status()["outcomes"], "outcomes")
    assert mapping(outcomes[1], "outcome")["correct"] is None


@pytest.mark.parametrize("reason", ["resize", "tabhide", "disconnect", "fullscreen_loss"])
def test_interrupt_excludes_correctness_when_technical(ready: Harness, reason: str) -> None:
    ready.start()
    ready.onset()
    ready.clock.value += 0.1
    ready.event("interrupt", reason=reason)
    report = reconstruct(ready.directory)
    assert report["state"] == "interrupted"
    assert mapping(sequence(report["trials"], "trials")[0], "trial")["outcome"] == "interrupted"
    with pytest.raises(StudyError):
        ready.start()


def test_rest_is_enforced_when_declared(tmp_path: Path) -> None:
    harness = Harness(tmp_path, rest=2)
    harness.ready()
    harness.start()
    harness.onset()
    harness.answer()
    with pytest.raises(StudyError):
        harness.start(1)
    harness.clock.value += 2
    harness.start(1)
    assert harness.session.status()["state"] == "awaiting_onset"


@pytest.mark.parametrize(
    "change", ["capture_stop", "epoch", "backward", "duplicate_frame", "wrong_clock"]
)
def test_capture_invalidation_preserves_native_when_discontinuous(
    ready: Harness, change: str
) -> None:
    ready.start()
    ready.onset()
    match change:
        case "capture_stop":
            ready.session.capture_state(False, "capture1", "failed")
        case "epoch":
            ready.session.capture_state(True, "capture2")
        case "backward":
            ready.sample(1, 2, timestamp=1)
        case "duplicate_frame":
            ready.frame_index -= 1
            ready.sample(1, 2)
        case "wrong_clock":
            ready.sample(1, 2, timestamp=ready.clock.value + 1)
    assert ready.session.status()["state"] == "interrupted"
    assert reconstruct(ready.directory)["correctness_denominator"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("participant_id", 1),
        ("storage_acknowledged", False),
        ("fullscreen", False),
        ("distance_m", True),
        ("dpr", float("nan")),
        ("screen_width_px", 1000.5),
        ("browser", ""),
        ("viewport_width_px", 0),
    ],
)
def test_setup_rejects_when_not_explicit_finite_geometry(
    harness: Harness, tmp_path: Path, field: str, value: JsonValue
) -> None:
    session = StudySession(harness.manifest, tmp_path / "other", harness.clock)
    payload = setup_payload()
    payload[field] = value
    with pytest.raises(StudyError):
        session.prepare(payload)


def test_prepare_rejects_when_replayed(harness: Harness) -> None:
    with pytest.raises(StudyError, match="cannot be replayed"):
        harness.session.prepare(setup_payload())


def test_start_rejects_when_calibration_absent(harness: Harness) -> None:
    with pytest.raises(StudyError):
        harness.start()


def test_native_clock_sequence_is_preserved_when_reset(harness: Harness, tmp_path: Path) -> None:
    clock = Clock(0)
    session = StudySession(harness.manifest, tmp_path / "clock-control", clock)
    session.capture_state(True, "native1")
    for frame, timestamp in enumerate((1.0, 2.0, 0.0, 1.0)):
        clock.value = frame + 1.0
        session.capture(
            CaptureSample(frame, timestamp, GazeSample(timestamp, 1, 2)), "native1", clock.value
        )
    directory = Path(str(session.status()["output_dir"]))
    with (directory / "capture.csv").open() as stream:
        native = [float(row["timestamp_s"]) for row in csv.DictReader(stream)]
    assert native == [1, 2, 0, 1]
    assert mapping(reconstruct(directory)["capture"], "capture")["clock_fault_rows"] == 2


def test_calibration_rejects_huge_finite_gaze_when_overflow_risk(harness: Harness) -> None:
    target = mapping(harness.session.status()["target"], "target")
    harness.event(
        "target_onset",
        target_id=target["target_id"],
        position={"x": 150, "y": 120},
        rect={"x": 145, "y": 115, "width": 10, "height": 10},
    )
    onset = harness.clock.value
    for index in range(20):
        harness.clock.value = onset + 0.51 + index * 0.04
        harness.sample(1e308, 1e308)
    harness.clock.value = onset + 1.51
    with pytest.raises(StudyError, match="insufficient"):
        harness.event("target_collect", target_id=target["target_id"])
