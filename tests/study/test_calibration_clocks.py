from __future__ import annotations

import pytest

from itrace.study import StudyError, reconstruct
from itrace.study.jsonio import mapping, number, sequence

from .helpers import Harness


def test_calibration_uses_distinct_heldout_targets_when_ready(ready: Harness) -> None:
    diagnostics = mapping(ready.session.status()["calibration"], "calibration")
    assert diagnostics["heldout_target_count"] == 5
    assert number(diagnostics["heldout_rms_css_px"], "rms") < 1e-8
    report = reconstruct(ready.directory)
    fit = sequence(report["calibration_dwells"], "fit")
    heldout = sequence(report["validation_dwells"], "heldout")
    fit_positions = {
        (mapping(row, "dwell")["target_x_css_px"], mapping(row, "dwell")["target_y_css_px"])
        for row in fit
    }
    assert all(
        (mapping(row, "dwell")["target_x_css_px"], mapping(row, "dwell")["target_y_css_px"])
        not in fit_positions
        for row in heldout
    )


@pytest.mark.parametrize("condition", ["pre_onset", "settle_only", "nan", "few", "no_face"])
def test_calibration_rejects_when_post_settle_window_unusable(
    harness: Harness, condition: str
) -> None:
    target = mapping(harness.session.status()["target"], "target")
    if condition == "pre_onset":
        for _ in range(20):
            harness.clock.value += 0.01
            harness.sample(1, 2)
    onset = harness.clock.value
    harness.event(
        "target_onset",
        target_id=target["target_id"],
        position={"x": 150, "y": 120},
        rect={"x": 145, "y": 115, "width": 10, "height": 10},
    )
    for index in range(5 if condition == "few" else 20):
        harness.clock.value = (
            onset + 0.01 + index * 0.02
            if condition == "settle_only"
            else onset + 0.51 + index * 0.04
        )
        if condition != "pre_onset":
            harness.sample(
                float("nan") if condition == "nan" else 1,
                2,
                face=0 if condition == "no_face" else 1,
            )
    harness.clock.value = onset + 1.51
    with pytest.raises(StudyError, match="insufficient"):
        harness.event("target_collect", target_id=target["target_id"])
    assert harness.session.status()["state"] == "calibration"
    assert harness.session.status()["ready"] is False


@pytest.mark.parametrize(
    "condition",
    ["wrong_target", "position", "rect", "duplicate_onset", "early_collect", "no_onset"],
)
def test_target_rejects_when_presentation_contract_invalid(
    harness: Harness, condition: str
) -> None:
    payload = harness.event_payload(
        "target_onset",
        target_id="target-1",
        position={"x": 150, "y": 120},
        rect={"x": 145, "y": 115, "width": 10, "height": 10},
    )
    match condition:
        case "wrong_target":
            payload["target_id"] = "target-2"
        case "position":
            payload["position"] = {"x": 120, "y": 100}
        case "rect":
            payload["rect"] = {"x": 1001, "y": 0, "width": 5, "height": 5}
        case "duplicate_onset" | "early_collect":
            harness.session.event(payload)
    if condition == "duplicate_onset":
        payload = harness.event_payload(
            "target_onset",
            target_id="target-1",
            position={"x": 150, "y": 120},
            rect={"x": 145, "y": 115, "width": 10, "height": 10},
        )
    if condition in {"early_collect", "no_onset"}:
        payload = harness.event_payload("target_collect", target_id="target-1")
    with pytest.raises(StudyError):
        harness.session.event(payload)


def test_validation_rejects_when_independent_targets_inaccurate(harness: Harness) -> None:
    with pytest.raises(StudyError, match="heldout error"):
        harness.ready(validation_offset=100)
    assert harness.session.status()["state"] == "interrupted"
    assert reconstruct(harness.directory)["state"] == "interrupted"


def test_rank_deficient_calibration_rejects_when_gaze_constant(harness: Harness) -> None:
    for _ in range(8):
        target = mapping(harness.session.status()["target"], "target")
        x, y = number(target["x_fraction"], "x") * 1000, number(target["y_fraction"], "y") * 800
        onset = harness.clock.value
        harness.event(
            "target_onset",
            target_id=target["target_id"],
            position={"x": x, "y": y},
            rect={"x": x - 5, "y": y - 5, "width": 10, "height": 10},
        )
        for index in range(20):
            harness.clock.value = onset + 0.51 + index * 0.04
            harness.sample(1, 2)
        harness.clock.value = onset + 1.51
        harness.event("target_collect", target_id=target["target_id"])
        harness.clock.value += 0.01
    target = mapping(harness.session.status()["target"], "target")
    harness.event(
        "target_onset",
        target_id=target["target_id"],
        position={"x": 850, "y": 680},
        rect={"x": 845, "y": 675, "width": 10, "height": 10},
    )
    onset = harness.clock.value
    for index in range(20):
        harness.clock.value = onset + 0.51 + index * 0.04
        harness.sample(1, 2)
    harness.clock.value = onset + 1.51
    with pytest.raises(StudyError, match="rank-deficient"):
        harness.event("target_collect", target_id=target["target_id"])


@pytest.mark.parametrize(
    "condition", ["negative_rtt", "uncertain", "stale_epoch", "duplicate", "unknown", "late"]
)
def test_sync_rejects_when_exchange_not_stable(harness: Harness, condition: str) -> None:
    answer = harness.session.sync(
        {"client_epoch": "browser1", "client_send_ms": harness.clock.value * 1000}
    )
    payload = {
        "sync_id": answer["sync_id"],
        "client_epoch": "browser1",
        "client_received_ms": harness.clock.value * 1000,
    }
    match condition:
        case "negative_rtt":
            payload["client_received_ms"] = harness.clock.value * 1000 - 10
        case "uncertain":
            payload["client_received_ms"] = harness.clock.value * 1000 + 1000
        case "stale_epoch":
            payload["client_epoch"] = "browser2"
        case "duplicate":
            harness.session.confirm_sync(payload)
        case "unknown":
            payload["sync_id"] = "unknown"
        case "late":
            harness.clock.value += 3
    with pytest.raises(StudyError):
        harness.session.confirm_sync(payload)


@pytest.mark.parametrize(
    "condition",
    ["future", "late", "stale_epoch", "expired", "unknown_sync", "unconfirmed", "before_sync"],
)
def test_event_rejects_and_invalidates_when_clock_unbound(harness: Harness, condition: str) -> None:
    payload = harness.event_payload("interrupt", reason="resize")
    match condition:
        case "future":
            payload["browser_ms"] = harness.clock.value * 1000 + 1000
        case "late":
            payload["browser_ms"] = harness.clock.value * 1000 - 3000
        case "stale_epoch":
            payload["client_epoch"] = "browser2"
        case "expired":
            harness.clock.value += 61
            payload["browser_ms"] = harness.clock.value * 1000
        case "unknown_sync":
            payload["sync_id"] = "unknown"
        case "unconfirmed":
            answer = harness.session.sync(
                {"client_epoch": "browser1", "client_send_ms": harness.clock.value * 1000}
            )
            payload["sync_id"] = answer["sync_id"]
        case "before_sync":
            payload["browser_ms"] = harness.clock.value * 1000 - 1
    with pytest.raises(StudyError):
        harness.session.event(payload)
    assert harness.session.status()["state"] == "interrupted"


def test_sync_binds_known_offset_and_error_when_four_timestamps(harness: Harness) -> None:
    start = harness.clock.value
    answer = harness.session.sync(
        {"client_epoch": "browser1", "client_send_ms": (start - 10) * 1000}
    )
    harness.clock.value += 0.02
    result = harness.session.confirm_sync(
        {
            "sync_id": answer["sync_id"],
            "client_epoch": "browser1",
            "client_received_ms": (start - 10 + 0.02) * 1000,
        }
    )
    assert result["offset_s"] == pytest.approx(9.99)
    assert result["error_bound_s"] == pytest.approx(0.01)


def test_host_clock_rejects_when_backwards(harness: Harness) -> None:
    harness.clock.value -= 1
    with pytest.raises(StudyError, match="backwards"):
        harness.session.status()
