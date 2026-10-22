from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .calibration import Frame
from .jsonio import JsonValue, StudyError, fields, number, parse_object, text
from .runtime import Runtime
from .setup import rectangle
from .storage import Journal


def validate_shape(data: Mapping[str, JsonValue]) -> str:
    base = "event_id client_epoch sync_id browser_ms kind"
    kind = text(data.get("kind"), "kind")
    match kind:
        case "target_onset":
            fields(data, base + " target_id rect position")
        case "target_collect":
            fields(data, base + " target_id")
        case "trial_onset":
            fields(data, base + " trial_id rect presentation")
        case "response":
            fields(data, base + " trial_id key")
        case "timeout":
            fields(data, base + " trial_id")
        case "interrupt":
            fields(data, base + " reason")
        case _:
            raise StudyError("kind", "unknown event kind")
    for key in ("event_id", "client_epoch", "sync_id"):
        text(data[key], key, identifier=True)
    number(data["browser_ms"], "browser_ms")
    return kind


def transition(
    runtime: Runtime,
    journal: Journal,
    payload: Mapping[str, JsonValue],
    kind: str,
    estimate: float,
    error: float,
    now: float,
    frames: Sequence[Frame],
) -> dict[str, JsonValue]:
    derived: dict[str, JsonValue] = {
        "estimate_s": estimate,
        "uncertainty_s": error,
        "presentation_timing": "browser estimate, not physical onset proof",
    }
    if estimate < runtime.last_event_s:
        raise StudyError("browser_ms", "out-of-order browser event")
    match kind:
        case "interrupt":
            text(payload["reason"], "reason")
            journal.append("events", {"event": dict(payload), **derived}, now)
            runtime.invalidate(text(payload["reason"], "reason"))
        case "target_onset" | "target_collect":
            if runtime.state not in {"calibration", "validation", "post_validation"}:
                raise StudyError("state", "target event outside calibration/validation")
            calibration = runtime.require_calibration()
            epoch = runtime.require_capture()
            match kind:
                case "target_onset":
                    onset = calibration.begin(payload, estimate, error, epoch)
                    journal.append("events", {"event": dict(payload), **derived}, now)
                    calibration.onset = onset
                case "target_collect":
                    dwell = calibration.collect(payload, estimate, error, frames, epoch)
                    journal.append("events", {"event": dict(payload), **derived}, now)
                    phase = (
                        "calibration"
                        if calibration.index < 9
                        else "validation"
                        if calibration.index < 14
                        else "post_validation"
                    )
                    journal.append(phase, parse_object(dwell.payload), now)
                    calibration.accept(dwell)
                    if calibration.index == 9:
                        runtime.state = "validation"
                    if calibration.index == 14:
                        runtime.state = "ready"
                    if calibration.index == 19:
                        runtime.state = "complete"
                    journal.append("calibration_diagnostics", calibration.diagnostics, now)
        case "trial_onset" | "response" | "timeout":
            runtime.require_capture()
            trial = runtime.trial()
            if payload["trial_id"] != trial.trial_id:
                raise StudyError("trial_id", "stale or unexpected trial")
            outcome = runtime.outcomes[runtime.index]
            match kind:
                case "trial_onset":
                    if runtime.state != "awaiting_onset" or payload["presentation"] != "raf":
                        raise StudyError("state", "start_trial and co-present rAF onset required")
                    rect = rectangle(payload["rect"], runtime.require_setup())
                    ratio = number(rect["width"], "width") / number(rect["height"], "height")
                    if abs(ratio / (trial.stimulus.width / trial.stimulus.height) - 1) > 0.02:
                        raise StudyError("rect", "stimulus aspect ratio changed")
                    outcome = replace(
                        outcome, outcome="presented", onset_s=estimate, uncertainty_s=error
                    )
                case "response" | "timeout":
                    if (
                        runtime.state != "trial"
                        or outcome.onset_s is None
                        or outcome.uncertainty_s is None
                    ):
                        raise StudyError("state", "acknowledged trial onset required")
                    earliest_deadline = outcome.onset_s - outcome.uncertainty_s + trial.deadline_s
                    latest_deadline = outcome.onset_s + outcome.uncertainty_s + trial.deadline_s
                    match kind:
                        case "response":
                            key = text(payload["key"], "key")
                            if key not in {option.key for option in trial.options}:
                                raise StudyError("key", "response key not bound to trial")
                            if estimate - error < outcome.onset_s + outcome.uncertainty_s:
                                raise StudyError("response", "early or onset-uncertain response")
                            if estimate + error >= earliest_deadline:
                                raise StudyError("response", "late or deadline-uncertain response")
                            outcome = replace(
                                outcome,
                                outcome="answered",
                                key=key,
                                response_s=estimate,
                                correct=key == trial.correct_key,
                            )
                        case "timeout":
                            if estimate - error < latest_deadline:
                                raise StudyError("timeout", "deadline has not certainly elapsed")
                            outcome = replace(outcome, outcome="timeout")
            journal.append("events", {"event": dict(payload), **derived}, now)
            runtime.outcomes[runtime.index] = outcome
            if kind == "trial_onset":
                runtime.state = "trial"
            else:
                runtime.index += 1
                if runtime.index == len(runtime.outcomes):
                    runtime.state = "post_validation"
                elif trial.rest_s:
                    runtime.rest_until_s = now + trial.rest_s
                    runtime.state = "rest"
                else:
                    runtime.state = "ready"
                if trial.role == "practice" and kind == "response":
                    derived["practice_correct"] = outcome.correct
        case _:
            raise StudyError("kind", "unknown event")
    runtime.event_ids.add(text(payload["event_id"], "event_id"))
    runtime.last_event_s = estimate
    return derived
