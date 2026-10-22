from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .calibration_replay import replay_calibration
from .capture_report import capture_diagnostics
from .clocks import Exchange, confirm, event_time
from .events import validate_shape
from .journal_read import read_journal
from .jsonio import (
    MAX_JSON,
    JsonValue,
    StudyError,
    mapping,
    number,
    parse_object,
    text,
)
from .manifest import load_manifest, read_bounded
from .replay_transitions import ReplayTransitions
from .runtime import Outcome


def _exchange(payload: Mapping[str, JsonValue]) -> Exchange:
    return Exchange(
        text(payload.get("sync_id"), "sync_id"),
        text(payload.get("client_epoch"), "client_epoch"),
        number(payload.get("client_send_ms"), "client_send_ms"),
        number(payload.get("server_received_s"), "server_received_s"),
        number(payload.get("server_sent_s"), "server_sent_s"),
    )


def reconstruct(session_dir: Path) -> dict[str, JsonValue]:
    """Recompute key, timing and outcomes from durable events; never resume trials."""
    manifest = load_manifest(session_dir / "manifest.json")
    metadata = parse_object(read_bounded(session_dir / "metadata.json", MAX_JSON))
    if metadata.get("manifest_digest") != manifest.digest:
        raise StudyError("manifest", "snapshot digest differs from metadata")
    records, incomplete = read_journal(session_dir)
    outcomes = [Outcome(trial.trial_id, trial.role) for trial in manifest.trials]
    exchanges: dict[str, Exchange] = {}
    seen: set[str] = set()
    index = 0
    transitions = ReplayTransitions(manifest)
    epoch: str | None = None
    last_event = -1.0
    pending = False
    rejected = 0
    fit_records: list[JsonValue] = []
    validation_records: list[JsonValue] = []
    post_records: list[JsonValue] = []
    dwell_records: list[JsonValue] = []
    setup_payload: dict[str, JsonValue] | None = None
    for row in records:
        payload = mapping(row["payload"], "payload")
        received = number(row["server_received_s"], "server_received_s")
        transitions.record(text(row["kind"], "kind"), payload, received)
        if transitions.terminal and pending:
            outcomes[index] = replace(outcomes[index], outcome=transitions.state, correct=None)
            pending = False
        match row["kind"]:
            case "created":
                if payload.get("manifest_digest") != manifest.digest:
                    raise StudyError("manifest", "journal manifest digest differs")
            case "prepare":
                if epoch is not None:
                    raise StudyError("prepare", "replayed preparation")
                epoch = text(payload.get("client_epoch"), "client_epoch")
                setup_payload = dict(payload)
            case "sync":
                exchange = _exchange(payload)
                if exchange.sync_id in exchanges or exchange.client_epoch != epoch:
                    raise StudyError("sync", "duplicate or stale exchange")
                exchanges[exchange.sync_id] = exchange
            case "confirm_sync":
                sync_id = text(payload.get("sync_id"), "sync_id")
                if sync_id not in exchanges:
                    raise StudyError("sync", "confirmation without exchange")
                if received - exchanges[sync_id].server_sent_s > manifest.limits.max_event_lag_s:
                    raise StudyError("sync", "confirmation too late")
                exchanges[sync_id] = confirm(
                    {
                        "sync_id": sync_id,
                        "client_epoch": payload.get("client_epoch"),
                        "client_received_ms": payload.get("client_received_ms"),
                    },
                    exchanges[sync_id],
                    manifest.limits,
                )
            case "start_trial":
                if not any(
                    exchange.offset_s is not None
                    and received - exchange.server_sent_s <= manifest.limits.sync_max_age_s
                    for exchange in exchanges.values()
                ):
                    raise StudyError("sync", "current confirmed synchronization required")
                if (
                    index >= len(outcomes)
                    or payload.get("trial_id") != outcomes[index].trial_id
                    or pending
                ):
                    raise StudyError("trial_id", "invalid replayed start order")
                pending = True
            case "events":
                event = mapping(payload.get("event"), "event")
                kind = validate_shape(event)
                event_id = text(event["event_id"], "event_id")
                sync_id = text(event["sync_id"], "sync_id")
                if event_id in seen or sync_id not in exchanges:
                    raise StudyError("events", "duplicate identity or unbound sync")
                seen.add(event_id)
                estimate, error = event_time(event, exchanges[sync_id], received, manifest.limits)
                if estimate < last_event:
                    raise StudyError("events", "backward event time")
                last_event = estimate
                transitions.event(event, estimate, error)
                match kind:
                    case "interrupt":
                        if pending:
                            outcomes[index] = replace(
                                outcomes[index], outcome="interrupted", correct=None
                            )
                            pending = False
                    case "trial_onset" | "response" | "timeout":
                        if (
                            index >= len(outcomes)
                            or not pending
                            or event["trial_id"] != outcomes[index].trial_id
                        ):
                            raise StudyError("trial_id", "invalid event trial order")
                        trial, outcome = manifest.trials[index], outcomes[index]
                        match kind:
                            case "trial_onset":
                                if outcome.onset_s is not None or event["presentation"] != "raf":
                                    raise StudyError("trial_onset", "duplicate/invalid onset")
                                outcome = replace(
                                    outcome,
                                    outcome="presented",
                                    onset_s=estimate,
                                    uncertainty_s=error,
                                )
                            case "response" | "timeout":
                                if outcome.onset_s is None or outcome.uncertainty_s is None:
                                    raise StudyError("response", "missing onset")
                                onset_s, onset_error_s = outcome.onset_s, outcome.uncertainty_s
                                match kind:
                                    case "response":
                                        key = text(event["key"], "key")
                                        if key not in {option.key for option in trial.options}:
                                            raise StudyError("key", "unbound option")
                                        if (
                                            estimate - error < onset_s + onset_error_s
                                            or estimate + error
                                            >= onset_s - onset_error_s + trial.deadline_s
                                        ):
                                            raise StudyError(
                                                "response", "ineligible response timing"
                                            )
                                        outcome = replace(
                                            outcome,
                                            outcome="answered",
                                            key=key,
                                            response_s=estimate,
                                            correct=key == trial.correct_key,
                                        )
                                    case "timeout":
                                        if (
                                            estimate - error
                                            < onset_s + onset_error_s + trial.deadline_s
                                        ):
                                            raise StudyError("timeout", "premature timeout")
                                        outcome = replace(outcome, outcome="timeout")
                                pending = False
                        outcomes[index] = outcome
                        if not pending:
                            index += 1
                            transitions.finish_trial(received)
            case "capture_state" | "capture_fault" | "abort":
                pass
            case "rejected":
                rejected += 1
            case "calibration":
                fit_records.append(dict(payload))
                dwell_records.append(dict(payload))
            case "validation":
                validation_records.append(dict(payload))
                dwell_records.append(dict(payload))
            case "post_validation":
                post_records.append(dict(payload))
                dwell_records.append(dict(payload))
            case "calibration_diagnostics":
                pass
            case _:
                raise StudyError("journal", "unknown record kind")
    scored = [
        outcome
        for outcome in outcomes
        if outcome.role == "scored" and outcome.outcome == "answered"
    ]
    return {
        "schema": "itrace.private-study-report",
        "version": 1,
        "session_id": session_dir.name,
        "manifest_digest": manifest.digest,
        "capture": capture_diagnostics(session_dir),
        "state": "incomplete" if incomplete else transitions.report_state,
        "incomplete_final_line": incomplete,
        "journal_records": len(records),
        "rejected_events": rejected,
        "trials": [outcome.payload(private=True) for outcome in outcomes],
        "scored_explicit_responses": len(scored),
        "scored_correct": sum(outcome.correct is True for outcome in scored),
        "correctness_denominator": len(scored),
        "practice_feedback_separate": True,
        "scene_counts": manifest.public_payload()["scene_counts"],
        "calibration_dwells": fit_records,
        "validation_dwells": validation_records,
        "post_validation_dwells": post_records,
        "behavioral_trials_finished": index == len(outcomes),
        "calibration_diagnostics": replay_calibration(
            setup_payload, dwell_records, manifest.limits
        ),
        "calibration_accuracy": "saved engineering diagnostics; no reference-device accuracy claim",
        "provenance": metadata.get("provenance"),
        "timing": (
            "recomputed from raw four-timestamp exchanges and browser events; "
            "not physical onset proof"
        ),
        "recovery": "valid journal prefix only; no resume/replay of scored collection",
    }
