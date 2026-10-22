"""Exact replay of capture invocations against acknowledged journal prefixes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .journal_read import read_journal
from .jsonio import (
    MAX_JSON,
    JsonValue,
    StudyError,
    fields,
    integer,
    mapping,
    number,
    parse_object,
    text,
)
from .manifest import read_bounded

_REASON: Final = "native clock/frame discontinuity or inactive epoch"


@dataclass(frozen=True, slots=True)
class Recording:
    epoch: str
    frame: int
    receipt: float
    recorded: float
    sequence: int
    timestamp: float | None
    valid: bool

    @classmethod
    def parse(cls, row: dict[str, JsonValue]) -> Recording:
        if not {"host_recorded_s", "journal_sequence"} <= row.keys():
            raise StudyError(
                "capture_schema",
                "incompatible capture recording epoch; host_recorded_s/journal_sequence required",
            )
        fields(
            row,
            "epoch_id frame_index host_received_s host_recorded_s journal_sequence "
            "native_timestamp_s clock_valid reason",
        )
        if type(row["clock_valid"]) is not bool or row["reason"] != (
            None if row["clock_valid"] else _REASON
        ):
            raise StudyError("capture", "invalid clock flag/reason")
        return cls(
            text(row["epoch_id"], "epoch_id", identifier=True),
            integer(row["frame_index"], "frame_index", 0, 2**53 - 1),
            number(row["host_received_s"], "host_received_s"),
            number(row["host_recorded_s"], "host_recorded_s"),
            integer(row["journal_sequence"], "journal_sequence", 1, 200000),
            None
            if row["native_timestamp_s"] is None
            else number(row["native_timestamp_s"], "native_timestamp_s", -1e12),
            row["clock_valid"] is True,
        )


@dataclass(frozen=True, slots=True)
class CaptureState:
    epoch: str = ""
    active: bool = False
    generation: int = 0


class JournalBindings:
    """Accumulate ordered recording checks; state changes reset native progression."""

    def __init__(self, session_dir: Path) -> None:
        metadata = parse_object(read_bounded(session_dir / "metadata.json", MAX_JSON))
        if metadata.get("session_id") != session_dir.name:
            raise StudyError("capture", "metadata session identity mismatch")
        records, self.incomplete = read_journal(session_dir)
        self.states = [CaptureState()]
        self.lower = [0.0]
        self.upper = [math.inf] * (len(records) + 1)
        self.faults: dict[int, float] = {}
        self.matched_faults: set[int] = set()
        self.previous: tuple[float, int, int] | None = None
        self.last_recorded = 0.0
        self.last_sequence = 0
        for index, record in enumerate(records, 1):
            state, when = self.states[-1], self.lower[-1]
            received = number(record["server_received_s"], "server_received_s")
            payload = mapping(record["payload"], "payload")
            # Fault receipts are native-arrival times, not host invocation times.
            if record["kind"] != "capture_fault":
                if received < when:
                    raise StudyError("capture", "journal host clock reversed")
                when = received
            if record["kind"] == "capture_state":
                fields(payload, "active epoch_id reason")
                epoch = text(payload["epoch_id"], "epoch_id", identifier=True)
                if type(payload["active"]) is not bool:
                    raise StudyError("capture", "invalid journal capture active flag")
                if payload["reason"] is not None:
                    text(payload["reason"], "reason")
                state = CaptureState(
                    epoch, payload["active"] is True, state.generation + (epoch != state.epoch)
                )
            elif record["kind"] == "capture_fault":
                fields(payload, "reason")
                if payload["reason"] != _REASON:
                    raise StudyError("capture", "unexpected capture fault reason")
                self.faults[index] = received
            self.states.append(state)
            self.lower.append(when)
        for index in range(len(records) - 1, -1, -1):
            self.upper[index] = (
                self.upper[index + 1]
                if records[index]["kind"] == "capture_fault"
                else number(records[index]["server_received_s"], "server_received_s")
            )

    def check(self, row: Recording) -> None:
        if (
            row.sequence >= len(self.states)
            or row.sequence < self.last_sequence
            or row.recorded < self.last_recorded
        ):
            raise StudyError("capture", "recording sequence/time outside acknowledged order")
        if not self.lower[row.sequence] <= row.recorded <= self.upper[row.sequence]:
            raise StudyError("capture", "recording time contradicts journal sequence")
        state = self.states[row.sequence]
        previous = self.previous
        valid = (
            row.timestamp is not None
            and 0 <= row.recorded - row.receipt <= 5
            and 0 <= row.receipt - row.timestamp <= 5
            and state.epoch == row.epoch
            and state.active
            and (
                previous is None
                or previous[2] != state.generation
                or (row.timestamp > previous[0] and row.frame > previous[1])
            )
        )
        if row.valid != valid:
            raise StudyError("capture", "declared clock flag contradicts bound invocation")
        self.last_sequence, self.last_recorded = row.sequence, row.recorded
        if row.valid:
            assert row.timestamp is not None
            self.previous = row.timestamp, row.frame, state.generation
        else:
            fault = row.sequence + 1
            if self.faults.get(fault) != row.receipt or fault in self.matched_faults:
                raise StudyError("capture", "clock fault lacks matching next journal receipt")
            self.matched_faults.add(fault)
            self.last_sequence = fault
