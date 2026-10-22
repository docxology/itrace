from __future__ import annotations

import csv
import math
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path

from .capture_bindings import JournalBindings, Recording
from .jsonio import MAX_JSON, JsonValue, StudyError, integer, number, parse_object

_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "gaze_x_deg",
    "gaze_y_deg",
    "pupil_size",
    "pupil_unit",
    "fps_estimate_hz",
    "quality_json",
]


def _lines(path: Path, partial: dict[str, bool]) -> Iterator[bytes]:
    if path.stat().st_size > 512 * 1024 * 1024:
        raise StudyError("capture", "raw capture file exceeds allocation bound")
    with path.open("rb") as stream:
        for index, line in enumerate(iter(lambda: stream.readline(MAX_JSON + 1), b"")):
            if len(line) > MAX_JSON or index > 1000000:
                raise StudyError("capture", "capture row/count exceeds allocation bound")
            if not line.endswith(b"\n"):
                partial[path.name] = True
                break
            yield line


def _native(path: Path, partial: dict[str, bool]) -> Iterator[tuple[int, float, bool]]:
    lines = _lines(path, partial)
    if next(csv.reader([next(lines, b"").decode()]), []) != _COLUMNS:
        raise StudyError("capture", "unexpected raw capture columns")
    for line in lines:
        values = next(csv.reader([line.decode()], strict=True))
        if len(values) != len(_COLUMNS):
            raise StudyError("capture", "capture row shape invalid")
        row = dict(zip(_COLUMNS, values, strict=True))
        frame = integer(int(row["frame_index"]), "frame_index", 0, 2**53 - 1)
        if str(frame) != row["frame_index"]:
            raise StudyError("capture", "native frame is not an exact integer")
        timestamp = float(row["timestamp_s"])
        if math.isfinite(timestamp):
            number(timestamp, "timestamp_s", -1e12)
        number(float(row["fps_estimate_hz"]), "fps_estimate_hz")
        gaze = [float(row[key]) for key in ("gaze_x_deg", "gaze_y_deg")]
        for value in gaze + ([float(row["pupil_size"])] if row["pupil_size"] else []):
            if not math.isnan(value):
                number(value, "native measurement", -1e12)
        if row["pupil_size"]:
            if row["pupil_unit"] not in {"mm", "relative", "pixels"}:
                raise StudyError("capture", "invalid native pupil unit")
        elif row["pupil_unit"]:
            raise StudyError("capture", "pupil unit without pupil measurement")
        parse_object(row["quality_json"].encode())
        yield frame, timestamp, all(math.isfinite(value) for value in gaze)


def _epochs(path: Path, partial: dict[str, bool]) -> Iterator[Recording]:
    for line in _lines(path, partial):
        yield Recording.parse(dict(parse_object(line)))


def capture_diagnostics(session_dir: Path) -> dict[str, JsonValue]:
    """Check local consistency, retaining recoverable defects; this is not authentication."""
    native_rows, finite_gaze, clock_faults, epoch_rows = 0, 0, 0, 0
    partial: dict[str, bool] = {}
    defects: list[JsonValue] = []
    try:
        bindings = JournalBindings(session_dir)
        for native, epoch in zip_longest(
            _native(session_dir / "capture.csv", partial),
            _epochs(session_dir / "capture_epochs.jsonl", partial),
        ):
            native_rows += native is not None
            epoch_rows += epoch is not None
            if max(native_rows, epoch_rows) > 1000000:
                raise StudyError("capture", "capture count exceeds allocation bound")
            if native is not None:
                finite_gaze += native[2]
            if epoch is not None:
                bindings.check(epoch)
                clock_faults += not epoch.valid
            if native is None or epoch is None:
                continue
            frame, timestamp, _ = native
            if frame != epoch.frame or (
                timestamp != epoch.timestamp
                if math.isfinite(timestamp)
                else epoch.timestamp is not None
            ):
                raise StudyError("capture", "native/sidecar row identity mismatch")
        if len(bindings.faults) != clock_faults:
            if native_rows == epoch_rows and not partial:
                raise StudyError(
                    "capture", "durable clock faults contradict complete capture flags"
                )
            defects.append("unmatched_journal_capture_faults")
        if bindings.incomplete:
            defects.append("incomplete_journal_final_line")
    except (OSError, ValueError, UnicodeError, csv.Error, OverflowError) as exc:
        raise StudyError(
            "capture", "unreadable/malformed raw capture; partial files retained"
        ) from exc
    if partial.get("capture.csv", False):
        defects.append("incomplete_native_final_line")
    if partial.get("capture_epochs.jsonl", False):
        defects.append("incomplete_epoch_final_line")
    if native_rows != epoch_rows:
        defects.append("unmatched_partial_rows")
    if not native_rows:
        defects.append("no_native_capture_rows")
    if clock_faults:
        defects.append("native_clock_faults")
    if native_rows != finite_gaze:
        defects.append("missing_gaze")
    return {
        "native_rows": native_rows,
        "finite_gaze_rows": finite_gaze,
        "epoch_rows": epoch_rows,
        "clock_fault_rows": clock_faults,
        "unmatched_partial_rows": abs(native_rows - epoch_rows),
        "incomplete_native_final_line": partial.get("capture.csv", False),
        "incomplete_epoch_final_line": partial.get("capture_epochs.jsonl", False),
        "missing_gaze_rows": native_rows - finite_gaze,
        "ambiguous_clock_rows": 0,
        "unverifiable_clock_rows": 0,
        "integrity_ok": not defects,
        "integrity_defects": defects,
        "clock_verification": "native/receipt/invocation consistency at exact journal sequence",
        "authenticity": "writable local records; consistency checks are not authentication",
        "integrity_scope": "structural consistency only; capture completeness is not established",
        "pupil_units": "native units retained; relative proxy has no millimetre interpretation",
    }
