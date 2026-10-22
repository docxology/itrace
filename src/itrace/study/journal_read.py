from __future__ import annotations

from pathlib import Path

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


def read_journal(session_dir: Path) -> tuple[list[dict[str, JsonValue]], bool]:
    records: list[dict[str, JsonValue]] = []
    incomplete = False
    try:
        path = session_dir / "journal.jsonl"
        if path.stat().st_size > 512 * 1024 * 1024:
            raise StudyError("journal", "journal exceeds allocation bound")
        with path.open("rb") as stream:
            while line := stream.readline(MAX_JSON + 1):
                if len(line) > MAX_JSON:
                    raise StudyError("journal", "journal line exceeds allocation bound")
                if not line.endswith(b"\n"):
                    incomplete = True
                    break
                row = dict(parse_object(line))
                fields(row, "schema version sequence session_id server_received_s kind payload")
                if (
                    row["schema"] != "itrace.study-journal"
                    or integer(row["version"], "version", 1, 1) != 1
                ):
                    raise StudyError("journal", "unsupported journal schema")
                if integer(row["sequence"], "sequence", 1, 200000) != len(records) + 1:
                    raise StudyError("journal", "sequence gap or duplicate")
                if text(row["session_id"], "session_id") != session_dir.name:
                    raise StudyError("journal", "session identity mismatch")
                number(row["server_received_s"], "server_received_s")
                text(row["kind"], "kind")
                mapping(row["payload"], "payload")
                records.append(row)
    except OSError as exc:
        raise StudyError("storage", "cannot read journal; partial directory retained") from exc
    return records, incomplete
