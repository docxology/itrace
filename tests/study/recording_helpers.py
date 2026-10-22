import shutil
from pathlib import Path

from itrace.study.jsonio import JsonValue, encoded, integer, parse_object

from .helpers import Harness


def finish_behavior(harness: Harness) -> None:
    for index, key in enumerate(("a", "a", "b")):
        harness.start(index)
        harness.onset(index)
        harness.answer(index, key)


def copy_session(harness: Harness, root: Path) -> Path:
    target = root / "copied" / harness.directory.name
    shutil.copytree(harness.directory, target)
    return target


def rewrite(path: Path, rows: list[dict[str, JsonValue]]) -> None:
    path.write_bytes(b"".join(encoded(row) + b"\n" for row in rows))


def rows(path: Path) -> list[dict[str, JsonValue]]:
    return [dict(parse_object(line)) for line in path.read_bytes().splitlines()]


def capture_prefix(native: bytes, epochs: bytes, sequence: int) -> tuple[bytes, bytes]:
    retained = [
        line
        for line in epochs.splitlines(keepends=True)
        if integer(parse_object(line)["journal_sequence"], "journal_sequence", 1, 200000)
        <= sequence
    ]
    return b"".join(native.splitlines(keepends=True)[: len(retained) + 1]), b"".join(retained)
