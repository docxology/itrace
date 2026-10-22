from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ..capture import CaptureSample
from .jsonio import JsonValue, StudyError, encoded
from .model import StudyManifest
from .paths import validate_private_root
from .provenance import provenance


class Journal:
    def __init__(self, manifest: StudyManifest, output_root: Path) -> None:
        self.sequence = 0
        self.failed = False
        self.session_id = uuid4().hex
        self.path = output_root / self.session_id
        try:
            validate_private_root(output_root)
            output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.mkdir(mode=0o700)
            self.write_new(self.path / "manifest.json", manifest.original_bytes)
            for trial in manifest.trials:
                target = self.path / trial.stimulus.path
                if trial.stimulus.path in {"manifest.json", "metadata.json", "journal.jsonl"}:
                    raise StudyError("path", "reserved bundle path")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    self.write_new(target, trial.stimulus.data)
            self.write_new(self.path / "journal.jsonl", b"")
            self.write_new(self.path / "capture_epochs.jsonl", b"")
            self.write_new(
                self.path / "capture.csv",
                self.csv_row(
                    [
                        "frame_index",
                        "timestamp_s",
                        "gaze_x_deg",
                        "gaze_y_deg",
                        "pupil_size",
                        "pupil_unit",
                        "fps_estimate_hz",
                        "quality_json",
                    ]
                ),
            )
            self.write_new(
                self.path / "metadata.json",
                encoded(
                    {
                        "session_id": self.session_id,
                        "manifest_digest": manifest.digest,
                        "provenance": provenance(),
                        "privacy": "N=1 private derived records; no video/crops",
                    }
                ),
            )
        except OSError as exc:
            self.failed = True
            raise StudyError(
                "storage", "cannot allocate private session; partial files retained"
            ) from exc

    @staticmethod
    def csv_row(values: list[str | int | float]) -> bytes:
        stream = io.StringIO(newline="")
        csv.writer(stream).writerow(values)
        return stream.getvalue().encode()

    @staticmethod
    def write_new(path: Path, data: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        with _directory(path.parent) as descriptor:
            os.fsync(descriptor)

    def _append(self, path: Path, data: bytes) -> None:
        if self.failed:
            raise StudyError("storage", "previous storage failure; session cannot continue")
        try:
            if path.stat().st_size + len(data) > 512 * 1024 * 1024:
                self.failed = True
                raise StudyError("storage", "file allocation bound reached")
            with path.open("ab") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            self.failed = True
            raise StudyError("storage", "durable append failed; partial files retained") from exc

    def append(self, kind: str, payload: Mapping[str, JsonValue], now: float) -> int:
        if self.sequence >= 200000:
            self.failed = True
            raise StudyError("journal", "event allocation bound reached")
        sequence = self.sequence + 1
        self._append(
            self.path / "journal.jsonl",
            encoded(
                {
                    "schema": "itrace.study-journal",
                    "version": 1,
                    "sequence": sequence,
                    "session_id": self.session_id,
                    "server_received_s": now,
                    "kind": kind,
                    "payload": dict(payload),
                }
            )
            + b"\n",
        )
        self.sequence = sequence
        return sequence

    def capture(
        self,
        sample: CaptureSample,
        epoch_id: str,
        receipt: float,
        valid_clock: bool,
        reason: str | None,
        host_recorded_s: float,
    ) -> None:
        pupil = sample.pupil
        quality: dict[str, JsonValue] = {
            key: value if math.isfinite(value) else str(value)
            for key, value in sample.quality.items()
        }
        self._append(
            self.path / "capture.csv",
            self.csv_row(
                [
                    sample.frame_index,
                    sample.timestamp_s,
                    sample.gaze.x,
                    sample.gaze.y,
                    pupil.size if pupil is not None else "",
                    pupil.unit.value if pupil is not None else "",
                    sample.fps_estimate_hz,
                    encoded(quality).decode(),
                ]
            ),
        )
        self._append(
            self.path / "capture_epochs.jsonl",
            encoded(
                {
                    "epoch_id": epoch_id,
                    "frame_index": sample.frame_index,
                    "host_received_s": receipt,
                    "host_recorded_s": host_recorded_s,
                    "journal_sequence": self.sequence,
                    "native_timestamp_s": (
                        sample.timestamp_s if math.isfinite(sample.timestamp_s) else None
                    ),
                    "clock_valid": valid_clock,
                    "reason": reason,
                }
            )
            + b"\n",
        )

    def export_files(
        self, report: Mapping[str, JsonValue], records: list[dict[str, JsonValue]]
    ) -> Path:
        target = self.path / f"export-{uuid4().hex}"
        try:
            target.mkdir(mode=0o700)
            self.write_new(target / "report.json", encoded(report))
            for kind in ("clocks", "calibration", "validation", "post_validation", "events"):
                selected = (
                    [row for row in records if row["kind"] in {"sync", "confirm_sync"}]
                    if kind == "clocks"
                    else [row for row in records if row["kind"] == kind]
                )
                self.write_new(
                    target / f"{kind}.jsonl", b"".join(encoded(row) + b"\n" for row in selected)
                )
            trials = report["trials"]
            if not isinstance(trials, list):
                raise StudyError("report", "trial list missing")
            rows = [self.csv_row(["trial_id", "role", "outcome", "key", "correct", "rt_s"])]
            for trial in trials:
                if not isinstance(trial, dict):
                    raise StudyError("report", "invalid trial")
                rows.append(
                    self.csv_row(
                        [
                            str(trial.get(key, ""))
                            for key in ("trial_id", "role", "outcome", "key", "correct", "rt_s")
                        ]
                    )
                )
            self.write_new(target / "trials.csv", b"".join(rows))
            self.write_new(target / "responses.csv", b"".join(rows))
            checksums: dict[str, JsonValue] = {}
            for path in sorted(self.path.rglob("*")):
                if path.is_file() and path.name != "checksums.json":
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(65536), b""):
                            digest.update(chunk)
                    checksums[path.relative_to(self.path).as_posix()] = digest.hexdigest()
            self.write_new(target / "checksums.json", encoded(checksums))
        except OSError as exc:
            raise StudyError("storage", "export failed; partial export retained") from exc
        return target


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        yield descriptor
    finally:
        os.close(descriptor)
