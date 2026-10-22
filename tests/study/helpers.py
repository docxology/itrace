from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from itrace.capture import CaptureSample
from itrace.study import JsonValue, StudySession, load_manifest
from itrace.study.jsonio import mapping, number, text
from itrace.types import GazeSample, PupilSample


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png(color: int = 0, filter_byte: int = 0) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((bytes([filter_byte]) + bytes([color, 40, 80]) * 16) * 16))
        + chunk(b"IEND", b"")
    )


def manifest_data(root: Path, *, rest: float = 0) -> dict[str, JsonValue]:
    trials: list[JsonValue] = []
    for index, role in enumerate(("practice", "scored", "scored")):
        raw = png(index * 20)
        name = f"image{index}.png"
        (root / name).write_bytes(raw)
        trials.append(
            {
                "trial_id": f"trial{index}",
                "role": role,
                "block_id": role,
                "task_id": "choice",
                "family_id": role + "-family",
                "query": "Choose a shape",
                "stimulus": {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "width": 16,
                    "height": 16,
                },
                "options": [{"key": "a", "label": "Circle"}, {"key": "b", "label": "Square"}],
                "correct_key": "a" if index != 2 else "b",
                "deadline_s": 3,
                "rest_s": rest if index == 0 else 0,
            }
        )
    return {
        "schema": "itrace.private-study",
        "version": 1,
        "study_id": "synthetic",
        "title": "Synthetic contract control",
        "trials": trials,
        "limits": {
            "max_sync_error_s": 0.05,
            "sync_max_age_s": 60,
            "max_event_lag_s": 2,
            "max_validation_error_fraction": 0.15,
            "settle_s": 0.5,
            "collect_s": 1,
            "min_samples": 10,
            "min_valid_fraction": 0.6,
        },
    }


def write_manifest(root: Path, data: dict[str, JsonValue] | None = None) -> Path:
    path = root / "study.json"
    path.write_text(json.dumps(data if data is not None else manifest_data(root)))
    return path


def setup_payload() -> dict[str, JsonValue]:
    return {
        "participant_id": "synthetic-only",
        "device_id": "numeric-stream",
        "display_width_m": 0.5,
        "display_height_m": 0.3,
        "distance_m": 0.6,
        "screen_width_px": 1000,
        "screen_height_px": 800,
        "dpr": 1,
        "viewport_width_px": 1000,
        "viewport_height_px": 800,
        "browser": "synthetic",
        "client_epoch": "browser1",
        "storage_acknowledged": True,
        "geometry_confirmed": True,
        "fullscreen": True,
    }


@dataclass
class Clock:
    value: float = 100.0

    def __call__(self) -> float:
        return self.value


class Harness:
    def __init__(self, root: Path, *, rest: float = 0) -> None:
        self.clock = Clock()
        self.manifest = load_manifest(write_manifest(root, manifest_data(root, rest=rest)))
        self.session = StudySession(self.manifest, root / "private", self.clock)
        self.session.prepare(setup_payload())
        self.sync_id = ""
        self.serial = 0
        self.frame_index = 0
        self.sync()
        self.session.capture_state(True, "capture1")

    @property
    def directory(self) -> Path:
        return Path(text(self.session.status()["output_dir"], "output_dir"))

    def sync(self) -> None:
        answer = self.session.sync(
            {"client_epoch": "browser1", "client_send_ms": self.clock.value * 1000}
        )
        self.sync_id = text(answer["sync_id"], "sync_id")
        self.session.confirm_sync(
            {
                "sync_id": self.sync_id,
                "client_epoch": "browser1",
                "client_received_ms": self.clock.value * 1000,
            }
        )

    def event_payload(self, kind: str, **extra: JsonValue) -> dict[str, JsonValue]:
        self.serial += 1
        return {
            "event_id": f"event{self.serial}",
            "client_epoch": "browser1",
            "sync_id": self.sync_id,
            "browser_ms": self.clock.value * 1000,
            "kind": kind,
            **extra,
        }

    def event(self, kind: str, **extra: JsonValue) -> dict[str, JsonValue]:
        return self.session.event(self.event_payload(kind, **extra))

    def sample(
        self,
        x: float,
        y: float,
        *,
        timestamp: float | None = None,
        face: float = 1,
        epoch: str = "capture1",
    ) -> None:
        native = self.clock.value if timestamp is None else timestamp
        self.frame_index += 1
        self.session.capture(
            CaptureSample(
                self.frame_index,
                native,
                GazeSample(native, x, y),
                PupilSample(native, 0.2),
                30,
                {"face_detected": face},
            ),
            epoch,
            self.clock.value,
        )

    def target(self, *, offset: float = 0, finite: bool = True, count: int = 20) -> None:
        target = mapping(self.session.status()["target"], "target")
        x = number(target["x_fraction"], "x") * 1000
        y = number(target["y_fraction"], "y") * 800
        self.event(
            "target_onset",
            target_id=target["target_id"],
            position={"x": x, "y": y},
            rect={"x": x - 5, "y": y - 5, "width": 10, "height": 10},
        )
        onset = self.clock.value
        for index in range(count):
            self.clock.value = onset + 0.51 + index * 0.9 / max(count - 1, 1)
            self.sample((x - 100) / 20 + offset if finite else float("nan"), (y - 50) / -15)
        self.clock.value = onset + 1.51
        self.event("target_collect", target_id=target["target_id"])
        self.clock.value += 0.01

    def ready(self, *, validation_offset: float = 0) -> None:
        for index in range(14):
            self.target(offset=validation_offset if index >= 9 else 0)

    def start(self, index: int = 0) -> None:
        self.session.start_trial({"trial_id": f"trial{index}"})

    def onset(self, index: int = 0) -> None:
        self.event(
            "trial_onset",
            trial_id=f"trial{index}",
            presentation="raf",
            rect={"x": 100, "y": 100, "width": 500, "height": 500},
        )

    def answer(self, index: int = 0, key: str = "a") -> None:
        self.clock.value += 0.5
        self.event("response", trial_id=f"trial{index}", key=key)
