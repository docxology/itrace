from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from itrace.study import JsonValue, StudyError, load_manifest
from itrace.study.jsonio import mapping, sequence

from .helpers import chunk, manifest_data, write_manifest


def test_manifest_rejects_compression_bomb_when_aggregate_decode_exceeds_cap(
    tmp_path: Path,
) -> None:
    data = manifest_data(tmp_path)
    base = mapping(sequence(data["trials"], "trials")[0], "trial")
    trials: list[JsonValue] = []
    for index in range(6):
        compressor = zlib.compressobj()
        row = b"\x00" + bytes([index, 0, 0]) * 4096
        compressed = b"".join(compressor.compress(row) for _ in range(4096)) + compressor.flush()
        raw = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 4096, 4096, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )
        name = f"large{index}.png"
        (tmp_path / name).write_bytes(raw)
        trials.append(
            {
                **base,
                "trial_id": f"large{index}",
                "role": "scored",
                "stimulus": {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "width": 4096,
                    "height": 4096,
                },
            }
        )
    data["trials"] = trials
    with pytest.raises(StudyError, match="decoded PNG allocation"):
        load_manifest(write_manifest(tmp_path, data))
