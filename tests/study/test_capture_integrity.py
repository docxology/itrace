from __future__ import annotations

import csv
from pathlib import Path

import pytest

from itrace.study import StudyError, reconstruct

from .helpers import Harness
from .recording_helpers import copy_session


@pytest.mark.parametrize("field", ["fps_estimate_hz", "pupil_size", "pupil_unit"])
def test_capture_rejects_when_native_numeric_or_unit_column_corrupted(
    harness: Harness, tmp_path: Path, field: str
) -> None:
    harness.sample(1, 2)
    target = copy_session(harness, tmp_path)
    path = target / "capture.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        records = list(reader)
    assert header is not None
    records[0][field] = "corrupt"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(StudyError):
        reconstruct(target)
