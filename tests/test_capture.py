"""Capture-orchestrator tests (ISC-43..45), with or without hardware deps."""

from __future__ import annotations

import base64
import csv
import sys
from importlib.util import find_spec

import numpy as np
import pytest

from itrace import capture, cli, io
from itrace.capture import CaptureSample
from itrace.types import GazeSample, PupilUnit


def _landmarks_with_iris(offset_x: float) -> np.ndarray:
    """Build a (478, 3) normalised landmark array; right iris shifted by offset_x."""
    lm = np.full((478, 3), 0.5, dtype=np.float64)
    # right eye corners (33 outer, 133 inner) define the half-width
    lm[33] = (0.40, 0.50, 0.0)
    lm[133] = (0.50, 0.50, 0.0)
    lm[362] = (0.50, 0.50, 0.0)
    lm[263] = (0.60, 0.50, 0.0)
    eye_center_x = 0.45
    iris_offsets = [(0.0, 0.0), (0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01)]
    for idx, (dx, dy) in zip(capture.RIGHT_IRIS, iris_offsets, strict=True):
        lm[idx] = (eye_center_x + offset_x + dx, 0.50 + dy, 0.0)
    for idx, (dx, dy) in zip(capture.LEFT_IRIS, iris_offsets, strict=True):
        lm[idx] = (0.55 + offset_x + dx, 0.50 + dy, 0.0)
    return lm


def test_iris_landmarks_to_sample_returns_gaze() -> None:  # ISC-43
    sample = capture.iris_landmarks_to_sample(_landmarks_with_iris(0.0), t=0.1)
    assert isinstance(sample, GazeSample)
    assert sample.t == 0.1
    assert sample.x == pytest.approx(0.0, abs=1e-6)  # centred iris -> ~0 yaw


def test_iris_offset_monotonic_in_yaw() -> None:  # ISC-43
    left = capture.iris_landmarks_to_sample(_landmarks_with_iris(-0.02), t=0.0)
    center = capture.iris_landmarks_to_sample(_landmarks_with_iris(0.0), t=0.0)
    right = capture.iris_landmarks_to_sample(_landmarks_with_iris(0.02), t=0.0)
    assert left.x < center.x < right.x


def test_iris_landmarks_to_pupil_sample_is_relative_proxy() -> None:
    pupil = capture.iris_landmarks_to_pupil_sample(_landmarks_with_iris(0.0), t=0.2)
    assert pupil.t == 0.2
    assert pupil.unit is PupilUnit.RELATIVE
    assert pupil.size > 0.0


def test_landmarks_to_capture_sample_carries_quality() -> None:
    sample = capture.iris_landmarks_to_capture_sample(
        _landmarks_with_iris(0.0),
        t=0.3,
        frame_index=7,
        fps_estimate_hz=29.0,
    )
    assert isinstance(sample, CaptureSample)
    assert sample.frame_index == 7
    assert sample.gaze.t == sample.timestamp_s == 0.3
    assert sample.pupil is not None
    assert sample.quality["pupil_proxy_relative"] == 1.0


def test_eye_box_from_landmarks_bounds_both_eyes_and_clamps() -> None:
    box = capture.eye_box_from_landmarks(
        _landmarks_with_iris(0.0),
        frame_width=640,
        frame_height=480,
    )

    assert box.x >= 0
    assert box.y >= 0
    assert box.x + box.width <= 640
    assert box.y + box.height <= 480
    assert box.width > box.height
    assert box.width >= 128

    edge_landmarks = _landmarks_with_iris(0.0)
    edge_landmarks[:, 0] = np.clip(edge_landmarks[:, 0] - 0.45, 0.0, 1.0)
    edge_box = capture.eye_box_from_landmarks(
        edge_landmarks,
        frame_width=64,
        frame_height=48,
        padding_fraction=1.0,
    )
    assert edge_box.x == 0
    assert edge_box.width <= 64


def test_eye_crop_jpeg_base64_uses_supplied_encoder() -> None:
    class FakeCv2:
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def imencode(_ext: str, image: np.ndarray, params: list[int]) -> tuple[bool, np.ndarray]:
            assert image.shape == (12, 20, 3)
            assert params == [FakeCv2.IMWRITE_JPEG_QUALITY, 90]
            return True, np.frombuffer(b"jpeg-bytes", dtype=np.uint8)

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    uri = capture.encode_eye_crop_jpeg_base64(
        frame,
        capture.EyeBox(x=5, y=6, width=20, height=12),
        cv2_module=FakeCv2,
        jpeg_quality=90,
    )

    prefix = "data:image/jpeg;base64,"
    assert uri.startswith(prefix)
    assert base64.b64decode(uri.removeprefix(prefix)) == b"jpeg-bytes"


def test_capture_samples_write_gaze_and_pupil_csv(tmp_path) -> None:
    samples = [
        capture.iris_landmarks_to_capture_sample(
            _landmarks_with_iris(offset),
            t=i / 30.0,
            frame_index=i,
        )
        for i, offset in enumerate((0.0, 0.01, 0.02))
    ]
    gaze_path = tmp_path / "gaze.csv"
    pupil_path = tmp_path / "pupil.csv"

    n = cli.write_capture_samples(samples, gaze_path, pupil_path)

    assert n == 3
    gaze = io.read_gaze_csv(gaze_path)
    pupil = io.read_pupil_csv(pupil_path)
    assert len(gaze) == 3
    assert len(pupil) == 3
    assert pupil.unit is PupilUnit.RELATIVE


def test_capture_records_csv_includes_quality_and_timing(tmp_path) -> None:
    samples = [
        capture.iris_landmarks_to_capture_sample(
            _landmarks_with_iris(offset),
            t=0.1 + i / 30.0,
            frame_index=i,
            fps_estimate_hz=30.0,
        )
        for i, offset in enumerate((0.0, 0.01))
    ]
    records_path = tmp_path / "capture_records.csv"

    cli.write_capture_records_csv(samples, records_path)

    rows = list(csv.DictReader(records_path.open()))
    assert len(rows) == 2
    assert rows[0]["frame_index"] == "0"
    assert float(rows[0]["timestamp_s"]) == pytest.approx(0.1)
    assert "gaze_x_deg" in rows[0]
    assert rows[0]["pupil_unit"] == PupilUnit.RELATIVE.value
    assert rows[0]["quality_face_detected"] == "1.0"


def test_iris_landmarks_no_mediapipe_imported() -> None:  # ISC-43/ISC-6
    capture.iris_landmarks_to_sample(_landmarks_with_iris(0.0), t=0.0)
    assert "mediapipe" not in sys.modules
    assert "cv2" not in sys.modules


def test_iris_landmarks_rejects_bad_shape_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="N, >=2"):
        capture.iris_landmarks_to_sample(np.zeros((478, 1)), t=0.0)
    bad = _landmarks_with_iris(0.0)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        capture.iris_landmarks_to_sample(bad, t=0.0)


def test_require_capture_deps_raises_clear_error() -> None:  # ISC-44
    if find_spec("cv2") is not None and find_spec("mediapipe") is not None:
        cv2, mp = capture._require_capture_deps()
        assert cv2 is not None
        assert hasattr(mp, "solutions")
        assert hasattr(mp.solutions, "face_mesh")
    else:
        with pytest.raises(RuntimeError, match="capture"):
            capture._require_capture_deps()


def test_webcam_source_construction_errors_without_deps() -> None:  # ISC-44
    if find_spec("cv2") is not None and find_spec("mediapipe") is not None:
        source = capture.WebcamSource()
        assert hasattr(source, "frames")
    else:
        with pytest.raises(RuntimeError, match="uv sync --extra capture"):
            capture.WebcamSource()


def test_capture_exposes_frame_interface() -> None:  # ISC-45
    assert hasattr(capture.WebcamSource, "frames")
    assert callable(capture.iris_landmarks_to_sample)
