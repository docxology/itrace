"""CaptureBackend conformance tests for non-camera adapters."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from itrace import capture
from itrace.types import GazeSample, PupilSample, PupilUnit


@dataclass(frozen=True, slots=True)
class SyntheticFrameBackend:
    """Deterministic backend shaped like recorded-video or WebRTC adapters."""

    n_samples: int = 6

    def frames(self, max_frames: int | None = None) -> Iterable[capture.CaptureSample]:
        limit = self.n_samples if max_frames is None else min(self.n_samples, max_frames)
        for idx in range(limit):
            yield capture.CaptureSample(
                frame_index=idx,
                timestamp_s=idx / 30.0,
                gaze=GazeSample(t=idx / 30.0, x=0.25 * idx, y=-0.10 * idx),
                pupil=PupilSample(t=idx / 30.0, size=3.0 + 0.01 * idx, unit=PupilUnit.RELATIVE),
                fps_estimate_hz=30.0,
                quality={"face_detected": 1.0, "synthetic_confidence": 0.99},
            )

    def live_frames(self, max_frames: int | None = None) -> Iterable[capture.LiveFrameSample]:
        eye_box = capture.EyeBox(x=4, y=5, width=10, height=8)
        for sample in self.frames(max_frames=max_frames):
            yield capture.LiveFrameSample(
                capture=sample,
                frame_width=30,
                frame_height=20,
                eye_box=eye_box,
                eye_crop_jpeg="data:image/jpeg;base64,ZmFrZQ==",
            )


def test_synthetic_frame_backend_satisfies_runtime_protocol() -> None:
    backend = SyntheticFrameBackend()

    assert isinstance(backend, capture.CaptureBackend)


def test_capture_backend_frames_convert_to_streams_and_csv(tmp_path) -> None:
    backend = SyntheticFrameBackend(n_samples=5)

    samples = list(backend.frames(max_frames=4))
    gaze, pupil = capture.samples_to_streams(samples)
    records_path = capture.write_capture_records_csv(samples, tmp_path / "capture_records.csv")

    assert len(samples) == 4
    assert np.all(np.diff(gaze.t) > 0.0)
    assert np.allclose(gaze.x, [0.0, 0.25, 0.50, 0.75])
    assert len(pupil) == 4
    rows = list(csv.DictReader(records_path.open()))
    assert rows[0]["quality_face_detected"] == "1.0"
    assert rows[0]["quality_synthetic_confidence"] == "0.99"


def test_capture_backend_live_frames_preserve_visual_context() -> None:
    backend = SyntheticFrameBackend(n_samples=3)

    frames = list(backend.live_frames(max_frames=2))

    assert len(frames) == 2
    assert frames[0].capture.frame_index == 0
    assert frames[0].frame_width == 30
    assert frames[0].frame_height == 20
    assert frames[0].eye_box.to_dict() == {"x": 4, "y": 5, "width": 10, "height": 8}
    assert frames[0].eye_crop_jpeg.startswith("data:image/jpeg;base64,")
