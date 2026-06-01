"""Thin, optional webcam/MediaPipe capture orchestrator.

This is the only hardware-facing module. Two hard rules keep the rest of the
package CI-able:

1. **Lazy imports.** ``cv2`` / ``mediapipe`` are imported *inside* methods, never
   at module load, so ``import itrace.capture`` works with neither installed.
2. **Pure landmark math.** :func:`iris_landmarks_to_sample` turns an
   already-extracted array of normalised face landmarks into a
   :class:`~itrace.types.GazeSample` with no third-party dependency, so the
   landmark-to-gaze logic is fully unit-testable without a camera.

MediaPipe Face Mesh (refine_landmarks=True) iris indices default to
468-472 (right) and 473-477 (left); eye-corner indices 33/133 (right) and
362/263 (left) bound the horizontal extent for offset normalisation.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .geometry import iris_offset_to_gaze_angle
from .types import FloatArray, GazeSample, PupilSample, PupilUnit

# MediaPipe Face Mesh canonical indices (refine_landmarks=True).
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)
RIGHT_EYE_CORNERS = (33, 133)  # outer, inner
LEFT_EYE_CORNERS = (362, 263)  # inner, outer
EYE_BOX_LANDMARKS = RIGHT_IRIS + LEFT_IRIS + RIGHT_EYE_CORNERS + LEFT_EYE_CORNERS


@dataclass(frozen=True, slots=True)
class EyeBox:
    """Pixel-space bounding box around both eyes in one camera frame."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        """JSON-friendly representation used by the live HTML orchestrator."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class CaptureSample:
    """One webcam-derived sample with gaze, optional pupil proxy and quality."""

    frame_index: int
    timestamp_s: float
    gaze: GazeSample
    pupil: PupilSample | None = None
    fps_estimate_hz: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveFrameSample:
    """Capture sample plus visual context for the live HTML orchestrator."""

    capture: CaptureSample
    frame_width: int
    frame_height: int
    eye_box: EyeBox
    eye_crop_jpeg: str


class CaptureBackend(Protocol):
    """Minimal protocol for webcam, video-file, WebRTC, or synthetic backends."""

    def frames(self, max_frames: int | None = None) -> Iterable[CaptureSample]:
        """Yield analysis-ready capture samples."""
        ...

    def live_frames(self, max_frames: int | None = None) -> Iterable[LiveFrameSample]:
        """Yield capture samples plus eye-crop visual context."""
        ...


def _landmarks_array(landmarks: FloatArray | list[list[float]]) -> FloatArray:
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        msg = "landmarks must be an (N, >=2) array of normalised coordinates"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "landmarks contain non-finite values"
        raise ValueError(msg)
    return arr


def eye_box_from_landmarks(
    landmarks: FloatArray | list[list[float]],
    *,
    frame_width: int,
    frame_height: int,
    padding_fraction: float = 0.70,
    min_padding_px: int = 12,
    indices: tuple[int, ...] = EYE_BOX_LANDMARKS,
) -> EyeBox:
    """Return a clamped pixel-space box around both eyes from normalised landmarks."""
    if frame_width <= 0 or frame_height <= 0:
        msg = "frame_width and frame_height must be positive"
        raise ValueError(msg)
    if padding_fraction < 0.0:
        msg = "padding_fraction must be non-negative"
        raise ValueError(msg)
    arr = _landmarks_array(landmarks)
    points = arr[list(indices), :2]
    xs = points[:, 0] * frame_width
    ys = points[:, 1] * frame_height
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)
    pad_x = max(span_x * padding_fraction, float(min_padding_px))
    pad_y = max(span_y * padding_fraction, float(min_padding_px))

    x0 = max(int(np.floor(x_min - pad_x)), 0)
    y0 = max(int(np.floor(y_min - pad_y)), 0)
    x1 = min(int(np.ceil(x_max + pad_x)), frame_width)
    y1 = min(int(np.ceil(y_max + pad_y)), frame_height)
    if x1 <= x0:
        x1 = min(x0 + 1, frame_width)
    if y1 <= y0:
        y1 = min(y0 + 1, frame_height)
    return EyeBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def crop_frame_to_eye_box(frame: FloatArray, eye_box: EyeBox) -> FloatArray:
    """Crop a frame to an :class:`EyeBox`, validating bounds and dimensions."""
    arr = np.asarray(frame)
    if arr.ndim not in {2, 3}:
        msg = "frame must be a 2-D or 3-D image array"
        raise ValueError(msg)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if eye_box.x < 0 or eye_box.y < 0 or eye_box.width <= 0 or eye_box.height <= 0:
        msg = "eye_box must have non-negative origin and positive size"
        raise ValueError(msg)
    x1 = min(eye_box.x + eye_box.width, w)
    y1 = min(eye_box.y + eye_box.height, h)
    if eye_box.x >= x1 or eye_box.y >= y1:
        msg = "eye_box lies outside frame"
        raise ValueError(msg)
    return arr[eye_box.y : y1, eye_box.x : x1]


def encode_eye_crop_jpeg_base64(
    frame: FloatArray,
    eye_box: EyeBox,
    *,
    cv2_module: Any | None = None,
    jpeg_quality: int = 85,
) -> str:
    """JPEG-encode an eye crop as a browser-ready data URI."""
    if not 1 <= jpeg_quality <= 100:
        msg = "jpeg_quality must be in [1, 100]"
        raise ValueError(msg)
    cv2 = cv2_module if cv2_module is not None else _require_cv2()
    crop = crop_frame_to_eye_box(frame, eye_box)
    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        msg = "failed to JPEG-encode eye crop"
        raise RuntimeError(msg)
    payload = base64.b64encode(bytes(encoded)).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _default_eye_box(frame_width: int, frame_height: int) -> EyeBox:
    """Return a stable face-missing crop area where eyes usually appear."""
    width = max(1, round(frame_width * 0.72))
    height = max(1, round(frame_height * 0.34))
    return EyeBox(
        x=max(0, (frame_width - width) // 2),
        y=max(0, round(frame_height * 0.20)),
        width=min(width, frame_width),
        height=min(height, frame_height),
    )


def _eye_gaze(
    landmarks: FloatArray,
    iris_idx: tuple[int, ...],
    corner_idx: tuple[int, int],
    max_angle_deg: float,
) -> tuple[float, float]:
    """Return (yaw_deg, pitch_deg) for one eye from normalised landmarks."""
    iris = landmarks[list(iris_idx)]
    iris_center = iris.mean(axis=0)
    c0 = landmarks[corner_idx[0]]
    c1 = landmarks[corner_idx[1]]
    eye_center = (c0 + c1) / 2.0
    half_width = np.linalg.norm(c1[:2] - c0[:2]) / 2.0
    if half_width <= 0:
        msg = "degenerate eye width; corner landmarks coincide"
        raise ValueError(msg)
    offset_x = (iris_center[0] - eye_center[0]) / half_width
    offset_y = (iris_center[1] - eye_center[1]) / half_width
    yaw = float(iris_offset_to_gaze_angle(offset_x, max_angle_deg))
    # screen-y is down; iris_offset_to_gaze_angle keeps sign, geometry layer
    # negates for direction. Here pitch is returned in the gaze-up convention.
    pitch = float(-iris_offset_to_gaze_angle(offset_y, max_angle_deg))
    return yaw, pitch


def iris_landmarks_to_sample(
    landmarks: FloatArray | list[list[float]],
    t: float,
    *,
    max_angle_deg: float = 25.0,
    right_iris: tuple[int, ...] = RIGHT_IRIS,
    left_iris: tuple[int, ...] = LEFT_IRIS,
    right_corners: tuple[int, int] = RIGHT_EYE_CORNERS,
    left_corners: tuple[int, int] = LEFT_EYE_CORNERS,
) -> GazeSample:
    """Convert one frame's normalised face landmarks to a gaze sample (deg).

    ``landmarks`` is an ``(N, 2)`` or ``(N, 3)`` array of normalised coordinates
    (MediaPipe convention). The binocular gaze is the mean of the two eyes'
    estimates. No MediaPipe import occurs here -- the array is plain floats.
    """
    arr = _landmarks_array(landmarks)
    ryaw, rpitch = _eye_gaze(arr, right_iris, right_corners, max_angle_deg)
    lyaw, lpitch = _eye_gaze(arr, left_iris, left_corners, max_angle_deg)
    return GazeSample(t=t, x=(ryaw + lyaw) / 2.0, y=(rpitch + lpitch) / 2.0)


def _iris_radius_over_half_width(
    landmarks: FloatArray,
    iris_idx: tuple[int, ...],
    corner_idx: tuple[int, int],
) -> float:
    iris = landmarks[list(iris_idx)]
    center = iris.mean(axis=0)
    c0 = landmarks[corner_idx[0]]
    c1 = landmarks[corner_idx[1]]
    half_width = float(np.linalg.norm(c1[:2] - c0[:2]) / 2.0)
    if half_width <= 0.0:
        msg = "degenerate eye width; corner landmarks coincide"
        raise ValueError(msg)
    radius = float(np.mean(np.linalg.norm(iris[:, :2] - center[:2], axis=1)))
    return radius / half_width


def iris_landmarks_to_pupil_sample(
    landmarks: FloatArray | list[list[float]],
    t: float,
    *,
    right_iris: tuple[int, ...] = RIGHT_IRIS,
    left_iris: tuple[int, ...] = LEFT_IRIS,
    right_corners: tuple[int, int] = RIGHT_EYE_CORNERS,
    left_corners: tuple[int, int] = LEFT_EYE_CORNERS,
) -> PupilSample:
    """Return a relative webcam pupil/iris proxy from MediaPipe iris geometry.

    MediaPipe Face Mesh does not provide calibrated millimetres or a true pupil
    boundary. This proxy is the mean iris-landmark radius divided by eye
    half-width, so callers must treat it as relative and camera-dependent.
    """
    arr = _landmarks_array(landmarks)
    right = _iris_radius_over_half_width(arr, right_iris, right_corners)
    left = _iris_radius_over_half_width(arr, left_iris, left_corners)
    return PupilSample(t=t, size=(right + left) / 2.0, unit=PupilUnit.RELATIVE)


def iris_landmarks_to_capture_sample(
    landmarks: FloatArray | list[list[float]],
    *,
    t: float,
    frame_index: int,
    fps_estimate_hz: float = 0.0,
    max_angle_deg: float = 25.0,
) -> CaptureSample:
    """Convert one frame's landmarks into a timestamped capture sample."""
    gaze = iris_landmarks_to_sample(landmarks, t=t, max_angle_deg=max_angle_deg)
    pupil = iris_landmarks_to_pupil_sample(landmarks, t=t)
    return CaptureSample(
        frame_index=frame_index,
        timestamp_s=t,
        gaze=gaze,
        pupil=pupil,
        fps_estimate_hz=fps_estimate_hz,
        quality={
            "face_detected": 1.0,
            "pupil_proxy_relative": 1.0,
        },
    )


@dataclass(slots=True)
class WebcamSource:
    """Live webcam capture + MediaPipe landmark extraction (optional backend).

    Construction validates that the ``capture`` extra is installed and surfaces
    a clear, actionable error otherwise -- callers never see a bare
    ``ModuleNotFoundError`` from deep inside the stack.
    """

    camera_index: int = 0
    max_angle_deg: float = 25.0
    _cv2: object = field(default=None, repr=False)
    _mp_face: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._cv2, mp = _require_capture_deps()
        self._mp_face = mp.solutions.face_mesh.FaceMesh

    def frames(self, max_frames: int | None = None) -> Iterator[CaptureSample]:  # pragma: no cover
        """Yield timestamped capture samples from the live camera."""
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.camera_index)  # type: ignore[attr-defined]
        if not cap.isOpened():
            msg = f"could not open camera index {self.camera_index}"
            raise RuntimeError(msg)
        try:
            face_mesh_cls = self._mp_face
            face_mesh = face_mesh_cls(refine_landmarks=True, max_num_faces=1)  # type: ignore[operator]
            read_count = 0
            yielded = 0
            start = time.monotonic()
            while max_frames is None or read_count < max_frames:
                ok, frame = cap.read()
                frame_index = read_count
                read_count += 1
                if not ok:
                    break
                t = time.monotonic() - start
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
                result = face_mesh.process(rgb)  # type: ignore[attr-defined]
                if not result.multi_face_landmarks:
                    continue
                lm = result.multi_face_landmarks[0].landmark
                arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
                yielded += 1
                fps = yielded / t if t > 0.0 else 0.0
                yield iris_landmarks_to_capture_sample(
                    arr,
                    t=t,
                    frame_index=frame_index,
                    fps_estimate_hz=fps,
                    max_angle_deg=self.max_angle_deg,
                )
        finally:
            cap.release()
            if "face_mesh" in locals():
                face_mesh.close()

    def gaze_frames(
        self, max_frames: int | None = None
    ) -> Iterator[GazeSample]:  # pragma: no cover
        """Compatibility iterator yielding only gaze samples."""
        for sample in self.frames(max_frames=max_frames):
            yield sample.gaze

    def live_frames(
        self,
        max_frames: int | None = None,
        *,
        crop_padding_fraction: float = 0.70,
        jpeg_quality: int = 85,
    ) -> Iterator[LiveFrameSample]:  # pragma: no cover
        """Yield capture samples with live eye crops for the HTML orchestrator."""
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.camera_index)  # type: ignore[attr-defined]
        if not cap.isOpened():
            msg = f"could not open camera index {self.camera_index}"
            raise RuntimeError(msg)
        try:
            face_mesh_cls = self._mp_face
            face_mesh = face_mesh_cls(refine_landmarks=True, max_num_faces=1)  # type: ignore[operator]
            read_count = 0
            yielded = 0
            start = time.monotonic()
            while max_frames is None or read_count < max_frames:
                ok, frame = cap.read()
                frame_index = read_count
                read_count += 1
                if not ok:
                    break
                t = time.monotonic() - start
                frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
                fps_read = read_count / t if t > 0.0 else 0.0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
                result = face_mesh.process(rgb)  # type: ignore[attr-defined]
                if not result.multi_face_landmarks:
                    eye_box = _default_eye_box(frame_width, frame_height)
                    yield LiveFrameSample(
                        capture=CaptureSample(
                            frame_index=frame_index,
                            timestamp_s=t,
                            gaze=GazeSample(t=t, x=float("nan"), y=float("nan")),
                            pupil=PupilSample(
                                t=t,
                                size=float("nan"),
                                unit=PupilUnit.RELATIVE,
                            ),
                            fps_estimate_hz=fps_read,
                            quality={"face_detected": 0.0},
                        ),
                        frame_width=frame_width,
                        frame_height=frame_height,
                        eye_box=eye_box,
                        eye_crop_jpeg=encode_eye_crop_jpeg_base64(
                            frame,
                            eye_box,
                            cv2_module=cv2,
                            jpeg_quality=jpeg_quality,
                        ),
                    )
                    continue
                lm = result.multi_face_landmarks[0].landmark
                arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
                yielded += 1
                fps = yielded / t if t > 0.0 else 0.0
                capture = iris_landmarks_to_capture_sample(
                    arr,
                    t=t,
                    frame_index=frame_index,
                    fps_estimate_hz=fps,
                    max_angle_deg=self.max_angle_deg,
                )
                eye_box = eye_box_from_landmarks(
                    arr,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    padding_fraction=crop_padding_fraction,
                )
                yield LiveFrameSample(
                    capture=capture,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    eye_box=eye_box,
                    eye_crop_jpeg=encode_eye_crop_jpeg_base64(
                        frame,
                        eye_box,
                        cv2_module=cv2,
                        jpeg_quality=jpeg_quality,
                    ),
                )
        finally:
            cap.release()
            if "face_mesh" in locals():
                face_mesh.close()


def _require_cv2() -> Any:
    """Import cv2 lazily for JPEG encoding helpers."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        msg = (
            "JPEG eye-crop encoding needs the 'capture' extra. Install it with:\n"
            "    uv sync --extra capture\n"
            f"Missing module: {exc.name}"
        )
        raise RuntimeError(msg) from exc
    return cv2


def _require_capture_deps() -> tuple[Any, Any]:
    """Import cv2 + mediapipe lazily, raising an actionable error if absent."""
    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError as exc:
        msg = (
            "Webcam capture needs the 'capture' extra. Install it with:\n"
            "    uv sync --extra capture\n"
            "(provides opencv-python and mediapipe). "
            f"Missing module: {exc.name}"
        )
        raise RuntimeError(msg) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        msg = (
            "Webcam capture needs MediaPipe's legacy Face Mesh solutions API. "
            "Use the project lockfile or install the 'capture' extra, which pins "
            "a compatible mediapipe range."
        )
        raise RuntimeError(msg)
    return cv2, mp
