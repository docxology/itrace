"""3D eyeball forward model and pinhole projection to webcam landmarks.

This is the *forward model* of the closed loop: a parameterised 3D eyeball is
rotated to a true gaze direction and dilated to a true pupil size, then
**perspective-projected** through a pinhole camera to the 2-D, MediaPipe-shaped
normalised landmark array that the capture layer consumes. Feeding those
landmarks back through :func:`itrace.capture.iris_landmarks_to_sample` recovers
the gaze — closing the loop from 3-D truth to estimated signal.

Crucially the forward model (3-D sphere + perspective projection) and the
estimator (an arcsine sphere approximation in :mod:`itrace.geometry`) are
**independent formulations**. Their agreement is therefore a genuine validation
of the geometry/landmark path, not a tautology, and the small residual error is
the cost of the estimator's approximation.

Conventions (consistent with :mod:`itrace.types`)
--------------------------------------------------
* World axes: ``X`` right, ``Y`` up, ``Z`` toward the scene. The pinhole camera
  sits at the origin looking along ``+Z``; the eye centre is at ``Z = distance``.
* Gaze: ``yaw`` rotates about ``Y`` (``+`` looks right), ``pitch`` about ``X``
  (``+`` looks up). ``yaw = pitch = 0`` looks straight back at the camera.
* Image: normalised, ``x`` right and ``y`` *down* (MediaPipe convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import LEFT_EYE_CORNERS, LEFT_IRIS, RIGHT_EYE_CORNERS, RIGHT_IRIS
from .types import FloatArray

# Number of MediaPipe Face Mesh landmarks (refine_landmarks=True).
N_LANDMARKS = 478


@dataclass(frozen=True, slots=True)
class Camera:
    """Pinhole camera + scene scale for projecting the eye to normalised pixels."""

    focal: float = 1.0
    distance_mm: float = 600.0
    interocular_mm: float = 64.0
    norm_scale: float = 6.0  # world-to-normalised-image gain (keeps coords ~[0,1])


@dataclass(frozen=True, slots=True)
class Eye3D:
    """A 3-D eyeball at a given gaze and dilation."""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    pupil_radius_mm: float = 2.0
    iris_radius_mm: float = 6.0
    eyeball_radius_mm: float = 12.0

    def gaze_vector(self) -> FloatArray:
        """Unit gaze vector in world axes (``yaw=pitch=0`` -> ``(0,0,-1)``)."""
        y, p = np.radians(self.yaw_deg), np.radians(self.pitch_deg)
        v = np.array(
            [np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)],
            dtype=np.float64,
        )
        unit: FloatArray = v / np.linalg.norm(v)
        return unit


def project_pinhole(point_mm: FloatArray, camera: Camera) -> FloatArray:
    """Project a 3-D world point (mm) to normalised image coordinates.

    A point on the optical axis (``X = Y = 0``) maps to the principal point
    ``(0.5, 0.5)``. Image ``y`` grows downward, so world-up maps to smaller
    ``y``.
    """
    p = np.asarray(point_mm, dtype=np.float64)
    z = p[2]
    if z <= 0:
        msg = "point must be in front of the camera (Z > 0)"
        raise ValueError(msg)
    nx = 0.5 + camera.norm_scale * camera.focal * p[0] / z
    ny = 0.5 - camera.norm_scale * camera.focal * p[1] / z
    return np.array([nx, ny], dtype=np.float64)


def _orthonormal_basis(normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Two unit vectors spanning the plane perpendicular to ``normal``."""
    helper = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _disc_points(center: FloatArray, normal: FloatArray, radius: float, n: int) -> FloatArray:
    """``n`` points on a circle of ``radius`` centred at ``center`` in the plane
    perpendicular to ``normal``."""
    u, v = _orthonormal_basis(normal)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts: FloatArray = center + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    return pts


def _eye_center(side: str, camera: Camera) -> FloatArray:
    sign = +1.0 if side == "right" else -1.0
    return np.array([sign * camera.interocular_mm / 2.0, 0.0, camera.distance_mm], dtype=np.float64)


def _fill_eye(
    landmarks: FloatArray,
    eye: Eye3D,
    camera: Camera,
    center: FloatArray,
    iris_idx: tuple[int, ...],
    corner_idx: tuple[int, int],
) -> None:
    g = eye.gaze_vector()
    iris_center = center + eye.eyeball_radius_mm * g
    # iris ring lies in the plane perpendicular to the gaze axis
    ring = _disc_points(iris_center, g, eye.iris_radius_mm, len(iris_idx))
    for k, idx in enumerate(iris_idx):
        nx, ny = project_pinhole(ring[k], camera)
        landmarks[idx] = (nx, ny, 0.0)
    # eye corners are anatomically fixed (do NOT rotate with the eyeball);
    # half-aperture chosen so the corner span subtends ~25deg, matching the
    # estimator's default max_angle so recovery is well-scaled.
    corner_half = eye.eyeball_radius_mm * np.sin(np.radians(25.0))
    for idx, dx in zip(corner_idx, (-corner_half, +corner_half), strict=True):
        corner = center + np.array([dx, 0.0, 0.0])
        nx, ny = project_pinhole(corner, camera)
        landmarks[idx] = (nx, ny, 0.0)


def eye_to_landmarks(
    yaw_deg: float,
    pitch_deg: float,
    *,
    pupil_radius_mm: float = 2.0,
    camera: Camera | None = None,
) -> FloatArray:
    """Render a binocular gaze to a MediaPipe-shaped ``(478, 3)`` landmark array.

    Both eyes share the gaze (binocular fixation). The returned array is exactly
    what :func:`itrace.capture.iris_landmarks_to_sample` expects.
    """
    cam = camera or Camera()
    eye = Eye3D(yaw_deg=yaw_deg, pitch_deg=pitch_deg, pupil_radius_mm=pupil_radius_mm)
    landmarks = np.full((N_LANDMARKS, 3), 0.5, dtype=np.float64)
    _fill_eye(landmarks, eye, cam, _eye_center("right", cam), RIGHT_IRIS, RIGHT_EYE_CORNERS)
    _fill_eye(landmarks, eye, cam, _eye_center("left", cam), LEFT_IRIS, LEFT_EYE_CORNERS)
    return landmarks


def projected_pupil_ratio(
    yaw_deg: float,
    pitch_deg: float,
    pupil_radius_mm: float,
    *,
    iris_radius_mm: float = 6.0,
    camera: Camera | None = None,
) -> float:
    """Projected pupil-radius / iris-radius ratio (the image pupillometry signal).

    Both discs sit at the same depth, so under perspective their projected radii
    scale identically with distance; the ratio is therefore a clean,
    head-distance-invariant proxy for pupil dilation.
    """
    cam = camera or Camera()
    eye = Eye3D(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        pupil_radius_mm=pupil_radius_mm,
        iris_radius_mm=iris_radius_mm,
    )
    center = _eye_center("right", cam)
    g = eye.gaze_vector()
    iris_center = center + eye.eyeball_radius_mm * g

    def _img_radius(radius_mm: float) -> float:
        ring = _disc_points(iris_center, g, radius_mm, 8)
        pts = np.array([project_pinhole(p, cam) for p in ring])
        c = np.array(project_pinhole(iris_center, cam))
        return float(np.mean(np.linalg.norm(pts - c, axis=1)))

    iris_r = _img_radius(eye.iris_radius_mm)
    pupil_r = _img_radius(eye.pupil_radius_mm)
    return pupil_r / iris_r if iris_r > 0 else 0.0


# A blink frame: landmarks present but iris invalid (NaN) so downstream code
# treats it as missing rather than a bogus gaze.
def blink_landmarks() -> FloatArray:
    """A landmark frame for a closed eye: iris indices are NaN (invalid)."""
    landmarks = np.full((N_LANDMARKS, 3), 0.5, dtype=np.float64)
    for idx in (*RIGHT_IRIS, *LEFT_IRIS):
        landmarks[idx] = (np.nan, np.nan, np.nan)
    return landmarks
