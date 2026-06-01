"""3-D eye forward-model tests (ISC-54..61)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import capture, eyemodel
from itrace.eyemodel import Camera, Eye3D


def test_gaze_vector_unit_and_origin() -> None:  # ISC-54/ISC-55
    g0 = Eye3D().gaze_vector()
    assert np.allclose(np.linalg.norm(g0), 1.0)
    assert np.allclose(g0, [0.0, 0.0, -1.0], atol=1e-9)  # looks at camera


def test_gaze_vector_monotonic() -> None:  # ISC-55
    gx = [Eye3D(yaw_deg=y).gaze_vector()[0] for y in (-10, 0, 10, 20)]
    assert gx[0] < gx[1] < gx[2] < gx[3]
    gy = [Eye3D(pitch_deg=p).gaze_vector()[1] for p in (-10, 0, 10, 20)]
    assert gy[0] < gy[1] < gy[2] < gy[3]


def test_project_pinhole_principal_point() -> None:  # ISC-56
    cam = Camera()
    nx, ny = eyemodel.project_pinhole(np.array([0.0, 0.0, cam.distance_mm]), cam)
    assert nx == pytest.approx(0.5)
    assert ny == pytest.approx(0.5)


def test_project_pinhole_rejects_behind_camera() -> None:
    with pytest.raises(ValueError, match="Z > 0"):
        eyemodel.project_pinhole(np.array([0.0, 0.0, -1.0]), Camera())


def test_project_up_is_smaller_y() -> None:  # ISC-56 (image y grows downward)
    cam = Camera()
    up = eyemodel.project_pinhole(np.array([0.0, 5.0, cam.distance_mm]), cam)
    assert up[1] < 0.5


def test_eye_to_landmarks_shape_and_indices() -> None:  # ISC-57
    lm = eyemodel.eye_to_landmarks(0.0, 0.0)
    assert lm.shape == (eyemodel.N_LANDMARKS, 3)
    for idx in (*capture.RIGHT_IRIS, *capture.LEFT_IRIS, *capture.RIGHT_EYE_CORNERS):
        assert np.all(np.isfinite(lm[idx]))


def test_centred_gaze_recovers_zero() -> None:  # ISC-58
    lm = eyemodel.eye_to_landmarks(0.0, 0.0)
    s = capture.iris_landmarks_to_sample(lm, t=0.0)
    assert s.x == pytest.approx(0.0, abs=1.0)
    assert s.y == pytest.approx(0.0, abs=1.0)


def test_rightward_gaze_positive_increasing() -> None:  # ISC-59
    recovered = []
    for yaw in (5.0, 10.0, 15.0):
        lm = eyemodel.eye_to_landmarks(yaw, 0.0)
        recovered.append(capture.iris_landmarks_to_sample(lm, 0.0).x)
    assert recovered[0] > 0
    assert recovered[0] < recovered[1] < recovered[2]


def test_upward_gaze_positive_pitch() -> None:  # ISC-59 (vertical)
    lm = eyemodel.eye_to_landmarks(0.0, 10.0)
    s = capture.iris_landmarks_to_sample(lm, 0.0)
    assert s.y > 1.0  # recovered pitch is clearly positive (up)


def test_pupil_ratio_monotonic() -> None:  # ISC-60
    from itertools import pairwise

    ratios = [eyemodel.projected_pupil_ratio(0.0, 0.0, p) for p in (1.0, 2.0, 3.0, 4.0)]
    assert all(b > a for a, b in pairwise(ratios))


def test_blink_landmarks_invalid_iris() -> None:  # ISC-61
    lm = eyemodel.blink_landmarks()
    assert np.all(np.isnan(lm[capture.RIGHT_IRIS[0]]))
    with pytest.raises(ValueError, match="non-finite"):
        capture.iris_landmarks_to_sample(lm, 0.0)
