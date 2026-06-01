"""Gaze-geometry tests (ISC-11..14)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import geometry


def test_pix2deg_deg2pix_roundtrip() -> None:  # ISC-11
    px = np.array([-200.0, -50.0, 0.0, 75.0, 300.0])
    deg = geometry.pix2deg(px, screen_px=1920, screen_cm=52.0, viewing_distance_cm=60.0)
    back = geometry.deg2pix(deg, screen_px=1920, screen_cm=52.0, viewing_distance_cm=60.0)
    assert np.allclose(back, px, atol=1e-9)


def test_pix2deg_rejects_nonpositive_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        geometry.pix2deg(10.0, screen_px=0, screen_cm=52.0, viewing_distance_cm=60.0)
    with pytest.raises(ValueError, match="positive"):
        geometry.deg2pix(10.0, screen_px=1920, screen_cm=-1, viewing_distance_cm=60.0)


def test_iris_offset_monotonic() -> None:  # ISC-12
    offsets = np.linspace(-1.0, 1.0, 21)
    angles = geometry.iris_offset_to_gaze_angle(offsets)
    assert np.all(np.diff(angles) > 0)
    assert angles[10] == pytest.approx(0.0, abs=1e-9)
    assert angles[0] == pytest.approx(-angles[-1], abs=1e-9)


def test_iris_offset_rejects_nonfinite() -> None:  # ISC-14
    with pytest.raises(ValueError, match="non-finite"):
        geometry.iris_offset_to_gaze_angle(np.array([0.1, np.nan]))


def test_normalize_by_interocular_scale_invariant() -> None:  # ISC-13
    offset, iod = 30.0, 64.0
    base = geometry.normalize_by_interocular(offset, iod)
    for k in (0.5, 2.0, 3.7):  # uniform scale of both offset and IOD
        scaled = geometry.normalize_by_interocular(offset * k, iod * k)
        assert scaled == pytest.approx(base, rel=1e-6)


def test_normalize_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        geometry.normalize_by_interocular(10.0, 0.0)


def test_direction_convention() -> None:  # supports ISC-25
    assert geometry.direction_deg(1.0, 0.0) == pytest.approx(0.0)  # right
    assert geometry.direction_deg(0.0, -1.0) == pytest.approx(90.0)  # up (screen-y up)
    assert abs(geometry.direction_deg(-1.0, 0.0)) == pytest.approx(180.0)  # left (+/-180)
    assert geometry.direction_deg(0.0, 1.0) == pytest.approx(-90.0)  # down
