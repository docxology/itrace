"""No-mocks tests for pure eye-crop pupil segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import pupilseg
from itrace.types import PupilUnit


def _synthetic_eye_crop(radius: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[:64, :96]
    image = np.full((64, 96), 180.0, dtype=np.float64)
    mask = (xx - 42.0) ** 2 + (yy - 31.0) ** 2 <= radius**2
    image[mask] = 25.0
    image += 4.0 * np.sin(xx / 7.0)
    return image


def test_segment_dark_pupil_recovers_synthetic_center_and_radius() -> None:
    segmentation = pupilseg.segment_dark_pupil(_synthetic_eye_crop(), threshold=60.0)

    assert segmentation.center_x_px == pytest.approx(42.0, abs=0.5)
    assert segmentation.center_y_px == pytest.approx(31.0, abs=0.5)
    assert segmentation.radius_px == pytest.approx(8.0, rel=0.08)
    assert segmentation.diameter_px == pytest.approx(segmentation.radius_px * 2.0)
    assert segmentation.confidence > 0.5
    assert segmentation.quality["component_count"] == 1.0


def test_segment_pupil_sample_reports_pixels_or_relative_units() -> None:
    segmentation, pixel_sample = pupilseg.segment_pupil_sample(
        _synthetic_eye_crop(radius=6.0),
        t=0.25,
        threshold=60.0,
    )
    relative = pupilseg.segmentation_to_pupil_sample(segmentation, t=0.25, iris_radius_px=12.0)

    assert pixel_sample.t == 0.25
    assert pixel_sample.unit is PupilUnit.PIXELS
    assert pixel_sample.size == pytest.approx(segmentation.diameter_px)
    assert relative.unit is PupilUnit.RELATIVE
    assert relative.size == pytest.approx(segmentation.radius_px / 12.0)


def test_segment_dark_pupil_rejects_missing_or_tiny_candidates() -> None:
    with pytest.raises(ValueError, match="no finite"):
        pupilseg.segment_dark_pupil(np.full((4, 4), np.nan))
    tiny = np.ones((8, 8), dtype=np.float64)
    tiny[3, 4] = 0.0
    with pytest.raises(ValueError, match="smaller"):
        pupilseg.segment_dark_pupil(tiny, threshold=0.0, min_area_px=2)
    segmentation = pupilseg.segment_dark_pupil(np.dstack([_synthetic_eye_crop()] * 3), threshold=60)
    assert segmentation.area_px > 0
    with pytest.raises(ValueError, match="iris_radius_px"):
        pupilseg.segmentation_to_pupil_sample(segmentation, t=0.0, iris_radius_px=0.0)
