"""Pure NumPy/SciPy pupil segmentation for cropped eye images.

The live webcam path currently exports a relative iris-landmark proxy. This
module adds a small, explicit image-space segmentation path for callers that
already have an eye crop. It reports pixel or relative units only; it does not
claim millimetre pupil diameter or reference-device validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np
from scipy import ndimage

from .types import FloatArray, PupilSample, PupilUnit


@dataclass(frozen=True, slots=True)
class PupilSegmentation:
    """Dark-pupil segmentation result in image-pixel coordinates."""

    center_x_px: float
    center_y_px: float
    radius_px: float
    area_px: float
    threshold: float
    confidence: float
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def diameter_px(self) -> float:
        """Equivalent circular pupil diameter in pixels."""
        return 2.0 * self.radius_px


def _grayscale(image: FloatArray | list[list[float]]) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[2] < 3:
            msg = "RGB eye crops must have at least three channels"
            raise ValueError(msg)
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        msg = "eye crop must be a 2-D grayscale or 3-D RGB image"
        raise ValueError(msg)
    if not np.any(np.isfinite(arr)):
        msg = "eye crop contains no finite pixels"
        raise ValueError(msg)
    return arr


def segment_dark_pupil(
    image: FloatArray | list[list[float]],
    *,
    threshold: float | None = None,
    min_area_px: int = 8,
) -> PupilSegmentation:
    """Segment the largest dark connected component in an eye crop.

    ``threshold`` defaults to the 20th percentile of finite intensities, which
    keeps the method deterministic and camera-agnostic for synthetic fixtures.
    The returned diameter is an image-space measurement unless the caller
    separately supplies an iris-radius normalizer.
    """
    if min_area_px < 1:
        msg = "min_area_px must be positive"
        raise ValueError(msg)
    gray = _grayscale(image)
    finite = gray[np.isfinite(gray)]
    cutoff = float(np.percentile(finite, 20.0)) if threshold is None else float(threshold)
    if not np.isfinite(cutoff):
        msg = "threshold must be finite"
        raise ValueError(msg)

    dark = np.isfinite(gray) & (gray <= cutoff)
    labels, n_labels = ndimage.label(dark)
    if n_labels == 0:
        msg = "no dark pupil candidate found"
        raise ValueError(msg)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    label = int(np.argmax(counts))
    area = int(counts[label])
    if area < min_area_px:
        msg = "dark pupil candidate is smaller than min_area_px"
        raise ValueError(msg)

    ys, xs = np.nonzero(labels == label)
    pupil_values = gray[ys, xs]
    background_values = gray[np.isfinite(gray) & (labels != label)]
    background_mean = float(np.mean(background_values)) if background_values.size else float(cutoff)
    pupil_mean = float(np.mean(pupil_values))
    contrast = max(0.0, background_mean - pupil_mean)
    denom = max(abs(background_mean), 1.0)
    contrast_score = min(1.0, contrast / denom)
    area_fraction = area / float(gray.size)
    confidence = float(
        max(0.0, min(1.0, 0.65 * contrast_score + 0.35 * min(area_fraction * 20.0, 1.0)))
    )

    return PupilSegmentation(
        center_x_px=float(np.mean(xs)),
        center_y_px=float(np.mean(ys)),
        radius_px=float(sqrt(area / pi)),
        area_px=float(area),
        threshold=cutoff,
        confidence=confidence,
        quality={
            "area_fraction": float(area_fraction),
            "contrast": float(contrast),
            "component_count": float(n_labels),
            "pupil_mean_intensity": pupil_mean,
            "background_mean_intensity": background_mean,
        },
    )


def segmentation_to_pupil_sample(
    segmentation: PupilSegmentation,
    *,
    t: float,
    iris_radius_px: float | None = None,
) -> PupilSample:
    """Convert a segmentation to a timestamped pupil sample.

    If ``iris_radius_px`` is supplied the sample is a relative pupil/iris-radius
    ratio; otherwise it is an equivalent pupil diameter in pixels.
    """
    if iris_radius_px is not None:
        if iris_radius_px <= 0.0 or not np.isfinite(iris_radius_px):
            msg = "iris_radius_px must be positive and finite"
            raise ValueError(msg)
        return PupilSample(
            t=t,
            size=segmentation.radius_px / float(iris_radius_px),
            unit=PupilUnit.RELATIVE,
        )
    return PupilSample(t=t, size=segmentation.diameter_px, unit=PupilUnit.PIXELS)


def segment_pupil_sample(
    image: FloatArray | list[list[float]],
    *,
    t: float,
    threshold: float | None = None,
    min_area_px: int = 8,
    iris_radius_px: float | None = None,
) -> tuple[PupilSegmentation, PupilSample]:
    """Segment an eye crop and return both segmentation metadata and sample."""
    segmentation = segment_dark_pupil(image, threshold=threshold, min_area_px=min_area_px)
    sample = segmentation_to_pupil_sample(segmentation, t=t, iris_radius_px=iris_radius_px)
    return segmentation, sample
