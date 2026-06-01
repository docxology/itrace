"""Gaze geometry: pixels <-> degrees of visual angle, iris-offset -> gaze angle.

The relationships here are the deterministic, head-distance-aware core that
turns raw landmark pixels into the calibrated degree-of-visual-angle quantities
the rest of the pipeline assumes. Everything is pure NumPy and exactly
invertible where a round-trip is claimed.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .types import FloatArray


def pix2deg(
    pixels: FloatArray | float,
    screen_px: float,
    screen_cm: float,
    viewing_distance_cm: float,
) -> FloatArray:
    """Convert a pixel *offset from screen centre* to degrees of visual angle.

    Uses the standard small-and-large-angle-exact formula
    ``theta = atan(size_cm / distance_cm)`` (pymovements ``pix2deg`` convention).

    Parameters
    ----------
    pixels:
        Offset(s) from the centre of the screen, in pixels.
    screen_px:
        Screen resolution along this axis, in pixels.
    screen_cm:
        Physical screen size along this axis, in centimetres.
    viewing_distance_cm:
        Eye-to-screen distance, in centimetres.
    """
    if screen_px <= 0 or screen_cm <= 0 or viewing_distance_cm <= 0:
        msg = "screen_px, screen_cm and viewing_distance_cm must be positive"
        raise ValueError(msg)
    cm_per_px = screen_cm / screen_px
    offsets_cm = np.asarray(pixels, dtype=np.float64) * cm_per_px
    return cast(FloatArray, np.degrees(np.arctan2(offsets_cm, viewing_distance_cm)))


def deg2pix(
    degrees: FloatArray | float,
    screen_px: float,
    screen_cm: float,
    viewing_distance_cm: float,
) -> FloatArray:
    """Inverse of :func:`pix2deg` (exact round-trip)."""
    if screen_px <= 0 or screen_cm <= 0 or viewing_distance_cm <= 0:
        msg = "screen_px, screen_cm and viewing_distance_cm must be positive"
        raise ValueError(msg)
    cm_per_px = screen_cm / screen_px
    offsets_cm = np.tan(np.radians(np.asarray(degrees, dtype=np.float64))) * viewing_distance_cm
    return cast(FloatArray, offsets_cm / cm_per_px)


def iris_offset_to_gaze_angle(
    iris_offset: FloatArray | float,
    max_angle_deg: float = 25.0,
) -> FloatArray:
    """Map a normalised iris offset in ``[-1, 1]`` to a gaze angle in degrees.

    The eye is modelled as a sphere; the iris centre's displacement within the
    palpebral fissure maps approximately sinusoidally to eyeball rotation. A
    normalised offset of ``+-1`` corresponds to ``+-max_angle_deg``. The mapping
    is monotonic and odd.

    Raises
    ------
    ValueError
        If any input is non-finite (NaN/inf are never silently coerced).
    """
    arr = np.asarray(iris_offset, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        msg = "iris_offset contains non-finite values (NaN/inf)"
        raise ValueError(msg)
    clipped = np.clip(arr, -1.0, 1.0)
    angles: FloatArray = np.degrees(np.arcsin(clipped * np.sin(np.radians(max_angle_deg))))
    return angles


def normalize_by_interocular(
    pixel_offset: FloatArray | float,
    interocular_px: float,
    reference_interocular_px: float = 64.0,
) -> FloatArray:
    """Scale a pixel offset to a head-distance-invariant reference.

    The inter-ocular distance in pixels shrinks as the head moves away from the
    camera. Dividing by it (and multiplying by a fixed reference) removes that
    distance dependence, so the same physical eye movement yields the same
    normalised value regardless of how far the user sits. The output is
    therefore invariant to any uniform scaling applied to *both* the offset and
    the inter-ocular distance.
    """
    if interocular_px <= 0 or reference_interocular_px <= 0:
        msg = "interocular distances must be positive"
        raise ValueError(msg)
    arr = np.asarray(pixel_offset, dtype=np.float64)
    out: FloatArray = arr / interocular_px * reference_interocular_px
    return out


def direction_deg(dx: float, dy_screen: float) -> float:
    """Direction of a displacement in the *gaze* convention (up = +90deg).

    ``dy_screen`` is in screen coordinates (down is positive); it is negated so
    that an upward movement yields ``+90deg``. Result is wrapped to
    ``(-180, 180]``.
    """
    return float(np.degrees(np.arctan2(-dy_screen, dx)))
