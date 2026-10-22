from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .jsonio import JsonValue


def compare_validation(
    coefficients: Sequence[Sequence[float]],
    pre_residuals: Sequence[tuple[float, float]],
    post_points: Sequence[tuple[float, float, float, float]],
    diagonal: float,
    threshold: float,
) -> dict[str, JsonValue]:
    points: list[JsonValue] = []
    errors: list[float] = []
    for index, (gx, gy, tx, ty) in enumerate(post_points):
        pre_x, pre_y = pre_residuals[index]
        pre_error = math.hypot(pre_x, pre_y)
        dx: float | None = None
        dy: float | None = None
        error: float | None = None
        if math.isfinite(gx) and math.isfinite(gy):
            dx = gx * coefficients[0][0] + gy * coefficients[1][0] + coefficients[2][0] - tx
            dy = gx * coefficients[0][1] + gy * coefficients[1][1] + coefficients[2][1] - ty
            error = math.hypot(dx, dy)
            errors.append(error)
        points.append(
            {
                "validation_target_id": f"target-{index + 10}",
                "post_target_id": f"target-{index + 15}",
                "target_x_css_px": tx,
                "target_y_css_px": ty,
                "pre_error_css_px": pre_error,
                "post_error_css_px": error,
                "pre_dx_css_px": pre_x,
                "pre_dy_css_px": pre_y,
                "post_dx_css_px": dx,
                "post_dy_css_px": dy,
                "error_change_css_px": error - pre_error if error is not None else None,
                "residual_change_x_css_px": dx - pre_x if dx is not None else None,
                "residual_change_y_css_px": dy - pre_y if dy is not None else None,
            }
        )
    p95 = float(np.percentile(errors, 95)) if errors else None
    rms = math.sqrt(math.fsum(value * value for value in errors) / len(errors)) if errors else None
    return {
        "post_validation_target_count": len(post_points),
        "post_validation_valid_target_count": len(errors),
        "pre_post_residuals": points,
        "post_validation_p95_css_px": p95,
        "post_validation_rms_css_px": rms,
        "post_validation_adequate": len(errors) == 5
        and p95 is not None
        and p95 / diagonal <= threshold,
        "post_validation_fit": "frozen initial nine-target fit; no refitting",
        "post_validation_scope": "engineering gaze adequacy, separate from behavioral outcomes",
    }
