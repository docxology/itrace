from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .calibration import Calibration, Dwell
from .jsonio import JsonValue, StudyError, encoded, mapping, number, text
from .model import Limits
from .setup import parse_setup


def replay_calibration(
    setup: Mapping[str, JsonValue] | None, records: Sequence[JsonValue], limits: Limits
) -> dict[str, JsonValue]:
    if setup is None:
        return {"available": False, "post_validation_adequate": None}
    calibration = Calibration(parse_setup(setup), limits)
    try:
        for value in records:
            payload = mapping(value, "dwell")
            target = calibration.target()
            if target is None or text(payload.get("target_id"), "target_id") != target["target_id"]:
                raise StudyError("calibration", "saved target order mismatch")
            gx = payload.get("gaze_x_deg")
            gy = payload.get("gaze_y_deg")
            missing = gx is None or gy is None
            if missing and calibration.index < 14:
                raise StudyError("calibration", "initial fit/validation has missing mean")
            calibration.accept(
                Dwell(
                    math.nan if missing else number(gx, "gaze_x_deg", -1e6, 1e6),
                    math.nan if missing else number(gy, "gaze_y_deg", -1e6, 1e6),
                    number(
                        payload.get("target_x_css_px"),
                        "target_x_css_px",
                        0,
                        calibration.setup.width,
                    ),
                    number(
                        payload.get("target_y_css_px"),
                        "target_y_css_px",
                        0,
                        calibration.setup.height,
                    ),
                    encoded(payload),
                )
            )
    except StudyError as exc:
        return {
            "available": True,
            **calibration.diagnostics,
            "initial_validation_adequate": False,
            "post_validation_adequate": None,
            "reconstruction_error": {"field": exc.field, "reason": exc.reason},
        }
    return {
        "available": True,
        **calibration.diagnostics,
        "initial_validation_adequate": calibration.index >= 14,
        "post_validation_adequate": calibration.diagnostics.get("post_validation_adequate"),
        "reconstruction": (
            "affine fit/residuals recomputed from saved dwell means; "
            "no reference-device accuracy claim"
        ),
    }
