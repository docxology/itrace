import math
from collections.abc import Sequence
from itertools import pairwise

from ..capture import CaptureSample
from .calibration import Frame
from .jsonio import JsonValue, StudyError
from .model import Limits


def capture_quality(
    frames: Sequence[Frame], now: float, limits: Limits, *, active: bool
) -> dict[str, JsonValue]:
    recent = [frame for frame in frames if now - limits.collect_s <= frame.timestamp_s <= now]
    count = len(recent)
    valid = sum(frame.valid for frame in recent)
    fraction = valid / count if count else 0.0
    return {
        "window_s": limits.collect_s,
        "recent_samples": count,
        "valid_samples": valid,
        "valid_fraction": fraction,
        "last_sample_age_s": now - frames[-1].timestamp_s if frames else None,
        "ready_to_calibrate": (
            active
            and valid >= limits.min_samples
            and fraction >= limits.min_valid_fraction
            and bool(recent)
            and recent[-1].valid
        ),
        "scope": "live sample availability; held-out calibration remains required",
    }


def probe_quality(samples: Sequence[CaptureSample]) -> dict[str, JsonValue]:
    if not samples:
        raise StudyError("camera", "no frames acquired")
    times = [sample.timestamp_s for sample in samples]
    if not all(math.isfinite(value) for value in times) or any(
        right <= left for left, right in pairwise(times)
    ):
        raise StudyError("camera", "nonfinite or nonincreasing native timestamps")
    faces = sum(sample.quality.get("face_detected", 0) > 0 for sample in samples)
    finite = sum(
        sample.quality.get("face_detected", 0) > 0
        and math.isfinite(sample.gaze.x)
        and math.isfinite(sample.gaze.y)
        for sample in samples
    )
    if finite == 0:
        raise StudyError(
            "camera",
            f"{len(samples)} frames acquired; no face-valid finite gaze; "
            "check framing and illumination",
        )
    return {
        "read_frames": len(samples),
        "detected_face_samples": faces,
        "finite_gaze_samples": finite,
        "duration_s": times[-1] - times[0],
        "scope": "camera access and finite samples only; not gaze accuracy or calibration",
    }
