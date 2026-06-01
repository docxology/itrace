"""Lightweight report payload validation.

The package intentionally avoids a runtime schema dependency. This module checks
the JSON-friendly dictionary emitted by :meth:`itrace.types.SessionReport.to_dict`
with small, explicit type predicates.
"""

from __future__ import annotations

from collections.abc import Mapping


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_report_payload(
    payload: Mapping[str, object],
    *,
    raise_on_error: bool = False,
) -> dict[str, object]:
    """Validate the stable top-level shape of a report payload."""
    errors: list[str] = []
    required: dict[str, type[object] | tuple[type[object], ...]] = {
        "n_samples": int,
        "duration_s": (int, float),
        "n_fixations": int,
        "n_saccades": int,
        "scanpath": str,
        "saccades": list,
        "fixations": list,
        "quality": dict,
        "config": dict,
    }
    for key, expected in required.items():
        if key not in payload:
            errors.append(f"missing required field {key!r}")
            continue
        value = payload[key]
        if expected == (int, float):
            if not _is_number(value):
                errors.append(f"{key} must be numeric")
        elif not isinstance(value, expected):
            errors.append(f"{key} must be {expected}")

    if "n_samples" in payload and not isinstance(payload["n_samples"], int):
        errors.append("n_samples must be an integer")
    if (
        "n_saccades" in payload
        and isinstance(payload.get("saccades"), list)
        and isinstance(payload["n_saccades"], int)
        and payload["n_saccades"] != len(payload["saccades"])  # type: ignore[arg-type]
    ):
        errors.append("n_saccades does not match saccades length")

    if errors and raise_on_error:
        msg = "report payload validation failed: " + "; ".join(errors)
        raise ValueError(msg)
    return {
        "valid": not errors,
        "errors": errors,
        "fields": sorted(str(key) for key in payload),
    }
