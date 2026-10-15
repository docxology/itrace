"""Lightweight report payload validation.

The package intentionally avoids a runtime schema dependency. This module checks
the JSON-friendly dictionary emitted by :meth:`itrace.types.SessionReport.to_dict`
with small, explicit type predicates.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from .types import NumericReportDict, ReportPayload, SessionReport


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value)


def validate_report_payload(
    payload: Mapping[str, object],
    *,
    raise_on_error: bool = False,
) -> ReportPayload:
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

    for key in (
        "n_samples",
        "n_fixations",
        "n_saccades",
        "n_microsaccades",
        "n_smooth_pursuits",
        "n_psos",
    ):
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{key} must be a non-negative integer")
    duration = payload.get("duration_s")
    if isinstance(duration, int | float) and (_is_number(duration) is False or duration < 0.0):
        errors.append("duration_s must be finite and non-negative")
    for key in ("fixations", "saccades", "smooth_pursuits", "psos"):
        events = payload.get(key)
        if isinstance(events, list) and payload.get(f"n_{key}") != len(events):
            errors.append(f"n_{key} does not match {key} length")

    if errors and raise_on_error:
        msg = "report payload validation failed: " + "; ".join(errors)
        raise ValueError(msg)
    return {
        "valid": not errors,
        "errors": errors,
        "fields": sorted(str(key) for key in payload),
    }


def session_report_dict(
    report: SessionReport,
    *,
    analysis_error: str | None = None,
) -> ReportPayload:
    """Return a validated JSON payload from a :class:`SessionReport`."""
    payload = report.to_dict()
    if analysis_error is not None:
        payload = {**payload, "analysis_error": analysis_error}
    validate_report_payload(payload, raise_on_error=True)
    return payload


def empty_session_report_dict(
    *,
    n_samples: int = 0,
    duration_s: float = 0.0,
    quality: NumericReportDict | None = None,
    config: ReportPayload | None = None,
) -> ReportPayload:
    """Minimal report payload for empty or not-yet-analysable sessions."""
    return session_report_dict(
        SessionReport(
            n_samples=n_samples,
            duration_s=duration_s,
            fixations=[],
            saccades=[],
            quality=quality or {},
            config=config or {},
        )
    )


def partial_session_report_dict(
    *,
    n_samples: int,
    duration_s: float,
    quality: NumericReportDict,
    config: ReportPayload | None = None,
) -> ReportPayload:
    """Report payload when samples exist but full analysis is not yet possible."""
    return empty_session_report_dict(
        n_samples=n_samples,
        duration_s=duration_s,
        quality=quality,
        config=config,
    )


def error_session_report_dict(
    *,
    n_samples: int,
    duration_s: float,
    quality: NumericReportDict,
    error: str,
    config: ReportPayload | None = None,
) -> ReportPayload:
    """Report payload when analysis fails but session metadata is known."""
    return session_report_dict(
        SessionReport(
            n_samples=n_samples,
            duration_s=duration_s,
            fixations=[],
            saccades=[],
            quality=quality,
            config=config or {},
        ),
        analysis_error=error,
    )
