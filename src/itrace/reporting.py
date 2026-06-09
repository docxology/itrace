"""Lightweight report payload validation.

The package intentionally avoids a runtime schema dependency. This module checks
the JSON-friendly dictionary emitted by :meth:`itrace.types.SessionReport.to_dict`
with small, explicit type predicates.
"""

from __future__ import annotations

from collections.abc import Mapping

from .types import NumericReportDict, ReportPayload, SessionReport


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


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


def session_report_dict(
    report: SessionReport,
    *,
    analysis_error: str | None = None,
) -> ReportPayload:
    """Return a validated JSON payload from a :class:`SessionReport`."""
    payload = report.to_dict()
    if analysis_error is not None:
        payload = {**payload, "analysis_error": analysis_error}
    return payload


def empty_session_report_dict(
    *,
    n_samples: int = 0,
    duration_s: float = 0.0,
    quality: NumericReportDict | None = None,
) -> ReportPayload:
    """Minimal report payload for empty or not-yet-analysable sessions."""
    return session_report_dict(
        SessionReport(
            n_samples=n_samples,
            duration_s=duration_s,
            fixations=[],
            saccades=[],
            quality=quality or {},
        )
    )


def partial_session_report_dict(
    *,
    n_samples: int,
    duration_s: float,
    quality: NumericReportDict,
) -> ReportPayload:
    """Report payload when samples exist but full analysis is not yet possible."""
    return empty_session_report_dict(
        n_samples=n_samples,
        duration_s=duration_s,
        quality=quality,
    )


def error_session_report_dict(
    *,
    n_samples: int,
    duration_s: float,
    quality: NumericReportDict,
    error: str,
) -> ReportPayload:
    """Report payload when analysis fails but session metadata is known."""
    return session_report_dict(
        SessionReport(
            n_samples=n_samples,
            duration_s=duration_s,
            fixations=[],
            saccades=[],
            quality=quality,
        ),
        analysis_error=error,
    )
