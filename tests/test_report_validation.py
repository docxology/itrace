"""Report payload validation tests."""

from __future__ import annotations

import pytest

from itrace import pipeline
from itrace.live.analysis import analysis_payload
from itrace.reporting import (
    empty_session_report_dict,
    error_session_report_dict,
    partial_session_report_dict,
    validate_report_payload,
)
from itrace.synthetic import gaze_with_saccade


def test_validate_report_payload_accepts_session_report_dict() -> None:
    gaze, _ = gaze_with_saccade()
    payload = pipeline.analyze_gaze(gaze).to_dict()

    result = validate_report_payload(payload)

    assert result["valid"] is True
    assert result["errors"] == []
    assert "n_saccades" in result["fields"]


def test_validate_report_payload_rejects_missing_or_wrong_typed_fields() -> None:
    payload = {"n_samples": "bad", "duration_s": 1.0}

    result = validate_report_payload(payload)

    assert result["valid"] is False
    assert result["errors"]
    assert any("n_samples" in error for error in result["errors"])


def test_validate_report_payload_can_raise() -> None:
    with pytest.raises(ValueError, match="report payload"):
        validate_report_payload({"n_samples": "bad"}, raise_on_error=True)


def test_reporting_helpers_emit_validated_payloads() -> None:
    for payload in (
        empty_session_report_dict(),
        partial_session_report_dict(
            n_samples=2,
            duration_s=0.1,
            quality={"finite_sample_fraction": 1.0},
        ),
        error_session_report_dict(
            n_samples=4,
            duration_s=0.2,
            quality={"finite_sample_fraction": 0.5},
            error="too few finite samples",
        ),
    ):
        result = validate_report_payload(payload)
        assert result["valid"] is True
        assert result["errors"] == []


def test_live_analysis_payloads_validate() -> None:
    empty = analysis_payload(
        [],
        method="ivt",
        velocity_threshold_deg_s=30.0,
        include_pso=False,
    )
    empty_result = validate_report_payload(empty["report"])
    assert empty_result["valid"] is True
