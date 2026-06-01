"""Report payload validation tests."""

from __future__ import annotations

import pytest

from itrace import pipeline
from itrace.reporting import validate_report_payload
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
