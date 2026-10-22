from __future__ import annotations

import json

import pytest

from itrace.study import reconstruct
from itrace.study.jsonio import mapping, sequence

from .helpers import Harness
from .recording_helpers import finish_behavior


def test_post_validation_repeats_heldout_with_frozen_fit_when_behavior_finished(
    ready: Harness,
) -> None:
    before = mapping(ready.session.status()["calibration"], "calibration")[
        "coefficients_gaze_deg_to_css_px"
    ]
    finish_behavior(ready)
    assert ready.session.status()["state"] == "post_validation"
    target = mapping(ready.session.status()["target"], "target")
    assert (target["phase"], target["x_fraction"], target["y_fraction"]) == (
        "post_validation",
        0.3,
        0.3,
    )
    for _ in range(5):
        ready.target()
    diagnostics = mapping(ready.session.status()["calibration"], "calibration")
    report = reconstruct(ready.directory)
    rebuilt = mapping(report["calibration_diagnostics"], "diagnostics")
    assert ready.session.status()["state"] == report["state"] == "complete"
    assert diagnostics["coefficients_gaze_deg_to_css_px"] == before
    assert diagnostics["post_validation_adequate"] is True
    assert rebuilt["post_validation_adequate"] is True
    assert len(sequence(rebuilt["pre_post_residuals"], "residuals")) == 5
    assert report["correctness_denominator"] == report["scored_correct"] == 2


@pytest.mark.parametrize("condition", ["missing", "inaccurate", "few"])
def test_post_quality_failure_retained_without_redo_when_behavior_valid(
    ready: Harness, condition: str
) -> None:
    finish_behavior(ready)
    for _ in range(5):
        ready.target(
            finite=condition != "missing",
            offset=100 if condition == "inaccurate" else 0,
            count=5 if condition == "few" else 20,
        )
    report = reconstruct(ready.directory)
    diagnostics = mapping(report["calibration_diagnostics"], "diagnostics")
    assert report["state"] == "complete"
    assert report["correctness_denominator"] == report["scored_correct"] == 2
    assert diagnostics["post_validation_adequate"] is False
    assert len(sequence(report["post_validation_dwells"], "post")) == 5
    assert all(
        mapping(row, "trial")["outcome"] == "answered"
        for row in sequence(report["trials"], "trials")
    )
    json.dumps(report, allow_nan=False)


def test_post_partial_export_preserves_answers_when_aborted(ready: Harness) -> None:
    finish_behavior(ready)
    ready.target()
    ready.session.abort({"reason": "escape"})
    report = mapping(ready.session.export()["report"], "report")
    assert report["state"] == "aborted"
    assert report["behavioral_trials_finished"] is True
    assert report["correctness_denominator"] == 2
    assert len(sequence(report["post_validation_dwells"], "post")) == 1


def test_post_capture_stop_preserves_behavior_when_technical_interruption(ready: Harness) -> None:
    finish_behavior(ready)
    ready.session.capture_state(False, "capture1", "failed")
    report = reconstruct(ready.directory)
    assert report["state"] == "interrupted"
    assert report["correctness_denominator"] == 2


def test_post_error_change_reports_separate_residual_when_quality_shifted(ready: Harness) -> None:
    finish_behavior(ready)
    for _ in range(5):
        ready.target(offset=1)
    diagnostics = mapping(reconstruct(ready.directory)["calibration_diagnostics"], "diagnostics")
    point = mapping(sequence(diagnostics["pre_post_residuals"], "residuals")[0], "point")
    assert point["error_change_css_px"] == pytest.approx(20)
    assert point["residual_change_x_css_px"] == pytest.approx(20)
    assert point["residual_change_y_css_px"] == pytest.approx(0, abs=1e-8)
