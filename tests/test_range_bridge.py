"""Tests for synthetic-vs-empirical range bridge payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itrace.stats.range_bridge import (
    build_synthetic_empirical_range_bridge,
    load_synthetic_empirical_range_bridge,
)

ROOT = Path(__file__).resolve().parent.parent


def _minimal_synthetic_validation() -> dict[str, object]:
    return {
        "kind": "synthetic_domain_validation",
        "truth_boundary": "Synthetic domains have ground truth; live webcam diagnostics do not.",
        "domains": [
            {
                "domain": "clean_desktop",
                "label": "Clean desktop",
                "summary": {
                    "gaze_finite_fraction": {"mean": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                    "pupil_valid_fraction": {"mean": 0.98, "ci_low": 0.98, "ci_high": 0.98},
                    "saccade_f1": {"mean": 0.82, "ci_low": 0.70, "ci_high": 0.92},
                    "amplitude_mae_deg": {"mean": 0.06, "ci_low": 0.03, "ci_high": 0.09},
                },
            },
            {
                "domain": "low_light_dropout",
                "label": "Low-light dropout",
                "summary": {
                    "gaze_finite_fraction": {"mean": 0.95, "ci_low": 0.94, "ci_high": 0.96},
                    "pupil_valid_fraction": {"mean": 0.92, "ci_low": 0.92, "ci_high": 0.92},
                    "saccade_f1": {"mean": 0.65, "ci_low": 0.56, "ci_high": 0.77},
                    "amplitude_mae_deg": {"mean": 0.94, "ci_low": 0.31, "ci_high": 2.05},
                },
            },
        ],
    }


def _minimal_noise_metrics() -> dict[str, object]:
    return {
        "kind": "itrace_noise_sweep_metrics",
        "thresholds": {
            "saccade_f1_0_8": {"pixels_at_640": 0.92, "sigma_normalised": 0.0014},
            "gaze_rms_2deg": {"pixels_at_640": 3.05, "sigma_normalised": 0.0048},
        },
        "records": [
            {"noise_px": 0.0, "gaze_rms_deg_mean": 0.16, "saccade_f1_mean": 1.0},
            {"noise_px": 2.56, "gaze_rms_deg_mean": 1.66, "saccade_f1_mean": 0.22},
        ],
    }


def _minimal_statistical_diagnostics() -> dict[str, object]:
    return {
        "kind": "itrace_statistical_diagnostics",
        "amplitude_model_comparison": {
            "best_family": "lognormal",
            "model_selection_bootstrap": {
                "available": True,
                "top_family": "lognormal",
                "top_frequency": 0.75,
            },
        },
        "amplitude_shape_summary": {
            "available": True,
            "median": 0.45,
            "iqr": 10.02,
        },
    }


def _available_empirical() -> dict[str, object]:
    return {
        "kind": "empirical_pilot_metrics",
        "available": True,
        "source_report": "output/empirical_pilot/local_pilot_001/experiment/experiment_report.json",
        "sample_count": 3701,
        "completed_trial_count": 3,
        "finite_gaze_fraction": 1.0,
        "sampling_rate_hz": 30.0,
        "sampling_interval_cv": 0.047,
        "max_drift_deg_s": 0.025,
        "pupil_valid_fraction": 1.0,
        "heldout_target_error": {"available": True, "rms_error_deg": 16.26},
        "target_acquisition_latency_s": {"available": True, "median_latency_s": 0.42},
    }


def test_range_bridge_payload_available_empirical_keeps_boundaries() -> None:
    payload = build_synthetic_empirical_range_bridge(
        empirical_metrics=_available_empirical(),
        synthetic_validation=_minimal_synthetic_validation(),
        noise_metrics=_minimal_noise_metrics(),
        statistical_diagnostics=_minimal_statistical_diagnostics(),
        empirical_report={
            "trials": {
                "fixed_center": {"sampling_rate_hz": 30.1, "sampling_interval_cv": 0.05},
                "reading": {"sampling_rate_hz": 29.9, "sampling_interval_cv": 0.04},
            }
        },
    )

    assert payload["kind"] == "itrace_synthetic_empirical_range_bridge"
    assert payload["empirical_available"] is True
    assert "not reference-device validation" in payload["truth_boundary"]
    assert "output/empirical_pilot/local_pilot_001/experiment/experiment_report.json" in json.dumps(
        payload["sources"]
    )
    assert payload["trial_spread"]["available"] is True
    assert payload["trial_spread"]["metrics"]["sampling_rate_hz"]["min"] == pytest.approx(29.9)
    rows = {row["id"]: row for row in payload["metrics"]}
    assert rows["finite_gaze_fraction"]["empirical"]["display"] == "100.0%"
    assert rows["finite_gaze_fraction"]["synthetic"]["display"] == "clean 100.0%; stress min 95.0%"
    assert rows["heldout_target_rms_deg"]["comparability"] == "not_directly_comparable"
    assert rows["target_acquisition_latency_s"]["interpretation"].startswith("UI/capture")
    assert rows["model_selection_stability"]["empirical"]["available"] is False
    assert rows["model_selection_stability"]["statistical_model"]["display"] == "lognormal top 75%"
    assert "raw eye video" in payload["storage_boundary"]
    assert "persisted eye-crop images" in payload["storage_boundary"]
    assert "validates accuracy" not in json.dumps(payload).lower()
    assert "device performance" not in json.dumps(payload).lower()


def test_range_bridge_payload_unavailable_empirical_stays_explicit() -> None:
    payload = build_synthetic_empirical_range_bridge(
        empirical_metrics={"available": False},
        synthetic_validation=_minimal_synthetic_validation(),
        noise_metrics=_minimal_noise_metrics(),
        statistical_diagnostics=_minimal_statistical_diagnostics(),
    )

    assert payload["empirical_available"] is False
    assert payload["trial_spread"]["available"] is False
    rows = {row["id"]: row for row in payload["metrics"]}
    assert rows["finite_gaze_fraction"]["empirical"]["available"] is False
    assert rows["finite_gaze_fraction"]["empirical"]["display"] == "unavailable"
    assert rows["heldout_target_rms_deg"]["empirical"]["display"] == "unavailable"


def test_range_bridge_rejects_absolute_or_raw_video_provenance() -> None:
    empirical = _available_empirical()
    empirical["source_report"] = "/tmp/outside/experiment_report.json"
    with pytest.raises(ValueError, match="repo-relative"):
        build_synthetic_empirical_range_bridge(
            empirical_metrics=empirical,
            synthetic_validation=_minimal_synthetic_validation(),
            noise_metrics=_minimal_noise_metrics(),
            statistical_diagnostics=_minimal_statistical_diagnostics(),
        )

    empirical = _available_empirical()
    empirical["source_report"] = "output/empirical_pilot/raw_eye_video.mp4"
    with pytest.raises(ValueError, match="raw video"):
        build_synthetic_empirical_range_bridge(
            empirical_metrics=empirical,
            synthetic_validation=_minimal_synthetic_validation(),
            noise_metrics=_minimal_noise_metrics(),
            statistical_diagnostics=_minimal_statistical_diagnostics(),
        )


def test_checked_in_range_bridge_inputs_have_repo_relative_provenance() -> None:
    payload = load_synthetic_empirical_range_bridge(ROOT)
    encoded = json.dumps(payload)

    assert payload["empirical_available"] is True
    assert payload["source_report_available"] is True
    assert payload["trial_spread"]["available"] is True
    assert all(not Path(path).is_absolute() for path in payload["sources"].values() if path)
    assert ".mp4" not in encoded.lower()
    assert ".mov" not in encoded.lower()
    assert ".webm" not in encoded.lower()
