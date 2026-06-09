"""Tests for statistical interpretation ledgers."""

from __future__ import annotations

import json
from pathlib import Path

from itrace.stats.evidence import (
    build_statistical_interpretation_ledger,
    load_statistical_interpretation_ledger,
)

ROOT = Path(__file__).resolve().parent.parent


def _minimal_statistical_diagnostics() -> dict[str, object]:
    return {
        "kind": "itrace_statistical_diagnostics",
        "amplitude_shape_summary": {
            "available": True,
            "median": 0.45,
            "iqr": 10.0,
            "iqr_outlier_count": 0,
            "n": 15,
        },
        "amplitude_model_comparison": {
            "available": True,
            "best_family": "lognormal",
            "families": [{"family": "lognormal", "akaike_weight": 0.63}],
            "model_selection_bootstrap": {
                "available": True,
                "top_family": "lognormal",
                "top_frequency": 0.75,
            },
        },
        "amplitude_quantile_diagnostics": {
            "available": True,
            "residual_rmse_deg": 1.2,
        },
        "amplitude_cdf_diagnostics": {
            "available": True,
            "max_abs_cdf_residual": 0.2,
            "dkw_epsilon": 0.35,
        },
        "main_sequence_exponent": {
            "available": True,
            "estimate": 0.91,
            "ci_low": 0.88,
            "ci_high": 0.92,
        },
        "spatial_stability": {
            "gaze_dispersion": 14.86,
            "bcea": 449.54,
            "fixation_position_entropy": 2.43,
        },
        "transition_matrix": {"symbols": ["L", "R"], "scanpath": "LRRL"},
    }


def test_statistical_interpretation_ledger_summarizes_methods_and_boundaries() -> None:
    payload = build_statistical_interpretation_ledger(
        statistical_diagnostics=_minimal_statistical_diagnostics(),
        range_bridge={
            "kind": "itrace_synthetic_empirical_range_bridge",
            "summary": {
                "metric_count": 10,
                "stress_domain_count": 3,
                "not_directly_comparable_count": 2,
            },
        },
        noise_metrics={
            "thresholds": {
                "saccade_f1_0_8": {"pixels_at_640": 0.92},
                "gaze_rms_2deg": {"pixels_at_640": 3.05},
            }
        },
        empirical_metrics={"available": True},
    )

    assert payload["kind"] == "itrace_statistical_interpretation_ledger"
    assert payload["summary"]["row_count"] == 8
    assert payload["summary"]["available_count"] == 8
    assert "reference-device validation" in payload["truth_boundary"]
    rows = {row["id"]: row for row in payload["rows"]}
    assert (
        rows["robust_amplitude_shape"]["reported_value"]
        == "median 0.45 deg; IQR 10.00 deg; outliers 0/15"
    )
    assert (
        rows["relative_model_comparison"]["reported_value"]
        == "best lognormal; w(AIC) 0.63; boot lognormal 75%"
    )
    assert (
        rows["distribution_residual_checks"]["reported_value"]
        == "QQ RMSE 1.20 deg; P-P max |res| 0.20; DKW +/- 0.35"
    )
    assert (
        rows["synthetic_empirical_context"]["reported_value"]
        == "10 rows; 3 stress; 2 non-comparable"
    )
    encoded = json.dumps(payload).lower()
    assert "posterior probability" in encoded
    assert "device validation" in encoded
    assert "validates accuracy" not in encoded
    assert "validates device performance" not in encoded


def test_statistical_interpretation_ledger_keeps_missing_sources_unavailable() -> None:
    payload = build_statistical_interpretation_ledger(
        statistical_diagnostics={},
        range_bridge={},
        noise_metrics={},
        empirical_metrics={"available": False},
    )

    assert payload["empirical_available"] is False
    assert payload["summary"]["row_count"] == 8
    assert payload["summary"]["available_count"] < 8
    rows = {row["id"]: row for row in payload["rows"]}
    assert rows["robust_amplitude_shape"]["available"] is False
    assert rows["robust_amplitude_shape"]["reported_value"] == "shape unavailable"
    assert rows["idealized_noise_sensitivity"]["reported_value"] == "noise thresholds unavailable"


def test_checked_in_statistical_interpretation_ledger_loads_repo_artifacts() -> None:
    payload = load_statistical_interpretation_ledger(ROOT)
    encoded = json.dumps(payload)

    assert payload["empirical_available"] is True
    assert payload["summary"]["available_count"] >= 7
    assert "statistical_diagnostics.json" in encoded
    assert "synthetic_empirical_range_bridge.json" in encoded
    assert "validates accuracy" not in encoded.lower()
