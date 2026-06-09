"""Tests for pure statistical diagnostics payloads."""

from __future__ import annotations

import json

import numpy as np
import pytest

from itrace import pipeline
from itrace.stats.diagnostics import session_statistical_diagnostics
from itrace.synthetic import gaze_with_saccade
from itrace.types import GazeStream, Saccade, SessionReport


def _multi_saccade_report(seed: int = 5) -> SessionReport:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(3.0, 24.0, 20):
        gaze, _truth = gaze_with_saccade(
            amplitude_deg=float(amp),
            direction_deg=float(rng.uniform(-180.0, 180.0)),
            fixation_s=0.06,
        )
        xs.append(gaze.x)
        ys.append(gaze.y)
        ts.append(gaze.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    stream = GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))
    return pipeline.analyze_session(stream)


def test_session_statistical_diagnostics_is_json_serializable() -> None:
    report = _multi_saccade_report()
    payload = session_statistical_diagnostics(report, n_boot=120, seed=77)

    assert payload["kind"] == "itrace_statistical_diagnostics"
    json.dumps(payload)
    assert payload["n_saccades"] == len(report.saccades)
    assert "not population physiology" in payload["truth_boundary"]

    model = payload["amplitude_model_comparison"]
    assert model["available"] is True
    fits = model["families"]
    assert len(fits) >= 3
    assert fits[0]["delta_aic"] == 0.0
    assert fits[0]["evidence_ratio"] == pytest.approx(1.0)
    assert [row["delta_aic"] for row in fits] == sorted(row["delta_aic"] for row in fits)
    assert model["weight_criterion"] == "aic"
    assert model["aicc_weight_criterion"] == "aicc"
    assert sum(row["akaike_weight"] for row in fits) == pytest.approx(1.0)
    assert sum(row["aicc_weight"] for row in fits) == pytest.approx(1.0)
    assert all(0.0 <= row["akaike_weight"] <= 1.0 for row in fits)
    assert all(0.0 <= row["aicc_weight"] <= 1.0 for row in fits)
    assert all(row["aicc"] is None or row["aicc"] >= row["aic"] for row in fits)
    assert all(row["delta_aicc"] is None or row["delta_aicc"] >= 0.0 for row in fits)
    stability = model["model_selection_bootstrap"]
    assert stability["available"] is True
    assert stability["n_boot"] == 40
    assert stability["seed"] == 279
    assert stability["criterion"] == "aic"
    assert stability["successful_bootstraps"] <= stability["n_boot"]
    assert stability["successful_bootstraps"] == sum(stability["winner_counts"].values())
    assert sum(stability["winner_frequencies"].values()) == pytest.approx(1.0)
    assert stability["top_family"] in stability["winner_frequencies"]
    assert 0.0 <= stability["top_frequency"] <= 1.0
    assert stability["unique_winner_count"] == len(stability["winner_frequencies"])

    shape = payload["amplitude_shape_summary"]
    assert shape["available"] is True
    assert shape["n"] == payload["n_saccades"]
    assert shape["median"] > 0.0
    assert shape["q25"] <= shape["median"] <= shape["q75"]
    assert shape["iqr"] >= 0.0
    assert -1.0 <= shape["bowley_skewness"] <= 1.0
    assert shape["iqr_outlier_count"] >= 0
    assert 0.0 <= shape["iqr_outlier_fraction"] <= 1.0
    assert shape["bootstrap_n"] == 120
    assert shape["bootstrap_seed"] == 178
    assert shape["confidence"] == 0.95
    assert shape["median_ci_low"] <= shape["median"] <= shape["median_ci_high"]
    assert shape["iqr_ci_low"] <= shape["iqr"] <= shape["iqr_ci_high"]

    quantile = payload["amplitude_quantile_diagnostics"]
    assert quantile["available"] is True
    assert quantile["family"] == model["best_family"]
    assert quantile["n"] == payload["n_saccades"]
    assert len(quantile["probabilities"]) == quantile["n"]
    assert len(quantile["empirical_quantiles_deg"]) == quantile["n"]
    assert len(quantile["fitted_quantiles_deg"]) == quantile["n"]
    assert len(quantile["residuals_deg"]) == quantile["n"]
    assert quantile["residual_rmse_deg"] >= 0.0
    assert quantile["central_80_rmse_deg"] >= 0.0
    assert -1.0 <= quantile["qq_correlation"] <= 1.0

    cdf = payload["amplitude_cdf_diagnostics"]
    assert cdf["available"] is True
    assert cdf["family"] == model["best_family"]
    assert cdf["n"] == payload["n_saccades"]
    assert len(cdf["amplitude_deg"]) == cdf["n"]
    assert len(cdf["empirical_cdf"]) == cdf["n"]
    assert len(cdf["fitted_cdf"]) == cdf["n"]
    assert len(cdf["cdf_residuals"]) == cdf["n"]
    assert cdf["cdf_rmse"] >= 0.0
    assert cdf["max_abs_cdf_residual"] >= cdf["mean_abs_cdf_residual"]
    assert cdf["fit_ks_statistic"] == model["families"][0]["ks_statistic"]
    assert cdf["dkw_alpha"] == 0.05
    assert cdf["dkw_confidence"] == 0.95
    assert cdf["dkw_epsilon"] == pytest.approx(
        float(np.sqrt(np.log(2.0 / 0.05) / (2.0 * cdf["n"])))
    )
    assert cdf["exceeds_dkw_band"] is (cdf["max_abs_cdf_residual"] > cdf["dkw_epsilon"])

    exponent = payload["main_sequence_exponent"]
    assert exponent["available"] is True
    assert exponent["ci_low"] <= exponent["estimate"] <= exponent["ci_high"]
    assert exponent["seed"] == 77

    transition = payload["transition_matrix"]
    matrix = np.asarray(transition["matrix"], dtype=np.float64)
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1] == len(transition["symbols"])
    row_sums = matrix.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))


def test_session_statistical_diagnostics_empty_session_keeps_boundaries() -> None:
    report = SessionReport(n_samples=0, duration_s=0.0, fixations=[], saccades=[])
    payload = session_statistical_diagnostics(report)

    assert payload["amplitude_model_comparison"]["available"] is False
    assert payload["amplitude_shape_summary"]["available"] is False
    assert payload["amplitude_shape_summary"]["reason"] == "need >=1 positive event"
    assert payload["amplitude_quantile_diagnostics"]["available"] is False
    assert payload["amplitude_quantile_diagnostics"]["reason"] == "need >=3 positive events"
    assert payload["amplitude_cdf_diagnostics"]["available"] is False
    assert payload["amplitude_cdf_diagnostics"]["reason"] == "need >=3 positive events"
    assert payload["main_sequence_exponent"]["available"] is False
    assert payload["spatial_stability"]["bcea"] == 0.0
    assert payload["transition_matrix"]["symbols"] == []
    assert payload["transition_matrix"]["matrix"] == []


def test_session_statistical_diagnostics_reports_robust_amplitude_shape() -> None:
    amplitudes = [1.0, 2.0, 3.0, 100.0]
    saccades = [
        Saccade(
            onset_idx=idx * 10,
            offset_idx=idx * 10 + 2,
            onset_t=float(idx),
            offset_t=float(idx) + 0.02,
            amplitude_deg=amplitude,
            direction_deg=0.0,
            peak_velocity_deg_s=200.0 + amplitude,
        )
        for idx, amplitude in enumerate(amplitudes)
    ]
    report = SessionReport(n_samples=0, duration_s=4.0, fixations=[], saccades=saccades)
    payload = session_statistical_diagnostics(report, n_boot=20, seed=12)

    shape = payload["amplitude_shape_summary"]
    assert shape["available"] is True
    assert shape["mean"] == np.mean(amplitudes)
    assert shape["median"] == np.median(amplitudes)
    assert shape["q25"] == np.percentile(amplitudes, 25.0)
    assert shape["q75"] == np.percentile(amplitudes, 75.0)
    assert shape["iqr"] == np.percentile(amplitudes, 75.0) - np.percentile(amplitudes, 25.0)
    assert shape["iqr_outlier_count"] == 1
    assert shape["iqr_outlier_fraction"] == 0.25
    assert shape["iqr_outlier_upper"] < 100.0
    assert shape["bowley_skewness"] == pytest.approx(24.0 / 25.5)
    assert shape["median_ci_low"] <= shape["median"] <= shape["median_ci_high"]
    assert shape["iqr_ci_low"] <= shape["iqr"] <= shape["iqr_ci_high"]


def test_session_statistical_diagnostics_reports_quantile_residuals() -> None:
    report = _multi_saccade_report(seed=8)
    payload = session_statistical_diagnostics(report, n_boot=40, seed=3)

    quantile = payload["amplitude_quantile_diagnostics"]
    assert quantile["available"] is True
    residuals = np.asarray(quantile["residuals_deg"], dtype=np.float64)
    assert quantile["residual_rmse_deg"] == pytest.approx(float(np.sqrt(np.mean(residuals**2))))
    assert quantile["median_abs_residual_deg"] == pytest.approx(float(np.median(np.abs(residuals))))
    assert quantile["max_abs_residual_deg"] == pytest.approx(float(np.max(np.abs(residuals))))
    probabilities = np.asarray(quantile["probabilities"], dtype=np.float64)
    assert np.all(np.diff(probabilities) > 0.0)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_session_statistical_diagnostics_reports_cdf_residuals() -> None:
    report = _multi_saccade_report(seed=9)
    payload = session_statistical_diagnostics(report, n_boot=40, seed=4)

    cdf = payload["amplitude_cdf_diagnostics"]
    assert cdf["available"] is True
    empirical = np.asarray(cdf["empirical_cdf"], dtype=np.float64)
    fitted = np.asarray(cdf["fitted_cdf"], dtype=np.float64)
    residuals = np.asarray(cdf["cdf_residuals"], dtype=np.float64)
    assert np.all(np.diff(empirical) > 0.0)
    assert np.all((empirical > 0.0) & (empirical < 1.0))
    assert np.all((fitted >= 0.0) & (fitted <= 1.0))
    assert np.allclose(residuals, empirical - fitted)
    assert cdf["cdf_rmse"] == pytest.approx(float(np.sqrt(np.mean(residuals**2))))
    assert cdf["mean_abs_cdf_residual"] == pytest.approx(float(np.mean(np.abs(residuals))))
    assert cdf["max_abs_cdf_residual"] == pytest.approx(float(np.max(np.abs(residuals))))
    assert cdf["lower_tail_max_abs_cdf_residual"] >= 0.0
    assert cdf["upper_tail_max_abs_cdf_residual"] >= 0.0
    assert cdf["tail_max_abs_cdf_residual"] == pytest.approx(
        max(cdf["lower_tail_max_abs_cdf_residual"], cdf["upper_tail_max_abs_cdf_residual"])
    )
    assert cdf["cramer_von_mises_statistic"] == pytest.approx(
        float((1.0 / (12.0 * cdf["n"])) + np.sum((fitted - empirical) ** 2))
    )
    clipped = np.clip(fitted, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    indices = np.arange(1, cdf["n"] + 1, dtype=np.float64)
    assert cdf["anderson_darling_statistic"] == pytest.approx(
        float(
            -cdf["n"]
            - np.mean((2.0 * indices - 1.0) * (np.log(clipped) + np.log1p(-clipped[::-1])))
        )
    )
    assert cdf["dkw_epsilon"] == pytest.approx(
        float(np.sqrt(np.log(2.0 / cdf["dkw_alpha"]) / (2.0 * cdf["n"])))
    )


def test_session_statistical_diagnostics_rejects_invalid_cdf_band_alpha() -> None:
    report = _multi_saccade_report(seed=10)
    payload = session_statistical_diagnostics(report, cdf_band_alpha=1.5)

    cdf = payload["amplitude_cdf_diagnostics"]
    assert cdf["available"] is False
    assert cdf["reason"] == "DKW alpha must be between 0 and 1"


def test_session_statistical_diagnostics_shape_bootstrap_is_deterministic() -> None:
    report = _multi_saccade_report(seed=11)
    first = session_statistical_diagnostics(report, n_boot=80, seed=21)
    second = session_statistical_diagnostics(report, n_boot=80, seed=21)

    first_shape = first["amplitude_shape_summary"]
    second_shape = second["amplitude_shape_summary"]
    assert first_shape["median_ci_low"] == second_shape["median_ci_low"]
    assert first_shape["median_ci_high"] == second_shape["median_ci_high"]
    assert first_shape["iqr_ci_low"] == second_shape["iqr_ci_low"]
    assert first_shape["iqr_ci_high"] == second_shape["iqr_ci_high"]


def test_session_statistical_diagnostics_model_selection_bootstrap_is_deterministic() -> None:
    report = _multi_saccade_report(seed=12)
    first = session_statistical_diagnostics(report, model_boot=24, seed=31)
    second = session_statistical_diagnostics(report, model_boot=24, seed=31)

    first_stability = first["amplitude_model_comparison"]["model_selection_bootstrap"]
    second_stability = second["amplitude_model_comparison"]["model_selection_bootstrap"]
    assert first_stability["winner_counts"] == second_stability["winner_counts"]
    assert first_stability["winner_frequencies"] == second_stability["winner_frequencies"]
    assert first_stability["top_family"] == second_stability["top_family"]
    assert first_stability["top_frequency"] == second_stability["top_frequency"]
