from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from itrace import mainsequence
from itrace.stats import bootstrap, distributions, scanpath_metrics


def test_fixed_location_changes_information_criterion_parameter_count() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0])
    fit = distributions.fit_distribution(data, "normal", floc=0.0)
    assert fit.k == 1
    assert fit.aic == pytest.approx(2.0 - 2.0 * fit.loglik)
    assert fit.bic == pytest.approx(np.log(data.size) - 2.0 * fit.loglik)


def test_no_successful_exponent_resample_is_an_error() -> None:
    amplitude = np.array([2.0, 6.0, 18.0])
    velocity = 50.0 * amplitude**0.5
    with pytest.raises(ValueError, match="no valid bootstrap"):
        scanpath_metrics.main_sequence_exponent_ci(amplitude, velocity, n_boot=1, seed=4)


def test_constant_amplitude_has_no_identifiable_power_law() -> None:
    with pytest.raises(ValueError, match="distinct amplitudes"):
        mainsequence.fit(np.ones(5), np.arange(1.0, 6.0))


def test_two_distinct_amplitudes_identify_a_resampled_log_linear_fit() -> None:
    amplitude = np.array([2.0, 6.0, 18.0])
    velocity = 50 * amplitude**0.5
    result = scanpath_metrics.main_sequence_exponent_bootstrap(
        amplitude, velocity, n_boot=1, seed=0
    )
    assert result.n_successful == 1
    assert result.estimate == pytest.approx(0.5)


def test_moving_blocks_preserve_order_and_have_exact_draw_size() -> None:
    draws = list(bootstrap.bootstrap_indices(11, n_boot=8, block_length=3, seed=7))
    assert len(draws) == 8
    assert all(draw.shape == (11,) for draw in draws)
    for draw in draws:
        assert np.all((draw >= 0) & (draw < 11))
        for start in (0, 3, 6):
            np.testing.assert_array_equal(np.diff(draw[start : start + 3]), [1, 1])
        assert draw[10] == draw[9] + 1
    repeated = list(bootstrap.bootstrap_indices(11, n_boot=8, block_length=3, seed=7))
    np.testing.assert_array_equal(draws, repeated)


@pytest.mark.parametrize("block_length", [0, -1, 12, 1.5, True])
def test_invalid_block_length_is_rejected(block_length) -> None:
    with pytest.raises(ValueError, match="block_length"):
        list(bootstrap.bootstrap_indices(11, n_boot=2, block_length=block_length))


@pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
def test_invalid_bootstrap_count_is_rejected(n_boot) -> None:
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap.bootstrap_statistic([1.0, 2.0], np.mean, n_boot=n_boot)


@pytest.mark.parametrize("values", [[1.0, np.nan], [1.0, np.inf], [[1.0, 2.0]]])
def test_bootstrap_rejects_nonfinite_or_multidimensional_data(values) -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        bootstrap.bootstrap_statistic(values, np.mean, n_boot=2)


def test_bootstrap_rejects_nonfinite_statistic() -> None:
    with pytest.raises(ValueError, match="statistic returned a nonfinite"):
        bootstrap.bootstrap_statistic([1.0, 2.0], lambda _: float("nan"), n_boot=2)


def test_block_bootstrap_retains_dependence_in_seeded_ar1_control() -> None:
    rng = np.random.default_rng(381)
    innovations = rng.normal(size=1024)
    series = np.empty(innovations.size)
    series[0] = innovations[0]
    for i in range(1, series.size):
        series[i] = 0.85 * series[i - 1] + innovations[i]
    iid = bootstrap.bootstrap_statistic(series, np.mean, n_boot=600, seed=9)
    blocks = bootstrap.bootstrap_statistic(series, np.mean, n_boot=600, seed=9, block_length=16)
    assert np.std(blocks) > 1.8 * np.std(iid)
    assert np.std(blocks) < 4.0 * np.std(iid)


def test_parametric_ks_refits_each_draw_against_independent_normal_oracle() -> None:
    data = np.random.default_rng(33).normal(4.0, 2.0, 60)
    result = distributions.parametric_ks_test(data, "normal", n_boot=39, seed=13)
    observed = stats.kstest(data, stats.norm(data.mean(), data.std()).cdf).statistic
    rng = np.random.default_rng(13)
    exceedances = 0
    for _ in range(39):
        draw = rng.normal(data.mean(), data.std(), data.size)
        distance = stats.kstest(draw, stats.norm(draw.mean(), draw.std()).cdf).statistic
        exceedances += distance >= observed
    assert result.statistic == pytest.approx(observed)
    assert result.exceedances == exceedances
    assert result.pvalue == pytest.approx((exceedances + 1) / 40)
    assert result.minimum_pvalue == pytest.approx(1 / 40)
    assert distributions.parametric_ks_test(data, "normal", n_boot=39, seed=13) == result


def test_parametric_ks_detects_misspecified_normal_distribution() -> None:
    rng = np.random.default_rng(8)
    mixture = np.r_[rng.normal(-4, 0.2, 100), rng.normal(4, 0.2, 100)]
    result = distributions.parametric_ks_test(mixture, "normal", n_boot=99, seed=5)
    assert result.pvalue == pytest.approx(0.01)


def test_parametric_ks_respects_fixed_location_and_rejects_invalid_count() -> None:
    data = np.random.default_rng(12).normal(0, 2, 50)
    result = distributions.parametric_ks_test(data, "normal", floc=0.0, n_boot=9, seed=3)
    assert result.fixed_location == 0.0
    assert 0.1 <= result.pvalue <= 1.0
    with pytest.raises(ValueError, match="n_boot"):
        distributions.parametric_ks_test(data, "normal", n_boot=0)


def test_exponent_bootstrap_reports_failed_draws_and_block_design() -> None:
    amplitude = np.linspace(2.0, 20.0, 30)
    velocity = 40.0 * amplitude**0.65
    result = scanpath_metrics.main_sequence_exponent_bootstrap(
        amplitude, velocity, n_boot=30, seed=7, block_length=4
    )
    assert result.estimate == pytest.approx(0.65)
    assert result.n_successful + result.n_failed == result.n_boot == 30
    assert result.block_length == 4
    assert result.ci_low == pytest.approx(0.65)
    assert result.ci_high == pytest.approx(0.65)


def test_resampling_demo_analytic_variance_matches_full_covariance_matrix() -> None:
    payload = bootstrap.resampling_sensitivity(n=32, n_boot=20, rho=0.7, seed=18)
    index = np.arange(32)
    covariance = 0.7 ** np.abs(index[:, None] - index[None, :]) / (1 - 0.7**2)
    assert payload["analytic_mean_se"] == pytest.approx(np.sqrt(covariance.sum() / 32**2))
    assert payload == bootstrap.resampling_sensitivity(n=32, n_boot=20, rho=0.7, seed=18)
    iid = bootstrap.resampling_sensitivity(n=32, n_boot=20, rho=0, seed=18)
    assert iid["analytic_mean_se"] == pytest.approx(1 / np.sqrt(32))


@pytest.mark.parametrize("kwargs", [{"n": 3}, {"rho": 1.0}, {"rho": np.nan}, {"n_boot": 1}])
def test_resampling_demo_rejects_invalid_design(kwargs) -> None:
    with pytest.raises(ValueError, match=r"n|rho|n_boot"):
        bootstrap.resampling_sensitivity(**kwargs)


@pytest.mark.parametrize(
    ("predicted", "truth"),
    [([(0, 100)], [(10, 20), (30, 40)]), ([(10, 20), (30, 40)], [(10, 40)])],
)
def test_closed_loop_penalizes_merging_and_fragmentation(predicted, truth) -> None:
    from itrace.scene import _saccade_prf

    precision, recall, f1 = _saccade_prf(predicted, truth)
    assert f1 == pytest.approx(2 / 3)
    assert min(precision, recall) == 0.5


def test_closed_loop_touching_endpoints_have_no_positive_overlap() -> None:
    from itrace.scene import _saccade_prf

    assert _saccade_prf([(10, 20)], [(20, 30)]) == (0.0, 0.0, 0.0)


def test_noise_table_preserves_asymmetric_percentile_endpoints() -> None:
    from itrace import power

    curve = power.MetricCurve("saccade_f1", True, [0.0], [0.8], [0.2], [0.9], [0.1])
    result = power.NoiseSweepResult([0.0], 25, {"saccade_f1": curve})
    result.curves["gaze_rms_deg"] = curve
    result.curves["pupil_corr"] = curve
    assert "0.800 [0.200, 0.900]" in power.format_summary_markdown(result)
    assert power.summary_records(result)[0]["saccade_f1_mcse"] == pytest.approx(0.02)


@pytest.mark.parametrize("levels", [[], [-1.0], [np.nan], [0.1, 0.0], [0.0, 0.0]])
def test_noise_sweep_rejects_invalid_grid_before_running(levels) -> None:
    from itrace import power

    with pytest.raises(ValueError, match="noise_levels"):
        power.run_noise_sweep(levels, n_trials=2)


@pytest.mark.parametrize("values", [[], [np.nan], [[1.0, 2.0]]])
def test_noise_interval_rejects_invalid_trials(values) -> None:
    from itrace import power

    with pytest.raises(ValueError, match="trial values"):
        power._mean_ci(np.asarray(values))
