"""Tests for itrace.stats.distributions (maximum-likelihood fitting).

No mocks: every test uses deterministic synthetic samples drawn with an
explicitly seeded ``np.random.default_rng`` and asserts against the known
generating parameters, plus real PNG output for any plotting path.
"""

from __future__ import annotations

import numpy as np
import pytest

from itrace.stats.distributions import (
    FAMILIES,
    FitResult,
    best_fit,
    compare_distributions,
    fit_distribution,
    frozen_from_result,
    information_weights,
)


def _gamma_sample(n: int = 600, shape: float = 2.5, scale: float = 0.18) -> np.ndarray:
    """Deterministic gamma sample with known shape/scale."""
    rng = np.random.default_rng(0)
    return rng.gamma(shape, scale, n)


def test_families_constant_matches_registry() -> None:
    assert FAMILIES == ("normal", "gamma", "lognormal", "weibull", "exgaussian")


def test_gamma_fit_recovers_parameters() -> None:
    shape_true, scale_true = 2.5, 0.18
    data = _gamma_sample(shape=shape_true, scale=scale_true)
    result = fit_distribution(data, "gamma")
    assert result.family == "gamma"
    # floc=0 -> 2 free params (shape, scale).
    assert result.k == 2
    assert set(result.params) == {"shape", "scale"}
    assert result.params["shape"] == pytest.approx(shape_true, rel=0.20)
    assert result.params["scale"] == pytest.approx(scale_true, rel=0.20)
    assert result.n == data.size
    assert np.isfinite(result.loglik)
    # AIC/BIC consistency with k, n, loglik.
    assert result.aic == pytest.approx(2 * result.k - 2 * result.loglik)
    assert result.bic == pytest.approx(result.k * np.log(result.n) - 2 * result.loglik)
    assert result.aicc == pytest.approx(
        result.aic + (2 * result.k * (result.k + 1)) / (result.n - result.k - 1)
    )


def test_gamma_beats_normal_on_gamma_data() -> None:
    data = _gamma_sample()
    gamma = fit_distribution(data, "gamma")
    normal = fit_distribution(data, "normal")
    assert gamma.aic < normal.aic


def test_gamma_fit_ks_pvalue_reasonable() -> None:
    data = _gamma_sample()
    result = fit_distribution(data, "gamma")
    assert 0.0 <= result.ks_pvalue <= 1.0
    # The true family should not be rejected at the 1% level.
    assert result.ks_pvalue > 0.01
    assert 0.0 <= result.ks_statistic <= 1.0


def test_normal_fit_has_three_scipy_params_two_readable() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 2.0, 500)
    result = fit_distribution(data, "normal")
    assert result.k == 2
    assert set(result.params) == {"mu", "sigma"}
    assert result.params["mu"] == pytest.approx(5.0, abs=0.4)
    assert result.params["sigma"] == pytest.approx(2.0, rel=0.20)
    # scipy norm.fit returns (loc, scale).
    assert len(result.scipy_params) == 2


def test_lognormal_and_weibull_fit_on_positive_data() -> None:
    data = _gamma_sample()
    for family in ("lognormal", "weibull"):
        result = fit_distribution(data, family)
        assert result.family == family
        assert result.k == 2
        assert set(result.params) == {"shape", "scale"}
        # floc=0 fixes the location at the origin.
        assert result.scipy_params[1] == pytest.approx(0.0, abs=1e-12)


def test_exgaussian_fits_exponnorm_sample() -> None:
    rng = np.random.default_rng(2)
    # ex-Gaussian = Gaussian + exponential; build it directly.
    data = rng.normal(0.3, 0.04, 500) + rng.exponential(0.12, 500)
    result = fit_distribution(data, "exgaussian")
    assert result.family == "exgaussian"
    assert result.k == 3
    assert set(result.params) == {"K", "mu", "sigma"}
    assert np.isfinite(result.loglik)
    assert len(result.scipy_params) == 3


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        fit_distribution(_gamma_sample(), "cauchy")


def test_too_few_finite_samples_raises() -> None:
    with pytest.raises(ValueError, match=">=2 finite samples"):
        fit_distribution(np.array([1.0]), "gamma")


def test_non_finite_values_dropped_before_count() -> None:
    # After dropping the NaN/inf only one finite sample remains -> raises.
    with pytest.raises(ValueError, match=">=2 finite samples"):
        fit_distribution(np.array([2.0, np.nan, np.inf, -np.inf]), "gamma")


def test_positive_support_rejects_nonpositive_data() -> None:
    data = np.array([1.0, 2.0, -1.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="positive support"):
        fit_distribution(data, "gamma")


def test_normal_accepts_nonpositive_data() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, 400)  # straddles zero
    result = fit_distribution(data, "normal")
    assert result.family == "normal"
    assert np.isfinite(result.loglik)


def test_explicit_floc_overrides_default() -> None:
    data = _gamma_sample() + 0.05
    result = fit_distribution(data, "gamma", floc=0.05)
    assert result.scipy_params[1] == pytest.approx(0.05, abs=1e-9)


def test_frozen_from_result_pdf_and_cdf() -> None:
    data = _gamma_sample()
    result = fit_distribution(data, "gamma")
    frozen = frozen_from_result(result)
    grid = np.linspace(0.01, 2.0, 50)
    pdf = frozen.pdf(grid)
    cdf = frozen.cdf(grid)
    assert pdf.shape == grid.shape
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0.0)
    # CDF is monotone non-decreasing and bounded in [0, 1].
    assert np.all(np.diff(cdf) >= -1e-9)
    assert cdf[0] >= 0.0
    assert cdf[-1] <= 1.0


def test_frozen_from_result_unknown_family_raises() -> None:
    data = _gamma_sample()
    result = fit_distribution(data, "gamma")
    bad = FitResult(
        family="weird",
        params=result.params,
        loglik=result.loglik,
        aic=result.aic,
        bic=result.bic,
        ks_statistic=result.ks_statistic,
        ks_pvalue=result.ks_pvalue,
        n=result.n,
        scipy_params=result.scipy_params,
    )
    with pytest.raises(ValueError, match="unknown family"):
        frozen_from_result(bad)


def test_compare_distributions_sorted_by_aic() -> None:
    data = _gamma_sample()
    results = compare_distributions(data)
    assert len(results) == len(FAMILIES)
    aics = [r.aic for r in results]
    assert aics == sorted(aics)
    # The gamma family should be among the strongest fits on gamma data.
    assert results[0].family in {"gamma", "lognormal", "weibull", "exgaussian"}


def test_information_weights_are_normalized_relative_support() -> None:
    data = _gamma_sample()
    results = compare_distributions(data)

    weights = information_weights(results, criterion="aic")
    assert set(weights) == {result.family for result in results}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(0.0 <= weight <= 1.0 for weight in weights.values())
    assert weights[results[0].family] == max(weights.values())

    aicc_weights = information_weights(results, criterion="aicc")
    assert sum(aicc_weights.values()) == pytest.approx(1.0)


def test_information_weights_reject_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="criterion must be"):
        information_weights([fit_distribution(_gamma_sample(), "gamma")], criterion="hqic")


def test_aicc_is_infinite_when_small_sample_correction_is_undefined() -> None:
    result = fit_distribution(np.array([1.0, 2.0, 3.0]), "gamma")
    assert np.isinf(result.aicc)


def test_compare_distributions_skips_failing_families() -> None:
    # Data straddling zero: positive-support families must be skipped.
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 1.0, 400)
    results = compare_distributions(data)
    families = {r.family for r in results}
    assert "normal" in families
    assert "gamma" not in families
    assert "lognormal" not in families
    assert "weibull" not in families


def test_compare_distributions_custom_family_list() -> None:
    data = _gamma_sample()
    results = compare_distributions(data, families=["normal", "gamma"])
    assert {r.family for r in results} == {"normal", "gamma"}


def test_compare_distributions_can_be_empty() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, 300)
    # Only positive-support families on zero-straddling data -> all skipped.
    results = compare_distributions(data, families=["gamma", "lognormal", "weibull"])
    assert results == []


def test_best_fit_returns_lowest_aic() -> None:
    data = _gamma_sample()
    results = compare_distributions(data)
    best = best_fit(data)
    assert best.family == results[0].family
    assert best.aic == min(r.aic for r in results)


def test_best_fit_raises_when_nothing_fits() -> None:
    rng = np.random.default_rng(6)
    data = rng.normal(0.0, 1.0, 300)
    with pytest.raises(ValueError, match="no distribution family"):
        best_fit(data, families=["gamma", "lognormal", "weibull"])


def test_fitresult_is_frozen() -> None:
    result = fit_distribution(_gamma_sample(), "gamma")
    with pytest.raises((AttributeError, TypeError)):
        result.family = "normal"  # type: ignore[misc]
