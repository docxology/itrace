"""Maximum-likelihood distribution fitting for eye-movement statistics.

Saccade and fixation summaries (durations, amplitudes, inter-saccadic
intervals) are routinely modelled by a handful of parametric families. This
module fits those families by maximum likelihood via :mod:`scipy.stats`,
scores them with information criteria, relative information-criterion weights,
and a goodness-of-fit test, and lets the caller reconstruct a frozen
distribution for plotting.

Supported families
------------------
* ``normal``     -- :class:`scipy.stats.norm` (location-scale: ``mu``, ``sigma``).
* ``gamma``      -- :class:`scipy.stats.gamma`, positive support, ``floc=0``.
* ``lognormal``  -- :class:`scipy.stats.lognorm`, positive support, ``floc=0``.
* ``weibull``    -- :class:`scipy.stats.weibull_min`, positive support, ``floc=0``.
* ``exgaussian`` -- :class:`scipy.stats.exponnorm`, the ex-Gaussian convolution
  of a Gaussian and an exponential (``K``, ``mu``, ``sigma``); the canonical
  model for reaction-time-like and fixation-duration distributions.

The positive-support families are fitted with ``floc=0`` so that the location
is pinned to the origin and only the shape and scale are free (two free
parameters); this is the standard choice for strictly-positive quantities and
keeps the parameter count -- and therefore AIC/BIC -- well defined.

All public entry points drop non-finite samples first and raise
:class:`ValueError` on degenerate input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import stats

from ..types import FloatArray

FAMILIES: tuple[str, ...] = ("normal", "gamma", "lognormal", "weibull", "exgaussian")
"""Names of the distribution families this module can fit, in canonical order."""


@dataclass(frozen=True)
class _FamilySpec:
    """Internal description of how to fit and label one family."""

    dist_name: str
    positive_support: bool
    param_names: tuple[str, ...]


# Single source of truth: family name -> (scipy dist, support, readable params).
# ``param_names`` describes the *readable* params exposed in ``FitResult.params``;
# the raw scipy parameter tuple is always preserved verbatim in ``scipy_params``.
_REGISTRY: dict[str, _FamilySpec] = {
    "normal": _FamilySpec("norm", positive_support=False, param_names=("mu", "sigma")),
    "gamma": _FamilySpec("gamma", positive_support=True, param_names=("shape", "scale")),
    "lognormal": _FamilySpec("lognorm", positive_support=True, param_names=("shape", "scale")),
    "weibull": _FamilySpec("weibull_min", positive_support=True, param_names=("shape", "scale")),
    "exgaussian": _FamilySpec(
        "exponnorm", positive_support=False, param_names=("K", "mu", "sigma")
    ),
}


@dataclass(frozen=True)
class FitResult:
    """Result of fitting one parametric family by maximum likelihood.

    Attributes
    ----------
    family:
        Family name (one of :data:`FAMILIES`).
    params:
        Readable, named parameters (e.g. ``{"shape": ..., "scale": ...}``).
    loglik:
        Total log-likelihood ``sum(log pdf(x))`` of the data under the fit.
    aic:
        Akaike information criterion ``2k - 2L`` (lower is better).
    bic:
        Bayesian information criterion ``k ln(n) - 2L`` (lower is better).
    aicc:
        Small-sample corrected AIC, exposed as a property because it is fully
        determined by ``aic``, ``k``, and ``n``.
    ks_statistic:
        Kolmogorov-Smirnov statistic between the data and the fitted CDF.
    ks_pvalue:
        Two-sided p-value of the KS test (large = consistent with the fit).
    n:
        Number of finite samples used in the fit.
    scipy_params:
        Exact parameter tuple returned by ``scipy`` (shape args + ``loc`` +
        ``scale``), suitable for losslessly reconstructing the frozen dist.
    """

    family: str
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    n: int
    scipy_params: tuple[float, ...]

    @property
    def k(self) -> int:
        """Number of free (readable) parameters of the fitted family."""
        return len(self.params)

    @property
    def aicc(self) -> float:
        """Small-sample corrected AIC, or ``inf`` when undefined.

        The correction is ``AIC + 2k(k+1)/(n-k-1)``. It is finite only when the
        sample size exceeds ``k + 1``; callers can still report ordinary AIC
        for extremely small candidate sets.
        """
        denom = self.n - self.k - 1
        if denom <= 0:
            return float("inf")
        return float(self.aic + (2.0 * self.k * (self.k + 1)) / denom)


def _clean(data: FloatArray | object) -> FloatArray:
    """Coerce to ``float64``, flatten, and drop non-finite samples.

    Parameters
    ----------
    data:
        Sequence or array of samples.

    Returns
    -------
    FloatArray
        1-D array of the finite samples.

    Raises
    ------
    ValueError
        If fewer than two finite samples remain.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        msg = f"distribution fit needs >=2 finite samples; got {arr.size}"
        raise ValueError(msg)
    return cast(FloatArray, arr)


def _readable_params(spec: _FamilySpec, scipy_params: tuple[float, ...]) -> dict[str, float]:
    """Map a raw scipy parameter tuple to readable named parameters."""
    if spec.dist_name == "norm":
        loc, scale = scipy_params
        return {"mu": float(loc), "sigma": float(scale)}
    if spec.dist_name == "exponnorm":
        shape, loc, scale = scipy_params
        return {"K": float(shape), "mu": float(loc), "sigma": float(scale)}
    # Positive-support families fitted with floc=0: (shape, loc=0, scale).
    shape, _loc, scale = scipy_params
    return {"shape": float(shape), "scale": float(scale)}


def fit_distribution(
    data: FloatArray | object,
    family: str = "gamma",
    *,
    floc: float | None = None,
) -> FitResult:
    """Fit one parametric family to ``data`` by maximum likelihood.

    Parameters
    ----------
    data:
        Samples to fit; non-finite values are dropped first.
    family:
        One of :data:`FAMILIES`.
    floc:
        Optional fixed location. If ``None``, positive-support families default
        to ``floc=0`` (so only shape and scale are free); ``normal`` and
        ``exgaussian`` fit the location freely. Passing an explicit ``floc``
        overrides the default for any family.

    Returns
    -------
    FitResult
        The fitted family with log-likelihood, AIC/BIC, and a KS goodness-of-fit.

    Raises
    ------
    ValueError
        If ``family`` is unknown, if fewer than two finite samples remain, if a
        positive-support family receives non-positive data, or if the resulting
        log-likelihood is non-finite.
    """
    if family not in _REGISTRY:
        msg = f"unknown family {family!r}; expected one of {FAMILIES}"
        raise ValueError(msg)
    spec = _REGISTRY[family]
    x = _clean(data)

    if spec.positive_support and np.any(x <= 0):
        msg = f"family {family!r} has positive support; data contains non-positive values"
        raise ValueError(msg)

    dist = getattr(stats, spec.dist_name)

    effective_floc = floc
    if effective_floc is None and spec.positive_support:
        effective_floc = 0.0

    fit_kwargs: dict[str, float] = {}
    if effective_floc is not None:
        fit_kwargs["floc"] = effective_floc

    scipy_params = tuple(float(p) for p in dist.fit(x, **fit_kwargs))
    frozen = dist(*scipy_params)

    loglik = float(np.sum(frozen.logpdf(x)))
    if not np.isfinite(loglik):
        msg = f"family {family!r} produced a non-finite log-likelihood"
        raise ValueError(msg)

    params = _readable_params(spec, scipy_params)
    n = int(x.size)
    k = len(params)
    aic = 2.0 * k - 2.0 * loglik
    bic = k * float(np.log(n)) - 2.0 * loglik

    ks = stats.kstest(x, frozen.cdf)
    ks_statistic = float(ks.statistic)
    ks_pvalue = float(ks.pvalue)

    return FitResult(
        family=family,
        params=params,
        loglik=loglik,
        aic=aic,
        bic=bic,
        ks_statistic=ks_statistic,
        ks_pvalue=ks_pvalue,
        n=n,
        scipy_params=scipy_params,
    )


def frozen_from_result(result: FitResult) -> Any:
    """Reconstruct a frozen scipy distribution from a :class:`FitResult`.

    The returned object exposes the usual frozen-distribution methods
    (``pdf``, ``cdf``, ``ppf``, ...), built from the exact ``scipy_params`` of
    the fit so it matches the fitted curve byte-for-byte.

    Parameters
    ----------
    result:
        A result previously produced by :func:`fit_distribution`.

    Returns
    -------
    Any
        A frozen ``scipy.stats`` distribution (exposing ``pdf``/``cdf``/``ppf``).

    Raises
    ------
    ValueError
        If ``result.family`` is not a known family.
    """
    if result.family not in _REGISTRY:
        msg = f"unknown family {result.family!r}; expected one of {FAMILIES}"
        raise ValueError(msg)
    dist = getattr(stats, _REGISTRY[result.family].dist_name)
    return dist(*result.scipy_params)


def compare_distributions(
    data: FloatArray | object,
    families: tuple[str, ...] | list[str] | None = None,
    *,
    floc: float | None = None,
) -> list[FitResult]:
    """Fit several families and return their results sorted by AIC (ascending).

    Families that cannot be fitted for the given data (e.g. a positive-support
    family on non-positive samples, or a fit yielding a non-finite
    log-likelihood) are skipped rather than raising.

    Parameters
    ----------
    data:
        Samples to fit.
    families:
        Families to try; defaults to all of :data:`FAMILIES`.
    floc:
        Optional fixed location forwarded to :func:`fit_distribution`.

    Returns
    -------
    list[FitResult]
        Successful fits, best (lowest AIC) first. May be empty.
    """
    names = FAMILIES if families is None else tuple(families)
    results: list[FitResult] = []
    for name in names:
        try:
            results.append(fit_distribution(data, name, floc=floc))
        except ValueError:
            continue
    results.sort(key=lambda r: r.aic)
    return results


def information_weights(
    results: list[FitResult] | tuple[FitResult, ...],
    *,
    criterion: str = "aic",
) -> dict[str, float]:
    """Return normalized relative weights for fitted candidate models.

    Parameters
    ----------
    results:
        Candidate fits, typically from :func:`compare_distributions`.
    criterion:
        One of ``"aic"``, ``"aicc"``, or ``"bic"``. Weights are computed as
        ``exp(-0.5 * delta)`` and normalized over the finite criterion values.

    Returns
    -------
    dict[str, float]
        Family name to normalized weight. Fits with non-finite criterion values
        receive zero weight; an empty or all-nonfinite candidate set yields all
        zero weights.

    Raises
    ------
    ValueError
        If ``criterion`` is not supported.
    """
    if criterion not in {"aic", "aicc", "bic"}:
        msg = "criterion must be one of 'aic', 'aicc', or 'bic'"
        raise ValueError(msg)

    values = np.array([float(getattr(result, criterion)) for result in results], dtype=np.float64)
    weights = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(finite):
        finite_values = values[finite]
        delta = finite_values - float(np.min(finite_values))
        raw = np.exp(-0.5 * delta)
        denom = float(np.sum(raw))
        if denom > 0.0 and np.isfinite(denom):
            weights[finite] = raw / denom
    return {result.family: float(weight) for result, weight in zip(results, weights, strict=True)}


def best_fit(
    data: FloatArray | object,
    families: tuple[str, ...] | list[str] | None = None,
    *,
    floc: float | None = None,
) -> FitResult:
    """Return the single best-fitting family (lowest AIC).

    Parameters
    ----------
    data:
        Samples to fit.
    families:
        Families to try; defaults to all of :data:`FAMILIES`.
    floc:
        Optional fixed location forwarded to :func:`fit_distribution`.

    Returns
    -------
    FitResult
        The lowest-AIC fit.

    Raises
    ------
    ValueError
        If no family could be fitted to ``data``.
    """
    results = compare_distributions(data, families, floc=floc)
    if not results:
        msg = "no distribution family could be fitted to the supplied data"
        raise ValueError(msg)
    return results[0]
