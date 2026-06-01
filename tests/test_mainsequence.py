"""Main-sequence fitting tests (ISC-28..30)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import mainsequence


def test_recovers_saturating_parameters() -> None:  # ISC-28
    rng = np.random.default_rng(0)
    amp = np.linspace(0.5, 25.0, 60)
    v_max_true, c_true = 600.0, 8.0
    vel = v_max_true * (1.0 - np.exp(-amp / c_true))
    vel = vel + rng.normal(0.0, 3.0, amp.size)
    fit = mainsequence.fit(amp, vel)
    assert fit["v_max"] == pytest.approx(v_max_true, rel=0.10)
    assert fit["C"] == pytest.approx(c_true, rel=0.10)


def test_power_law_exponent_in_range() -> None:  # ISC-29
    rng = np.random.default_rng(1)
    amp = np.linspace(0.5, 25.0, 60)
    vel = 30.0 * amp**0.6  # canonical-ish power law
    vel = vel * (1.0 + rng.normal(0.0, 0.02, amp.size))
    fit = mainsequence.fit(amp, vel)
    assert 0.4 <= fit["power_b"] <= 0.9
    assert fit["r_squared_power"] > 0.95


def test_too_few_points_raises() -> None:  # ISC-30
    with pytest.raises(ValueError, match=">=3"):
        mainsequence.fit(np.array([1.0, 2.0]), np.array([10.0, 20.0]))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        mainsequence.fit(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]))


def test_filters_nonpositive_points() -> None:
    amp = np.array([1.0, 2.0, np.nan, 5.0, 10.0, -3.0, 15.0])
    vel = np.array([30.0, 45.0, 50.0, 70.0, 95.0, 10.0, 120.0])
    fit = mainsequence.fit(amp, vel)
    assert np.isfinite(fit["power_b"])
