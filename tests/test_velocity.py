"""Velocity-estimation tests (ISC-15..17)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import velocity


def test_savgol_recovers_linear_ramp_velocity() -> None:  # ISC-15
    sr = 250.0
    n = 250
    slope = 4.0  # deg/s
    t = np.arange(n) / sr
    pos = slope * t
    v = velocity.pos2vel_savgol(pos, sr)
    interior = v[10:-10]
    assert np.allclose(interior, slope, rtol=0.01)


def test_savgol_constant_is_zero() -> None:  # ISC-16
    v = velocity.pos2vel_savgol(np.full(200, 3.14), 250.0)
    assert np.all(np.abs(v) < 1e-6)


def test_savgol_short_signal_returns_zeros() -> None:
    assert np.all(velocity.pos2vel_savgol(np.array([1.0, 2.0]), 250.0) == 0)


def test_savgol_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="positive"):
        velocity.pos2vel_savgol(np.zeros(10), 0.0)
    with pytest.raises(ValueError, match="1-D"):
        velocity.pos2vel_savgol(np.zeros((4, 4)), 250.0)


def test_gradient_variable_dt() -> None:  # ISC-17
    t = np.array([0.0, 0.1, 0.25, 0.45, 0.7])  # non-uniform
    pos = 2.0 * t
    v = velocity.pos2vel_gradient(pos, t)
    assert np.allclose(v, 2.0, rtol=1e-6)


def test_gradient_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        velocity.pos2vel_gradient(np.zeros(3), np.zeros(4))


def test_gradient_single_sample() -> None:
    assert np.all(velocity.pos2vel_gradient(np.array([1.0]), np.array([0.0])) == 0)


def test_speed_2d() -> None:
    s = velocity.speed_2d(np.array([3.0]), np.array([4.0]))
    assert s[0] == pytest.approx(5.0)
