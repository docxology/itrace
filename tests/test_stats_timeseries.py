"""Tests for itrace.stats.timeseries (no mocks; real data + ground truth)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace.stats import timeseries as ts
from itrace.synthetic import pupil_sine_with_blink
from itrace.types import Fixation, Microsaccade, PupilStream, PupilUnit, Saccade


def _fix(onset_t: float) -> Fixation:
    return Fixation(0, 1, onset_t, onset_t + 0.1, 0.0, 0.0)


def _sacc(onset_t: float) -> Saccade:
    return Saccade(0, 1, onset_t, onset_t + 0.02, 10.0, 0.0, 300.0)


def _micro() -> Microsaccade:
    return Microsaccade(0, 1, 0.5, 50.0, 0.0)


# --- event_rate ------------------------------------------------------------


def test_event_rate_uniform_onsets_flat_rate() -> None:
    # One onset in the middle of each 1 s bin over 10 s -> flat 1 Hz.
    onsets = [b + 0.5 for b in range(10)]
    centers, rate = ts.event_rate(onsets, 10.0, bin_s=1.0)
    assert centers.shape == (10,)
    assert np.allclose(centers, np.arange(10) + 0.5)
    assert np.allclose(rate, 1.0)


def test_event_rate_known_counts_per_bin() -> None:
    # bin_s=2: 3 onsets in [0,2), 1 in [2,4) -> rate 1.5 Hz, 0.5 Hz.
    onsets = [0.1, 0.5, 1.9, 3.0]
    centers, rate = ts.event_rate(onsets, 4.0, bin_s=2.0)
    assert np.allclose(centers, [1.0, 3.0])
    assert np.allclose(rate, [1.5, 0.5])


def test_event_rate_partial_final_bin() -> None:
    # duration 2.5 with bin 1.0 -> 3 bins (0-1, 1-2, 2-3 remainder).
    centers, rate = ts.event_rate([2.2], 2.5, bin_s=1.0)
    assert centers.shape == (3,)
    assert np.allclose(rate, [0.0, 0.0, 1.0])


def test_event_rate_empty_is_zero() -> None:
    centers, rate = ts.event_rate([], 5.0, bin_s=1.0)
    assert centers.shape == (5,)
    assert np.allclose(rate, 0.0)


def test_event_rate_ignores_out_of_range() -> None:
    centers, rate = ts.event_rate([-1.0, 0.5, 9.0], 2.0, bin_s=1.0)
    assert np.allclose(rate, [1.0, 0.0])
    assert centers.shape == (2,)


def test_event_rate_bad_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        ts.event_rate([0.5], 0.0, bin_s=1.0)


def test_event_rate_bad_bin_raises() -> None:
    with pytest.raises(ValueError, match="bin_s"):
        ts.event_rate([0.5], 5.0, bin_s=0.0)


# --- fixation/saccade rate series ------------------------------------------


def test_fixation_rate_series_matches_event_rate() -> None:
    fixations = [_fix(0.5), _fix(1.5), _fix(1.6)]
    centers, rate = ts.fixation_rate_series(fixations, 2.0, bin_s=1.0)
    assert np.allclose(rate, [1.0, 2.0])
    assert centers.shape == (2,)


def test_fixation_rate_series_empty() -> None:
    centers, rate = ts.fixation_rate_series([], 3.0, bin_s=1.0)
    assert centers.shape == (3,)
    assert np.allclose(rate, 0.0)


def test_saccade_rate_series_matches_event_rate() -> None:
    saccades = [_sacc(0.2), _sacc(2.1)]
    centers, rate = ts.saccade_rate_series(saccades, 4.0, bin_s=2.0)
    assert np.allclose(rate, [0.5, 0.5])
    assert centers.shape == (2,)


def test_saccade_rate_series_empty() -> None:
    _centers, rate = ts.saccade_rate_series([], 2.0, bin_s=1.0)
    assert np.allclose(rate, 0.0)


# --- blink_rate_hz ---------------------------------------------------------


def test_blink_rate_one_blink_over_span() -> None:
    # 10 s recording, single blink window -> ~1 blink / span (~9.98 s span).
    stream, _peaks = pupil_sine_with_blink(duration_s=10.0)
    span = float(stream.t[-1] - stream.t[0])
    rate = ts.blink_rate_hz(stream)
    assert rate == pytest.approx(1.0 / span, rel=1e-9)
    assert 0.09 < rate < 0.11


def test_blink_rate_no_blink_is_zero() -> None:
    stream, _peaks = pupil_sine_with_blink(duration_s=5.0, blink_window_s=None)
    assert ts.blink_rate_hz(stream) == 0.0


def test_blink_rate_too_short_is_zero() -> None:
    stream = PupilStream(t=np.array([0.0]), size=np.array([3.0]), unit=PupilUnit.MM)
    assert ts.blink_rate_hz(stream) == 0.0


def test_blink_rate_zero_span_is_zero() -> None:
    stream = PupilStream(
        t=np.array([1.0, 1.0]),
        size=np.array([np.nan, np.nan]),
        unit=PupilUnit.MM,
    )
    assert ts.blink_rate_hz(stream) == 0.0


# --- microsaccade_rate_hz --------------------------------------------------


def test_microsaccade_rate_known_count() -> None:
    micros = [_micro() for _ in range(6)]
    assert ts.microsaccade_rate_hz(micros, 3.0) == pytest.approx(2.0)


def test_microsaccade_rate_empty() -> None:
    assert ts.microsaccade_rate_hz([], 5.0) == 0.0


def test_microsaccade_rate_bad_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        ts.microsaccade_rate_hz([_micro()], 0.0)


# --- sliding_window_stat ---------------------------------------------------


def test_sliding_window_mean_on_ramp() -> None:
    # values == times (ramp); window 2 s, step 2 s, non-overlapping.
    times = np.arange(0.0, 6.0, 1.0)  # 0..5
    values = times.copy()
    centers, means = ts.sliding_window_stat(values, times, window_s=2.0, step_s=2.0, stat="mean")
    # windows start at 0,2,4 -> [0,1], [2,3], [4,5]; means 0.5, 2.5, 4.5.
    assert np.allclose(centers, [1.0, 3.0, 5.0])
    assert np.allclose(means, [0.5, 2.5, 4.5])


def test_sliding_window_count_on_ramp() -> None:
    times = np.arange(0.0, 6.0, 1.0)
    values = times.copy()
    _centers, counts = ts.sliding_window_stat(values, times, window_s=2.0, step_s=2.0, stat="count")
    assert np.allclose(counts, [2.0, 2.0, 2.0])


def test_sliding_window_std_constant_is_zero() -> None:
    times = np.arange(0.0, 4.0, 1.0)
    values = np.full_like(times, 7.0)
    _centers, stds = ts.sliding_window_stat(values, times, window_s=2.0, step_s=2.0, stat="std")
    assert np.allclose(stds, 0.0)


def test_sliding_window_empty_window_yields_zero() -> None:
    # Gap in time axis leaves a window with no samples.
    times = np.array([0.0, 0.1, 5.0, 5.1])
    values = np.array([1.0, 1.0, 2.0, 2.0])
    centers, means = ts.sliding_window_stat(values, times, window_s=1.0, step_s=1.0, stat="mean")
    # window starting at t=1,2,3,4 are empty -> 0.0.
    assert means[0] == pytest.approx(1.0)
    assert np.any(means == 0.0)
    assert centers.shape == means.shape


def test_sliding_window_empty_input() -> None:
    centers, out = ts.sliding_window_stat([], [], window_s=1.0, step_s=1.0, stat="mean")
    assert centers.shape == (0,)
    assert out.shape == (0,)


def test_sliding_window_bad_window_raises() -> None:
    with pytest.raises(ValueError, match="window_s"):
        ts.sliding_window_stat([1.0], [0.0], window_s=0.0, step_s=1.0)


def test_sliding_window_bad_step_raises() -> None:
    with pytest.raises(ValueError, match="step_s"):
        ts.sliding_window_stat([1.0], [0.0], window_s=1.0, step_s=0.0)


def test_sliding_window_unknown_stat_raises() -> None:
    with pytest.raises(ValueError, match="stat"):
        ts.sliding_window_stat([1.0], [0.0], window_s=1.0, step_s=1.0, stat="median")


def test_sliding_window_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ts.sliding_window_stat([1.0, 2.0], [0.0], window_s=1.0, step_s=1.0)
