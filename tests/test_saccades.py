"""Saccade / fixation / microsaccade detection tests (ISC-18..27)."""

from __future__ import annotations

import numpy as np
import pytest
from reference_impl import ivt_reference, microsaccade_reference  # tests/fixtures

from itrace import saccades
from itrace.saccades import _ek_threshold, _ek_velocity
from itrace.synthetic import gaze_fixation_noise, gaze_with_microsaccade, gaze_with_saccade


def test_ivt_recovers_three_events(saccade_stream_and_truth) -> None:  # ISC-18
    stream, _truth = saccade_stream_and_truth
    fixations, saccs = saccades.detect_ivt(stream, velocity_threshold_deg_s=30.0)
    assert len(saccs) == 1
    # fixation, saccade, fixation -> two fixations bracket the one saccade
    assert len(fixations) == 2
    assert fixations[0].offset_idx < saccs[0].onset_idx
    assert saccs[0].offset_idx < fixations[1].onset_idx


def test_ivt_recovers_amplitude(saccade_stream_and_truth) -> None:  # ISC-19
    stream, truth = saccade_stream_and_truth
    _f, saccs = saccades.detect_ivt(stream)
    assert saccs[0].amplitude_deg == pytest.approx(truth.amplitude_deg, rel=0.05)


def test_ivt_recovers_peak_velocity(saccade_stream_and_truth) -> None:  # ISC-20
    stream, truth = saccade_stream_and_truth
    _f, saccs = saccades.detect_ivt(stream)
    assert saccs[0].peak_velocity_deg_s == pytest.approx(truth.peak_velocity_deg_s, rel=0.10)


def test_idt_holds_one_fixation() -> None:  # ISC-21
    stream = gaze_fixation_noise(noise_deg=0.05, duration_s=0.5)
    fixations = saccades.detect_idt(stream, dispersion_threshold_deg=1.0, min_duration_s=0.1)
    assert len(fixations) == 1
    assert fixations[0].duration_s >= 0.1


def test_microsaccade_recovered(microsaccade_stream_and_onset) -> None:  # ISC-22
    stream, onset_t = microsaccade_stream_and_onset
    micro = saccades.detect_microsaccades(stream)
    assert len(micro) >= 1
    closest = min(micro, key=lambda m: abs(stream.t[m.onset_idx] - onset_t))
    assert abs(stream.t[closest.onset_idx] - onset_t) < 0.03
    assert closest.amplitude_deg == pytest.approx(0.5, rel=0.25)


def test_microsaccade_threshold_is_median_based() -> None:  # ISC-23
    # A velocity series with extreme outliers: plain std >> median-based sigma.
    v = np.concatenate([np.full(100, 1.0), np.array([1000.0, -1000.0])])
    eta = _ek_threshold(v, lam=1.0)
    median_sigma = np.sqrt(max(np.median(v**2) - np.median(v) ** 2, 0.0))
    plain_std = float(np.std(v))
    assert eta == pytest.approx(median_sigma, rel=1e-9)
    assert eta < 0.5 * plain_std  # robust estimator ignores the outliers


def test_ek_velocity_zero_on_constant() -> None:
    v = _ek_velocity(np.full(20, 5.0), dt=1.0 / 500.0)
    assert np.all(np.abs(v) < 1e-9)


def test_saccade_properties_and_direction() -> None:  # ISC-24/ISC-25
    right, _ = gaze_with_saccade(amplitude_deg=8.0, direction_deg=0.0)
    up, _ = gaze_with_saccade(amplitude_deg=8.0, direction_deg=90.0)
    _f, sr = saccades.detect_ivt(right)
    _f2, su = saccades.detect_ivt(up)
    assert sr[0].direction_deg == pytest.approx(0.0, abs=1.0)
    assert su[0].direction_deg == pytest.approx(90.0, abs=1.0)
    props = saccades.saccade_properties(sr)
    assert set(props) == {"amplitude_deg", "direction_deg", "duration_s", "peak_velocity_deg_s"}
    assert props["amplitude_deg"][0] == pytest.approx(8.0, rel=0.05)


def test_pure_noise_yields_no_saccades() -> None:  # ISC-26 (Anti)
    stream = gaze_fixation_noise(noise_deg=0.05, duration_s=2.0, seed=3)
    _f, saccs = saccades.detect_ivt(stream, velocity_threshold_deg_s=30.0)
    assert saccs == []


def test_agreement_with_independent_reference() -> None:  # ISC-27
    stream, _ = gaze_with_saccade(amplitude_deg=12.0, direction_deg=0.0, sampling_rate_hz=250.0)
    _f, saccs = saccades.detect_ivt(stream, velocity_threshold_deg_s=30.0)
    ref_events = ivt_reference(stream.x, stream.y, stream.t, velocity_threshold=30.0)
    ref_saccades = [e for e in ref_events if e["kind"] == 1.0]
    assert len(saccs) == len(ref_saccades) == 1
    # onset times agree within a few milliseconds despite different velocity math
    assert abs(saccs[0].onset_t - ref_saccades[0]["onset_t"]) < 0.02

    # microsaccade detectors agree on count
    micro_stream, _onset = gaze_with_microsaccade(seed=1)
    mine = saccades.detect_microsaccades(micro_stream)
    theirs = microsaccade_reference(micro_stream.x, micro_stream.y, micro_stream.sampling_rate_hz)
    assert len(mine) == len(theirs)


def test_empty_and_short_streams() -> None:
    from itrace.types import GazeStream

    empty = GazeStream(t=np.array([0.0]), x=np.array([0.0]), y=np.array([0.0]))
    assert saccades.detect_ivt(empty) == ([], [])
    assert saccades.detect_idt(empty) == []
    assert saccades.detect_microsaccades(empty) == []
