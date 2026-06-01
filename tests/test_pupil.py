"""Pupillometry preprocessing tests (ISC-31..36)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import pupil
from itrace.synthetic import pupil_sine_with_blink
from itrace.types import PupilStream, PupilUnit


def test_detect_blink_span() -> None:  # ISC-31
    stream, _peaks = pupil_sine_with_blink(blink_window_s=(4.0, 4.3))
    blinks = pupil.detect_blinks(stream)
    assert len(blinks) == 1
    onset, offset = blinks[0]
    assert stream.t[onset] == pytest.approx(4.0, abs=0.05)
    assert stream.t[offset] == pytest.approx(4.3, abs=0.05)


def test_interpolate_removes_nans() -> None:  # ISC-32
    stream, _peaks = pupil_sine_with_blink(blink_window_s=(4.0, 4.3))
    clean = pupil.interpolate_blinks(stream)
    assert np.all(np.isfinite(clean.size))
    assert len(clean) == len(stream)


def test_interpolate_all_invalid_raises() -> None:  # ISC-36 (Anti)
    t = np.linspace(0.0, 1.0, 60)
    bad = PupilStream(t=t, size=np.full(60, np.nan), unit=PupilUnit.MM)
    with pytest.raises(ValueError, match="unusable"):
        pupil.interpolate_blinks(bad)


def test_blink_threshold_marks_low_confidence_pupil_samples() -> None:
    t = np.arange(7, dtype=np.float64) * 0.01
    stream = PupilStream(
        t=t,
        size=np.array([3.0, 3.1, 0.12, 3.1, 3.0, 0.18, 3.0], dtype=np.float64),
        unit=PupilUnit.MM,
    )

    assert pupil.detect_blinks(stream) == []
    assert pupil.detect_blinks(stream, min_valid=0.2) == [(2, 2), (5, 5)]
    clean = pupil.interpolate_blinks(stream, min_valid=0.2, pad_samples=0)
    assert clean.size[2] == pytest.approx(3.1)
    assert clean.size[5] == pytest.approx(3.0)

    summary = pupil.quality_summary(stream, min_valid=0.2)
    assert summary["n_blinks"] == 2.0
    assert summary["valid_sample_fraction"] == pytest.approx(5.0 / 7.0)


def test_baseline_correct_centers_window() -> None:  # ISC-33
    t = np.linspace(0.0, 10.0, 600)
    size = 3.0 + 0.5 * np.sin(2 * np.pi * t / 2.0)
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)
    corrected = pupil.baseline_correct(stream, baseline_window_s=(0.0, 2.0))
    mask = (t >= 0.0) & (t <= 2.0)
    assert float(np.mean(corrected.size[mask])) == pytest.approx(0.0, abs=1e-9)


def test_baseline_divisive_and_errors() -> None:
    t = np.linspace(0.0, 4.0, 240)
    stream = PupilStream(t=t, size=np.full(240, 4.0), unit=PupilUnit.MM)
    div = pupil.baseline_correct(stream, (0.0, 1.0), mode="divisive")
    assert np.allclose(div.size, 1.0)
    with pytest.raises(ValueError, match="end must exceed"):
        pupil.baseline_correct(stream, (1.0, 1.0))
    with pytest.raises(ValueError, match="unknown baseline mode"):
        pupil.baseline_correct(stream, (0.0, 1.0), mode="bogus")
    with pytest.raises(ValueError, match="no samples"):
        pupil.baseline_correct(stream, (100.0, 200.0))


def test_smooth_reduces_noise_variance() -> None:  # ISC-34
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.0, 600)
    noise = rng.normal(0.0, 0.3, 600)
    stream = PupilStream(t=t, size=3.0 + noise, unit=PupilUnit.MM)
    smoothed = pupil.smooth(stream, cutoff_hz=2.0)
    assert float(np.var(smoothed.size)) < 0.5 * float(np.var(stream.size))


def test_smooth_requires_gapfree() -> None:
    t = np.linspace(0.0, 1.0, 60)
    s = PupilStream(t=t, size=np.concatenate([np.full(59, 3.0), [np.nan]]), unit=PupilUnit.MM)
    with pytest.raises(ValueError, match="gap-free"):
        pupil.smooth(s)


def test_mad_flags_spikes_not_clean() -> None:  # ISC-35
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 10.0, 600)
    size = 3.0 + rng.normal(0.0, 0.05, 600)
    size[100] = 50.0  # injected spike
    size[400] = -20.0  # injected negative spike
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)
    flags = pupil.mad_reject(stream, n_mad=5.0)
    assert flags[100]
    assert flags[400]
    assert flags.sum() <= 5  # clean samples not flagged


def test_mad_all_nan() -> None:
    t = np.linspace(0.0, 1.0, 10)
    stream = PupilStream(t=t, size=np.full(10, np.nan))
    flags = pupil.mad_reject(stream)
    assert flags.all()


def test_quality_summary_counts_blinks_invalid_samples_and_velocity() -> None:
    t = np.linspace(0.0, 2.0, 201)
    size = 3.0 + 0.2 * np.sin(2 * np.pi * t)
    size[30:40] = np.nan
    size[100] = 0.0
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)

    summary = pupil.quality_summary(stream)

    assert summary["n_samples"] == 201.0
    assert summary["valid_sample_fraction"] < 1.0
    assert summary["blink_fraction"] > 0.0
    assert summary["n_blinks"] == 2.0
    assert summary["median_dt_s"] == pytest.approx(0.01)
    assert summary["peak_dilation_velocity"] > 0.0
    assert summary["peak_constriction_velocity"] < 0.0


def test_response_features_report_latency_auc_and_phase_fractions() -> None:
    t = np.linspace(0.0, 4.0, 401)
    size = np.full_like(t, 3.0)
    response = (t >= 1.0) & (t <= 3.0)
    size[response] += np.sin((t[response] - 1.0) / 2.0 * np.pi)
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)

    features = pupil.response_features(
        stream,
        baseline_window_s=(0.0, 0.9),
        response_window_s=(1.0, 3.0),
    )

    assert features["baseline_size"] == pytest.approx(3.0)
    assert features["latency_to_peak_s"] == pytest.approx(1.0, abs=0.05)
    assert features["dilation_auc"] > 1.0
    assert features["baseline_relative_peak_change"] > 0.3
    assert features["phase_fraction_peak"] > 0.0
    assert features["mean_dilation_velocity"] > 0.0
