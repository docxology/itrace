"""Statistical-power / noise-sweep tests (ISC-79..86)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import power
from itrace.scene import closed_loop


def test_sweep_shape_and_ci_ordering() -> None:  # ISC-79/ISC-82
    res = power.run_noise_sweep([0.0, 0.004, 0.016], n_trials=5)
    assert res.n_trials == 5
    for m in ("gaze_rms_deg", "saccade_f1", "pupil_corr"):
        c = res.curve(m)
        assert len(c.mean) == 3
        for lo, mean, hi in zip(c.ci_low, c.mean, c.ci_high, strict=True):
            assert lo <= mean <= hi


def test_gaze_error_increases_with_noise() -> None:  # ISC-80 (falsifiable core)
    res = power.run_noise_sweep([0.0, 0.002, 0.008, 0.016], n_trials=6)
    means = res.curve("gaze_rms_deg").mean
    # strictly increasing: more webcam noise -> worse gaze recovery
    from itertools import pairwise

    assert all(b > a for a, b in pairwise(means))
    assert means[0] < 0.5  # near-perfect at zero noise


def test_saccades_are_most_fragile() -> None:  # ISC-81
    res = power.run_noise_sweep([0.0, 0.004, 0.016], n_trials=6)
    assert res.curve("saccade_f1").mean[0] > 0.95  # perfect at zero noise
    assert res.curve("saccade_f1").mean[-1] < 0.95  # degraded under heavy noise


def test_pupil_more_robust_than_saccades() -> None:  # ISC-81
    res = power.run_noise_sweep([0.0, 0.002, 0.004], n_trials=6)
    # at a moderate noise level pupil correlation degrades more slowly than F1
    sigma_idx = 2
    assert res.curve("pupil_corr").mean[sigma_idx] > res.curve("saccade_f1").mean[sigma_idx]


def test_recovery_threshold_gaze() -> None:  # ISC-83
    res = power.run_noise_sweep([0.0, 0.002, 0.004, 0.008, 0.016], n_trials=6)
    thr = power.recovery_threshold(res.curve("gaze_rms_deg"), 2.0)
    assert thr is not None
    assert 0.002 < thr < 0.016  # crosses 2 deg somewhere in the swept range


def test_recovery_threshold_none_when_never_crossed() -> None:  # ISC-83
    res = power.run_noise_sweep([0.0, 0.001], n_trials=4)
    # gaze RMS never reaches 100 deg, so no crossing
    assert power.recovery_threshold(res.curve("gaze_rms_deg"), 100.0) is None


def test_sweep_is_deterministic() -> None:  # ISC-84
    a = power.run_noise_sweep([0.0, 0.008], n_trials=5)
    b = power.run_noise_sweep([0.0, 0.008], n_trials=5)
    assert a.curve("gaze_rms_deg").mean == b.curve("gaze_rms_deg").mean


def test_mean_ci_helper() -> None:  # ISC-82
    mean, lo, hi, std = power._mean_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert mean == 3.0
    assert lo < 3.0 < hi
    assert std > 0
    # single value -> zero-width interval
    m2, lo2, hi2, s2 = power._mean_ci(np.array([7.0]))
    assert m2 == lo2 == hi2 == 7.0
    assert s2 == 0.0


def test_bounded_metric_ci_never_exceeds_bounds() -> None:  # ISC-99 (Advisor fix)
    # bootstrap percentile CI must respect F1 in [0,1] and r in [-1,1] near 1.0
    res = power.run_noise_sweep([0.0, 0.001, 0.004], n_trials=12)
    for metric, hi_bound in (("saccade_f1", 1.0), ("pupil_corr", 1.0)):
        c = res.curve(metric)
        assert max(c.ci_high) <= hi_bound + 1e-9
        assert min(c.ci_low) >= -1.0 - 1e-9


def test_ci_is_deterministic() -> None:  # ISC-99 (bootstrap is seeded)
    a = power.run_noise_sweep([0.0, 0.004], n_trials=10)
    b = power.run_noise_sweep([0.0, 0.004], n_trials=10)
    assert a.curve("saccade_f1").ci_high == b.curve("saccade_f1").ci_high


def test_bootstrap_ci_respects_bound_where_t_would_not() -> None:  # ISC-99 (discriminating)
    # near-1.0 bounded sample: a symmetric t-interval would push hi > 1.0;
    # the bootstrap percentile cannot exceed the in-bound observations.
    vals = np.array([0.90, 0.95, 1.0, 1.0, 1.0])
    mean, lo, hi, _std = power._mean_ci(vals)
    assert hi <= 1.0 + 1e-9  # would be ~1.02 under symmetric Student-t
    assert lo <= mean <= hi


def test_saccade_prf_extremes() -> None:  # ISC-85
    truth = [(10, 20), (40, 50)]
    assert power_scene_prf([(10, 20), (40, 50)], truth)[2] == 1.0  # perfect F1
    assert power_scene_prf([], truth) == (0.0, 0.0, 0.0)  # nothing detected
    assert power_scene_prf([(100, 110)], truth)[2] == 0.0  # no overlap


def test_landmark_noise_increases_error() -> None:  # ISC-86
    clean = closed_loop(landmark_noise_sd=0.0, seed=1).metrics["gaze_rms_deg"]
    noisy = closed_loop(landmark_noise_sd=0.01, seed=1).metrics["gaze_rms_deg"]
    assert noisy > clean


def test_summary_records_shape_and_pixels() -> None:  # ISC-94
    res = power.run_noise_sweep([0.0, 0.004, 0.016], n_trials=5)
    rows = power.summary_records(res, image_width_px=640.0)
    assert len(rows) == 3
    for r in rows:
        assert r["noise_px"] == pytest.approx(r["noise_sigma"] * 640.0)
        for m in ("gaze_rms_deg", "saccade_f1", "pupil_corr"):
            assert f"{m}_mean" in r
            assert r[f"{m}_ci_low"] <= r[f"{m}_mean"] <= r[f"{m}_ci_high"]
        assert r["n_trials"] == 5


def test_format_summary_markdown_is_valid_table() -> None:  # ISC-94
    res = power.run_noise_sweep([0.0, 0.008], n_trials=5)
    md = power.format_summary_markdown(res)
    lines = md.splitlines()
    assert lines[0].startswith("| sigma (norm) |")
    assert set(lines[1]) <= set("|-")  # separator row
    assert "+/-" in md  # every cell carries uncertainty
    assert "n=5" in md  # n disclosed
    # one data row per noise level + header + separator (+ caption block)
    data_rows = [ln for ln in lines if ln.startswith("| 0.")]
    assert len(data_rows) == 2


def test_sigma_to_pixels() -> None:  # ISC-93
    assert power.sigma_to_pixels(0.0014, 640) == pytest.approx(0.896, abs=1e-3)
    assert power.sigma_to_pixels(0.005, 640) == pytest.approx(3.2, abs=1e-3)
    # saccade-breakdown sigma is below the gaze-breakdown sigma in pixels too
    assert power.sigma_to_pixels(0.0014) < power.sigma_to_pixels(0.0048)


# helper imported from scene (kept here to test the scoring directly)
from itrace.scene import _saccade_prf as power_scene_prf  # noqa: E402
