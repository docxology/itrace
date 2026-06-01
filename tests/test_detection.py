"""Advanced oculomotor detection tests (itrace.detection).

No mocks: every test runs the real detector over real synthetic arrays whose
ground truth is known by construction. The adaptive-threshold tests use *noisy*
streams on purpose -- the Nystrom & Holmqvist (2010) scheme estimates a velocity
threshold from the intersaccadic noise floor, so a strictly zero-noise signal
has no floor to find and is a degenerate input rather than a useful test case.
"""

from __future__ import annotations

import numpy as np
import pytest

from itrace import detection, saccades
from itrace.synthetic import gaze_with_saccade
from itrace.types import GazeStream


def _multi_saccade_stream(
    *,
    n: int = 4,
    noise_deg: float = 0.03,
    fixation_s: float = 0.2,
    sampling_rate_hz: float = 250.0,
    seed: int = 2,
) -> GazeStream:
    """Concatenate ``n`` saccades of growing amplitude into one noisy stream."""
    rng = np.random.default_rng(0)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(8.0, 20.0, n):
        g, _ = gaze_with_saccade(
            sampling_rate_hz=sampling_rate_hz,
            amplitude_deg=float(amp),
            direction_deg=float(rng.uniform(-180.0, 180.0)),
            fixation_s=fixation_s,
            noise_deg=noise_deg,
            seed=seed,
        )
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / sampling_rate_hz
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def _stream_with_brief_saccade_gap() -> GazeStream:
    """Two fast ramp segments separated by a brief low-velocity plateau."""
    dt = 0.01
    x: list[float] = [0.0] * 20
    cur = 0.0
    for _ in range(10):
        cur += 1.0
        x.append(cur)
    x.extend([cur] * 4)
    for _ in range(10):
        cur += 1.0
        x.append(cur)
    x.extend([cur] * 20)
    arr = np.array(x, dtype=np.float64)
    return GazeStream(t=np.arange(arr.size) * dt, x=arr, y=np.zeros_like(arr))


# --------------------------------------------------------------------------- #
# adaptive_ivt_threshold / detect_ivt_adaptive
# --------------------------------------------------------------------------- #
def test_adaptive_threshold_is_positive_and_below_peaks() -> None:
    stream = _multi_saccade_stream()
    thr = detection.adaptive_ivt_threshold(stream)
    _vx, _vy, speed = saccades.velocities(stream)
    assert thr > 0.0
    # The threshold must sit between the noise floor and the saccadic peaks.
    assert thr < float(np.max(speed))
    assert thr > float(np.median(speed))


def test_adaptive_threshold_scales_with_noise() -> None:
    low = detection.adaptive_ivt_threshold(_multi_saccade_stream(noise_deg=0.01))
    high = detection.adaptive_ivt_threshold(_multi_saccade_stream(noise_deg=0.05))
    # A noisier recording yields a higher adaptive threshold.
    assert high > low > 0.0


def test_adaptive_recovers_same_saccade_count_as_fixed() -> None:
    stream = _multi_saccade_stream()
    _f_a, sacc_adaptive = detection.detect_ivt_adaptive(stream)
    _f_f, sacc_fixed = saccades.detect_ivt(stream, velocity_threshold_deg_s=30.0)
    assert len(sacc_adaptive) >= 1
    # Count agrees with the fixed-threshold detector within a small tolerance.
    assert abs(len(sacc_adaptive) - len(sacc_fixed)) <= 1


def test_ivt_merge_gap_closes_short_low_velocity_hole_in_one_saccade() -> None:
    stream = _stream_with_brief_saccade_gap()

    _f_plain, split = saccades.detect_ivt(
        stream,
        velocity_threshold_deg_s=30.0,
        min_saccade_duration_s=0.0,
        merge_gap_s=0.0,
    )
    _f_merged, merged = saccades.detect_ivt(
        stream,
        velocity_threshold_deg_s=30.0,
        min_saccade_duration_s=0.0,
        merge_gap_s=0.04,
    )

    assert len(split) > len(merged)
    assert len(merged) == 1
    assert merged[0].amplitude_deg == pytest.approx(20.0, rel=0.08)


def test_ivt_rejects_negative_merge_gap() -> None:
    stream = _stream_with_brief_saccade_gap()
    with pytest.raises(ValueError, match="merge_gap_s"):
        saccades.detect_ivt(stream, merge_gap_s=-0.01)


def test_detect_smooth_pursuit_flags_sustained_moderate_motion() -> None:
    t = np.arange(150, dtype=np.float64) / 60.0
    stream = GazeStream(t=t, x=8.0 * t, y=np.zeros_like(t))

    pursuits = detection.detect_smooth_pursuit(
        stream,
        min_velocity_deg_s=4.0,
        max_velocity_deg_s=20.0,
        min_duration_s=0.5,
    )

    assert len(pursuits) == 1
    assert pursuits[0].duration_s >= 0.5
    assert pursuits[0].mean_velocity_deg_s == pytest.approx(8.0, rel=0.15)


def test_ivt_edge_and_duration_filters_are_opt_in() -> None:
    stream = GazeStream(
        t=np.arange(12, dtype=np.float64) * 0.01,
        x=np.array([0, 2, 4, 6, 8, 10, 12, 12, 12, 12, 12, 12], dtype=np.float64),
        y=np.zeros(12, dtype=np.float64),
    )

    _fix, plain = saccades.detect_ivt(
        stream,
        velocity_threshold_deg_s=30.0,
        min_saccade_duration_s=0.0,
    )
    _fix, filtered = saccades.detect_ivt(
        stream,
        velocity_threshold_deg_s=30.0,
        min_saccade_duration_s=0.0,
        reject_edge_events=True,
        max_saccade_duration_s=0.02,
    )

    assert len(plain) == 1
    assert filtered == []


def test_adaptive_lambda_factor_monotone() -> None:
    stream = _multi_saccade_stream()
    low_lambda = detection.adaptive_ivt_threshold(stream, lambda_factor=3.0)
    high_lambda = detection.adaptive_ivt_threshold(stream, lambda_factor=9.0)
    # More standard deviations above the floor -> a higher threshold.
    assert high_lambda > low_lambda


def test_adaptive_threshold_short_stream_is_zero() -> None:
    short = GazeStream(t=np.array([0.0, 0.004]), x=np.array([0.0, 0.0]), y=np.array([0.0, 0.0]))
    assert detection.adaptive_ivt_threshold(short) == 0.0


def test_adaptive_threshold_constant_stream_returns_max_speed() -> None:
    # Constant position -> zero speed everywhere -> no separable noise floor.
    const = GazeStream(
        t=np.arange(50) / 250.0,
        x=np.full(50, 3.0),
        y=np.full(50, 3.0),
    )
    thr = detection.adaptive_ivt_threshold(const)
    assert thr == pytest.approx(0.0, abs=1e-6)
    # Downstream adaptive I-VT must therefore flag no saccades.
    _f, sacc = detection.detect_ivt_adaptive(const)
    assert sacc == []


def test_adaptive_threshold_near_flat_trace_is_near_zero() -> None:
    # A single tiny isolated bump on an otherwise-flat trace has essentially no
    # noise floor: the iteration relaxes onto a near-zero threshold.
    x = np.zeros(40)
    x[20] = 0.001
    stream = GazeStream(t=np.arange(40) / 250.0, x=x, y=np.zeros(40))
    thr = detection.adaptive_ivt_threshold(stream)
    assert thr == pytest.approx(0.0, abs=1e-3)


def test_adaptive_threshold_max_iter_caps_iteration() -> None:
    # With max_iter=1 the loop runs once; the value still lies below the peaks.
    stream = _multi_saccade_stream()
    one_step = detection.adaptive_ivt_threshold(stream, max_iter=1)
    _vx, _vy, speed = saccades.velocities(stream)
    assert 0.0 < one_step < float(np.max(speed))


# --------------------------------------------------------------------------- #
# detect_pso
# --------------------------------------------------------------------------- #
def _stream_with_post_saccadic_oscillation() -> tuple[GazeStream, list]:
    """A clean saccade with a small damped ring injected after its offset."""
    g, _truth = gaze_with_saccade(
        sampling_rate_hz=500.0,
        amplitude_deg=12.0,
        direction_deg=0.0,
        fixation_s=0.2,
        noise_deg=0.0,
        seed=0,
    )
    _f, sacc = saccades.detect_ivt(g, velocity_threshold_deg_s=30.0)
    offset = sacc[0].offset_idx
    x = g.x.copy()
    k = np.arange(20)
    ring = 1.0 * np.exp(-k / 6.0) * np.sin(2.0 * np.pi * k / 8.0)
    x[offset + 1 : offset + 1 + 20] += ring
    ringed = GazeStream(t=g.t, x=x, y=g.y)
    _f2, sacc2 = saccades.detect_ivt(ringed, velocity_threshold_deg_s=30.0)
    return ringed, sacc2


def test_detect_pso_flags_injected_oscillation() -> None:
    stream, sacc = _stream_with_post_saccadic_oscillation()
    pso = detection.detect_pso(stream, sacc, window_s=0.05)
    assert len(pso) >= 1
    rec = pso[0]
    assert set(rec) == {"onset_t", "offset_t", "peak_velocity_deg_s"}
    assert rec["offset_t"] > rec["onset_t"]
    assert rec["peak_velocity_deg_s"] > 0.0


def test_detect_pso_none_on_clean_minimum_jerk() -> None:
    # A pure minimum-jerk saccade decelerates monotonically: no secondary peak.
    g, _ = gaze_with_saccade(
        sampling_rate_hz=500.0, amplitude_deg=12.0, direction_deg=0.0, fixation_s=0.2, seed=0
    )
    _f, sacc = saccades.detect_ivt(g, velocity_threshold_deg_s=30.0)
    assert detection.detect_pso(g, sacc, window_s=0.04) == []


def test_detect_pso_empty_inputs() -> None:
    stream = _multi_saccade_stream()
    assert detection.detect_pso(stream, []) == []
    short = GazeStream(t=np.array([0.0, 0.004]), x=np.array([0.0, 0.0]), y=np.array([0.0, 0.0]))
    _f, sacc = saccades.detect_ivt(_multi_saccade_stream())
    assert detection.detect_pso(short, sacc) == []


def test_detect_pso_tiny_window_has_no_samples() -> None:
    # A window shorter than the sample interval admits no post-offset samples,
    # so the search window is empty for every saccade (exercises end<=start).
    stream, sacc = _stream_with_post_saccadic_oscillation()
    assert detection.detect_pso(stream, sacc, window_s=1e-6) == []


def test_detect_pso_high_peak_fraction_suppresses_all() -> None:
    stream, sacc = _stream_with_post_saccadic_oscillation()
    # Require the secondary peak to exceed the saccade peak itself: impossible.
    assert detection.detect_pso(stream, sacc, window_s=0.05, peak_fraction=2.0) == []


def test_detect_pso_skips_saccade_at_stream_end() -> None:
    # A saccade whose offset is the final sample leaves no post-window to scan.
    g, _ = gaze_with_saccade(
        sampling_rate_hz=500.0, amplitude_deg=12.0, direction_deg=0.0, fixation_s=0.05, seed=0
    )
    _f, sacc = saccades.detect_ivt(g, velocity_threshold_deg_s=30.0)
    # Forge a saccade pinned to the last index.
    last = len(g) - 1
    end_saccade = saccades.Saccade(
        onset_idx=last - 1,
        offset_idx=last,
        onset_t=float(g.t[last - 1]),
        offset_t=float(g.t[last]),
        amplitude_deg=sacc[0].amplitude_deg,
        direction_deg=sacc[0].direction_deg,
        peak_velocity_deg_s=sacc[0].peak_velocity_deg_s,
    )
    assert detection.detect_pso(g, [end_saccade], window_s=0.05) == []


# --------------------------------------------------------------------------- #
# intersaccadic_intervals
# --------------------------------------------------------------------------- #
def test_intersaccadic_intervals_length_and_sign() -> None:
    stream = _multi_saccade_stream()
    _f, sacc = detection.detect_ivt_adaptive(stream)
    isi = detection.intersaccadic_intervals(sacc)
    assert isi.shape == (len(sacc) - 1,)
    assert isi.dtype == np.float64
    assert np.all(isi > 0.0)


def test_intersaccadic_intervals_values() -> None:
    a = saccades.Saccade(0, 1, 0.0, 0.05, 5.0, 0.0, 200.0)
    b = saccades.Saccade(2, 3, 0.20, 0.25, 5.0, 0.0, 200.0)
    c = saccades.Saccade(4, 5, 0.50, 0.55, 5.0, 0.0, 200.0)
    isi = detection.intersaccadic_intervals([a, b, c])
    assert isi.tolist() == pytest.approx([0.15, 0.25])


def test_intersaccadic_intervals_too_few_saccades() -> None:
    assert detection.intersaccadic_intervals([]).shape == (0,)
    single = saccades.Saccade(0, 1, 0.0, 0.05, 5.0, 0.0, 200.0)
    assert detection.intersaccadic_intervals([single]).shape == (0,)


# --------------------------------------------------------------------------- #
# saccade_peak_accelerations
# --------------------------------------------------------------------------- #
def test_peak_accelerations_finite_positive_one_per_saccade() -> None:
    stream = _multi_saccade_stream()
    _f, sacc = detection.detect_ivt_adaptive(stream)
    acc = detection.saccade_peak_accelerations(stream, sacc)
    assert acc.shape == (len(sacc),)
    assert acc.dtype == np.float64
    assert np.all(np.isfinite(acc))
    assert np.all(acc > 0.0)


def test_peak_accelerations_empty_inputs() -> None:
    stream = _multi_saccade_stream()
    assert detection.saccade_peak_accelerations(stream, []).shape == (0,)
    short = GazeStream(t=np.array([0.0]), x=np.array([0.0]), y=np.array([0.0]))
    single = saccades.Saccade(0, 0, 0.0, 0.0, 5.0, 0.0, 200.0)
    assert detection.saccade_peak_accelerations(short, [single]).shape == (0,)


def test_peak_acceleration_larger_for_faster_saccade() -> None:
    # A bigger-amplitude saccade reaches a higher peak speed faster -> larger
    # peak acceleration than a small one of equal duration.
    big, _ = gaze_with_saccade(amplitude_deg=18.0, direction_deg=0.0, seed=0)
    small, _ = gaze_with_saccade(amplitude_deg=4.0, direction_deg=0.0, seed=0)
    _fb, sb = saccades.detect_ivt(big, velocity_threshold_deg_s=30.0)
    _fs, ss = saccades.detect_ivt(small, velocity_threshold_deg_s=30.0)
    acc_big = detection.saccade_peak_accelerations(big, sb)
    acc_small = detection.saccade_peak_accelerations(small, ss)
    assert float(acc_big[0]) > float(acc_small[0])
