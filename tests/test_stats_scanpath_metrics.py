"""Tests for itrace.stats.scanpath_metrics (no mocks; real data + ground truth)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import saccades
from itrace.stats import scanpath_metrics as sm
from itrace.synthetic import gaze_with_saccade
from itrace.types import Fixation, GazeStream, Saccade, SessionReport


def _fix(cx: float, cy: float) -> Fixation:
    return Fixation(0, 1, 0.0, 0.1, cx, cy)


def _sacc(direction: float, amplitude: float) -> Saccade:
    return Saccade(0, 1, 0.0, 0.02, amplitude, direction, 300.0)


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


# --- shannon_entropy -------------------------------------------------------


def test_shannon_entropy_uniform_four_cells_is_two_bits() -> None:
    assert sm.shannon_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2.0)


def test_shannon_entropy_counts_normalised_defensively() -> None:
    # Raw counts of a uniform 2-outcome distribution -> 1 bit.
    assert sm.shannon_entropy([5.0, 5.0]) == pytest.approx(1.0)


def test_shannon_entropy_zero_prob_entries_contribute_zero() -> None:
    # A point mass has zero entropy even with explicit zero entries present.
    assert sm.shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_shannon_entropy_empty_and_zero_mass() -> None:
    assert sm.shannon_entropy([]) == 0.0
    assert sm.shannon_entropy([0.0, 0.0]) == 0.0


def test_shannon_entropy_base_e() -> None:
    # Uniform over 4 -> ln(4) nats.
    assert sm.shannon_entropy([1, 1, 1, 1], base=np.e) == pytest.approx(np.log(4.0))


# --- gaze_dispersion -------------------------------------------------------


def test_gaze_dispersion_known_value() -> None:
    # Four points at distance 1 from the origin centroid -> RMS = 1.
    xs = [1.0, -1.0, 0.0, 0.0]
    ys = [0.0, 0.0, 1.0, -1.0]
    assert sm.gaze_dispersion(xs, ys) == pytest.approx(1.0)


def test_gaze_dispersion_identical_points_is_zero() -> None:
    assert sm.gaze_dispersion([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]) == 0.0


def test_gaze_dispersion_empty() -> None:
    assert sm.gaze_dispersion([], []) == 0.0


def test_gaze_path_length_and_efficiency_filter_nonfinite_samples() -> None:
    xs = [0.0, 3.0, float("nan"), 3.0]
    ys = [0.0, 4.0, 99.0, 0.0]
    assert sm.gaze_path_length(xs, ys) == pytest.approx(9.0)
    assert sm.gaze_path_efficiency(xs, ys) == pytest.approx(3.0 / 9.0)


def test_raw_gaze_spatial_summary_is_json_friendly() -> None:
    summary = sm.raw_gaze_spatial_summary([0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0])
    assert summary["finite_sample_count"] == 4.0
    assert summary["path_length_deg"] == pytest.approx(3.0)
    assert 0.0 <= summary["path_efficiency"] <= 1.0
    assert summary["convex_hull_area_deg2"] == pytest.approx(1.0)


def test_raw_gaze_spatial_summary_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        sm.raw_gaze_spatial_summary([0.0, 1.0], [0.0])


# --- convex_hull_area ------------------------------------------------------


def test_convex_hull_area_unit_square() -> None:
    xs = [0.0, 1.0, 1.0, 0.0, 0.5]
    ys = [0.0, 0.0, 1.0, 1.0, 0.5]
    assert sm.convex_hull_area(xs, ys) == pytest.approx(1.0)


def test_convex_hull_area_triangle() -> None:
    # Triangle with base 2 and height 3 -> area 3.
    xs = [0.0, 2.0, 0.0]
    ys = [0.0, 0.0, 3.0]
    assert sm.convex_hull_area(xs, ys) == pytest.approx(3.0)


def test_convex_hull_area_collinear_is_zero() -> None:
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 2.0, 3.0]
    assert sm.convex_hull_area(xs, ys) == 0.0


def test_convex_hull_area_too_few_points() -> None:
    assert sm.convex_hull_area([0.0, 1.0], [0.0, 1.0]) == 0.0


# --- bcea ------------------------------------------------------------------


def test_bcea_positive_on_noisy_cloud() -> None:
    rng = np.random.default_rng(7)
    xs = rng.normal(0.0, 1.0, 200)
    ys = rng.normal(0.0, 1.0, 200)
    assert sm.bcea(xs, ys) > 0.0


def test_bcea_zero_on_identical_points() -> None:
    assert sm.bcea([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0


def test_bcea_too_few_points() -> None:
    assert sm.bcea([1.0], [2.0]) == 0.0


def test_bcea_independent_axes_matches_formula() -> None:
    # With rho ~ 0 the area approaches 2*k*pi*sx*sy. Use a large independent
    # sample and compare against the closed form with the observed moments.
    rng = np.random.default_rng(11)
    xs = rng.normal(0.0, 2.0, 5000)
    ys = rng.normal(0.0, 3.0, 5000)
    sx = float(np.std(xs, ddof=1))
    sy = float(np.std(ys, ddof=1))
    rho = float(np.corrcoef(xs, ys)[0, 1])
    k = -np.log(1.0 - 0.68)
    expected = 2.0 * k * np.pi * sx * sy * np.sqrt(1.0 - rho**2)
    assert sm.bcea(xs, ys, probability=0.68) == pytest.approx(expected, rel=1e-9)


def test_bcea_invalid_probability_raises() -> None:
    with pytest.raises(ValueError, match="probability must be in"):
        sm.bcea([0.0, 1.0], [0.0, 1.0], probability=0.0)
    with pytest.raises(ValueError, match="probability must be in"):
        sm.bcea([0.0, 1.0], [0.0, 1.0], probability=1.0)


# --- fixation_position_entropy --------------------------------------------


def test_fixation_position_entropy_uniform_grid_is_two_bits() -> None:
    # One fixation in each of the four cells of a 2x2 grid over [0,1]x[0,1].
    fixations = [
        _fix(0.25, 0.25),
        _fix(0.75, 0.25),
        _fix(0.25, 0.75),
        _fix(0.75, 0.75),
    ]
    entropy = sm.fixation_position_entropy(fixations, grid=(2, 2), extent=(0.0, 1.0, 0.0, 1.0))
    assert entropy == pytest.approx(2.0)


def test_fixation_position_entropy_single_cell_is_zero() -> None:
    fixations = [_fix(0.1, 0.1), _fix(0.12, 0.11)]
    entropy = sm.fixation_position_entropy(fixations, grid=(2, 2), extent=(0.0, 1.0, 0.0, 1.0))
    assert entropy == pytest.approx(0.0)


def test_fixation_position_entropy_empty() -> None:
    assert sm.fixation_position_entropy([]) == 0.0


def test_fixation_position_entropy_inferred_extent_degenerate_axis() -> None:
    # All centroids share an x; the inferred-range path must still produce a
    # finite entropy without raising.
    fixations = [_fix(1.0, 0.0), _fix(1.0, 1.0)]
    entropy = sm.fixation_position_entropy(fixations, grid=(2, 2))
    assert entropy >= 0.0


def test_fixation_position_entropy_invalid_grid_raises() -> None:
    with pytest.raises(ValueError, match="grid cells"):
        sm.fixation_position_entropy([_fix(0.0, 0.0)], grid=(0, 2))


# --- direction_transition_entropy -----------------------------------------


def test_direction_transition_entropy_deterministic_is_zero() -> None:
    # R always followed by U and U always followed by R -> fully predictable.
    saccs = [
        _sacc(0.0, 10.0),
        _sacc(90.0, 10.0),
        _sacc(0.0, 10.0),
        _sacc(90.0, 10.0),
        _sacc(0.0, 10.0),
    ]
    assert sm.direction_transition_entropy(saccs) == pytest.approx(0.0)


def test_direction_transition_entropy_positive_when_branching() -> None:
    # From R the next symbol is sometimes U, sometimes R -> positive entropy.
    saccs = [
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
        _sacc(0.0, 10.0),  # R
        _sacc(0.0, 10.0),  # R
    ]
    assert sm.direction_transition_entropy(saccs) > 0.0


def test_direction_transition_entropy_too_few_saccades() -> None:
    assert sm.direction_transition_entropy([]) == 0.0
    assert sm.direction_transition_entropy([_sacc(0.0, 10.0)]) == 0.0


# --- main_sequence_exponent_ci ---------------------------------------------


def test_exponent_ci_brackets_point_estimate_on_power_law() -> None:
    # Synthetic clean power law V = 50 * A^0.6.
    amp = np.linspace(2.0, 20.0, 40)
    vel = 50.0 * amp**0.6
    b, lo, hi = sm.main_sequence_exponent_ci(amp, vel, n_boot=500, seed=42)
    assert b == pytest.approx(0.6, abs=1e-6)
    assert lo <= b <= hi


def test_exponent_ci_is_deterministic() -> None:
    stream = _multi_saccade_stream()
    _f, saccs = saccades.detect_ivt(stream)
    props = saccades.saccade_properties(saccs)
    amp, vel = props["amplitude_deg"], props["peak_velocity_deg_s"]
    first = sm.main_sequence_exponent_ci(amp, vel, n_boot=300, seed=99)
    second = sm.main_sequence_exponent_ci(amp, vel, n_boot=300, seed=99)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_exponent_ci_invalid_arguments_raise() -> None:
    amp = np.linspace(2.0, 20.0, 10)
    vel = 40.0 * amp**0.5
    with pytest.raises(ValueError, match="n_boot"):
        sm.main_sequence_exponent_ci(amp, vel, n_boot=0)
    with pytest.raises(ValueError, match="confidence"):
        sm.main_sequence_exponent_ci(amp, vel, confidence=1.5)


def test_exponent_ci_degenerate_resamples_report_unavailable() -> None:
    amp = np.array([2.0, 6.0, 18.0])
    vel = 50.0 * amp**0.5
    with pytest.raises(ValueError, match="no valid bootstrap"):
        sm.main_sequence_exponent_ci(amp, vel, n_boot=1, seed=4)


# --- scanpath_summary ------------------------------------------------------


def test_scanpath_summary_from_explicit_lists() -> None:
    fixations = [_fix(0.0, 0.0), _fix(1.0, 0.0), _fix(0.0, 1.0), _fix(1.0, 1.0)]
    saccs = [_sacc(0.0, 10.0), _sacc(90.0, 10.0), _sacc(180.0, 10.0)]
    out = sm.scanpath_summary(fixations, saccs)
    assert set(out) == {
        "gaze_dispersion",
        "gaze_path_length",
        "gaze_path_efficiency",
        "convex_hull_area",
        "bcea",
        "fixation_position_entropy",
        "direction_transition_entropy",
    }
    assert out["convex_hull_area"] == pytest.approx(1.0)
    assert out["gaze_path_length"] > 0.0
    assert 0.0 <= out["gaze_path_efficiency"] <= 1.0
    assert all(isinstance(v, float) for v in out.values())


def test_scanpath_summary_from_session_report() -> None:
    fixations = [_fix(0.0, 0.0), _fix(2.0, 0.0), _fix(0.0, 2.0), _fix(2.0, 2.0)]
    saccs = [_sacc(0.0, 10.0), _sacc(90.0, 10.0)]
    report = SessionReport(
        n_samples=100,
        duration_s=1.0,
        fixations=fixations,
        saccades=saccs,
    )
    out = sm.scanpath_summary(report)
    assert out["convex_hull_area"] == pytest.approx(4.0)
    assert out["gaze_dispersion"] > 0.0


def test_scanpath_aoi_dict_inputs() -> None:
    from itrace.stats.scanpath_metrics import aoi_dwell_statistics, aoi_transition_matrix
    from itrace.types import Fixation

    aoi_dict = {"name": "target", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}
    fix = [
        Fixation(
            onset_idx=0,
            offset_idx=5,
            onset_t=0.0,
            offset_t=0.5,
            centroid_x=5.0,
            centroid_y=5.0,
        )
    ]
    stats = aoi_dwell_statistics(fix, [aoi_dict])
    assert stats["target"]["dwell_time_s"] == 0.5

    labels, _mat = aoi_transition_matrix(fix, [aoi_dict])
    assert labels == ["target", "outside"]


def test_scanpath_aoi_empty_and_no_match() -> None:
    from itrace.stats.scanpath_metrics import aoi_dwell_statistics, aoi_transition_matrix
    from itrace.types import AOI, Fixation

    aoi = AOI(name="target", x=100.0, y=100.0, w=10.0, h=10.0)
    fix = [
        Fixation(
            onset_idx=0,
            offset_idx=5,
            onset_t=0.0,
            offset_t=0.5,
            centroid_x=0.0,
            centroid_y=0.0,
        )
    ]
    stats = aoi_dwell_statistics(fix, [aoi])
    assert stats["target"]["dwell_time_s"] == 0.0
    assert stats["target"]["first_fixation_latency_s"] == -1.0
    assert stats["outside"]["dwell_time_s"] == 0.5
    assert stats["outside"]["first_fixation_latency_s"] == 0.0

    labels, mat = aoi_transition_matrix([], [aoi])
    assert labels == ["target", "outside"]
    assert mat.shape == (2, 2)


def test_scanpath_summary_missing_saccades_raises() -> None:
    with pytest.raises(ValueError, match="saccades must be provided"):
        sm.scanpath_summary([_fix(0.0, 0.0)])


def test_aoi_dataclass_and_methods() -> None:
    from itrace.stats.scanpath_metrics import aoi_dwell_statistics, aoi_transition_matrix
    from itrace.types import AOI, Fixation

    aoi_left = AOI(name="left_panel", x=-10.0, y=-10.0, w=10.0, h=20.0)
    aoi_right = AOI(name="right_panel", x=0.0, y=-10.0, w=10.0, h=20.0)

    assert aoi_left.contains(-5.0, 0.0) is True
    assert aoi_left.contains(5.0, 0.0) is False
    assert aoi_right.contains(5.0, 0.0) is True

    fixations = [
        Fixation(
            onset_idx=0,
            offset_idx=10,
            onset_t=0.0,
            offset_t=0.3,
            centroid_x=-5.0,
            centroid_y=0.0,
        ),
        Fixation(
            onset_idx=11,
            offset_idx=20,
            onset_t=0.35,
            offset_t=0.75,
            centroid_x=5.0,
            centroid_y=0.0,
        ),
        Fixation(
            onset_idx=21,
            offset_idx=30,
            onset_t=0.8,
            offset_t=1.0,
            centroid_x=-5.0,
            centroid_y=0.0,
        ),
        Fixation(
            onset_idx=31,
            offset_idx=40,
            onset_t=1.05,
            offset_t=1.25,
            centroid_x=50.0,
            centroid_y=50.0,
        ),
    ]

    stats = aoi_dwell_statistics(fixations, [aoi_left, aoi_right])
    assert stats["left_panel"]["fixation_count"] == 2.0
    assert stats["left_panel"]["dwell_time_s"] == pytest.approx(0.5)
    assert stats["left_panel"]["first_fixation_latency_s"] == pytest.approx(0.0)
    assert stats["right_panel"]["fixation_count"] == 1.0
    assert stats["right_panel"]["dwell_time_s"] == pytest.approx(0.4)
    assert stats["right_panel"]["first_fixation_latency_s"] == pytest.approx(0.35)
    assert stats["outside"]["fixation_count"] == 1.0
    assert stats["outside"]["dwell_time_s"] == pytest.approx(0.2)

    labels, matrix = aoi_transition_matrix(fixations, [aoi_left, aoi_right])
    assert labels == ["left_panel", "right_panel", "outside"]
    # Transitions: left -> right (1), right -> left (1), left -> outside (1)
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(0.5)  # from left, 1 to right and 1 to outside
    assert matrix[0, 2] == pytest.approx(0.5)
    assert matrix[1, 0] == pytest.approx(1.0)  # from right, 1 to left
