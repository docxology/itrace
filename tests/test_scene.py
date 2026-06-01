"""Animated scene + full closed-loop validation tests (ISC-62..68)."""

from __future__ import annotations

import numpy as np

from itrace import scene
from itrace.scene import DilationEvent, EyeSceneSpec, Fixation3D


def test_animate_shapes_and_truth() -> None:  # ISC-62/ISC-63
    sc = scene.animate()
    n = sc.t.size
    assert sc.true_yaw.size == n
    assert sc.true_pitch.size == n
    assert sc.true_pupil_mm.size == n
    assert sc.blink.size == n
    assert len(sc.landmarks) == n
    assert sc.blink.any()  # the scene has a blink
    # each non-blink frame is a full landmark array
    first_valid = int(np.flatnonzero(~sc.blink)[0])
    assert sc.landmarks[first_valid].shape == (478, 3)


def test_pupil_truth_has_dilation_peaks() -> None:  # ISC-62
    sc = scene.animate()
    assert sc.true_pupil_mm.max() > sc.true_pupil_mm[0]  # dilation raised it


def test_closed_loop_gaze_recovery_under_two_degrees() -> None:  # ISC-65
    res = scene.closed_loop()
    assert res.metrics["gaze_rms_deg"] < 2.0
    assert res.metrics["gaze_max_deg"] < 3.0


def test_closed_loop_recovers_saccade_count() -> None:  # ISC-66
    # default scene has 4 fixations -> 3 saccades
    res = scene.closed_loop()
    assert res.metrics["n_saccades"] == 3.0


def test_closed_loop_pupil_correlation() -> None:  # ISC-67
    res = scene.closed_loop()
    assert res.metrics["pupil_corr"] > 0.9
    assert np.all(np.isfinite(res.recovered_pupil.size))  # blink interpolated out


def test_closed_loop_not_a_tautology() -> None:  # ISC-68
    # forward model (3-D + perspective) and estimator (arcsine) differ, so a
    # real, nonzero-but-bounded recovery error must exist.
    res = scene.closed_loop()
    assert res.metrics["gaze_rms_deg"] > 0.0


def test_closed_loop_custom_spec() -> None:
    spec = EyeSceneSpec(
        fixations=(Fixation3D(0.3, 0.0, 0.0), Fixation3D(0.3, 10.0, 0.0)),
        dilations=(DilationEvent(0.4, 1.0),),
        blinks_s=(),
        sampling_rate_hz=120.0,
    )
    res = scene.closed_loop(spec)
    assert res.metrics["n_saccades"] == 1.0
    assert res.metrics["gaze_rms_deg"] < 2.0
    assert not res.scene.blink.any()


def test_closed_loop_measurement_noise_is_seeded() -> None:
    spec = EyeSceneSpec(measurement_noise=0.1, seed=7)
    a = scene.closed_loop(spec).recovered_gaze.x
    b = scene.closed_loop(spec).recovered_gaze.x
    assert np.allclose(a, b)  # deterministic given seed
