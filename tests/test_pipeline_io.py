"""Pipeline + IO integration tests (ISC-46..47, ISC-49 shape)."""

from __future__ import annotations

import numpy as np

from itrace import io, pipeline
from itrace.config import AnalysisConfig, DetectionConfig, PupilConfig
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink
from itrace.types import SessionReport


def test_analyze_session_end_to_end() -> None:  # ISC-46
    gaze, truth = gaze_with_saccade(amplitude_deg=10.0, direction_deg=0.0)
    pstream, _peaks = pupil_sine_with_blink()
    report = pipeline.analyze_session(gaze, pstream)
    assert isinstance(report, SessionReport)
    import pytest

    assert len(report.saccades) == 1
    assert report.saccades[0].amplitude_deg == pytest.approx(truth.amplitude_deg, rel=0.05)
    assert report.scanpath  # non-empty encoded scanpath
    assert report.pupil  # pupil summary populated
    assert report.pupil["n_blinks"] >= 1.0


def test_analyze_session_without_pupil() -> None:
    gaze, _ = gaze_with_saccade()
    report = pipeline.analyze_session(gaze)
    assert report.pupil == {}


def test_analyze_gaze_accepts_adaptive_config_and_reports_quality() -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=12.0, noise_deg=0.02)
    cfg = AnalysisConfig(detection=DetectionConfig(method="adaptive_ivt", include_pso=True))
    report = pipeline.analyze_gaze(gaze, config=cfg)
    assert report.config["detection"]["method"] == "adaptive_ivt"  # type: ignore[index]
    assert report.quality["detection_threshold_deg_s"] > 0.0
    assert "pso_rate_hz" in report.quality


def test_analyze_gaze_reports_smooth_pursuit_when_enabled() -> None:
    from itrace.types import GazeStream

    t = np.arange(160, dtype=np.float64) / 80.0
    stream = GazeStream(t=t, x=6.0 * t, y=np.zeros_like(t))
    cfg = AnalysisConfig(
        detection=DetectionConfig(
            velocity_threshold_deg_s=30.0,
            include_microsaccades=False,
            include_smooth_pursuit=True,
        )
    )

    report = pipeline.analyze_gaze(stream, config=cfg)
    payload = report.to_dict()

    assert len(report.smooth_pursuits) == 1
    assert payload["n_smooth_pursuits"] == 1
    assert payload["smooth_pursuits"][0]["mean_velocity_deg_s"] > 0.0


def test_analysis_config_merge_gap_reduces_split_saccades() -> None:
    from itrace.types import GazeStream

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
    stream = GazeStream(t=np.arange(arr.size) * dt, x=arr, y=np.zeros_like(arr))

    plain = pipeline.analyze_gaze(
        stream,
        config=AnalysisConfig(detection=DetectionConfig(min_saccade_duration_s=0.0)),
    )
    robust = pipeline.analyze_gaze(
        stream,
        config=AnalysisConfig(
            detection=DetectionConfig(min_saccade_duration_s=0.0, merge_gap_s=0.04)
        ),
    )

    assert len(plain.saccades) > len(robust.saccades)
    assert len(robust.saccades) == 1
    assert robust.config["detection"]["merge_gap_s"] == 0.04  # type: ignore[index]


def test_main_sequence_populated_with_many_saccades() -> None:
    # concatenate several saccades of differing amplitude into one stream
    import numpy as np

    from itrace.types import GazeStream

    xs, ys, ts = [], [], []
    t_offset = 0.0
    for amp in (4.0, 8.0, 12.0, 16.0, 20.0):
        g, _ = gaze_with_saccade(amplitude_deg=amp, fixation_s=0.1)
        xs.append(g.x)
        ys.append(g.y - g.y[0])
        ts.append(g.t + t_offset)
        t_offset = ts[-1][-1] + 1.0 / 250.0
    stream = GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))
    report = pipeline.analyze_gaze(stream)
    assert len(report.saccades) >= 3
    assert "power_b" in report.main_sequence


def test_gaze_csv_roundtrip(tmp_path) -> None:  # ISC-47
    gaze, _ = gaze_with_saccade()
    path = tmp_path / "gaze.csv"
    io.write_gaze_csv(gaze, path)
    back = io.read_gaze_csv(path)
    assert np.allclose(back.t, gaze.t)
    assert np.allclose(back.x, gaze.x)
    assert np.allclose(back.y, gaze.y)


def test_pupil_csv_roundtrip(tmp_path) -> None:
    stream, _ = pupil_sine_with_blink(blink_window_s=None)
    path = tmp_path / "pupil.csv"
    io.write_pupil_csv(stream, path)
    back = io.read_pupil_csv(path)
    assert np.allclose(back.size, stream.size)
    assert back.unit == stream.unit


def test_io_missing_columns(tmp_path) -> None:
    import pandas as pd
    import pytest

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"t": [0.0], "x": [0.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        io.read_gaze_csv(bad)
    pd.DataFrame({"t": [0.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        io.read_pupil_csv(bad)


def test_report_to_dict_serialisable() -> None:  # ISC-49 shape
    import json

    gaze, _ = gaze_with_saccade()
    report = pipeline.analyze_session(gaze)
    payload = json.dumps(report.to_dict())
    parsed = json.loads(payload)
    assert "saccades" in parsed
    assert "fixations" in parsed
    assert parsed["n_saccades"] == len(report.saccades)


def test_pupil_summary_includes_validity_and_velocity_metrics() -> None:
    pstream, _ = pupil_sine_with_blink(blink_window_s=(2.0, 2.2))
    report = pipeline.analyze_session(gaze_with_saccade()[0], pstream)

    assert report.pupil["valid_sample_fraction"] < 1.0
    assert report.pupil["blink_fraction"] > 0.0
    assert report.pupil["std_size"] > 0.0
    assert report.pupil["peak_dilation_velocity"] > 0.0
    assert report.pupil["peak_constriction_velocity"] < 0.0


def test_pupil_config_threshold_flows_into_session_summary() -> None:
    pstream = pupil_sine_with_blink(blink_window_s=None)[0]
    low_samples = pstream.size.copy()
    low_samples[:8] = np.array([3.0, 3.1, 0.15, 3.1, 3.0, 0.1, 3.2, 3.1])
    custom = type(pstream)(t=pstream.t, size=low_samples, unit=pstream.unit)
    cfg = AnalysisConfig(pupil=PupilConfig(blink_threshold=0.2, blink_pad_samples=0))

    report = pipeline.analyze_session(gaze_with_saccade()[0], custom, config=cfg)

    assert report.pupil["n_blinks"] >= 2.0
    assert report.pupil["valid_sample_fraction"] < 1.0
    assert report.config["pupil"]["blink_threshold"] == 0.2  # type: ignore[index]
