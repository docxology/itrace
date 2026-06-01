"""Descriptive-statistics tests (no mocks, real synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace import pipeline, saccades
from itrace.stats import descriptive
from itrace.synthetic import gaze_fixation_noise, gaze_with_saccade
from itrace.types import Fixation, GazeStream, PupilStream, PupilUnit, Saccade


def _multi_saccade_stream() -> GazeStream:
    """Concatenate several synthetic saccades into one continuous stream."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in (4.0, 8.0, 12.0, 16.0, 20.0):
        g, _ = gaze_with_saccade(amplitude_deg=amp, fixation_s=0.1)
        xs.append(g.x)
        ys.append(g.y - g.y[0])
        ts.append(g.t + t_offset)
        t_offset = ts[-1][-1] + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def test_describe_known_values() -> None:
    stats = descriptive.describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5.0
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(np.sqrt(2.0))  # population std
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["sum"] == pytest.approx(15.0)
    assert stats["p50"] == pytest.approx(3.0)
    assert stats["p25"] == pytest.approx(2.0)
    assert stats["p90"] == pytest.approx(4.6)


def test_describe_empty_is_zeroed() -> None:
    stats = descriptive.describe([])
    assert stats["n"] == 0.0
    for key in ("mean", "std", "min", "max", "sum", "p25", "p50", "p75", "p90"):
        assert stats[key] == 0.0


def test_describe_custom_percentiles() -> None:
    stats = descriptive.describe([10.0, 20.0, 30.0], percentiles=(10, 95))
    assert "p10" in stats
    assert "p95" in stats
    assert "p50" not in stats


def test_coefficient_of_variation_known() -> None:
    values = [2.0, 4.0, 6.0]
    cv = descriptive.coefficient_of_variation(values)
    expected = float(np.std(values, ddof=0) / np.mean(values))
    assert cv == pytest.approx(expected)


def test_coefficient_of_variation_empty_and_zero_mean() -> None:
    assert descriptive.coefficient_of_variation([]) == 0.0
    assert descriptive.coefficient_of_variation([-1.0, 0.0, 1.0]) == 0.0


def test_fixation_summary_hand_constructed() -> None:
    fixations = [
        Fixation(0, 10, 0.0, 0.2, 0.0, 0.0),
        Fixation(11, 20, 0.5, 0.9, 1.0, 1.0),
    ]
    summary = descriptive.fixation_summary(fixations)
    assert summary["count"] == 2.0
    # durations are 0.2 and 0.4
    assert summary["total_dwell_s"] == pytest.approx(0.6)
    assert summary["mean_duration_s"] == pytest.approx(0.3)
    assert summary["median_duration_s"] == pytest.approx(0.3)
    # span = 0.9 - 0.0 = 0.9, two fixations -> rate = 2 / 0.9
    assert summary["fixation_rate_hz"] == pytest.approx(2.0 / 0.9)


def test_fixation_summary_empty() -> None:
    summary = descriptive.fixation_summary([])
    assert summary["count"] == 0.0
    assert summary["total_dwell_s"] == 0.0
    assert summary["fixation_rate_hz"] == 0.0


def test_fixation_summary_zero_span_rate_zero() -> None:
    # single fixation: span is its own duration; build one with zero span.
    fixations = [Fixation(0, 0, 1.0, 1.0, 0.0, 0.0)]
    summary = descriptive.fixation_summary(fixations)
    assert summary["count"] == 1.0
    assert summary["fixation_rate_hz"] == 0.0


def test_saccade_summary_hand_constructed() -> None:
    saccs = [
        Saccade(0, 5, 0.0, 0.05, 10.0, 0.0, 300.0),
        Saccade(6, 11, 0.1, 0.16, 20.0, 90.0, 500.0),
    ]
    summary = descriptive.saccade_summary(saccs)
    assert summary["count"] == 2.0
    assert summary["amplitude_deg_mean"] == pytest.approx(15.0)
    assert summary["peak_velocity_deg_s_max"] == pytest.approx(500.0)
    assert summary["duration_s_mean"] == pytest.approx((0.05 + 0.06) / 2.0)
    # ratios: 300/10 = 30, 500/20 = 25 -> mean 27.5
    assert summary["main_seq_ratio_mean"] == pytest.approx(27.5)


def test_saccade_summary_empty() -> None:
    summary = descriptive.saccade_summary([])
    assert summary["count"] == 0.0
    assert summary["amplitude_deg_n"] == 0.0
    assert summary["main_seq_ratio_mean"] == 0.0


def test_saccade_summary_zero_amplitude_excluded() -> None:
    saccs = [
        Saccade(0, 5, 0.0, 0.05, 0.0, 0.0, 300.0),
        Saccade(6, 11, 0.1, 0.16, 10.0, 90.0, 400.0),
    ]
    summary = descriptive.saccade_summary(saccs)
    # only the second saccade contributes: 400 / 10 = 40
    assert summary["main_seq_ratio_mean"] == pytest.approx(40.0)


def test_summaries_from_detected_events() -> None:
    stream = _multi_saccade_stream()
    fixations, saccs = saccades.detect_ivt(stream)
    assert len(saccs) >= 3
    fix_summary = descriptive.fixation_summary(fixations)
    sac_summary = descriptive.saccade_summary(saccs)
    assert fix_summary["count"] == float(len(fixations))
    assert sac_summary["count"] == float(len(saccs))
    assert sac_summary["amplitude_deg_mean"] > 0.0
    assert sac_summary["main_seq_ratio_mean"] > 0.0


def test_summarize_report_structure() -> None:
    stream = _multi_saccade_stream()
    report = pipeline.analyze_gaze(stream)
    summary = descriptive.summarize_report(report)
    assert summary["fixations"]["count"] == float(len(report.fixations))  # type: ignore[index]
    assert summary["saccades"]["count"] == float(len(report.saccades))  # type: ignore[index]
    assert summary["n_microsaccades"] == len(report.microsaccades)
    assert summary["scanpath_length"] == len(report.scanpath)


def test_summarize_report_pure_fixation_stream() -> None:
    stream = gaze_fixation_noise(duration_s=1.0, noise_deg=0.02)
    report = pipeline.analyze_gaze(stream)
    summary = descriptive.summarize_report(report)
    assert summary["saccades"]["count"] == 0.0  # type: ignore[index]
    assert summary["n_microsaccades"] >= 0
    assert summary["scanpath_length"] == len(report.scanpath)


def test_session_statistics_summarizes_recording_events_gaze_and_pupil() -> None:
    stream = _multi_saccade_stream()
    pupil = PupilStream(
        t=stream.t,
        size=0.25 + 0.03 * np.sin(stream.t),
        unit=PupilUnit.RELATIVE,
    )
    report = pipeline.analyze_session(stream, pupil)

    summary = descriptive.session_statistics(stream, pupil, report)

    recording = summary["recording"]  # type: ignore[assignment]
    gaze = summary["gaze"]  # type: ignore[assignment]
    events = summary["events"]  # type: ignore[assignment]
    pupil_summary = summary["pupil"]  # type: ignore[assignment]
    assert recording["sample_count"] == float(len(stream))  # type: ignore[index]
    assert recording["sample_rate_hz"] > 0.0  # type: ignore[index]
    assert recording["finite_gaze_fraction"] == pytest.approx(1.0)  # type: ignore[index]
    assert gaze["path_length_deg"] > 0.0  # type: ignore[index]
    assert 0.0 <= gaze["path_efficiency"] <= 1.0  # type: ignore[index]
    assert events["saccade_rate_per_min"] > 0.0  # type: ignore[index]
    assert events["fixation_dwell_fraction"] >= 0.0  # type: ignore[index]
    assert pupil_summary["valid_fraction"] == pytest.approx(1.0)  # type: ignore[index]


def test_session_statistics_handles_empty_streams() -> None:
    empty = np.zeros(0, dtype=np.float64)
    gaze = GazeStream(t=empty, x=empty, y=empty)
    pupil = PupilStream(t=empty, size=empty, unit=PupilUnit.RELATIVE)

    summary = descriptive.session_statistics(gaze, pupil)

    assert summary["recording"]["sample_count"] == 0.0  # type: ignore[index]
    assert summary["gaze"]["path_length_deg"] == 0.0  # type: ignore[index]
    assert summary["events"]["saccade_rate_per_min"] == 0.0  # type: ignore[index]
    assert summary["pupil"]["valid_fraction"] == 0.0  # type: ignore[index]
