"""Descriptive statistics over fixations, saccades and session reports.

These are pure, side-effect-free functions that reduce typed oculomotor events
into plain ``dict`` summaries built from native ``float``/``int`` values, so the
results serialise straight to JSON. The conventions follow the rest of the
toolkit:

* spread is reported as a *population* standard deviation (``ddof=0``);
* percentiles use NumPy's linear interpolation (``method="linear"``);
* empty inputs never raise -- they collapse to a zero-filled summary with
  ``n == 0`` so downstream reporting code has a uniform shape to consume.

Durations come from each event's :pyattr:`~itrace.types.Fixation.duration_s`
property; amplitudes and peak velocities come straight off
:class:`~itrace.types.Saccade`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ..types import Fixation, FloatArray, GazeStream, PupilStream, Saccade, SessionReport
from . import scanpath_metrics

__all__ = [
    "coefficient_of_variation",
    "describe",
    "fixation_summary",
    "saccade_summary",
    "session_statistics",
    "summarize_report",
]


def _as_float_array(values: Iterable[float]) -> FloatArray:
    """Coerce any iterable of numbers into a 1-D ``float64`` array."""
    return np.asarray(list(values), dtype=np.float64).reshape(-1)


def describe(
    values: Iterable[float],
    percentiles: Sequence[int] = (25, 50, 75, 90),
) -> dict[str, float]:
    """Summarise a sample of scalar values.

    Parameters
    ----------
    values
        Any iterable of numbers (e.g. a list of durations).
    percentiles
        Percentile ranks (0-100) to report. Each becomes a key ``"pNN"`` where
        ``NN`` is the integer rank, e.g. ``50`` -> ``"p50"``.

    Returns
    -------
    dict[str, float]
        Keys ``n`` (count, as a float), ``mean``, ``std`` (population, ``ddof=0``),
        ``min``, ``max``, ``sum`` and one ``pNN`` per requested percentile. An
        empty sample yields zeros everywhere with ``n == 0.0`` (percentile keys
        are still present and set to ``0.0``).

    Notes
    -----
    All values are returned as native ``float`` for JSON friendliness.
    """
    arr = _as_float_array(values)
    summary: dict[str, float] = {}
    if arr.size == 0:
        summary = {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }
        for p in percentiles:
            summary[f"p{int(p)}"] = 0.0
        return summary

    summary = {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }
    for p in percentiles:
        summary[f"p{int(p)}"] = float(np.percentile(arr, float(p)))
    return summary


def coefficient_of_variation(values: Iterable[float]) -> float:
    """Return the coefficient of variation ``std / mean`` of a sample.

    Returns ``0.0`` for an empty sample or whenever the mean is exactly zero
    (avoids a divide-by-zero and keeps the result JSON-friendly).
    """
    arr = _as_float_array(values)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    return float(np.std(arr, ddof=0) / mean)


def fixation_summary(fixations: list[Fixation]) -> dict[str, float]:
    """Summarise a list of fixations.

    Parameters
    ----------
    fixations
        Detected fixations (e.g. the first element of
        :func:`itrace.saccades.detect_ivt`).

    Returns
    -------
    dict[str, float]
        ``count`` (int-valued float), ``total_dwell_s`` (summed durations),
        ``mean_duration_s``, ``median_duration_s``, ``std_duration_s`` and
        ``p90_duration_s`` (all derived from :func:`describe` on the durations),
        plus ``fixation_rate_hz`` -- the number of fixations divided by the time
        spanned from the earliest ``onset_t`` to the latest ``offset_t``. The
        rate is ``0.0`` when there are no fixations or the span is non-positive.
    """
    count = len(fixations)
    durations = [f.duration_s for f in fixations]
    stats = describe(durations, percentiles=(50, 90))
    total_dwell = float(stats["sum"])

    rate = 0.0
    if count > 0:
        span = max(f.offset_t for f in fixations) - min(f.onset_t for f in fixations)
        if span > 0.0:
            rate = float(count) / float(span)

    return {
        "count": float(count),
        "total_dwell_s": total_dwell,
        "mean_duration_s": float(stats["mean"]),
        "median_duration_s": float(stats["p50"]),
        "std_duration_s": float(stats["std"]),
        "p90_duration_s": float(stats["p90"]),
        "fixation_rate_hz": rate,
    }


def saccade_summary(saccades: list[Saccade]) -> dict[str, float]:
    """Summarise a list of saccades.

    Parameters
    ----------
    saccades
        Detected saccades (e.g. the second element of
        :func:`itrace.saccades.detect_ivt`).

    Returns
    -------
    dict[str, float]
        ``count`` plus three :func:`describe` blocks flattened with prefixes:
        ``amplitude_deg_*`` (amplitudes), ``peak_velocity_deg_s_*`` (peak
        velocities) and ``duration_s_*`` (durations). It also reports
        ``main_seq_ratio_mean`` -- the mean of
        ``peak_velocity_deg_s / amplitude_deg`` taken only over saccades with a
        strictly positive amplitude (``0.0`` when none qualify).
    """
    count = len(saccades)
    amplitudes = [s.amplitude_deg for s in saccades]
    peak_velocities = [s.peak_velocity_deg_s for s in saccades]
    durations = [s.duration_s for s in saccades]

    summary: dict[str, float] = {"count": float(count)}
    for prefix, block in (
        ("amplitude_deg", describe(amplitudes)),
        ("peak_velocity_deg_s", describe(peak_velocities)),
        ("duration_s", describe(durations)),
    ):
        for key, value in block.items():
            summary[f"{prefix}_{key}"] = value

    ratios = [s.peak_velocity_deg_s / s.amplitude_deg for s in saccades if s.amplitude_deg > 0.0]
    summary["main_seq_ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
    return summary


def summarize_report(report: SessionReport) -> dict[str, object]:
    """Roll a :class:`~itrace.types.SessionReport` into nested summaries.

    Returns
    -------
    dict[str, object]
        ``fixations`` -> :func:`fixation_summary`, ``saccades`` ->
        :func:`saccade_summary`, ``n_microsaccades`` (int) and
        ``scanpath_length`` (number of characters in the encoded scanpath).
    """
    return {
        "fixations": fixation_summary(report.fixations),
        "saccades": saccade_summary(report.saccades),
        "n_microsaccades": len(report.microsaccades),
        "scanpath_length": len(report.scanpath),
    }


def _sample_interval_summary(times: FloatArray) -> dict[str, float]:
    """Summarise monotonic sample cadence for session/live diagnostics."""
    if times.size < 2:
        return {
            "sample_rate_hz": 0.0,
            "median_interval_s": 0.0,
            "mean_interval_s": 0.0,
            "sampling_interval_cv": 0.0,
        }
    intervals = np.diff(times)
    finite = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if finite.size == 0:
        return {
            "sample_rate_hz": 0.0,
            "median_interval_s": 0.0,
            "mean_interval_s": 0.0,
            "sampling_interval_cv": 0.0,
        }
    median_interval = float(np.median(finite))
    mean_interval = float(np.mean(finite))
    return {
        "sample_rate_hz": 1.0 / median_interval if median_interval > 0.0 else 0.0,
        "median_interval_s": median_interval,
        "mean_interval_s": mean_interval,
        "sampling_interval_cv": coefficient_of_variation(finite),
    }


def _pupil_statistics(pupil: PupilStream | None) -> dict[str, float | str]:
    """Return live-safe pupil validity and range metrics."""
    if pupil is None or len(pupil) == 0:
        return {
            "sample_count": 0.0,
            "valid_fraction": 0.0,
            "mean_size": 0.0,
            "range": 0.0,
            "unit": "",
        }
    finite = pupil.size[np.isfinite(pupil.size)]
    if finite.size == 0:
        return {
            "sample_count": float(len(pupil)),
            "valid_fraction": 0.0,
            "mean_size": 0.0,
            "range": 0.0,
            "unit": pupil.unit.value,
        }
    return {
        "sample_count": float(len(pupil)),
        "valid_fraction": float(finite.size / len(pupil)),
        "mean_size": float(np.mean(finite)),
        "range": float(np.max(finite) - np.min(finite)),
        "unit": pupil.unit.value,
    }


def session_statistics(
    gaze: GazeStream,
    pupil: PupilStream | None = None,
    report: SessionReport | None = None,
) -> dict[str, object]:
    """Summarise recording, raw gaze, events, and pupil for reports/UI.

    This function intentionally avoids optional visualization/web dependencies
    so the same statistics can be used by CLI reports, generated manuscripts,
    and the local live HTML interface.
    """
    duration = (
        float(report.duration_s)
        if report is not None
        else float(gaze.t[-1] - gaze.t[0])
        if len(gaze) >= 2
        else 0.0
    )
    finite_gaze = np.isfinite(gaze.x) & np.isfinite(gaze.y)
    cadence = _sample_interval_summary(gaze.t)
    fixations = report.fixations if report is not None else []
    saccades = report.saccades if report is not None else []
    microsaccades = report.microsaccades if report is not None else []
    psos = report.psos if report is not None else []

    fix_summary = fixation_summary(fixations)
    sac_summary = saccade_summary(saccades)
    dwell = float(fix_summary["total_dwell_s"])
    saccade_rate_hz = float(len(saccades) / duration) if duration > 0.0 else 0.0
    fixation_rate_hz = float(len(fixations) / duration) if duration > 0.0 else 0.0
    microsaccade_rate_hz = float(len(microsaccades) / duration) if duration > 0.0 else 0.0
    pso_rate_hz = float(len(psos) / duration) if duration > 0.0 else 0.0

    return {
        "recording": {
            "sample_count": float(len(gaze)),
            "duration_s": duration,
            "finite_gaze_fraction": float(np.mean(finite_gaze)) if len(gaze) else 0.0,
            **cadence,
        },
        "gaze": scanpath_metrics.raw_gaze_spatial_summary(gaze.x, gaze.y),
        "events": {
            "fixation_count": float(len(fixations)),
            "saccade_count": float(len(saccades)),
            "microsaccade_count": float(len(microsaccades)),
            "pso_count": float(len(psos)),
            "fixation_dwell_s": dwell,
            "fixation_dwell_fraction": dwell / duration if duration > 0.0 else 0.0,
            "fixation_rate_hz": fixation_rate_hz,
            "saccade_rate_hz": saccade_rate_hz,
            "saccade_rate_per_min": saccade_rate_hz * 60.0,
            "microsaccade_rate_hz": microsaccade_rate_hz,
            "pso_rate_hz": pso_rate_hz,
            "mean_fixation_duration_s": float(fix_summary["mean_duration_s"]),
            "mean_saccade_amplitude_deg": float(sac_summary["amplitude_deg_mean"]),
            "mean_saccade_peak_velocity_deg_s": float(sac_summary["peak_velocity_deg_s_mean"]),
            "fixation_position_entropy_bits": scanpath_metrics.fixation_position_entropy(fixations),
            "direction_transition_entropy_bits": scanpath_metrics.direction_transition_entropy(
                saccades
            ),
        },
        "pupil": _pupil_statistics(pupil),
    }
