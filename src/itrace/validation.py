"""Synthetic-domain validation and live-recording diagnostics.

This module is deliberately pure NumPy/iTrace core. Synthetic validation uses
known simulator truth; live webcam diagnostics report only observability and
plausibility because there is no reference eye tracker in the loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import sqrt

import numpy as np

from . import pipeline
from .calibration import interpolate_gaze_gaps, robust_gaze_quality
from .config import AnalysisConfig, DetectionConfig
from .stats import bootstrap
from .stats.events import event_prf, interval_overlap_s
from .synthetic import SyntheticSaccadeTruth, SyntheticSessionSpec, synthetic_session
from .types import GazeStream, PupilStream, Saccade


@dataclass(frozen=True, slots=True)
class SyntheticDomain:
    """A deterministic validation domain with its own signal stressors."""

    name: str
    label: str
    spec: SyntheticSessionSpec
    detection: DetectionConfig = field(
        default_factory=lambda: DetectionConfig(method="adaptive_ivt", include_pso=True)
    )


def default_synthetic_domains() -> tuple[SyntheticDomain, ...]:
    """Return the standard synthetic domains used for recovery validation."""
    base = SyntheticSessionSpec(seed=0, n_saccades=7, duration_s=7.0)
    return (
        SyntheticDomain(
            name="clean_desktop",
            label="Clean desktop",
            spec=replace(base, noise_deg=0.01, pupil_noise=0.01, dropout_fraction=0.0),
        ),
        SyntheticDomain(
            name="webcam_jitter",
            label="Webcam jitter",
            spec=replace(
                base,
                noise_deg=0.05,
                timestamp_jitter_s=0.001,
                correlated_noise_deg=0.02,
                pupil_noise=0.035,
            ),
        ),
        SyntheticDomain(
            name="head_drift",
            label="Head-pose drift",
            spec=replace(
                base,
                noise_deg=0.035,
                correlated_noise_deg=0.03,
                head_pose_drift_deg=1.75,
                pupil_noise=0.025,
            ),
        ),
        SyntheticDomain(
            name="low_light_dropout",
            label="Low-light dropout",
            spec=replace(
                base,
                noise_deg=0.06,
                pupil_noise=0.055,
                dropout_fraction=0.04,
                lighting_dropouts_s=((1.1, 1.28), (4.2, 4.42)),
            ),
        ),
    )


def _truth_as_saccades(truth: Sequence[SyntheticSaccadeTruth]) -> list[Saccade]:
    return [
        Saccade(
            onset_idx=0,
            offset_idx=0,
            onset_t=item.onset_t,
            offset_t=item.offset_t,
            amplitude_deg=item.amplitude_deg,
            direction_deg=item.direction_deg,
            peak_velocity_deg_s=item.peak_velocity_deg_s,
        )
        for item in truth
    ]


def _match_saccades(
    truth: Sequence[Saccade],
    predicted: Sequence[Saccade],
    *,
    min_overlap_s: float = 0.005,
) -> list[tuple[Saccade, Saccade]]:
    matched: set[int] = set()
    pairs: list[tuple[Saccade, Saccade]] = []
    truth_sorted = sorted(truth, key=lambda event: event.onset_t)
    pred_sorted = sorted(predicted, key=lambda event: event.onset_t)
    for pred in pred_sorted:
        best_idx = -1
        best_overlap = min_overlap_s
        for idx, item in enumerate(truth_sorted):
            if idx in matched:
                continue
            overlap = interval_overlap_s(pred.onset_t, pred.offset_t, item.onset_t, item.offset_t)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx >= 0:
            matched.add(best_idx)
            pairs.append((truth_sorted[best_idx], pred))
    return pairs


def _mean_absolute(values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64)))) if values else 0.0


def _rmse(values: list[float]) -> float:
    return (
        float(sqrt(float(np.mean(np.square(np.asarray(values, dtype=np.float64))))))
        if values
        else 0.0
    )


def _circular_error_deg(a: float, b: float) -> float:
    return float(((a - b + 180.0) % 360.0) - 180.0)


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        out = float(value)
        return out if np.isfinite(out) else default
    return default


def _finite_gaze_stream(gaze: GazeStream) -> GazeStream:
    mask = np.isfinite(gaze.t) & np.isfinite(gaze.x) & np.isfinite(gaze.y)
    if int(np.sum(mask)) < 3:
        msg = "synthetic validation needs at least 3 finite gaze samples"
        raise ValueError(msg)
    return GazeStream(t=gaze.t[mask], x=gaze.x[mask], y=gaze.y[mask])


def validate_synthetic_domain(
    domain: SyntheticDomain,
    *,
    seed: int | None = None,
    min_overlap_s: float = 0.005,
) -> dict[str, object]:
    """Analyse one synthetic session and compare detected events with truth."""
    spec = replace(domain.spec, seed=domain.spec.seed if seed is None else seed)
    gaze, pupil, truth = synthetic_session(spec)
    analysis_gaze = gaze
    interpolated = False
    dropped_invalid = False
    config = AnalysisConfig(detection=domain.detection)
    try:
        report = pipeline.analyze_session(analysis_gaze, pupil, config=config)
    except ValueError:
        analysis_gaze = interpolate_gaze_gaps(gaze, max_gap_s=0.08)
        interpolated = True
        if not (np.all(np.isfinite(analysis_gaze.x)) and np.all(np.isfinite(analysis_gaze.y))):
            analysis_gaze = _finite_gaze_stream(analysis_gaze)
            dropped_invalid = True
        report = pipeline.analyze_session(analysis_gaze, pupil, config=config)
    truth_saccades = _truth_as_saccades(truth.saccades)
    recovery = event_prf(truth_saccades, report.saccades, min_overlap_s=min_overlap_s)
    pairs = _match_saccades(truth_saccades, report.saccades, min_overlap_s=min_overlap_s)
    amplitude_errors = [pred.amplitude_deg - true.amplitude_deg for true, pred in pairs]
    direction_errors = [
        _circular_error_deg(pred.direction_deg, true.direction_deg) for true, pred in pairs
    ]
    peak_errors = [pred.peak_velocity_deg_s - true.peak_velocity_deg_s for true, pred in pairs]
    duration_errors = [pred.duration_s - true.duration_s for true, pred in pairs]
    pupil_finite = np.isfinite(pupil.size)
    pupil_values = pupil.size[pupil_finite]
    pupil_dynamic_range = (
        float(np.max(pupil_values) - np.min(pupil_values)) if pupil_values.size else 0.0
    )

    quality = report.quality
    raw_quality = robust_gaze_quality(gaze)
    return {
        "domain": domain.name,
        "label": domain.label,
        "seed": spec.seed,
        "n_truth_saccades": len(truth_saccades),
        "n_detected_saccades": len(report.saccades),
        "n_matched_saccades": len(pairs),
        "saccade_recovery": recovery,
        "gaze": {
            "finite_sample_fraction": _number(raw_quality.get("valid_sample_fraction")),
            "sampling_rate_hz": _number(quality.get("sampling_rate_hz")),
            "detection_threshold_deg_s": _number(quality.get("detection_threshold_deg_s")),
            "dropout_fraction": _number(raw_quality.get("dropout_fraction")),
            "large_gap_count": _number(raw_quality.get("large_gap_count")),
            "short_gap_interpolation_used": float(interpolated),
            "invalid_sample_drop_used": float(dropped_invalid),
        },
        "saccade_error": {
            "amplitude_mae_deg": _mean_absolute(amplitude_errors),
            "amplitude_bias_deg": float(np.mean(amplitude_errors)) if amplitude_errors else 0.0,
            "direction_mae_deg": _mean_absolute(direction_errors),
            "peak_velocity_rmse_deg_s": _rmse(peak_errors),
            "duration_mae_s": _mean_absolute(duration_errors),
        },
        "pupil": {
            "truth_event_count": float(len(truth.pupil_events)),
            "truth_blink_count": float(len(truth.blink_windows_s)),
            "truth_lighting_dropout_count": _number(
                truth.quality_flags.get("lighting_dropout_count")
            ),
            "valid_sample_fraction": float(np.mean(pupil_finite)) if pupil.size.size else 0.0,
            "dynamic_range": pupil_dynamic_range,
            "detected_blinks": _number(report.pupil.get("n_blinks")),
            "latency_to_peak_s": _number(report.pupil.get("latency_to_peak_s")),
            "dilation_auc": _number(report.pupil.get("dilation_auc")),
        },
    }


def _summary_block(values: Sequence[float], *, seed: int = 12345) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if finite.size == 1:
        value = float(finite[0])
        return {"n": 1.0, "mean": value, "std": 0.0, "ci_low": value, "ci_high": value}
    samples = bootstrap.bootstrap_statistic(
        finite, lambda sample: float(np.mean(sample)), seed=seed
    )
    ci_low, ci_high = bootstrap.percentile_interval(samples)
    return {
        "n": float(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _path_number(payload: Mapping[str, object], *keys: str) -> float:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return 0.0
        current = current.get(key)
    return _number(current)


def within_domain_validation(
    domain: SyntheticDomain,
    *,
    repetitions: int = 5,
    first_seed: int = 0,
) -> dict[str, object]:
    """Run repeated seeded sessions inside one synthetic domain."""
    if repetitions < 1:
        msg = "repetitions must be >= 1"
        raise ValueError(msg)
    runs = [
        validate_synthetic_domain(domain, seed=first_seed + index) for index in range(repetitions)
    ]
    return {
        "domain": domain.name,
        "label": domain.label,
        "repetitions": repetitions,
        "runs": runs,
        "summary": {
            "saccade_f1": _summary_block(
                [_path_number(run, "saccade_recovery", "f1") for run in runs]
            ),
            "saccade_precision": _summary_block(
                [_path_number(run, "saccade_recovery", "precision") for run in runs]
            ),
            "saccade_recall": _summary_block(
                [_path_number(run, "saccade_recovery", "recall") for run in runs]
            ),
            "amplitude_mae_deg": _summary_block(
                [_path_number(run, "saccade_error", "amplitude_mae_deg") for run in runs]
            ),
            "direction_mae_deg": _summary_block(
                [_path_number(run, "saccade_error", "direction_mae_deg") for run in runs]
            ),
            "peak_velocity_rmse_deg_s": _summary_block(
                [_path_number(run, "saccade_error", "peak_velocity_rmse_deg_s") for run in runs]
            ),
            "pupil_valid_fraction": _summary_block(
                [_path_number(run, "pupil", "valid_sample_fraction") for run in runs]
            ),
            "gaze_finite_fraction": _summary_block(
                [_path_number(run, "gaze", "finite_sample_fraction") for run in runs]
            ),
        },
    }


def synthetic_validation_suite(
    *,
    domains: Sequence[SyntheticDomain] | None = None,
    repetitions: int = 5,
    first_seed: int = 0,
) -> dict[str, object]:
    """Run within- and across-domain synthetic recovery validation."""
    selected = tuple(domains) if domains is not None else default_synthetic_domains()
    domain_results = [
        within_domain_validation(domain, repetitions=repetitions, first_seed=first_seed)
        for domain in selected
    ]
    f1s = [_path_number(domain, "summary", "saccade_f1", "mean") for domain in domain_results]
    worst_idx = int(np.argmin(f1s)) if f1s else -1
    return {
        "kind": "synthetic_domain_validation",
        "truth_boundary": ("Synthetic domains have ground truth; live webcam diagnostics do not."),
        "domain_count": len(domain_results),
        "repetitions": repetitions,
        "domains": domain_results,
        "cross_domain": {
            "macro_saccade_f1": float(np.mean(f1s)) if f1s else 0.0,
            "min_saccade_f1": float(np.min(f1s)) if f1s else 0.0,
            "max_saccade_f1": float(np.max(f1s)) if f1s else 0.0,
            "stability_gap_f1": float(np.max(f1s) - np.min(f1s)) if f1s else 0.0,
            "worst_domain": domain_results[worst_idx]["domain"] if worst_idx >= 0 else None,
        },
    }


def live_recording_diagnostics(
    gaze: GazeStream,
    pupil: PupilStream,
    report_payload: Mapping[str, object],
) -> dict[str, object]:
    """Summarise live recording quality without claiming ground-truth validity."""
    n = len(gaze)
    finite_gaze = np.isfinite(gaze.x) & np.isfinite(gaze.y)
    valid_fraction = float(np.mean(finite_gaze)) if n else 0.0
    dt = np.diff(gaze.t) if n >= 2 else np.zeros(0, dtype=np.float64)
    finite_dt = dt[np.isfinite(dt) & (dt > 0.0)]
    median_dt = float(np.median(finite_dt)) if finite_dt.size else 0.0
    sampling_rate = 1.0 / median_dt if median_dt > 0.0 else 0.0
    sampling_cv = (
        float(np.std(finite_dt, ddof=0) / np.mean(finite_dt))
        if finite_dt.size and float(np.mean(finite_dt)) > 0.0
        else 0.0
    )
    dx = np.diff(gaze.x)
    dy = np.diff(gaze.y)
    finite_step = np.isfinite(dx) & np.isfinite(dy)
    path_length = float(np.sum(np.hypot(dx[finite_step], dy[finite_step])))
    if np.any(finite_gaze):
        centered_x = gaze.x[finite_gaze] - float(np.mean(gaze.x[finite_gaze]))
        centered_y = gaze.y[finite_gaze] - float(np.mean(gaze.y[finite_gaze]))
        dispersion = float(np.sqrt(np.mean(centered_x**2 + centered_y**2)))
    else:
        dispersion = 0.0
    pupil_valid = np.isfinite(pupil.size)
    pupil_fraction = float(np.mean(pupil_valid)) if len(pupil) else 0.0
    pupil_values = pupil.size[pupil_valid]
    pupil_range = float(np.max(pupil_values) - np.min(pupil_values)) if pupil_values.size else 0.0

    quality_payload = report_payload.get("quality")
    quality = quality_payload if isinstance(quality_payload, Mapping) else {}
    f1_unavailable_reason = "no live reference truth"
    warnings: list[str] = []
    if n < 30:
        warnings.append("low sample count")
    if valid_fraction < 0.85:
        warnings.append("low finite gaze fraction")
    if sampling_cv > 0.2:
        warnings.append("irregular sampling interval")
    if pupil_fraction < 0.5:
        warnings.append("low pupil-valid fraction")
    if _number(report_payload.get("n_saccades")) == 0.0 and n >= 90:
        warnings.append("no saccades detected in current window")

    score = 100.0 * (
        0.35 * valid_fraction
        + 0.2 * min(sampling_rate / 30.0, 1.0)
        + 0.2 * max(0.0, 1.0 - min(sampling_cv, 1.0))
        + 0.15 * pupil_fraction
        + 0.1 * min(_number(report_payload.get("duration_s")) / 5.0, 1.0)
    )
    return {
        "truth_boundary": f1_unavailable_reason,
        "quality_index": float(max(0.0, min(score, 100.0))),
        "sample_count": n,
        "sampling_rate_hz": sampling_rate,
        "sampling_interval_cv": sampling_cv,
        "finite_gaze_fraction": valid_fraction,
        "gaze_path_length_deg": path_length,
        "gaze_dispersion_deg": dispersion,
        "pupil_valid_fraction": pupil_fraction,
        "pupil_dynamic_range": pupil_range,
        "detected_threshold_deg_s": _number(quality.get("detection_threshold_deg_s")),
        "warnings": warnings,
    }
