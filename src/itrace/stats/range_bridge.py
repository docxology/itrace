"""Synthetic-to-empirical range bridge payloads.

This module is pure Python and JSON-oriented: it does not render figures or
touch camera/video paths. The payload it returns is intended to make one local
derived empirical pilot comparable with synthetic-domain and noise-model
evidence without turning that comparison into a device-validation claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, median
from typing import Any

DEFAULT_RANGE_BRIDGE_SOURCES: dict[str, str] = {
    "empirical_metrics": "docs/empirical_pilot_metrics.json",
    "synthetic_validation": "output/synthetic_validation.json",
    "noise_metrics": "output/figures/noise_metrics.json",
    "statistical_diagnostics": "output/figures/statistical_diagnostics.json",
}
"""Repo-relative evidence files consumed by the standard bridge payload."""

RAW_VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
"""File suffixes rejected in provenance because raw eye video is out of scope."""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nested_number(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        current = _mapping(current).get(key)
    return _finite_float(current)


def _display_unavailable() -> dict[str, object]:
    return {"available": False, "value": None, "display": "unavailable"}


def _display_value(value: float | None, unit: str) -> dict[str, object]:
    if value is None:
        return _display_unavailable()
    if unit == "fraction":
        display = f"{100.0 * value:.1f}%"
    elif unit == "hz":
        display = f"{value:.1f} Hz"
    elif unit == "cv":
        display = f"{value:.3f}"
    elif unit == "deg_s":
        display = f"{value:.3f} deg/s"
    elif unit == "deg":
        display = f"{value:.2f} deg"
    elif unit == "s":
        display = f"{value:.2f} s"
    else:
        display = f"{value:.3g}"
    return {"available": True, "value": float(value), "display": display}


def _validate_repo_relative_path(path: str | None, *, field: str) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        msg = f"{field} must be repo-relative provenance, got {path!r}"
        raise ValueError(msg)
    if candidate.suffix.lower() in RAW_VIDEO_SUFFIXES:
        msg = f"{field} points to raw video, which is outside the derived-data contract"
        raise ValueError(msg)
    return path


def _domain_summary(
    synthetic_validation: dict[str, Any],
    domain: str,
    metric: str,
) -> dict[str, Any]:
    domains = synthetic_validation.get("domains", [])
    if not isinstance(domains, list):
        return {}
    for row in domains:
        item = _mapping(row)
        if item.get("domain") == domain:
            return _mapping(_mapping(item.get("summary")).get(metric))
    return {}


def _domain_mean(
    synthetic_validation: dict[str, Any],
    domain: str,
    metric: str,
) -> float | None:
    return _finite_float(_domain_summary(synthetic_validation, domain, metric).get("mean"))


def _domain_metric_means(synthetic_validation: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    domains = synthetic_validation.get("domains", [])
    if not isinstance(domains, list):
        return values
    for row in domains:
        value = _finite_float(_mapping(_mapping(row).get("summary")).get(metric, {}).get("mean"))
        if value is not None:
            values.append(value)
    return values


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{100.0 * value:.1f}%"


def _clean_stress_fraction_display(
    synthetic_validation: dict[str, Any],
    metric: str,
) -> tuple[bool, str, float | None]:
    clean = _domain_mean(synthetic_validation, "clean_desktop", metric)
    all_values = _domain_metric_means(synthetic_validation, metric)
    if clean is None or not all_values:
        return False, "unavailable", None
    stress_min = min(all_values)
    return True, f"clean {_percent(clean)}; stress min {_percent(stress_min)}", stress_min


def _noise_threshold(noise_metrics: dict[str, Any], threshold: str) -> dict[str, Any]:
    return _mapping(_mapping(noise_metrics.get("thresholds")).get(threshold))


def _noise_records_range(
    noise_metrics: dict[str, Any], field: str
) -> tuple[float | None, float | None]:
    records = noise_metrics.get("records", [])
    if not isinstance(records, list):
        return None, None
    values = [_finite_float(_mapping(row).get(field)) for row in records]
    finite = [value for value in values if value is not None]
    if not finite:
        return None, None
    return min(finite), max(finite)


def _bootstrap_model_display(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    bootstrap = _mapping(
        _mapping(statistical_diagnostics.get("amplitude_model_comparison")).get(
            "model_selection_bootstrap"
        )
    )
    top_family = bootstrap.get("top_family")
    top_frequency = _finite_float(bootstrap.get("top_frequency"))
    if not bootstrap.get("available") or not top_family or top_frequency is None:
        reason = str(bootstrap.get("reason", "bootstrap model stability unavailable"))
        return {"available": False, "value": None, "display": "unavailable", "reason": reason}
    return {
        "available": True,
        "family": str(top_family),
        "top_frequency": float(top_frequency),
        "display": f"{top_family} top {100.0 * top_frequency:.0f}%",
    }


def _empirical_value(
    empirical_metrics: dict[str, Any],
    key: str,
    unit: str,
    *nested_path: str,
) -> dict[str, object]:
    if not empirical_metrics.get("available"):
        return _display_unavailable()
    if nested_path:
        value = _nested_number(empirical_metrics, *nested_path)
    else:
        value = _finite_float(empirical_metrics.get(key))
    return _display_value(value, unit)


def _trial_spread(empirical_report: dict[str, Any] | None) -> dict[str, object]:
    report = empirical_report or {}
    trials = report.get("trials")
    if not isinstance(trials, dict) or not trials:
        return {
            "available": False,
            "reason": "source experiment_report.json with per-trial metrics unavailable",
        }

    fields = (
        "sample_count",
        "duration_s",
        "finite_gaze_fraction",
        "sampling_rate_hz",
        "sampling_interval_cv",
        "gaze_dispersion_deg",
        "drift_slope_deg_s",
        "pupil_valid_fraction",
        "pupil_dynamic_range",
        "n_saccades",
        "n_fixations",
    )
    rows: list[dict[str, object]] = []
    metrics: dict[str, dict[str, float | int]] = {}
    for trial_id, raw in trials.items():
        trial = _mapping(raw)
        item: dict[str, object] = {"id": str(trial_id)}
        for field in fields:
            value = _finite_float(trial.get(field))
            if value is not None:
                item[field] = float(value)
        rows.append(item)

    for field in fields:
        values = [value for row in rows if (value := _finite_float(row.get(field))) is not None]
        if not values:
            continue
        metrics[field] = {
            "n": len(values),
            "min": float(min(values)),
            "median": float(median(values)),
            "max": float(max(values)),
            "mean": float(fmean(values)),
        }
        if len(values) > 1:
            mean_value = fmean(values)
            variance = fmean([(value - mean_value) ** 2 for value in values])
            metrics[field]["std"] = float(math.sqrt(variance))
        else:
            metrics[field]["std"] = 0.0

    return {
        "available": True,
        "trial_count": len(rows),
        "trials": rows,
        "metrics": metrics,
        "interpretation": (
            "Per-trial spread describes one local derived session; it is not "
            "population variability or reference-device accuracy."
        ),
    }


def _metric_row(
    *,
    row_id: str,
    label: str,
    empirical: dict[str, object],
    synthetic: dict[str, object] | None = None,
    noise_model: dict[str, object] | None = None,
    statistical_model: dict[str, object] | None = None,
    comparability: str,
    status: str,
    interpretation: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": row_id,
        "label": label,
        "empirical": empirical,
        "comparability": comparability,
        "status": status,
        "interpretation": interpretation,
    }
    if synthetic is not None:
        row["synthetic"] = synthetic
    if noise_model is not None:
        row["noise_model"] = noise_model
    if statistical_model is not None:
        row["statistical_model"] = statistical_model
    return row


def _metric_rows(
    *,
    empirical_metrics: dict[str, Any],
    synthetic_validation: dict[str, Any],
    noise_metrics: dict[str, Any],
    statistical_diagnostics: dict[str, Any],
) -> list[dict[str, object]]:
    finite_available, finite_display, finite_stress = _clean_stress_fraction_display(
        synthetic_validation, "gaze_finite_fraction"
    )
    pupil_available, pupil_display, _ = _clean_stress_fraction_display(
        synthetic_validation, "pupil_valid_fraction"
    )
    f1_clean = _domain_mean(synthetic_validation, "clean_desktop", "saccade_f1")
    f1_values = _domain_metric_means(synthetic_validation, "saccade_f1")
    amplitude_clean = _domain_mean(synthetic_validation, "clean_desktop", "amplitude_mae_deg")
    amplitude_stress = _domain_metric_means(synthetic_validation, "amplitude_mae_deg")
    gaze_rms_lo, gaze_rms_hi = _noise_records_range(noise_metrics, "gaze_rms_deg_mean")
    f1_threshold = _noise_threshold(noise_metrics, "saccade_f1_0_8")
    gaze_threshold = _noise_threshold(noise_metrics, "gaze_rms_2deg")
    bootstrap_model = _bootstrap_model_display(statistical_diagnostics)

    return [
        _metric_row(
            row_id="finite_gaze_fraction",
            label="finite gaze fraction",
            empirical=_empirical_value(empirical_metrics, "finite_gaze_fraction", "fraction"),
            synthetic={
                "available": finite_available,
                "value": finite_stress,
                "display": finite_display,
                "source": "synthetic domain summaries",
            },
            comparability="same_unit_contextual",
            status="observed local scale",
            interpretation=(
                "The pilot and synthetic domains use the same finite-gaze fraction; "
                "the clean N=1 session contextualizes clean-session examples without "
                "replacing dropout or low-light stress domains."
            ),
        ),
        _metric_row(
            row_id="pupil_valid_fraction",
            label="valid pupil fraction",
            empirical=_empirical_value(empirical_metrics, "pupil_valid_fraction", "fraction"),
            synthetic={
                "available": pupil_available,
                "value": None,
                "display": pupil_display,
                "source": "synthetic domain summaries",
            },
            comparability="same_unit_contextual",
            status="observed local scale",
            interpretation=(
                "The local run shows whether the current capture path produced a "
                "usable pupil proxy; synthetic dropout domains remain stress tests."
            ),
        ),
        _metric_row(
            row_id="sampling_rate_hz",
            label="sampling rate",
            empirical=_empirical_value(empirical_metrics, "sampling_rate_hz", "hz"),
            synthetic={
                "available": False,
                "value": None,
                "display": "synthetic sessions can be generated at chosen rates",
                "source": "synthetic generator parameter",
            },
            comparability="configuration_contextual",
            status="observed local scale",
            interpretation=(
                "A webcam-rate local session contextualizes live-style tests and "
                "timestamp-jitter assumptions; it is not a hardware benchmark."
            ),
        ),
        _metric_row(
            row_id="sampling_interval_cv",
            label="sampling interval CV",
            empirical=_empirical_value(empirical_metrics, "sampling_interval_cv", "cv"),
            synthetic={
                "available": False,
                "value": None,
                "display": "timestamp-jitter stress domain, if configured",
                "source": "session diagnostics",
            },
            comparability="configuration_contextual",
            status="observed local scale",
            interpretation=(
                "The coefficient of variation is a session-timing diagnostic used "
                "to sanity-check live-style jitter magnitudes."
            ),
        ),
        _metric_row(
            row_id="max_drift_deg_s",
            label="maximum drift slope",
            empirical=_empirical_value(empirical_metrics, "max_drift_deg_s", "deg_s"),
            synthetic={
                "available": True,
                "value": _domain_mean(synthetic_validation, "head_drift", "saccade_f1"),
                "display": "head-drift domain is a stress test",
                "source": "synthetic head_drift domain",
            },
            comparability="stress_domain_only",
            status="stress-domain only",
            interpretation=(
                "The pilot supplies one low-drift operating scale; synthetic "
                "head-drift cases deliberately probe harder conditions."
            ),
        ),
        _metric_row(
            row_id="heldout_target_rms_deg",
            label="held-out target RMS",
            empirical=_empirical_value(
                empirical_metrics,
                "heldout_target_rms_deg",
                "deg",
                "heldout_target_error",
                "rms_error_deg",
            ),
            noise_model={
                "available": gaze_rms_lo is not None and gaze_rms_hi is not None,
                "value": gaze_rms_hi,
                "display": (
                    f"idealized landmark-noise RMS {gaze_rms_lo:.2f}-{gaze_rms_hi:.2f} deg"
                    if gaze_rms_lo is not None and gaze_rms_hi is not None
                    else "unavailable"
                ),
                "source": "noise sweep over synthetic landmarks",
                "threshold_display": (
                    f"2 deg at {float(gaze_threshold['pixels_at_640']):.2f} px"
                    if "pixels_at_640" in gaze_threshold
                    else "unavailable"
                ),
            },
            comparability="not_directly_comparable",
            status="not directly comparable",
            interpretation=(
                "Held-out screen-target residual includes prompt following, "
                "calibration, webcam session effects, and user behavior; it is not "
                "iid landmark noise or gaze accuracy."
            ),
        ),
        _metric_row(
            row_id="target_acquisition_latency_s",
            label="target acquisition latency",
            empirical=_empirical_value(
                empirical_metrics,
                "target_acquisition_latency_s",
                "s",
                "target_acquisition_latency_s",
                "median_latency_s",
            ),
            synthetic={
                "available": False,
                "value": None,
                "display": "not a synthetic saccade-latency oracle",
                "source": "live UI protocol timing",
            },
            comparability="not_directly_comparable",
            status="not directly comparable",
            interpretation=(
                "UI/capture target-acquisition timing measures the local protocol "
                "loop, not physiological saccade latency."
            ),
        ),
        _metric_row(
            row_id="saccade_detection_stress",
            label="saccade detection stress",
            empirical={"available": False, "value": None, "display": "unavailable"},
            synthetic={
                "available": f1_clean is not None and bool(f1_values),
                "value": min(f1_values) if f1_values else None,
                "display": (
                    f"clean F1 {_percent(f1_clean)}; stress min {_percent(min(f1_values))}"
                    if f1_clean is not None and f1_values
                    else "unavailable"
                ),
                "source": "synthetic truth domains",
            },
            noise_model={
                "available": "pixels_at_640" in f1_threshold,
                "value": _finite_float(f1_threshold.get("pixels_at_640")),
                "display": (
                    f"F1=0.8 edge at {float(f1_threshold['pixels_at_640']):.2f} px"
                    if "pixels_at_640" in f1_threshold
                    else "unavailable"
                ),
                "source": "noise sweep sidecar",
            },
            comparability="synthetic_truth_only",
            status="stress-domain only",
            interpretation=(
                "Synthetic truth and oracle comparisons, not the N=1 pilot, test "
                "event-detector correctness under known labels."
            ),
        ),
        _metric_row(
            row_id="amplitude_recovery_mae_deg",
            label="amplitude recovery MAE",
            empirical={"available": False, "value": None, "display": "unavailable"},
            synthetic={
                "available": amplitude_clean is not None and bool(amplitude_stress),
                "value": max(amplitude_stress) if amplitude_stress else None,
                "display": (
                    f"clean {amplitude_clean:.2f} deg; stress max {max(amplitude_stress):.2f} deg"
                    if amplitude_clean is not None and amplitude_stress
                    else "unavailable"
                ),
                "source": "synthetic truth domains",
            },
            comparability="synthetic_truth_only",
            status="stress-domain only",
            interpretation=(
                "Amplitude MAE is defined only where synthetic truth labels exist; "
                "the pilot supplies operating-scale context, not truth labels."
            ),
        ),
        _metric_row(
            row_id="model_selection_stability",
            label="amplitude model stability",
            empirical={"available": False, "value": None, "display": "unavailable"},
            statistical_model=bootstrap_model,
            comparability="not_directly_comparable",
            status="descriptive model check",
            interpretation=(
                "Bootstrap winner frequency describes ranking stability in the "
                "synthetic diagnostic report, not a biological population model."
            ),
        ),
    ]


def build_synthetic_empirical_range_bridge(
    *,
    empirical_metrics: dict[str, Any],
    synthetic_validation: dict[str, Any],
    noise_metrics: dict[str, Any],
    statistical_diagnostics: dict[str, Any],
    empirical_report: dict[str, Any] | None = None,
    sources: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Build a JSON-friendly bridge from generated evidence artifacts.

    The returned payload deliberately separates observed local-session values
    from synthetic truth, idealized noise sweeps, and descriptive statistical
    diagnostics. It should never be read as device validation.
    """
    empirical = _mapping(empirical_metrics)
    source_report = _validate_repo_relative_path(
        str(empirical.get("source_report")) if empirical.get("source_report") else None,
        field="source_report",
    )
    merged_sources: dict[str, str | None] = dict(DEFAULT_RANGE_BRIDGE_SOURCES)
    if sources:
        merged_sources.update(sources)
    if source_report:
        merged_sources["source_report"] = source_report
    for key, value in list(merged_sources.items()):
        merged_sources[key] = _validate_repo_relative_path(value, field=key)

    rows = _metric_rows(
        empirical_metrics=empirical,
        synthetic_validation=_mapping(synthetic_validation),
        noise_metrics=_mapping(noise_metrics),
        statistical_diagnostics=_mapping(statistical_diagnostics),
    )
    row_statuses = [str(row.get("status", "")) for row in rows]
    trial_spread = _trial_spread(empirical_report)
    payload: dict[str, object] = {
        "kind": "itrace_synthetic_empirical_range_bridge",
        "version": 1,
        "truth_boundary": (
            "N=1 derived-session diagnostics contextualize one local operating scale; "
            "synthetic truth and reference-oracle tests remain the correctness evidence; "
            "this is not reference-device validation."
        ),
        "storage_boundary": (
            "Derived gaze/pupil/capture records and summary reports only; no raw eye video "
            "or persisted eye-crop images are required or persisted by this bridge."
        ),
        "empirical_available": bool(empirical.get("available")),
        "source_report_available": empirical_report is not None,
        "sources": merged_sources,
        "metrics": rows,
        "trial_spread": trial_spread,
        "summary": {
            "metric_count": len(rows),
            "observed_local_scale_count": row_statuses.count("observed local scale"),
            "stress_domain_count": row_statuses.count("stress-domain only"),
            "not_directly_comparable_count": row_statuses.count("not directly comparable"),
        },
    }
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _mapping(json.loads(path.read_text(encoding="utf-8")))


def load_synthetic_empirical_range_bridge(root: str | Path = ".") -> dict[str, object]:
    """Load standard repo artifacts and build the range-bridge payload."""
    repo_root = Path(root)
    empirical_metrics = _load_json(repo_root / DEFAULT_RANGE_BRIDGE_SOURCES["empirical_metrics"])
    synthetic_validation = _load_json(
        repo_root / DEFAULT_RANGE_BRIDGE_SOURCES["synthetic_validation"]
    )
    noise_metrics = _load_json(repo_root / DEFAULT_RANGE_BRIDGE_SOURCES["noise_metrics"])
    statistical_diagnostics = _load_json(
        repo_root / DEFAULT_RANGE_BRIDGE_SOURCES["statistical_diagnostics"]
    )

    source_report = _validate_repo_relative_path(
        str(empirical_metrics.get("source_report"))
        if empirical_metrics.get("source_report")
        else None,
        field="source_report",
    )
    empirical_report = None
    if source_report:
        report_path = repo_root / source_report
        if report_path.exists():
            empirical_report = _load_json(report_path)

    return build_synthetic_empirical_range_bridge(
        empirical_metrics=empirical_metrics,
        synthetic_validation=synthetic_validation,
        noise_metrics=noise_metrics,
        statistical_diagnostics=statistical_diagnostics,
        empirical_report=empirical_report,
        sources=DEFAULT_RANGE_BRIDGE_SOURCES,
    )
