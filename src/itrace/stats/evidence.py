"""Interpretation ledger for statistical evidence artifacts.

The diagnostics payloads contain numerical results. This module adds the
publication-facing interpretation layer: what each statistic estimates, where it
comes from, and what claim it explicitly does *not* support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STATISTICAL_EVIDENCE_SOURCES: dict[str, str] = {
    "statistical_diagnostics": "output/figures/statistical_diagnostics.json",
    "synthetic_empirical_range_bridge": "output/figures/synthetic_empirical_range_bridge.json",
    "noise_metrics": "output/figures/noise_metrics.json",
    "empirical_metrics": "docs/empirical_pilot_metrics.json",
}
"""Repo-relative evidence files consumed by the standard interpretation ledger."""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    number = _float(value)
    if number is None:
        return "unavailable"
    return f"{number:.{digits}f}{suffix}"


def _available_display(available: bool, display: str, reason: str = "unavailable") -> str:
    return display if available else reason


def _first_family(model: dict[str, Any]) -> dict[str, Any]:
    families = _list(model.get("families"))
    first = families[0] if families else {}
    return _mapping(first)


def _metric_row(
    *,
    row_id: str,
    label: str,
    source_artifact: str,
    source_path: str,
    reported_value: str,
    estimand: str,
    interpretation: str,
    does_not_prove: str,
    scholarship_basis: list[str],
    evidence_class: str,
    available: bool = True,
) -> dict[str, object]:
    return {
        "id": row_id,
        "label": label,
        "source_artifact": source_artifact,
        "source_path": source_path,
        "reported_value": reported_value,
        "estimand": estimand,
        "interpretation": interpretation,
        "does_not_prove": does_not_prove,
        "scholarship_basis": scholarship_basis,
        "evidence_class": evidence_class,
        "available": bool(available),
    }


def _shape_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    shape = _mapping(statistical_diagnostics.get("amplitude_shape_summary"))
    available = bool(shape.get("available"))
    display = _available_display(
        available,
        (
            f"median {_fmt(shape.get('median'), suffix=' deg')}; "
            f"IQR {_fmt(shape.get('iqr'), suffix=' deg')}; "
            f"outliers {int(_float(shape.get('iqr_outlier_count')) or 0)}/"
            f"{int(_float(shape.get('n')) or 0)}"
        ),
        str(shape.get("reason", "shape unavailable")),
    )
    return _metric_row(
        row_id="robust_amplitude_shape",
        label="Robust amplitude shape",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="finite-sample distribution shape of detected synthetic saccade amplitudes",
        interpretation=(
            "Quartiles, Bowley skew, and outside values describe amplitude shape before "
            "parametric fits."
        ),
        does_not_prove="a population amplitude law or real-eye physiology",
        scholarship_basis=["robust quartiles", "IQR outside-value rule", "percentile bootstrap"],
        evidence_class="descriptive_statistic",
        available=available,
    )


def _model_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    model = _mapping(statistical_diagnostics.get("amplitude_model_comparison"))
    family = _first_family(model)
    bootstrap = _mapping(model.get("model_selection_bootstrap"))
    available = bool(model.get("available")) and bool(family)
    top_frequency = _float(bootstrap.get("top_frequency"))
    top_family = str(bootstrap.get("top_family", model.get("best_family", "model")))
    display = _available_display(
        available,
        (
            f"best {model.get('best_family', family.get('family', 'model'))}; "
            f"w(AIC) {_fmt(family.get('akaike_weight'))}; "
            f"boot {top_family} {100.0 * top_frequency:.0f}%"
        )
        if top_frequency is not None
        else (
            f"best {model.get('best_family', family.get('family', 'model'))}; "
            f"w(AIC) {_fmt(family.get('akaike_weight'))}"
        ),
        str(model.get("reason", "model comparison unavailable")),
    )
    return _metric_row(
        row_id="relative_model_comparison",
        label="Relative model comparison",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="relative fit ranking among candidate amplitude distributions",
        interpretation=(
            "Information criteria and bootstrap winner frequency describe ranking stability."
        ),
        does_not_prove="true family membership or posterior probability",
        scholarship_basis=["AIC", "AICc", "BIC", "Kolmogorov-Smirnov statistic"],
        evidence_class="relative_model_diagnostic",
        available=available,
    )


def _cdf_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    quantile = _mapping(statistical_diagnostics.get("amplitude_quantile_diagnostics"))
    cdf = _mapping(statistical_diagnostics.get("amplitude_cdf_diagnostics"))
    available = bool(quantile.get("available")) and bool(cdf.get("available"))
    display = _available_display(
        available,
        (
            f"QQ RMSE {_fmt(quantile.get('residual_rmse_deg'), suffix=' deg')}; "
            f"P-P max |res| {_fmt(cdf.get('max_abs_cdf_residual'))}; "
            f"DKW +/- {_fmt(cdf.get('dkw_epsilon'))}"
        ),
        str(quantile.get("reason") or cdf.get("reason") or "CDF diagnostics unavailable"),
    )
    return _metric_row(
        row_id="distribution_residual_checks",
        label="Distribution residual checks",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="empirical-versus-fitted amplitude quantile/CDF residual scale",
        interpretation=("QQ/P-P residuals and a DKW/Massart band show selected-family fit gaps."),
        does_not_prove="a formal model-selection theorem or a device-accuracy result",
        scholarship_basis=["QQ plot", "P-P/CDF residuals", "DKW/Massart band"],
        evidence_class="model_residual_diagnostic",
        available=available,
    )


def _main_sequence_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    exponent = _mapping(statistical_diagnostics.get("main_sequence_exponent"))
    available = bool(exponent.get("available"))
    display = _available_display(
        available,
        (
            f"b {_fmt(exponent.get('estimate'))} "
            f"[{_fmt(exponent.get('ci_low'))}, {_fmt(exponent.get('ci_high'))}]"
        ),
        str(exponent.get("reason", "main-sequence interval unavailable")),
    )
    return _metric_row(
        row_id="main_sequence_uncertainty",
        label="Main-sequence uncertainty",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="finite-sample uncertainty for the detected synthetic main-sequence exponent",
        interpretation="Seeded bootstrap interval for this report's exponent variability.",
        does_not_prove="human physiological population bounds or camera timing precision",
        scholarship_basis=["seeded bootstrap", "main-sequence power-law diagnostic"],
        evidence_class="bootstrap_uncertainty",
        available=available,
    )


def _spatial_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    spatial = _mapping(statistical_diagnostics.get("spatial_stability"))
    available = bool(spatial)
    display = _available_display(
        available,
        (
            f"disp {_fmt(spatial.get('gaze_dispersion'), suffix=' deg')}; "
            f"BCEA {_fmt(spatial.get('bcea'), suffix=' deg2')}; "
            f"entropy {_fmt(spatial.get('fixation_position_entropy'), suffix=' bits')}"
        ),
        "spatial descriptors unavailable",
    )
    return _metric_row(
        row_id="spatial_stability_descriptors",
        label="Spatial stability descriptors",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="within-report fixation/gaze spread and entropy descriptors",
        interpretation="Dispersion, hull/BCEA, and entropy summarize within-report spread.",
        does_not_prove="absolute gaze accuracy or participant-level stability",
        scholarship_basis=["BCEA-style spread", "entropy descriptor"],
        evidence_class="descriptive_statistic",
        available=available,
    )


def _transition_row(statistical_diagnostics: dict[str, Any]) -> dict[str, object]:
    transition = _mapping(statistical_diagnostics.get("transition_matrix"))
    symbols = _list(transition.get("symbols"))
    available = bool(symbols)
    display = _available_display(
        available,
        f"{len(symbols)} symbols; scanpath {transition.get('scanpath', '')}",
        "transition matrix unavailable",
    )
    return _metric_row(
        row_id="scanpath_transition_structure",
        label="Scanpath transition structure",
        source_artifact="statistical_diagnostics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"],
        reported_value=display,
        estimand="first-order direction-symbol transition structure in one report",
        interpretation="Transition matrix summarizes local scanpath grammar for QA.",
        does_not_prove="identity, biometrics, cognition, or anomaly labels",
        scholarship_basis=[
            "Levenshtein/string encoding",
            "transition matrix",
            "cosine-style vector comparison",
        ],
        evidence_class="scanpath_descriptor",
        available=available,
    )


def _noise_row(noise_metrics: dict[str, Any]) -> dict[str, object]:
    thresholds = _mapping(noise_metrics.get("thresholds"))
    f1_edge = _mapping(thresholds.get("saccade_f1_0_8"))
    rms_edge = _mapping(thresholds.get("gaze_rms_2deg"))
    available = "pixels_at_640" in f1_edge or "pixels_at_640" in rms_edge
    f1_px = _float(f1_edge.get("pixels_at_640"))
    rms_px = _float(rms_edge.get("pixels_at_640"))
    display = _available_display(
        available,
        (
            f"F1 edge {f1_px:.2f} px; RMS edge {rms_px:.2f} px"
            if f1_px is not None and rms_px is not None
            else "partial threshold evidence"
        ),
        "noise thresholds unavailable",
    )
    return _metric_row(
        row_id="idealized_noise_sensitivity",
        label="Idealized noise sensitivity",
        source_artifact="noise_metrics",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["noise_metrics"],
        reported_value=display,
        estimand="closed-loop estimator sensitivity to seeded synthetic landmark jitter",
        interpretation=(
            "Threshold crossings show where idealized landmark jitter degrades recovery."
        ),
        does_not_prove="webcam optics, MediaPipe bias, lighting, or hardware noise",
        scholarship_basis=["Monte-Carlo perturbation", "bounded bootstrap interval"],
        evidence_class="synthetic_stress_diagnostic",
        available=available,
    )


def _range_bridge_row(range_bridge: dict[str, Any]) -> dict[str, object]:
    summary = _mapping(range_bridge.get("summary"))
    available = range_bridge.get("kind") == "itrace_synthetic_empirical_range_bridge"
    display = _available_display(
        available,
        (
            f"{int(_float(summary.get('metric_count')) or 0)} rows; "
            f"{int(_float(summary.get('stress_domain_count')) or 0)} stress; "
            f"{int(_float(summary.get('not_directly_comparable_count')) or 0)} non-comparable"
        ),
        "range bridge unavailable",
    )
    return _metric_row(
        row_id="synthetic_empirical_context",
        label="Synthetic/empirical context",
        source_artifact="synthetic_empirical_range_bridge",
        source_path=DEFAULT_STATISTICAL_EVIDENCE_SOURCES["synthetic_empirical_range_bridge"],
        reported_value=display,
        estimand="relationship between one local N=1 session scale and synthetic/stress variables",
        interpretation="Local-scale, stress-only, and non-comparable rows remain separated.",
        does_not_prove="device validation, device performance, or universal webcam accuracy",
        scholarship_basis=["data-quality ledger", "truth-boundary taxonomy"],
        evidence_class="interpretation_boundary",
        available=available,
    )


def build_statistical_interpretation_ledger(
    *,
    statistical_diagnostics: dict[str, Any],
    range_bridge: dict[str, Any],
    noise_metrics: dict[str, Any],
    empirical_metrics: dict[str, Any],
    sources: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a JSON-friendly ledger for interpreting statistical diagnostics."""
    source_map = dict(DEFAULT_STATISTICAL_EVIDENCE_SOURCES)
    if sources is not None:
        source_map.update(sources)
    rows = [
        _shape_row(statistical_diagnostics),
        _model_row(statistical_diagnostics),
        _cdf_row(statistical_diagnostics),
        _main_sequence_row(statistical_diagnostics),
        _spatial_row(statistical_diagnostics),
        _transition_row(statistical_diagnostics),
        _noise_row(noise_metrics),
        _range_bridge_row(range_bridge),
    ]
    available = [row for row in rows if row["available"]]
    return {
        "kind": "itrace_statistical_interpretation_ledger",
        "version": 1,
        "truth_boundary": (
            "Rows interpret generated statistics and visual evidence; they do not "
            "convert synthetic or N=1 diagnostics into population physiology or "
            "reference-device validation."
        ),
        "source_artifacts": source_map,
        "empirical_available": bool(empirical_metrics.get("available")),
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "available_count": len(available),
            "evidence_classes": sorted({str(row["evidence_class"]) for row in rows}),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_statistical_interpretation_ledger(root: str | Path = ".") -> dict[str, object]:
    """Load standard repo artifacts and build the interpretation ledger."""
    repo_root = Path(root)
    return build_statistical_interpretation_ledger(
        statistical_diagnostics=_load_json(
            repo_root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["statistical_diagnostics"]
        ),
        range_bridge=_load_json(
            repo_root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["synthetic_empirical_range_bridge"]
        ),
        noise_metrics=_load_json(repo_root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["noise_metrics"]),
        empirical_metrics=_load_json(
            repo_root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["empirical_metrics"]
        ),
        sources=DEFAULT_STATISTICAL_EVIDENCE_SOURCES,
    )
