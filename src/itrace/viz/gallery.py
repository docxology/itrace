"""Render the standard iTrace figure gallery."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from .. import pipeline, saccades
from ..config import AnalysisConfig, DetectionConfig
from ..stats.diagnostics import session_statistical_diagnostics
from ..stats.evidence import (
    DEFAULT_STATISTICAL_EVIDENCE_SOURCES,
    build_statistical_interpretation_ledger,
)
from ..stats.range_bridge import (
    DEFAULT_RANGE_BRIDGE_SOURCES,
    build_synthetic_empirical_range_bridge,
)
from ..stats.timeseries import saccade_rate_series
from ..synthetic import SyntheticSessionSpec, synthetic_session
from .dashboard import render_dashboard, session_dashboard
from .distributions import (
    figure_amplitude_histogram,
    figure_duration_histogram,
    figure_main_sequence,
)
from .evidence import figure_statistical_interpretation_ledger
from .palette import WONG, apply_house_style
from .quality import (
    figure_calibration_residuals,
    figure_dropout_raster,
    figure_pupil_velocity,
    figure_sampling_intervals,
)
from .range_bridge import figure_synthetic_empirical_range_bridge
from .scanpath import figure_microsaccades, figure_scanpath
from .spatial import figure_aoi, figure_fixation_heatmap, figure_gaze_density
from .statistics import figure_statistical_diagnostics
from .timeline import figure_event_raster, figure_pupil_psd, figure_rate
from .traces import figure_pupil_trace, figure_velocity_trace


def _save(fig: Figure, path: Path, *, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _range_bridge_payload(statistical_payload: dict[str, object]) -> dict[str, object]:
    root = Path.cwd()
    empirical_metrics = _load_json(root / DEFAULT_RANGE_BRIDGE_SOURCES["empirical_metrics"])
    synthetic_validation = _load_json(root / DEFAULT_RANGE_BRIDGE_SOURCES["synthetic_validation"])
    noise_metrics = _load_json(root / DEFAULT_RANGE_BRIDGE_SOURCES["noise_metrics"])
    empirical_report = None
    source_report = empirical_metrics.get("source_report")
    if isinstance(source_report, str) and source_report:
        report_path = root / source_report
        if report_path.exists():
            empirical_report = _load_json(report_path)
    return build_synthetic_empirical_range_bridge(
        empirical_metrics=empirical_metrics,
        synthetic_validation=synthetic_validation,
        noise_metrics=noise_metrics,
        statistical_diagnostics=statistical_payload,
        empirical_report=empirical_report,
        sources=DEFAULT_RANGE_BRIDGE_SOURCES,
    )


def _statistical_evidence_payload(
    statistical_payload: dict[str, object],
    range_bridge_payload: dict[str, object],
) -> dict[str, object]:
    root = Path.cwd()
    return build_statistical_interpretation_ledger(
        statistical_diagnostics=statistical_payload,
        range_bridge=range_bridge_payload,
        noise_metrics=_load_json(root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["noise_metrics"]),
        empirical_metrics=_load_json(
            root / DEFAULT_STATISTICAL_EVIDENCE_SOURCES["empirical_metrics"]
        ),
        sources=DEFAULT_STATISTICAL_EVIDENCE_SOURCES,
    )


def _default_aois() -> list[dict[str, object]]:
    return [
        {"name": "left", "x": -20.0, "y": -20.0, "w": 20.0, "h": 40.0},
        {"name": "right", "x": 0.0, "y": -20.0, "w": 20.0, "h": 40.0},
    ]


def _render_replay_animation(out_dir: Path, seed: int, *, n_frames: int = 80) -> Path:
    gaze, pupil, _truth = synthetic_session(SyntheticSessionSpec(seed=seed))
    fig, (ax_xy, ax_pupil) = plt.subplots(1, 2, figsize=(10, 4))
    ax_xy.plot(gaze.x, gaze.y, color=WONG[5], lw=1.0)
    ax_xy.set_title("synthetic gaze replay")
    ax_xy.set_xlabel("horizontal gaze (deg)")
    ax_xy.set_ylabel("vertical gaze (deg, screen)")
    ax_xy.invert_yaxis()
    (dot,) = ax_xy.plot([], [], "o", color=WONG[3], ms=7)

    ax_pupil.plot(pupil.t, pupil.size, color=WONG[2], lw=1.2)
    ax_pupil.set_title("pupil proxy")
    ax_pupil.set_xlabel("time (s)")
    cursor = ax_pupil.axvline(float(pupil.t[0]), color="0.2", lw=1.0)
    fig.tight_layout()

    step = max(len(gaze) // n_frames, 1)
    frames = list(range(0, len(gaze), step))

    def update(idx: int) -> tuple[Artist, Artist]:
        dot.set_data([gaze.x[idx]], [gaze.y[idx]])
        cursor.set_xdata([gaze.t[idx], gaze.t[idx]])
        return dot, cursor

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "synthetic_replay.gif"
    anim.save(path, writer=PillowWriter(fps=15))
    plt.close(fig)
    return path


def render_gallery(
    out_dir: str | Path,
    *,
    seed: int = 0,
    animations: bool = False,
    dpi: int = 300,
) -> list[Path]:
    """Render the standard static gallery and optional animation."""
    apply_house_style()  # readable font floor + clean print defaults for every figure
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gaze, pupil, _truth = synthetic_session(SyntheticSessionSpec(seed=seed))
    cfg = AnalysisConfig(detection=DetectionConfig(method="adaptive_ivt", include_pso=True))
    report = pipeline.analyze_session(gaze, pupil, config=cfg)
    props = saccades.saccade_properties(report.saccades)
    statistical_payload = session_statistical_diagnostics(report)
    range_bridge_payload = _range_bridge_payload(statistical_payload)
    evidence_payload = _statistical_evidence_payload(statistical_payload, range_bridge_payload)

    paths: list[Path] = [
        _write_json(statistical_payload, out / "statistical_diagnostics.json"),
        _write_json(range_bridge_payload, out / "synthetic_empirical_range_bridge.json"),
        _write_json(evidence_payload, out / "statistical_interpretation_ledger.json"),
        render_dashboard(out / "session_dashboard.png", seed=seed),
        _save(
            session_dashboard(report, gaze, pupil), out / "session_dashboard_synthetic.png", dpi=dpi
        ),
        _save(figure_velocity_trace(gaze), out / "velocity_trace.png", dpi=dpi),
        _save(figure_pupil_trace(pupil), out / "pupil_trace.png", dpi=dpi),
        _save(figure_scanpath(report), out / "scanpath.png", dpi=dpi),
        _save(figure_fixation_heatmap(report.fixations), out / "fixation_heatmap.png", dpi=dpi),
        _save(figure_gaze_density(gaze.x, gaze.y), out / "gaze_density.png", dpi=dpi),
        _save(figure_aoi(report.fixations, _default_aois()), out / "aoi_dwell.png", dpi=dpi),
        _save(figure_event_raster(report), out / "event_raster.png", dpi=dpi),
        _save(figure_pupil_psd(pupil), out / "pupil_psd.png", dpi=dpi),
        _save(figure_dropout_raster(gaze), out / "dropout_raster.png", dpi=dpi),
        _save(figure_sampling_intervals(gaze), out / "sampling_intervals.png", dpi=dpi),
        _save(
            figure_calibration_residuals([0.0, 0.05, 0.02, 0.04]),
            out / "calibration_residuals.png",
            dpi=dpi,
        ),
        _save(figure_pupil_velocity(pupil), out / "pupil_velocity.png", dpi=dpi),
        _save(
            figure_amplitude_histogram(props["amplitude_deg"]),
            out / "amplitude_histogram.png",
            dpi=dpi,
        ),
        _save(
            figure_duration_histogram(props["duration_s"]),
            out / "duration_histogram.png",
            dpi=dpi,
        ),
        _save(
            figure_main_sequence(props["amplitude_deg"], props["peak_velocity_deg_s"]),
            out / "main_sequence_diagnostics.png",
            dpi=dpi,
        ),
        _save(
            figure_statistical_diagnostics(report, payload=statistical_payload),
            out / "statistical_diagnostics.png",
            dpi=dpi,
        ),
        _save(
            figure_synthetic_empirical_range_bridge(range_bridge_payload),
            out / "synthetic_empirical_range_bridge.png",
            dpi=dpi,
        ),
        _save(
            figure_statistical_interpretation_ledger(evidence_payload),
            out / "statistical_interpretation_ledger.png",
            dpi=dpi,
        ),
        _save(figure_microsaccades(report.microsaccades), out / "microsaccades.png", dpi=dpi),
    ]

    if report.duration_s > 0.0:
        times, rates = saccade_rate_series(report.saccades, report.duration_s, bin_s=0.5)
        paths.append(
            _save(figure_rate(times, rates, label="saccades/s"), out / "saccade_rate.png", dpi=dpi)
        )
    if animations:
        paths.append(_render_replay_animation(out, seed))
    return paths
