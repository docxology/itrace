"""Render the standard iTrace figure gallery."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from .. import pipeline, saccades
from ..config import AnalysisConfig, DetectionConfig
from ..stats.timeseries import saccade_rate_series
from ..synthetic import SyntheticSessionSpec, synthetic_session
from .dashboard import render_dashboard, session_dashboard
from .distributions import (
    figure_amplitude_histogram,
    figure_duration_histogram,
    figure_main_sequence,
)
from .quality import (
    figure_calibration_residuals,
    figure_dropout_raster,
    figure_pupil_velocity,
    figure_sampling_intervals,
)
from .scanpath import figure_microsaccades, figure_scanpath
from .spatial import figure_aoi, figure_fixation_heatmap, figure_gaze_density
from .timeline import figure_event_raster, figure_pupil_psd, figure_rate
from .palette import apply_house_style
from .traces import figure_pupil_trace, figure_velocity_trace


def _save(fig: Figure, path: Path, *, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _default_aois() -> list[dict[str, object]]:
    return [
        {"name": "left", "x": -20.0, "y": -20.0, "w": 20.0, "h": 40.0},
        {"name": "right", "x": 0.0, "y": -20.0, "w": 20.0, "h": 40.0},
    ]


def _render_replay_animation(out_dir: Path, seed: int, *, n_frames: int = 80) -> Path:
    gaze, pupil, _truth = synthetic_session(SyntheticSessionSpec(seed=seed))
    fig, (ax_xy, ax_pupil) = plt.subplots(1, 2, figsize=(10, 4))
    ax_xy.plot(gaze.x, gaze.y, color="#56B4E9", lw=1.0)
    ax_xy.set_title("synthetic gaze replay")
    ax_xy.set_xlabel("horizontal gaze (deg)")
    ax_xy.set_ylabel("vertical gaze (deg, screen)")
    ax_xy.invert_yaxis()
    (dot,) = ax_xy.plot([], [], "o", color="#D55E00", ms=7)

    ax_pupil.plot(pupil.t, pupil.size, color="#009E73", lw=1.2)
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

    paths: list[Path] = [
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
