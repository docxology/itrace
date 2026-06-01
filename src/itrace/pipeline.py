"""End-to-end analysis pipeline composing the core modules.

Takes raw gaze (and optional pupil) streams and produces a single
:class:`~itrace.types.SessionReport`: fixations, saccades, microsaccades,
main-sequence parameters, an encoded scanpath, and pupil summary statistics.
"""

from __future__ import annotations

from contextlib import suppress

import numpy as np

from . import detection, encoding, mainsequence, pupil, saccades
from .calibration import robust_gaze_quality
from .config import AnalysisConfig, DetectionConfig, PupilConfig
from .pupilphase import Phase, PhaseDetector
from .types import Fixation, GazeStream, PupilStream, Saccade, SessionReport


def _resolve_analysis_config(
    config: AnalysisConfig | None,
    *,
    velocity_threshold_deg_s: float,
    long_saccade_deg: float,
    include_microsaccades: bool,
) -> AnalysisConfig:
    if config is not None:
        return config
    return AnalysisConfig(
        detection=DetectionConfig(
            velocity_threshold_deg_s=velocity_threshold_deg_s,
            long_saccade_deg=long_saccade_deg,
            include_microsaccades=include_microsaccades,
        )
    )


def _detect_events(
    stream: GazeStream, cfg: DetectionConfig
) -> tuple[list[Fixation], list[Saccade], float]:
    if cfg.method == "adaptive_ivt":
        threshold = detection.adaptive_ivt_threshold(
            stream,
            lambda_factor=cfg.adaptive_lambda_factor,
        )
        fixations, saccs = saccades.detect_ivt(
            stream,
            velocity_threshold_deg_s=threshold,
            min_saccade_duration_s=cfg.min_saccade_duration_s,
            merge_gap_s=cfg.merge_gap_s,
            min_inter_event_gap_s=cfg.min_inter_event_gap_s,
            max_saccade_duration_s=cfg.max_saccade_duration_s,
            reject_edge_events=cfg.reject_edge_events,
        )
    else:
        threshold = cfg.velocity_threshold_deg_s
        fixations, saccs = saccades.detect_ivt(
            stream,
            velocity_threshold_deg_s=cfg.velocity_threshold_deg_s,
            min_saccade_duration_s=cfg.min_saccade_duration_s,
            merge_gap_s=cfg.merge_gap_s,
            min_inter_event_gap_s=cfg.min_inter_event_gap_s,
            max_saccade_duration_s=cfg.max_saccade_duration_s,
            reject_edge_events=cfg.reject_edge_events,
        )
    return fixations, saccs, threshold


def _quality_summary(
    stream: GazeStream,
    cfg: DetectionConfig,
    threshold_deg_s: float,
    n_fixations: int,
    n_saccades: int,
    n_microsaccades: int,
    n_psos: int,
) -> dict[str, float]:
    duration = float(stream.t[-1] - stream.t[0]) if len(stream) else 0.0
    robust = robust_gaze_quality(stream)
    quality = {
        "finite_sample_fraction": robust["valid_sample_fraction"],
        "sampling_rate_hz": float(stream.sampling_rate_hz) if len(stream) >= 2 else 0.0,
        "detection_threshold_deg_s": float(threshold_deg_s),
        "adaptive_lambda_factor": float(cfg.adaptive_lambda_factor)
        if cfg.method == "adaptive_ivt"
        else 0.0,
        "fixation_rate_hz": float(n_fixations / duration) if duration > 0.0 else 0.0,
        "saccade_rate_hz": float(n_saccades / duration) if duration > 0.0 else 0.0,
        "microsaccade_rate_hz": float(n_microsaccades / duration) if duration > 0.0 else 0.0,
        "pso_rate_hz": float(n_psos / duration) if duration > 0.0 else 0.0,
    }
    quality.update(robust)
    return quality


def analyze_gaze(
    stream: GazeStream,
    *,
    velocity_threshold_deg_s: float = 30.0,
    long_saccade_deg: float = 5.0,
    include_microsaccades: bool = True,
    config: AnalysisConfig | None = None,
) -> SessionReport:
    """Run gaze -> events -> main-sequence -> scanpath encoding."""
    resolved = _resolve_analysis_config(
        config,
        velocity_threshold_deg_s=velocity_threshold_deg_s,
        long_saccade_deg=long_saccade_deg,
        include_microsaccades=include_microsaccades,
    )
    detect_cfg = resolved.detection
    fixations, saccs, threshold = _detect_events(stream, detect_cfg)
    micro = saccades.detect_microsaccades(stream) if detect_cfg.include_microsaccades else []
    pursuits = (
        detection.detect_smooth_pursuit(
            stream,
            min_velocity_deg_s=detect_cfg.smooth_pursuit_min_velocity_deg_s,
            max_velocity_deg_s=detect_cfg.smooth_pursuit_max_velocity_deg_s,
            min_duration_s=detect_cfg.smooth_pursuit_min_duration_s,
        )
        if detect_cfg.include_smooth_pursuit
        else []
    )
    psos = (
        detection.detect_pso_events(
            stream,
            saccs,
            window_s=detect_cfg.pso_window_s,
            peak_fraction=detect_cfg.pso_peak_fraction,
        )
        if detect_cfg.include_pso
        else []
    )

    main_seq: dict[str, float] = {}
    if len(saccs) >= 3:
        props = saccades.saccade_properties(saccs)
        try:
            main_seq = mainsequence.fit(props["amplitude_deg"], props["peak_velocity_deg_s"])
        except ValueError:
            main_seq = {}

    scanpath = encoding.encode_directions(saccs, long_threshold_deg=detect_cfg.long_saccade_deg)
    duration = float(stream.t[-1] - stream.t[0]) if len(stream) else 0.0
    return SessionReport(
        n_samples=len(stream),
        duration_s=duration,
        fixations=fixations,
        saccades=saccs,
        microsaccades=micro,
        smooth_pursuits=pursuits,
        psos=psos,
        scanpath=scanpath,
        main_sequence=main_seq,
        quality=_quality_summary(
            stream,
            detect_cfg,
            threshold,
            len(fixations),
            len(saccs),
            len(micro),
            len(psos),
        ),
        config=resolved.to_dict(),
    )


def analyze_pupil(stream: PupilStream, config: PupilConfig | None = None) -> dict[str, float]:
    """Deblink -> smooth -> summarise + phase-event counts."""
    cfg = config or PupilConfig()
    if not cfg.enabled:
        return {}
    min_valid = max(cfg.blink_threshold, 1e-6)
    clean = pupil.interpolate_blinks(
        stream,
        pad_samples=cfg.blink_pad_samples,
        min_valid=min_valid,
    )
    smoothed = pupil.smooth(
        clean,
        cutoff_hz=cfg.smooth_cutoff_hz,
        order=cfg.smooth_order,
    )
    phases = PhaseDetector().run(smoothed.size.tolist())
    baseline_window = cfg.baseline_window_s
    response_window = cfg.response_window_s
    if baseline_window is None or response_window is None:
        span = float(stream.t[-1] - stream.t[0]) if len(stream) >= 2 else 0.0
        start = float(stream.t[0]) if len(stream) else 0.0
        baseline_window = baseline_window or (start, start + max(span * 0.1, 1e-6))
        response_window = response_window or (
            baseline_window[1],
            float(stream.t[-1]) if len(stream) else baseline_window[1] + 1e-6,
        )
    summary = {
        "mean_size": float(np.mean(smoothed.size)),
        "std_size": float(np.std(smoothed.size)),
        "min_size": float(np.min(smoothed.size)),
        "max_size": float(np.max(smoothed.size)),
        "n_peaks": float(sum(p is Phase.PEAK for p in phases)),
        "n_troughs": float(sum(p is Phase.TROUGH for p in phases)),
        "n_blinks": float(len(pupil.detect_blinks(stream, min_valid=min_valid))),
    }
    summary.update(pupil.quality_summary(stream, min_valid=min_valid))
    with suppress(ValueError):
        summary.update(
            pupil.response_features(
                smoothed,
                baseline_window_s=baseline_window,
                response_window_s=response_window,
            )
        )
    return summary


def analyze_session(
    gaze: GazeStream,
    pupil_stream: PupilStream | None = None,
    *,
    velocity_threshold_deg_s: float = 30.0,
    config: AnalysisConfig | None = None,
) -> SessionReport:
    """Full session analysis: gaze events plus optional pupil summary."""
    report = analyze_gaze(
        gaze,
        velocity_threshold_deg_s=velocity_threshold_deg_s,
        config=config,
    )
    if pupil_stream is not None and len(pupil_stream) > 0:
        pupil_cfg = config.pupil if config is not None else None
        object.__setattr__(report, "pupil", analyze_pupil(pupil_stream, pupil_cfg))
    return report
