"""Typer command-line interface for iTrace.

Commands
--------
* ``itrace demo``       - synthesize gaze + pupil, analyse, print recovered events.
* ``itrace analyze``    - analyse a gaze CSV (optionally a pupil CSV); write JSON.
* ``itrace record``     - live webcam capture (requires the ``capture`` extra).
* ``itrace dashboard``  - launch the Streamlit dashboard (``dashboard`` extra).
* ``itrace live-html``  - launch the local HTML orchestrator (``web`` + ``capture`` extras).
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer

from . import io, pipeline, saccades, validation
from .calibration import (
    AffineCalibration,
    calibration_error,
    interpolate_gaze_gaps,
    robust_gaze_quality,
)
from .capture import CaptureSample
from .config import AnalysisConfig, analysis_config_from_json
from .pupil import quality_summary
from .reporting import validate_report_payload
from .stats import bootstrap, descriptive, distributions, scanpath_metrics
from .synthetic import gaze_with_saccade, pupil_sine_with_blink
from .types import GazeStream, PupilStream, PupilUnit, SessionReport
from .version import __version__

app = typer.Typer(add_completion=False, help="iTrace - webcam eye-movement analysis toolkit.")


@contextmanager
def _native_stderr(backend_logs: bool) -> Iterator[None]:
    """Optionally silence noisy native OpenCV/MediaPipe stderr diagnostics."""
    if backend_logs:
        yield
        return
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def write_capture_records_csv(samples: list[CaptureSample], out: Path) -> Path:
    """Write full capture records: frame, timing, gaze, pupil proxy, FPS, quality."""
    quality_keys = sorted({key for sample in samples for key in sample.quality})
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "gaze_x_deg",
        "gaze_y_deg",
        "pupil_size",
        "pupil_unit",
        "fps_estimate_hz",
        *[f"quality_{key}" for key in quality_keys],
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row: dict[str, object] = {
                "frame_index": sample.frame_index,
                "timestamp_s": sample.timestamp_s,
                "gaze_x_deg": sample.gaze.x,
                "gaze_y_deg": sample.gaze.y,
                "pupil_size": sample.pupil.size if sample.pupil is not None else "",
                "pupil_unit": sample.pupil.unit.value if sample.pupil is not None else "",
                "fps_estimate_hz": sample.fps_estimate_hz,
            }
            row.update({f"quality_{key}": sample.quality.get(key, "") for key in quality_keys})
            writer.writerow(row)
    return out


def write_capture_samples(
    samples: list[CaptureSample], out: Path, pupil_out: Path | None = None
) -> int:
    """Write capture samples to gaze and optional pupil CSV files.

    The capture module itself is import-safe; constructing ``WebcamSource`` is
    still the only place that imports hardware dependencies.
    """
    gaze = GazeStream(
        t=np.array([s.gaze.t for s in samples], dtype=np.float64),
        x=np.array([s.gaze.x for s in samples], dtype=np.float64),
        y=np.array([s.gaze.y for s in samples], dtype=np.float64),
    )
    io.write_gaze_csv(gaze, out)
    if pupil_out is not None:
        pupil_samples = [s.pupil for s in samples if s.pupil is not None]
        pstream = PupilStream(
            t=np.array([p.t for p in pupil_samples], dtype=np.float64),
            size=np.array([p.size for p in pupil_samples], dtype=np.float64),
            unit=pupil_samples[0].unit if pupil_samples else PupilUnit.RELATIVE,
        )
        io.write_pupil_csv(pstream, pupil_out)
    return len(samples)


def load_analysis_config(config_json: Path | None) -> AnalysisConfig:
    """Load an analysis config JSON file or return defaults."""
    return analysis_config_from_json(config_json) if config_json is not None else AnalysisConfig()


def _config_with_cli_overrides(
    config_json: Path | None,
    *,
    velocity_threshold: float | None = None,
) -> AnalysisConfig:
    cfg = load_analysis_config(config_json)
    if velocity_threshold is not None:
        cfg = replace(
            cfg,
            detection=replace(cfg.detection, velocity_threshold_deg_s=velocity_threshold),
        )
    return cfg


def _event_ci(report: SessionReport) -> dict[str, object]:
    saccades_list = report.saccades
    fixations_list = report.fixations
    out: dict[str, object] = {}
    for name, values in (
        ("saccade_duration_mean_ci_s", [s.duration_s for s in saccades_list]),
        ("saccade_amplitude_mean_ci_deg", [s.amplitude_deg for s in saccades_list]),
        ("saccade_peak_velocity_mean_ci_deg_s", [s.peak_velocity_deg_s for s in saccades_list]),
        ("fixation_duration_mean_ci_s", [f.duration_s for f in fixations_list]),
    ):
        if len(values) < 2:
            out[name] = {"n": float(len(values)), "mean": float(np.mean(values)) if values else 0.0}
            continue
        boot = bootstrap.bootstrap_statistic(values, lambda arr: float(np.mean(arr)), n_boot=500)
        lo, hi = bootstrap.percentile_interval(boot)
        out[name] = {
            "n": float(len(values)),
            "mean": float(np.mean(values)),
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


@app.command()
def version() -> None:
    """Print the installed iTrace version."""
    typer.echo(__version__)


@app.command()
def demo(amplitude: float = 10.0, direction: float = 0.0) -> None:
    """Synthesize a saccade + pupil signal, analyse it, and print the result."""
    gaze, truth = gaze_with_saccade(amplitude_deg=amplitude, direction_deg=direction)
    pstream, _ = pupil_sine_with_blink()
    report = pipeline.analyze_session(gaze, pstream)
    typer.echo(f"samples={report.n_samples} duration={report.duration_s:.3f}s")
    typer.echo(f"fixations={len(report.fixations)} saccades={len(report.saccades)}")
    if report.saccades:
        s = report.saccades[0]
        typer.echo(
            f"saccade: amplitude={s.amplitude_deg:.2f}deg "
            f"(truth {truth.amplitude_deg:.2f}) "
            f"direction={s.direction_deg:.1f}deg peak_vel={s.peak_velocity_deg_s:.0f}deg/s"
        )
    typer.echo(f"scanpath={report.scanpath!r}")
    if report.pupil:
        typer.echo(f"pupil: {report.pupil}")


@app.command()
def analyze(
    gaze_csv: Path,
    out: Path = Path("report.json"),
    pupil_csv: Path | None = None,
    velocity_threshold: float | None = None,
    config_json: Path | None = None,
) -> None:
    """Analyse a gaze CSV (and optional pupil CSV); write a JSON report."""
    gaze = io.read_gaze_csv(gaze_csv)
    pstream = io.read_pupil_csv(pupil_csv) if pupil_csv is not None else None
    cfg = _config_with_cli_overrides(config_json, velocity_threshold=velocity_threshold)
    report = pipeline.analyze_session(gaze, pstream, config=cfg)
    payload = report.to_dict()
    validate_report_payload(payload, raise_on_error=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"wrote {out} ({len(report.saccades)} saccades, {len(report.fixations)} fixations)")


@app.command()
def stats(
    gaze_csv: Path,
    out: Path = Path("stats.json"),
    pupil_csv: Path | None = None,
    velocity_threshold: float | None = None,
    config_json: Path | None = None,
    family: str = "gamma",
) -> None:
    """Analyse a gaze CSV and write descriptive/distribution/scanpath statistics.

    Computes the descriptive fixation/saccade summaries, the scanpath spread
    metrics, and a maximum-likelihood fit of the saccade-amplitude distribution
    (when at least three saccades are present), then writes them as JSON.
    """
    gaze = io.read_gaze_csv(gaze_csv)
    pstream = io.read_pupil_csv(pupil_csv) if pupil_csv is not None else None
    cfg = _config_with_cli_overrides(config_json, velocity_threshold=velocity_threshold)
    report = pipeline.analyze_session(gaze, pstream, config=cfg)

    result: dict[str, object] = {
        "descriptive": descriptive.summarize_report(report),
        "scanpath": scanpath_metrics.scanpath_summary(report),
        "quality": report.quality,
        "event_ci": _event_ci(report),
        "config": report.config,
    }
    if report.pupil:
        result["pupil"] = report.pupil
    amplitudes = saccades.saccade_properties(report.saccades)["amplitude_deg"]
    if amplitudes.size >= 3:
        fit = distributions.fit_distribution(amplitudes, family=family)
        result["amplitude_fit"] = {
            "family": fit.family,
            "params": fit.params,
            "aic": fit.aic,
            "bic": fit.bic,
            "ks_statistic": fit.ks_statistic,
            "ks_pvalue": fit.ks_pvalue,
            "n": fit.n,
        }
    out.write_text(json.dumps(result, indent=2))
    typer.echo(f"wrote {out} ({len(report.saccades)} saccades, {len(report.fixations)} fixations)")


@app.command()
def calibrate(
    points_csv: Path,
    out: Path = Path("calibration.json"),
    apply_gaze: Path | None = None,
    calibrated_out: Path | None = None,
) -> None:
    """Fit an affine calibration from target-point CSV data."""
    import pandas as pd

    table = pd.read_csv(points_csv)
    required = {"raw_x", "raw_y", "target_x", "target_y"}
    missing = sorted(required - set(table.columns))
    if missing:
        msg = f"calibration CSV missing columns: {', '.join(missing)}"
        raise typer.BadParameter(msg)
    cal = AffineCalibration.fit(
        table["raw_x"].to_numpy(),
        table["raw_y"].to_numpy(),
        table["target_x"].to_numpy(),
        table["target_y"].to_numpy(),
    )
    payload = cal.to_dict()
    payload.update(
        calibration_error(
            cal,
            table["raw_x"].to_numpy(),
            table["raw_y"].to_numpy(),
            table["target_x"].to_numpy(),
            table["target_y"].to_numpy(),
        )
    )
    out.write_text(json.dumps(payload, indent=2))
    if apply_gaze is not None:
        target = calibrated_out or apply_gaze.with_name(f"{apply_gaze.stem}_calibrated.csv")
        io.write_gaze_csv(cal.apply_stream(io.read_gaze_csv(apply_gaze)), target)
    typer.echo(f"wrote {out}")


@app.command("validate-recording")
def validate_recording(
    gaze_csv: Path,
    out: Path = Path("validation.json"),
    pupil_csv: Path | None = None,
    config_json: Path | None = None,
    calibration_json: Path | None = None,
) -> None:
    """Validate recording quality before analysis."""
    gaze = io.read_gaze_csv(gaze_csv)
    pstream = io.read_pupil_csv(pupil_csv) if pupil_csv is not None else None
    cfg = load_analysis_config(config_json)
    quality = robust_gaze_quality(gaze)
    try:
        analysis_gaze = interpolate_gaze_gaps(gaze, max_gap_s=0.05)
    except ValueError:
        analysis_gaze = gaze
    try:
        report = pipeline.analyze_session(analysis_gaze, pstream, config=cfg)
        report_validation = validate_report_payload(report.to_dict())
    except ValueError as exc:
        report = None
        report_validation = {"valid": False, "errors": [str(exc)], "warnings": []}
    warnings: list[str] = []
    errors: list[str] = []
    if quality["dropout_fraction"] > 0.0:
        warnings.append("gaze contains non-finite samples")
    if quality["nonmonotonic_timestamp_count"] > 0.0:
        errors.append("gaze timestamps are nonmonotonic")
    if quality["large_gap_count"] > 0.0:
        warnings.append("gaze contains large timestamp gaps")
    if report is None:
        errors.append("recording could not be analyzed with current configuration")
    elif len(report.saccades) == 0:
        warnings.append("no saccades detected with current configuration")
    pupil_payload: dict[str, float] = {}
    if pstream is not None:
        pupil_payload = quality_summary(pstream, min_valid=max(cfg.pupil.blink_threshold, 1e-6))
        if pupil_payload.get("blink_fraction", 0.0) > 0.0:
            warnings.append("pupil trace contains blink or invalid samples")
    payload = {
        "n_samples": len(gaze),
        "duration_s": float(gaze.t[-1] - gaze.t[0]) if len(gaze) >= 2 else 0.0,
        "quality": quality,
        "events": {
            "n_fixations": len(report.fixations) if report is not None else 0,
            "n_saccades": len(report.saccades) if report is not None else 0,
            "n_microsaccades": len(report.microsaccades) if report is not None else 0,
            "n_smooth_pursuits": len(report.smooth_pursuits) if report is not None else 0,
            "n_psos": len(report.psos) if report is not None else 0,
        },
        "pupil": pupil_payload,
        "event_ci": _event_ci(report) if report is not None else {},
        "calibration": {
            "available": calibration_json is not None and calibration_json.exists(),
            "path": str(calibration_json) if calibration_json is not None else None,
        },
        "warnings": warnings,
        "errors": errors,
        "report_validation": report_validation,
        "config": report.config if report is not None else cfg.to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"wrote {out} ({len(warnings)} warnings, {len(errors)} errors)")


@app.command("synthetic-validation")
def synthetic_validation(
    out: Path = Path("synthetic_validation.json"),
    repetitions: int = 5,
    first_seed: int = 0,
) -> None:
    """Run within- and across-domain synthetic recovery validation."""
    payload = validation.synthetic_validation_suite(
        repetitions=repetitions,
        first_seed=first_seed,
    )
    out.write_text(json.dumps(payload, indent=2))
    cross = payload["cross_domain"]
    assert isinstance(cross, dict)
    typer.echo(
        "wrote "
        f"{out} (domains={payload['domain_count']}, "
        f"macro_f1={float(cross['macro_saccade_f1']):.3f}, "
        f"worst={cross['worst_domain']})"
    )


@app.command()
def figures(
    out_dir: Path = Path("output/figures"),
    seed: int = 0,
    animations: bool = False,
) -> None:
    """Render the multi-panel session dashboard (requires the 'figures' extra).

    Uses :mod:`itrace.viz` (matplotlib) to synthesise a deterministic session and
    write a publication-quality dashboard PNG into ``out_dir``.
    """
    from .viz.gallery import render_gallery

    paths = render_gallery(out_dir, seed=seed, animations=animations)
    for path in paths:
        typer.echo(f"wrote {path}")


@app.command()
def record(
    camera: int = 0,
    max_frames: int = 300,
    out: Path = Path("gaze.csv"),
    pupil_out: Path | None = None,
    records_out: Path | None = None,
    backend_logs: bool = False,
) -> None:
    """Capture live gaze from a webcam and write CSV files."""
    from .capture import WebcamSource

    with _native_stderr(backend_logs):
        source = WebcamSource(camera_index=camera)  # pragma: no cover - hardware
        samples = list(source.frames(max_frames=max_frames))  # pragma: no cover - hardware
    write_capture_samples(samples, out, pupil_out)  # pragma: no cover - hardware
    wrote = [str(out)]
    if pupil_out is not None:
        wrote.append(str(pupil_out))
    if records_out is not None:
        write_capture_records_csv(samples, records_out)
        wrote.append(str(records_out))
    typer.echo(
        f"captured {len(samples)} detected face samples -> {', '.join(wrote)}"
    )  # pragma: no cover - hardware


@app.command("camera-probe")
def camera_probe(camera: int = 0, frames: int = 5, backend_logs: bool = False) -> None:
    """Probe webcam dependency and camera access without writing data."""
    from .capture import WebcamSource

    try:
        with _native_stderr(backend_logs):
            source = WebcamSource(camera_index=camera)  # pragma: no cover - hardware
            samples = list(source.frames(max_frames=frames))  # pragma: no cover - hardware
    except RuntimeError as exc:  # pragma: no cover - hardware/environment
        typer.echo(f"camera probe failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"camera={camera} read_frames={frames} detected_face_samples={len(samples)}")


@app.command("live-html")
def live_html(
    camera: int = 0,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: Path | None = None,
    backend_logs: bool = False,
    open_browser: bool = False,
) -> None:
    """Launch the local HTML live webcam orchestrator (requires the 'web' extra)."""
    from .live import serve_live_html

    try:
        serve_live_html(
            camera_index=camera,
            host=host,
            port=port,
            output_dir=output_dir,
            backend_logs=backend_logs,
            open_browser=open_browser,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def dashboard() -> None:
    """Launch the Streamlit live dashboard (requires the 'dashboard' extra)."""
    from .dashboard import run_app

    run_app()  # pragma: no cover - launches server


def main() -> None:
    """Entry point for the ``itrace`` console script."""
    app()


if __name__ == "__main__":
    main()
