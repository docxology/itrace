"""Tests for the ``itrace stats`` and ``itrace figures`` CLI commands.

No mocks: commands are exercised end-to-end over real synthetic recordings
written to real temp CSVs, and their JSON / PNG outputs are read back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from itrace import cli, io
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink
from itrace.types import GazeStream


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    rng = np.random.default_rng(seed)
    xs, ys, ts = [], [], []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def test_stats_command_writes_full_summary(tmp_path) -> None:
    gaze_csv = io.write_gaze_csv(_multi_saccade_stream(), tmp_path / "gaze.csv")
    pstream, _ = pupil_sine_with_blink()
    pupil_csv = io.write_pupil_csv(pstream, tmp_path / "pupil.csv")
    out = tmp_path / "stats.json"

    cli.stats(gaze_csv=gaze_csv, out=out, pupil_csv=pupil_csv)

    data = json.loads(out.read_text())
    assert "descriptive" in data
    assert "scanpath" in data
    assert "amplitude_fit" in data  # >= 3 saccades present
    assert data["amplitude_fit"]["family"] == "gamma"
    assert data["descriptive"]["saccades"]["count"] >= 3.0


def test_stats_command_skips_fit_with_few_saccades(tmp_path) -> None:
    gaze, _ = gaze_with_saccade(amplitude_deg=10.0)  # single saccade
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    out = tmp_path / "stats.json"

    cli.stats(gaze_csv=gaze_csv, out=out)

    data = json.loads(out.read_text())
    assert "amplitude_fit" not in data
    assert "descriptive" in data


def test_figures_command_writes_dashboard_png(tmp_path) -> None:
    cli.figures(out_dir=tmp_path, seed=1)
    png = tmp_path / "session_dashboard.png"
    assert png.exists()
    assert png.stat().st_size > 0
    for name in (
        "scanpath.png",
        "pupil_trace.png",
        "event_raster.png",
        "main_sequence_diagnostics.png",
        "statistical_diagnostics.png",
        "statistical_diagnostics.json",
        "synthetic_empirical_range_bridge.png",
        "synthetic_empirical_range_bridge.json",
        "statistical_interpretation_ledger.png",
        "statistical_interpretation_ledger.json",
        "dropout_raster.png",
        "sampling_intervals.png",
        "calibration_residuals.png",
        "pupil_velocity.png",
    ):
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0
    diagnostics = json.loads((tmp_path / "statistical_diagnostics.json").read_text())
    assert diagnostics["kind"] == "itrace_statistical_diagnostics"
    assert "not population physiology" in diagnostics["truth_boundary"]
    assert diagnostics["amplitude_shape_summary"]["available"] is True
    assert diagnostics["amplitude_shape_summary"]["iqr_outlier_count"] >= 0
    assert (
        diagnostics["amplitude_shape_summary"]["median_ci_low"]
        <= diagnostics["amplitude_shape_summary"]["median"]
    )
    assert diagnostics["amplitude_quantile_diagnostics"]["available"] is True
    assert diagnostics["amplitude_quantile_diagnostics"]["residual_rmse_deg"] >= 0.0
    model = diagnostics["amplitude_model_comparison"]
    assert model["weight_criterion"] == "aic"
    assert sum(row["akaike_weight"] for row in model["families"]) == pytest.approx(1.0)
    assert all(row["evidence_ratio"] >= 1.0 for row in model["families"])
    stability = model["model_selection_bootstrap"]
    assert stability["available"] is True
    assert stability["criterion"] == "aic"
    assert stability["successful_bootstraps"] <= stability["n_boot"]
    assert stability["successful_bootstraps"] == sum(stability["winner_counts"].values())
    assert sum(stability["winner_frequencies"].values()) == pytest.approx(1.0)
    assert stability["top_family"] in stability["winner_frequencies"]
    assert 0.0 <= stability["top_frequency"] <= 1.0
    assert stability["unique_winner_count"] == len(stability["winner_frequencies"])
    assert diagnostics["amplitude_cdf_diagnostics"]["available"] is True
    assert diagnostics["amplitude_cdf_diagnostics"]["max_abs_cdf_residual"] >= 0.0
    assert diagnostics["amplitude_cdf_diagnostics"]["dkw_epsilon"] > 0.0
    assert diagnostics["amplitude_cdf_diagnostics"]["cramer_von_mises_statistic"] >= 0.0
    assert diagnostics["amplitude_cdf_diagnostics"]["anderson_darling_statistic"] >= 0.0
    assert diagnostics["amplitude_cdf_diagnostics"]["tail_max_abs_cdf_residual"] >= 0.0
    bridge = json.loads((tmp_path / "synthetic_empirical_range_bridge.json").read_text())
    assert bridge["kind"] == "itrace_synthetic_empirical_range_bridge"
    assert "not reference-device validation" in bridge["truth_boundary"]
    bridge_rows = {row["id"]: row for row in bridge["metrics"]}
    assert bridge_rows["finite_gaze_fraction"]["empirical"]["display"] == "100.0%"
    assert bridge_rows["heldout_target_rms_deg"]["comparability"] == "not_directly_comparable"
    ledger = json.loads((tmp_path / "statistical_interpretation_ledger.json").read_text())
    assert ledger["kind"] == "itrace_statistical_interpretation_ledger"
    assert "reference-device validation" in ledger["truth_boundary"]
    rows = {row["id"]: row for row in ledger["rows"]}
    assert rows["relative_model_comparison"]["evidence_class"] == "relative_model_diagnostic"
    assert rows["synthetic_empirical_context"]["available"] is True
    assert "device performance" in rows["synthetic_empirical_context"]["does_not_prove"]


def test_figures_command_writes_animation_when_requested(tmp_path) -> None:
    cli.figures(out_dir=tmp_path, seed=1, animations=True)
    gif = tmp_path / "synthetic_replay.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0
