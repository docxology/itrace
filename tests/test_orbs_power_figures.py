"""Orb-animation and power-figure rendering tests (ISC-87..89). Skips w/o matplotlib."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.mark.timeout(180)
def test_orbs_still_png(tmp_path) -> None:  # ISC-87
    import generate_orbs_animation as goa

    path = goa.generate_orbs_still(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 1000


@pytest.mark.timeout(180)
def test_orbs_gif(tmp_path) -> None:  # ISC-87/ISC-88
    import generate_orbs_animation as goa

    path = goa.generate_orbs_gif(tmp_path, n_frames=10)
    assert path.exists()
    assert path.suffix == ".gif"
    assert path.stat().st_size > 1000


@pytest.mark.timeout(180)
def test_power_figure_png(tmp_path) -> None:  # ISC-89
    import generate_power_figure as gpf

    path = gpf.generate_power_figure(tmp_path, n_trials=5)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 1000


@pytest.mark.timeout(120)
def test_power_summary_table_written(tmp_path) -> None:  # ISC-94
    import generate_power_figure as gpf

    path = gpf.write_summary_table(tmp_path, n_trials=5)
    assert path.exists()
    assert path.name == "noise_summary.md"
    text = path.read_text()
    assert "| sigma (norm) |" in text
    assert "+/-" in text


@pytest.mark.timeout(120)
def test_power_noise_metrics_sidecar_written(tmp_path) -> None:
    import generate_power_figure as gpf

    path = gpf.write_noise_metrics(tmp_path, n_trials=5)
    assert path.exists()
    assert path.name == "noise_metrics.json"
    payload = json.loads(path.read_text())
    assert payload["kind"] == "itrace_noise_sweep_metrics"
    assert payload["n_trials"] == 5
    assert payload["records"]
    saccade = payload["thresholds"]["saccade_f1_0_8"]
    assert saccade["metric"] == "saccade_f1"
    assert saccade["bound"] == 0.8
    assert saccade["pixels_at_640"] is None or saccade["pixels_at_640"] >= 0.0
