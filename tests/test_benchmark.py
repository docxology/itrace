"""Tests for external event benchmark helpers and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from itrace import benchmark, cli, io
from itrace.synthetic import gaze_with_saccade
from itrace.types import Saccade


def _events_csv(path: Path, events: list[Saccade]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "onset_t",
                "offset_t",
                "amplitude_deg",
                "direction_deg",
                "peak_velocity_deg_s",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "onset_t": event.onset_t,
                    "offset_t": event.offset_t,
                    "amplitude_deg": event.amplitude_deg,
                    "direction_deg": event.direction_deg,
                    "peak_velocity_deg_s": event.peak_velocity_deg_s,
                }
            )
    return path


def _event(onset: float, offset: float, amplitude: float = 10.0) -> Saccade:
    return Saccade(
        onset_idx=0,
        offset_idx=0,
        onset_t=onset,
        offset_t=offset,
        amplitude_deg=amplitude,
        direction_deg=0.0,
        peak_velocity_deg_s=400.0,
    )


def test_benchmark_payload_compares_user_supplied_truth_and_systems() -> None:
    truth = [_event(0.10, 0.13), _event(0.40, 0.43)]
    good = [_event(0.101, 0.131), _event(0.40, 0.431)]
    poor = [_event(0.70, 0.73)]

    payload = benchmark.benchmark_payload(
        truth,
        {"good": good, "poor": poor},
        min_overlap_s=0.005,
    )

    assert payload["truth_boundary"] == benchmark.TRUTH_BOUNDARY
    assert payload["systems"]["good"]["recovery"]["f1"] == pytest.approx(1.0)
    assert payload["systems"]["poor"]["recovery"]["f1"] == pytest.approx(0.0)
    assert payload["systems"]["good"]["timing"]["onset_mae_s"] < 0.002


def test_load_saccade_events_csv_validates_shape(tmp_path: Path) -> None:
    events = [_event(0.1, 0.2)]
    path = _events_csv(tmp_path / "events.csv", events)
    loaded = benchmark.load_saccade_events_csv(path)
    assert loaded[0].onset_t == pytest.approx(0.1)

    bad = tmp_path / "bad.csv"
    bad.write_text("start,end\n0,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        benchmark.load_saccade_events_csv(bad)


def test_benchmark_cli_writes_itrace_and_comparator_results(tmp_path: Path) -> None:
    gaze, truth = gaze_with_saccade(amplitude_deg=10.0)
    gaze_csv = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    truth_event = _event(truth.onset_t, truth.offset_t, truth.amplitude_deg)
    truth_csv = _events_csv(tmp_path / "truth.csv", [truth_event])
    comparator_csv = _events_csv(tmp_path / "comparator.csv", [_event(0.8, 0.9)])
    out = tmp_path / "benchmark.json"

    cli.benchmark_events(
        truth_csv=truth_csv,
        gaze_csv=gaze_csv,
        comparator_events_csv=comparator_csv,
        out=out,
    )

    payload = json.loads(out.read_text())
    assert payload["kind"] == "external_saccade_benchmark"
    assert payload["systems"]["itrace"]["recovery"]["recall"] == pytest.approx(1.0)
    assert payload["systems"]["comparator"]["recovery"]["recall"] == pytest.approx(0.0)
