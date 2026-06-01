"""Synthetic-domain validation and live diagnostics."""

from __future__ import annotations

import json

import numpy as np

from itrace import io, validation
from itrace.synthetic import SyntheticSessionSpec, synthetic_session


def test_synthetic_domain_validation_is_deterministic_and_truth_bounded() -> None:
    domain = validation.default_synthetic_domains()[0]

    first = validation.validate_synthetic_domain(domain, seed=12)
    second = validation.validate_synthetic_domain(domain, seed=12)

    assert first == second
    assert first["domain"] == "clean_desktop"
    assert first["n_truth_saccades"] == 7
    assert first["saccade_recovery"]["f1"] > 0.5
    assert first["pupil"]["truth_event_count"] == 3.0


def test_within_and_cross_domain_validation_summaries() -> None:
    domains = validation.default_synthetic_domains()[:2]

    suite = validation.synthetic_validation_suite(domains=domains, repetitions=2, first_seed=3)

    assert suite["kind"] == "synthetic_domain_validation"
    assert suite["domain_count"] == 2
    assert suite["truth_boundary"].startswith("Synthetic domains")
    assert suite["cross_domain"]["macro_saccade_f1"] > 0.0
    assert suite["cross_domain"]["stability_gap_f1"] >= 0.0
    for domain in suite["domains"]:
        assert domain["repetitions"] == 2
        assert domain["summary"]["saccade_f1"]["n"] == 2.0
        assert "peak_velocity_rmse_deg_s" in domain["summary"]


def test_default_dropout_domain_validation_uses_safe_gap_interpolation() -> None:
    domain = validation.default_synthetic_domains()[-1]

    result = validation.within_domain_validation(domain, repetitions=5, first_seed=0)

    assert result["domain"] == "low_light_dropout"
    assert result["summary"]["saccade_f1"]["n"] == 5.0
    assert any(
        run["gaze"]["short_gap_interpolation_used"] == 1.0
        or run["gaze"]["invalid_sample_drop_used"] == 1.0
        for run in result["runs"]
    )


def test_live_recording_diagnostics_do_not_claim_reference_truth() -> None:
    gaze, pupil, _truth = synthetic_session(
        SyntheticSessionSpec(seed=5, n_saccades=3, duration_s=4.0, dropout_fraction=0.02)
    )
    report = {"n_saccades": 3, "duration_s": 4.0, "quality": {"detection_threshold_deg_s": 42.0}}

    diagnostics = validation.live_recording_diagnostics(gaze, pupil, report)

    assert diagnostics["truth_boundary"] == "no live reference truth"
    assert diagnostics["quality_index"] > 0.0
    assert diagnostics["sample_count"] == len(gaze)
    assert diagnostics["sampling_rate_hz"] > 0.0
    assert diagnostics["gaze_path_length_deg"] > 0.0


def test_synthetic_validation_cli_writes_json(tmp_path) -> None:
    from itrace import cli

    out = tmp_path / "validation.json"

    cli.synthetic_validation(out=out, repetitions=1, first_seed=9)

    payload = json.loads(out.read_text())
    assert payload["domain_count"] >= 4
    assert payload["cross_domain"]["worst_domain"]


def test_validation_module_keeps_io_roundtrip_independent(tmp_path) -> None:
    gaze, _pupil, _truth = synthetic_session(SyntheticSessionSpec(seed=8, n_saccades=1))
    path = io.write_gaze_csv(gaze, tmp_path / "gaze.csv")
    loaded = io.read_gaze_csv(path)

    assert np.allclose(loaded.t, gaze.t)
    assert np.allclose(loaded.x, gaze.x, equal_nan=True)
