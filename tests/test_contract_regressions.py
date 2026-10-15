from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from itrace import io, pupil, saccades, velocity
from itrace.config import analysis_config_from_mapping
from itrace.reporting import empty_session_report_dict, validate_report_payload
from itrace.types import GazeStream, PupilStream


def test_streams_own_read_only_snapshots() -> None:
    timestamps = np.arange(4, dtype=np.float64)
    positions = timestamps.copy()
    gaze = GazeStream(t=timestamps, x=positions, y=positions)
    sizes = positions.copy()
    pupils = PupilStream(t=timestamps, size=sizes)
    timestamps[:] = 100.0
    positions[:] = 99.0
    sizes[:] = 50.0
    np.testing.assert_array_equal(gaze.t, np.arange(4))
    np.testing.assert_array_equal(gaze.x, np.arange(4))
    np.testing.assert_array_equal(pupils.size, np.arange(4))
    for array in (gaze.t, gaze.x, gaze.y, pupils.t, pupils.size):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 123.0


@pytest.mark.parametrize(
    "timestamps",
    [[0.0, 1.0, 1.0, 3.0], [0.0, 2.0, 1.0, 3.0], [0.0, np.nan, 2.0, 3.0], [0.0, 1.0, 2.0, np.inf]],
)
def test_differentiation_rejects_invalid_clocks(timestamps: list[float]) -> None:
    t = np.array(timestamps)
    stream = GazeStream(t=t, x=np.zeros(4), y=np.zeros(4))
    with pytest.raises(ValueError, match="timestamps"):
        _ = stream.sampling_rate_hz
    with pytest.raises(ValueError, match="timestamps"):
        velocity.pos2vel_gradient(stream.x, t)


@pytest.mark.parametrize("sizes", [2, 3, 8, 9, 10, 12, 15])
def test_short_pupil_smoothing_preserves_length_and_constant(sizes: int) -> None:
    t = np.arange(sizes, dtype=np.float64) / 60.0
    result = pupil.smooth(PupilStream(t=t, size=np.full(sizes, 3.0)))
    assert len(result) == sizes
    np.testing.assert_allclose(result.size, 3.0, atol=1e-12)


def test_blink_interpolation_uses_elapsed_time() -> None:
    stream = PupilStream(
        t=np.array([0.0, 0.1, 0.9, 1.0]), size=np.array([1.0, np.nan, np.nan, 3.0])
    )
    clean = pupil.interpolate_blinks(stream, pad_samples=0)
    np.testing.assert_allclose(clean.size, [1.0, 1.2, 2.8, 3.0])


def test_gaze_velocity_uses_actual_irregular_clock() -> None:
    t = np.array([0.0, 0.1, 0.15, 0.4, 0.8, 1.1, 1.2])
    vx, vy, speed = saccades.velocities(GazeStream(t=t, x=2.0 * t, y=np.zeros_like(t)))
    np.testing.assert_allclose(vx, 2.0)
    np.testing.assert_allclose(vy, 0.0)
    np.testing.assert_allclose(speed, 2.0)


def test_padding_cannot_remove_every_interpolation_anchor() -> None:
    stream = PupilStream(t=np.arange(3, dtype=np.float64), size=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"padding.*no valid|no valid.*padding"):
        pupil.interpolate_blinks(stream, pad_samples=1)


def test_zero_mad_still_rejects_isolated_spike() -> None:
    size = np.full(11, 3.0)
    size[5] = 30.0
    flags = pupil.mad_reject(PupilStream(t=np.arange(11, dtype=np.float64), size=size))
    np.testing.assert_array_equal(np.flatnonzero(flags), [5])


def test_missing_gaze_is_never_a_fixation_or_bridged_saccade() -> None:
    t = np.arange(100, dtype=np.float64) / 100.0
    x = np.zeros(100)
    x[40:60] = np.nan
    stream = GazeStream(t=t, x=x, y=np.zeros(100))
    for detector in (saccades.detect_ivt,):
        fixations, events = detector(stream, merge_gap_s=1.0)
        assert not events
        assert len(fixations) == 2
        assert all(np.all(np.isfinite(x[f.onset_idx : f.offset_idx + 1])) for f in fixations)
    fixations = saccades.detect_idt(stream)
    assert len(fixations) == 2
    assert all(np.isfinite(f.centroid_x) for f in fixations)


@pytest.mark.parametrize("units", ["mm\nrelative", "mm\ninvalid", "mm\n"])
def test_pupil_csv_rejects_mixed_or_missing_units(tmp_path: Path, units: str) -> None:
    first, second = units.split("\n")
    path = tmp_path / "pupil.csv"
    path.write_text(f"t,size,unit\n0,3,{first}\n1,3,{second}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unit"):
        io.read_pupil_csv(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("detection", "velocity_threshold_deg_s", np.nan),
        ("detection", "merge_gap_s", np.inf),
        ("detection", "include_pso", "false"),
        ("detection", "velocity_threshold_deg_s", True),
        ("pupil", "smooth_order", 1.5),
        ("pupil", "blink_pad_samples", True),
        ("pupil", "baseline_window_s", [0.0, np.inf]),
        ("pupil", "response_window_s", [0.0]),
    ],
)
def test_configuration_rejects_invalid_scalar_and_window_types(
    section: str,
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        analysis_config_from_mapping({section: {field: value}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_samples", True),
        ("n_samples", -1),
        ("duration_s", np.nan),
        ("duration_s", -0.5),
        ("n_fixations", 1),
        ("n_saccades", True),
    ],
)
def test_report_validation_rejects_invalid_counts_and_duration(field: str, value: object) -> None:
    payload = empty_session_report_dict()
    payload[field] = value
    assert validate_report_payload(payload)["valid"] is False


def _renderer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_manuscript.py"
    spec = importlib.util.spec_from_file_location("itrace_render_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renderer_source_set_matches_audited_paper() -> None:
    from itrace.manuscript import source_files

    renderer = _renderer()
    assert renderer._source_files() == source_files(renderer.MANUSCRIPT)


def test_missing_empirical_ledger_never_invents_measurements(tmp_path: Path) -> None:
    values = _renderer().load_empirical_tokens(tmp_path / "missing.json")
    assert all("unavailable" in value.lower() for value in values.values())
