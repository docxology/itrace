import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from itrace.cli import app
from itrace.study.jsonio import mapping

from .helpers import Harness


def test_live_quality_distinguishes_no_samples_finite_gaze_and_face_loss(harness: Harness) -> None:
    empty = mapping(harness.session.status()["capture_quality"], "quality")
    assert empty["ready_to_calibrate"] is False
    for _ in range(12):
        harness.clock.value += 0.03
        harness.sample(1, 2)
    healthy = mapping(harness.session.status()["capture_quality"], "quality")
    assert healthy["valid_samples"] == 12
    assert healthy["ready_to_calibrate"] is True
    harness.clock.value += 0.03
    harness.sample(1, 2, face=0)
    lost = mapping(harness.session.status()["capture_quality"], "quality")
    assert lost["ready_to_calibrate"] is False
    assert lost["valid_samples"] == 12


def test_live_quality_rejects_stale_samples(harness: Harness) -> None:
    for _ in range(12):
        harness.clock.value += 0.03
        harness.sample(1, 2)
    harness.clock.value += 2
    quality = mapping(harness.session.status()["capture_quality"], "quality")
    assert quality["ready_to_calibrate"] is False
    assert quality["recent_samples"] == 0


def test_preflight_checks_frozen_inputs_without_allocating_sessions(
    harness: Harness, tmp_path: Path
) -> None:
    from itrace.study.preflight import check_study

    target = tmp_path / "preflight"
    first = check_study(tmp_path / "study.json", target, require_runtime=False)
    second = check_study(tmp_path / "study.json", target, require_runtime=False)
    assert first["manifest_sha256"] == second["manifest_sha256"] == harness.manifest.digest
    assert first["practice_trials"] == 1
    assert first["scored_trials"] == 2
    assert first["unique_images"] == 3
    assert list(target.iterdir()) == []


def test_preflight_rejects_changed_image(harness: Harness, tmp_path: Path) -> None:
    from itrace.study import StudyError
    from itrace.study.preflight import check_study

    (tmp_path / "image0.png").write_bytes(b"altered")
    with pytest.raises(StudyError):
        check_study(tmp_path / "study.json", tmp_path / "records", require_runtime=False)
    assert not (tmp_path / "records").exists()


def test_preflight_rejects_public_or_symlink_storage(harness: Harness, tmp_path: Path) -> None:
    from itrace.study import StudyError
    from itrace.study.preflight import check_study

    public = tmp_path / "public"
    (public / "src/itrace").mkdir(parents=True)
    (public / "src/itrace/__init__.py").touch()
    (public / "pyproject.toml").touch()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "private")
    for path in (public / "private", link / "nested"):
        with pytest.raises(StudyError):
            check_study(tmp_path / "study.json", path, require_runtime=False)
        assert not path.exists()


def test_preflight_cli_rejects_bad_manifest_without_camera_access(tmp_path: Path) -> None:
    assert CliRunner().invoke(app, ["study-check", "--help"]).exit_code == 0
    result = CliRunner().invoke(
        app, ["study-check", "--manifest", str(tmp_path / "missing"), "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert not list(tmp_path.glob("*/journal.jsonl"))


def test_probe_quality_reports_actual_frames_and_rejects_absent_gaze() -> None:
    from itrace.capture import CaptureSample
    from itrace.study import StudyError
    from itrace.study.readiness import probe_quality
    from itrace.types import GazeSample

    sample = CaptureSample(0, 1.0, GazeSample(1.0, 2, 3), None, 30, {"face_detected": 1})
    assert probe_quality([sample])["read_frames"] == 1
    assert probe_quality([sample])["finite_gaze_samples"] == 1
    missing = CaptureSample(
        1, 2.0, GazeSample(2.0, float("nan"), 3), None, 30, {"face_detected": 0}
    )
    for samples in ([], [missing], [sample, sample]):
        with pytest.raises(StudyError):
            probe_quality(samples)


@pytest.mark.parametrize("arguments", [["--frames", "0"], ["--frames", "301"], ["--camera", "-1"]])
def test_probe_cli_rejects_invalid_limits_before_camera_access(arguments: list[str]) -> None:
    assert CliRunner().invoke(app, ["camera-probe", *arguments]).exit_code == 2


def test_preflight_runtime_requirement_uses_actual_installed_extras(
    harness: Harness, tmp_path: Path
) -> None:
    from itrace.study import StudyError
    from itrace.study.preflight import check_study

    names = ("cv2", "mediapipe", "fastapi", "uvicorn")
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        with pytest.raises(StudyError) as rejected:
            check_study(tmp_path / "study.json", tmp_path / "required")
        assert rejected.value.field == "dependencies"
    else:
        report = check_study(tmp_path / "study.json", tmp_path / "required")
        assert report["missing_runtime_modules"] == []


def test_preflight_receipt_refuses_overwrite(harness: Harness, tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_text("keep this receipt")
    result = CliRunner().invoke(
        app,
        [
            "study-check",
            "--manifest",
            str(tmp_path / "study.json"),
            "--output-dir",
            str(tmp_path / "records"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 2
    assert out.read_text() == "keep this receipt"
