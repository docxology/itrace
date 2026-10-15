from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from itrace.verification import run_check, run_verification, source_digest, verification_commands


def test_failed_check_retains_stderr_and_exit_status(tmp_path: Path) -> None:
    command = (
        sys.executable,
        "-c",
        "import sys; print('negative control', file=sys.stderr); sys.exit(7)",
    )
    result = run_check("failure", command, root=tmp_path, evidence_dir=tmp_path)
    assert result.returncode == 7
    assert "negative control" in Path(result.log).read_text()
    with pytest.raises(FileExistsError):
        run_check("failure", command, root=tmp_path, evidence_dir=tmp_path)


def test_success_and_missing_executable_are_distinguished(tmp_path: Path) -> None:
    passed = run_check(
        "success",
        (sys.executable, "-c", "print('actual success')"),
        root=tmp_path,
        evidence_dir=tmp_path,
    )
    missing = run_check(
        "missing", (str(tmp_path / "nonexistent"),), root=tmp_path, evidence_dir=tmp_path
    )
    assert passed.returncode == 0
    assert "actual success" in Path(passed.log).read_text()
    assert missing.returncode == 127
    assert "FileNotFoundError" in Path(missing.log).read_text()


def test_source_digest_changes_with_content_and_names(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    path = source / "module.py"
    path.write_text("value = 1\n")
    before = source_digest(tmp_path)
    path.write_text("value = 2\n")
    after = source_digest(tmp_path)
    assert before != after
    path.rename(source / "renamed.py")
    assert source_digest(tmp_path) != after
    commands = dict(verification_commands(tmp_path, tmp_path / "evidence"))
    assert set(commands) == {"lock", "ruff", "format", "mypy", "crossrefs", "release", "pytest"}
    assert commands["pytest"][0] == sys.executable


def test_verification_stops_after_real_lock_failure_and_preserves_receipt(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project\n")
    evidence = tmp_path / "receipt"
    assert run_verification(root, evidence) == 1
    receipt = json.loads((evidence / "summary.json").read_text())
    assert receipt["status"] == "failed"
    assert len(receipt["checks"]) == 1
    assert receipt["checks"][0]["name"] == "lock"
    assert receipt["checks"][0]["returncode"] != 0
    assert (evidence / "lock.log").stat().st_size > 0
    with pytest.raises(FileExistsError):
        run_verification(root, evidence)
