from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    log: str
    duration_s: float


def run_check(
    name: str,
    command: tuple[str, ...],
    *,
    root: Path,
    evidence_dir: Path,
    environment: dict[str, str] | None = None,
) -> CheckResult:
    """Run a real check, retaining both output streams and its actual exit status."""
    log = evidence_dir / f"{name}.log"
    start = monotonic()
    with log.open("x", encoding="utf-8") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = result.returncode
        except OSError as exc:
            stream.write(f"{type(exc).__name__}: {exc}\n")
            returncode = 127
    return CheckResult(name, command, returncode, str(log), monotonic() - start)


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    extensions = {".py", ".js", ".css", ".html", ".typed", ".csv", ".json"}
    paths = [
        p
        for directory in ("src", "tests", "scripts")
        for p in (root / directory).rglob("*")
        if p.suffix in extensions
    ]
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def verification_commands(root: Path, evidence_dir: Path) -> list[tuple[str, tuple[str, ...]]]:
    python = sys.executable
    return [
        ("lock", ("uv", "lock", "--check", "--offline")),
        ("ruff", (python, "-m", "ruff", "check", "src/", "tests/", "scripts/")),
        ("format", (python, "-m", "ruff", "format", "--check", "src/", "tests/", "scripts/")),
        ("mypy", (python, "-m", "mypy", "src/itrace")),
        ("crossrefs", (python, str(root / "scripts" / "audit_crossrefs.py"))),
        ("release", (python, str(root / "scripts" / "audit_release.py"))),
        (
            "pytest",
            (
                python,
                "-m",
                "pytest",
                "tests/",
                "--cov=itrace",
                "--cov-report=term-missing",
                f"--cov-report=json:{evidence_dir / 'coverage.json'}",
                f"--junitxml={evidence_dir / 'tests.xml'}",
                "-o",
                f"cache_dir={evidence_dir / 'pytest-cache'}",
            ),
        ),
    ]


def run_verification(root: Path, evidence_dir: Path) -> int:
    """Run repository gates once into a new directory; stop and report on failure."""
    root, evidence_dir = root.resolve(), evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        COVERAGE_FILE=str(evidence_dir / "coverage.data"),
        MYPY_CACHE_DIR=str(evidence_dir / "mypy-cache"),
        RUFF_CACHE_DIR=str(evidence_dir / "ruff-cache"),
    )
    original = source_digest(root)
    receipt: dict[str, object] = {
        "kind": "itrace_verification",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": original,
        "python": sys.executable,
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "status": "running",
        "checks": [],
    }
    checks: list[dict[str, object]] = []
    for name, command in verification_commands(root, evidence_dir):
        result = run_check(
            name, command, root=root, evidence_dir=evidence_dir, environment=environment
        )
        checks.append(asdict(result))
        receipt["checks"] = checks
        receipt["status"] = "failed" if result.returncode else "running"
        (evidence_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"{name}: exit {result.returncode}; {result.log}", flush=True)
        if result.returncode:
            return 1
    unchanged = original == source_digest(root)
    receipt["status"] = "passed" if unchanged else "source_changed_during_verification"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (evidence_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if unchanged else 1
