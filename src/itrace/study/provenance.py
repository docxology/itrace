"""Read-only source/runtime bindings without optional dependencies or Git mutation."""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .jsonio import JsonValue


def source_bindings(package: Path) -> dict[str, JsonValue]:
    checkout = package.parent.name == "src" and (package.parent.parent / "pyproject.toml").is_file()
    root = package.parent.parent if checkout else package.parent
    paths = sorted(package.rglob("*.py"))
    paths += sorted(
        path
        for path in (package / "web_static").glob("*")
        if path.suffix in {".html", ".js", ".css"}
    )
    if checkout:
        paths += [root / "pyproject.toml", root / "uv.lock"]
    digest = hashlib.sha256()
    hashes: dict[str, JsonValue] = {}
    for path in paths:
        if path.is_file():
            raw = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            hashed = hashlib.sha256(raw).hexdigest()
            hashes[relative] = hashed
            digest.update(relative.encode() + b"\0" + bytes.fromhex(hashed))
    commit: str | None = None
    git = root / ".git"
    if git.is_file():
        pointer = git.read_text().strip().removeprefix("gitdir: ")
        git = (root / pointer).resolve()
    if (git / "HEAD").is_file():
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = git / head.removeprefix("ref: ")
            if ref.is_file():
                commit = ref.read_text().strip()
        else:
            commit = head
    return {
        "source_commit": commit,
        "source_digest": digest.hexdigest(),
        "source_files": hashes,
        "installation_kind": "source-checkout" if checkout else "installed-package",
    }


def provenance() -> dict[str, JsonValue]:
    try:
        package_version: str | None = version("itrace")
    except PackageNotFoundError:
        package_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
        "package_version": package_version,
        **source_bindings(Path(__file__).resolve().parent.parent),
        "dirty_status": "source_digest binds current bytes; Git index not inspected",
        "capture_timestamp": "native host monotonic seconds; software acquisition boundary",
        "presentation_timing": "browser rAF estimate; not physical display onset proof",
        "scope": "N=1 private derived records; no device accuracy claim",
    }
