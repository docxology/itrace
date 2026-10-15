from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from itrace.version import CONCEPT_DOI, RELEASE_DATE, VERSION_DOI, __version__


def release_issues(root: Path) -> list[str]:
    root = root.resolve()
    issues: list[str] = []
    metadata_paths = (
        ".zenodo.json",
        "docs/verification_metrics.json",
        "docs/figure_manifest.json",
        "docs/TRACEABILITY_MATRIX.json",
    )
    payloads = {name: json.loads((root / name).read_text()) for name in metadata_paths}
    for name, payload in payloads.items():
        if payload.get("version") != __version__:
            issues.append(f"{name}: version differs from {__version__}")
    published = date.fromisoformat(RELEASE_DATE)
    cover_date = f"{published:%B} {published.day}, {published.year}"
    required = {
        "pyproject.toml": (f'version = "{__version__}"',),
        "uv.lock": (f'name = "itrace"\nversion = "{__version__}"',),
        "CITATION.cff": (
            f'version: "{__version__}"',
            f'date-released: "{RELEASE_DATE}"',
            f'doi: "{VERSION_DOI}"',
            CONCEPT_DOI,
        ),
        "README.md": (VERSION_DOI, CONCEPT_DOI),
        "docs/manuscript/config.yaml": (f'date: "{cover_date}"',),
        "docs/manuscript/00_abstract.md": (VERSION_DOI, CONCEPT_DOI),
        "docs/manuscript/_build.md": (VERSION_DOI, CONCEPT_DOI),
        f"docs/releases/v{__version__}.md": (f"iTrace v{__version__}", VERSION_DOI, CONCEPT_DOI),
    }
    for name, values in required.items():
        text = (root / name).read_text()
        for value in values:
            if value not in text:
                issues.append(f"{name}: missing {value!r}")
    zenodo = payloads[".zenodo.json"]
    if zenodo.get("publication_date") != RELEASE_DATE:
        issues.append(".zenodo.json: publication date differs")
    if zenodo.get("license") != "mit" or zenodo.get("access_right") != "open":
        issues.append(".zenodo.json: expected open MIT software")
    metrics = payloads["docs/verification_metrics.json"]
    for key, value in (("doi_version", VERSION_DOI), ("doi_concept", CONCEPT_DOI)):
        if metrics.get(key) != value:
            issues.append(f"docs/verification_metrics.json: {key} differs")
    figures = payloads["docs/figure_manifest.json"].get("figures", [])
    if not figures:
        issues.append("docs/figure_manifest.json: empty artifact inventory")
    seen: set[Path] = set()
    for item in figures:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            issues.append(f"figure path escapes repository or is duplicated: {item['path']}")
            continue
        seen.add(path)
        content = path.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            issues.append(f"figure integrity differs: {item['path']}")
    if not (root / "docs/manuscript/_build.pdf").read_bytes().startswith(b"%PDF-"):
        issues.append("docs/manuscript/_build.pdf: invalid PDF header")
    return issues


def write_asset_manifest(directory: Path, commit: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("commit must be an exact 40-character Git SHA")
    manifest = directory / "release-manifest.json"
    checksums = directory / "SHA256SUMS"
    if manifest.exists() or checksums.exists():
        raise FileExistsError(
            "release inventory already exists; preserve it and use a new directory"
        )
    paths = sorted(directory.iterdir())
    if not paths:
        raise ValueError("release payload is empty")
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.name) is None
        ):
            raise ValueError(
                f"release assets must be regular files with portable names: {path.name}"
            )
    files = []
    for path in paths:
        content = path.read_bytes()
        if not content:
            raise ValueError(f"release asset is empty: {path.name}")
        files.append(
            {
                "name": path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    with checksums.open("x") as stream:
        stream.writelines(f"{item['sha256']}  {item['name']}\n" for item in files)
    content = checksums.read_bytes()
    files.append(
        {
            "name": checksums.name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    with manifest.open("x") as stream:
        json.dump(
            {"version": __version__, "commit": commit, "doi": VERSION_DOI, "files": files},
            stream,
            indent=2,
        )
        stream.write("\n")
    return manifest
