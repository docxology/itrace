from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from itrace.release import release_issues, write_asset_manifest
from itrace.version import __version__

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    for name in (
        "pyproject.toml",
        "uv.lock",
        "CITATION.cff",
        ".zenodo.json",
        "README.md",
        "docs/verification_metrics.json",
        "docs/figure_manifest.json",
        "docs/TRACEABILITY_MATRIX.json",
        "docs/manuscript/config.yaml",
        "docs/manuscript/00_abstract.md",
        "docs/manuscript/_build.md",
        "docs/manuscript/_build.pdf",
        f"docs/releases/v{__version__}.md",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    content = b"independent integrity control"
    (tmp_path / "figure.bin").write_bytes(content)
    path = tmp_path / "docs/figure_manifest.json"
    payload = json.loads(path.read_text())
    payload["figures"] = [
        {"path": "figure.bin", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    ]
    path.write_text(json.dumps(payload))
    return tmp_path


def test_release_metadata_and_real_artifacts_agree() -> None:
    assert release_issues(ROOT) == []


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "date",
        "license",
        "doi",
        "citation",
        "artifact",
        "duplicate",
        "escape",
        "empty",
        "pdf",
    ],
)
def test_release_gate_rejects_corruption(release_tree: Path, damage: str) -> None:
    assert release_issues(release_tree) == []
    if damage in {"version", "date", "license"}:
        path = release_tree / ".zenodo.json"
        payload = json.loads(path.read_text())
        payload[
            {"version": "version", "date": "publication_date", "license": "license"}[damage]
        ] = "wrong"
        path.write_text(json.dumps(payload))
    elif damage == "doi":
        path = release_tree / "docs/verification_metrics.json"
        payload = json.loads(path.read_text())
        payload["doi_version"] = "wrong"
        path.write_text(json.dumps(payload))
    elif damage == "citation":
        (release_tree / "CITATION.cff").write_text("version: wrong\n")
    elif damage == "artifact":
        (release_tree / "figure.bin").write_bytes(b"corrupted")
    elif damage == "pdf":
        (release_tree / "docs/manuscript/_build.pdf").write_bytes(b"broken PDF")
    else:
        path = release_tree / "docs/figure_manifest.json"
        payload = json.loads(path.read_text())
        if damage == "duplicate":
            payload["figures"].append(payload["figures"][0])
        elif damage == "escape":
            payload["figures"][0]["path"] = "../outside"
        else:
            payload["figures"] = []
        path.write_text(json.dumps(payload))
    assert release_issues(release_tree)


def test_asset_inventory_binds_exact_commit_and_bytes(tmp_path: Path) -> None:
    payload = b"actual release bytes"
    (tmp_path / "package.whl").write_bytes(payload)
    manifest = write_asset_manifest(tmp_path, "a" * 40)
    receipt = json.loads(manifest.read_text())
    assert receipt["commit"] == "a" * 40
    assert receipt["version"] == __version__
    expected = hashlib.sha256(payload).hexdigest()
    assert receipt["files"][0] == {"name": "package.whl", "bytes": len(payload), "sha256": expected}
    assert (tmp_path / "SHA256SUMS").read_text() == f"{expected}  package.whl\n"
    for item in receipt["files"]:
        assert hashlib.sha256((tmp_path / item["name"]).read_bytes()).hexdigest() == item["sha256"]
    with pytest.raises(FileExistsError):
        write_asset_manifest(tmp_path, "b" * 40)


@pytest.mark.parametrize("damage", ["commit", "empty", "zero", "symlink", "directory", "name"])
def test_asset_inventory_rejects_invalid_payload(tmp_path: Path, damage: str) -> None:
    commit = "a" * 40
    if damage == "commit":
        commit = "main"
    elif damage == "zero":
        (tmp_path / "zero.txt").touch()
    elif damage == "symlink":
        (tmp_path / "link.txt").symlink_to(ROOT / "LICENSE")
    elif damage == "directory":
        (tmp_path / "directory").mkdir()
    elif damage == "name":
        (tmp_path / "bad\nname").write_bytes(b"ambiguous checksum input")
    with pytest.raises(ValueError, match=r"commit|empty|regular files"):
        write_asset_manifest(tmp_path, commit)
    assert not (tmp_path / "release-manifest.json").exists()
    assert not (tmp_path / "SHA256SUMS").exists()
