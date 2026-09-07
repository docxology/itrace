"""Tests for the manuscript cross-reference audit (itrace.manuscript)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from itrace.manuscript import audit_crossrefs

BIB = "@article{doe2026,\n  title = {Example},\n  author = {Doe, Jane},\n  year = {2026},\n}\n"


def _write_manuscript(tmp_path: Path, body: str, bib: str = BIB) -> Path:
    (tmp_path / "00_module.md").write_text(body, encoding="utf-8")
    (tmp_path / "references.bib").write_text(bib, encoding="utf-8")
    return tmp_path


def test_clean_manuscript_passes(tmp_path: Path) -> None:
    manuscript = _write_manuscript(
        tmp_path,
        "# Section {#sec:intro}\n\nSee @fig:trace and [@doe2026].\n\n"
        "![Trace](../../output/figures/x.png){#fig:trace}\n",
    )
    assert audit_crossrefs(manuscript) == 0


def test_dangling_xref_dead_figure_and_missing_cite_fail(tmp_path: Path) -> None:
    manuscript = _write_manuscript(
        tmp_path,
        "# Section {#sec:intro}\n\nSee @fig:missing and [@ghost2026].\n\n"
        "![Trace](../../output/figures/x.png){#fig:trace}\n\n"
        "Manual Figure 2 reference.\n",
    )
    assert audit_crossrefs(manuscript) == 1


def test_render_skip_files_are_not_audited(tmp_path: Path) -> None:
    manuscript = _write_manuscript(tmp_path, "# Section {#sec:intro}\n\n[@doe2026].\n")
    (manuscript / "AGENTS.md").write_text("dangling @fig:none\n", encoding="utf-8")
    (manuscript / "MANUSCRIPT_STATUS.md").write_text("cites [@nowhere2026]\n", encoding="utf-8")
    (manuscript / "_build.md").write_text("dangling @fig:none\n", encoding="utf-8")
    (manuscript / "README.md").write_text("skipped\n", encoding="utf-8")
    assert audit_crossrefs(manuscript) == 0


def test_script_audit_entrypoint_matches_module() -> None:
    """The thin orchestrator returns the module audit verdict for docs/manuscript."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    audit_script = importlib.import_module("audit_crossrefs")
    expected = audit_crossrefs(audit_script.MANUSCRIPT)
    assert audit_script.main() == expected
    assert (audit_script.ROOT / "docs" / "manuscript" / "references.bib").exists()
