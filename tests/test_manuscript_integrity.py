"""Manuscript structural integrity checks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
NON_RENDERED_MARKDOWN = {"README.md", "preamble.md"}


def _source_markdown() -> list[Path]:
    return sorted(
        p
        for p in MANUSCRIPT.glob("*.md")
        if not p.name.startswith("_") and p.name not in NON_RENDERED_MARKDOWN
    )


def test_manuscript_modules_have_one_labelled_h1() -> None:
    bad: list[str] = []
    for path in _source_markdown():
        text = path.read_text()
        h1s = re.findall(r"^# (.+)$", text, flags=re.MULTILINE)
        labelled_h1s = [
            heading for heading in h1s if re.search(r"\{#sec:[-A-Za-z0-9_]+\}$", heading)
        ]
        first_content = next(line for line in text.splitlines() if line.strip())
        if len(h1s) != 1 or len(labelled_h1s) != 1 or not first_content.startswith("# "):
            bad.append(path.name)
    assert bad == []


def test_manuscript_has_no_manual_section_numbering() -> None:
    numbered: list[str] = []
    for path in _source_markdown():
        text = path.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            if re.match(r"^#{1,6}\s+\d+(?:\.\d+)*[\s.]", line):
                numbered.append(f"{path.name}:{line_no}: {line}")
    assert numbered == []


def test_manuscript_crossrefs_resolve() -> None:
    files = _source_markdown()
    text_by_file = {path: path.read_text() for path in files}
    labels = set()
    refs: list[tuple[Path, str]] = []
    for path, text in text_by_file.items():
        labels.update(re.findall(r"\{#((?:sec|fig|tbl|eq):[-A-Za-z0-9_]+)(?:\s+[^}]*)?\}", text))
        refs.extend(
            (path, ref) for ref in re.findall(r"@((?:sec|fig|tbl|eq):[-A-Za-z0-9_]+)", text)
        )
    missing = [(path.name, ref) for path, ref in refs if ref not in labels]
    assert missing == []


def test_manuscript_citations_resolve() -> None:
    bib = (MANUSCRIPT / "references.bib").read_text()
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))
    citation_keys: set[str] = set()
    for path in _source_markdown():
        text = path.read_text()
        for key in re.findall(r"(?<![\w])@([A-Za-z][A-Za-z0-9_:-]+)", text):
            if not key.startswith(("sec:", "fig:", "tbl:", "eq:")):
                citation_keys.add(key)
    assert sorted(citation_keys - bib_keys) == []


def test_manuscript_figure_links_exist() -> None:
    missing: list[str] = []
    for path in _source_markdown():
        text = path.read_text()
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if link.startswith("../output/figures/"):
                target = (MANUSCRIPT / link).resolve()
                if not target.exists():
                    missing.append(f"{path.name}: {link}")
    assert missing == []


def test_public_docs_do_not_carry_stale_metrics() -> None:
    haystack = "\n".join(
        [
            (ROOT / "README.md").read_text(),
            (ROOT / "ISA.md").read_text(),
            (MANUSCRIPT / "README.md").read_text(),
            *(path.read_text() for path in _source_markdown()),
        ]
    )
    for stale in (
        "125 tests",
        "242 tests",
        "384 tests",
        "414 tests",
        "415 tests",
        "420 tests",
        "421 tests",
        "422 tests",
        "424 tests",
        "90.94%",
        "91.01%",
        "91.02%",
        "91.03%",
        "91.05%",
        "92.90%",
        "v0.3.0",
        "0.3.0. Gates",
    ):
        assert stale not in haystack
    assert "429 tests" in haystack
    assert "91.18%" in haystack


def test_rendered_manuscript_has_resolved_crossrefs_and_tokens() -> None:
    rendered = [
        MANUSCRIPT / "_build.md",
        MANUSCRIPT / "_build.txt",
        MANUSCRIPT / "_build.tex",
    ]
    missing = [path.name for path in rendered if not path.exists()]
    assert missing == []
    haystack = "\n".join(path.read_text(errors="ignore") for path in rendered)
    assert re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", haystack) is None
    assert "[@" not in haystack
    assert re.search(r"(?:sec|fig|tbl|eq):[-A-Za-z0-9_]+\?", haystack) is None
