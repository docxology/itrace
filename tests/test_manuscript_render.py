from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from itrace.manuscript import (
    load_empirical_session_tokens,
    load_empirical_tokens,
    load_metric_tokens,
    load_noise_tokens,
    render_manuscript,
    substitute_tokens,
    validate_rendered_text,
)


@pytest.mark.parametrize("text", ["{{MISSING}}", "¿tbl:missing?", "[@tbl:missing]", "??"])
def test_unresolved_render_output_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="unresolved"):
        validate_rendered_text(text)


def test_tokens_require_explicit_sources(tmp_path: Path) -> None:
    assert substitute_tokens("Count {{TEST_COUNT}}", {"TEST_COUNT": "3"}) == "Count 3"
    with pytest.raises(KeyError, match="MISSING"):
        substitute_tokens("{{MISSING}}", {})
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"test_count": 3}')
    with pytest.raises(ValueError, match="verification metrics"):
        load_metric_tokens(metrics)
    metrics.write_text('{"test_count": 3, "demo_amplitude": "10", "coverage_pct": "95.0%"}')
    assert load_metric_tokens(metrics)["TEST_COUNT"] == "3"
    for loader in (load_empirical_tokens, load_empirical_session_tokens):
        assert all("unavailable" in v for v in loader(tmp_path / "absent.json").values())
        metrics.write_text(json.dumps({"manuscript_tokens": {}}))
        with pytest.raises(ValueError, match="missing or invalid"):
            loader(metrics)
        metrics.write_text("[]")
        with pytest.raises(ValueError, match="must be an object"):
            loader(metrics)


def test_noise_tokens_require_an_existing_complete_ledger(tmp_path: Path) -> None:
    path = tmp_path / "noise.json"
    with pytest.raises(FileNotFoundError, match="required noise-sweep"):
        load_noise_tokens(path)
    path.write_text('{"manuscript_tokens": {"NOISE_TABLE": "incomplete"}}')
    with pytest.raises(ValueError, match="missing or invalid"):
        load_noise_tokens(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text(
        'title: "Render contract"\ndate: "October 15, 2026"\n'
        'author:\n  - name: "Test Author"\n    orcid: "0000-0000-0000-0000"\n'
    )
    (manuscript / "references.bib").write_text("")
    (manuscript / "_render_preamble.tex").write_text("")
    (manuscript / "AGENTS.md").write_text("AGENT INSTRUCTIONS MUST NOT APPEAR")
    (manuscript / "MANUSCRIPT_STATUS.md").write_text("STATUS MUST NOT APPEAR")
    (manuscript / "00_section.md").write_text(
        "# Results {#sec:results}\n\nSee @tbl:first and @tbl:second.\n\n"
        ": First result. {#tbl:first}\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        ": Second result. {#tbl:second}\n\n| A | B |\n|---|---|\n| 3 | 4 |\n"
    )
    return manuscript, tmp_path / "rendered"


def _require_render_tools() -> None:
    if any(shutil.which(name) is None for name in ("pandoc", "pandoc-crossref", "xelatex")):
        pytest.skip("Pandoc, pandoc-crossref, and XeLaTeX are required for real render checks")


def test_real_render_numbers_tables_and_excludes_nonpaper_files(tmp_path: Path) -> None:
    _require_render_tools()
    manuscript, output = _fixture(tmp_path)
    paths = render_manuscript({}, manuscript_dir=manuscript, root=tmp_path, output_dir=output)
    assert len(paths) == 4
    plain = (output / "_build.txt").read_text()
    assert "See tbl. 1 and tbl. 2." in plain.replace("\u00a0", " ")
    assert "AGENT INSTRUCTIONS" not in plain
    assert "STATUS MUST NOT APPEAR" not in plain
    assert (output / "_build.pdf").read_bytes().startswith(b"%PDF-")
    assert not list(output.glob(".itrace-render-*"))


def test_failed_render_retains_evidence_and_previous_outputs(tmp_path: Path) -> None:
    _require_render_tools()
    manuscript, output = _fixture(tmp_path)
    output.mkdir()
    for extension in ("md", "txt", "tex", "pdf"):
        (output / f"_build.{extension}").write_bytes(b"previous accepted artifact")
    with (manuscript / "00_section.md").open("a") as stream:
        stream.write("\n\nSee @fig:absent.\n\n![Unavailable](absent.png){#fig:absent}\n")
    with pytest.raises(subprocess.CalledProcessError):
        render_manuscript({}, manuscript_dir=manuscript, root=tmp_path, output_dir=output)
    assert all(
        path.read_bytes() == b"previous accepted artifact" for path in output.glob("_build.*")
    )
    stages = list(output.glob(".itrace-render-*"))
    assert len(stages) == 1
    assert (stages[0] / "input.md").exists()
