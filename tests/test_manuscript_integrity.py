"""Manuscript structural integrity checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
METRICS = ROOT / "docs" / "verification_metrics.json"
NON_RENDERED_MARKDOWN = {"README.md", "preamble.md"}


def _source_markdown() -> list[Path]:
    return sorted(
        p
        for p in MANUSCRIPT.glob("*.md")
        if not p.name.startswith("_") and p.name not in NON_RENDERED_MARKDOWN
    )


def _verification_metrics() -> dict[str, object]:
    return json.loads(METRICS.read_text())


def _public_doc_haystack() -> str:
    return "\n".join(
        [
            (ROOT / "README.md").read_text(),
            (ROOT / "ISA.md").read_text(),
            *(
                path.read_text()
                for path in sorted((ROOT / "docs").glob("*.md"))
                if path.name != "SCHOLARSHIP_AUDIT.md"
            ),
            (MANUSCRIPT / "README.md").read_text(),
            *(path.read_text() for path in _source_markdown()),
        ]
    )


def _normalise_noise_table(text: str) -> list[str]:
    lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("|") and "sigma" in line.lower():
            in_table = True
        if in_table and line.startswith("|"):
            lines.append(
                line.replace("σ", "sigma").replace("±", "+/-").replace(" px@640", " px@640").strip()
            )
            continue
        if in_table:
            break
    return lines


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


def test_manuscript_labels_are_unique() -> None:
    locations: dict[str, list[str]] = {}
    for path in _source_markdown():
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            for label in re.findall(r"\{#((?:sec|fig|tbl|eq):[-A-Za-z0-9_]+)(?:\s+[^}]*)?\}", line):
                locations.setdefault(label, []).append(f"{path.name}:{line_no}")
    duplicates = {label: refs for label, refs in sorted(locations.items()) if len(refs) > 1}
    assert duplicates == {}


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


def test_abstract_is_citation_free_and_declares_availability() -> None:
    abstract = (MANUSCRIPT / "00_abstract.md").read_text(encoding="utf-8")
    normalized_abstract = re.sub(r"\s+", " ", abstract)
    assert "[@" not in abstract
    assert re.search(r"(?<![\w])@[A-Za-z][A-Za-z0-9_:-]+", abstract) is None
    for required in (
        "MIT-licensed and openly released",
        "https://github.com/docxology/itrace",
        "10.5281/zenodo.20614027",
    ):
        assert required in normalized_abstract

    rendered = (MANUSCRIPT / "_build.md").read_text(encoding="utf-8")
    rendered_abstract = rendered.split("# Abstract", 1)[1].split("# Introduction", 1)[0]
    normalized_rendered_abstract = re.sub(r"\s+", " ", rendered_abstract)
    assert "[@" not in rendered_abstract
    assert "#ref-" not in rendered_abstract
    assert "Salvucci and Goldberg" not in rendered_abstract
    assert "Engbert and Kliegl" not in rendered_abstract
    assert "Bahill et al." not in rendered_abstract
    assert "Kronemer et al." not in rendered_abstract
    assert "MIT-licensed and openly released" in normalized_rendered_abstract
    assert "https://github.com/docxology/itrace" in normalized_rendered_abstract
    assert "10.5281/zenodo.20614027" in normalized_rendered_abstract


def test_manuscript_cover_uses_release_date() -> None:
    config = (MANUSCRIPT / "config.yaml").read_text(encoding="utf-8")
    tex = (MANUSCRIPT / "_build.tex").read_text(encoding="utf-8")
    assert 'date: "June 9, 2026"' in config
    assert r"\date{June 9, 2026}" in tex
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext is required for rendered cover-date verification")
    pdf_text = subprocess.run(
        ["pdftotext", str(MANUSCRIPT / "_build.pdf"), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "June 9, 2026" in pdf_text


def test_bibliography_metadata_is_publication_ready() -> None:
    bib = (MANUSCRIPT / "references.bib").read_text()
    source = "\n".join(path.read_text() for path in _source_markdown())
    assert " and others" not in bib
    assert 'Nystr{"o}m' not in bib
    assert "boccignone2011generative" not in bib
    assert "boccignone2004gaze" in bib
    assert "@boccignone2011generative" not in source


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


def test_manuscript_figures_are_referenced_in_prose() -> None:
    text_by_file = {path: path.read_text() for path in _source_markdown()}
    labels: set[str] = set()
    refs: set[str] = set()
    manual_numbering: list[str] = []
    for path, text in text_by_file.items():
        labels.update(re.findall(r"\{#(fig:[-A-Za-z0-9_]+)(?:\s+[^}]*)?\}", text))
        refs.update(re.findall(r"@((?:fig):[-A-Za-z0-9_]+)", text))
        for line_no, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\b(?:Figure|Fig\.|Table)\s+\d+\b", line):
                manual_numbering.append(f"{path.name}:{line_no}: {line}")
    assert sorted(labels - refs) == []
    assert manual_numbering == []


def test_manuscript_cover_author_and_layout_contract() -> None:
    config = (MANUSCRIPT / "config.yaml").read_text()
    assert "0000-0001-6232-9096" in config
    assert "FractAI" not in config
    assert "margin=0.42in" in config
    assert (ROOT / "output" / "figures" / "cover_visual.png").exists()
    assert (ROOT / "output" / "figures" / "graphical_abstract.png").exists()
    assert (ROOT / "output" / "figures" / "statistical_diagnostics.png").exists()
    assert (ROOT / "output" / "figures" / "synthetic_empirical_range_bridge.png").exists()
    assert (ROOT / "output" / "figures" / "synthetic_empirical_range_bridge.json").exists()
    assert (ROOT / "output" / "figures" / "statistical_interpretation_ledger.png").exists()
    assert (ROOT / "output" / "figures" / "statistical_interpretation_ledger.json").exists()
    assert (ROOT / "docs" / "empirical_pilot_metrics.json").exists()
    assert (ROOT / "output" / "figures" / "empirical_pilot_summary.png").exists()


def test_rendered_latex_uses_red_hyperlink_contract() -> None:
    config = (MANUSCRIPT / "config.yaml").read_text()
    rendered_tex = (MANUSCRIPT / "_build.tex").read_text(errors="ignore")
    renderer = (ROOT / "scripts" / "render_manuscript.py").read_text()
    for required in (
        "colorlinks: true",
        "linkcolor: red",
        "citecolor: red",
        "urlcolor: red",
        "toccolor: red",
    ):
        assert required in config
    for required in (
        "colorlinks=true",
        "linkcolor={red}",
        "citecolor={red}",
        "urlcolor={red}",
    ):
        assert required in rendered_tex
    assert "toccolor=red" in renderer


def test_public_docs_do_not_carry_stale_metrics() -> None:
    haystack = _public_doc_haystack()
    metrics = _verification_metrics()
    current_tests = f"{metrics['test_count']} tests"
    current_coverage = str(metrics["coverage_pct"])
    current_test_count = int(metrics["test_count"])
    stale_tests = sorted(
        {
            int(match)
            for match in re.findall(r"\b(\d{2,4})\s+tests\b", haystack)
            if int(match) != current_test_count
        }
    )
    stale_coverage = sorted(
        {
            match
            for match in re.findall(r"(?<!\d)(?:8\d|9\d|100)\.\d{2}%", haystack)
            if match != current_coverage
        }
    )
    stale_test_constants = sorted(
        {
            int(match)
            for match in re.findall(r"\bTEST_COUNT\s*=\s*(\d{2,4})\b", haystack)
            if int(match) != current_test_count
        }
    )
    assert stale_tests == []
    assert stale_coverage == []
    assert stale_test_constants == []
    assert current_tests in haystack
    assert current_coverage in haystack


def test_readme_points_to_metric_ledger_instead_of_hardcoding_gate_metrics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/verification_metrics.json" in readme
    assert re.findall(r"\b\d{2,4}\s+tests\b", readme) == []
    assert re.findall(r"(?<!\d)(?:8\d|9\d|100)\.\d{2}%", readme) == []


def test_public_claim_surfaces_reject_stale_or_overclaim_phrases() -> None:
    metrics = _verification_metrics()
    current_test_count = int(metrics["test_count"])
    current_coverage = str(metrics["coverage_pct"])
    claim_paths = [
        ROOT / "README.md",
        ROOT / "ISA.md",
        ROOT / "pyproject.toml",
        ROOT / "CITATION.cff",
        ROOT / ".zenodo.json",
        *(
            path
            for path in sorted((ROOT / "docs").glob("*.md"))
            if path.name != "SCHOLARSHIP_AUDIT.md"
        ),
        *sorted((ROOT / "docs" / "releases").glob("*.md")),
        *(MANUSCRIPT / name for name in ("_build.md", "_build.txt", "_build.tex")),
        *(path for path in _source_markdown()),
    ]
    haystack = "\n".join(path.read_text(errors="ignore") for path in claim_paths)
    stale_test_claims = sorted(
        {
            int(match)
            for match in re.findall(r"\b(\d{2,4})\s+tests\b", haystack)
            if int(match) != current_test_count
        }
    )
    stale_pytest_claims = sorted(
        {
            int(match)
            for match in re.findall(r"\b(\d{2,4})\s+passed\b", haystack)
            if int(match) >= 500 and int(match) != current_test_count
        }
    )
    stale_coverage_claims = sorted(
        {
            match
            for match in re.findall(r"(?<!\d)(?:8\d|9\d|100)\.\d{2}%", haystack)
            if match != current_coverage
        }
    )
    forbidden_patterns = {
        # Scientific-honesty boundary that holds for every release: the toolkit
        # is algorithmically verified, never device-accuracy validated.
        r"\b(?:validates|validated|proves|establishes)\s+"
        r"(?:webcam|device|real[- ]eye|gaze|pupil)[^.]{0,80}\baccuracy\b": (
            "unsupported accuracy validation"
        ),
    }
    failures = {
        label: pattern
        for pattern, label in forbidden_patterns.items()
        if re.search(pattern, haystack, flags=re.IGNORECASE)
    }
    assert stale_test_claims == []
    assert stale_pytest_claims == []
    assert stale_coverage_claims == []
    assert failures == {}
    assert f"{current_test_count} tests" in haystack
    assert current_coverage in haystack
    assert "v0.4.1" in haystack
    assert "v0.4.0" not in haystack
    assert "MIT-licensed and openly released" in haystack
    assert "10.5281/zenodo.20614027" in haystack
    assert "ready for diagnostic v1" in haystack
    assert "0 reference-backed rows" in haystack


def test_verification_metric_summaries_match_current_file_surface() -> None:
    metrics = _verification_metrics()
    ruff_files = sum(
        1
        for root in ("src", "tests", "scripts")
        for path in (ROOT / root).rglob("*.py")
        if path.is_file()
    )
    mypy_files = sum(1 for path in (ROOT / "src" / "itrace").rglob("*.py") if path.is_file())
    assert metrics["ruff_format_summary"] == f"{ruff_files} files already formatted"
    assert metrics["mypy_summary"] == f"Success: no issues found in {mypy_files} source files"

    thermo = (ROOT / "docs" / "THERMO_NUCLEAR_AUDIT.md").read_text(encoding="utf-8")
    assert f"{mypy_files} source files" in thermo
    assert "53 source files" not in thermo


def test_pyproject_urls_point_to_itrace_repo() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    urls = pyproject["project"]["urls"]
    metrics = _verification_metrics()
    repository = metrics["repository"]
    assert all("docxology/template" not in value for value in urls.values())
    assert urls["Homepage"] == repository
    assert urls["Repository"] == repository
    assert urls["Brief"] == repository


def test_release_license_and_citation_metadata_are_consistent() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    metrics = _verification_metrics()
    repository = str(metrics["repository"])
    license_name = str(metrics["license"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    release_notes = (ROOT / "docs" / "releases" / "v0.4.1.md").read_text(encoding="utf-8")
    abstract = (MANUSCRIPT / "00_abstract.md").read_text(encoding="utf-8")

    assert license_name == "MIT"
    assert pyproject["project"]["license"]["text"] == license_name
    assert "License :: OSI Approved :: MIT License" in pyproject["project"]["classifiers"]
    assert pyproject["project"]["urls"]["Repository"] == repository
    assert f'version: "{pyproject["project"]["version"]}"' in citation
    assert "license: MIT" in citation
    assert f'repository-code: "{repository}"' in citation
    assert f'url: "{repository}"' in citation
    assert 'date-released: "2026-06-09"' in citation
    assert zenodo["version"] == pyproject["project"]["version"]
    assert zenodo["license"] == "mit"
    assert zenodo["publication_date"] == "2026-06-09"
    assert zenodo["access_right"] == "open"
    assert zenodo["upload_type"] == "software"
    assert any(
        item.get("identifier") == repository
        for item in zenodo.get("related_identifiers", [])
        if isinstance(item, dict)
    )
    assert "reference-device accuracy" in zenodo["description"]
    assert "webcam-accuracy validation" in zenodo["description"]
    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text
    assert "Copyright (c) 2026 Daniel Ari Friedman" in license_text
    assert "iTrace v0.4.1" in release_notes
    assert repository in release_notes
    assert "10.5281/zenodo.20614027" in release_notes
    assert "10.5281/zenodo.20614026" in release_notes

    for text in (readme, citation, abstract, release_notes):
        normalized_text = re.sub(r"\s+", " ", text)
        assert repository in normalized_text
        assert "MIT" in normalized_text


def test_prepublication_review_matches_current_ledgers() -> None:
    metrics = _verification_metrics()
    sessions = json.loads((ROOT / "docs" / "empirical_sessions_summary.json").read_text())
    review = (ROOT / "docs" / "PREPUBLICATION_REVIEW.md").read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", review)

    assert "# Final pre-publication RedTeam/Science review" in review
    assert str(metrics["version"]) in normalized
    assert str(metrics["gate_date"]) in normalized
    assert f"{metrics['test_count']} tests" in normalized
    assert str(metrics["coverage_pct"]) in normalized
    assert str(metrics["license"]) in normalized
    assert str(metrics["repository"]) in normalized
    assert f"{sessions['available_session_count']} available sessions" in normalized
    assert f"{sessions['replicate_count']} replicate IDs" in normalized
    assert f"{sessions['condition_count']} conditions" in normalized
    assert f"{sessions['reference_evidence_count']} reference-backed rows" in normalized
    assert "archived at Zenodo DOI 10.5281/zenodo.20614027" in normalized
    assert "not device validation or webcam accuracy" in normalized
    assert "does not approve public claims of webcam/device accuracy" in normalized
    assert "raw eye video and persisted eye crops remain outside the default workflow" in normalized
    assert "Visualization and claim sweep" in normalized


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
    assert "{=tex}" not in haystack
    assert "at ~." not in haystack
    assert r"\SI{" not in (MANUSCRIPT / "_build.md").read_text(errors="ignore")
    assert "0000-0001-6232-9096" in haystack
    assert "FractAI" not in haystack
    assert "margin=0.42in" in haystack
    assert "cover_visual.png" in haystack
    assert "Figure 1: Graphical abstract" in haystack
    assert "statistical_diagnostics.png" in haystack
    assert "AIC" in haystack
    assert "bootstrap lowest-AIC winner stability" in haystack
    assert "not as posterior model probabilities" in haystack
    assert "Levenshtein" in haystack
    assert "Results: single-pilot empirical diagnostics" in haystack
    assert "empirical_pilot_summary.png" in haystack
    assert "synthetic_empirical_range_bridge.png" in haystack
    assert "Synthetic-to-empirical range bridge" in haystack
    assert "not directly comparable" in haystack
    assert "statistical_interpretation_ledger.png" in haystack
    assert "Statistical interpretation ledger" in haystack
    assert "does not prove" in haystack


def test_rendered_artifacts_match_metric_ledgers() -> None:
    metrics = _verification_metrics()
    empirical = json.loads((ROOT / "docs" / "empirical_pilot_metrics.json").read_text())
    rendered_paths = [
        MANUSCRIPT / "_build.md",
        MANUSCRIPT / "_build.txt",
        MANUSCRIPT / "_build.tex",
    ]
    haystack = "\n".join(path.read_text(errors="ignore") for path in rendered_paths)
    pdf_text = ""
    if shutil.which("pdftotext") and (MANUSCRIPT / "_build.pdf").exists():
        pdf_text = subprocess.run(
            ["pdftotext", str(MANUSCRIPT / "_build.pdf"), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    combined = haystack + "\n" + pdf_text

    expected_tests = f"{metrics['test_count']} tests"
    expected_coverage = str(metrics["coverage_pct"])
    assert expected_tests in combined
    assert expected_coverage in combined
    stale_tests = sorted(
        {
            int(match)
            for match in re.findall(r"\b(\d{2,4})\s+tests\b", combined)
            if int(match) != int(metrics["test_count"])
        }
    )
    stale_coverage = sorted(
        {
            match
            for match in re.findall(r"(?<!\d)(?:8\d|9\d|100)\.\d{2}%", combined)
            if match != expected_coverage
        }
    )
    assert stale_tests == []
    assert stale_coverage == []

    tokens = empirical.get("manuscript_tokens", {})
    assert isinstance(tokens, dict)
    normalized_combined = re.sub(r"\s+", " ", combined)
    for value in tokens.values():
        assert str(value) in normalized_combined


def test_rendered_pdf_smoke_matches_public_contract() -> None:
    pdf = MANUSCRIPT / "_build.pdf"
    if not shutil.which("pdfinfo") or not shutil.which("pdftotext"):
        pytest.skip("pdfinfo/pdftotext not installed")
    assert pdf.exists()
    info = subprocess.run(
        ["pdfinfo", str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
    assert pages_match is not None
    assert int(pages_match.group(1)) >= 20
    encrypted_match = re.search(r"^Encrypted:\s+(.+)$", info, flags=re.MULTILINE)
    if encrypted_match is not None:
        assert encrypted_match.group(1).strip().lower() == "no"
    assert pdf.stat().st_size > 1_000_000

    text = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    metrics = _verification_metrics()
    empirical = json.loads((ROOT / "docs" / "empirical_pilot_metrics.json").read_text())
    tokens = empirical.get("manuscript_tokens", {})
    assert "iTrace" in text
    assert "0000-0001-6232-9096" in text
    assert f"{metrics['test_count']} tests" in text
    assert str(metrics["coverage_pct"]) in text
    assert "N=1" in text
    assert "order-of-magnitude" in text
    assert "Synthetic-to-empirical range bridge" in text
    assert "not directly comparable" in text
    if isinstance(tokens, dict):
        for key in (
            "EMPIRICAL_PILOT_FINITE_GAZE",
            "EMPIRICAL_PILOT_SAMPLING_CV",
            "EMPIRICAL_PILOT_HELDOUT_RMS",
            "EMPIRICAL_PILOT_LATENCY",
        ):
            assert str(tokens[key]) in text
    forbidden = ("{{", "}}", "??", "[@", "Citation", "pending local recording")
    for marker in forbidden:
        assert marker not in text


def test_noise_table_matches_generated_summary() -> None:
    generated = (ROOT / "output" / "figures" / "noise_summary.md").read_text()
    manuscript = (MANUSCRIPT / "03d_noise_sensitivity.md").read_text()
    assert _normalise_noise_table(manuscript) == _normalise_noise_table(generated)


def test_pending_empirical_values_are_hydrated_honestly() -> None:
    metrics_path = ROOT / "docs" / "empirical_pilot_metrics.json"
    payload = json.loads(metrics_path.read_text())
    if payload.get("available"):
        return
    haystack = "\n".join(
        path.read_text(errors="ignore")
        for path in (MANUSCRIPT / "_build.md", MANUSCRIPT / "_build.txt", MANUSCRIPT / "_build.tex")
    )
    normalized = re.sub(r"[*\\{}]+", "", haystack)
    normalized = re.sub(r"\s+", " ", normalized)
    assert "pending local recording" in haystack
    assert "finite-gaze fraction unavailable" in normalized
    assert "held-out target RMS unavailable" in normalized
    assert "target acquisition latency unavailable" in normalized


def test_available_empirical_pilot_is_framed_as_n1_diagnostic() -> None:
    metrics_path = ROOT / "docs" / "empirical_pilot_metrics.json"
    payload = json.loads(metrics_path.read_text())
    if not payload.get("available"):
        return
    rendered_paths = [
        MANUSCRIPT / "_build.md",
        MANUSCRIPT / "_build.txt",
        MANUSCRIPT / "_build.tex",
    ]
    haystack = "\n".join(path.read_text(errors="ignore") for path in rendered_paths)
    normalized = re.sub(r"\s+", " ", haystack)
    tokens = payload["manuscript_tokens"]
    session_summary = json.loads((ROOT / "docs" / "empirical_sessions_summary.json").read_text())
    session_tokens = session_summary["manuscript_tokens"]

    for token in (
        "EMPIRICAL_PILOT_STATUS",
        "EMPIRICAL_PILOT_FINITE_GAZE",
        "EMPIRICAL_PILOT_SAMPLING_HZ",
        "EMPIRICAL_PILOT_SAMPLING_CV",
        "EMPIRICAL_PILOT_DRIFT",
        "EMPIRICAL_PILOT_HELDOUT_RMS",
        "EMPIRICAL_PILOT_LATENCY",
    ):
        assert tokens[token] in normalized

    for token in (
        "EMPIRICAL_SESSIONS_STATUS",
        "EMPIRICAL_SESSIONS_READINESS",
        "EMPIRICAL_SESSIONS_FUTURE_SCOPE",
    ):
        assert session_tokens[token] in normalized

    assert "N=1" in haystack
    assert "order-of-magnitude" in haystack
    assert "case study" in normalized
    assert "five-session diagnostic pilot" in normalized
    assert "ready for diagnostic v1" in normalized
    assert (
        "The stronger 12-session, three-condition, reference-backed validation plan" in normalized
    )
    assert "future scope rather than counted as a current v1 blocker" in normalized
    assert "Synthetic truth recovery" in normalized
    assert "practical scale at which the package's live workflow operates" in normalized
    assert "physiological saccade latency or reference-device gaze accuracy" in normalized
    assert "N=1 pilot scale compared with synthetic/model variables" in normalized
    assert "Synthetic-to-empirical range bridge" in normalized
    assert "generated JSON sidecar" in normalized
    assert "observed local scale" in normalized
    assert "stress-domain only" in normalized
    assert "not directly comparable" in normalized
    assert "sanity-checks one local operating scale for synthetic data generation" in normalized
    assert "webcam-rate synthetic/session examples" in normalized
    assert "timestamp jitter in synthetic sessions" in normalized
    assert "not iid landmark noise, closed-loop residual, or gaze accuracy" in normalized
    assert "contextualized by a local demonstration" in normalized
    assert "tested against future reference data" in normalized
    assert "empirically anchored" not in normalized
    assert "before the clean local pilot is recorded" not in normalized
    assert "fig. 7 remains an explicit pending-data artifact" not in normalized
    assert "readiness gate must therefore remain blocked" not in normalized
    assert "validates accuracy" not in normalized
    assert "validates device performance" not in normalized
    assert "validates device accuracy" not in normalized


def test_scholarship_brief_avoids_unsupported_superlatives() -> None:
    brief = (ROOT / "docs" / "RESEARCH_BRIEF.md").read_text(encoding="utf-8")
    forbidden = (
        "rivals commercial systems",
        "universal entry point",
        "most rigorously validated open-source pupillometry tool",
        "using nothing but a standard consumer webcam",
        "most immediately accessible",
        "canonical tool",
        "outperforms fixed-threshold",
        "richest suite",
        "primary option",
        "standard choice",
        "most capable option",
        "fastest path",
        "first large-scale publicly available dataset",
    )
    for phrase in forbidden:
        assert phrase not in brief
    assert "discovery/non-evidence context" in brief


def test_scholarship_audit_matches_bibliography_source_urls() -> None:
    audit = (ROOT / "docs" / "SCHOLARSHIP_AUDIT.md").read_text(encoding="utf-8")
    bib = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    assert "https://www.ijcai.org/Proceedings/16/Papers/540.pdf" in audit
    assert "https://www.ijcai.org/Proceedings/16/Papers/540.pdf" in bib
    assert "https://www.ijcai.org/Proceedings/16/Papers/522.pdf" not in audit
    assert "10.3758/s13428-024-02545-7" in audit
    assert "10.3758/s13428-024-02545-7" in bib
    assert "10.3758/s13428-024-02374-8" not in audit
