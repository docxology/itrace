"""Traceability checks for documentation, figures, and analysis claims."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "TRACEABILITY_MATRIX.json"

ALLOWED_CATEGORIES = {
    "algorithmic_verification",
    "analysis_output",
    "bounded_diagnostic",
    "display_only_ui",
    "external_truth_interface",
    "pure_core",
    "reproducibility_gate",
    "session_diagnostic",
}


def _load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def _path_exists(reference: str) -> bool:
    return (ROOT / reference).exists()


def _resolve_symbol(symbol: str) -> object:
    parts = symbol.split(".")
    for idx in range(len(parts), 0, -1):
        candidate_module = ".".join(parts[:idx])
        try:
            module = importlib.import_module(candidate_module)
        except ModuleNotFoundError:
            continue
        value: object = module
        for attr in parts[idx:]:
            value = getattr(value, attr)
        return value
    msg = f"could not resolve traceability symbol {symbol!r}"
    raise AssertionError(msg)


def test_traceability_matrix_schema_and_boundaries() -> None:
    payload = _load_matrix()
    assert payload["kind"] == "itrace_documentation_visualization_analysis_traceability"
    assert payload["version"] == "0.4.1"
    assert "Device-level accuracy requires" in payload["truth_boundary"]

    claim_ids: set[str] = set()
    for entry in payload["claims"]:
        assert re.fullmatch(r"[a-z0-9_]+", entry["id"])
        assert entry["id"] not in claim_ids
        claim_ids.add(entry["id"])
        assert entry["category"] in ALLOWED_CATEGORIES
        assert entry["claim"].strip()
        assert entry["boundary"].strip()
        assert entry["docs"]
        assert entry["methods"]
        assert entry["tests"]
        assert entry["evidence"]

    assert len(payload["figures"]) >= 6
    assert {fig["id"] for fig in payload["figures"]} >= {
        "graphical_abstract",
        "empirical_pilot_summary",
        "empirical_sessions_summary",
        "main_sequence",
        "quality_gallery",
        "statistical_diagnostics",
        "synthetic_empirical_range_bridge",
        "statistical_interpretation_ledger",
    }
    graphical = next(fig for fig in payload["figures"] if fig["id"] == "graphical_abstract")
    assert "MIT/GitHub availability" in graphical["boundary"]
    assert "no-reference-device-claim" in graphical["boundary"]
    assert "validating webcam accuracy" in graphical["boundary"]
    statistical = next(fig for fig in payload["figures"] if fig["id"] == "statistical_diagnostics")
    assert "bootstrap lowest-AIC winner stability" in statistical["boundary"]
    assert "posterior model probability" in statistical["boundary"]
    bridge = next(
        fig for fig in payload["figures"] if fig["id"] == "synthetic_empirical_range_bridge"
    )
    assert "observed local-scale" in bridge["boundary"]
    assert "non-comparable fields" in bridge["boundary"]
    ledger = next(
        fig for fig in payload["figures"] if fig["id"] == "statistical_interpretation_ledger"
    )
    assert "what each statistic estimates" in ledger["boundary"]
    assert "does not prove" in ledger["boundary"]
    empirical_sessions = next(
        fig for fig in payload["figures"] if fig["id"] == "empirical_sessions_summary"
    )
    assert "diagnostic-v1 readiness" in empirical_sessions["boundary"]
    assert "future backlit-condition" in empirical_sessions["boundary"]
    assert "reference-backed validation scope" in empirical_sessions["boundary"]


def test_traceability_paths_and_symbols_exist() -> None:
    payload = _load_matrix()
    for entry in [*payload["claims"], *payload["figures"]]:
        for key in ("docs", "methods", "tests", "evidence", "source_data"):
            for reference in entry.get(key, []):
                assert _path_exists(reference), f"{entry['id']} missing {key}: {reference}"
        if "path" in entry:
            assert _path_exists(entry["path"]), f"{entry['id']} missing figure path"
        if "script" in entry:
            assert _path_exists(entry["script"]), f"{entry['id']} missing script"
        for symbol in entry.get("method_symbols", []):
            assert _resolve_symbol(symbol) is not None


def test_public_docs_link_to_traceability_matrix() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs" / "VALIDATION_PROTOCOLS.md").read_text(encoding="utf-8")
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    isa = (ROOT / "ISA.md").read_text(encoding="utf-8")
    for text in (readme, validation, todo, isa):
        assert "TRACEABILITY_MATRIX.json" in text
    assert "browser canvas/svg is display-only" in re.sub(r"\s+", " ", validation.lower())
    assert "not a reference-device validation" in readme


def test_public_docs_distinguish_transient_eye_crop_from_persisted_exports() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs" / "VALIDATION_PROTOCOLS.md").read_text(encoding="utf-8")
    capture_section = (ROOT / "manuscript" / "02f_capture_shell.md").read_text(encoding="utf-8")
    combined = re.sub(r"\s+", " ", "\n".join([readme, validation, capture_section]).lower())
    assert "transient eye-crop jpeg" in combined
    assert "persisted eye-crop images" in combined
    assert "raw eye video" in combined
    assert "separate consent/privacy workflow" in combined


def test_traceability_figures_are_referenced_or_documented() -> None:
    payload = _load_matrix()
    public_docs = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [
            ROOT / "README.md",
            ROOT / "docs" / "VALIDATION_PROTOCOLS.md",
            ROOT / "ISA.md",
            ROOT / "manuscript" / "_build.md",
        ]
    )
    for figure in payload["figures"]:
        basename = Path(figure["path"]).name
        assert basename in public_docs or figure["id"] == "quality_gallery", basename
