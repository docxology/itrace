"""Figure-generation test (ISC-51). Skips cleanly if matplotlib is absent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (figures extra) not installed",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "manuscript"
FIGURE_MANIFEST = ROOT / "docs" / "figure_manifest.json"


def _manuscript_figure_paths() -> set[str]:
    paths: set[str] = set()
    for path in MANUSCRIPT.glob("*.md"):
        if path.name.startswith("_") or path.name in {"README.md", "preamble.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"!\[[^\]]*\]\((../output/figures/[^)]+)\)", text):
            paths.add(str((MANUSCRIPT / link).resolve().relative_to(ROOT)))
    return paths


def test_generate_main_sequence_png(tmp_path) -> None:  # ISC-51
    import generate_figures

    path = generate_figures.generate_main_sequence(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 1000  # a real PNG, not empty


def test_generate_direction_polar_png(tmp_path) -> None:  # ISC-51
    import generate_figures

    path = generate_figures.generate_direction_polar(tmp_path)
    assert path.exists()
    assert path.suffix == ".png"


def test_generate_cover_and_graphical_abstract_pngs(tmp_path) -> None:
    import matplotlib.image as mpimg

    import generate_graphical_abstract

    cover = generate_graphical_abstract.generate_cover_visual(tmp_path)
    abstract = generate_graphical_abstract.generate_graphical_abstract(tmp_path)
    assert cover.exists()
    assert abstract.exists()
    assert cover.stat().st_size > 1000
    assert abstract.stat().st_size > 1000
    image = mpimg.imread(abstract)
    height, width = image.shape[:2]
    assert width >= 3000
    assert height >= 1600
    assert abs((width / height) - (16 / 9)) < 0.02
    assert (
        generate_graphical_abstract.AVAILABILITY_FOOTER
        == "MIT License | github.com/docxology/itrace | DOI 10.5281/zenodo.20614027"
    )
    assert generate_graphical_abstract.NO_REFERENCE_DEVICE_BOUNDARY == "no reference-device claim"


def test_publication_figure_manifest_covers_manuscript_figures() -> None:
    import matplotlib.image as mpimg

    payload = json.loads(FIGURE_MANIFEST.read_text(encoding="utf-8"))
    assert payload["kind"] == "itrace_publication_figure_manifest"
    metrics = json.loads((ROOT / "docs" / "verification_metrics.json").read_text())
    assert payload["version"] == metrics["version"]
    assert payload["updated"] == metrics["gate_date"]
    entries = {entry["path"]: entry for entry in payload["figures"]}
    assert len(entries) == len(payload["figures"])
    statistical_entry = entries["output/figures/statistical_diagnostics.png"]
    assert "output/figures/statistical_diagnostics.json" in statistical_entry["source_data"]
    graphical_entry = entries["output/figures/graphical_abstract.png"]
    assert graphical_entry["script"] == "scripts/generate_graphical_abstract.py"
    assert "docs/verification_metrics.json" in graphical_entry["source_data"]
    assert "docs/empirical_sessions_summary.json" in graphical_entry["source_data"]
    assert "LICENSE" in graphical_entry["source_data"]
    assert "CITATION.cff" in graphical_entry["source_data"]
    bridge_entry = entries["output/figures/synthetic_empirical_range_bridge.png"]
    assert "output/figures/synthetic_empirical_range_bridge.json" in bridge_entry["source_data"]
    sessions_entry = entries["output/figures/empirical_sessions_summary.png"]
    assert "docs/empirical_sessions_manifest.json" in sessions_entry["source_data"]
    assert "docs/empirical_sessions_summary.json" in sessions_entry["source_data"]
    bridge_json_entry = entries["output/figures/synthetic_empirical_range_bridge.json"]
    assert "docs/empirical_pilot_metrics.json" in bridge_json_entry["source_data"]
    assert "output/synthetic_validation.json" in bridge_json_entry["source_data"]
    ledger_entry = entries["output/figures/statistical_interpretation_ledger.png"]
    assert "output/figures/statistical_interpretation_ledger.json" in ledger_entry["source_data"]
    ledger_json_entry = entries["output/figures/statistical_interpretation_ledger.json"]
    assert "output/figures/statistical_diagnostics.json" in ledger_json_entry["source_data"]
    assert (
        "output/figures/synthetic_empirical_range_bridge.json" in ledger_json_entry["source_data"]
    )
    assert "src/itrace/stats/evidence.py" in ledger_json_entry["source_data"]
    missing = sorted(_manuscript_figure_paths() - set(entries))
    assert missing == []

    for figure_path, entry in entries.items():
        path = ROOT / figure_path
        assert path.exists(), figure_path
        assert path.is_relative_to(ROOT / "output" / "figures")
        assert entry["artifact_kind"] in {"image", "animation", "table", "data", "artifact"}
        assert entry["script"].startswith("scripts/")
        assert (ROOT / str(entry["script"])).exists()
        assert entry["source_data"]
        for source in entry["source_data"]:
            assert not Path(str(source)).is_absolute()
            assert (ROOT / str(source)).exists(), f"{figure_path} source missing: {source}"
        assert entry["bytes"] == path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["sha256"] == digest
        if entry["artifact_kind"] in {"image", "animation"}:
            assert entry["width_px"] >= 900
            assert entry["height_px"] >= 450
            assert entry["pixel_std"] > 0.005
        if entry["artifact_kind"] == "image":
            image = mpimg.imread(path)
            assert image.shape[1] == entry["width_px"]
            assert image.shape[0] == entry["height_px"]
        if entry["artifact_kind"] == "animation":
            assert entry.get("frame_count", 1) >= 1

    if (ROOT / "docs" / "empirical_pilot_metrics.json").exists():
        empirical = json.loads((ROOT / "docs" / "empirical_pilot_metrics.json").read_text())
        if empirical.get("available"):
            source_report = empirical["source_report"]
            source_data = entries["output/figures/empirical_pilot_summary.png"]["source_data"]
            assert source_report in source_data
            bridge_source_data = entries["output/figures/synthetic_empirical_range_bridge.json"][
                "source_data"
            ]
            assert source_report in bridge_source_data
            ledger_source_data = entries["output/figures/statistical_interpretation_ledger.json"][
                "source_data"
            ]
            assert source_report in ledger_source_data

    public_artifacts = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "output" / "figures").iterdir()
        if path.suffix.lower() in {".png", ".gif", ".md", ".json"}
    }
    assert sorted(public_artifacts - set(entries)) == []
