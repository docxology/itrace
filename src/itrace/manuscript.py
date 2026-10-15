"""Cross-reference and citation integrity audit for the iTrace manuscript.

Oracle for the manuscript review pass: enumerates every label, cross-reference,
and citation across the renderable modules and reports
  - cross-references with no matching label (would render as "??"),
  - figure labels that are never referenced (dead figure anchors),
  - citation keys used but absent from references.bib (build-fatal under
    fail_on_missing), and citation keys defined but never cited.
  - manual Figure N / Table N prose that bypasses pandoc-crossref.

``scripts/audit_crossrefs.py`` orchestrates this over ``docs/manuscript/``.
The citation-key regex intentionally mirrors tests/test_manuscript_integrity.py.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import mkdtemp

RENDER_SKIP = {"README.md", "preamble.md", "AGENTS.md", "MANUSCRIPT_STATUS.md"}

LABEL_RE = re.compile(r"\{#((?:sec|fig|tbl|eq):[A-Za-z0-9_:+-]+)")
XREF_RE = re.compile(r"(?<![A-Za-z0-9_])@((?:sec|fig|tbl|eq):[A-Za-z0-9_:+-]+)")
# citation keys: @key not followed by a crossref prefix, allowing [@a; @b] and
# digitless keys. This intentionally mirrors tests/test_manuscript_integrity.py.
CITE_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_:-]+)")
BIBKEY_RE = re.compile(r"^@[a-zA-Z]+\{([^,]+),", re.MULTILINE)
MANUAL_NUMBER_RE = re.compile(r"\b(?:Figure|Fig\.|Table)\s+\d+\b")


def source_files(manuscript_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in manuscript_dir.glob("*.md")
        if not p.name.startswith("_") and p.name not in RENDER_SKIP
    )


def audit_crossrefs(manuscript_dir: Path) -> int:
    """Print the audit report; return 1 if any build-fatal problem is found."""
    labels: dict[str, str] = {}
    label_locations: dict[str, list[str]] = {}
    xrefs: dict[str, list[str]] = {}
    cites: dict[str, list[str]] = {}
    manual_numbering: list[str] = []

    for path in source_files(manuscript_dir):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for m in LABEL_RE.finditer(line):
                label = m.group(1)
                location = f"{path.name}:{line_no}"
                labels.setdefault(label, location)
                label_locations.setdefault(label, []).append(location)
        for m in XREF_RE.finditer(text):
            xrefs.setdefault(m.group(1), []).append(path.name)
        for m in CITE_RE.finditer(text):
            key = m.group(1)
            if key.split(":", 1)[0] in {"sec", "fig", "tbl", "eq"}:
                continue
            cites.setdefault(key, []).append(path.name)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if MANUAL_NUMBER_RE.search(line):
                manual_numbering.append(f"{path.name}:{line_no}: {line.strip()}")

    bib_text = (manuscript_dir / "references.bib").read_text(encoding="utf-8")
    bibkeys = set(BIBKEY_RE.findall(bib_text))

    label_names = set(labels)
    xref_names = set(xrefs)
    cite_names = set(cites)

    dangling = sorted(xref_names - label_names)
    duplicate_labels = {
        label: locations for label, locations in label_locations.items() if len(locations) > 1
    }
    dead_labels = sorted(label_names - xref_names)
    dead_figures = sorted(label for label in dead_labels if label.startswith("fig:"))
    missing_cites = sorted(cite_names - bibkeys)
    uncited = sorted(bibkeys - cite_names)

    print(
        f"labels={len(label_names)} xrefs={len(xref_names)} "
        f"cites={len(cite_names)} bibkeys={len(bibkeys)}"
    )
    print("\n[DANGLING cross-refs] (referenced, no label — renders as ??):")
    print("  " + (", ".join(dangling) if dangling else "none ✓"))
    print("\n[DUPLICATE labels] (same anchor defined more than once):")
    if duplicate_labels:
        print(
            "  "
            + "\n  ".join(
                f"{label}: {', '.join(locations)}" for label, locations in duplicate_labels.items()
            )
        )
    else:
        print("  none ✓")
    print("\n[DEAD labels] (labelled, never referenced):")
    print("  " + (", ".join(dead_labels) if dead_labels else "none ✓"))
    print("\n[DEAD figure labels] (labelled figures must be referenced in prose):")
    print("  " + (", ".join(dead_figures) if dead_figures else "none ✓"))
    print("\n[MISSING citations] (cited, absent from .bib — build-fatal):")
    print("  " + (", ".join(missing_cites) if missing_cites else "none ✓"))
    print("\n[UNCITED bib entries] (in .bib, never cited):")
    print("  " + (", ".join(uncited) if uncited else "none ✓"))
    print("\n[MANUAL figure/table numbering] (use pandoc-crossref refs instead):")
    print("  " + ("\n  ".join(manual_numbering) if manual_numbering else "none ✓"))

    fatal = bool(dangling or duplicate_labels or dead_figures or missing_cites or manual_numbering)
    print("\nRESULT:", "FAIL" if fatal else "PASS")
    return 1 if fatal else 0


TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
PILOT_TOKENS = (
    "EMPIRICAL_PILOT_STATUS",
    "EMPIRICAL_PILOT_FINITE_GAZE",
    "EMPIRICAL_PILOT_SAMPLING_HZ",
    "EMPIRICAL_PILOT_SAMPLING_CV",
    "EMPIRICAL_PILOT_DRIFT",
    "EMPIRICAL_PILOT_HELDOUT_RMS",
    "EMPIRICAL_PILOT_LATENCY",
)
SESSION_TOKENS = (
    "EMPIRICAL_SESSIONS_READINESS",
    "EMPIRICAL_SESSIONS_STATUS",
    "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL",
    "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_COUNT_ONLY",
    "EMPIRICAL_SESSIONS_CONDITIONS_REMAINING",
    "EMPIRICAL_SESSIONS_REFERENCE_REQUIREMENT",
    "EMPIRICAL_SESSIONS_FUTURE_SCOPE",
)


def substitute_tokens(text: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            msg = f"missing manuscript token value: {key}"
            raise KeyError(msg)
        return values[key]

    return TOKEN_RE.sub(replace, text)


def load_metric_tokens(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"demo_amplitude", "test_count", "coverage_pct"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        msg = "verification metrics require demo_amplitude, test_count, and coverage_pct"
        raise ValueError(msg)
    return {key.upper(): str(payload[key]) for key in required}


def _empirical_tokens(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    if not path.exists():
        return dict.fromkeys(keys, "unavailable (source ledger missing)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("manuscript_tokens"), dict):
        msg = f"{path}: manuscript_tokens must be an object"
        raise ValueError(msg)
    tokens = payload["manuscript_tokens"]
    if any(not isinstance(tokens.get(key), str) or not tokens[key].strip() for key in keys):
        msg = f"{path}: missing or invalid manuscript token values"
        raise ValueError(msg)
    return {key: tokens[key] for key in keys}


def load_empirical_tokens(path: Path) -> dict[str, str]:
    return _empirical_tokens(path, PILOT_TOKENS)


def load_empirical_session_tokens(path: Path) -> dict[str, str]:
    return _empirical_tokens(path, SESSION_TOKENS)


def validate_rendered_text(text: str) -> None:
    """Reject unresolved tokens or references; never invent replacement numbers."""
    if TOKEN_RE.search(text) or re.search(r"¿[^?\n]+\?|\[@[A-Za-z]|\?\?", text):
        msg = "rendered manuscript contains unresolved tokens or references"
        raise ValueError(msg)


def _simple_config_value(manuscript_dir: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*[\"']?(.*?)[\"']?\s*$")
    for line in (manuscript_dir / "config.yaml").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _author_string(manuscript_dir: Path) -> str | None:
    text = (manuscript_dir / "config.yaml").read_text(encoding="utf-8")
    name_match = re.search(r"^\s*-\s*name:\s*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
    if not name_match:
        return None
    name = name_match.group(1)
    orcid_match = re.search(r"^\s*orcid:\s*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
    parts = [a for a in re.findall(r'^\s*-\s*"([^"]+)"\s*$', text, re.MULTILINE) if a != name]
    if orcid_match:
        parts.append(f"ORCID: {orcid_match.group(1)}")
    return f"{name} ({'; '.join(parts)})" if parts else name


def _pandoc_base(build_input: Path, manuscript_dir: Path, root: Path) -> list[str]:
    command = [
        "pandoc",
        str(build_input),
        "--fail-if-warnings",
        "--filter",
        "pandoc-crossref",
        "--citeproc",
        "--metadata",
        "link-citations=true",
        "--bibliography",
        str(manuscript_dir / "references.bib"),
        "--resource-path",
        f"{root}:{manuscript_dir}:{root / 'output' / 'figures'}",
        "--number-sections",
    ]
    for setting in (
        "geometry:margin=0.42in",
        "colorlinks=true",
        "linkcolor=red",
        "citecolor=red",
        "urlcolor=red",
        "toccolor=red",
    ):
        command.extend(["-V", setting])
    for key in ("title", "date"):
        value = _simple_config_value(manuscript_dir, key)
        if value:
            command.extend(["--metadata", f"{key}={value}"])
    author = _author_string(manuscript_dir)
    if author:
        command.extend(["--metadata", f"author={author}"])
    return command


def render_manuscript(
    values: dict[str, str],
    *,
    manuscript_dir: Path,
    root: Path,
    output_dir: Path,
) -> list[Path]:
    """Stage and validate all four formats before replacing any reader output."""
    manuscript_dir, root, output_dir = (
        manuscript_dir.resolve(),
        root.resolve(),
        output_dir.resolve(),
    )
    if audit_crossrefs(manuscript_dir):
        msg = "manuscript source audit failed; render aborted"
        raise ValueError(msg)
    text = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in source_files(manuscript_dir))
    hydrated = substitute_tokens(text + "\n", values)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(mkdtemp(prefix=".itrace-render-", dir=output_dir))
    print(f"Render staging directory (retained on failure): {stage_dir}")
    build_input = stage_dir / "input.md"
    build_input.write_text(hydrated, encoding="utf-8")
    base = _pandoc_base(build_input, manuscript_dir, root)
    preamble = manuscript_dir / "_render_preamble.tex"
    commands = {
        "md": ["-t", "markdown"],
        "txt": ["-t", "plain"],
        "tex": ["-s", "-H", str(preamble)],
        "pdf": ["-H", str(preamble), "--pdf-engine=xelatex"],
    }
    for extension, options in commands.items():
        candidate = stage_dir / f"_build.{extension}"
        subprocess.run([*base, *options, "-o", str(candidate)], cwd=root, check=True)
        if not candidate.is_file() or not candidate.stat().st_size:
            msg = f"renderer produced an empty {extension} artifact"
            raise ValueError(msg)
        if extension != "pdf":
            validate_rendered_text(candidate.read_text(encoding="utf-8"))
    outputs = []
    for extension in commands:
        destination = output_dir / f"_build.{extension}"
        (stage_dir / destination.name).replace(destination)
        outputs.append(destination)
    shutil.rmtree(stage_dir)
    return outputs
