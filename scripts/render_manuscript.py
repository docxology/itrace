"""Standalone manuscript renderer for the working/iTrace checkout.

The sibling template pipeline is the canonical renderer after promotion to
``active/``. This script keeps the working project honest before promotion by
doing the same local essentials: concatenate renderable manuscript modules,
substitute explicit gate-derived tokens, and run Pandoc with pandoc-crossref
before citeproc.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
DEFAULT_METRICS = ROOT / "docs" / "verification_metrics.json"
DEFAULT_EMPIRICAL_METRICS = ROOT / "docs" / "empirical_pilot_metrics.json"
DEFAULT_EMPIRICAL_SESSIONS = ROOT / "docs" / "empirical_sessions_summary.json"
RENDER_SKIP = {"README.md", "preamble.md"}


def _source_files() -> list[Path]:
    return sorted(
        path
        for path in MANUSCRIPT.glob("*.md")
        if not path.name.startswith("_") and path.name not in RENDER_SKIP
    )


def _substitute_tokens(text: str, values: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return values[key]
        except KeyError as exc:
            raise KeyError(f"missing manuscript token value: {key}") from exc

    return re.sub(r"\{\{([A-Z][A-Z0-9_]*)\}\}", replace, text)


def _write_build_input(values: dict[str, str]) -> Path:
    sections = []
    for path in _source_files():
        sections.append(path.read_text(encoding="utf-8").strip())
    rendered = _substitute_tokens("\n\n".join(sections) + "\n", values)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z][A-Z0-9_]*)\}\}", rendered)))
    if unresolved:
        raise ValueError(f"unresolved manuscript tokens: {', '.join(unresolved)}")
    out = MANUSCRIPT / "_build_input.md"
    out.write_text(rendered, encoding="utf-8")
    return out


def _simple_config_value(key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*[\"']?(.*?)[\"']?\s*$")
    for line in (MANUSCRIPT / "config.yaml").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _author_string() -> str | None:
    """Flatten the structured config.yaml author to a one-line ``\\author{...}``.

    config.yaml stores the author as a structured block (name / orcid /
    affiliations) for the canonical template pipeline, which renders it richly.
    The standalone working render uses pandoc's *default* LaTeX template, whose
    ``\\author`` slot takes a plain string — so without this flattening the title
    page renders authorless. We emit ``Name (Affiliation; ORCID: id)`` from the
    first author entry; passing the metadata also populates ``pdfauthor``.
    """
    text = (MANUSCRIPT / "config.yaml").read_text(encoding="utf-8")
    name_match = re.search(r"^\s*-\s*name:\s*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
    if not name_match:
        return None
    name = name_match.group(1)
    orcid_match = re.search(r"^\s*orcid:\s*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
    orcid = orcid_match.group(1) if orcid_match else ""
    # Quoted bare list items are the affiliation entries (the `- name:` line and
    # the unquoted `- margin=...` geometry item do not match this shape).
    affiliations = [
        a for a in re.findall(r'^\s*-\s*"([^"]+)"\s*$', text, re.MULTILINE) if a != name
    ]
    parts = [*affiliations]
    if orcid:
        parts.append(f"ORCID: {orcid}")
    return f"{name} ({'; '.join(parts)})" if parts else name


def _pandoc_base(build_md: Path) -> list[str]:
    command = [
        "pandoc",
        str(build_md),
        "--filter",
        "pandoc-crossref",
        "--citeproc",
        # Hyperlink each in-text citation to its bibliography entry so citecolor
        # applies (otherwise author-date cites render as plain black text).
        "--metadata",
        "link-citations=true",
        "--bibliography",
        str(MANUSCRIPT / "references.bib"),
        "--resource-path",
        f"{ROOT}:{MANUSCRIPT}",
        "--number-sections",
        # Page + link styling (template-idiomatic pandoc variables): tight 0.42in
        # margins and red hyperlinks / citations / cross-references. Mirrors the
        # keys in config.yaml so the canonical template pipeline renders the same.
        "-V",
        "geometry:margin=0.42in",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=red",
        "-V",
        "citecolor=red",
        "-V",
        "urlcolor=red",
        "-V",
        "toccolor=red",
    ]
    for key in ("title", "date"):
        value = _simple_config_value(key)
        if value:
            command.extend(["--metadata", f"{key}={value}"])
    author = _author_string()
    if author:
        command.extend(["--metadata", f"author={author}"])
    return command


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def render(values: dict[str, str]) -> None:
    build_input = _write_build_input(values)
    try:
        base = _pandoc_base(build_input)
        _run([*base, "-t", "markdown", "-o", str(MANUSCRIPT / "_build.md")])
        _run([*base, "-t", "plain", "-o", str(MANUSCRIPT / "_build.txt")])
        _run(
            [
                *base,
                "-s",
                "-H",
                str(MANUSCRIPT / "_render_preamble.tex"),
                "-o",
                str(MANUSCRIPT / "_build.tex"),
            ]
        )
        _run(
            [
                *base,
                "-H",
                str(MANUSCRIPT / "_render_preamble.tex"),
                "--pdf-engine=xelatex",
                "-o",
                str(MANUSCRIPT / "_build.pdf"),
            ]
        )
    finally:
        build_input.unlink(missing_ok=True)


def load_metric_tokens(path: Path = DEFAULT_METRICS) -> dict[str, str]:
    """Load render tokens from the repository verification metrics file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"demo_amplitude", "test_count", "coverage_pct"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"verification metrics missing keys: {', '.join(missing)}")
    return {
        "DEMO_AMPLITUDE": str(payload["demo_amplitude"]),
        "TEST_COUNT": str(payload["test_count"]),
        "COVERAGE_PCT": str(payload["coverage_pct"]),
    }


def load_empirical_tokens(path: Path = DEFAULT_EMPIRICAL_METRICS) -> dict[str, str]:
    """Load empirical-pilot render tokens, using explicit unavailable defaults."""
    unavailable = {
        "EMPIRICAL_PILOT_STATUS": "pending local recording",
        "EMPIRICAL_PILOT_FINITE_GAZE": "unavailable",
        "EMPIRICAL_PILOT_SAMPLING_HZ": "unavailable",
        "EMPIRICAL_PILOT_SAMPLING_CV": "unavailable",
        "EMPIRICAL_PILOT_DRIFT": "unavailable",
        "EMPIRICAL_PILOT_HELDOUT_RMS": "unavailable",
        "EMPIRICAL_PILOT_LATENCY": "unavailable",
    }
    if not path.exists():
        return unavailable
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("manuscript_tokens", {})
    if not isinstance(tokens, dict):
        return unavailable
    return {key: str(tokens.get(key, value)) for key, value in unavailable.items()}


def load_empirical_session_tokens(path: Path = DEFAULT_EMPIRICAL_SESSIONS) -> dict[str, str]:
    """Load repeated-session readiness tokens from the generated summary."""
    unavailable = {
        "EMPIRICAL_SESSIONS_READINESS": "no repeated-session readiness available",
        "EMPIRICAL_SESSIONS_STATUS": "no repeated-session summary available",
        "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL": "unavailable",
        "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_COUNT_ONLY": "unavailable",
        "EMPIRICAL_SESSIONS_CONDITIONS_REMAINING": "unavailable",
        "EMPIRICAL_SESSIONS_REFERENCE_REQUIREMENT": "unavailable",
        "EMPIRICAL_SESSIONS_FUTURE_SCOPE": "unavailable",
    }
    if not path.exists():
        return unavailable
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("manuscript_tokens", {})
    if not isinstance(tokens, dict):
        return unavailable
    return {key: str(tokens.get(key, value)) for key, value in unavailable.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the iTrace manuscript with explicit gate values."
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS,
        help="Verification metrics JSON used for render tokens.",
    )
    parser.add_argument("--demo-amplitude", help="Override {{DEMO_AMPLITUDE}} from metrics JSON.")
    parser.add_argument("--test-count", help="Override {{TEST_COUNT}} from metrics JSON.")
    parser.add_argument("--coverage-pct", help="Override {{COVERAGE_PCT}} from metrics JSON.")
    parser.add_argument(
        "--empirical-json",
        type=Path,
        default=DEFAULT_EMPIRICAL_METRICS,
        help="Empirical pilot metrics JSON used for render tokens.",
    )
    parser.add_argument(
        "--empirical-sessions-json",
        type=Path,
        default=DEFAULT_EMPIRICAL_SESSIONS,
        help="Repeated empirical-session summary JSON used for render tokens.",
    )
    args = parser.parse_args()

    values = load_metric_tokens(args.metrics_json)
    values.update(load_empirical_tokens(args.empirical_json))
    values.update(load_empirical_session_tokens(args.empirical_sessions_json))
    if args.demo_amplitude is not None:
        values["DEMO_AMPLITUDE"] = args.demo_amplitude
    if args.test_count is not None:
        values["TEST_COUNT"] = args.test_count
    if args.coverage_pct is not None:
        values["COVERAGE_PCT"] = args.coverage_pct
    render(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
