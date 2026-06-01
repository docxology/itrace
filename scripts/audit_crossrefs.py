"""Cross-reference and citation integrity auditor for the iTrace manuscript.

Oracle for the manuscript review pass: enumerates every label, cross-reference,
and citation across the renderable modules and reports
  - cross-references with no matching label (would render as "??"),
  - labels that are never referenced (dead anchors),
  - citation keys used but absent from references.bib (build-fatal under
    fail_on_missing), and citation keys defined but never cited.

Run:  uv run python scripts/audit_crossrefs.py
Exit non-zero if any build-fatal problem (dangling ref or missing citation).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
RENDER_SKIP = {"README.md", "preamble.md"}

LABEL_RE = re.compile(r"\{#((?:sec|fig|tbl|eq):[A-Za-z0-9_:+-]+)")
XREF_RE = re.compile(r"(?<![A-Za-z0-9_])@((?:sec|fig|tbl|eq):[A-Za-z0-9_:+-]+)")
# citation keys: @key not followed by a crossref prefix, allowing [@a; @b]
CITE_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_+-]*[0-9][A-Za-z0-9_+-]*)")
BIBKEY_RE = re.compile(r"^@[a-zA-Z]+\{([^,]+),", re.MULTILINE)


def source_files() -> list[Path]:
    return sorted(
        p
        for p in MANUSCRIPT.glob("*.md")
        if not p.name.startswith("_") and p.name not in RENDER_SKIP
    )


def main() -> int:
    labels: dict[str, str] = {}
    xrefs: dict[str, list[str]] = {}
    cites: dict[str, list[str]] = {}

    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for m in LABEL_RE.finditer(text):
            labels.setdefault(m.group(1), path.name)
        for m in XREF_RE.finditer(text):
            xrefs.setdefault(m.group(1), []).append(path.name)
        for m in CITE_RE.finditer(text):
            key = m.group(1)
            if key.split(":", 1)[0] in {"sec", "fig", "tbl", "eq"}:
                continue
            cites.setdefault(key, []).append(path.name)

    bib_text = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    bibkeys = set(BIBKEY_RE.findall(bib_text))

    label_names = set(labels)
    xref_names = set(xrefs)
    cite_names = set(cites)

    dangling = sorted(xref_names - label_names)
    dead_labels = sorted(label_names - xref_names)
    missing_cites = sorted(cite_names - bibkeys)
    uncited = sorted(bibkeys - cite_names)

    print(f"labels={len(label_names)} xrefs={len(xref_names)} "
          f"cites={len(cite_names)} bibkeys={len(bibkeys)}")
    print("\n[DANGLING cross-refs] (referenced, no label — renders as ??):")
    print("  " + (", ".join(dangling) if dangling else "none ✓"))
    print("\n[DEAD labels] (labelled, never referenced):")
    print("  " + (", ".join(dead_labels) if dead_labels else "none ✓"))
    print("\n[MISSING citations] (cited, absent from .bib — build-fatal):")
    print("  " + (", ".join(missing_cites) if missing_cites else "none ✓"))
    print("\n[UNCITED bib entries] (in .bib, never cited):")
    print("  " + (", ".join(uncited) if uncited else "none ✓"))

    fatal = bool(dangling or missing_cites)
    print("\nRESULT:", "FAIL" if fatal else "PASS")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
