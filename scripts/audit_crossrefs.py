"""Cross-reference and citation integrity auditor for the iTrace manuscript.

Oracle for the manuscript review pass over ``docs/manuscript/``: reports
dangling cross-refs, dead figure anchors, duplicate labels, citations absent
from references.bib (build-fatal), uncited bib entries, and manual
Figure/Table numbering that bypasses pandoc-crossref.

Thin orchestrator: the audit logic lives in :mod:`itrace.manuscript`.

Run:  uv run python scripts/audit_crossrefs.py
Exit non-zero if any build-fatal problem is found.
"""

from __future__ import annotations

from pathlib import Path

from itrace.manuscript import audit_crossrefs

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript"


def main() -> int:
    return audit_crossrefs(MANUSCRIPT)


if __name__ == "__main__":
    raise SystemExit(main())
