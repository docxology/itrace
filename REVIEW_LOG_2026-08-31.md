# Agent-ergonomics review log — 2026-08-31 (fleet lane: itrace)

Round 1 (orientation audit):
- Cold-start audit: entry docs (README/AGENTS) pass all three orientation tasks
  (status → docs/verification_metrics.json with gate date; next → TODO.md;
  verification → uv run pytest --cov=itrace). Score: 3/3.
- Link check over 21 markdown files: 0 broken relative links (script-checked).
- Findings fixed: TODO.md had a stale unchecked duplicate of the already-shipped
  "visible target presentation" item (removed); unscoped Major roadmap entries
  given acceptance lines; live calibration target presentation lacked
  click/Space confirmation (implemented, see Round 2); pytest collect-only takes
  ~3m20s on the external drive — the "fast gate" is not <2 min, documented here
  rather than invented as passing.
- Pre-existing dirty tree (76 paths: docs-manuscript relocation M/D/?? files)
  left untouched; only path-scoped adds of files edited in this pass.
