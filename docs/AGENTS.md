# AGENTS.md — iTrace docs/

## Layout

This directory holds research ledgers, protocols, and audits (Markdown + JSON),
not software documentation. `docs/manuscript/` at the repo root holds the manuscript
sources. `docs/README.md` is the index.

## Key artifacts

- Protocols: `VALIDATION_PROTOCOLS.md`, `manual_annotation_protocol.md`, `PREPUBLICATION_REVIEW.md`.
- Ledgers (JSON, machine-readable): `TRACEABILITY_MATRIX.json`, `figure_manifest.json`,
  `verification_metrics.json`, `empirical_*` files, `releases/`.

## Conventions observed

- Numeric claims in the manuscript are injected at render time via `{{TOKEN}}`
  placeholders (documented in `docs/manuscript/README.md`); never hand-edit resolved
  values into section sources.
- `_build.*` files under `docs/manuscript/` are render outputs, not sources.

## Maintenance

When a ledger JSON changes, update the corresponding protocol/summary Markdown in
the same change. Keep this index consistent with actual filenames.
