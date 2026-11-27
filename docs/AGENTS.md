# AGENTS.md — iTrace docs/

## Layout

This directory holds the maintained guides, research ledgers, protocols,
and audits. `docs/manuscript/` at the repo root holds the manuscript
sources. `docs/README.md` is the index.

## Key artifacts

- Protocols: `VALIDATION_PROTOCOLS.md`, `manual_annotation_protocol.md`, `PREPUBLICATION_REVIEW.md`.
- Ledgers (JSON, machine-readable): `TRACEABILITY_MATRIX.json`, `figure_manifest.json`,
  `verification_metrics.json`, `empirical_*` files, `releases/`.

## Conventions observed

- Numeric claims in the manuscript are injected at render time via `{{TOKEN}}`
  placeholders (documented in `docs/manuscript/README.md`); never hand-edit resolved
  values into section sources.
- The prior `_build.*` artifacts are retained under `docs/releases/retained-render/`.

## Maintenance

When a ledger JSON changes, update the corresponding protocol/summary Markdown in
the same change. Keep this index consistent with actual filenames.
