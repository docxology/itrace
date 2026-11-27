# docs/manuscript/

iTrace methods-paper sources are numbered Markdown sections, supplementary
sections, `config.yaml`, and `references.bib`. The Markdown and TeX
preambles support Template and the standalone renderer respectively. The
standalone preamble is stored in `assets/` with the paper inputs.
The [retained standalone build](../releases/retained-render/README.md)
sits with the release documentation.

The corrected 30-page manuscript from source commit `310be17` is published at
[DOI 10.5281/zenodo.23419306](https://doi.org/10.5281/zenodo.23419306)
([PDF](https://zenodo.org/records/23419306/files/itrace-manuscript-2026-10-16.pdf?download=1)).
See the [revision notes and publication receipt](../releases/manuscript-2026-10-16.md)
for the source binding, checksums, scientific limits, and distinction from the
v0.5.0 software archive cited inside the PDF.

```bash
uv run python scripts/audit_crossrefs.py
uv run python scripts/render_manuscript.py --metrics-json docs/verification_metrics.json
```

The standalone publication renderer hydrates `{{TEST_COUNT}}`, `{{COVERAGE_PCT}}`, and
`{{DEMO_AMPLITUDE}}` from the current verification ledger and empirical tokens
from their respective ledgers. An absent empirical ledger produces explicit
unavailable fields. An existing malformed ledger, missing token, source audit
failure, Pandoc warning, or unresolved rendered reference stops the build.

Noise thresholds and the complete noise table hydrate from
`output/figures/noise_metrics.json` through `load_noise_tokens`. This deterministic
ledger is required; a missing or incomplete one fails instead of inserting guessed
thresholds. Regenerate it with the canonical figure pipeline after scoring changes.
The archived v0.5.0 verification ledger is retained under `docs/releases/`;
current development is identified by its own source digest and measured ledger.

Markdown, plain text, LaTeX, and PDF are staged and validated before reader outputs
are replaced. A failed `.itrace-render-*` directory retains the attempted input
and intermediate artifacts. Review builds default to `output/manuscript-preview/`. Use `--output-dir`
for a different isolated destination.
Inspect the resulting PDF after changing sources; text checks alone do not prove
layout quality. The standalone renderer always emits these four formats; the
`render.formats` block in `config.yaml` belongs to the sibling template workflow.

See [Development and verification](../DEVELOPMENT.md) for runtime bindings,
gate receipts, release-history separation, and empirical-source limitations.

The [offline published-reader render](../PUBLISHED_READER.md) builds the full
canonical paper with the public edition's checksum-pinned values and figures.
The [synthetic-only review](../SYNTHETIC_REVIEW.md) remains available for
verification that excludes the participant-derived figures and values.
