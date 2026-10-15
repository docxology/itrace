# docs/manuscript/

iTrace methods-paper sources are numbered Markdown sections, supplementary
sections, `config.yaml`, and `references.bib`. Files beginning with `_` are
generated. `README.md`, `AGENTS.md`, `MANUSCRIPT_STATUS.md`, and `preamble.md`
are excluded from paper assembly.

```bash
uv run python scripts/audit_crossrefs.py
uv run python scripts/render_manuscript.py --metrics-json docs/verification_metrics.json
```

The renderer hydrates `{{TEST_COUNT}}`, `{{COVERAGE_PCT}}`, and
`{{DEMO_AMPLITUDE}}` from the current verification ledger and empirical tokens
from their respective ledgers. An absent empirical ledger produces explicit
unavailable fields. An existing malformed ledger, missing token, source audit
failure, Pandoc warning, or unresolved rendered reference stops the build.

Markdown, plain text, LaTeX, and PDF are staged and validated before reader outputs
are replaced. A failed `.itrace-render-*` directory retains the attempted input
and intermediate artifacts. Use `--output-dir` for an isolated review render.
Inspect the resulting PDF after changing sources; text checks alone do not prove
layout quality. The standalone renderer always emits these four formats; the
`render.formats` block in `config.yaml` belongs to the sibling template workflow.

See [Development and verification](../DEVELOPMENT.md) for runtime bindings,
gate receipts, release-history separation, and empirical-source limitations.
