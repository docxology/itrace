# Offline published-reader render

This workflow renders the canonical iTrace manuscript using the
[MIT-licensed reader edition](https://doi.org/10.5281/zenodo.23419306). It reuses
the twelve values called out by the manuscript and the four figure files
listed in the release manifest. The retained Markdown, token mapping, and
figure bytes are each checksum-pinned. The archived edition stays unchanged.

The published figures and values describe one participant and one device
across five sessions. They provide a historical diagnostic. They establish
neither independent anonymization nor device accuracy, generalizability, or
new participant evidence. This build uses the published reader's visible
findings. Evidence collection and replay are outside the render.

## Materialize the reader inputs

Use a disposable blobless sparse checkout. Include the root configuration,
`docs/`, `scripts/`, and `src/`. Exclude `data/`, capture files, and the
participant manifest and reports
`docs/empirical_pilot_metrics.json`, `docs/empirical_sessions_manifest.json`,
and `docs/empirical_sessions_summary.json`. From the
[retained build directory](releases/retained-render/), only `_build.md` is
needed. Include the [synthetic-review inputs](SYNTHETIC_REVIEW.md) plus
`resampling_sensitivity.json` and `statistical_diagnostics.json` under
`output/figures/` and these four PNGs from the same directory.

- `empirical_pilot_summary.png`
- `empirical_sessions_summary.png`
- `synthetic_empirical_range_bridge.png`
- `statistical_interpretation_ledger.png`

## Render and validate

Use [Template at revision `87704feae0a4490ef7dad9c931d268e8c30273c2`](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2)
with Pandoc, a matching pandoc-crossref, XeLaTeX, and BibTeX. Set
`itrace_checkout` to the absolute path of a disposable checkout whose physical
directory is named `itrace`. Run from Template with an unused
`projects/working/itrace` slot.

```sh
uv sync --locked --no-dev
PYTHONPATH="$itrace_checkout/src" uv run --no-sync python \
  "$itrace_checkout/scripts/z_generate_manuscript_variables.py" --published-reader
mkdir -p projects/working
ln -s "$itrace_checkout" projects/working/itrace
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/itrace --skip-manuscript-hydration
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/itrace
```

The explicit hydration command checks the release artifacts and recovers only
the twelve permitted manuscript values. Two additional fields required by the
native report helper have no canonical callout and are omitted. The software
verification metrics and synthetic diagnostics continue to use the existing
inputs from the synthetic review. This workflow requires no participant report
JSON, recording, webcam, or empirical replay.

Open `output/pdf/itrace_combined.pdf` and `output/web/index.html` in the
project checkout. The reader provenance receipt is
`output/data/published_reader_receipt.json`, alongside the normal Template
validation and provenance reports under `output/reports/`. See the
[license](../LICENSE) and [release record](releases/manuscript-2026-10-16.md)
for attribution and the limits of the historical evidence.
