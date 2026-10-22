# iTrace — Documentation

iTrace is an open-source Python toolkit for webcam eye-movement analysis — gaze
trajectory, saccade direction & dynamics, and pupil diameter — built as a pure
NumPy/SciPy analysis core with an optional thin webcam + MediaPipe capture shell.
See the [root README](../README.md) for the overview and validation boundaries.
For the new private webcam visual-task workflow, start with
[Private study](PRIVATE_STUDY.md). Stock `live-html` remains a separate diagnostic
workflow; original scientific publications and historical evidence remain unchanged.

## Directory map

| Path | Contents |
| ---- | -------- |
| `src/` | Analysis core and capture shell |
| `tests/` | Test suite |
| `scripts/` | Analysis, rendering, and manuscript pipeline scripts |
| `docs/manuscript/` | Manuscript markdown sources (SECTION files + config.yaml) |
| `output/` | Run outputs and reports (local working artifacts) |
| `docs/` | This documentation tree (research ledgers and protocols) |

## How to run

Commands are documented in the root `README.md` and `pyproject.toml` (uv-managed;
`uv.lock` present). Per-paper pattern: `uv sync` then `uv run pytest tests/ -q`.
Verify exact analysis entry points against `scripts/` and `tasks.yaml` before
running; this index does not invent commands.

## Document index (selected)

- [Research brief](RESEARCH_BRIEF.md)
- [Validation protocols](VALIDATION_PROTOCOLS.md)
- [Private visual study: manifest, measured setup, capture, trials, and recovery](PRIVATE_STUDY.md)
- [Manual annotation protocol](manual_annotation_protocol.md)
- [Scholarship audit](SCHOLARSHIP_AUDIT.md) / [Traceability matrix](TRACEABILITY_MATRIX.json)
- [Primary-source scholarship and statistics review, October 2026](SCHOLARSHIP_REVIEW_20261015.md)
- [Statistical design and interpretation guide](STATISTICAL_DESIGN.md)
- [Prepublication review](PREPUBLICATION_REVIEW.md)
- [Development, verification, numerical contracts, and manuscript rendering](DEVELOPMENT.md)
- [Comprehensive development review and remaining limits](COMPREHENSIVE_REVIEW.md)
- [Release procedure and artifact verification](RELEASING.md)
- [v0.5.0 release notes](releases/v0.5.0.md)
- [Corrected manuscript revision, DOI, and publication verification (2026-10-16)](releases/manuscript-2026-10-16.md)

`verification_metrics.json` records current development checks. The previous
release ledgers are retained in `releases/v0.4.1-verification.json` and
`releases/v0.5.0-verification.json`; dated audit
reports retain their original evidence rather than claiming a later gate run.

Empirical session ledgers: `empirical_sessions_manifest.json`,
`empirical_sessions_summary.json`, `empirical_pilot_metrics.json`,
`figure_manifest.json`, `verification_metrics.json` — machine-readable evidence
files; see the manuscript for their interpretation.
