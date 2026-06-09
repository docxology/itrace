# Final pre-publication RedTeam/Science review

**Date:** 2026-06-09  
**Release state reviewed:** v0.4.1 diagnostic v1  
**Execution mode:** internal RedTeam VectorSpecialists panel plus Science FullCycle synthesis over repo-local artifacts and generated outputs.  
**Target classification:** structured artifact with existing oracles. Q1=yes, public claims reside in docs, manuscript, ledgers, figures, and rendered builds. Q2=yes, defects can be cited by path, section, field, or rendered output. Q3=yes, the repo has test, manifest, traceability, figure, and rendered-artifact oracles.

## Source ledgers

- `docs/verification_metrics.json`: v0.4.1, 2026-06-09 gate date, 528 tests, 90.23% coverage, MIT license, `https://github.com/docxology/itrace`, `ruff`, `ruff format`, and `mypy` summaries.
- `docs/empirical_sessions_summary.json`: 5 available sessions, 5 replicate IDs, 2 conditions, 0 reference-backed rows, diagnostic-v1 ready, and future 12-session/3-condition/reference-backed validation scope.
- `docs/empirical_sessions_manifest.json`: derived-record-only storage boundary and no current `reference_kind` beyond `none`.
- `docs/TRACEABILITY_MATRIX.json`: claim and figure boundaries for synthetic verification, display-only UI, empirical diagnostics, and device-validation non-claims.
- `docs/figure_manifest.json`: 39 generated figure/data artifacts with source paths, hashes, dimensions, and nonblank image checks.
- `manuscript/_build.md`, `manuscript/_build.txt`, `manuscript/_build.tex`, and `manuscript/_build.pdf`: rendered reader-facing outputs from the current ledgers.

## Verifier status

The verifier suite is strong but not omniscient. It currently checks stale metrics, rendered unresolved tokens, Abstract citation absence, license/citation/repository agreement, empirical readiness and reference-evidence counting, figure provenance and nonblank outputs, traceability paths, and the rendered PDF smoke contract. The final-audit record is now part of that verifier surface so this review cannot claim stale metrics or a different empirical scope without failing tests.

## Vector findings

| Vector | Verdict | Evidence checked | Result |
|---|---|---|---|
| Availability, license, citation | Clean after prior fixes | README, Abstract, `pyproject.toml`, `LICENSE`, `CITATION.cff`, verification metrics, rendered builds | MIT/GitHub wording is aligned; at publication the repository was made public and this version archived at Zenodo DOI 10.5281/zenodo.20614909. |
| Empirical diagnostic scope | Clean | session manifest, session summary, empirical manuscript section, tests | Five sessions are framed as single-participant/single-device diagnostic v1; 0 reference-backed rows remain explicit. |
| Validation and accuracy boundaries | Clean | README, manuscript methods/results/discussion/limitations, traceability matrix, rendered builds | Synthetic and closed-loop evidence are described as algorithmic verification, not device validation or webcam accuracy. |
| Citation and scholarship support | Minor fix applied | research brief, scholarship audit, bibliography tests | One research-brief link label looked like a GitHub URL while targeting an ecosyste.ms mirror; label was narrowed to the actual mirror. No new external evidence claim was added. |
| Figure and provenance claims | Clean | figure manifest, graphical abstract, statistical ledgers, generated JSON sidecars | Graphical abstract and figure captions now communicate architecture/evidence boundaries, not webcam validation. |
| Privacy, storage, and UI workflow | Clean | README, capture-shell manuscript section, empirical manifest, live/export tests | Public docs state derived records only by default; raw eye video and persisted eye crops remain outside the default workflow. |

## Residual future-scope claims

- Device-level gaze accuracy still requires public-dataset, reference-device, or validated manual-annotation evidence.
- Population generality, cross-device generality, lighting/head-pose generality, and reference-device agreement remain future scope.
- The planned validation expansion remains 12 sessions, 12 replicate IDs, 3 conditions including `indoor_office_backlit`, and at least 1 validated reference artifact.
- The GitHub repository URL is fixed for citation and release metadata; at publication the repository was made public and this version archived on Zenodo (DOI 10.5281/zenodo.20614909).

## Visualization and claim sweep

Final visualization polish targeted reader-facing composites only: the range bridge, statistical interpretation ledger, statistical diagnostics composite, and empirical-session intake summary. The changes improve spacing, wrapping, and boundary labels without changing evidence values, readiness criteria, or source JSON. Regression tests now check rendered table text for severe row overlap and reject stale test counts, stale coverage, and positive device/webcam accuracy claims across public claim surfaces.

RedTeam claim review after the polish treats `docs/verification_metrics.json`, `docs/empirical_sessions_summary.json`, `docs/empirical_sessions_manifest.json`, `docs/TRACEABILITY_MATRIX.json`, and `docs/figure_manifest.json` as the controlling ledgers. The canonical claim remains: v0.4.1 is MIT-licensed, openly released at `https://github.com/docxology/itrace` and archived at Zenodo DOI 10.5281/zenodo.20614909, with 528 tests, 90.23% coverage, five diagnostic sessions, two conditions, and 0 reference-backed rows. Device validation, population generality, cross-device claims, and reference-device agreement remain explicitly future scope.

## Final gate commands

Run these after this review record and any wording fixes:

```bash
uv run python scripts/generate_figures.py
uv run python scripts/render_manuscript.py --metrics-json docs/verification_metrics.json --empirical-json docs/empirical_pilot_metrics.json --empirical-sessions-json docs/empirical_sessions_summary.json
uv run pytest tests/test_figures.py tests/test_manuscript_integrity.py tests/test_documentation_traceability.py tests/test_empirical_sessions.py tests/test_experiments.py --tb=short
uv run pytest --cov=itrace
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/itrace
```

## Release verdict

Pre-publication status is acceptable for a diagnostic v1 manuscript if and only if the final gate commands pass after regeneration. The review does not approve public claims of webcam/device accuracy, population validity, cross-device validity, or reference-device agreement.
