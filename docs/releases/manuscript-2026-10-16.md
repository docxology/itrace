# iTrace corrected manuscript - 2026-10-16

The corrected manuscript is published on
[Zenodo](https://zenodo.org/records/23419306) at DOI
[10.5281/zenodo.23419306](https://doi.org/10.5281/zenodo.23419306).
[Download the 30-page PDF](https://zenodo.org/records/23419306/files/itrace-manuscript-2026-10-16.pdf?download=1).

| Identity | Value |
| --- | --- |
| Manuscript revision | `manuscript-2026-10-16-310be17` |
| Exact reviewed source | [`310be17ace8ae61621afe01329356427a3e6c861`](https://github.com/docxology/itrace/tree/310be17ace8ae61621afe01329356427a3e6c861) |
| Concept DOI | [10.5281/zenodo.20614908](https://doi.org/10.5281/zenodo.20614908) |
| Software baseline | [v0.5.0](v0.5.0.md), DOI [10.5281/zenodo.23403300](https://doi.org/10.5281/zenodo.23403300) |
| PDF length | 5,732,341 bytes |
| PDF SHA-256 | `419ca0113ee3816864c5ac8b6441886f471215962359a4601b804bc650c9688e` |

This is a manuscript revision after the v0.5.0 software release. The package
version remains 0.5.0; its tag, release assets, and previous Zenodo archive are
preserved. References to the v0.5.0 DOI inside the PDF identify that baseline.
The source link above identifies the subsequent corrections archived in this
manuscript. Later documentation commits do not replace its frozen source or PDF.

## Corrections

- Shared one-to-one, strictly positive overlap scoring prevents merged,
  fragmented, or merely touching events from receiving perfect recovery scores.
- Noise tables retain actual percentile endpoints and separate Monte Carlo
  standard errors. Thresholds and tables hydrate from the regenerated ledger.
- Moving-block resampling, explicit seeds and lengths, and an AR(1) sensitivity
  illustration expose dependence assumptions. Fit uncertainty reports failed
  draws and never turns total fitting failure into a zero-width interval.
- Information criteria count free parameters. Fitted-null KS diagnostics refit
  unknown parameters and disclose simulation resolution and selection limits.
- Figures show uncertainty endpoints, valid resample counts, and transition
  denominators. All 41 figure artifacts and four manuscript formats were
  regenerated; the PDF was rendered and inspected.
- Primary-source scholarship narrows physiological interpretations, real-error
  bounds, competitor comparisons, entropy claims, and bootstrap guarantees. The
  final revision distinguishes a parameter-recovery fixture from a descriptive
  recording fit and names the closed-loop residual as conditional.

See the [scholarship register](../SCHOLARSHIP_REVIEW_20261015.md) and
[statistical design guide](../STATISTICAL_DESIGN.md) for methods and limitations.

## Verification and publication

The source gate recorded 737 passed, 2 skipped, and 91.05% combined
statement-and-branch coverage. The core-only environment recorded 612 passed and
70 skipped with optional packages absent. All 44 final artifact checks passed.
Independent review approved the exact source commit after correcting two
manuscript contradictions.

[GitHub Actions run 38834808643](https://github.com/docxology/itrace/actions/runs/38834808643)
passed all ten jobs: Python 3.10-3.13, development gates, and the figures, web,
dashboard, capture, and all-extra package configurations.

The new draft's complete inventory was checked before publication. All eight
published payloads were downloaded without authentication and matched local
length, MD5 and SHA-256 values. The old v0.5.0 record retained its complete
filename/size/checksum inventory. The
[machine-readable receipt](manuscript-2026-10-16-verification.json) records the
published filenames, checksums, source identity, and successful hosted run.

## Remaining scientific limits

This publication adds no participant or reference-device acquisition. It does
not establish device accuracy, population or device generality, or nominal
bootstrap interval coverage. The existing empirical evidence remains a
single-participant, single-device, five-session diagnostic pilot. External
agreement studies and broader sampling remain in the [roadmap](../../TODO.md)
and [validation protocols](../VALIDATION_PROTOCOLS.md).
