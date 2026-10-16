# iTrace development review - 2026-10-15

This report records the initial development pass, before the v0.5.0 release
follow-up. Its source digest and command receipts identify that earlier cut;
current release evidence is in [verification_metrics.json](verification_metrics.json)
and [the v0.5.0 release notes](releases/v0.5.0.md).

The later statistical and scholarly correction pass and its published manuscript
are documented separately in the
[2026-10-16 revision record](releases/manuscript-2026-10-16.md). The measurements
below describe this initial pass, not the later correction or publication.

That pass corrected reproduced defects in the analysis core, input/report
contracts, live orchestration, publication generation, and verification. It is
an unreleased working-tree change against
`88e9a08a9ea47ac3cd261382601f4878482140e9`.

## Corrections

| Surface | Reproduced problem | Result |
|---|---|---|
| Streams | Caller arrays could mutate supposedly immutable recordings | Owned read-only float64 snapshots |
| Timing/events | Irregular samples used a uniform derivative; NaN runs became fixations | Actual-clock differentiation; invalid clocks fail; unknown samples split events |
| Pupil preprocessing | Index interpolation distorted irregular timing; short smoothing changed length or failed padding; zero-MAD spikes were missed | Elapsed-time interpolation, explicit anchor rejection, SOS-aware/edge-padded smoothing, zero-MAD outlier handling |
| Units/configuration | Later CSV units were ignored; nonfinite and wrong-type settings passed | Every unit row checked; scalar and window validation at configuration boundaries |
| Reports | Boolean/negative counts, nonfinite duration, and mismatched totals passed validation | Builders invoke the strengthened report validator |
| Live analysis | Exports used defaults; plots could show the requested fixed threshold instead of the adaptive result | Exported configuration follows displayed settings; effective threshold, including zero, drives the plot |
| Manuscript | Administrative files entered the paper; unresolved references were replaced by invented values; missing empirical ledgers had numerical defaults | Shared section discovery, explicit unavailable evidence, staged four-format builds that reject unresolved output |
| Figures | Full orchestrator omitted a required argument; a pilot image disagreed with its populated ledger | Correct repository-root binding; full regeneration; explicit retained-summary mode and source-mode provenance |
| Verification | Core-only suite failed collection; release metrics disagreed internally; checks lacked consolidated receipts | Actual absence controls, optional test guards, archived historical ledger, fail-fast verification command, CI and installed-wheel checks |

New regression inputs first produced real failures. Subsequent numerical tests
use actual arrays/files, synthetic ground truth, and the independent existing
oracle. No detector or forward-model substitution was introduced, and no test
threshold was lowered.

## Evidence and reproducibility

Current numerical verification is in [verification_metrics.json](verification_metrics.json).
[DEVELOPMENT.md](DEVELOPMENT.md) documents the runtime, numerical contracts,
commands, and limits. The complete command receipts are retained under
`${GITVAULT_ROOT}/Local/evidence/itrace-comprehensive-20261015/`, including
negative controls, core-only results, wheel smoke, browser screenshots/exports,
gallery generation, actionlint, and real renderer failure controls.

The executable/test input digest is
`6fb2b00826e74ac3077b552a86df9c3a8dc84286d6252bedd999840df06e899d`.
This digest excludes documentation and generated reader artifacts; their
integrity is checked separately. The final manuscript contains 28 pages and is
visually inspected through the complete page contact sheet and selected full
pages, including the empirical cards and table. The publication manifest indexes
39 artifacts and records `empirical_source=retained_summary_ledgers`.

The four manuscript formats, synthetic validation output, gallery figures,
animations, and provenance manifest were regenerated. Existing empirical JSON
ledgers were preserved; empirical images were rendered from those recorded
summaries. The prior release verification ledger remains unchanged at
[releases/v0.4.1-verification.json](releases/v0.4.1-verification.json).

## Limits

- Python 3.12 development/core-only and isolated-wheel checks ran locally.
  Python 3.10.20 is blocked on this macOS host by the locked SciPy 1.15.3 native
  library import, before iTrace loads. The exact failure is retained in
  `core310-tests.log`; it was not replaced by a passing result from different
  dependencies. The configured Linux CI matrix has not run remotely here.
- Browser checks use visibly labelled synthetic frames, including a zero-velocity
  control. Hardware performance, new empirical collection, reference-device
  agreement, and public-dataset validation are not established by this pass.
- The full pupil pipeline uses surrounding samples for interpolation/filtering;
  only the standalone phase classifier has the documented causal guarantee.
- The user's pre-existing manuscript format configuration and untracked
  `output/pdf`, `output/reports`, and `output/web` artifacts were preserved.
  The initial baseline command did overwrite the ignored repository `.coverage`
  file before coverage output was rebound to the evidence directory; the prior
  coverage file had not been snapshotted. Later failures and receipts are kept
  separately.

The [independent development gate review](releases/v0.5.0-development-review.md)
returned **APPROVE**, with no criterion-backed blockers. It independently
reproduced numerical boundary controls and verified all 39 figure hashes against
the final manifest. The final source gate receipt is `gates-final2/summary.json`
under the evidence root above. No release or remote publication is part of this
pass.
