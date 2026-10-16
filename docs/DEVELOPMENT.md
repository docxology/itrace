# Development and verification

iTrace separates synthetic algorithm verification, recording diagnostics, and
reference-device validation. Passing software gates does not turn the existing
single-participant pilot into a device-accuracy study.

## Runtime and gates

Use Python 3.10 or later and the committed `uv.lock`. On Daniel's workstation,
source, environments, caches, and temporary files belong on the mounted GitVault
volume. Select a GitVault interpreter and a separate environment before syncing;
check both `sys.prefix` and `sys.base_prefix`, since a virtual environment on the
SSD can still point to an internal interpreter. Preserve environments belonging
to other active sessions.

```bash
export UV_PROJECT_ENVIRONMENT="${GITVAULT_ROOT}/Local/envs/itrace-dev"
export UV_CACHE_DIR="${GITVAULT_ROOT}/Local/cache/itrace-uv"
export TMPDIR="${GITVAULT_ROOT}/Local/tmp/itrace"
export MPLCONFIGDIR="${GITVAULT_ROOT}/Local/cache/itrace-matplotlib"
export UV_LINK_MODE=copy
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR"
uv sync --locked --extra dev --python "$GITVAULT_PYTHON"
uv run --frozen --no-sync python -c \
  'import sys; print(sys.prefix); print(sys.base_prefix)'
uv run --frozen --no-sync python scripts/verify.py \
  --out-dir "output/verification/$(date -u +%Y%m%dT%H%M%SZ)"
```

Set `GITVAULT_ROOT` to the verified mounted volume and `GITVAULT_PYTHON` to the
interpreter selected there. Other machines may use their local environment and
cache directories.

The verification command runs lock validation, lint, formatting, strict mypy,
the manuscript source audit, release metadata and figure integrity, and the full test suite with branch measurement
enabled and the existing 90% total coverage floor. Each run requires a new
directory. Its `summary.json` records commands, statuses, durations, runtime
prefixes, and a digest of Python, browser assets, fixture data, package metadata,
and lock inputs. Logs, JUnit XML, and coverage JSON remain available after a
failure. A source change during execution prevents a passing receipt.

The source digest covers executable/test inputs, not every document or generated
artifact. Inspect rendered products separately. The runner stops on failure and
does not update manuscript metrics or publish a release.

## Core-only and installed-package checks

CI runs the suite with core dependencies plus pytest infrastructure on Python
3.10-3.13, separately from the development-extra job. Tests needing matplotlib or
FastAPI skip when those extras are actually absent; numerical, I/O, configuration,
and manuscript-source tests still run. The missing-web-extra control uses a real
missing dependency rather than replacing Python's import function.

Build and test the installed wheel in a fresh environment. Keep that environment
separate from the development environment; run its Python with `-I` so the source
checkout cannot satisfy imports. Check `py.typed`, dashboard assets, a synthetic
analysis, and the installed `itrace demo` command. CI includes those checks. The
optional-package jobs also install the sdist with each advertised extra and
exercise the real imports. A declared workflow is distinct from a successful
hosted run; publication receipts record the exact commit and hosted result.

The [2026-10-16 manuscript publication receipt](releases/manuscript-2026-10-16-verification.json)
binds the corrected PDF to source commit `310be17`, its successful ten-job hosted
run, and anonymously verified Zenodo files. Its
[revision notes](releases/manuscript-2026-10-16.md) separate the manuscript
archive from the v0.5.0 software release.

## Numerical contracts

The post-v0.5.0 statistical correction pass adds one-to-one overlap scoring to the
closed-loop sweep, actual percentile endpoints, explicit Monte Carlo standard
errors, moving-block resampling, failed-fit counts, and fixed-family KS refitting.
The [statistical design guide](STATISTICAL_DESIGN.md) specifies assumptions,
compatibility changes, and interpretation. Primary-source checks and their access
limits are retained in [the scholarship review](SCHOLARSHIP_REVIEW_20261015.md).
The archived v0.5.0 ledger is preserved byte-for-byte at
`releases/v0.5.0-verification.json`; current development does not change the
published Git tag or Zenodo payload.

- Synthetic-domain validation chooses gaze-gap preprocessing from finite input
  masks before analysis. Bounded gaps of at most 0.08 s are interpolated; remaining
  invalid edge samples are dropped and reported. Long bounded gaps remain errors.
  This explicit benchmark policy does not change the raw recording pipeline's
  missing-data policy or use library exceptions to select a method.
- Streams own read-only float64 snapshots. Raw NaN measurements and defective
  clocks remain representable for diagnostics; differentiation requires finite,
  strictly increasing timestamps.
- Irregular gaze timing selects a gradient on the actual clock. Uniform timing
  selects Savitzky-Golay differentiation. Missing measurements are not fixations,
  and saccade gap merging cannot bridge unknown samples.
- A pupil CSV with a unit column must use one valid unit on every row. With no
  unit column, the documented default remains `relative`.
- Pupil interpolation uses elapsed time and rejects a trace with no remaining
  anchors after blink padding. Short-input smoothing preserves length and constant
  values; `smooth_window` controls its edge-padded fallback. The SOS padding rule
  follows [SciPy's documented default](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
- A zero-MAD pupil trace flags nonmedian values. No variance floor hides spikes.
- Configuration rejects nonfinite values, booleans masquerading as numbers,
  fractional integer settings, and malformed windows. Report validation checks
  nonnegative integer counts, finite duration, and event-list totals.

## Live analysis and exports

The dashboard plots the effective threshold computed by the Python analysis,
including zero. Export sends the last displayed detector method, requested
threshold, and PSO selection. The report records those settings, including partial
and failed analyses. Export analyses the retained session; the display analyses
its rolling window, so their event counts can legitimately differ when the
retained session exceeds that window. Calibration continues to provide a separate
coordinate series; the primary detector operates on raw gaze.

Blink interpolation and zero-phase filtering are offline transforms. The standalone
`PhaseDetector` is causal; placing it after those transforms does not make the full
pipeline causal. A browser smoke test with synthetic frames verifies orchestration
and display behavior, not webcam performance.

## Manuscript and evidence

```bash
uv run --frozen --no-sync python scripts/audit_crossrefs.py
uv run --frozen --no-sync python scripts/render_manuscript.py \
  --metrics-json docs/verification_metrics.json
```

Rendering and auditing use the same section discovery. Administrative Markdown
never enters the paper. Rendering fails on warnings or unresolved references and
stages Markdown, text, LaTeX, and PDF before replacing reader outputs. Failed
staging directories retain the input and intermediate files. Missing empirical
ledgers yield explicit unavailable tokens; malformed existing ledgers fail.

`docs/verification_metrics.json` describes the current development verification;
`docs/releases/v0.4.1-verification.json` preserves the prior release ledger intact,
including its historical summary discrepancies. Dated audit notes retain their
original numbers. Updating the current ledger requires actual JUnit/coverage/gate
receipts, followed by manuscript hydration and visual PDF inspection.

Empirical summaries describe their recorded collection epoch. If local pilot
exports are absent, do not regenerate those summaries as though new raw evidence
were available. Their figures can be rendered from retained summary JSON, with
that source stated explicitly; this is not a reanalysis of missing session
records. Regenerate synthetic figures and verification badges separately.

For a full refresh using the recorded empirical summaries, run
`uv run python scripts/generate_figures.py --reuse-empirical-summaries`.
The default command requires the source exports and preflights them before any
figure/ledger writes. `figure_manifest.json` records the empirical source mode.

[RELEASING.md](RELEASING.md) describes the version/DOI checks, distribution
smoke tests, commit-bound asset inventory, and public download verification.
