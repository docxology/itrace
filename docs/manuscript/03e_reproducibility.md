# Results: cross-implementation agreement and gates {#sec:reproducibility}

On a shared synthetic trace the core I-VT and microsaccade detectors agree with
an independent reference implementation (`reference_impl`, raw finite-difference
velocity and an explicit-loop microsaccade detector) on event count and onset
timing, despite using different velocity formulations — a guard against a bug
shared by a single implementation. The two paths share only their input and the
published algorithm definitions, so agreement is evidence the implementation
matches the algorithm, not that two copies of one mistake agree.

The complete toolchain is green and reproducible: {{TEST_COUNT}} no-mocks tests
pass with {{COVERAGE_PCT}} statement-and-branch coverage, `ruff` and
`ruff format` report no issues, and `mypy --strict` finds no type errors across
the source tree. The test count, coverage, version, gate date, and render
evidence are stored once in `docs/verification_metrics.json`; the manuscript
renderer hydrates these tokens from that JSON so stale literal metrics are caught
by tests rather than silently carried into public artifacts. Every synthetic
generator and every Monte-Carlo trial is explicitly seeded with
`numpy.random.default_rng`, so each figure and table number is reproducible
byte-for-byte from a clean checkout — re-running the figure pipeline twice
produces identical PNGs, and the noise sweep of [@sec:noiseresults] reproduces
its table cells exactly. No test mocks a computation: detectors run on real
arrays with known ground truth, file I/O uses real temporary files, and figures
are asserted to write non-empty PNGs. The full module map, dependency contract,
and test strategy are in [@sec:software].

The documentation and visualization layer now follows the same evidence rule.
`docs/TRACEABILITY_MATRIX.json` maps each public validation claim and manuscript
figure to the documentation surface where it appears, the tested Python method
or script that computes it, the backing test file, the generated evidence
artifact, and the truth boundary. That matrix is deliberately machine-readable:
tests verify its paths and pure-core symbols, so a future figure, README badge,
or live-analysis claim cannot become orphaned prose detached from the tested
methods underneath it.

The validation-domain suite extends that reproducibility contract from a single
trace to stress domains. `itrace synthetic-validation` writes JSON for repeated
clean, webcam-jitter, head-drift, and low-light/dropout sessions, summarising
within-domain recovery and across-domain stability. The live HTML page calls the
same Python route when the user runs synthetic validation from the browser; the
browser only renders the returned statistics and does not reimplement event
detection or recovery scoring.

Finally, `itrace benchmark` provides a release-ready shape for external
validation without pretending such data are bundled. It loads caller-supplied
truth and comparator event CSV files, can score a gaze CSV through the iTrace
pipeline, and writes one JSON report with interval-overlap recovery plus timing
and amplitude errors for each system. That design matters because it separates
the *mechanism* for validation from the *evidence* required to make a validation
claim: the package can compute the same metrics against a reference tracker,
public dataset, or comparator detector, but the report must carry the supplied
truth source and boundary. A high score against another detector is detector
agreement; a high score against an independent reference device would be a
different and stronger claim.

The live experiment report follows the same reproducibility pattern for real
webcam sessions. `itrace live-html` auto-saves a derived experiment bundle after
the final required guided trial, and `itrace experiment-report` rebuilds the
empirical summary from the exported manifest plus capture-record CSV. That
command path is intentionally file-based:
the reported jitter, drift, finite fraction, calibration residual, and target
latency can be regenerated without a browser or camera, while the payload still
states that prompted screen targets are not a reference-device validation study.
The manuscript-facing single-pilot summary in [@fig:empirical-pilot] is hydrated
from that report when it exists; when it does not, the same render path exposes
pending or unavailable fields rather than replacing missing empirical data with
guessed values.
