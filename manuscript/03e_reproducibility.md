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
the source tree. Every synthetic generator and every Monte-Carlo trial is
explicitly seeded with `numpy.random.default_rng`, so each figure and table
number is reproducible byte-for-byte from a clean checkout — re-running the
figure pipeline twice produces identical PNGs, and the noise sweep of
[@sec:noiseresults] reproduces its table cells exactly. No test mocks a
computation: detectors run on real arrays with known ground truth, file I/O uses
real temporary files, and figures are asserted to write non-empty PNGs. The full
module map, dependency contract, and test strategy are in [@sec:software].

The validation-domain suite extends that reproducibility contract from a single
trace to stress domains. `itrace synthetic-validation` writes JSON for repeated
clean, webcam-jitter, head-drift, and low-light/dropout sessions, summarising
within-domain recovery and across-domain stability. The live HTML page calls the
same Python route when the user runs synthetic validation from the browser; the
browser only renders the returned statistics and does not reimplement event
detection or recovery scoring.
