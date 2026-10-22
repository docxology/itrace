# AGENTS.md — iTrace

Standalone `uv` package: develop and test in this checkout. Read
[development and verification](docs/DEVELOPMENT.md) for runtime bindings and gates.
For private visual-study manifests, capture, browser presentation, calibration,
responses, exports, or recovery, read [the private-study protocol](docs/PRIVATE_STUDY.md)
before editing. Keep private stimuli, answer keys, identifiers, and session data
outside public source; documentation examples must be generic.

## Architecture invariants (do not break)

1. **Import safety.** `import itrace` and the entire test suite MUST succeed
   with zero optional dependencies installed. Never add a top-level
   `import cv2` / `mediapipe` / `streamlit` / `matplotlib` / `fastapi` in
   `src/itrace/` root modules. Hardware/dashboard/figure/web deps are imported
   lazily inside functions (`capture._require_capture_deps`,
   `dashboard._require_streamlit`, `live.server.create_app`). The `viz/`
   subpackage is matplotlib-gated like `capture/` is OpenCV-gated.
2. **Pure core vs thin shell.** Keep scientific computation and `study/` state,
   manifest, clock, and storage logic camera-independent. `capture.py` and the
   optional live/dashboard/CLI shells own hardware and serving.
   `iris_landmarks_to_sample` takes plain float arrays for headless checks.
3. **No mocks.** Validate detectors against `synthetic.*` ground truth and the
   independent `tests/fixtures/reference_impl.py` oracle. Never assert against a
   mock that encodes the author's expectation.
4. **Units never leak.** Angles in degrees of visual angle, time in seconds,
   pupil size carries a `PupilUnit`. Direction convention: 0° = right,
   +90° = up (screen-y is negated in `geometry.direction_deg`).
5. **Determinism.** Every synthetic generator and stochastic routine takes a
   `seed`.

## Private-study invariants

- Preserve frozen manifest order and PNG bytes, measured setup, native capture
  timestamps, confirmed clock exchanges, and durable event acknowledgements.
- Keep nine calibration targets, five held-out targets, and five post-session
  repeats distinct. Post-validation uses the initial frozen fit; completion and
  gaze adequacy are separate. Preserve explicit responses, omissions, and technical
  interruptions as separate outcomes.
- Keep `live-html` available as its own diagnostic workflow. Private-study data
  must not update historical pilot ledgers, manuscript inputs, or release evidence.
- Report N=1 session diagnostics only: no population, device-accuracy, physical
  pupil-diameter, or saccade-onset claims from this interface.

## Gates

```bash
uv run pytest --cov=itrace      # ≥90% coverage, no-mocks
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run mypy src/itrace          # --strict (configured in pyproject)
```

## Layout

```
src/itrace/      pure analysis core + thin capture/dashboard shells + cli
  capture.py     optional webcam backend + canonical capture I/O helpers
  live/          local HTML orchestrator (state, analysis, server submodules)
  study/         private manifest, session state, calibration, journal/reconstruction
  study_cli.py   study and study-report CLI adapters
  web_static/    separate diagnostic and private-study browser assets
  eyemodel.py    3-D eyeball forward model + pinhole projection → landmarks
  scene.py       animated gaze/pupil trajectory + full closed-loop validation
tests/           no-mocks suite; tests/fixtures/reference_impl.py = oracle
scripts/         generate_figures.py, generate_loop_animation.py → output/figures/
docs/manuscript/      Pandoc manuscript (template numbering/citation conventions)
docs/            RESEARCH_BRIEF.md (the ecosystem survey behind the design)
ISA.md           ideal-state articulation, criteria, and verification evidence
```

The 3-D forward model (`eyemodel`) and closed loop (`scene`) must stay an
*independent formulation* from the estimator (`geometry`'s arcsine model): the
forward model uses a 3-D sphere + perspective pinhole projection. If you ever
make the forward model reuse the estimator's inverse, the closed-loop test
becomes a tautology — keep them distinct (see ISC-68).

## Source of truth

`ISA.md` records the ideal state, acceptance criteria (ISCs), and dated verification
evidence. Extend it when behaviour changes; historical receipts do not certify a
new interface. Preserve manuscript, release, and research ledgers unless the task
explicitly owns their regeneration.
