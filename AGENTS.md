# AGENTS.md — iTrace

Working-tree project: a standalone, self-contained `uv` package. Develop and
test directly here (`uv run pytest tests/`). It is not part of default template
pipeline discovery; render it explicitly from the sibling template checkout with
`--project working/iTrace`.

## Architecture invariants (do not break)

1. **Import safety.** `import itrace` and the entire test suite MUST succeed
   with zero optional dependencies installed. Never add a top-level
   `import cv2` / `mediapipe` / `streamlit` / `matplotlib` / `fastapi` in
   `src/itrace/` root modules. Hardware/dashboard/figure/web deps are imported
   lazily inside functions (`capture._require_capture_deps`,
   `dashboard._require_streamlit`, `live.server.create_app`). The `viz/`
   subpackage is matplotlib-gated like `capture/` is OpenCV-gated.
2. **Pure core vs thin shell.** Everything except `capture.py` (and the
   hardware path of the CLI `record`/`dashboard` commands) is pure NumPy/SciPy
   and must stay unit-testable without a camera. `iris_landmarks_to_sample`
   takes plain float arrays so capture logic is testable headless.
3. **No mocks.** Validate detectors against `synthetic.*` ground truth and the
   independent `tests/fixtures/reference_impl.py` oracle. Never assert against a
   mock that encodes the author's expectation.
4. **Units never leak.** Angles in degrees of visual angle, time in seconds,
   pupil size carries a `PupilUnit`. Direction convention: 0° = right,
   +90° = up (screen-y is negated in `geometry.direction_deg`).
5. **Determinism.** Every synthetic generator and stochastic routine takes a
   `seed`.

## Gates (all must pass before promoting to active/)

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
  eyemodel.py    3-D eyeball forward model + pinhole projection → landmarks
  scene.py       animated gaze/pupil trajectory + full closed-loop validation
tests/           no-mocks suite; tests/fixtures/reference_impl.py = oracle
scripts/         generate_figures.py, generate_loop_animation.py → output/figures/
manuscript/      Pandoc manuscript (template numbering/citation conventions)
docs/            RESEARCH_BRIEF.md (the ecosystem survey behind the design)
ISA.md           ideal-state articulation, criteria, and verification evidence
```

The 3-D forward model (`eyemodel`) and closed loop (`scene`) must stay an
*independent formulation* from the estimator (`geometry`'s arcsine model): the
forward model uses a 3-D sphere + perspective pinhole projection. If you ever
make the forward model reuse the estimator's inverse, the closed-loop test
becomes a tautology — keep them distinct (see ISC-68).

## Source of truth

`ISA.md` is the system of record: the ideal state, the 77 criteria (ISCs), and
the verification evidence. Read it first; extend it when you change behaviour.
