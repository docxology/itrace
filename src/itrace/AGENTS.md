# src/itrace/ — Agent Notes

The iTrace core package. Modules by concern:
- signal core: `geometry.py`, `saccades.py`, `detection.py`, `mainsequence.py`,
  `pupil.py`, `pupilseg.py`, `pupilphase.py`, `encoding.py`, `eyemodel.py`
- pipeline & capture: `pipeline.py`, `capture.py`, `calibration.py`, `scene.py`
- experiment/eval: `experiments.py`, `empirical.py`, `power.py`, `benchmark.py`, `synthetic.py`
- output: `reporting.py`, `dashboard.py`, `io.py`, `cli.py`, `config.py`, `types.py`
- verification and publication: `verification.py`, `manuscript.py`, `release.py`; consult `docs/RELEASING.md` before preparing or publishing a release
- subpackages: `stats/` (statistics), `viz/` (figures), `live/` (webcam shell), `web_static/` (dashboard assets)

Invariants: pure NumPy/SciPy kernel; synthetic-ground-truth verification
(`synthetic.py` + `tests/fixtures/validation/`); headless-safe; no mocks.
