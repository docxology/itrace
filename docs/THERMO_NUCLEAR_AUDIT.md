# Thermo-Nuclear Code Quality Audit — iTrace v0.4.1

**Date:** 2026-06-02  
**Gate refresh:** 2026-06-09  
**Scope:** Full package at `working/iTrace`  
**Gates verified:** 528 tests / 90.23% coverage, `mypy --strict` clean (56 source files), `ruff` clean, `ruff format` clean, `import itrace` safe without optional extras  
**Verdict:** REJECT (initial) → **APPROVE after remediation** (2026-06-02)

Remediation items 1–6 implemented; June 9 verifier refresh gates: 528 tests,
90.23% coverage, ruff format over 121 Python files, and mypy over 56 source
files clean.

---

## Executive summary

The pure analysis core (geometry → velocity → detection → pipeline → stats → validation) is coherent, typed, and testable headless. Optional layers mostly respect lazy-import boundaries.

The shell orchestration layer (`cli.py`, `live.py`, `capture.WebcamSource`) carried confirmed duplication and a 644-line god module that hand-built JSON contracts instead of reusing canonical types. That failed the thermo-nuclear bar despite green CI.

**Minimum bar to APPROVE:** canonical capture I/O, unified webcam loop, shared minimum-jerk, pipeline judo + `replace()`, validated report builders for live, and `live/` decomposition.

---

## 1. Structural regressions / architecture-boundary leaks

| ID | Location | Severity | Finding | Remedy |
|----|----------|----------|---------|--------|
| 1.1 | `viz/*.py` | medium | Matplotlib at module load in optional viz layer; `import itrace` stays clean | Document in AGENTS as matplotlib-gated (like capture) |
| 1.2 | `live._analysis_payload`, `LiveState.export` | high | Hand-built report dicts bypass `SessionReport` / `reporting.validate_report_payload` | `reporting.empty_session_report` / `error_session_report` |
| 1.3 | `live.create_app` globals hack | medium | `globals()["WebSocket"] = WebSocket` | Move routes to `live/server.py` with lazy FastAPI import |

## 2. Missed code-judo simplifications

| ID | Duplication | Remedy | Status |
|----|-------------|--------|--------|
| 2.1 | samples→streams (live, cli, export) | `capture.samples_to_streams` | remediated |
| 2.2 | capture-records CSV (cli, live) | `capture.write_capture_records_csv` | remediated |
| 2.3 | `_native_stderr` (cli, live) | `capture.native_stderr` | remediated |
| 2.4 | `WebcamSource.frames` / `live_frames` | `_iter_webcam_face_mesh_frames` | remediated |
| 2.5 | `_detect_events` dual `detect_ivt` | compute threshold once | remediated |
| 2.6 | minimum-jerk in scene + synthetic | `synthetic.minimum_jerk_profile` | remediated |
| 2.7 | `object.__setattr__` on SessionReport | `dataclasses.replace` | remediated |

## 3. Spaghetti / branching complexity

| ID | Location | Severity | Remedy |
|----|----------|----------|--------|
| 3.1 | `live._analysis_payload` | high | Split + delegate to `reporting` builders |
| 3.2 | `cli.validate_recording` | medium | Future: `validation.recording_validation_payload` |
| 3.3 | adaptive vs fixed IVT | medium | See 2.5 |

## 4. Type-contract problems

| ID | Location | Severity | Notes |
|----|----------|----------|-------|
| 4.1 | `create_app() -> Any` | medium | Typed when FastAPI loaded in `live/server.py` |
| 4.2 | `dict[str, object]` payloads | medium | TypedDict deferred; validated via `reporting` |
| 4.3 | WebSocket calibration parse | low–medium | Acceptable for now |

## 5. File-size / decomposition

| File | LOC | Action |
|------|-----|--------|
| `live.py` | 644 | Split into `live/` package |
| `cli.py` | 477 | Optional future split |
| `capture.py` | 491 | Unified loop reduces effective complexity |

## 6. Modularity issues

| ID | Finding | Severity |
|----|---------|----------|
| 6.1 | CLI + live duplicate capture persistence | high → remediated |
| 6.2 | Per-frame full rolling analysis | medium (documented hot path) |
| 6.3 | `session_statistics` correctly canonical | positive |

## 7. Legibility nits

- `viz/__init__.py` duplicate gallery import (low)
- `_method_name` maps unknown to `"ivt"` (low)
- FPS inconsistency on no-face frames in `live_frames` (medium → remediated)

---

## Positive signals

- `import itrace` safe without optional deps
- Pure landmark math in `capture.py`
- Clean `pipeline.analyze_gaze` → `SessionReport` composition
- `reporting.validate_report_payload` on CLI analyze path
- Closed-loop independence (scene vs geometry) preserved
- No file >1k LOC

## Remediation order (implemented)

1. Canonical capture I/O in `capture.py`
2. Unified `WebcamSource` frame loop
3. Shared `minimum_jerk_profile` in `synthetic.py`
4. Pipeline `_detect_events` + `analyze_session` cleanup
5. `reporting` empty/error report builders + live adoption
6. Split `live.py` into `live/` subpackage
7. Fix ruff import-order issues in scripts/viz
