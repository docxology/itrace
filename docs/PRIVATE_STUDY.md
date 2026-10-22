# Private webcam visual study

`study` runs a local, single-participant PNG visual task with webcam-derived gaze
and relative pupil measurements. It is separate from the stock `live-html`
diagnostic dashboard and the historical manuscript pilot. Original scientific
publications, manuscript, ledgers, figures, and release evidence remain untouched.

## Launch and measured setup

Use an existing iTrace environment with the `capture` and `web` extras. Select
private storage outside the public checkout for the manifest, PNGs, identifiers,
answers, and all session/report output. The placeholders below are operator
choices, not bundled data or repository directories.

```bash
itrace study --manifest PRIVATE/study.json --output-dir PRIVATE/sessions
itrace study-report SESSION_DIR --out NEW_REPORT
```

Open the local server at `http://127.0.0.1:8766/study` (default host/port). Keep
the service on loopback. `--camera` defaults to 0. Launching creates a unique
session directory; capture begins only when **Start webcam** is pressed.
`SESSION_DIR` is that directory, not an export subdirectory. `NEW_REPORT` must be
a new file with an existing parent; reconstruction needs no camera or browser.

1. Close other camera applications and use one display. Measure the display's
   physical width/height and eye-to-display distance in millimetres. Enter display
   pseudonymous participant/device identifiers. The page detects window CSS
   pixels and estimates display device pixels from the browser screen and scale;
   it refreshes the measurement after fullscreen settles. Physical dimensions
   and viewing distance must still be measured, never inferred from pixels.
2. Acknowledge private derived storage and confirm the measurements. **Prepare in
   fullscreen** binds browser, viewport, device-pixel ratio, and browser epoch;
   physical measurements are stored in metres. Keep geometry and camera fixed.
3. Press **Start webcam** and inspect camera/face/gaze status. Missing permission,
   missing dependencies, or unavailable finite gaze must be resolved before
   calibration. A camera-access probe does not establish real gaze quality;
   the live face/gaze check and held-out calibration must pass for this session.
4. Press **Calibrate and validate**. Follow nine fit targets and five distinct
   held-out targets. Each uses the manifest's settle/collect window. Readiness
   requires sufficient valid samples, acceptable held-out residuals, active
   capture, and a current confirmed clock exchange.
5. Start each trial in manifest order, respecting any rest interval. The decoded
   PNG, query, and choices appear together. Use a printed response key or click
   a choice. Practice feedback is separate; scored correctness stays in private
   reports. A deadline without an eligible response is an explicit omission.
6. After the final trial, keep capture running and press **Finish gaze validation**.
   Five held-out positions repeat with the initial fit frozen. Each is collected
   once; inadequate post data remain explicit rather than triggering a refit.
   Save the completed session, then inspect its report.

## Version 1 manifest

The JSON object has exact top-level fields `schema` (`itrace.private-study`),
`version` (1), `study_id`, `title`, `limits`, and `trials`. The loader rejects
unknown fields, duplicate JSON keys, nonfinite values, and unsupported versions.
The list contains 1–1024 uniquely identified trials, at least one scored trial,
and any practice trials before scored trials. Its realized order is immutable.

| Trial field | Contract |
| --- | --- |
| `trial_id`, `block_id`, `task_id`, `family_id` | Explicit identifiers; trial IDs are unique |
| `role` | `practice` or `scored`; practice/scored family IDs and stimulus hashes are disjoint |
| `stimulus` | Object with manifest-relative `path`, lowercase `sha256`, integer `width` and `height` |
| `query`, `options` | Prompt text and 2–12 `{key, label}` choices; unique single lowercase letter/digit keys |
| `correct_key` | Private answer key matching one option; omitted from browser protocol data |
| `deadline_s`, `rest_s` | Response deadline in seconds; optional `rest_s` defaults to 0 |

Use canonical relative PNG paths without traversal or symlinks. The loader
checks exact file hashes, PNG structure/CRC, dimensions, and bounded decoding.
Supported PNGs are noninterlaced 8-bit grayscale, RGB, grayscale-alpha, or RGBA.
Manifest size is bounded to 2 MiB, individual PNGs to 16 MiB compressed/64 MiB
decoded, and aggregate PNGs to 64 MiB compressed/256 MiB decoded. Within-role
repeats are allowed; reports disclose trial, unique-image-hash, and family counts.
Private answers and local paths are excluded from the browser manifest payload;
the session snapshot and exports remain private.

All `limits` fields are required; choose and freeze them before collection:

| Field | Accepted range / meaning |
| --- | --- |
| `max_sync_error_s` | 0.001–0.25 s clock-mapping uncertainty bound |
| `sync_max_age_s` | 1–600 s maximum age of confirmed synchronization |
| `max_event_lag_s` | 0.05–5 s event/confirmation lag bound |
| `max_validation_error_fraction` | 0.001–0.15 held-out p95 error as a screen-diagonal fraction |
| `settle_s`, `collect_s` | 0.5–5 s settling, then 1–10 s collection per target |
| `min_samples` | Integer 10–10000 valid samples per dwell |
| `min_valid_fraction` | 0.6–1 valid-sample fraction |

## Timing, outcomes, and interpretation

The browser records actual requestAnimationFrame callback timestamps for target
and trial presentation, the displayed image's CSS rectangle, and key/click events.
Confirmed four-timestamp browser/host exchanges provide offset and uncertainty;
native capture timestamps remain absolute host-monotonic seconds. Stale clocks,
invalid timing, and duplicate event identities are rejected. rAF timestamps are
software presentation estimates, not measured physical display onset or camera
sensor exposure. Reaction times carry that limitation.

Only an eligible explicit response counts as an answer. Reports separate
`answered`, `timeout`, `unattempted`, `interrupted`, and `aborted` outcomes.
`correctness_denominator` counts scored explicit responses; omissions and technical
losses must be reported alongside it. Practice does not enter scored correctness.

The affine fit maps native gaze degrees to viewport CSS pixels. Held-out
residuals describe target agreement in that session, not reference-device
accuracy. Post-validation repeats the five initial held-out coordinates without
changing the fit and records pre/post residual changes. `behavioral_trials_finished`,
session `state`, and `calibration_diagnostics.post_validation_adequate` answer
different questions: collecting five post dwells can complete a session even
when gaze quality is inadequate. Reconstruction recomputes fit/residuals from
saved dwell means, not a second replay of raw gaze.

Trials and repeated sessions from one participant remain N=1. This workflow
supports no population generalization, device-accuracy validation, physical
pupil diameter in millimetres, or saccade-onset claims. Relative pupil proxies
remain sensitive to illumination, pose, and camera/detector conditions.

## Storage and interruption

Session directories are created privately (mode 0700). Exact manifest/PNG
snapshots, `metadata.json`, `journal.jsonl`, native `capture.csv`, and
`capture_epochs.jsonl` remain on disk. Acknowledged journal events are flushed
and fsynced. Capture retains frame IDs, timestamps, measurements/units, FPS, and
per-row quality; preview delivery does not drive acquisition. Capture sidecars
bind each native frame to its host receipt, locked recording time, and
last acknowledged journal sequence. Reconstruction recomputes clock validity
against that exact capture state. Undeclared earlier development formats are
rejected explicitly; retain their original software and receipts for replay.
Runtime/source
provenance describes the actual execution bindings and is not a clean-Git claim.

**Save partial session** works before completion. Each export uses a fresh
directory containing `report.json`, `trials.csv`, `responses.csv`, clock,
calibration, validation, post-validation and event JSONL files, plus checksums.
Earlier exports remain intact; an export during collection is a partial snapshot.
Storage is derived gaze/pupil/response data plus the private input PNGs and answer
manifest. No raw webcam video or persisted eye-crop images are written; a transient
eye preview can be displayed during preparation. These derived records are still
private data, not anonymization guarantees.

Escape or **Stop session** aborts collection. Tab hiding, viewport changes,
fullscreen exit, capture faults, or browser connection loss invalidate readiness.
Stop and retain the partial session; start a new session after resolving the
cause. Do not treat a reload/reconnect as resuming scored trials. Connection-loss
handling does not promise an exact hardware shutdown deadline.

The recovery panel's **Save records and start a new session** first saves the
stopped session, then allocates a separate session directory and capture service.
It never clears or rewrites the earlier journal, captures or exports. Measurements
remain in the form for a same-page retry, but both acknowledgements are cleared
and calibration must be repeated. Identifiers and measurements are not saved in
browser local storage. After a reload, enter them again.

Terminal saves, repeated Stop, identical setup retries and new-session request
IDs are idempotent within the running server. The server retains the latest
new-session receipt so a lost acknowledgement can be retried without allocating
another session. Controls carry the intended session ID; an old tab cannot stop
or overwrite its replacement. An active session must be stopped before it can
be replaced. Repeated saves during active collection remain distinct snapshots.
If a request times out, retry its visible action; a timeout is not proof that the
server failed. Storage failures retain partial files and prevent replacement.

After a crash or failed export, preserve original files and run `study-report`
with a new report path. Reconstruction accepts a valid journal prefix with an
incomplete final line, flags incomplete/capture defects, and rejects interior
corruption. It recomputes outcomes, correctness and timing from saved events and
clock exchanges; it never resumes collection or silently repairs original files.

See [validation boundaries](VALIDATION_PROTOCOLS.md#private-visual-study-diagnostics)
and [development checks](DEVELOPMENT.md#private-study-development). Historical
publication traceability remains in [TRACEABILITY_MATRIX.json](TRACEABILITY_MATRIX.json).
