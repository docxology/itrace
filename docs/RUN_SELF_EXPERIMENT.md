# Run a private self-experiment

Use the [private-study contract](PRIVATE_STUDY.md) for manifest/schema details.
This is an exploratory one-person workflow, without device-accuracy or population
claims. Software checks do not substitute for your measured setup and calibration.

## Before collecting

1. Freeze the manifest, task rules, trial order, deadlines and image hashes. Keep
   the original input packet and all outputs outside the public source checkout.
2. Use an installed environment with the `web` and `capture` extras. Run
   `itrace study-check --manifest PRIVATE/study.json --output-dir PRIVATE/sessions`.
   `--out PRIVATE/new-preflight.json` retains a new receipt. Repeating the check
   creates no sessions and does not overwrite an earlier receipt.
3. Close other camera applications, face the camera with stable lighting, and
   run `itrace camera-probe --camera 0 --frames 15`. It must report actual frames
   and finite face-valid gaze. A camera that opens without finding your face
   is not ready for calibration; reposition and repeat the probe.
4. Measure lit screen width/height and eye-to-screen distance in millimetres.
   Have your private participant and device labels ready. Pixel geometry is
   detected automatically; it cannot supply physical distances.

## Launch and complete

Run `itrace study --manifest PRIVATE/study.json --output-dir PRIVATE/sessions`.
Use a dedicated terminal or an operator-managed process outside an agent host's
cancelled background commands. The server binds loopback, defaults to port 8766,
and keeps capture off until you explicitly start it. `--host`/`--port`/`--camera`
are explicit operator choices; never terminate a foreign server to claim its port.

Open `/study` in a dedicated browser, read the task-rule disclosure, enter measured
setup, acknowledge private storage and prepare in fullscreen. Keep that browser,
display, camera and viewing position unchanged. Then:

- Start webcam and wait for fresh valid gaze. The page explains missing samples.
- Calibrate: nine fit targets, followed by five separate held-out targets. Look at
  each dot until it moves. Passing these checks enables practice.
- Answer each practice/scored trial with a displayed key or click. Honor rests.
  Only practice provides correctness feedback; deadlines remain explicit timeouts.
- Finish five post-session gaze checks using the frozen initial calibration.
  Poor gaze can be recorded as inadequate; do not refit it into apparent success.
- Save the session. Check the saved state, answers/timeouts/unattempted trials,
  capture consistency and post-validation adequacy. Keep the entire directory,
  including inputs, native captures, journal, metadata and export checksums.

## When interrupted

Escape/Stop aborts; tab hiding, fullscreen exit, resizing and connection/capture
faults can invalidate the session. Save partial records. **Save records and start
a new session** retains the old export and allocates a separate session; it never
resumes scored trials. Retry the same action after an ambiguous network timeout.
Same-page measurements remain available, but acknowledgements/calibration repeat.
After a server crash, retain the original files and use
`itrace study-report SESSION_DIR --out PRIVATE/new-report.json` offline.

Readiness before the participant acts means input/software/storage checks passed,
the real camera is accessible, and the presenter works. Readiness to score trials
requires that session's measured geometry, live capture, held-out calibration and
clock checks. Neither boundary establishes physiological saccade onset, physical
pupil diameter, ethics status or broader generalization.
