# Results: single-pilot empirical diagnostics {#sec:empirical-pilot}

The live empirical workflow is included here as an **N=1, order-of-magnitude
case study**, not as a participant study or device validation claim. One local
webcam session records fixed-gaze, reading, and center/four-corner target trials;
the backend stores derived capture records, a protocol manifest, a target
schedule, and an experiment report; and the manuscript renderer reads only the
small summary JSON produced from that report. The current pilot source reports
**{{EMPIRICAL_PILOT_STATUS}}**. From that derived bundle, the renderer hydrates
the session metrics as follows: finite-gaze fraction
**{{EMPIRICAL_PILOT_FINITE_GAZE}}**, sampling rate
**{{EMPIRICAL_PILOT_SAMPLING_HZ}}**, sampling-interval coefficient of variation
**{{EMPIRICAL_PILOT_SAMPLING_CV}}**, maximum drift
**{{EMPIRICAL_PILOT_DRIFT}}**, held-out target RMS
**{{EMPIRICAL_PILOT_HELDOUT_RMS}}**, and target acquisition latency
**{{EMPIRICAL_PILOT_LATENCY}}**. The values are stored in
`docs/empirical_pilot_metrics.json`; when the pilot export is absent, the same
render path leaves each field explicitly unavailable rather than filling a
guessed value.

![N=1 empirical session summary generated from the derived experiment report. The cards report finite-gaze fraction, sampling rate, sampling regularity, drift, held-out target residuals, and target acquisition latency as order-of-magnitude session diagnostics when a local pilot export exists; pending fields remain visible before recording. The figure demonstrates the package workflow and current-session quality ledger, not reference-device validation.](../output/figures/empirical_pilot_summary.png){#fig:empirical-pilot width=100%}

For adding more data, the release path now uses
`docs/empirical_sessions_manifest.json` as an intake ledger. Each available
session must name the de-identified participant, device, session group,
replicate identifier, condition, protocol, consent scope, reference-evidence
type, and repo-relative derived report path; planned sessions can be listed
before their exports exist. A non-`none` reference-evidence type counts only
when the row also names a valid repo-relative `reference_artifact` JSON; for the
manual-annotation lane, that artifact stores sampled target-window judgments
over derived report/record files rather than raw video.

The v1 empirical release is intentionally scoped as a single-participant,
single-device, five-session diagnostic pilot, not as a device-validation study.
The generated `docs/empirical_sessions_summary.json` and
[@fig:empirical-sessions-summary] therefore treat participant/device counts as
scope metadata while using five available sessions, five replicate IDs, and the
two collected lighting conditions as the diagnostic readiness criteria. The
current ledger reports **{{EMPIRICAL_SESSIONS_STATUS}}** and is
**{{EMPIRICAL_SESSIONS_READINESS}}**. Under those checked-in diagnostic criteria,
the minimum additional collection needed for v1 is
**{{EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL}}** additional session exports, with
**{{EMPIRICAL_SESSIONS_CONDITIONS_REMAINING}}** additional current-v1
condition(s) required.

The stronger 12-session, three-condition, reference-backed validation plan is
preserved as future scope rather than counted as a current v1 blocker:
**{{EMPIRICAL_SESSIONS_FUTURE_SCOPE}}**. Prompt-only replicates improve
within-participant/device operating-scale coverage, but device validation still
requires reference-device, public-dataset, or manual-annotation evidence backed
by a valid artifact.

![Empirical-session intake summary generated from `docs/empirical_sessions_manifest.json`. The figure reports available sessions, distinct replicates, conditions, reference-evidence count, participant/device scope, held-out residual scale, finite-gaze fraction, total derived samples, and current diagnostic-v1 status. The five-session prompt-only ledger is sufficient for diagnostic v1 readiness while population, cross-device, and device-accuracy claims remain future reference-backed validation scope.](../output/figures/empirical_sessions_summary.png){#fig:empirical-sessions-summary width=100%}

The pilot adds a different kind of evidence from the synthetic results. Synthetic
truth recovery and the 3-D closed loop test the estimator under known gaze,
events, and geometry ([@sec:results]; [@sec:closedloop]); the noise sweep then
shows how idealised landmark jitter propagates through the real estimator
([@sec:noiseresults]). The single local run instead contextualizes the scale of
one real session: finite capture, webcam sampling regularity, short-session drift,
prompted-target residuals, and UI/capture target acquisition latency. In that
interpretation, the finite-gaze fraction and sampling CV are capture-stability
checks, the drift value is a short-session stability check, the held-out RMS is a
prompted-target residual scale, and the latency is a coarse task/UI timing
quantity. None of these should be read as physiological saccade latency or
reference-device gaze accuracy.

The pilot therefore sanity-checks one local operating scale for synthetic data
generation without turning those variables into measured population parameters.
The comparison is deliberately coarse, and [@fig:synthetic-empirical-range-bridge]
renders the same comparison from a generated JSON sidecar so the visual and prose
share one evidence boundary:

: N=1 pilot scale compared with synthetic/model variables. {#tbl:empirical-synthetic-range}

| Pilot diagnostic | Hydrated value | Context | Range interpretation |
|---|---:|---|---|
| Finite gaze | {{EMPIRICAL_PILOT_FINITE_GAZE}} | clean-session finite-sample examples; dropout domains | Supports the clean synthetic demos as plausible for a cooperative local run, while preserving low-light/dropout stress cases. |
| Sampling rate | {{EMPIRICAL_PILOT_SAMPLING_HZ}} | live-style sampling cadence and timestamp grid | Contextualizes webcam-rate synthetic/session examples near a consumer-camera cadence rather than the higher-rate ideal traces used for algorithmic recovery. |
| Sampling CV | {{EMPIRICAL_PILOT_SAMPLING_CV}} | timestamp jitter in synthetic sessions | Gives an order-of-magnitude jitter target for live-style stress tests, separate from deterministic oracle traces. |
| Maximum drift | {{EMPIRICAL_PILOT_DRIFT}} | head-drift and fixation-stability domains | Supports a low-drift baseline example; larger drift domains remain stress tests, not observed claims from this pilot. |
| Held-out target RMS | {{EMPIRICAL_PILOT_HELDOUT_RMS}} | prompted-target residual / calibration-bias scale | Contextualizes screen-prompt and calibration residual scale; it is not iid landmark noise, closed-loop residual, or gaze accuracy against a reference device. |
| Target acquisition latency | {{EMPIRICAL_PILOT_LATENCY}} | target-schedule and UI/capture timing | Contextualizes task-response timing for the live protocol; it is not physiological saccade latency. |

![Synthetic-to-empirical range bridge generated from `output/figures/synthetic_empirical_range_bridge.json`. The rows place N=1 empirical session values beside synthetic-domain summaries, idealized landmark-noise sweep values, or descriptive statistical diagnostics, and each row is labelled as observed local scale, stress-domain only, or not directly comparable. The figure uses the pilot to contextualize synthetic defaults and stress ranges; it does not validate device accuracy, device performance, or webcam generality.](../output/figures/synthetic_empirical_range_bridge.png){#fig:synthetic-empirical-range-bridge width=100%}

This comparison is intentionally asymmetric. The synthetic closed loop reports a
near-ideal residual because both truth and measurements are generated inside a
controlled model, while the N=1 pilot folds in calibration quality, user
compliance, webcam timing, landmark behaviour, screen geometry, and target
eccentricity. A much larger prompted-target RMS in
[@fig:synthetic-empirical-range-bridge] is
therefore not a contradiction of the synthetic validation; it is the practical
scale at which the package's live workflow operates before any reference tracker
is introduced. That makes the pilot useful for choosing realistic demo defaults,
stress-test ranges, and caveats, while leaving real-device validation to a future
public-dataset or reference-device study.
