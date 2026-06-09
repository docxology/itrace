# Results: closed-loop recovery {#sec:closedloop}

Running the full loop — 3-D eyeball → perspective projection → landmark extraction
→ estimation → recovery — on the default animated scene (four fixations joined by
three saccades, two pupil dilation events, one blink, at 120 Hz)
recovers gaze with an RMS residual of **0.16°** against the 3-D truth (maximum
0.23°) for gaze within ±15°, and recovers the three scripted saccades exactly
([@fig:loop]). The residual is small but strictly non-zero, confirming the forward
model and estimator are independent ([@sec:forward]).

This is an *internal-consistency verification*: it proves the estimator correctly
inverts the forward model and would expose sign flips, axis swaps, unit errors,
and coordinate-frame mismatches. It is **not** device validation. The 0.16° is a
*lower bound* on real-world error — any idealization the two paths share (pinhole
projection, no corneal refraction, a spherical eye, zero kappa) contributes
exactly zero residual by construction ([@sec:limitations]). The recovered pupil
($r = 1.0$) is a near-identity consistency check (the projected pupil/iris ratio
is a monotone function of pupil radius), confirming the deblink/pipeline preserve
the signal — not pupil-size accuracy. This is the strongest possible evidence for
software geometry consistency inside the current assumptions and deliberately
weaker evidence for the real optical system.

![Closed-loop recovery on the default synthetic scene. Panel A shows the projected binocular landmark geometry and the two iris-centre paths. Panel B overlays true and recovered yaw/pitch, with scripted and detected saccade intervals shaded. Panel C overlays true and recovered pupil dynamics after normalisation, with the blink interval shaded. The figure verifies consistency between an independent 3-D generator and the estimator; shared idealisations mean the residual is a lower bound, not a measured webcam error.](../output/figures/closed_loop_summary.png){#fig:loop width=100%}

The synthetic generator supplies all three target signals jointly — gaze
direction, saccade intervals, and pupil diameter — animated as floating 3-D
eyeball orbs ([@fig:orbs]).

![Synthetic truth and recovered signals rendered together. The 3-D panel shows the true eyeball orientation, gaze rays, iris, and pupil state at one representative frame; the trace panels repeat the recovered yaw, pitch, and pupil signals with true values, saccades, and the blink interval visible for audit. This panel is a visual audit of the synthetic scene and recovery path, not evidence from a recorded participant.](../output/figures/eye_orbs_still.png){#fig:orbs width=100%}
