# Methods: 3-D forward model and the closed loop {#sec:forward}

Detectors verified against a one-dimensional position signal are only as
auditable as the path that produced that signal. To exercise the
geometry/landmark path end-to-end — the part a real webcam pipeline actually runs
— we add a 3-D forward model (`eyemodel`) and an animated scene (`scene`) that
drive the *real* estimator from synthetic anatomy with known truth.

## The eyeball-to-landmark forward model

A parameterised eyeball — centre, radius, gaze yaw/pitch, and pupil and iris
radii — is rotated to a true gaze direction. The iris ring and pupil disc are
placed on the sphere surface at that orientation and **perspective-projected**
through a pinhole camera to the normalised, MediaPipe-shaped landmark array the
capture layer consumes. The anatomical eye corners are fixed: they sit on the
face, not on the rotating globe, so they do not move with gaze. Because the
corners are stationary while the iris translates across the visible sclera, the
iris-to-corner offset encodes gaze exactly as a real face mesh would, which is
what lets the same `geometry` routine that an actual recording exercises be the
one under test. The projection is a forward map from a physical pose to image
coordinates; it is deliberately not the inverse the estimator computes.

## The animated scene

The scene (`scene`) drives a deterministic trajectory of fixation targets joined
by minimum-jerk saccades — the same velocity profile the synthetic single-saccade
generator uses — together with a pupil baseline carrying Gaussian dilation events
and blink intervals. At each frame it emits both the per-frame 3-D truth (gaze
yaw/pitch, pupil size, blink state) and the projected landmark array, so the
entire recording has a ground truth that no detector could have peeked at.

## The closed loop

The **closed loop** then feeds those landmarks back through the production
estimator — `iris_landmarks_to_sample` reconstructs a `GazeStream`, which the
standard `pipeline` analyses into fixations, saccades, and a pupil summary — and
measures recovery against the 3-D truth: gaze angular error in degrees, saccade
detection against the scripted saccades, and pupil correlation with the scripted
baseline.

The design is deliberate on one point: the forward model (3-D sphere + perspective
projection) and the estimator (the arcsine sphere approximation of
[@sec:geometry]) are **independent formulations**, not two halves of one
identity. Their agreement therefore tests the geometry rather than restating it —
a tautology would yield exactly zero residual — while the small non-zero residual
is the price of the estimator's approximation, an honest, bounded quantity rather
than a hidden one. What this construction can and cannot establish is taken up in
[@sec:closedloop] and [@sec:limitations]: it certifies that the analysis code
recovers truth from a geometrically faithful but *idealized* eye, and it is
silent — by construction — about corneal refraction, the kappa angle, lens
distortion, and every other physical effect the pinhole model omits.
