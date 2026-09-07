# Limitations and accuracy bounds {#sec:limitations}

The conveniences of the pure core do not change the physics of consumer hardware,
and several limits should temper interpretation. The headline limitation is the
unclosed **device validation gap**: iTrace's correctness claims rest on
constructed ground truth and an internal-consistency loop, never on agreement
with a reference measurement of real eyes. Everything below follows from that.

**Gaze and saccades.** Real-frame webcam gaze accuracy depends strongly on the
dataset, camera, lighting, pose, calibration, and model class; large public
benchmarks report errors in physical screen units or degrees under their own
protocols rather than a universal webcam constant [@krafka2016gazecapture;
@zhang2017mpiigaze; @hempel2022l2cs]. iTrace's noise-sensitivity result sharpens
the method-level risk: [@sec:noiseresults] shows saccade *timing* breaks down at
a landmark perturbation of ≈0.9 px, so velocity-thresholded webcam saccade timing
is theoretically marginal before real-world lens, illumination, head-pose,
landmark-bias, and calibration effects are added.

**Pupil.** Webcam pupil diameter now has dedicated learned-model and dataset work
[@shah2024eyedentify], mature pupil-detection algorithms and platforms
[@fuhl2015else; @santini2018pure; @zandi2021pupilext], and downstream packages
for transparent preprocessing, visualization, and statistical modelling once
pupil samples exist [@zhang2026pupeyes; @mittner2020pypillometry]. iTrace now
includes a pure eye-crop segmentation helper, but it reports image pixels or
pupil/iris-relative values only. It does not implement an absolute millimetre
calibration path, and its live MediaPipe pupil channel remains a relative proxy.
The package surfaces this rather than implying IR-grade precision, and the pupil
noise-sensitivity result is conditional on a modelling assumption
([@sec:noisemodel]), not measured.

**The expanded analysis layer describes, it does not validate.** The
descriptive statistics, distribution fits, scanpath spread and entropy metrics,
bootstrap confidence interval of [@sec:diststats], and external benchmark report
all operate on the events or truth files they are handed. They quantify the
*shape* of a scanpath and the agreement with a supplied label set precisely, but
a crisp gamma fit, a narrow exponent interval, a small BCEA, or high agreement
with a comparator detector says nothing about whether the underlying gaze tracked
a real eye — it is description downstream of the same unvalidated estimates or
caller-supplied labels, not independent corroboration of them.

**The live HTML app is a diagnostic interface, not a validation study.** A large
zoomed eye crop, stable FPS counter, and plausible rolling saccade/pupil plots
prove that the webcam, MediaPipe, and Python analysis path are wired together.
They do not prove gaze accuracy, pupil-diameter accuracy, or event timing
agreement with an independent reference device. The guided empirical-session
workflow improves that diagnostic layer by recording fixed-gaze, reading, and
center/four-corner target trials and reporting drift, jitter, finite fractions,
held-out target residuals, and target acquisition latency. Those quantities are
valuable for the current session and screen geometry, but the prompts are not a
reference eye tracker; they do not establish accuracy across cameras,
participants, lighting, head pose, or tasks.

**Prompted sessions support diagnostic release, not device validation.** The
empirical-session manifest and summary figure provide a disciplined way to add
repeated sessions from the planned single participant and single device. The
five collected sessions are enough for the declared diagnostic-v1 scope because
they exercise the guided workflow across the two recorded lighting conditions
and preserve derived records for rerendering. They do not sample population
variability, cross-device variability, or independent gaze truth. The stronger
validation scope therefore remains future work: more sessions, the backlit
condition, and at least one reference-device, public-dataset, or
manual-annotation lane under the preregistered split/metric policy.

**Idealizations the validation cannot see.** The closed loop and the noise model
share assumptions — pinhole optics, no refraction, spherical eye, zero kappa, iid
landmark noise — and are blind to any error those introduce ([@sec:discussion]).
Spatial/temporal landmark-noise correlation, heteroscedasticity, and systematic
bias are unmodelled.

## Future work

The honest path to closing the device validation gap is empirical: a head-to-head
against a reference eye-tracker or a public webcam dataset (MPIIGaze, GazeCapture)
on real frames, reporting error in degrees and pupil correlation. The guided
session protocol added here should become a preregistered calibration and
screen-target block inside that larger study, not a substitute for it. Beyond that, a
calibrated absolute-diameter pupil model trained on PupilSense/EyeDentify data
[@shah2024eyedentify] or validated against PupilEXT-style measurement protocols
[@zandi2021pupilext],
smooth-pursuit and post-saccadic-oscillation classification to match REMoDNaV's
four-class output [@dar2021remodnav], screen-space calibration, and binocular
disparity each slot in behind the existing typed interfaces without disturbing the
verified core.
