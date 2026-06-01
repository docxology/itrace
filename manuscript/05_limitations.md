# Limitations and accuracy bounds {#sec:limitations}

The conveniences of the pure core do not change the physics of consumer hardware,
and several limits should temper interpretation. The headline limitation is the
unclosed **device-validation gap**: iTrace's correctness claims rest on
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
[@shah2024eyedentify], while downstream packages emphasise transparent
preprocessing, visualization, and statistical modelling once pupil samples exist
[@zhang2026pupeyes; @mittner2020pypillometry]. iTrace does not implement an
absolute pupil segmentation or millimetre calibration path, so its live pupil
channel remains a relative proxy. The package surfaces this rather than implying
IR-grade precision, and the pupil noise-sensitivity result is conditional on a
modelling assumption ([@sec:noisemodel]), not measured.

**The expanded analysis layer describes, it does not validate.** The
descriptive statistics, distribution fits, scanpath spread and entropy metrics,
and the bootstrap confidence interval of [@sec:diststats] all operate on the
events a detector returns. They quantify the *shape* of a scanpath precisely, but
a crisp gamma fit, a narrow exponent interval, or a small BCEA says nothing about
whether the underlying gaze tracked a real eye — it is description downstream of
the same unvalidated estimates, not independent corroboration of them.

**The live HTML app is a diagnostic interface, not a validation study.** A large
zoomed eye crop, stable FPS counter, and plausible rolling saccade/pupil plots
prove that the webcam, MediaPipe, and Python analysis path are wired together.
They do not prove gaze accuracy, pupil-diameter accuracy, or event timing
agreement with an independent reference device.

**Idealizations the validation cannot see.** The closed loop and the noise model
share assumptions — pinhole optics, no refraction, spherical eye, zero kappa, iid
landmark noise — and are blind to any error those introduce ([@sec:discussion]).
Spatial/temporal landmark-noise correlation, heteroscedasticity, and systematic
bias are unmodelled.

## Future work

The honest path to closing the device-validation gap is empirical: a head-to-head
against a reference eye-tracker or a public webcam dataset (MPIIGaze, GazeCapture)
on real frames, reporting error in degrees and pupil correlation. Beyond that, an
absolute-diameter pupil model trained on PupilSense/EyeDentify data
[@shah2024eyedentify],
smooth-pursuit and post-saccadic-oscillation classification to match REMoDNaV's
four-class output [@dar2021remodnav], screen-space calibration, and binocular
disparity each slot in behind the existing typed interfaces without disturbing the
verified core.
