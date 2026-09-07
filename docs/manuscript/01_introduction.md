# Introduction {#sec:introduction}

Eye movements are a rich, non-invasive window onto attention, cognition, and
oculomotor health. Three signal families dominate research use: the **gaze
trajectory** (where the eyes point over time), **saccades** (the fast ballistic
movements between fixations, with characteristic direction, amplitude, and
peak-velocity dynamics), and **pupil diameter** (a marker of arousal and
cognitive load). Dedicated infrared eye-trackers measure all three at high
precision, but their cost restricts access; commodity cameras offer reach but
force software to confront poorer optics, lower frame rates, browser/camera
permission boundaries, illumination changes, and weaker calibration. A maturing
open-source ecosystem now estimates related signals from these imperfect inputs:
browser trackers such as
WebGazer [@papoutsaki2016webgazer], large webcam and mobile gaze datasets such as
GazeCapture [@krafka2016gazecapture] and MPIIGaze [@zhang2017mpiigaze],
appearance-based gaze estimators such as L2CS-Net [@hempel2022l2cs],
landmark-based capture stacks such as MediaPipe Iris [@vakunov2020mediapipeiris],
velocity- and dispersion-threshold event classifiers
[@salvucci2000identifying; @nystrom2010adaptive; @dar2021remodnav],
microsaccade detectors [@engbert2003microsaccades], file-oriented processing and
quality-reporting pipelines [@krause2023pymovements; @jakobi2024quality], and
webcam or offline pupillometry tools [@shah2024eyedentify; @zandi2021pupilext;
@fuhl2015else; @santini2018pure; @zhang2026pupeyes; @mittner2020pypillometry;
@cai2024opendpsm].

Two problems recur across these tools. First, the *capture* layer (camera
drivers, MediaPipe, browser WebRTC [@papoutsaki2016webgazer], integrated
open-source platforms such as Pupil [@kassner2014pupil]) is tightly coupled
to the *analysis* layer, so the scientific algorithms cannot be exercised or
checked without hardware — which makes continuous integration, regression
testing, and reproducible benchmarking awkward. Second, "validation" is often
demonstrated on recorded data without a ground-truth oracle, so a detector that
silently mis-handles noise, blinks, or NaNs can pass unnoticed.

The existing tools each occupy a different point in this design space, and
iTrace is positioned deliberately against them. WebGazer
[@papoutsaki2016webgazer] pioneered in-browser webcam gaze but fuses capture,
calibration, and estimation into one runtime. GazeCapture and MPIIGaze provide
the public-data substrate for real-frame gaze evaluation
[@krafka2016gazecapture; @zhang2017mpiigaze], while L2CS-Net demonstrates how
appearance models can predict pitch/yaw from unconstrained images
[@hempel2022l2cs]. pymovements [@krause2023pymovements] offers a mature,
file-oriented processing pipeline and now anchors explicit data-quality
reporting standards [@jakobi2024quality], but it assumes the gaze samples
already exist. REMoDNaV [@dar2021remodnav] is a strong event classifier whose
four-class fixation / saccade / smooth-pursuit / post-saccadic-oscillation output
is richer than ours, but it classifies an input signal rather than reasoning
about how that signal was estimated from a camera. PupilSense/EyeDentify
[@shah2024eyedentify] addresses webcam pupil diameter with learned models and a
dataset; PupilEXT, ElSe, and PuRe represent mature pupil-detection platform and
algorithm lines [@zandi2021pupilext; @fuhl2015else; @santini2018pure]; PupEyes
and pypillometry focus on reproducible pupil preprocessing and analysis after
samples have been recorded [@zhang2026pupeyes; @mittner2020pypillometry];
Open-DPSM models luminance-driven pupil responses in dynamic stimuli
[@cai2024opendpsm]. iTrace does not try to out-measure any of these on real eyes.
It occupies a narrower gap: a *verification-first*, hardware-decoupled
implementation of established gaze, saccade, microsaccade, pupil, and descriptive
scanpath algorithms whose correctness is pinned to constructed ground truth and
whose outputs can later be compared against public datasets, reference devices,
or other packages when the caller supplies truth.

iTrace makes three contributions. First, it implements established algorithms
in a pure NumPy/SciPy core with no hardware dependency, then verifies every
detector against synthetic signals whose events are known by construction. Second,
it keeps webcam frames, MediaPipe iris landmarks, live HTML rendering, and
dashboard/figure backends in optional shells whose outputs are ordinary typed
gaze/pupil/capture records. The landmark-to-gaze mathematics is therefore
testable without a camera, and the browser never becomes a second analysis
implementation. Third, it adds bounded interfaces around that verified core:
descriptive event statistics, maximum-likelihood distribution fitting, scanpath
spread and entropy metrics, a bootstrap interval on the main-sequence exponent,
a deterministic publication-figure gallery, a pure eye-crop pupil-segmentation
helper that reports pixels or pupil/iris-relative units only, a benchmark report
for caller-supplied truth files, and a guided live empirical workflow that writes
derived records rather than raw eye video. The graphical summary in
[@fig:graphical-abstract] shows the same separation visually. [@sec:methods]
describes the architecture and algorithms; [@sec:results] reports the
ground-truth verification, N=1 local empirical diagnostics, and figure gallery;
[@sec:discussion] and [@sec:limitations] state the accuracy boundaries that still
constrain webcam eye tracking.
