# Methods: pupillometry {#sec:pupillometry}

The pupil pipeline is a sequence of pure transforms over a `PupilStream`, each
independently testable. It follows the conservative preprocessing vocabulary
common to open pupil-analysis tools: blink handling, interpolation,
baseline-correction, quality reporting, and transparent preprocessing records
[@zhang2026pupeyes; @mittner2020pypillometry]. Blinks are detected as runs of
NaN or sub-threshold samples and removed by linear interpolation with edge
padding, so the partial-occlusion ramp on either side of a closure does not leak
into the signal; a trace with no valid samples is reported as unusable rather
than silently flattened to a constant. Outliers are flagged by the scaled median
absolute deviation ($1.4826\,\mathrm{MAD}$), which is robust to the very spikes it
must catch. The deblinked trace is low-pass filtered with a deterministic
zero-phase Butterworth implementation choice (with a short-trace moving-average
fallback) [@butterworth1930theory] and baseline-corrected, subtractively or
divisively, against a chosen pre-event time window.

The runtime `PupilConfig` now controls the same decisions the standalone
functions expose: the low-confidence pupil threshold, interpolation padding, and
Butterworth cutoff/order. Reports include both cleaned-signal summaries
(mean/standard deviation/min/max, phase peak and trough counts) and raw-quality
summaries (valid fraction, blink fraction/count, median sample interval, peak
dilation velocity, and peak constriction velocity). The live webcam pupil channel
therefore remains a relative proxy, but the analysis path records how much of the
trace was usable and how strongly the cleaned pupil changed over time.
Brightness and contrast effects are not modelled away in this version; tools such
as Open-DPSM show why dynamic visual confounds need explicit modelling when pupil
size is interpreted cognitively [@cai2024opendpsm].

The separate `pupilseg` module covers a narrower, testable image-space step for
callers that already have an eye crop. It thresholds the finite crop intensities,
selects the largest dark connected component, reports centroid/radius/area and
contrast-derived confidence, and converts that segmentation into either a
`PupilUnit.PIXELS` sample or a pupil/iris-relative sample when an iris-radius
normalizer is supplied. This deliberately does less than established pupil
detectors and platforms such as ElSe, PuRe, or PupilEXT
[@fuhl2015else; @santini2018pure; @zandi2021pupilext]: it is a pure-core
fixture-verified path for pixels/relative measurements, not a millimetre
calibration or device validation claim.

For closed-loop and online use, a causal `PhaseDetector` classifies each streamed
sample as dilation, constriction, peak, or trough using only the current and
prior samples [@kronemer2024rtpupilphase]. Causality is not merely asserted but
verified: feeding the detector a prefix of a signal reproduces exactly the labels
it assigns to that prefix in a full run, proving no future sample influences a
past label ([@sec:pupilresults]).
