# iTrace Scholarship Audit

Checked on 2026-06-04 for the manuscript scholarship/PDF refresh.

Red-team spot refresh on 2026-06-06 rechecked the most volatile or
implementation-relevant sources: MediaPipe Iris as a commodity-camera landmark
and depth-estimation source, L2CS-Net's protocol-specific MPIIGaze/Gaze360
angular-error results, EyeDentify/PupilSense webcam pupil-diameter scope,
PupEyes' 2026 publication metadata and supported input formats, PupilEXT's six
pupil-detection algorithms, rtPupilPhase's real-time phase-detection scope, and
Starlette's current `httpx2` TestClient transition. No device-validation,
millimetre-pupil, or universal webcam-accuracy claim was added.

## Research path

- Perplexity retry: `llm -m sonar ...` failed with `401 insufficient_quota`.
  The refresh therefore uses direct primary or official sources rather than
  Perplexity synthesis.
- Primary-source policy: bibliography entries were added only for sources with a
  stable DOI, arXiv DOI, official conference page, official JOSS page, official
  Springer page, or official Google Research page.

## Sources added or corrected

| Area | Source | Primary/official URL | Why it matters |
|---|---|---|---|
| Browser gaze | Papoutsaki et al. 2016, WebGazer | <https://www.ijcai.org/Proceedings/16/Papers/540.pdf> | Widely cited in-browser webcam gaze tracker already cited; retained as browser-gaze baseline. |
| Commodity-camera landmarks | Vakunov et al. 2020, MediaPipe Iris | <https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/> | Official source for real-time iris tracking/depth estimation; cited as a landmark source, not as validation evidence. |
| Appearance gaze | Abdelrahman et al. 2022, L2CS-Net | <https://arxiv.org/abs/2203.03339> | Representative open appearance-based pitch/yaw estimator evaluated on MPIIGaze and Gaze360. |
| Public gaze datasets | Krafka et al. 2016, GazeCapture; Zhang et al. 2017, MPIIGaze | <https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html>; <https://arxiv.org/abs/1711.09017> | Public real-frame substrate for future device/dataset validation. |
| Event processing | Dar et al. 2021, REMoDNaV; Krakowczyk et al. 2023, pymovements | <https://link.springer.com/article/10.3758/s13428-020-01428-x>; <https://arxiv.org/abs/2304.09859> | Event-classification and processing comparators for future benchmark harnesses. |
| Data quality | Jakobi et al. 2024 | <https://arxiv.org/abs/2404.00620> | Supports explicit quality-reporting language instead of unqualified validation claims. |
| Webcam pupil diameter | Shah et al. 2025, PupilSense/EyeDentify | <https://arxiv.org/abs/2407.11204> | Corrects the previous EyeDentify-only title to the current arXiv PupilSense application/dataset paper. |
| Pupil preprocessing | Zhang and Jonides 2026, PupEyes; Mittner 2020, pypillometry | <https://pmc.ncbi.nlm.nih.gov/articles/PMC12769653/>; <https://joss.theoj.org/papers/10.21105/joss.02348> | Places iTrace's pupil preprocessing in relation to current open pupil-analysis tools. |
| Pupil segmentation platforms | Zandi et al. 2021, PupilEXT | <https://doi.org/10.3389/fnins.2021.676220>; <https://github.com/openPupil/Open-PupilEXT> | Primary source for an open pupillometry platform that integrates Starburst, Swirski, ExCuSe, ElSe, PuRe, and PuReST; used to bound iTrace's smaller pure eye-crop helper. |
| Classical pupil segmentation | Fuhl et al. 2015, ElSe; Santini et al. 2018, PuRe | <https://arxiv.org/abs/1511.06575>; <https://arxiv.org/abs/1712.08900> | Supports describing iTrace's `pupilseg` as a minimal deterministic connected-component helper, not a replacement for validated ellipse/edge-segment pupil detectors. |
| Pupil light confounds | Cai et al. 2024, Open-DPSM | <https://link.springer.com/article/10.3758/s13428-023-02292-1> | Supports the limitation that dynamic luminance/contrast confounds need explicit modelling. |
| Pupil phase | Kronemer et al. 2025, rtPupilPhase | <https://doi.org/10.3758/s13428-024-02545-7> | Retained as the causal pupil-phase reference. |
| Velocity smoothing | Savitzky and Golay 1964 | <https://pubs.acs.org/doi/10.1021/ac60214a047> | Supports naming the uniform-sampling velocity derivative as a Savitzky-Golay smoothing/differentiation method. |
| Pupil low-pass filter | Butterworth 1930 | <https://cir.nii.ac.jp/crid/1370869854360999320> | Supports naming the zero-phase pupil preprocessing filter family without implying a physiological model. |
| Model selection | Akaike 1974; Burnham and Anderson 2002; Schwarz 1978 | <https://doi.org/10.1109/TAC.1974.1100705>; <https://link.springer.com/book/10.1007/b97636>; <https://doi.org/10.1214/aos/1176344136> | Supports AIC/AICc/BIC ranking, relative Akaike weights, and the finite-candidate interpretation boundary in distribution diagnostics. |
| Distribution distance | Massey 1951 | <https://www.tandfonline.com/doi/abs/10.1080/01621459.1951.10500769> | Supports the Kolmogorov-Smirnov goodness-of-fit statistic while the manuscript keeps the fitted-parameter caveat. |
| Robust exploratory summaries | Bowley 1901; Tukey 1977 | <https://openlibrary.org/books/OL6904986M/Elements_of_statistics>; <https://openlibrary.org/books/OL4877620M/Exploratory_data_analysis> | Supports quartile skewness, IQR summaries, and outside-value framing in the statistical diagnostics payload. |
| Bootstrap intervals and stability | Efron and Tibshirani 1993 | <https://doi.org/10.1201/9780429246593> | Supports seeded percentile-bootstrap intervals for finite-sample median/IQR and main-sequence exponent summaries, plus bootstrap resampling of the model-selection winner as a ranking-stability diagnostic. |
| Probability plotting | Wilk and Gnanadesikan 1968 | <https://doi.org/10.1093/biomet/55.1.1> | Supports the QQ/probability-plot adequacy panel and residual sidecar fields for fitted amplitude distributions. |
| Empirical-CDF residual diagnostics | Anderson and Darling 1952; Massey 1951; Massart 1990; Wilk and Gnanadesikan 1968 | <https://doi.org/10.1214/aoms/1177729437>; <https://www.tandfonline.com/doi/abs/10.1080/01621459.1951.10500769>; <https://doi.org/10.1214/aop/1176990746>; <https://doi.org/10.1093/biomet/55.1.1> | Supports the P-P/CDF residual inset, integrated/tail-weighted fitted-CDF distances, and DKW/Massart reference band as descriptive fitted-model adequacy views, with the fitted-parameter caveat retained. |
| Scanpath sequence comparison | Levenshtein 1966; Salton et al. 1975 | <https://ui.adsabs.harvard.edu/abs/1966SPhD...10..707L/abstract>; <https://dblp.org/rec/journals/cacm/SaltonWY75> | Supports edit-distance and vector-space/cosine n-gram scanpath comparisons as descriptive sequence metrics. |

## Claim changes

- Changed manuscript title from "validated" to "algorithmically verified" to
  avoid implying reference-device validation.
- Reworked the introduction around representative source classes: browser gaze,
  appearance gaze, public gaze datasets, event processors, pupil-processing
  packages, and data-quality standards.
- Replaced universal webcam accuracy prose with conditional language: real-frame
  accuracy depends on protocol, camera, lighting, pose, calibration, and model.
- Replaced the unsupported "real landmark-localisation floor" phrasing with a
  narrower statement: the iTrace sweep shows a method-level fragility at about
  0.9 px before real capture/camera effects are added.
- Kept live pupil output framed as a relative proxy. PupilSense/EyeDentify is
  cited as the relevant absolute-diameter learned-model path, not as evidence
  that iTrace currently estimates millimetres.
- Added a narrower pupil-segmentation distinction: `pupilseg` can segment a
  caller-supplied eye crop and emit pixels or pupil/iris-relative units, but it
  is not an ElSe/PuRe/PupilEXT-equivalent validation platform and does not infer
  millimetres.
- Added the external benchmark truth boundary: iTrace can score outputs against
  user-supplied truth/comparator files, but detector agreement is not biological
  ground truth without reference annotations or a reference device.
- Added the guided empirical-session boundary: fixed-gaze, reading, and
  center/four-corner target prompts estimate session quality and held-out target
  residuals from derived records only, but do not constitute reference-device
  validation.
- Added method-foundation citations for the statistics/visualization layer:
  Savitzky-Golay differentiation, Butterworth filtering, AIC/BIC, the
  Kolmogorov-Smirnov statistic, robust quartile/IQR exploratory summaries,
  QQ/probability plotting, bootstrap model-selection winner stability,
  empirical-CDF residual diagnostics, Levenshtein edit distance, and vector-space cosine
  comparison. These support method names and formulas only; they do not add
  validation claims.
- Added a generated statistical-diagnostics payload
  (`output/figures/statistical_diagnostics.json`) so the publication statistics
  figure is display over tested Python outputs rather than plot-local
  calculation.
- Added a generated synthetic-to-empirical range-bridge payload
  (`output/figures/synthetic_empirical_range_bridge.json`) and figure. This
  introduces no new device-accuracy claim: it maps the N=1 pilot onto existing
  synthetic-domain, idealized landmark-noise, and descriptive-statistics
  evidence using the already-cited model-comparison, bootstrap, CDF-diagnostic,
  and data-quality methods.
- Added a generated statistical interpretation ledger
  (`output/figures/statistical_interpretation_ledger.json`) and figure. This
  introduces no new method claim; it maps already-cited robust summaries,
  AIC/AICc/BIC/KS diagnostics, bootstrap intervals, CDF residual checks,
  scanpath encodings, idealized noise sweeps, and the N=1 range bridge to their
  estimands and explicit non-claims.

## Rejected or deferred claims

- No claim that iTrace is device-validated.
- No claim that the 3-D closed-loop residual is real webcam gaze accuracy.
- No universal "2-5 degree" or "~0.1 mm" webcam-performance constant in the
  manuscript limitations; those values are protocol-dependent and belong in a
  future dataset/reference-device comparison.
- No claim that agreement with REMoDNaV, pymovements, or another detector would
  constitute biological ground truth without reference annotations or a
  reference eye tracker.
- No claim that prompted screen targets alone establish real eye-tracker
  accuracy across devices, participants, lighting, or head pose.
