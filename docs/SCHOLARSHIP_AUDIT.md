# iTrace Scholarship Audit

Checked on 2026-06-01 for the manuscript scholarship/PDF refresh.

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
| Browser gaze | Papoutsaki et al. 2016, WebGazer | <https://www.ijcai.org/Proceedings/16/Papers/522.pdf> | Canonical in-browser webcam gaze tracker already cited; retained as browser-gaze baseline. |
| Commodity-camera landmarks | Vakunov et al. 2020, MediaPipe Iris | <https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/> | Official source for real-time iris tracking/depth estimation; cited as a landmark source, not as validation evidence. |
| Appearance gaze | Abdelrahman et al. 2022, L2CS-Net | <https://arxiv.org/abs/2203.03339> | Representative open appearance-based pitch/yaw estimator evaluated on MPIIGaze and Gaze360. |
| Public gaze datasets | Krafka et al. 2016, GazeCapture; Zhang et al. 2017, MPIIGaze | <https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html>; <https://arxiv.org/abs/1711.09017> | Public real-frame substrate for future device/dataset validation. |
| Event processing | Dar et al. 2021, REMoDNaV; Krakowczyk et al. 2023, pymovements | <https://link.springer.com/article/10.3758/s13428-020-01428-x>; <https://arxiv.org/abs/2304.09859> | Event-classification and processing comparators for future benchmark harnesses. |
| Data quality | Jakobi et al. 2024 | <https://arxiv.org/abs/2404.00620> | Supports explicit quality-reporting language instead of unqualified validation claims. |
| Webcam pupil diameter | Shah et al. 2025, PupilSense/EyeDentify | <https://arxiv.org/abs/2407.11204> | Corrects the previous EyeDentify-only title to the current arXiv PupilSense application/dataset paper. |
| Pupil preprocessing | Zhang and Jonides 2026, PupEyes; Mittner 2020, pypillometry | <https://pmc.ncbi.nlm.nih.gov/articles/PMC12769653/>; <https://joss.theoj.org/papers/10.21105/joss.02348> | Places iTrace's pupil preprocessing in relation to current open pupil-analysis tools. |
| Pupil light confounds | Cai et al. 2024, Open-DPSM | <https://link.springer.com/article/10.3758/s13428-023-02292-1> | Supports the limitation that dynamic luminance/contrast confounds need explicit modelling. |
| Pupil phase | Kronemer et al. 2024, rtPupilPhase | <https://link.springer.com/article/10.3758/s13428-024-02374-8> | Retained as the causal pupil-phase reference. |

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

## Rejected or deferred claims

- No claim that iTrace is device-validated.
- No claim that the 3-D closed-loop residual is real webcam gaze accuracy.
- No universal "2-5 degree" or "~0.1 mm" webcam-performance constant in the
  manuscript limitations; those values are protocol-dependent and belong in a
  future dataset/reference-device comparison.
- No claim that agreement with REMoDNaV, pymovements, or another detector would
  constitute biological ground truth without reference annotations or a
  reference eye tracker.
