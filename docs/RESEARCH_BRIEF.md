# Open-Source Webcam Eye Tracking: Saccades, Gaze Dynamics, Pupil Diameter & Live Dashboards

## Executive Summary

A rich and rapidly maturing open-source ecosystem now supports all three target signals — saccade direction and dynamics, gaze trajectory inference, and pupil diameter estimation — using nothing but a standard consumer webcam. The landscape divides naturally into four interoperating layers: (1) **low-level capture and landmark detection** (OpenCV, MediaPipe), (2) **mid-level event classification** (REMoDNaV, pymovements, gazeclassifier), (3) **pupillometry pipelines** (PupilEXT, jeelizPupillometry, PupEyes, pypillometry), and (4) **live dashboard and reporting frameworks** (Streamlit + streamlit-webrtc, Plotly Dash, Panel). By composing tools from these layers, research teams can construct a fully open, reproducible eye-tracking platform that rivals commercial systems at a fraction of the cost, though important accuracy constraints on consumer hardware must be understood.

***

## Part I: Foundations — Webcam Capture and Facial Landmark Detection

### OpenCV as the Capture Substrate

OpenCV (`opencv-python`) is the universal entry point for all Python-based webcam eye tracking. It provides `cv2.VideoCapture`, which abstracts hardware access for USB and built-in webcams, and offers a comprehensive suite of classical image processing functions including Canny edge detection, Hough transforms, and thresholding — all useful for iris segmentation and pupil boundary detection. OpenCV is available under the Apache 2.0 license, runs cross-platform, and is a dependency for virtually every Python-based eye tracker described below. On consumer hardware, frame capture typically runs at 30 fps (720p or 1080p), which is sufficient for fixation and coarse saccade detection but falls short of the millisecond precision achievable with dedicated IR cameras.[^1][^2]

### MediaPipe Face Mesh: 468 Landmark Real-Time Tracking

Google's MediaPipe Face Mesh provides 468 three-dimensional facial landmarks, including dedicated iris keypoints (indices for left iris: 474–478; right iris: 469–473). The iris detection module estimates the center and approximate radius of each iris in real time without any calibration, enabling immediate inference of relative pupil/iris position, gaze direction (left/right/center classification), and blink detection. MediaPipe runs efficiently on CPU-only hardware and is available as `pip install mediapipe`. Several community implementations pair it with a Kalman filter to smooth the noisy landmark coordinates into coherent gaze trajectories, substantially reducing jitter in the estimated saccade onset times.[^3][^4][^2][^5][^1]

A key limitation of MediaPipe for precise pupillometry is that the iris radius estimate is a relative measure — it reflects iris size in pixels, which varies with head distance — not an absolute pupil diameter in millimeters. For true pupillometry, downstream normalization or an additional deep learning model is required (see Part III).[^2]

### Dlib: Active Shape Models for Pupil Center Estimation

The Dlib library provides a 68-point facial landmark detector trained with ensemble of regression trees. The GazeTracking library by Antoine Lamé uses Dlib's eye-region landmarks to define a region of interest (ROI), then applies gradient-based pupil detection within that ROI to locate the pupil center with sub-pixel accuracy. The library exposes methods such as `gaze.is_right()`, `gaze.is_left()`, `gaze.is_center()`, and `gaze.pupil_left_coords()` / `gaze.pupil_right_coords()`, making it the most immediately accessible Python library for coarse gaze direction inference. It requires Python 3.10+ and is installable via `pip install -e .`.[^6][^7]

***

## Part II: Gaze Direction and Saccade Inference

### L2CS-Net: Deep Learning Gaze Estimation from Webcam Frames

L2CS-Net (Looking 2 Class Simultaneously Network) is a CNN-based gaze estimator trained on the MPIIGaze and Gaze360 datasets that achieves 3.92° mean angular error on MPIIGaze and 10.41° on Gaze360. The network takes a cropped facial image and outputs pitch and yaw angles representing the 3D gaze direction. Unlike landmark-based approaches, L2CS-Net performs full appearance-based regression, making it robust to lighting variation and partial occlusion. Installation is straightforward via pip from the official GitHub repository (`ahmednull/l2cs-net`), and the library exposes a simple `Pipeline` object for per-frame inference.[^8][^9][^10]

The pitch/yaw output of L2CS-Net can be differentiated temporally to infer saccade onset, direction, and amplitude. Because it provides a continuous angular time series rather than a discrete gaze point, it integrates naturally with the event classification algorithms described below. Roboflow's Inference platform also wraps L2CS-Net behind an HTTP API for server-side deployment, enabling separation of the ML backend from the visualization frontend.[^11]

### EyeGestures: Webcam-Native Gaze Tracking with Saccade Events

EyeGestures (`pip install eyeGestures`) is a Python library that combines MediaPipe facial landmarks with a calibration-based regression model to predict on-screen gaze coordinates. The V3 engine exposes a `gestures.step(frame, calibrate, width, height)` call that returns an `event` object containing `event.point` (gaze coordinates), `event.fixation` (fixation status), and `event.saccadess` (saccade list for the current frame). This makes EyeGestures one of the most directly usable libraries for saccade direction logging in a live pipeline without additional post-processing. The library is available in both Python and JavaScript variants, supporting desktop and browser deployment respectively.[^12][^13]

A calibration phase of approximately 20 seconds is required before prediction begins, during which the model learns the mapping from eye landmarks to screen coordinates. After calibration, gaze point accuracy on consumer hardware is typically 2–4° RMS, which is sufficient for coarse saccade direction classification (e.g., leftward vs. rightward vs. upward) but not for fine-grained amplitude measurements below 3–5°.[^14]

### WebGazer.js: Browser-Based Self-Calibrating Gaze Tracker

WebGazer.js is a JavaScript library that performs webcam-based gaze estimation entirely in the browser, requiring no server-side processing. It self-calibrates by observing mouse clicks and cursor movements to train a mapping between eye appearance features and screen positions. WebGazer has been open source since 2016 and remains the canonical tool for web-based UX studies. As of February 2026, the library is fully functional but is no longer actively maintained, with community support continuing via GitHub Issues. For research requiring saccade direction detection in web contexts, WebGazer outputs gaze coordinates at approximately 30 fps, and temporal differencing can be applied post-hoc to identify saccade-like ballistic movements.[^15][^16]

### REMoDNaV: Robust Saccade/Fixation/Pursuit Classification

REMoDNaV (Robust Eye Movement Detection for Natural Viewing) is a velocity-based event classification algorithm that extends the adaptive Nyström & Holmqvist (2010) algorithm. It classifies eye movement traces into four event types: **saccades**, **post-saccadic oscillations (PSO)**, **fixations**, and **smooth pursuit**. Unlike simpler threshold algorithms, REMoDNaV adapts its noise model to temporally varying signal quality, making it particularly robust for webcam data where lighting and head position fluctuate.[^17][^18]

The algorithm is available as `pip install remodnav`, takes tab-separated x/y coordinate files as input, and writes annotated event files as output. It was validated on three public datasets including manually annotated data for static images, moving dots, and short video clips. REMoDNaV is especially appropriate when gaze coordinates have been obtained from a preceding pipeline (e.g., L2CS-Net or EyeGestures) and need classification into oculomotor event categories prior to statistical analysis. The algorithm outperforms fixed-threshold I-VT and I-DT methods on data with variable noise, which is the norm for webcam-based systems.[^19][^20][^18][^17]

### pymovements: Full Eye-Movement Processing Pipeline

pymovements is a well-maintained Python package (installable via `pip install pymovements` or `conda install pymovements`) that provides an end-to-end preprocessing and event detection pipeline for eye-tracking data. Key capabilities include:[^21][^22]

- Parsing eye tracker data files from multiple vendor formats
- Transforming pixel coordinates to degrees of visual angle (`pix2deg()`)
- Computing instantaneous velocity (`pos2vel()`) using Savitzky-Golay smoothing
- Detecting fixations with I-VT and I-DT algorithms (`detect('ivt')`)
- Detecting saccades and microsaccades (`detect('microsaccades')`)
- Computing event properties including saccade amplitude, direction, duration, and peak velocity[^23][^24][^25]

pymovements stores data internally as polars DataFrames for high performance and integrates with the broader Python data science stack. When used downstream of a webcam gaze pipeline, it provides the richest suite of saccade metrics including the main sequence relationship between amplitude and peak velocity, which is a hallmark of normal oculomotor function.[^25][^21]

### gazeclassifier: Lightweight Saccade/Fixation Discriminator

For applications requiring only binary classification of gaze sequences (saccade vs. fixation vs. unknown), the `gazeclassifier` package from the Infant Cognition Laboratory at the University of Tampere provides a pretrained classifier accessible via `gc.predict(gaze_points)`. It accepts lists of (x, y) gaze point arrays and returns categorical labels. While simpler than REMoDNaV or pymovements, its minimal interface makes it useful for real-time streaming applications where low-latency classification is prioritized over detailed event metrics.[^26]

***

## Part III: Pupil Diameter Estimation from Webcam

### The Webcam Pupillometry Problem

Estimating absolute pupil diameter in millimeters from a standard webcam is significantly harder than tracking gaze direction. Commercial IR-based systems (Tobii, EyeLink) rely on the corneal reflection and operate under controlled illumination conditions, achieving submillimeter precision. Consumer webcams operate in visible light at variable distances and angles, introducing substantial noise. Several open-source approaches have been developed to address this gap, spanning classical image processing, deep learning, and hybrid methods.[^27][^28][^29][^30]

### PupilEXT (Open-PupilEXT): High-Resolution Offline Pupillometry

PupilEXT is a C++-based open-source platform developed for high-resolution pupillometry in vision research. It supports both real-time webcam streams and offline video processing, and implements six pupil detection algorithms including PuRe, PuReST, ElSe, ExCuSe, Starburst, and Swirski. A Python interface (`PyPupilEXT`) enables integration with Python pipelines. PupilEXT is the most rigorously validated open-source pupillometry tool and is particularly suited for offline analysis where frame rate constraints are relaxed. The software outputs pupil diameter time series, confidence values, and quality metrics.[^31][^32][^27]

### jeelizPupillometry: Real-Time WebGL Pupillometry

jeelizPupillometry is a JavaScript/WebGL library that runs entirely in the browser, using a 4K webcam feed to detect and measure the radius of both pupils in real time. The pipeline includes head detection via a deep neural network, iris segmentation via Circle Hough Transform, and pupil segmentation via a custom ray-tracing algorithm — all GPU-accelerated through WebGL. The output is a relative pupil radius (ratio of pupil radius to iris radius), which normalizes out head distance effects. This library enables web-based pupillometry studies without any server infrastructure and is the primary option for browser-deployed real-time dashboards requiring pupil data.[^33][^34]

### PupilSense / EyeDentify: Deep Learning Webcam Pupil Diameter

EyeDentify, developed at DFKI and RPTU Kaiserslautern-Landau, provides the first large-scale publicly available dataset (212,073 images from 51 participants) specifically designed for pupil diameter estimation from standard webcam images, with ground-truth measurements from a Tobii eye-tracker. Associated ResNet-based models achieve validation MAE of 0.0837 mm on the left eye using ResNet-18, with test MAE of 0.1340 mm. The EyeDentify dataset enables researchers to train or fine-tune models for webcam-based absolute pupil diameter estimation, moving beyond the relative-radius limitation of classical approaches.[^29][^35]

PupilSense, from Carnegie Mellon University, builds on EyeDentify and provides a complete application for webcam-based pupil diameter estimation, including class activation maps, graphs of predicted left and right pupil diameters, and eye aspect ratios during blinks. It is published on GitHub and supports batch video processing as well as frame-by-frame analysis.[^28]

### MEYE: Translational Real-Time Pupillometry Web App

MEYE (Measuring Eye) is a deep learning-based web application for real-time pupil size measurement that is cross-species compatible, having been validated on both humans and mice. MEYE uses a convolutional neural network trained on approximately 12,000 static grayscale eye images and runs in any standard web browser via TensorFlow.js. The web app can process both pre-recorded videos and live webcam streams. Comparative evaluations showed MEYE performing at equivalent accuracy to the EyeLink 1000 Plus and with higher frame rate than DeepLabCut for pupillometry. Source code is freely available under an open-source license.[^36][^37]

### PupEyes: Interactive Python Pupillometry Pipeline

PupEyes (published in 2026 in a peer-reviewed journal) is a Python package (`pip install pupeyes`) for preprocessing and visualizing pupil size and fixation data. It provides:[^38][^39]

- A principled, transparent pupil preprocessing pipeline including artifact removal, blink interpolation, and baseline correction[^40]
- Interactive Plotly Dash visualizations: a **Pupil Viewer** for examining preprocessing steps, a **Fixation Viewer** for visualizing fixation patterns, and an **AOI Drawing Tool** for defining areas of interest[^41][^42]
- Native compatibility with pandas DataFrames, enabling seamless integration with the broader Python analytics stack[^40]
- Support for EyeLink and Tobii data formats as well as any generic CSV input conforming to minimal formatting standards[^38]

PupEyes is particularly well-suited as the analysis layer in a research pipeline: raw pupil time series can be ingested from any of the capture libraries, preprocessed through PupEyes' pipeline, and explored interactively through its built-in Plotly Dash dashboards.[^41]

### pypillometry: Statistical Pupillometry and Modeling

pypillometry (published in the Journal of Open Source Software) is a Python package for comprehensive pupillometric analyses including preprocessing, blink handling, event-related pupil dilation modeling, and signal decomposition. It provides plotting functions, baseline correction routines, and a general linear deconvolution model for estimating the canonical pupil impulse response to cognitive events — directly relevant for applications coupling eye tracking with cognitive experiments. It is installable from PyPI and is the standard choice for hypothesis-driven statistical analyses of pupil time series.[^43][^44]

### rtPupilPhase: Real-Time Pupil Phase Detection

rtPupilPhase (NIH/NIMH, 2024) is the first open-source tool specifically designed for **online detection of trends in pupil size fluctuation** (dilation phase, constriction phase, peak, trough) in real time. Validated on human, rodent, and monkey pupil data, it enables closed-loop experimental designs where stimuli are presented at specific phases of the pupil cycle. The implementation uses PsychoPy for stimulus delivery and connects to an EyeLink eye tracker for high-precision pupil input; the statistical detection logic is portable to webcam-based input streams.[^45][^46][^47][^48]

***

## Part IV: Comprehensive Software Comparison

### Core Capture and Gaze Libraries

| Library | Language | Saccade Detection | Gaze Direction | Pupil Diameter | Live Stream | License |
|---------|----------|-------------------|----------------|----------------|-------------|---------|
| **GazeTracking** (antoinelame)[^6][^7] | Python | Indirect (coordinate diff.) | Left/Center/Right | Pupil center coords | Yes (OpenCV) | MIT |
| **EyeGestures** v3[^12][^13] | Python/JS | Direct (`event.saccadess`) | Screen coordinates | No | Yes | AGPL-3.0 |
| **L2CS-Net**[^9][^10] | Python (PyTorch) | Via pitch/yaw time series | Pitch + yaw angles | No | Yes | MIT |
| **WebGazer.js**[^15][^16] | JavaScript | Via coordinate differencing | Screen gaze point | No | Yes (browser) | Apache 2.0 |
| **MediaPipe Face Mesh**[^3][^1] | Python/C++ | Via landmark velocity | Iris-relative position | Relative iris radius | Yes | Apache 2.0 |
| **PyGaze**[^49][^50] | Python | Depends on backend | Backend-dependent | Backend-dependent | Partial | GPL-3.0 |
| **OpenIris**[^51][^52][^53] | C# | Yes (3D incl. torsion) | 3D eye vector | Yes | Yes (500+ Hz) | AGPL-3.0 |

### Event Classification Algorithms

| Algorithm | Library | Events Detected | Handles Dynamic Stimuli | Speed | Language |
|-----------|---------|-----------------|------------------------|-------|---------|
| **REMoDNaV**[^17][^18] | `remodnav` | Saccade, PSO, fixation, smooth pursuit | Yes | Offline batch | Python |
| **I-VT** | `pymovements`[^23][^25] | Saccade, fixation | Partial | Fast | Python |
| **I-DT** | `pymovements`[^23][^25] | Fixation | No | Fast | Python |
| **Microsaccades** | `pymovements`[^22] | Microsaccades | N/A | Fast | Python |
| **gazeclassifier**[^26] | `gazeclassifier` | Saccade/fixation/unknown | No | Real-time capable | Python |
| **Neural network** (Hafed Lab)[^54] | Custom | Saccades + blinks + PSO | Yes | Offline | Python/MATLAB |

### Pupillometry Libraries

| Library | Input | Diameter (mm) | Real-Time | Visualization | Platform |
|---------|-------|---------------|-----------|---------------|---------|
| **PupilEXT**[^27][^31] | Webcam/video | Yes (6 algorithms) | Yes | Yes | Desktop (C++) |
| **jeelizPupillometry**[^33][^34] | Webcam | Relative ratio | Yes | WebGL canvas | Browser (JS) |
| **MEYE**[^36][^37] | Webcam/video | Yes (DL) | Yes | Web app | Browser |
| **PupilSense/EyeDentify**[^28][^29] | Webcam | Yes (ResNet) | No (batch) | Graphs + CAM | Python |
| **PupEyes**[^38][^41][^42] | CSV (any source) | Post-hoc | No | Plotly Dash | Python |
| **pypillometry**[^43][^44] | CSV (any source) | Post-hoc | No | Matplotlib | Python |
| **rtPupilPhase**[^45][^46] | EyeLink/PsychoPy | Phase events | Yes | Basic | Python |

***

## Part V: Live Dashboard and Report Visualization Architecture

### Streamlit + streamlit-webrtc: Rapid Real-Time Dashboard Prototyping

Streamlit is widely regarded as the fastest path from Python code to interactive web application, requiring no frontend development experience. The `streamlit-webrtc` library extends Streamlit to handle real-time video streams from webcams via WebRTC, enabling browser-side webcam access with server-side Python processing. A minimal streaming eye-tracking dashboard requires only:[^55][^56][^57][^58]

```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # ... run MediaPipe / L2CS-Net / EyeGestures here ...
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(key="eye-tracker", video_frame_callback=video_frame_callback)
st.plotly_chart(live_gaze_figure)  # updated via st.session_state
```

Streamlit is best for rapid prototyping and small-scale deployments; its primary limitation is constrained customization relative to full web frameworks. For production dashboards serving many simultaneous users, Plotly Dash or Panel offer greater scalability and UI flexibility.[^59]

### Plotly Dash: Production-Grade Interactive Dashboards

Plotly Dash is the most capable option for building fully interactive, production-ready eye-tracking dashboards in Python. Dash applications are React-based under the hood, enabling rich client-side interactivity including real-time graph updates via `dcc.Interval`, synchronized multi-plot brushing, and custom callback logic. PupEyes uses Plotly Dash as its visualization backend for its Pupil Viewer, Fixation Viewer, and AOI Drawing Tool components, demonstrating a well-designed architecture for this use case.[^60][^42][^55][^41]

A comprehensive eye-tracking dashboard built on Dash would include:
- **Raw signal panel**: Real-time scrolling time series of pupil diameter (left/right), gaze x/y coordinates, and blink flags
- **Saccade metrics panel**: Bar charts of saccade direction distribution (polar or Cartesian histograms), scatter plots of amplitude vs. peak velocity (main sequence), and saccade rate over time
- **Gaze heatmap**: Kernel density estimate of fixation distribution over a stimulus image
- **Pupil event panel**: Dilation phases, constriction events, and event-locked pupil responses
- **Summary statistics**: Fixation count, mean saccade amplitude, mean pupil diameter, and blink rate reported as numeric cards

### Panel and Bokeh: Streaming Data Visualization

Panel (from HoloViz) and Bokeh support streaming data via `ColumnDataSource.stream()` and `pn.state.add_periodic_callback()`, making them well-suited for live physiological data visualization with low-latency updates. Bokeh's streaming architecture is particularly efficient for updating only new data points rather than re-rendering the entire plot, which matters for long-duration recording sessions. Panel additionally provides a widget layer compatible with both Plotly and Bokeh rendering backends, enabling mixed visualization libraries in a single dashboard.[^61][^60]

***

## Part VI: Recommended Architecture for a Complete System

### Layered Pipeline Design

A comprehensive open-source eye-tracking system with live dashboard and statistical reports is best organized into four processing layers:

**Layer 1 — Acquisition**: OpenCV `VideoCapture` at 30–60 fps, with MediaPipe Face Mesh for landmark extraction. This layer produces per-frame iris/pupil center coordinates, relative iris radius, and head pose (pitch, yaw, roll).

**Layer 2 — Inference**: 
- *Gaze direction*: L2CS-Net or EyeGestures (calibration-based) for screen-space gaze coordinates and pitch/yaw angles
- *Pupil diameter*: jeelizPupillometry (browser, real-time relative) or EyeDentify/PupilSense model (offline absolute mm)
- *Saccade classification*: REMoDNaV or pymovements I-VT applied to the gaze coordinate time series

**Layer 3 — Live Dashboard**: Streamlit + streamlit-webrtc for annotated video display; Plotly Dash for synchronized real-time charts of all signal streams

**Layer 4 — Reports & Statistics**: PupEyes for interactive pupil preprocessing visualization; pypillometry for statistical modeling; pymovements for saccade metric computation and export; matplotlib/Seaborn for publication-quality figures

### Data Flow and Synchronization

All four layers should share a common timestamp (Unix epoch or monotonic clock in milliseconds) to enable post-hoc synchronization of signals collected at different sampling rates. A lightweight message queue (Python `queue.Queue` for local applications, or ZMQ for distributed setups) effectively decouples the acquisition loop (which must be fast and unblocked) from the dashboard rendering loop (which can tolerate higher latency). Gaze event annotations from REMoDNaV can be written to an HDF5 or Parquet file in near-real-time, which PupEyes and pypillometry can then read for live updating analysis.[^3]

### Accuracy Expectations and Limitations

Consumer webcam-based eye tracking operates under significant constraints compared to dedicated IR systems. Angular accuracy for gaze direction using calibrated approaches (EyeGestures, WebGazer) is typically 2–5° RMS, sufficient for detecting saccades >5° in amplitude but not microsaccades (<1°). Pupil diameter estimation from visible-light webcams introduces additional error due to lighting-dependent pupil appearance changes, perspective distortion, and partial occlusion by eyelashes. EyeDentify models achieve test MAE of approximately 0.13 mm using ResNet-18, which is adequate for pharmacological pupillometry and cognitive load studies but falls short of the <0.05 mm precision of dedicated pupillometers. These limitations should inform study design and data interpretation, particularly when comparing results to literature using IR-based systems.[^62][^63][^29]

***

## Part VII: Active Research Frontiers and Notable Tools

### OpenIris: Modular C# Framework for High-Speed Research

OpenIris (UC Berkeley Ocular-Motor Lab) is a C# framework designed for customizable, high-speed eye tracking with binocular pupil tracking pipelines achieving over 500 Hz. While not primarily designed for consumer webcams (it targets research cameras), its modular plugin architecture and remote network control interface make it relevant for hybrid setups where high-speed IR cameras supplement a webcam-based stream. OpenIris tracks pupil, corneal reflections, and torsion in 3D, and is openly licensed under AGPL-3.0.[^51][^52][^53]

### EyeLoop: Closed-Loop Neuroscience Experiments

EyeLoop (Arvin, 2020; Danish Research Institute of Translational Neuroscience) was a Python 3 eye tracker explicitly designed for closed-loop neuroscience experiments on consumer hardware, claiming >1000 Hz performance without dedicated processing units. The repository has been archived (read-only as of April 2024), but its architecture and source code remain a valuable reference for building modular eye-tracking engines with experiment control integration. It demonstrated that Python-based systems can achieve performance previously associated only with compiled-language solutions.[^64][^65]

### GazeParser: Multiplatform Python Library with PsychoPy Integration

GazeParser is an open-source Python library providing a video-based eyetracker alongside libraries for data recording and analysis, integrating with PsychoPy and VisionEgg. A validation study showed sampling interval errors under 1 ms and spatial accuracy of 0.7°–1.2°, with saccade latency and amplitude in gap/overlap tasks matching those of a commercial eyetracker. GazeParser provides a complete solution for psychology researchers already using PsychoPy who wish to add eye tracking without purchasing commercial hardware.[^66][^67]

### Saccade Direction Encoding: N-Gram Analysis

For applications specifically interested in saccade direction patterns (e.g., Active Inference models of oculomotor behavior, cognitive security studies of gaze-based deception or attention), a directional encoding scheme can be applied to REMoDNaV or pymovements output. One published implementation from the University of Stuttgart encodes each saccade as a direction character (U = up, D = down, R = right, L = left, with capitalization indicating long vs. short saccades) and computes n-gram distributions over saccade sequences. This enables statistical characterization of habitual scan patterns and detection of anomalies in gaze behavior, directly relevant to cognitive security applications.[^68]

***

## Part VIII: Recommended Stack by Use Case

| Use Case | Recommended Stack |
|----------|-----------------|
| **Rapid research prototype** | MediaPipe + EyeGestures + Streamlit/webrtc + Plotly |
| **Saccade direction classification** | L2CS-Net + REMoDNaV + pymovements + Plotly Dash |
| **Real-time pupillometry (browser)** | jeelizPupillometry or MEYE web app |
| **Deep learning pupil diameter (Python)** | EyeDentify/PupilSense + PupEyes + pypillometry |
| **High-speed research (dedicated camera)** | OpenIris (C#) with camera |
| **Closed-loop cognitive experiments** | GazeParser + PsychoPy + rtPupilPhase |
| **Full offline analysis pipeline** | pymovements + PupEyes + pypillometry + REMoDNaV |
| **Browser-deployed study** | WebGazer.js (gaze) + jeelizPupillometry (pupil) |

***

## Conclusion

The open-source ecosystem for webcam-based eye tracking is now mature enough to support rigorous research on saccade dynamics, gaze trajectory, and pupil diameter from consumer hardware alone. The most effective approach integrates MediaPipe or L2CS-Net for per-frame signal extraction, EyeGestures or REMoDNaV/pymovements for event classification, MEYE or PupEyes for pupillometry, and Streamlit+webrtc or Plotly Dash for live dashboards. Each layer is independently replaceable, allowing iterative improvement as the field advances. Critical accuracy caveats must be acknowledged: webcam-based systems operate at 2–5° gaze accuracy and ~0.1 mm pupil diameter precision, which constrains but does not preclude meaningful research on fixation patterns, coarse saccade dynamics, cognitive load, and attention.[^19][^28][^29]

---

## References

1. [Part 1. Eye Tracking with Mediapipe and OpenCV (In ...](https://kh-monib.medium.com/title-gaze-tracking-with-opencv-and-mediapipe-318ac0c9c2c3) - In short these are the topics that will be covered in the article

2. [How to calculate gaze position? - python - Stack Overflow](https://stackoverflow.com/questions/74070699/how-to-calculate-gaze-position) - I am trying to code out an eye tracker using Python. I am using the Face Mesh solution from the Medi...

3. [Real-Time Eye & Face Tracking with Python - YouTube](https://www.youtube.com/watch?v=UgC2GggTks0) - ... Python-Gaze-Face-Tracker, a simplified tool for real-time iris tracking, blink detection, and fa...

4. [EsraaMeslam/-Real-Time-Eye-Tracking-and-Position-Estimation ...](https://github.com/EsraaMeslam/-Real-Time-Eye-Tracking-and-Position-Estimation-Using-OpenCV-and-MediaPipe-) - This project implements a real-time eye-tracking system using OpenCV and MediaPipe. The code tracks ...

5. [[PDF] MediaPipe Iris and Kalman Filter for Robust Eye Gaze Tracking](https://www.atlantis-press.com/article/126011300.pdf) - Here we present a simple but powerful approach using MediaPipe Iris in combination with Kalman Filte...

6. [antoinelame/GazeTracking: 👀 Eye Tracking library easily ...](https://github.com/antoinelame/GazeTracking) - 👀 Eye Tracking library easily implementable to your projects - antoinelame/GazeTracking

7. [GazeTracking/README.md at master · antoinelame/GazeTracking](https://github.com/antoinelame/GazeTracking/blob/master/README.md) - 👀 Eye Tracking library easily implementable to your projects - antoinelame/GazeTracking

8. [Fine-Grained Gaze Estimation in Unconstrained Environments - arXiv](https://arxiv.org/abs/2203.03339) - In this paper, we propose a robust CNN-based model for predicting gaze in unconstrained settings. We...

9. [Ahmednull/L2CS-Net: The official PyTorch implementation ... - GitHub](https://github.com/ahmednull/l2cs-net) - Gaze Detection and Eye Tracking: A How-To Guide: Use L2CS-Net through a HTTP interface with the open...

10. [GitHub - Ahmednull/L2CS-Net: The official PyTorch implementation of L2CS-Net for gaze estimation and tracking](https://github.com/Ahmednull/L2CS-Net) - The official PyTorch implementation of L2CS-Net for gaze estimation and tracking - Ahmednull/L2CS-Ne...

11. [Gaze - Roboflow Inference](https://inference.roboflow.com/foundation/gaze/) - L2CS-Net is a gaze estimation model. You can detect the direction in which someone is looking using ...

12. [GitHub - NativeSensors/EyeGestures: gaze tracking software](https://github.com/NativeSensors/EyeGestures/tree/main) - gaze tracking software. Contribute to NativeSensors/EyeGestures development by creating an account o...

13. [EyeGestures - Open Source Gaze Tracking Made Accessible](https://eyegestures.com) - Our gaze tracker is focused around openness and lowering entry point for building gaze driven interf...

14. [Real-Time Webcam Eye-Tracking [Open-Source] - Reddit](https://www.reddit.com/r/computervision/comments/1j0xeug/realtime_webcam_eyetracking_opensource/) - Top eye tracking apps available. Gaze tracking techniques in Python. Face tracking options for VR. F...

15. [WebGazer.js: Scalable Webcam EyeTracking Using User Interactions](https://github.com/brownhci/webgazer) - WebGazer.js is an eye tracking library that uses common webcams to infer the eye-gaze locations of w...

16. [WebGazer.js: Democratizing Webcam Eye Tracking on the Browser](https://webgazer.cs.brown.edu) - WebGazer.js is an eye tracking library that uses common webcams to infer the eye-gaze locations of w...

17. [remodnav/README.md at master · psychoinformatics-de/remodnav](https://github.com/psychoinformatics-de/remodnav/blob/master/README.md) - Robust Eye Movement Detection for Natural Viewing. Contribute to psychoinformatics-de/remodnav devel...

18. [REMoDNaV: robust eye-movement classification for dynamic ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880959/) - The algorithm is cross-platform compatible, implemented using the Python programming language, and r...

19. [Review and Evaluation of Eye Movement Event Detection Algorithms](https://pmc.ncbi.nlm.nih.gov/articles/PMC9699548/) - The velocity threshold algorithm is another algorithm and the foundation for an automated/objective ...

20. [Identification of fixations and saccades in eye-tracking data using ...](https://arxiv.org/html/2512.23926v1) - We find that a velocity threshold achieves the highest baseline accuracy (90-93%) across both free-v...

21. [pymovements: A Python Package for Eye Movement Data Processing](https://arxiv.org/abs/2304.09859) - We introduce pymovements: a Python package for analyzing eye-tracking data that follows best practic...

22. [GitHub - aeye-lab/pymovements: A python package for processing eye movement data](https://github.com/aeye-lab/pymovements) - A python package for processing eye movement data. Contribute to aeye-lab/pymovements development by...

23. [Welcome to the pymovements documentation! — pymovements ...](https://pymovements.readthedocs.io) - pymovements is an open-source python package for processing eye movement data. It provides a simple ...

24. [pymovements: A Python Package for Eye Movement Data Processing](https://ar5iv.labs.arxiv.org/html/2304.09859) - We introduce pymovements: a Python package for analyzing eye-tracking data that follows best practic...

25. [Detecting Gaze Events#](https://pymovements.readthedocs.io/en/v0.21.1/tutorials/detecting-events.html)

26. [gazeclassifier](https://pypi.org/project/gazeclassifier/) - Decides whether given gaze points represent a saccade, fixation, or some unknown pattern

27. [PupilEXT: Flexible Open-Source Platform for High-Resolution ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8249868/) - If cameras are connected to PupilEXT, a real-time pupil measurement with one of the six pupil detect...

28. [A Novel Application for Webcam-Based Pupil Diameter Estimation](https://arxiv.org/html/2407.11204v2) - This paper presents a novel application that enables pupil diameter estimation using standard webcam...

29. [EyeDentify: A Dataset for Pupil Diameter Estimation ... - GitHub Pages](https://vijulshah.github.io/eyedentify/) - In this work, we introduce EyeDentify, a dataset specifically designed for pupil diameter estimation...

30. [[Literature Review] EyeDentify: A Dataset for Pupil Diameter ...](https://www.themoonlight.io/en/review/eyedentify-a-dataset-for-pupil-diameter-estimation-based-on-webcam-images) - In summary, the EyeDentify dataset uniquely addresses the limitations of current pupil diameter esti...

31. [openPupil/Open-PupilEXT - GitHub](https://github.com/openPupil/Open-PupilEXT) - PupilEXT can record eye images using a stereo camera system or a single camera to measure the pupil ...

32. [Webcam pupillometry : r/matlab - Reddit](https://www.reddit.com/r/matlab/comments/rttscd/webcam_pupillometry/) - My understanding of pupillometry is such that you must have a camera capable of tracking the pupil+s...

33. [GitHub - jeeliz/jeelizPupillometry: Real time pupillometry in the web ...](https://github.com/jeeliz/jeelizPupillometry) - This JavaScript library detects, tracks and measures the radius of the 2 pupils of the eyes of the u...

34. [https://github.com/jeeliz/jeelizpupillometry](https://awesome.ecosyste.ms/projects/github.com%2Fjeeliz%2Fjeelizpupillometry) - Real-time pupillometry in the web browser using a 4K webcam video feed processed by this WebGL/Javas...

35. [A Dataset for Pupil Diameter Estimation based on Webcam Images](https://arxiv.org/html/2407.11204v1) - In this work, we introduce EyeDentify, a dataset specifically designed for pupil diameter estimation...

36. [MEYE: Web App for Translational and Real-Time Pupillometry - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8489024/) - Here we show an open-source web app that, through deep learning, can perform real-time pupil size me...

37. [Web app tracks pupil size in people, mice | The Transmitter](https://www.thetransmitter.org/spectrum/web-app-tracks-pupil-size-in-people-mice/) - A new open-source web app can measure changes in pupil size in both people and mice as accurately as...

38. [PupEyes: An interactive Python library for eye movement data ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12769653/) - We present PupEyes, an open-source Python package for preprocessing and visualizing pupil size and f...

39. [PupEyes: An interactive Python library for eye movement data processing - PubMed](https://pubmed.ncbi.nlm.nih.gov/41491873/) - We present PupEyes, an open-source Python package for preprocessing and visualizing pupil size and f...

40. [PupEyes: An Interactive Python Library for Eye Movement Data ...](https://sciety.org/articles/activity/10.31234/osf.io/h95ma_v2) - We present PupEyes, an open-source Python package for preprocessing and visualizing pupil size and f...

41. [PupEyes: Your Buddy for Pupil Size and Eye Movement Data ...](https://pupeyes.readthedocs.io) - PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil si...

42. [PupEyes: Your Buddy for Pupil Size and Eye Movement Data Analysis](https://github.com/HanZhang-psych/pupeyes) - PupEyes is a Python package for preprocessing and visualizing eye movement data. It handles pupil si...

43. [pypillometry · PyPI](https://pypi.org/project/pypillometry/) - This package implements functions for the analysis of pupillometric data. Features include preproces...

44. [pypillometry: A Python package for pupillometric analyses](https://joss.theoj.org/papers/10.21105/joss.02348) - A Python package for pupillometric analyses. Journal of Open Source Software, 5(51), 2348, https://d...

45. [Cross-species real time detection of trends in pupil size fluctuation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10896349/) - We introduce rtPupilPhase – an open source software that automatically detects trends in pupil size ...

46. [Cross-species real-time detection of trends in pupil size fluctuation](https://scholarworks.sjsu.edu/faculty_rsca/5855/) - We introduce rtPupilPhase—an open-source software that automatically detects trends in pupil size in...

47. [rtPupilPhase/README.md at main · nimh-sfim/rtPupilPhase](https://github.com/nimh-sfim/rtPupilPhase/blob/main/README.md) - A repository for code and data associated with Kronemer et al., 2024 - nimh-sfim/rtPupilPhase

48. [nimh-sfim/rtPupilPhase: A repository for code and data ... - GitHub](https://github.com/nimh-sfim/rtPupilPhase) - An object of class StimulusDecider stores data and detects pupil phase events in real time and in si...

49. [python-pygaze - PyPI](https://pypi.org/project/python-pygaze/) - PyGaze is a Python package for easily creating gaze contingent experiments or other software (as wel...

50. [esdalmaijer/PyGaze](https://github.com/esdalmaijer/PyGaze) - an open-source, cross-platform toolbox for minimal-effort programming of eye tracking experiments - ...

51. [An Open Source Framework for Video-Based Eye-Tracking ...](https://pubmed.ncbi.nlm.nih.gov/38463977/) - We present OpenIris, an adaptable and user-friendly open-source framework for video-based eye-tracki...

52. [An Open Source Framework for Video-Based Eye-Tracking ... - bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.27.582401v1) - We present OpenIris, a user-friendly and adaptable open-source framework developed in C# for video-b...

53. [GitHub - ocular-motor-lab/OpenIris](https://github.com/ocular-motor-lab/OpenIris) - Contribute to ocular-motor-lab/OpenIris development by creating an account on GitHub.

54. [Open-source code for saccade detection algorithm available](https://hafedlab.org/2018/07/01/open-source-code-for-saccade-detection-algorithm-available/) - We have developed a novel, state-of-the-art algorithm for detecting saccades and microsaccades in ey...

55. [How to Build Interactive Dashboards in Python with Plotly Using ...](https://python.plainenglish.io/how-to-build-interactive-dashboards-in-python-with-plotly-using-streamlit-047979344d93) - ... Streamlit Interactive dashboards offer a powerful way to visualize, interpret, and share data in...

56. [Building a Web-Based Real-Time Computer Vision App ...](https://dev.to/whitphx/build-a-web-based-real-time-computer-vision-app-with-streamlit-57l2) - This article is based on an older version of the library and out-of-date. See this new tutorial...

57. [Developing Web-Based Real-Time Video/Audio Processing Apps Quickly with Streamlit](https://towardsdatascience.com/developing-web-based-real-time-video-audio-processing-apps-quickly-with-streamlit-7c7bcd0bc5a8/) - In this article, we will see how we can create browser-ready real-time video/audio processing apps w...

58. [Realtime Webcam Processing Using Streamlit and OpenCV](https://thiagoalves.ai/building-webcam-streaming-applications-with-streamlit-and-opencv/) - In this post, I’ll show you how to build streaming application using Streamlit and OpenCV.

59. [Streamlit vs Dash | Python Tools Comparison - Firebolt](https://www.firebolt.io/python-tools-comparison/streamlit-vs-dash) - Streamlit vs Dash - which is the best Python web tools for your project? Read our comprehensive comp...

60. [A Comprehensive Guide to Streamlit, Dash, and Bokeh](https://dev.to/sergiocolqueponce/modern-data-visualization-tools-a-comprehensive-guide-to-streamlit-dash-and-bokeh-7ed) - Building interactive dashboards and reports has never been easier. Let's explore three powerful Pyth...

61. [Data Visualization Features with Streamlit, Dash, and Panel. Part 2](https://sunscrapers.com/blog/streamlit-dash-panel-features-part-2/) - Streamlit, Dash, and Panel excel in creating highly interactive applications with features like real...

62. [Best gaze tracking tool (webcam-based, accurate with calibration ...](https://www.reddit.com/r/EyeTracking/comments/1kf18p0/best_gaze_tracking_tool_webcambased_accurate_with/) - We're (https://www.1gaze.com) releasing an eye tracker that works with a webcam and doesn't require ...

63. [Evaluating Calibration-free Webcam-based Eye Tracking for Gaze ...](https://dl.acm.org/doi/pdf/10.1145/3536221.3556580) - Thus, saccade detection relies on accurate fxation detection, which is common in low-frequency eye t...

64. [eyeloop/README.md at master · simonarvin/eyeloop](https://github.com/simonarvin/eyeloop/blob/master/README.md) - EyeLoop is a Python 3-based eye-tracker tailored specifically to dynamic, closed-loop experiments on...

65. [We open-sourced our Python-based eye-tracker for brain research ...](https://www.reddit.com/r/programming/comments/hllg7c/we_opensourced_our_pythonbased_eyetracker_for/) - We just published a new Python-based eye-tracker, EyeLoop. All code files are online, and the softwa...

66. [GazeParser: an open-source and multiplatform library for low-cost ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC3745611/) - Eye movement analysis is an effective method for research on visual perception and cognition. Howeve...

67. [GazeParser: an open-source and multiplatform library for ...](https://pubmed.ncbi.nlm.nih.gov/23239074/) - Eye movement analysis is an effective method for research on visual perception and cognition. Howeve...

68. [eye_movements_personality/featureExtraction/gaze_analysis.py at 0403f2ce555ea70b98f52d833c4e5200077403c9](https://git.hcics.simtech.uni-stuttgart.de/public-projects/eye_movements_personality/src/commit/0403f2ce555ea70b98f52d833c4e5200077403c9/featureExtraction/gaze_analysis.py) - eye_movements_personality - Official code for "Eye movements during everyday behavior predict person...

