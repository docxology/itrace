"""iTrace - open-source webcam eye-movement analysis toolkit.

A pure NumPy/SciPy analysis core (gaze geometry, velocity, saccade/fixation/
microsaccade detection, main-sequence dynamics, pupillometry, scanpath
encoding) with an optional, thin webcam/MediaPipe capture shell. Importing this
package never requires any optional (hardware/dashboard) dependency.
"""

from __future__ import annotations

from . import (
    calibration,
    capture,
    config,
    dashboard,
    detection,
    encoding,
    eyemodel,
    geometry,
    io,
    mainsequence,
    pipeline,
    power,
    pupil,
    pupilphase,
    reporting,
    saccades,
    scene,
    stats,
    synthetic,
    validation,
    velocity,
)
from .pupilphase import Phase, PhaseDetector
from .types import (
    PSO,
    EventType,
    Fixation,
    GazeSample,
    GazeStream,
    Microsaccade,
    PupilSample,
    PupilStream,
    PupilUnit,
    Saccade,
    SessionReport,
    SmoothPursuit,
)
from .version import __version__

__all__ = [
    "PSO",
    "EventType",
    "Fixation",
    "GazeSample",
    "GazeStream",
    "Microsaccade",
    "Phase",
    "PhaseDetector",
    "PupilSample",
    "PupilStream",
    "PupilUnit",
    "Saccade",
    "SessionReport",
    "SmoothPursuit",
    "__version__",
    "calibration",
    "capture",
    "config",
    "dashboard",
    "detection",
    "encoding",
    "eyemodel",
    "geometry",
    "io",
    "mainsequence",
    "pipeline",
    "power",
    "pupil",
    "pupilphase",
    "reporting",
    "saccades",
    "scene",
    "stats",
    "synthetic",
    "validation",
    "velocity",
]
