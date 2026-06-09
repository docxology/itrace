"""Core data model for iTrace.

All public signal containers live here. The conventions are fixed and enforced
so that no ambiguous quantity ever leaks between layers:

Coordinate & unit conventions
------------------------------
* **Pixels**: screen coordinates, origin top-left, ``x`` increasing rightward,
  ``y`` increasing *downward* (the standard image/screen convention).
* **Degrees of visual angle (dva)**: produced by :mod:`itrace.geometry`. Gaze
  *direction* angles use the mathematical convention where ``0deg`` points right
  and ``+90deg`` points *up* (screen-down is negated on the way in), so a
  rightward saccade is ``~0deg`` and an upward saccade is ``~90deg``.
* **Time**: seconds (float). Either a monotonic clock or Unix epoch; only
  differences are used.
* **Pupil size**: always carries an explicit :class:`PupilUnit` — never a bare
  number whose unit a caller has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]
NumericReportDict: TypeAlias = dict[str, float]
ReportPayload: TypeAlias = dict[str, object]


class EventType(str, Enum):
    """Oculomotor event categories (REMoDNaV-compatible)."""

    FIXATION = "fixation"
    SACCADE = "saccade"
    PSO = "pso"  # post-saccadic oscillation
    SMOOTH_PURSUIT = "smooth_pursuit"
    BLINK = "blink"


class PupilUnit(str, Enum):
    """Unit attached to every pupil measurement."""

    MM = "mm"
    RELATIVE = "relative"  # pupil-radius / iris-radius ratio (head-distance free)
    PIXELS = "pixels"


@dataclass(frozen=True, slots=True)
class GazeSample:
    """A single timestamped gaze position in degrees of visual angle."""

    t: float
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class PupilSample:
    """A single timestamped pupil-size measurement with explicit unit."""

    t: float
    size: float
    unit: PupilUnit = PupilUnit.RELATIVE


@dataclass(frozen=True, slots=True)
class EyeGazeDiagnostic:
    """One eye's landmark-derived gaze diagnostic in degrees and normalized units."""

    eye: str
    yaw_deg: float
    pitch_deg: float
    iris_offset_x: float
    iris_offset_y: float
    pupil_proxy_relative: float


@dataclass(frozen=True, slots=True)
class BinocularGazeSample:
    """Binocular gaze estimate plus per-eye diagnostics from one landmark frame."""

    t: float
    gaze: GazeSample
    right: EyeGazeDiagnostic
    left: EyeGazeDiagnostic
    vergence_x_deg: float
    vertical_disparity_deg: float
    asymmetry_deg: float
    pupil_proxy_relative: float


@dataclass(frozen=True, slots=True)
class Saccade:
    """A detected saccade with its main dynamics."""

    onset_idx: int
    offset_idx: int
    onset_t: float
    offset_t: float
    amplitude_deg: float
    direction_deg: float
    peak_velocity_deg_s: float

    @property
    def duration_s(self) -> float:
        return self.offset_t - self.onset_t


@dataclass(frozen=True, slots=True)
class Microsaccade:
    """A detected (binocular-agnostic) microsaccade, Engbert & Kliegl style."""

    onset_idx: int
    offset_idx: int
    amplitude_deg: float
    peak_velocity_deg_s: float
    direction_deg: float


@dataclass(frozen=True, slots=True)
class SmoothPursuit:
    """A provisional smooth-pursuit interval from sustained moderate velocity."""

    onset_idx: int
    offset_idx: int
    onset_t: float
    offset_t: float
    mean_velocity_deg_s: float
    direction_deg: float

    @property
    def duration_s(self) -> float:
        return self.offset_t - self.onset_t


@dataclass(frozen=True, slots=True)
class PSO:
    """A candidate post-saccadic oscillation after a detected saccade."""

    onset_idx: int
    offset_idx: int
    onset_t: float
    offset_t: float
    peak_velocity_deg_s: float
    parent_saccade_idx: int

    @property
    def duration_s(self) -> float:
        return self.offset_t - self.onset_t


@dataclass(frozen=True, slots=True)
class Fixation:
    """A detected fixation (period of stable gaze)."""

    onset_idx: int
    offset_idx: int
    onset_t: float
    offset_t: float
    centroid_x: float
    centroid_y: float

    @property
    def duration_s(self) -> float:
        return self.offset_t - self.onset_t


@dataclass(frozen=True, slots=True)
class GazeStream:
    """Aligned x/y/t gaze arrays in degrees of visual angle.

    Equal length is validated on construction; the arrays are stored as
    ``float64`` copies so a stream is an immutable snapshot.
    """

    t: FloatArray
    x: FloatArray
    y: FloatArray

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if not (t.shape == x.shape == y.shape):
            msg = (
                f"GazeStream requires equal-length t/x/y; got t={t.shape}, x={x.shape}, y={y.shape}"
            )
            raise ValueError(msg)
        if t.ndim != 1:
            msg = f"GazeStream arrays must be 1-D; got ndim={t.ndim}"
            raise ValueError(msg)
        # Bypass frozen to store normalized copies.
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def sampling_rate_hz(self) -> float:
        """Median sampling rate inferred from timestamps."""
        if len(self) < 2:
            msg = "Need >=2 samples to infer sampling rate"
            raise ValueError(msg)
        dt = float(np.median(np.diff(self.t)))
        if dt <= 0:
            msg = "Non-increasing timestamps; cannot infer sampling rate"
            raise ValueError(msg)
        return 1.0 / dt


@dataclass(frozen=True, slots=True)
class PupilStream:
    """Aligned pupil-size / time arrays with a single unit."""

    t: FloatArray
    size: FloatArray
    unit: PupilUnit = PupilUnit.RELATIVE

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        size = np.asarray(self.size, dtype=np.float64)
        if t.shape != size.shape:
            msg = f"PupilStream requires equal-length t/size; got {t.shape} vs {size.shape}"
            raise ValueError(msg)
        if t.ndim != 1:
            msg = f"PupilStream arrays must be 1-D; got ndim={t.ndim}"
            raise ValueError(msg)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "size", size)

    def __len__(self) -> int:
        return int(self.t.shape[0])


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Structured result of analysing one recording."""

    n_samples: int
    duration_s: float
    fixations: list[Fixation]
    saccades: list[Saccade]
    microsaccades: list[Microsaccade] = field(default_factory=list)
    smooth_pursuits: list[SmoothPursuit] = field(default_factory=list)
    psos: list[PSO] = field(default_factory=list)
    scanpath: str = ""
    main_sequence: NumericReportDict = field(default_factory=dict)
    pupil: NumericReportDict = field(default_factory=dict)
    quality: NumericReportDict = field(default_factory=dict)
    config: ReportPayload = field(default_factory=dict)

    def to_dict(self) -> ReportPayload:
        """JSON-serialisable summary (events collapsed to plain records)."""
        return {
            "n_samples": self.n_samples,
            "duration_s": self.duration_s,
            "n_fixations": len(self.fixations),
            "n_saccades": len(self.saccades),
            "n_microsaccades": len(self.microsaccades),
            "n_smooth_pursuits": len(self.smooth_pursuits),
            "n_psos": len(self.psos),
            "scanpath": self.scanpath,
            "main_sequence": self.main_sequence,
            "pupil": self.pupil,
            "quality": self.quality,
            "config": self.config,
            "saccades": [
                {
                    "onset_t": s.onset_t,
                    "offset_t": s.offset_t,
                    "amplitude_deg": s.amplitude_deg,
                    "direction_deg": s.direction_deg,
                    "peak_velocity_deg_s": s.peak_velocity_deg_s,
                    "duration_s": s.duration_s,
                }
                for s in self.saccades
            ],
            "psos": [
                {
                    "onset_t": p.onset_t,
                    "offset_t": p.offset_t,
                    "duration_s": p.duration_s,
                    "peak_velocity_deg_s": p.peak_velocity_deg_s,
                    "parent_saccade_idx": p.parent_saccade_idx,
                }
                for p in self.psos
            ],
            "smooth_pursuits": [
                {
                    "onset_t": p.onset_t,
                    "offset_t": p.offset_t,
                    "duration_s": p.duration_s,
                    "mean_velocity_deg_s": p.mean_velocity_deg_s,
                    "direction_deg": p.direction_deg,
                }
                for p in self.smooth_pursuits
            ],
            "fixations": [
                {
                    "onset_t": f.onset_t,
                    "offset_t": f.offset_t,
                    "duration_s": f.duration_s,
                    "centroid_x": f.centroid_x,
                    "centroid_y": f.centroid_y,
                }
                for f in self.fixations
            ],
        }
