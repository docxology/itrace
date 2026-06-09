"""Animated 3-D eye scene and the full closed-loop validation.

A :class:`EyeSceneSpec` declares a deterministic gaze trajectory (fixation
targets joined by minimum-jerk saccades), pupil dynamics (a baseline plus
Gaussian dilation events), and blink intervals. :func:`animate` renders the
per-frame 3-D truth and the projected MediaPipe-shaped landmark arrays;
:func:`closed_loop` pushes those frames through the *real* estimation path
(:func:`itrace.capture.iris_landmarks_to_sample` -> :class:`GazeStream` ->
:mod:`itrace.pipeline`) and measures how well the recovered signals match the
3-D truth.

This is the "full loop": 3-D animation -> camera projection -> landmark
extraction -> gaze/pupil estimation -> event detection -> recovery error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import pipeline, pupil
from .capture import iris_landmarks_to_sample
from .eyemodel import Camera, blink_landmarks, eye_to_landmarks, projected_pupil_ratio
from .synthetic import minimum_jerk_profile
from .types import BoolArray, FloatArray, GazeStream, PupilStream, PupilUnit, SessionReport


@dataclass(frozen=True, slots=True)
class Fixation3D:
    """A fixation target held for ``duration_s`` at gaze (yaw, pitch) degrees."""

    duration_s: float
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True, slots=True)
class DilationEvent:
    """A Gaussian pupil dilation: ``+amplitude_mm`` peak at ``onset_s``."""

    onset_s: float
    amplitude_mm: float
    width_s: float = 0.4


@dataclass(frozen=True, slots=True)
class EyeSceneSpec:
    """Declarative, deterministic specification of an eye-movement scene."""

    sampling_rate_hz: float = 120.0
    saccade_duration_s: float = 0.05
    fixations: tuple[Fixation3D, ...] = (
        Fixation3D(0.4, 0.0, 0.0),
        Fixation3D(0.4, 12.0, 0.0),
        Fixation3D(0.4, 12.0, 8.0),
        Fixation3D(0.4, -8.0, -6.0),
    )
    pupil_baseline_mm: float = 2.0
    dilations: tuple[DilationEvent, ...] = (
        DilationEvent(0.6, 1.2),
        DilationEvent(1.3, 0.8),
    )
    blinks_s: tuple[tuple[float, float], ...] = ((1.0, 1.08),)
    measurement_noise: float = 0.0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Scene:
    """Per-frame ground truth and the projected landmark frames."""

    t: FloatArray
    true_yaw: FloatArray
    true_pitch: FloatArray
    true_pupil_mm: FloatArray
    blink: BoolArray
    landmarks: list[FloatArray] = field(default_factory=list)
    true_saccades: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    """Recovered signals plus error metrics for one closed-loop run."""

    scene: Scene
    recovered_gaze: GazeStream
    recovered_pupil: PupilStream
    report: SessionReport
    metrics: dict[str, float]


def _trajectory(
    spec: EyeSceneSpec,
) -> tuple[FloatArray, FloatArray, FloatArray, list[tuple[int, int]]]:
    """Build per-frame (t, yaw, pitch) with min-jerk saccades between fixations.

    Also returns the [onset, offset] sample index of each ground-truth saccade
    segment, for detection-power scoring.
    """
    sr = spec.sampling_rate_hz
    n_sac = max(round(spec.saccade_duration_s * sr), 3)
    yaw_parts: list[FloatArray] = []
    pitch_parts: list[FloatArray] = []
    saccades_idx: list[tuple[int, int]] = []
    cursor = 0
    fixs = spec.fixations
    for i, fix in enumerate(fixs):
        n_hold = max(round(fix.duration_s * sr), 1)
        yaw_parts.append(np.full(n_hold, fix.yaw_deg))
        pitch_parts.append(np.full(n_hold, fix.pitch_deg))
        cursor += n_hold
        if i + 1 < len(fixs):
            nxt = fixs[i + 1]
            prof = minimum_jerk_profile(n_sac)
            yaw_parts.append(fix.yaw_deg + (nxt.yaw_deg - fix.yaw_deg) * prof)
            pitch_parts.append(fix.pitch_deg + (nxt.pitch_deg - fix.pitch_deg) * prof)
            saccades_idx.append((cursor, cursor + n_sac - 1))
            cursor += n_sac
    yaw = np.concatenate(yaw_parts)
    pitch = np.concatenate(pitch_parts)
    t = np.arange(yaw.size) / sr
    return t, yaw, pitch, saccades_idx


def _pupil_trace(spec: EyeSceneSpec, t: FloatArray) -> FloatArray:
    size = np.full(t.size, spec.pupil_baseline_mm, dtype=np.float64)
    for ev in spec.dilations:
        size += ev.amplitude_mm * np.exp(-(((t - ev.onset_s) / ev.width_s) ** 2))
    return size


def animate(spec: EyeSceneSpec | None = None, camera: Camera | None = None) -> Scene:
    """Render per-frame 3-D truth and projected landmark frames."""
    spec = spec or EyeSceneSpec()
    cam = camera or Camera()
    t, yaw, pitch, saccades_idx = _trajectory(spec)
    pupil_mm = _pupil_trace(spec, t)
    blink = np.zeros(t.size, dtype=bool)
    for b0, b1 in spec.blinks_s:
        blink |= (t >= b0) & (t <= b1)

    frames: list[FloatArray] = []
    for i in range(t.size):
        if blink[i]:
            frames.append(blink_landmarks())
        else:
            frames.append(
                eye_to_landmarks(
                    float(yaw[i]), float(pitch[i]), pupil_radius_mm=float(pupil_mm[i]), camera=cam
                )
            )
    return Scene(
        t=t,
        true_yaw=yaw,
        true_pitch=pitch,
        true_pupil_mm=pupil_mm,
        blink=blink,
        landmarks=frames,
        true_saccades=saccades_idx,
    )


def closed_loop(
    spec: EyeSceneSpec | None = None,
    camera: Camera | None = None,
    *,
    landmark_noise_sd: float = 0.0,
    pupil_noise_scale: float = 5.0,
    seed: int | None = None,
) -> ClosedLoopResult:
    """Run the full loop and measure recovery against the 3-D forward-model truth.

    ``landmark_noise_sd`` adds Gaussian noise (in normalised image-coordinate
    units) to every landmark before estimation, modelling imperfect webcam /
    MediaPipe localisation. The pupil/iris ratio is read from a *separate*
    modelled measurement, so its observation noise is added directly with
    standard deviation ``landmark_noise_sd * pupil_noise_scale`` (a modelling
    assumption rather than a calibrated image-segmentation or millimetre pupil
    model; see the assumption ledger in the docs). See :mod:`itrace.power` for
    the noise sweep.
    """
    spec = spec or EyeSceneSpec()
    cam = camera or Camera()
    scene = animate(spec, cam)
    rng = np.random.default_rng(spec.seed if seed is None else seed)

    rec_x = np.empty(scene.t.size, dtype=np.float64)
    rec_y = np.empty(scene.t.size, dtype=np.float64)
    pupil_ratio = np.empty(scene.t.size, dtype=np.float64)
    last_x, last_y = 0.0, 0.0
    for i in range(scene.t.size):
        if scene.blink[i]:
            # tracker holds last valid gaze; pupil is missing during a blink
            rec_x[i], rec_y[i] = last_x, last_y
            pupil_ratio[i] = np.nan
            continue
        frame = scene.landmarks[i]
        if landmark_noise_sd > 0:
            frame = frame + rng.normal(0.0, landmark_noise_sd, frame.shape)
        sample = iris_landmarks_to_sample(frame, float(scene.t[i]))
        rec_x[i], rec_y[i] = sample.x, sample.y
        last_x, last_y = sample.x, sample.y
        ratio = projected_pupil_ratio(
            float(scene.true_yaw[i]),
            float(scene.true_pitch[i]),
            float(scene.true_pupil_mm[i]),
            camera=cam,
        )
        if landmark_noise_sd > 0:
            ratio += float(rng.normal(0.0, landmark_noise_sd * pupil_noise_scale))
        pupil_ratio[i] = ratio

    if spec.measurement_noise > 0:
        rec_x = rec_x + rng.normal(0.0, spec.measurement_noise, rec_x.size)
        rec_y = rec_y + rng.normal(0.0, spec.measurement_noise, rec_y.size)

    recovered_gaze = GazeStream(t=scene.t, x=rec_x, y=rec_y)
    raw_pupil = PupilStream(t=scene.t, size=pupil_ratio, unit=PupilUnit.RELATIVE)
    clean_pupil = pupil.interpolate_blinks(raw_pupil)
    report = pipeline.analyze_session(recovered_gaze, clean_pupil)

    metrics = _metrics(scene, rec_x, rec_y, clean_pupil, report)
    return ClosedLoopResult(
        scene=scene,
        recovered_gaze=recovered_gaze,
        recovered_pupil=clean_pupil,
        report=report,
        metrics=metrics,
    )


def _metrics(
    scene: Scene,
    rec_x: FloatArray,
    rec_y: FloatArray,
    clean_pupil: PupilStream,
    report: SessionReport,
) -> dict[str, float]:
    valid = ~scene.blink
    err = np.hypot(rec_x[valid] - scene.true_yaw[valid], rec_y[valid] - scene.true_pitch[valid])
    gaze_rms = float(np.sqrt(np.mean(err**2)))
    gaze_max = float(np.max(err))
    # pupil correlation: recovered (deblinked relative ratio) vs true mm
    r = float(np.corrcoef(clean_pupil.size, scene.true_pupil_mm)[0, 1])
    prec, rec, f1 = _saccade_prf(
        [(s.onset_idx, s.offset_idx) for s in report.saccades], scene.true_saccades
    )
    return {
        "gaze_rms_deg": gaze_rms,
        "gaze_max_deg": gaze_max,
        "n_saccades": float(len(report.saccades)),
        "saccade_precision": prec,
        "saccade_recall": rec,
        "saccade_f1": f1,
        "pupil_corr": r,
        "n_frames": float(scene.t.size),
    }


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _saccade_prf(
    detected: list[tuple[int, int]], truth: list[tuple[int, int]]
) -> tuple[float, float, float]:
    """Precision / recall / F1 of detected saccade intervals vs ground truth.

    A true saccade counts as recalled if any detected interval overlaps it; a
    detected interval counts as a true positive if it overlaps any true saccade.
    """
    if not truth:
        return 0.0, 0.0, 0.0
    tp_truth = sum(any(_overlaps(d, g) for d in detected) for g in truth)
    tp_det = sum(any(_overlaps(d, g) for g in truth) for d in detected)
    recall = tp_truth / len(truth)
    precision = tp_det / len(detected) if detected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)
