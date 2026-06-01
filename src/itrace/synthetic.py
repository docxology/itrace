"""Synthetic signal generators with known ground truth.

These are the backbone of validation: each generator returns both a signal and
the parameters used to build it, so a detector can be checked against truth it
could not have peeked at. Every generator takes an explicit ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from .types import FloatArray, GazeStream, IntArray, PupilStream, PupilUnit


@dataclass(frozen=True, slots=True)
class SyntheticSaccadeTruth:
    """Ground truth for a single synthesised saccade."""

    onset_t: float
    offset_t: float
    amplitude_deg: float
    direction_deg: float
    peak_velocity_deg_s: float


@dataclass(frozen=True, slots=True)
class SyntheticPupilEventTruth:
    """Ground truth for one synthetic pupil dilation event."""

    onset_t: float
    peak_t: float
    amplitude: float


@dataclass(frozen=True, slots=True)
class SyntheticSessionSpec:
    """Configurable synthetic gaze+pupil session specification."""

    sampling_rate_hz: float = 250.0
    duration_s: float = 6.0
    n_saccades: int = 6
    amplitude_range_deg: tuple[float, float] = (3.0, 16.0)
    saccade_duration_s: float = 0.04
    noise_deg: float = 0.02
    pso_amplitude_fraction: float = 0.08
    blink_windows_s: tuple[tuple[float, float], ...] = ((2.6, 2.75),)
    pupil_baseline: float = 3.0
    pupil_noise: float = 0.02
    dropout_fraction: float = 0.0
    timestamp_jitter_s: float = 0.0
    correlated_noise_deg: float = 0.0
    head_pose_drift_deg: float = 0.0
    lighting_dropouts_s: tuple[tuple[float, float], ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0.0:
            msg = "sampling_rate_hz must be positive"
            raise ValueError(msg)
        if self.duration_s <= 0.0:
            msg = "duration_s must be positive"
            raise ValueError(msg)
        if self.n_saccades < 0:
            msg = "n_saccades must be non-negative"
            raise ValueError(msg)
        if self.saccade_duration_s <= 0.0:
            msg = "saccade_duration_s must be positive"
            raise ValueError(msg)
        lo, hi = self.amplitude_range_deg
        if lo < 0.0 or hi < lo:
            msg = "amplitude_range_deg must be a non-negative (lo, hi) pair"
            raise ValueError(msg)
        if self.noise_deg < 0.0 or self.pupil_noise < 0.0:
            msg = "noise terms must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= self.dropout_fraction < 1.0:
            msg = "dropout_fraction must be in [0, 1)"
            raise ValueError(msg)
        if self.timestamp_jitter_s < 0.0:
            msg = "timestamp_jitter_s must be non-negative"
            raise ValueError(msg)
        if self.correlated_noise_deg < 0.0:
            msg = "correlated_noise_deg must be non-negative"
            raise ValueError(msg)
        if self.head_pose_drift_deg < 0.0:
            msg = "head_pose_drift_deg must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SyntheticSessionTruth:
    """Ground truth returned by :func:`synthetic_session`."""

    saccades: list[SyntheticSaccadeTruth]
    pupil_events: list[SyntheticPupilEventTruth]
    blink_windows_s: tuple[tuple[float, float], ...]
    lighting_dropouts_s: tuple[tuple[float, float], ...]
    quality_flags: dict[str, float]
    seed: int


def _minimum_jerk(n: int) -> FloatArray:
    """Normalised minimum-jerk displacement profile in [0, 1] over n samples."""
    tau = np.linspace(0.0, 1.0, n)
    return cast(FloatArray, 10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def gaze_with_saccade(
    *,
    sampling_rate_hz: float = 250.0,
    fixation_s: float = 0.3,
    amplitude_deg: float = 10.0,
    direction_deg: float = 0.0,
    saccade_duration_s: float = 0.05,
    noise_deg: float = 0.0,
    seed: int = 0,
) -> tuple[GazeStream, SyntheticSaccadeTruth]:
    """Fixation -> saccade -> fixation with a minimum-jerk velocity profile.

    Direction follows the gaze convention (0deg right, +90deg up). Optional
    Gaussian position noise (deg) is added. Returns the stream and the truth.
    """
    rng = np.random.default_rng(seed)
    n_fix = round(fixation_s * sampling_rate_hz)
    n_sac = max(round(saccade_duration_s * sampling_rate_hz), 3)

    profile = _minimum_jerk(n_sac)
    rad = np.radians(direction_deg)
    dx_total, dy_total = amplitude_deg * np.cos(rad), amplitude_deg * np.sin(rad)

    x = np.concatenate([np.zeros(n_fix), dx_total * profile, np.full(n_fix, dx_total)])
    # screen-y is down, gaze-up is +; store screen coords so detectors negate dy.
    y = np.concatenate([np.zeros(n_fix), -dy_total * profile, np.full(n_fix, -dy_total)])
    n = x.size
    t = np.arange(n) / sampling_rate_hz
    if noise_deg > 0:
        x = x + rng.normal(0.0, noise_deg, n)
        y = y + rng.normal(0.0, noise_deg, n)

    onset_t = float(t[n_fix])
    offset_t = float(t[n_fix + n_sac - 1])
    peak_v = amplitude_deg * 1.875 / saccade_duration_s  # min-jerk peak speed
    truth = SyntheticSaccadeTruth(
        onset_t=onset_t,
        offset_t=offset_t,
        amplitude_deg=amplitude_deg,
        direction_deg=direction_deg,
        peak_velocity_deg_s=peak_v,
    )
    return GazeStream(t=t, x=x, y=y), truth


def gaze_fixation_noise(
    *,
    sampling_rate_hz: float = 250.0,
    duration_s: float = 1.0,
    noise_deg: float = 0.05,
    seed: int = 0,
) -> GazeStream:
    """A pure fixation: small Gaussian jitter around the origin, no saccades."""
    rng = np.random.default_rng(seed)
    n = round(duration_s * sampling_rate_hz)
    t = np.arange(n) / sampling_rate_hz
    x = rng.normal(0.0, noise_deg, n)
    y = rng.normal(0.0, noise_deg, n)
    return GazeStream(t=t, x=x, y=y)


def gaze_with_microsaccade(
    *,
    sampling_rate_hz: float = 500.0,
    duration_s: float = 1.0,
    microsaccade_amplitude_deg: float = 0.5,
    onset_s: float = 0.5,
    micro_duration_s: float = 0.02,
    noise_deg: float = 0.01,
    seed: int = 0,
) -> tuple[GazeStream, float]:
    """Fixational jitter with a single embedded horizontal microsaccade.

    Returns the stream and the microsaccade onset time.
    """
    rng = np.random.default_rng(seed)
    n = round(duration_s * sampling_rate_hz)
    t = np.arange(n) / sampling_rate_hz
    x = rng.normal(0.0, noise_deg, n)
    y = rng.normal(0.0, noise_deg, n)
    i0 = round(onset_s * sampling_rate_hz)
    n_micro = max(round(micro_duration_s * sampling_rate_hz), 3)
    profile = _minimum_jerk(n_micro)
    x[i0 : i0 + n_micro] += microsaccade_amplitude_deg * profile
    x[i0 + n_micro :] += microsaccade_amplitude_deg
    return GazeStream(t=t, x=x, y=y), float(t[i0])


def pupil_sine_with_blink(
    *,
    sampling_rate_hz: float = 60.0,
    duration_s: float = 10.0,
    period_s: float = 2.0,
    baseline: float = 3.0,
    amplitude: float = 0.5,
    blink_window_s: tuple[float, float] | None = (4.0, 4.3),
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[PupilStream, IntArray]:
    """Sinusoidal pupil signal (mm) with an optional NaN blink.

    Returns the stream and the array of analytic peak indices (sine maxima).
    """
    rng = np.random.default_rng(seed)
    n = round(duration_s * sampling_rate_hz)
    t = np.arange(n) / sampling_rate_hz
    size = baseline + amplitude * np.sin(2 * np.pi * t / period_s)
    if noise > 0:
        size = size + rng.normal(0.0, noise, n)
    if blink_window_s is not None:
        b0, b1 = blink_window_s
        size[(t >= b0) & (t <= b1)] = np.nan
    # analytic maxima: sin peaks at phase pi/2 -> t = period*(1/4 + k)
    ks = np.arange(0, int(duration_s / period_s) + 1)
    peak_times = period_s * (0.25 + ks)
    peak_idx = np.array(
        [round(pt * sampling_rate_hz) for pt in peak_times if pt < duration_s],
        dtype=int,
    )
    return PupilStream(t=t, size=size, unit=PupilUnit.MM), peak_idx


def synthetic_session(
    spec: SyntheticSessionSpec | None = None,
) -> tuple[GazeStream, PupilStream, SyntheticSessionTruth]:
    """Generate a deterministic multi-event gaze+pupil recording.

    The gaze trace contains seeded fixations, minimum-jerk saccades, optional
    post-saccadic ringing, Gaussian position noise, and optional random dropout.
    The pupil trace contains smooth dilation events time-locked to the first few
    saccades plus explicit NaN blink windows. The returned truth object is
    independent of any detector and is intended for recovery tests.
    """
    cfg = spec or SyntheticSessionSpec()
    rng = np.random.default_rng(cfg.seed)
    n = max(round(cfg.duration_s * cfg.sampling_rate_hz), 2)
    t = np.arange(n, dtype=np.float64) / cfg.sampling_rate_hz
    if cfg.timestamp_jitter_s > 0.0:
        jitter = rng.normal(0.0, cfg.timestamp_jitter_s, n)
        jitter[0] = 0.0
        t = np.sort(t + jitter)
        t = t - float(t[0])
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    if cfg.n_saccades > 0:
        margin = max(0.25, cfg.saccade_duration_s * 2.0)
        event_times = np.linspace(margin, max(margin, cfg.duration_s - margin), cfg.n_saccades)
    else:
        event_times = np.zeros(0, dtype=np.float64)

    truths: list[SyntheticSaccadeTruth] = []
    cur_x = 0.0
    cur_y = 0.0
    last_end = 0
    n_sac = max(round(cfg.saccade_duration_s * cfg.sampling_rate_hz), 3)
    lo_amp, hi_amp = cfg.amplitude_range_deg
    for onset_t in event_times:
        onset = min(max(round(float(onset_t) * cfg.sampling_rate_hz), 0), n - 1)
        offset = min(onset + n_sac - 1, n - 1)
        if offset <= onset:
            continue
        x[last_end:onset] = cur_x
        y[last_end:onset] = cur_y
        amp = float(rng.uniform(lo_amp, hi_amp))
        direction = float(rng.uniform(-180.0, 180.0))
        rad = np.radians(direction)
        dx = amp * np.cos(rad)
        dy_screen = -amp * np.sin(rad)
        profile = _minimum_jerk(offset - onset + 1)
        x[onset : offset + 1] = cur_x + dx * profile
        y[onset : offset + 1] = cur_y + dy_screen * profile
        cur_x += dx
        cur_y += dy_screen

        if cfg.pso_amplitude_fraction > 0.0 and offset + 2 < n:
            ring_n = min(round(0.06 * cfg.sampling_rate_hz), n - offset - 1)
            k = np.arange(ring_n, dtype=np.float64)
            ring = (
                cfg.pso_amplitude_fraction * amp * np.exp(-k / 8.0) * np.sin(2.0 * np.pi * k / 8.0)
            )
            unit_x = np.cos(rad)
            unit_y = -np.sin(rad)
            x[offset + 1 : offset + 1 + ring_n] = cur_x + ring * unit_x
            y[offset + 1 : offset + 1 + ring_n] = cur_y + ring * unit_y
            last_end = offset + 1 + ring_n
        else:
            last_end = offset + 1

        truths.append(
            SyntheticSaccadeTruth(
                onset_t=float(t[onset]),
                offset_t=float(t[offset]),
                amplitude_deg=amp,
                direction_deg=direction,
                peak_velocity_deg_s=amp * 1.875 / cfg.saccade_duration_s,
            )
        )

    x[last_end:] = cur_x
    y[last_end:] = cur_y
    if cfg.noise_deg > 0.0:
        x += rng.normal(0.0, cfg.noise_deg, n)
        y += rng.normal(0.0, cfg.noise_deg, n)
    if cfg.correlated_noise_deg > 0.0:
        for target in (x, y):
            state = 0.0
            for i in range(n):
                state = 0.85 * state + float(rng.normal(0.0, cfg.correlated_noise_deg))
                target[i] += state
    if cfg.head_pose_drift_deg > 0.0:
        drift = cfg.head_pose_drift_deg * np.sin(2.0 * np.pi * t / max(cfg.duration_s, 1e-6))
        x += drift
        y += 0.5 * drift
    if cfg.dropout_fraction > 0.0:
        keep = rng.random(n) >= cfg.dropout_fraction
        x[~keep] = np.nan
        y[~keep] = np.nan

    pupil_size = np.full(n, cfg.pupil_baseline, dtype=np.float64)
    pupil_events: list[SyntheticPupilEventTruth] = []
    for truth in truths[:3]:
        pupil_onset = truth.onset_t + 0.25
        peak = pupil_onset + 0.35
        amp = 0.08 + 0.01 * truth.amplitude_deg
        response = (
            amp * (1.0 - np.exp(-(t - pupil_onset) / 0.25)) * np.exp(-(t - pupil_onset) / 1.4)
        )
        response[t < pupil_onset] = 0.0
        pupil_size += response
        pupil_events.append(
            SyntheticPupilEventTruth(onset_t=pupil_onset, peak_t=peak, amplitude=amp)
        )
    if cfg.pupil_noise > 0.0:
        pupil_size += rng.normal(0.0, cfg.pupil_noise, n)
    for b0, b1 in cfg.blink_windows_s:
        pupil_size[(t >= b0) & (t <= b1)] = np.nan
    for d0, d1 in cfg.lighting_dropouts_s:
        pupil_size[(t >= d0) & (t <= d1)] = np.nan

    return (
        GazeStream(t=t, x=x, y=y),
        PupilStream(t=t, size=pupil_size, unit=PupilUnit.MM),
        SyntheticSessionTruth(
            saccades=truths,
            pupil_events=pupil_events,
            blink_windows_s=cfg.blink_windows_s,
            lighting_dropouts_s=cfg.lighting_dropouts_s,
            quality_flags={
                "timestamp_jitter_s": cfg.timestamp_jitter_s,
                "correlated_noise_deg": cfg.correlated_noise_deg,
                "head_pose_drift_deg": cfg.head_pose_drift_deg,
                "dropout_fraction": cfg.dropout_fraction,
                "lighting_dropout_count": float(len(cfg.lighting_dropouts_s)),
            },
            seed=cfg.seed,
        ),
    )
