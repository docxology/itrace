"""Independent second implementation of the core detectors (numerical oracle).

Deliberately uses *different formulations* from ``itrace`` (raw finite-difference
velocity instead of Savitzky-Golay; an explicit per-axis loop for microsaccades)
so that agreement between the two is a genuine cross-check rather than a
tautology. NumPy/SciPy only; no import of ``itrace``.

(Originally scoped to the cross-vendor Forge/Codex producer; Codex hit a usage
limit, so this same-vendor independent-formulation oracle stands in. The true
cross-vendor audit is tracked as FOLLOWUP-XVENDOR.)
"""

from __future__ import annotations

import numpy as np


def ivt_reference(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, velocity_threshold: float
) -> list[dict[str, float]]:
    """Raw finite-difference I-VT. Returns collapsed events."""
    dt = np.diff(t)
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    speed = np.hypot(vx, vy)
    # sample-aligned: pad to length of t with leading 0
    speed_full = np.concatenate(([0.0], speed))
    label = speed_full > velocity_threshold
    events: list[dict[str, float]] = []
    i = 0
    n = label.size
    while i < n:
        j = i
        while j + 1 < n and label[j + 1] == label[i]:
            j += 1
        events.append(
            {
                "kind": 1.0 if label[i] else 0.0,  # 1=saccade, 0=fixation
                "onset_idx": float(i),
                "offset_idx": float(j),
                "onset_t": float(t[i]),
                "offset_t": float(t[j]),
            }
        )
        i = j + 1
    return events


def microsaccade_reference(
    x: np.ndarray,
    y: np.ndarray,
    sampling_rate: float,
    lam: float = 6.0,
    min_samples: int = 3,
) -> list[dict[str, float]]:
    """Engbert-Kliegl with an explicit-loop 5-point velocity (independent code)."""
    n = x.size
    dt = 1.0 / sampling_rate
    vx = np.zeros(n)
    vy = np.zeros(n)
    for i in range(2, n - 2):
        vx[i] = (x[i + 2] + x[i + 1] - x[i - 1] - x[i - 2]) / (6.0 * dt)
        vy[i] = (y[i + 2] + y[i + 1] - y[i - 1] - y[i - 2]) / (6.0 * dt)
    sx = np.sqrt(max(np.median(vx**2) - np.median(vx) ** 2, 0.0))
    sy = np.sqrt(max(np.median(vy**2) - np.median(vy) ** 2, 0.0))
    eta_x, eta_y = lam * sx, lam * sy
    if eta_x <= 0 or eta_y <= 0:
        return []
    test = (vx / eta_x) ** 2 + (vy / eta_y) ** 2 > 1.0
    events: list[dict[str, float]] = []
    i = 0
    while i < n:
        if test[i]:
            j = i
            while j + 1 < n and test[j + 1]:
                j += 1
            if j - i + 1 >= min_samples:
                events.append({"onset_idx": float(i), "offset_idx": float(j)})
            i = j + 1
        else:
            i += 1
    return events
