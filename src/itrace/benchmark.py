"""External event-benchmark helpers.

iTrace does not bundle real reference-device data. These helpers compare iTrace
or comparator event outputs against caller-supplied truth files and keep that
truth boundary explicit in every payload.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .stats.events import event_prf, interval_overlap_s
from .types import Saccade

TRUTH_BOUNDARY = (
    "user-supplied event truth/comparators; no bundled real-device validation is claimed"
)


def load_saccade_events_csv(path: Path) -> list[Saccade]:
    """Load saccade-like events from a CSV with onset_t and offset_t columns."""
    rows = list(csv.DictReader(path.open(newline="")))
    required = {"onset_t", "offset_t"}
    if not rows:
        return []
    missing = sorted(required - set(rows[0]))
    if missing:
        msg = f"event CSV missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    events: list[Saccade] = []
    for index, row in enumerate(rows):
        onset = float(row["onset_t"])
        offset = float(row["offset_t"])
        if not np.isfinite(onset) or not np.isfinite(offset) or offset < onset:
            msg = f"invalid event timing in row {index + 1}"
            raise ValueError(msg)
        events.append(
            Saccade(
                onset_idx=0,
                offset_idx=0,
                onset_t=onset,
                offset_t=offset,
                amplitude_deg=float(row.get("amplitude_deg") or 0.0),
                direction_deg=float(row.get("direction_deg") or 0.0),
                peak_velocity_deg_s=float(row.get("peak_velocity_deg_s") or 0.0),
            )
        )
    return events


def _match_events(
    truth: Sequence[Saccade],
    predicted: Sequence[Saccade],
    *,
    min_overlap_s: float,
) -> list[tuple[Saccade, Saccade]]:
    matched: set[int] = set()
    pairs: list[tuple[Saccade, Saccade]] = []
    truth_sorted = sorted(truth, key=lambda event: event.onset_t)
    for pred in sorted(predicted, key=lambda event: event.onset_t):
        best_idx = -1
        best_overlap = min_overlap_s
        for idx, item in enumerate(truth_sorted):
            if idx in matched:
                continue
            overlap = interval_overlap_s(pred.onset_t, pred.offset_t, item.onset_t, item.offset_t)
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx >= 0:
            matched.add(best_idx)
            pairs.append((truth_sorted[best_idx], pred))
    return pairs


def _mean_abs(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.abs(arr))) if arr.size else 0.0


def compare_saccade_events(
    truth: Sequence[Saccade],
    predicted: Sequence[Saccade],
    *,
    min_overlap_s: float = 0.005,
) -> dict[str, object]:
    """Compare predicted saccades with caller-supplied truth events."""
    if min_overlap_s < 0.0:
        msg = "min_overlap_s must be non-negative"
        raise ValueError(msg)
    recovery = event_prf(truth, predicted, min_overlap_s=min_overlap_s)
    pairs = _match_events(truth, predicted, min_overlap_s=min_overlap_s)
    timing_errors = [pred.onset_t - true.onset_t for true, pred in pairs]
    duration_errors = [pred.duration_s - true.duration_s for true, pred in pairs]
    amplitude_errors = [
        pred.amplitude_deg - true.amplitude_deg
        for true, pred in pairs
        if true.amplitude_deg > 0.0 and pred.amplitude_deg > 0.0
    ]
    return {
        "n_truth": len(truth),
        "n_predicted": len(predicted),
        "n_matched": len(pairs),
        "recovery": recovery,
        "timing": {
            "onset_mae_s": _mean_abs(timing_errors),
            "duration_mae_s": _mean_abs(duration_errors),
        },
        "amplitude": {"mae_deg": _mean_abs(amplitude_errors)},
    }


def benchmark_payload(
    truth: Sequence[Saccade],
    systems: Mapping[str, Sequence[Saccade]],
    *,
    min_overlap_s: float = 0.005,
) -> dict[str, object]:
    """Build a JSON-friendly benchmark payload for one or more systems."""
    if not systems:
        msg = "at least one prediction/comparator event set is required"
        raise ValueError(msg)
    return {
        "kind": "external_saccade_benchmark",
        "truth_boundary": TRUTH_BOUNDARY,
        "min_overlap_s": min_overlap_s,
        "systems": {
            name: compare_saccade_events(truth, events, min_overlap_s=min_overlap_s)
            for name, events in systems.items()
        },
    }
