"""Event-alignment metrics for synthetic truth recovery."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import PSO, Fixation, Saccade

IntervalEvent = Fixation | Saccade | PSO


def interval_overlap_s(a_onset: float, a_offset: float, b_onset: float, b_offset: float) -> float:
    """Return overlap duration between two closed-open event intervals."""
    return max(0.0, min(a_offset, b_offset) - max(a_onset, b_onset))


def event_prf(
    truth: Sequence[IntervalEvent],
    predicted: Sequence[IntervalEvent],
    *,
    min_overlap_s: float = 0.0,
) -> dict[str, float]:
    """Precision/recall/F1 using greedy one-to-one interval overlap.

    A predicted event is a true positive when it overlaps an unmatched truth
    event by more than ``min_overlap_s``. The greedy ordering is deterministic:
    predictions are processed in onset order, and each chooses the unmatched
    truth interval with the largest overlap.
    """
    if min_overlap_s < 0.0:
        msg = "min_overlap_s must be non-negative"
        raise ValueError(msg)
    truth_sorted = sorted(truth, key=lambda e: e.onset_t)
    pred_sorted = sorted(predicted, key=lambda e: e.onset_t)
    matched: set[int] = set()
    true_positive = 0
    for pred in pred_sorted:
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
            true_positive += 1

    precision = true_positive / len(pred_sorted) if pred_sorted else 0.0
    recall = true_positive / len(truth_sorted) if truth_sorted else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    return {
        "true_positive": float(true_positive),
        "false_positive": float(len(pred_sorted) - true_positive),
        "false_negative": float(len(truth_sorted) - true_positive),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
