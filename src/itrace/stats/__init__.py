"""Statistics subpackage for iTrace.

Pure NumPy/SciPy quantitative layers over detected oculomotor events (no
matplotlib, no optional dependency).
"""

from __future__ import annotations

from . import (
    bootstrap,
    descriptive,
    diagnostics,
    distributions,
    events,
    evidence,
    range_bridge,
    scanpath_metrics,
    similarity,
    timeseries,
)
from .bootstrap import bootstrap_statistic, percentile_interval
from .descriptive import (
    coefficient_of_variation,
    describe,
    fixation_summary,
    saccade_summary,
    session_statistics,
    summarize_report,
)
from .diagnostics import DEFAULT_DIAGNOSTIC_FAMILIES, session_statistical_diagnostics
from .distributions import (
    FAMILIES,
    FitResult,
    best_fit,
    compare_distributions,
    fit_distribution,
    frozen_from_result,
)
from .events import event_prf, interval_overlap_s
from .evidence import (
    build_statistical_interpretation_ledger,
    load_statistical_interpretation_ledger,
)
from .range_bridge import (
    build_synthetic_empirical_range_bridge,
    load_synthetic_empirical_range_bridge,
)
from .scanpath_metrics import (
    bcea,
    convex_hull_area,
    direction_transition_entropy,
    fixation_position_entropy,
    gaze_dispersion,
    gaze_path_efficiency,
    gaze_path_length,
    main_sequence_exponent_ci,
    raw_gaze_spatial_summary,
    scanpath_summary,
    shannon_entropy,
)
from .similarity import (
    levenshtein,
    ngram_cosine,
    normalized_levenshtein,
    scanpath_similarity,
    transition_matrix,
)
from .timeseries import (
    blink_rate_hz,
    event_rate,
    fixation_rate_series,
    microsaccade_rate_hz,
    saccade_rate_series,
    sliding_window_stat,
)

__all__ = [
    "DEFAULT_DIAGNOSTIC_FAMILIES",
    "FAMILIES",
    "FitResult",
    "bcea",
    "best_fit",
    "blink_rate_hz",
    "bootstrap",
    "bootstrap_statistic",
    "build_statistical_interpretation_ledger",
    "build_synthetic_empirical_range_bridge",
    "coefficient_of_variation",
    "compare_distributions",
    "convex_hull_area",
    "describe",
    "descriptive",
    "diagnostics",
    "direction_transition_entropy",
    "distributions",
    "event_prf",
    "event_rate",
    "events",
    "evidence",
    "fit_distribution",
    "fixation_position_entropy",
    "fixation_rate_series",
    "fixation_summary",
    "frozen_from_result",
    "gaze_dispersion",
    "gaze_path_efficiency",
    "gaze_path_length",
    "interval_overlap_s",
    "levenshtein",
    "load_statistical_interpretation_ledger",
    "load_synthetic_empirical_range_bridge",
    "main_sequence_exponent_ci",
    "microsaccade_rate_hz",
    "ngram_cosine",
    "normalized_levenshtein",
    "percentile_interval",
    "range_bridge",
    "raw_gaze_spatial_summary",
    "saccade_rate_series",
    "saccade_summary",
    "scanpath_metrics",
    "scanpath_similarity",
    "scanpath_summary",
    "session_statistical_diagnostics",
    "session_statistics",
    "shannon_entropy",
    "similarity",
    "sliding_window_stat",
    "summarize_report",
    "timeseries",
    "transition_matrix",
]
