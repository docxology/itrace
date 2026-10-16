from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .palette import INK, WONG, panel_label


def figure_resampling_sensitivity(payload: dict[str, Any]) -> Figure:
    """Plot the core AR(1) example, exact percentile endpoints, and mean SE reference."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    sequence, intervals, errors = axes
    values = np.asarray(payload["values"], dtype=np.float64)
    rows = payload["intervals"]
    sequence.plot(np.arange(values.size), values, color=WONG[0], lw=1)
    sequence.axhline(0, color=INK, ls="--", lw=1, label="generating mean")
    sequence.set(
        xlabel="ordered observation",
        ylabel="signal (arbitrary units)",
        title=f"Synthetic stationary AR(1), rho={payload['rho']:.2f}",
    )
    sequence.legend(loc="lower left", fontsize=9)
    for index, row in enumerate(rows):
        colour = WONG[1] if row["block_length"] == 1 else WONG[0]
        intervals.plot([row["ci_low"], row["ci_high"]], [index, index], color=colour, lw=3)
        intervals.plot(payload["mean"], index, "o", color=INK, ms=4)
    intervals.axvline(0, color=INK, ls="--", lw=1)
    intervals.set(
        yticks=range(len(rows)),
        yticklabels=[f"L={row['block_length']}" for row in rows],
        xlabel="resampled mean (arbitrary units)",
        title="95% percentile intervals",
    )
    intervals.invert_yaxis()
    positions = np.arange(len(rows))
    errors.bar(positions, [row["bootstrap_se"] for row in rows], color=WONG[0], alpha=0.85)
    errors.axhline(
        payload["analytic_mean_se"], color=WONG[1], ls="--", lw=2, label="analytic AR(1) mean SE"
    )
    errors.set(
        xticks=positions,
        xticklabels=[str(row["block_length"]) for row in rows],
        xlabel="block length (observations)",
        ylabel="SE (arbitrary units)",
        title="Sensitivity to the resampling unit",
    )
    errors.legend(fontsize=9)
    for label, axis in zip(("A", "B", "C"), axes, strict=True):
        panel_label(axis, label)
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle(
        f"Resampling design matters | one synthetic record, "
        f"n={payload['n']}, B={payload['n_boot']}\n"
        "Illustration only: block length requires justification; "
        "intervals are not population evidence",
        fontsize=12,
    )
    return figure
