"""Composite statistical diagnostics for a session report.

This module is intentionally in :mod:`itrace.viz`: matplotlib is optional for
the package as a whole, while all fitted values and scanpath metrics come from
the tested pure-core statistics modules. The publication figure produced here
composes four diagnostics that otherwise live in separate reports:

* amplitude-distribution shape and model comparison by AIC / BIC,
* best-fit quantile and empirical-CDF residual diagnostics for amplitude-model adequacy,
* bootstrap uncertainty for the main-sequence exponent,
* fixation-stability descriptors (dispersion, hull area, BCEA), and
* first-order scanpath transition probabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from itrace.stats.diagnostics import session_statistical_diagnostics
from itrace.types import Fixation, FloatArray, SessionReport

from .palette import FONT_FLOOR, INK, MUTED, WONG, panel_label, result_box

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _fixation_xy(fixations: list[Fixation]) -> tuple[FloatArray, FloatArray]:
    """Return fixation-centroid coordinates as float arrays."""
    x = np.array([fix.centroid_x for fix in fixations], dtype=np.float64)
    y = np.array([fix.centroid_y for fix in fixations], dtype=np.float64)
    return x, y


def _placeholder(ax: Axes, title: str, text: str) -> None:
    """Draw a consistent no-data placeholder."""
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def _shape_callout(shape: dict[str, Any]) -> str:
    """Format the robust amplitude-shape summary computed by the core."""
    if not shape.get("available"):
        return str(shape.get("reason", "shape unavailable"))
    median = f"median {float(shape['median']):.2f} deg"
    iqr = f"IQR {float(shape['iqr']):.2f} deg"
    if "median_ci_low" in shape and "median_ci_high" in shape:
        median = (
            f"{median}; CI {float(shape['median_ci_low']):.2f}-{float(shape['median_ci_high']):.2f}"
        )
    if "iqr_ci_low" in shape and "iqr_ci_high" in shape:
        iqr = f"{iqr}; CI {float(shape['iqr_ci_low']):.2f}-{float(shape['iqr_ci_high']):.2f}"
    return "\n".join(
        [
            median,
            iqr,
            (
                f"Bowley skew {float(shape['bowley_skewness']):.2f}; "
                f"IQR outliers {int(shape['iqr_outlier_count'])}/{int(shape['n'])}"
            ),
        ]
    )


def _plot_model_comparison(
    ax: Axes,
    model: dict[str, Any],
    shape: dict[str, Any],
) -> None:
    """AIC/BIC comparison plus robust shape summary for positive amplitudes."""
    ax.set_title("Model comparison")
    ax.set_xlabel("delta AIC (lower is better)")

    results = model.get("families", [])
    if not results:
        reason = str(model.get("reason", "no stable distribution fits"))
        _placeholder(ax, "Model comparison", reason)
        return

    fit_rows = [row for row in results if isinstance(row, dict)]
    y = np.arange(len(fit_rows))
    delta_aic = np.array([float(row["delta_aic"]) for row in fit_rows], dtype=np.float64)
    colours = [WONG[2] if idx == 0 else WONG[0] for idx in range(len(fit_rows))]

    ax.barh(y, delta_aic, color=colours, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["family"]) for row in fit_rows])
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0.0, max(float(np.max(delta_aic)), 1.0) * 1.65)

    for idx, (delta, row) in enumerate(zip(delta_aic, fit_rows, strict=True)):
        label = (
            f"w={float(row.get('akaike_weight', 0.0)):.2f} | "
            f"BIC={float(row['bic']):.1f} | KS={float(row['ks_statistic']):.2f}"
        )
        ax.text(
            max(float(delta), 0.05) + 0.05,
            idx,
            label,
            va="center",
            fontsize=FONT_FLOOR,
            color=MUTED,
        )

    bootstrap = model.get("model_selection_bootstrap", {})
    boot_line = ""
    if isinstance(bootstrap, dict) and bootstrap.get("available"):
        boot_line = f"\nboot top={float(bootstrap.get('top_frequency', 0.0)):.0%}"
    result_box(
        ax,
        (
            f"best: {model.get('best_family', fit_rows[0]['family'])}\n"
            f"w(AIC)={float(fit_rows[0].get('akaike_weight', 0.0)):.2f}"
            f"{boot_line}\n"
            f"n={model.get('n', fit_rows[0]['n'])}"
        ),
        xy=(0.98, 0.96),
        color=WONG[2],
        align="right",
    )
    result_box(
        ax,
        _shape_callout(shape),
        xy=(0.98, 0.62),
        color=WONG[1],
        align="right",
    )


def _plot_quantile_diagnostics(
    ax: Axes,
    quantile: dict[str, Any],
    cdf_payload: dict[str, Any],
) -> None:
    """QQ/probability-plot check plus empirical-CDF residuals for the best fit."""
    ax.set_title("Best-fit QQ")
    ax.set_xlabel("fitted amplitude quantile (deg)")
    ax.set_ylabel("empirical amplitude quantile (deg)")

    if not quantile.get("available"):
        reason = str(quantile.get("reason", "quantile diagnostics unavailable"))
        _placeholder(ax, "Best-fit QQ", reason)
        return

    fitted = np.asarray(quantile["fitted_quantiles_deg"], dtype=np.float64)
    empirical = np.asarray(quantile["empirical_quantiles_deg"], dtype=np.float64)
    residual = np.asarray(quantile["residuals_deg"], dtype=np.float64)
    if fitted.size == 0 or empirical.size == 0:
        _placeholder(ax, "Best-fit QQ", "quantile diagnostics unavailable")
        return

    lo = float(np.nanmin([np.min(fitted), np.min(empirical)]))
    hi = float(np.nanmax([np.max(fitted), np.max(empirical)]))
    pad = max((hi - lo) * 0.08, 0.5)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, alpha=0.55, label="identity")
    ax.axhline(0.0, color=INK, lw=0.8, alpha=0.10)

    abs_residual = np.abs(residual)
    sizes = 46.0 + 120.0 * abs_residual / max(float(np.max(abs_residual)), 1e-9)
    scatter = ax.scatter(
        fitted,
        empirical,
        s=sizes,
        c=abs_residual,
        cmap="magma",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|empirical - fitted| deg")
    result_box(
        ax,
        _quantile_callout(quantile, cdf_payload),
        xy=(0.05, 0.95),
        color=WONG[5],
    )
    _plot_cdf_residual_inset(ax, cdf_payload)


def _quantile_callout(quantile: dict[str, Any], cdf_payload: dict[str, Any]) -> str:
    """Format the QQ/P-P residual callout."""
    lines = [
        (
            f"{quantile.get('family', 'best')} fit; "
            f"QQ RMSE {float(quantile['residual_rmse_deg']):.2f} deg"
        ),
        f"slope {float(quantile['qq_slope']):.2f}; r {float(quantile['qq_correlation']):.2f}",
    ]
    if cdf_payload.get("available"):
        lines.append(f"P-P max |res| {float(cdf_payload['max_abs_cdf_residual']):.2f}")
        if "cramer_von_mises_statistic" in cdf_payload:
            lines.append(
                "CvM "
                f"{float(cdf_payload['cramer_von_mises_statistic']):.2f}; "
                f"AD {float(cdf_payload['anderson_darling_statistic']):.2f}"
            )
        if "dkw_epsilon" in cdf_payload:
            confidence = float(cdf_payload.get("dkw_confidence", 0.95))
            lines.append(f"DKW {confidence:.0%} +/- {float(cdf_payload['dkw_epsilon']):.2f}")
    return "\n".join(lines)


def _plot_cdf_residual_inset(ax: Axes, cdf_payload: dict[str, Any]) -> None:
    """Small P-P residual inset using core-computed empirical-CDF diagnostics."""
    if not cdf_payload.get("available"):
        return
    fitted = np.asarray(cdf_payload["fitted_cdf"], dtype=np.float64)
    residual = np.asarray(cdf_payload["cdf_residuals"], dtype=np.float64)
    if fitted.size == 0 or residual.size == 0:
        return

    inset = ax.inset_axes((0.56, 0.08, 0.36, 0.28))
    inset.patch.set_alpha(0.94)
    epsilon = float(cdf_payload.get("dkw_epsilon", 0.0))
    if epsilon > 0.0:
        inset.axhspan(-epsilon, epsilon, color=WONG[5], alpha=0.10)
        inset.axhline(epsilon, color=WONG[5], lw=0.8, ls="--", alpha=0.70)
        inset.axhline(-epsilon, color=WONG[5], lw=0.8, ls="--", alpha=0.70)
    inset.axhline(0.0, color=INK, lw=0.8, alpha=0.55)
    inset.plot(fitted, residual, color=WONG[2], marker="o", ms=3.0, lw=1.0)
    inset.set_title("P-P residual", fontsize=FONT_FLOOR)
    inset.set_ylabel("Emp - fit", fontsize=FONT_FLOOR - 1)
    inset.tick_params(labelsize=FONT_FLOOR - 1)
    inset.grid(True, alpha=0.20)
    max_abs = max(float(np.max(np.abs(residual))), epsilon, 0.05)
    inset.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    inset.set_xlim(0.0, 1.0)


def _plot_main_sequence_ci(ax: Axes, stats_payload: dict[str, Any]) -> None:
    """Bootstrap confidence interval for the main-sequence power exponent."""
    ax.set_title("Main-sequence exponent")
    ax.set_xlabel("power-law exponent b")
    ax.set_yticks([])

    if not stats_payload.get("available"):
        reason = str(stats_payload.get("reason", "fit unavailable"))
        _placeholder(ax, "Main-sequence exponent", reason)
        return

    estimate = float(stats_payload["estimate"])
    lo = float(stats_payload["ci_low"])
    hi = float(stats_payload["ci_high"])
    confidence = float(stats_payload.get("confidence", 0.95))
    n_boot = int(stats_payload.get("n_boot", 0))

    lower_err = max(estimate - lo, 0.0)
    upper_err = max(hi - estimate, 0.0)
    ax.errorbar(
        estimate,
        0,
        xerr=np.array([[lower_err], [upper_err]], dtype=np.float64),
        fmt="o",
        ms=9,
        capsize=7,
        color=WONG[4],
        ecolor=WONG[4],
        lw=2.5,
    )
    pad = max((hi - lo) * 1.8, 0.035)
    ax.set_xlim(min(lo, estimate) - pad, max(hi, estimate) + pad)
    ax.axvline(estimate, color=INK, lw=1.0, alpha=0.45)
    ax.grid(True, axis="x", alpha=0.25)
    result_box(
        ax,
        f"b={estimate:.2f}\n{confidence:.0%} CI {lo:.2f}-{hi:.2f}\nbootstrap B={n_boot}",
        xy=(0.05, 0.92),
        color=WONG[4],
    )


def _ellipse_patch(x: FloatArray, y: FloatArray, *, probability: float = 0.68) -> Ellipse | None:
    """Return a BCEA-aligned ellipse patch, or ``None`` on degenerate points."""
    if x.size < 2 or y.size < 2:
        return None
    cov = np.cov(np.column_stack((x, y)).T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, 0.0)
    if float(values.max()) <= 0.0:
        return None
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    k = -np.log(1.0 - probability)
    width, height = 2.0 * np.sqrt(2.0 * k * values)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    return Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor="none",
        edgecolor=WONG[3],
        lw=2.0,
        alpha=0.95,
    )


def _plot_spatial_stability(
    ax: Axes,
    fixations: list[Fixation],
    spatial_payload: dict[str, Any],
) -> None:
    """Fixation centroids plus core scanpath-spread metrics."""
    ax.set_title("Spatial stability")
    ax.set_xlabel("horizontal gaze (deg)")
    ax.set_ylabel("vertical gaze (deg, screen)")

    if not fixations:
        _placeholder(ax, "Spatial stability", "no fixations")
        ax.invert_yaxis()
        return

    x, y = _fixation_xy(fixations)
    duration = np.array([max(fix.duration_s, 0.0) for fix in fixations], dtype=np.float64)
    sizes = 45.0 + 240.0 * duration / max(float(duration.max()), 1e-9)
    ax.scatter(x, y, s=sizes, color=WONG[0], edgecolor="white", linewidth=0.8, zorder=3)
    if x.size > 1:
        ax.plot(x, y, color=WONG[5], lw=1.1, alpha=0.65, zorder=2)
    ellipse = _ellipse_patch(x, y)
    if ellipse is not None:
        ax.add_patch(ellipse)

    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    result_box(
        ax,
        "\n".join(
            [
                f"dispersion {float(spatial_payload['gaze_dispersion']):.2f} deg",
                f"BCEA68 {float(spatial_payload['bcea']):.2f} deg^2",
                f"hull {float(spatial_payload['convex_hull_area']):.2f} deg^2",
            ]
        ),
        xy=(0.05, 0.95),
        color=WONG[3],
    )


def _plot_transition_matrix(ax: Axes, transition_payload: dict[str, Any]) -> None:
    """First-order direction-transition matrix for the encoded scanpath."""
    ax.set_title("Scanpath transitions")
    symbols = [str(symbol) for symbol in transition_payload.get("symbols", [])]
    matrix = np.asarray(transition_payload.get("matrix", []), dtype=np.float64)
    if not symbols:
        _placeholder(ax, "Scanpath transitions", "no encoded saccades")
        return

    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)
    ax.set_xlabel("next symbol")
    ax.set_ylabel("current symbol")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            if value > 0.0:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.45 else INK,
                    fontsize=FONT_FLOOR,
                )
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p(next | current)")
    scanpath = str(transition_payload.get("scanpath", "")) or "empty"
    if len(scanpath) > 24:
        scanpath = scanpath[:21] + "..."
    ax.text(
        0.0,
        -0.22,
        f"scanpath={scanpath}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_FLOOR,
        color=MUTED,
        clip_on=False,
    )


def figure_statistical_diagnostics(
    report: SessionReport,
    *,
    payload: dict[str, Any] | None = None,
) -> Figure:
    """Render the publication statistical-diagnostics composite.

    Parameters
    ----------
    report
        Session report whose detected saccades/fixations are summarised.
    payload
        Optional precomputed :func:`itrace.stats.diagnostics.session_statistical_diagnostics`
        payload. When omitted it is computed with the publication defaults.

    Returns
    -------
    matplotlib.figure.Figure
        Five-panel figure: amplitude shape/model comparison, best-fit QQ/P-P
        residual checks, main-sequence uncertainty, fixation-stability
        descriptors, and transition probabilities.
    """
    diagnostics = payload if payload is not None else session_statistical_diagnostics(report)

    fig = plt.figure(figsize=(13.2, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    ax_model = fig.add_subplot(grid[0, 0])
    ax_qq = fig.add_subplot(grid[0, 1])
    ax_ci = fig.add_subplot(grid[1, 0])
    ax_spatial = fig.add_subplot(grid[1, 1])
    ax_transition = fig.add_subplot(grid[2, :])

    _plot_model_comparison(
        ax_model,
        diagnostics["amplitude_model_comparison"],
        diagnostics["amplitude_shape_summary"],
    )
    _plot_quantile_diagnostics(
        ax_qq,
        diagnostics["amplitude_quantile_diagnostics"],
        diagnostics["amplitude_cdf_diagnostics"],
    )
    _plot_main_sequence_ci(ax_ci, diagnostics["main_sequence_exponent"])
    _plot_spatial_stability(ax_spatial, report.fixations, diagnostics["spatial_stability"])
    _plot_transition_matrix(ax_transition, diagnostics["transition_matrix"])

    for label, ax in zip(
        ("A", "B", "C", "D", "E"),
        (ax_model, ax_qq, ax_ci, ax_spatial, ax_transition),
        strict=True,
    ):
        panel_label(ax, label)

    fig.suptitle(
        "Statistical diagnostics over a tested synthetic session",
        fontsize=16,
        fontweight="bold",
    )
    return fig
