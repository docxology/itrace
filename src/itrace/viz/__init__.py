"""Visualization subpackage for iTrace.

A gallery of publication-quality, deterministic plots over the analysis core:
velocity and pupil traces, spatial scanpaths, microsaccade panels,
distribution-with-fit histograms, main-sequence diagnostics, a multi-panel
session dashboard, spatial fixation-density / AOI plots, and temporal event
rasters and pupil power-spectra.

.. note::
   matplotlib is an **optional** dependency of iTrace (the ``figures`` extra).
   Importing the top-level :mod:`itrace` package never imports this subpackage,
   so the core stays headless-dependency-free. Importing ``itrace.viz`` (or any
   of its modules) requires matplotlib to be installed.
"""

from __future__ import annotations

from . import dashboard, distributions, gallery, scanpath, spatial, timeline, traces
from .dashboard import render_dashboard, session_dashboard
from .distributions import (
    figure_amplitude_histogram,
    figure_duration_histogram,
    figure_main_sequence,
    plot_amplitude_histogram,
    plot_duration_histogram,
    plot_main_sequence,
    plot_main_sequence_residuals,
)
from .scanpath import (
    figure_microsaccades,
    figure_scanpath,
    plot_microsaccade_main_sequence,
    plot_microsaccade_polar,
    plot_scanpath,
)
from .spatial import (
    assign_aoi,
    figure_aoi,
    figure_fixation_heatmap,
    figure_gaze_density,
    fixation_heatmap,
    gaze_density,
    plot_aoi_dwell,
)
from .timeline import (
    figure_event_raster,
    figure_pupil_psd,
    figure_rate,
    plot_event_raster,
    plot_pupil_psd,
    plot_rate,
)
from .traces import (
    figure_pupil_trace,
    figure_velocity_trace,
    plot_pupil_trace,
    plot_velocity_trace,
)

__all__ = [
    "assign_aoi",
    "dashboard",
    "distributions",
    "figure_amplitude_histogram",
    "figure_aoi",
    "figure_duration_histogram",
    "figure_event_raster",
    "figure_fixation_heatmap",
    "figure_gaze_density",
    "figure_main_sequence",
    "figure_microsaccades",
    "figure_pupil_psd",
    "figure_pupil_trace",
    "figure_rate",
    "figure_scanpath",
    "figure_velocity_trace",
    "fixation_heatmap",
    "gallery",
    "gaze_density",
    "plot_amplitude_histogram",
    "plot_aoi_dwell",
    "plot_duration_histogram",
    "plot_event_raster",
    "plot_main_sequence",
    "plot_main_sequence_residuals",
    "plot_microsaccade_main_sequence",
    "plot_microsaccade_polar",
    "plot_pupil_psd",
    "plot_pupil_trace",
    "plot_rate",
    "plot_scanpath",
    "plot_velocity_trace",
    "render_dashboard",
    "render_gallery",
    "scanpath",
    "session_dashboard",
    "spatial",
    "timeline",
    "traces",
]
from .gallery import render_gallery
