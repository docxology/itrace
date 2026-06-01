"""No-mocks tests for :mod:`itrace.viz.timeline`.

Real synthetic data, real :mod:`itrace.pipeline` output, real matplotlib
objects, and real PNG files written under the pytest ``tmp_path`` fixture.
Skips cleanly if matplotlib is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from itrace import pipeline
from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink
from itrace.types import GazeStream, PupilStream, PupilUnit, SessionReport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib (viz extra) not installed",
)


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    """Concatenate several synthetic saccades into one multi-event stream."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(4.0, 18.0, 6):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


def _session() -> SessionReport:
    """A deterministic analysed session with both fixations and saccades."""
    gaze = _multi_saccade_stream()
    pstream, _ = pupil_sine_with_blink()
    return pipeline.analyze_session(gaze, pstream)


# --------------------------------------------------------------------------- #
# Event raster
# --------------------------------------------------------------------------- #
def test_plot_event_raster_returns_axes_with_bars() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    report = _session()
    assert report.fixations  # the session must exercise the fixation row
    assert report.saccades  # ... and the saccade row
    ax = timeline.plot_event_raster(report)
    assert isinstance(ax, Axes)
    # broken_barh adds BrokenBarHCollection / PolyCollection objects.
    assert len(ax.collections) >= 1
    assert ax.get_xlabel() == "time (s)"
    assert list(ax.get_yticklabels()[0].get_text()) == list("saccades")


def test_plot_event_raster_reuses_supplied_axes() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    report = _session()
    fig, ax = plt.subplots()
    returned = timeline.plot_event_raster(report, ax=ax)
    assert returned is ax
    assert isinstance(returned, Axes)
    plt.close(fig)


def test_plot_event_raster_empty_report_draws_no_bars() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    report = SessionReport(n_samples=0, duration_s=0.0, fixations=[], saccades=[])
    ax = timeline.plot_event_raster(report)
    assert isinstance(ax, Axes)
    assert len(ax.collections) == 0
    # No events -> no legend was created.
    assert ax.get_legend() is None


def test_figure_event_raster_writes_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import timeline

    report = _session()
    fig = timeline.figure_event_raster(report)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Oculomotor event raster"
    out = tmp_path / "raster.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Rate trace
# --------------------------------------------------------------------------- #
def test_plot_rate_returns_axes_with_step_line() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    times = np.linspace(0.0, 5.0, 11)
    rates = np.abs(np.sin(times)) * 3.0
    ax = timeline.plot_rate(times, rates, label="saccade rate")
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 1
    assert ax.get_xlabel() == "time (s)"
    assert ax.get_ylabel() == "rate (1/s)"
    labels = [line.get_label() for line in ax.lines]
    assert "saccade rate" in labels


def test_plot_rate_reuses_supplied_axes() -> None:
    import matplotlib.pyplot as plt

    from itrace.viz import timeline

    fig, ax = plt.subplots()
    returned = timeline.plot_rate([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_rate_empty_series_draws_no_line() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    ax = timeline.plot_rate([], [])
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 0
    assert ax.get_legend() is None


def test_plot_rate_length_mismatch_raises() -> None:
    from itrace.viz import timeline

    with pytest.raises(ValueError, match="equal length"):
        timeline.plot_rate([0.0, 1.0, 2.0], [1.0, 2.0])


def test_figure_rate_writes_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import timeline

    times = np.linspace(0.0, 3.0, 7)
    rates = np.linspace(1.0, 4.0, 7)
    fig = timeline.figure_rate(times, rates, label="fix rate")
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Event rate"
    out = tmp_path / "rate.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Pupil PSD
# --------------------------------------------------------------------------- #
def test_plot_pupil_psd_returns_log_log_axes() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    stream, _peaks = pupil_sine_with_blink()
    ax = timeline.plot_pupil_psd(stream)
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 1
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "frequency (Hz)"


def test_plot_pupil_psd_reuses_supplied_axes() -> None:
    import matplotlib.pyplot as plt

    from itrace.viz import timeline

    stream, _peaks = pupil_sine_with_blink()
    fig, ax = plt.subplots()
    returned = timeline.plot_pupil_psd(stream, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_pupil_psd_short_trace_no_data_branch() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    # Fewer than three samples cannot yield a Welch spectrum -> "no data".
    stream = PupilStream(t=np.array([0.0, 0.1]), size=np.array([3.0, 3.1]), unit=PupilUnit.MM)
    ax = timeline.plot_pupil_psd(stream)
    assert isinstance(ax, Axes)
    # The "no data" annotation is drawn as text and no spectral line.
    assert len(ax.lines) == 0
    assert ax.get_xscale() == "linear"


def test_plot_pupil_psd_flat_signal_still_draws_spectrum() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import timeline

    # A nearly-constant signal still has (tiny, float-noise) positive power at
    # the non-DC bins, so the positive-power mask keeps at least one bin and the
    # log-log spectrum is drawn without error.
    t = np.linspace(0.0, 10.0, 600)
    size = np.full(t.size, 3.0)
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)
    ax = timeline.plot_pupil_psd(stream)
    assert isinstance(ax, Axes)
    assert len(ax.lines) >= 1
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_figure_pupil_psd_writes_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import timeline

    stream, _peaks = pupil_sine_with_blink()
    fig = timeline.figure_pupil_psd(stream)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Pupil power spectral density"
    out = tmp_path / "psd.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0
