"""No-mocks tests for :mod:`itrace.viz.traces`.

Real synthetic data, real matplotlib objects, real PNG files written under the
pytest ``tmp_path`` fixture. Skips cleanly if matplotlib is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from itrace.synthetic import gaze_with_saccade, pupil_sine_with_blink
from itrace.types import GazeStream, PupilStream, PupilUnit

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


def test_plot_velocity_trace_returns_axes_with_lines() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import traces

    stream = _multi_saccade_stream()
    ax = traces.plot_velocity_trace(stream)
    assert isinstance(ax, Axes)
    # speed line + threshold line at least.
    assert len(ax.lines) >= 2
    assert ax.get_xlabel() == "time (s)"
    assert ax.get_ylabel() == "speed (deg/s)"


def test_plot_velocity_trace_reuses_supplied_axes() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from itrace.viz import traces

    stream = _multi_saccade_stream()
    fig, ax = plt.subplots()
    returned = traces.plot_velocity_trace(stream, ax=ax)
    assert returned is ax
    assert isinstance(returned, Axes)
    plt.close(fig)


def test_velocity_trace_shades_saccade_spans() -> None:
    from itrace.viz import traces

    stream = _multi_saccade_stream()
    ax = traces.plot_velocity_trace(stream)
    # axvspan adds Polygon patches; with real saccades there must be >=1.
    assert len(ax.patches) >= 1


def test_velocity_trace_pure_fixation_no_saccade_spans() -> None:
    from itrace.viz import traces

    # A single small saccade well below the threshold should yield no spans
    # when the threshold is very high (covers the empty-saccades branch).
    stream, _ = gaze_with_saccade(amplitude_deg=10.0, fixation_s=0.2)
    ax = traces.plot_velocity_trace(stream, velocity_threshold_deg_s=1.0e6)
    assert len(ax.patches) == 0


def test_figure_velocity_trace_writes_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import traces

    stream = _multi_saccade_stream()
    fig = traces.figure_velocity_trace(stream)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Gaze velocity trace (I-VT)"
    out = tmp_path / "velocity.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_figure_velocity_trace_forwards_threshold() -> None:
    from itrace.viz import traces

    stream = _multi_saccade_stream()
    fig = traces.figure_velocity_trace(stream, velocity_threshold_deg_s=45.0)
    ax = fig.axes[0]
    # The dashed threshold line sits at the forwarded y-value.
    ydata = [float(line.get_ydata()[0]) for line in ax.lines if line.get_linestyle() == "--"]
    assert any(abs(y - 45.0) < 1e-9 for y in ydata)


def test_plot_pupil_trace_returns_axes_with_lines() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import traces

    stream, _peaks = pupil_sine_with_blink()
    ax = traces.plot_pupil_trace(stream)
    assert isinstance(ax, Axes)
    # raw line + cleaned overlay.
    assert len(ax.lines) >= 2
    assert ax.get_xlabel() == "time (s)"
    assert ax.get_ylabel() == "pupil size (mm)"


def test_pupil_trace_shades_blink_span() -> None:
    from itrace.viz import traces

    stream, _peaks = pupil_sine_with_blink(blink_window_s=(4.0, 4.3))
    ax = traces.plot_pupil_trace(stream)
    # The blink span is an axvspan Polygon patch.
    assert len(ax.patches) >= 1


def test_pupil_trace_marks_peaks_and_troughs() -> None:
    from itrace.viz import traces

    stream, _peaks = pupil_sine_with_blink(duration_s=10.0, period_s=2.0)
    ax = traces.plot_pupil_trace(stream)
    # A multi-cycle sine yields both peak and trough scatter collections.
    assert len(ax.collections) >= 1


def test_pupil_trace_handles_no_blink_trace() -> None:
    from matplotlib.axes import Axes

    from itrace.viz import traces

    # No blink window -> no NaNs -> no blink spans (covers the empty branch).
    stream, _peaks = pupil_sine_with_blink(blink_window_s=None)
    ax = traces.plot_pupil_trace(stream)
    assert isinstance(ax, Axes)
    assert len(ax.patches) == 0
    assert len(ax.lines) >= 2


def test_pupil_trace_reuses_supplied_axes() -> None:
    import matplotlib.pyplot as plt

    from itrace.viz import traces

    stream, _peaks = pupil_sine_with_blink()
    fig, ax = plt.subplots()
    returned = traces.plot_pupil_trace(stream, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_pupil_trace_flat_signal_no_extrema(tmp_path) -> None:
    from itrace.viz import traces

    # A monotonic ramp has neither a peak nor a trough; exercises the branch
    # where the peak/trough scatter lists stay empty.
    t = np.linspace(0.0, 5.0, 300)
    size = 3.0 + 0.1 * t
    stream = PupilStream(t=t, size=size, unit=PupilUnit.MM)
    ax = traces.plot_pupil_trace(stream)
    out = tmp_path / "ramp.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0


def test_figure_pupil_trace_writes_png(tmp_path) -> None:
    from matplotlib.figure import Figure

    from itrace.viz import traces

    stream, _peaks = pupil_sine_with_blink()
    fig = traces.figure_pupil_trace(stream)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Pupil trace (cleaned overlay)"
    out = tmp_path / "pupil.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0
