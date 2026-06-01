"""Read/write gaze and pupil time-series as tabular CSV (pandas-backed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import GazeStream, PupilStream, PupilUnit


def write_gaze_csv(stream: GazeStream, path: str | Path) -> Path:
    """Write a gaze stream to ``t,x,y`` CSV. Returns the written path."""
    df = pd.DataFrame({"t": stream.t, "x": stream.x, "y": stream.y})
    out = Path(path)
    df.to_csv(out, index=False)
    return out


def read_gaze_csv(path: str | Path) -> GazeStream:
    """Read a ``t,x,y`` CSV back into a GazeStream."""
    df = pd.read_csv(path)
    missing = {"t", "x", "y"} - set(df.columns)
    if missing:
        msg = f"gaze CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)
    return GazeStream(
        t=df["t"].to_numpy(dtype=np.float64),
        x=df["x"].to_numpy(dtype=np.float64),
        y=df["y"].to_numpy(dtype=np.float64),
    )


def write_pupil_csv(stream: PupilStream, path: str | Path) -> Path:
    """Write a pupil stream to ``t,size,unit`` CSV. Returns the written path."""
    df = pd.DataFrame(
        {"t": stream.t, "size": stream.size, "unit": [stream.unit.value] * len(stream)}
    )
    out = Path(path)
    df.to_csv(out, index=False)
    return out


def read_pupil_csv(path: str | Path) -> PupilStream:
    """Read a ``t,size[,unit]`` CSV back into a PupilStream."""
    df = pd.read_csv(path)
    missing = {"t", "size"} - set(df.columns)
    if missing:
        msg = f"pupil CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)
    unit = PupilUnit(df["unit"].iloc[0]) if "unit" in df.columns and len(df) else PupilUnit.RELATIVE
    return PupilStream(
        t=df["t"].to_numpy(dtype=np.float64),
        size=df["size"].to_numpy(dtype=np.float64),
        unit=unit,
    )
