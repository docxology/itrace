"""Package-level / import-safety tests (ISC-2..6)."""

from __future__ import annotations

import builtins
import importlib
import subprocess
import sys

import pytest

import itrace


def test_version_exported() -> None:
    assert isinstance(itrace.__version__, str)
    assert itrace.__version__.count(".") >= 2  # ISC-2


def test_public_api_present() -> None:
    for name in (
        "GazeStream",
        "Saccade",
        "PSO",
        "PhaseDetector",
        "config",
        "pipeline",
        "geometry",
        "validation",
    ):
        assert hasattr(itrace, name), name  # ISC-2


def test_no_optional_deps_imported_at_module_load() -> None:
    # ISC-5/ISC-6: importing itrace must not pull in any hardware/dashboard dep.
    forbidden = ("cv2", "mediapipe", "streamlit", "plotly", "matplotlib", "fastapi", "uvicorn")
    before = {name for name in forbidden if name in sys.modules}
    importlib.reload(itrace)
    after = {name for name in forbidden if name in sys.modules}
    assert after == before


def test_clean_process_import_does_not_load_optional_visual_or_hardware_deps() -> None:
    code = (
        "import sys, itrace\n"
        "for name in ('cv2','mediapipe','streamlit','plotly','matplotlib','fastapi','uvicorn'):\n"
        "    assert name not in sys.modules, name\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, text=True)
    assert result.returncode == 0


def test_live_module_import_does_not_load_web_or_capture_deps() -> None:
    code = (
        "import sys, itrace.live\n"
        "for name in ('cv2','mediapipe','streamlit','plotly','matplotlib','fastapi','uvicorn'):\n"
        "    assert name not in sys.modules, name\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, text=True)
    assert result.returncode == 0


def test_live_app_missing_web_extra_has_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from itrace import live

    real_import = builtins.__import__

    def import_without_fastapi(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastapi" or name.startswith("fastapi."):
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fastapi)
    with pytest.raises(RuntimeError, match="uv sync --extra web"):
        live.create_app()


def test_py_typed_marker_present() -> None:
    # ISC-3
    from pathlib import Path

    marker = Path(itrace.__file__).parent / "py.typed"
    assert marker.exists()
