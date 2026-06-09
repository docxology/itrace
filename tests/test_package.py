"""Package-level / import-safety tests (ISC-2..6)."""

from __future__ import annotations

import builtins
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

import itrace

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_IMPORT_MODULES = (
    "cv2",
    "mediapipe",
    "streamlit",
    "plotly",
    "matplotlib",
    "fastapi",
    "uvicorn",
)
OPTIONAL_DEPENDENCY_IMPORTS = {
    "opencv-python": "cv2",
    "mediapipe": "mediapipe",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "matplotlib": "matplotlib",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}


def _dependency_name(requirement: str) -> str:
    for marker in ("[", "<", ">", "=", ";", "!", "~"):
        requirement = requirement.split(marker, 1)[0]
    return requirement.strip().lower()


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
        "pupilseg",
        "benchmark",
        "experiments",
        "geometry",
        "validation",
        "BinocularGazeSample",
        "EyeGazeDiagnostic",
    ):
        assert hasattr(itrace, name), name  # ISC-2


def test_no_optional_deps_imported_at_module_load() -> None:
    # ISC-5/ISC-6: importing itrace must not pull in any hardware/dashboard dep.
    before = {name for name in OPTIONAL_IMPORT_MODULES if name in sys.modules}
    importlib.reload(itrace)
    after = {name for name in OPTIONAL_IMPORT_MODULES if name in sys.modules}
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
    marker = Path(itrace.__file__).parent / "py.typed"
    assert marker.exists()


def test_httpx2_is_dev_only_test_client_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert all("httpx2" not in dependency for dependency in extras["web"])
    assert all("httpx2" not in dependency for dependency in extras["all"])
    assert any("httpx2" in dependency for dependency in extras["dev"])


def test_optional_dependency_import_safety_denylist_tracks_runtime_extras() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    runtime_extra_names = set(extras) - {"all", "dev"}
    unknown: list[str] = []
    modules: set[str] = set()
    for extra_name in sorted(runtime_extra_names):
        for requirement in extras[extra_name]:
            dependency = _dependency_name(requirement)
            module = OPTIONAL_DEPENDENCY_IMPORTS.get(dependency)
            if module is None:
                unknown.append(f"{extra_name}:{dependency}")
            else:
                modules.add(module)

    assert unknown == []
    assert modules <= set(OPTIONAL_IMPORT_MODULES)
