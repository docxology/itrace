import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryFile

from .jsonio import JsonValue, StudyError
from .manifest import load_manifest
from .paths import validate_private_root
from .provenance import provenance


def check_study(
    manifest_path: Path, output_root: Path, *, require_runtime: bool = True
) -> dict[str, JsonValue]:
    manifest = load_manifest(manifest_path)
    target = validate_private_root(output_root)
    missing = [
        name
        for name in ("fastapi", "uvicorn", "cv2", "mediapipe")
        if importlib.util.find_spec(name) is None
    ]
    if require_runtime and missing:
        raise StudyError("dependencies", "install web/capture extras: " + ", ".join(missing))
    try:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryFile(dir=target) as stream:
            stream.write(b"itrace-storage-probe")
            stream.flush()
            os.fsync(stream.fileno())
        free = shutil.disk_usage(target).free
    except OSError as error:
        raise StudyError("output_root", "private storage is not writable") from error
    limits = manifest.limits
    dwell_seconds = 19 * (limits.settle_s + limits.collect_s)
    deadline_seconds = math.fsum(trial.deadline_s + trial.rest_s for trial in manifest.trials)
    return {
        "schema": "itrace.study-preflight",
        "version": 1,
        "manifest_sha256": manifest.digest,
        "practice_trials": sum(trial.role == "practice" for trial in manifest.trials),
        "scored_trials": sum(trial.role == "scored" for trial in manifest.trials),
        "unique_images": len({trial.stimulus.sha256 for trial in manifest.trials}),
        "task_ids": list[JsonValue](sorted({trial.task_id for trial in manifest.trials})),
        "timed_protocol_seconds": dwell_seconds + deadline_seconds,
        "duration_scope": "declared dwell/deadline/rest sum; setup and participant pauses add time",
        "output_root": str(target),
        "storage_probe_passed": True,
        "disk_free_bytes": free,
        "missing_runtime_modules": list[JsonValue](missing),
        "camera_probed": False,
        "operator_actions": [
            "measure physical geometry",
            "start webcam",
            "pass held-out calibration",
        ],
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "provenance": provenance(),
        "scope": "software/input/storage checks only; no participant or device validation",
    }
