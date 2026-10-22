"""Freeze local manifest and PNG bytes at the private-study trust boundary."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from .jsonio import (
    MAX_JSON,
    JsonValue,
    StudyError,
    fields,
    integer,
    mapping,
    number,
    parse_object,
    sequence,
    text,
)
from .model import Limits, Option, Stimulus, StudyManifest, Trial
from .png import MAX_PNG, validate_png


def read_bounded(path: Path, limit: int) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise StudyError("storage", f"cannot read {path.name}") from exc
    if len(data) > limit:
        raise StudyError("storage", "file exceeds allocation bound")
    return data


def _limits(value: JsonValue) -> Limits:
    data = mapping(value, "limits")
    fields(
        data,
        "max_sync_error_s sync_max_age_s max_event_lag_s "
        "max_validation_error_fraction settle_s collect_s min_samples min_valid_fraction",
    )
    return Limits(
        number(data["max_sync_error_s"], "max_sync_error_s", 0.001, 0.25),
        number(data["sync_max_age_s"], "sync_max_age_s", 1, 600),
        number(data["max_event_lag_s"], "max_event_lag_s", 0.05, 5),
        number(data["max_validation_error_fraction"], "max_validation_error_fraction", 0.001, 0.15),
        number(data["settle_s"], "settle_s", 0.5, 5),
        number(data["collect_s"], "collect_s", 1, 10),
        integer(data["min_samples"], "min_samples", 10, 10000),
        number(data["min_valid_fraction"], "min_valid_fraction", 0.6, 1),
    )


def _stimulus(value: JsonValue, root: Path, cache: dict[str, Stimulus]) -> Stimulus:
    data = mapping(value, "stimulus")
    fields(data, "path sha256 width height")
    relative = text(data["path"], "path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or ":" in relative
        or pure.suffix.lower() != ".png"
        or relative != pure.as_posix()
    ):
        raise StudyError("path", "expected canonical relative PNG path")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise StudyError("path", "symlinks forbidden")
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise StudyError("path", "stimulus is not a local file")
    digest = text(data["sha256"], "sha256")
    if not re.fullmatch("[a-f0-9]{64}", digest):
        raise StudyError("sha256", "expected lowercase SHA256")
    width = integer(data["width"], "width")
    height = integer(data["height"], "height")
    if relative in cache:
        previous = cache[relative]
        if (digest, width, height) != (previous.sha256, previous.width, previous.height):
            raise StudyError("stimulus", "conflicting repeated binding")
        return previous
    raw = read_bounded(current, MAX_PNG)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise StudyError("sha256", "stimulus bytes differ from hash")
    decoded_bytes = validate_png(raw, width, height)
    result = Stimulus(relative, digest, width, height, raw, decoded_bytes)
    cache[relative] = result
    if sum(len(item.data) for item in cache.values()) > 64 * 1024 * 1024:
        raise StudyError("stimulus", "total PNG allocation exceeds 64 MiB")
    if sum(item.decoded_bytes for item in cache.values()) > 256 * 1024 * 1024:
        raise StudyError("stimulus", "total decoded PNG allocation exceeds 256 MiB")
    return result


def _trial(value: JsonValue, root: Path, cache: dict[str, Stimulus]) -> Trial:
    data = mapping(value, "trial")
    fields(
        data,
        "trial_id role block_id task_id family_id stimulus query options correct_key deadline_s",
        "rest_s",
    )
    role = text(data["role"], "role")
    if role not in {"practice", "scored"}:
        raise StudyError("role", "expected practice or scored")
    options = sequence(data["options"], "options")
    if not 2 <= len(options) <= 12:
        raise StudyError("options", "expected 2..12 options")
    parsed: list[Option] = []
    for value in options:
        option = mapping(value, "option")
        fields(option, "key label")
        key = text(option["key"], "key", identifier=True)
        if len(key) != 1 or key not in "abcdefghijklmnopqrstuvwxyz0123456789":
            raise StudyError("key", "expected single lowercase letter or digit")
        parsed.append(Option(key, text(option["label"], "label")))
    keys = [option.key for option in parsed]
    correct_key = text(data["correct_key"], "correct_key")
    if len(set(keys)) != len(keys) or correct_key not in keys:
        raise StudyError("options", "duplicate option or unbound correct_key")
    return Trial(
        text(data["trial_id"], "trial_id", identifier=True),
        role,
        text(data["block_id"], "block_id", identifier=True),
        text(data["task_id"], "task_id", identifier=True),
        text(data["family_id"], "family_id", identifier=True),
        _stimulus(data["stimulus"], root, cache),
        text(data["query"], "query"),
        tuple(parsed),
        correct_key,
        number(data["deadline_s"], "deadline_s", 0.1, 600),
        number(data.get("rest_s", 0), "rest_s", 0, 3600),
    )


def load_manifest(path: Path) -> StudyManifest:
    """Read and bind an exact private manifest without optional image libraries."""
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise StudyError("path", "manifest and ancestors must not be symlinks")
    raw = read_bounded(path, MAX_JSON)
    data: Mapping[str, JsonValue] = parse_object(raw)
    fields(data, "schema version study_id title limits trials")
    if data["schema"] != "itrace.private-study" or integer(data["version"], "version", 1, 1) != 1:
        raise StudyError("schema", "unsupported manifest version/schema")
    values = sequence(data["trials"], "trials")
    if not 1 <= len(values) <= 1024:
        raise StudyError("trials", "expected 1..1024 frozen trials")
    cache: dict[str, Stimulus] = {}
    trials = tuple(_trial(value, path.parent, cache) for value in values)
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise StudyError("trial_id", "duplicate trial")
    if not any(trial.role == "scored" for trial in trials):
        raise StudyError("trials", "at least one scored trial required")
    practice = [trial for trial in trials if trial.role == "practice"]
    scored = [trial for trial in trials if trial.role == "scored"]
    if {trial.family_id for trial in practice} & {trial.family_id for trial in scored}:
        raise StudyError("family_id", "practice/scored families must be disjoint")
    if {trial.stimulus.sha256 for trial in practice} & {trial.stimulus.sha256 for trial in scored}:
        raise StudyError("stimulus", "practice/scored stimulus bytes must be disjoint")
    seen_scored = False
    for trial in trials:
        if seen_scored and trial.role == "practice":
            raise StudyError("role", "practice must precede scored trials")
        seen_scored = seen_scored or trial.role == "scored"
    return StudyManifest(
        text(data["study_id"], "study_id", identifier=True),
        text(data["title"], "title"),
        trials,
        _limits(data["limits"]),
        hashlib.sha256(raw).hexdigest(),
        raw,
    )
