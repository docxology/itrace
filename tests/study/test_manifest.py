from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from itrace.study import StudyError, load_manifest, parse_object
from itrace.study.jsonio import JsonValue, encoded, fields, integer, mapping, number, sequence, text
from itrace.study.png import validate_png

from .helpers import chunk, manifest_data, png, write_manifest


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e999}',
        b"[]",
        b"{",
        b"\xff",
        b'{"a":' + b"[" * 40 + b"0" + b"]" * 40 + b"}",
        b" " * (2 * 1024 * 1024 + 1),
    ],
)
def test_parse_rejects_malformed_when_untrusted(raw: bytes) -> None:
    with pytest.raises(StudyError):
        parse_object(raw)


def test_parse_retains_nested_finite_values_when_valid() -> None:
    value = parse_object(b'{"a":[1,2.5,true,null,{"b":"x"}]}')
    assert value == {"a": [1, 2.5, True, None, {"b": "x"}]}


def test_error_preserves_fields_when_pickled() -> None:
    original = StudyError("clock", "bad")
    recovered = pickle.loads(pickle.dumps(original))
    assert (recovered.field, recovered.reason, str(recovered)) == ("clock", "bad", "clock: bad")


@pytest.mark.parametrize("value", [True, None, "1", [], {}, float("nan"), float("inf"), -1])
def test_number_rejects_when_not_bounded_number(value: JsonValue) -> None:
    with pytest.raises(StudyError):
        number(value, "number")


@pytest.mark.parametrize("value", [True, 1.0, "1", 0, 20000])
def test_integer_rejects_when_not_exact_dimension(value: JsonValue) -> None:
    with pytest.raises(StudyError):
        integer(value, "dimension")


@pytest.mark.parametrize("value", [None, 1, "", "\n", "x" * 513, "bad/id"])
def test_identifier_rejects_when_invalid(value: JsonValue) -> None:
    with pytest.raises(StudyError):
        text(value, "id", identifier=True)


def test_structural_helpers_reject_when_wrong_shape() -> None:
    for operation in (
        lambda: mapping([], "x"),
        lambda: sequence({}, "x"),
        lambda: fields({"extra": None}, "required"),
        lambda: encoded({"bad": float("nan")}),
    ):
        with pytest.raises(StudyError):
            operation()


def test_manifest_freezes_bytes_and_hides_keys_when_valid(tmp_path: Path) -> None:
    path = write_manifest(tmp_path)
    manifest = load_manifest(path)
    original = manifest.stimulus_bytes("trial0")
    (tmp_path / "image0.png").write_bytes(b"changed")
    path.write_bytes(b"changed")
    assert manifest.stimulus_bytes("trial0") == original
    assert "correct_key" not in json.dumps(manifest.public_payload())
    assert manifest.digest == hashlib.sha256(manifest.original_bytes).hexdigest()
    assert "image0.png" not in json.dumps(manifest.public_payload())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("version", 2),
        ("schema", "other"),
        ("study_id", 42),
        ("title", ""),
        ("extra", 1),
        ("trials", []),
        ("trials", [None] * 1025),
        ("limits", {}),
    ],
)
def test_manifest_rejects_when_root_contract_violated(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    data = manifest_data(tmp_path)
    data[field] = value
    with pytest.raises(StudyError):
        load_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "task"),
        ("role", True),
        ("query", ""),
        ("trial_id", 3),
        ("deadline_s", True),
        ("deadline_s", -1),
        ("options", []),
        ("options", [None, None]),
        ("correct_key", "z"),
        ("options", [{"key": "aa", "label": "A"}, {"key": "b", "label": "B"}]),
        ("options", [{"key": "a", "label": "A"}, {"key": "a", "label": "B"}]),
    ],
)
def test_manifest_rejects_when_trial_contract_violated(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    data = manifest_data(tmp_path)
    trials = sequence(data["trials"], "trials")
    mapping(trials[0], "trial")[field] = value
    with pytest.raises(StudyError):
        load_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    "change", ["family", "bytes", "duplicate", "no_scored", "practice_after", "binding"]
)
def test_manifest_rejects_when_partition_or_order_leaks(tmp_path: Path, change: str) -> None:
    data = manifest_data(tmp_path)
    trials = sequence(data["trials"], "trials")
    first, second, third = (mapping(item, "trial") for item in trials)
    match change:
        case "family":
            second["family_id"] = first["family_id"]
        case "bytes":
            second["stimulus"] = first["stimulus"]
        case "duplicate":
            second["trial_id"] = first["trial_id"]
        case "no_scored":
            second["role"] = third["role"] = "practice"
        case "practice_after":
            third["role"] = "practice"
        case "binding":
            third["stimulus"] = dict(mapping(second["stimulus"], "stimulus"))
            mapping(third["stimulus"], "stimulus")["width"] = 17
    with pytest.raises(StudyError):
        load_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    "path",
    [
        "../image0.png",
        "/image0.png",
        "a/../image0.png",
        "https://x/a.png",
        "image0.jpg",
        "./image0.png",
        "x\\image0.png",
        "missing.png",
    ],
)
def test_manifest_rejects_when_stimulus_path_unsafe(tmp_path: Path, path: str) -> None:
    data = manifest_data(tmp_path)
    first = mapping(sequence(data["trials"], "trials")[0], "trial")
    mapping(first["stimulus"], "stimulus")["path"] = path
    with pytest.raises(StudyError):
        load_manifest(write_manifest(tmp_path, data))


def test_manifest_rejects_when_symlink_or_mutated_hash(tmp_path: Path) -> None:
    data = manifest_data(tmp_path)
    (tmp_path / "alias.png").symlink_to(tmp_path / "image0.png")
    first = mapping(sequence(data["trials"], "trials")[0], "trial")
    mapping(first["stimulus"], "stimulus")["path"] = "alias.png"
    with pytest.raises(StudyError, match="symlink"):
        load_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value"), [("sha256", "a"), ("sha256", "f" * 64), ("width", 17), ("height", True)]
)
def test_manifest_rejects_when_stimulus_binding_invalid(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    data = manifest_data(tmp_path)
    first = mapping(sequence(data["trials"], "trials")[0], "trial")
    mapping(first["stimulus"], "stimulus")[field] = value
    with pytest.raises(StudyError):
        load_manifest(write_manifest(tmp_path, data))


@pytest.mark.parametrize(
    "raw",
    [
        b"bad",
        png()[:-1],
        png() + b"extra",
        png()[:-12],
        png(0, 5),
        png()[:33] + chunk(b"IDAT", b"broken") + chunk(b"IEND", b""),
        png()[:33] + chunk(b"ABCD", b"") + png()[33:],
        png()[:33] + chunk(b"IEND", b""),
        png()[:33] + png()[:33] + png()[33:],
        png()[:45] + b"badcrc" + png()[51:],
        png()[:33] + chunk(b"IDAT", __import__("zlib").compress(b"")) + chunk(b"IEND", b""),
    ],
)
def test_png_rejects_when_integrity_or_decode_invalid(raw: bytes) -> None:
    with pytest.raises(StudyError):
        validate_png(raw, 16, 16)


@pytest.mark.parametrize("filter_byte", range(5))
def test_png_accepts_when_scanline_filter_supported(filter_byte: int) -> None:
    validate_png(png(20, filter_byte), 16, 16)


def test_unknown_stimulus_rejects_when_id_unbound(tmp_path: Path) -> None:
    manifest = load_manifest(write_manifest(tmp_path))
    with pytest.raises(StudyError, match="unknown trial"):
        manifest.stimulus_bytes("other")


def test_missing_manifest_reports_typed_storage_error_when_absent(tmp_path: Path) -> None:
    with pytest.raises(StudyError, match="cannot read"):
        load_manifest(tmp_path / "absent.json")


def test_png_rejects_when_reserved_chunk_bit_invalid() -> None:
    raw = png()[:33] + chunk(b"abca", b"") + png()[33:]
    with pytest.raises(StudyError, match="reserved chunk bit"):
        validate_png(raw, 16, 16)


@pytest.mark.parametrize("color", [0, 1, 3, 5, 7])
def test_png_rejects_when_color_or_scanline_format_invalid(color: int) -> None:
    import struct

    raw = (
        png()[:8] + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, color, 0, 0, 0)) + png()[33:]
    )
    with pytest.raises(StudyError):
        validate_png(raw, 16, 16)


def test_png_rejects_when_interlaced_without_decoder() -> None:
    import struct

    raw = png()[:8] + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 1)) + png()[33:]
    with pytest.raises(StudyError, match="unsupported PNG"):
        validate_png(raw, 16, 16)


def test_png_accepts_when_optional_palette_and_ancillary_chunks_valid() -> None:
    raw = (
        png()[:33] + chunk(b"PLTE", b"\x00\x00\x00") + chunk(b"tEXt", b"tag\x00value") + png()[33:]
    )
    validate_png(raw, 16, 16)


def test_repeated_scene_count_disclosed_when_within_role(tmp_path: Path) -> None:
    data = manifest_data(tmp_path)
    trials = sequence(data["trials"], "trials")
    mapping(trials[2], "trial")["stimulus"] = mapping(trials[1], "trial")["stimulus"]
    manifest = load_manifest(write_manifest(tmp_path, data))
    scored = mapping(
        mapping(manifest.public_payload()["scene_counts"], "counts")["scored"], "scored"
    )
    assert (scored["trials"], scored["unique_stimulus_hashes"]) == (2, 1)
