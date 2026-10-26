"""Bounded JSON and exact request parsing for private studies."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Final, TypeAlias

JsonValue: TypeAlias = "bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None"
MAX_JSON: Final = 2 * 1024 * 1024


class StudyError(Exception):
    """A field-specific failure safe for the HTTP boundary to expose."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")

    def __reduce__(self) -> tuple[type[StudyError], tuple[str, str]]:
        return type(self), (self.field, self.reason)


def _pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise StudyError(key, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise StudyError("json", f"nonfinite constant {value}")


def _finite(value: JsonValue, depth: int = 0) -> None:
    if depth > 32:
        raise StudyError("json", "nesting exceeds 32")
    match value:
        case float() as number:
            if not math.isfinite(number):
                raise StudyError("json", "nonfinite number")
        case list() as values:
            for child in values:
                _finite(child, depth + 1)
        case dict() as mapping:
            for child in mapping.values():
                _finite(child, depth + 1)
        case _:
            return


def parse_object(data: bytes) -> Mapping[str, JsonValue]:
    """Parse a size/depth-bounded finite JSON object without duplicate keys."""
    if len(data) > MAX_JSON:
        raise StudyError("json", "payload exceeds 2 MiB")
    try:
        value: JsonValue = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
        _finite(value)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise StudyError("json", "invalid JSON encoding or number") from exc
    return mapping(value, "json")


def encoded(value: Mapping[str, JsonValue]) -> bytes:
    """Encode finite deterministic JSON; never emit JavaScript NaN literals."""
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise StudyError("json", "cannot encode finite JSON") from exc


def mapping(value: JsonValue, field: str) -> dict[str, JsonValue]:
    match value:
        case dict() as result:
            return result
        case _:
            raise StudyError(field, "expected object")


def sequence(value: JsonValue, field: str) -> list[JsonValue]:
    match value:
        case list() as result:
            return result
        case _:
            raise StudyError(field, "expected array")


def fields(data: Mapping[str, JsonValue], required: str, optional: str = "") -> None:
    """Reject missing/unknown fields before reading typed fields."""
    expected = set(required.split())
    if not expected <= data.keys() or data.keys() - expected - set(optional.split()):
        raise StudyError("fields", f"expected {required}; optional {optional}")


def text(value: JsonValue, field: str, *, identifier: bool = False) -> str:
    match value:
        case str() as result:
            if not result.strip() or len(result) > 512 or any(ord(c) < 32 for c in result):
                raise StudyError(field, "expected nonempty bounded text")
            if identifier and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", result):
                raise StudyError(field, "expected string identifier")
            return result
        case _:
            raise StudyError(field, "expected string")


def number(value: JsonValue, field: str, low: float = 0, high: float = 1e12) -> float:
    match value:
        case bool():
            raise StudyError(field, "boolean is not numeric")
        case int() | float():
            if not low <= value <= high:
                raise StudyError(field, "number outside finite bounds")
            return float(value)
        case _:
            raise StudyError(field, "expected number")


def integer(value: JsonValue, field: str, low: int = 1, high: int = 16384) -> int:
    match value:
        case bool():
            raise StudyError(field, "boolean is not integer")
        case int() as result:
            if not low <= result <= high:
                raise StudyError(field, "integer outside bounds")
            return result
        case _:
            raise StudyError(field, "expected exact integer")


def acknowledged(value: JsonValue, field: str) -> None:
    if value is not True:
        raise StudyError(field, "explicit true acknowledgement required")
