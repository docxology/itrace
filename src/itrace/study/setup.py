"""Metric display measurements bound to a browser epoch and viewport CSS pixels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .jsonio import JsonValue, StudyError, acknowledged, fields, integer, mapping, number, text


@dataclass(frozen=True, slots=True)
class Setup:
    client_epoch: str
    width: float
    height: float
    payload: bytes


def parse_setup(data: Mapping[str, JsonValue]) -> Setup:
    from .jsonio import encoded

    fields(
        data,
        "participant_id device_id display_width_m display_height_m distance_m "
        "screen_width_px screen_height_px dpr viewport_width_px viewport_height_px "
        "browser client_epoch storage_acknowledged geometry_confirmed fullscreen",
    )
    for key in ("participant_id", "device_id", "client_epoch"):
        text(data[key], key, identifier=True)
    text(data["browser"], "browser")
    for key in ("storage_acknowledged", "geometry_confirmed", "fullscreen"):
        acknowledged(data[key], key)
    for key in ("display_width_m", "display_height_m", "distance_m"):
        number(data[key], key, 0.05, 10)
    for key in ("screen_width_px", "screen_height_px"):
        integer(data[key], key, 100, 32768)
    dpr = number(data["dpr"], "dpr", 0.25, 8)
    width = number(data["viewport_width_px"], "viewport_width_px", 100, 32768)
    height = number(data["viewport_height_px"], "viewport_height_px", 100, 32768)
    if width * dpr > float(integer(data["screen_width_px"], "screen_width_px", 100, 32768)) * 8:
        raise StudyError("viewport_width_px", "inconsistent display geometry")
    return Setup(text(data["client_epoch"], "client_epoch"), width, height, encoded(data))


def rectangle(value: JsonValue, setup: Setup) -> dict[str, JsonValue]:
    data = mapping(value, "rect")
    fields(data, "x y width height")
    x = number(data["x"], "rect.x", 0, setup.width)
    y = number(data["y"], "rect.y", 0, setup.height)
    width = number(data["width"], "rect.width", 0.01, setup.width)
    height = number(data["height"], "rect.height", 0.01, setup.height)
    if x + width > setup.width + 0.01 or y + height > setup.height + 0.01:
        raise StudyError("rect", "outside prepared viewport CSS pixels")
    return {"x": x, "y": y, "width": width, "height": height}
