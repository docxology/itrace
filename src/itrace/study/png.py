"""Bounded stdlib validation of noninterlaced 8-bit PNG scanlines."""

from __future__ import annotations

import struct
import zlib
from typing import Final

from .jsonio import StudyError

MAX_PNG: Final = 16 * 1024 * 1024
MAX_DECODED: Final = 64 * 1024 * 1024


def validate_png(data: bytes, width: int, height: int) -> int:
    """Check signature, chunks, CRCs, decompression length, and scanline filters.

    Only grayscale/RGB/gray-alpha/RGBA 8-bit, noninterlaced images are accepted.
    Filtering is bytewise invertible for these formats; every filtered byte value
    is legal, so exact decompression plus valid filter bytes proves decodability.
    """
    if len(data) > MAX_PNG or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise StudyError("stimulus", "invalid or oversized PNG signature")
    offset, chunks, channels = 8, 0, 0
    compressed = bytearray()
    idat_seen, idat_ended, ended = False, False, False
    while offset < len(data):
        if offset + 12 > len(data) or ended:
            raise StudyError("stimulus", "truncated PNG or bytes after IEND")
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data) or not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind):
            raise StudyError("stimulus", "invalid PNG chunk bounds/type")
        body = data[offset + 8 : end - 4]
        if kind[2] > 90:
            raise StudyError("stimulus", "PNG reserved chunk bit must be uppercase")
        crc = int.from_bytes(data[end - 4 : end], "big")
        if zlib.crc32(kind + body) != crc:
            raise StudyError("stimulus", "PNG CRC mismatch")
        if chunks == 0 and kind != b"IHDR":
            raise StudyError("stimulus", "IHDR must be first")
        match kind:
            case b"IHDR":
                if chunks != 0 or length != 13:
                    raise StudyError("stimulus", "duplicate/invalid IHDR")
                w, h, depth, color, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", body
                )
                if (w, h) != (width, height):
                    raise StudyError("stimulus", "PNG dimensions differ from manifest")
                if depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
                    raise StudyError("stimulus", "unsupported PNG encoding")
                channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color, 0)
                if not channels or (width * channels + 1) * height > MAX_DECODED:
                    raise StudyError("stimulus", "unsupported color or decoded image too large")
            case b"IDAT":
                if idat_ended:
                    raise StudyError("stimulus", "nonconsecutive IDAT chunks")
                idat_seen = True
                compressed.extend(body)
            case b"IEND":
                if length != 0 or not idat_seen:
                    raise StudyError("stimulus", "invalid IEND or missing IDAT")
                ended = True
            case b"PLTE":
                if idat_seen or length == 0 or length % 3 or length > 768:
                    raise StudyError("stimulus", "invalid palette")
            case _:
                if kind[0] < 97:
                    raise StudyError("stimulus", "unsupported critical PNG chunk")
        if idat_seen and kind != b"IDAT":
            idat_ended = True
        chunks += 1
        if chunks > 4096:
            raise StudyError("stimulus", "too many PNG chunks")
        offset = end
    if not ended:
        raise StudyError("stimulus", "missing IEND")
    expected = (width * channels + 1) * height
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise StudyError("stimulus", "invalid PNG deflate stream") from exc
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise StudyError("stimulus", "PNG decoded length/stream mismatch")
    stride = width * channels + 1
    if any(raw[row * stride] > 4 for row in range(height)):
        raise StudyError("stimulus", "invalid PNG scanline filter")
    return expected
