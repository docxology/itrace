"""Read checksum-pinned values from the public 2026-10-16 reader edition."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

EDITION_DOI = "10.5281/zenodo.23419306"
READER = "docs/releases/retained-render/_build.md"
READER_SHA256 = "8dee2750abb4cb4560a2e5957188dfe32733466a541b3aa315dfdd9a19925d48"
TOKEN_SHA256 = "af78585b1c5b4a9d6c3217e57a99b96d3e072bb0783164ae870ac8f9f3867898"
FIGURES = {
    "empirical_pilot_summary.png": (
        "712549a5634c1b16a1579fee68433d663a23f0c25d5028f460511f766863cbc9"
    ),
    "empirical_sessions_summary.png": (
        "586c2b961a9e846cd2d722e26df5d26d16174b194963fcba59f9c3072df42f25"
    ),
    "synthetic_empirical_range_bridge.png": (
        "70b29999ed87a1741460c5335d0ade99bab30e9266d0e99a99b664e378171f82"
    ),
    "statistical_interpretation_ledger.png": (
        "8852f1578e18a3b5dfc7d43b0c35ff2055700bf44811f877386d77fc256d7351"
    ),
}
TOKENS = frozenset(
    {
        "EMPIRICAL_PILOT_STATUS",
        "EMPIRICAL_PILOT_FINITE_GAZE",
        "EMPIRICAL_PILOT_SAMPLING_HZ",
        "EMPIRICAL_PILOT_SAMPLING_CV",
        "EMPIRICAL_PILOT_DRIFT",
        "EMPIRICAL_PILOT_HELDOUT_RMS",
        "EMPIRICAL_PILOT_LATENCY",
        "EMPIRICAL_SESSIONS_STATUS",
        "EMPIRICAL_SESSIONS_READINESS",
        "EMPIRICAL_SESSIONS_MIN_ADDITIONAL_ALL",
        "EMPIRICAL_SESSIONS_CONDITIONS_REMAINING",
        "EMPIRICAL_SESSIONS_FUTURE_SCOPE",
    }
)
_SECTION_HEADING = "# Results: single-pilot empirical diagnostics"
_TOKEN_PATTERN = re.compile(r"\{\{(EMPIRICAL_[A-Z0-9_]+)\}\}")


def _read_pinned(root: Path, relative: str, checksum: str) -> bytes:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Public edition artifact resolves outside the checkout: {relative}")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError(f"Public edition checksum mismatch: {relative}")
    return raw


def _edition_tokens(source: str, reader: str) -> dict[str, str]:
    """Recover only bold prose bindings in the corresponding released section."""
    sections = re.split(r"(?m)(?=^# )", reader)
    released = [section for section in sections if section.startswith(_SECTION_HEADING)]
    if len(released) != 1:
        raise ValueError("Expected one empirical diagnostics section in the retained reader")
    paragraph_text = " ".join(released[0].split())
    source_text = " ".join(source.split())
    values: dict[str, str] = {}
    for match in _TOKEN_PATTERN.finditer(source_text):
        key = match[1]
        before, after = source_text[: match.start()].split(), source_text[match.end() :].split()
        # Later appearances in tables reuse the value recovered from emphasized prose.
        if (
            not before
            or not after
            or not before[-1].endswith("**")
            or not after[0].startswith("**")
        ):
            continue
        if key not in TOKENS:
            raise ValueError(f"Unapproved published reader token: {key}")
        candidates = set()
        for width in (8, 6, 4, 3, 2):
            left, right = " ".join(before[-width:]), " ".join(after[:width])
            pattern = re.escape(left) + r"(.{1,500}?)" + re.escape(right)
            candidates.update(
                item for item in re.findall(pattern, paragraph_text) if "**" not in item
            )
        if len(candidates) != 1:
            raise ValueError(f"Expected one published reader binding for {key}")
        values[key] = candidates.pop()
    if values.keys() != TOKENS:
        raise ValueError("Published reader token set differs from the allowlist")
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if sha256(serialized.encode("utf-8")).hexdigest() != TOKEN_SHA256:
        raise ValueError("Published reader token checksum mismatch")
    return values


def load_published_reader(root: Path, manuscript: Path) -> dict[str, str]:
    """Verify the existing reader artifacts and recover the used manuscript values."""
    reader = _read_pinned(root, READER, READER_SHA256).decode("utf-8")
    for filename, checksum in FIGURES.items():
        _read_pinned(root, f"output/figures/{filename}", checksum)
    source = (manuscript / "03ez_empirical_pilot.md").read_text(encoding="utf-8")
    return _edition_tokens(source, reader)
