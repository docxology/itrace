"""Aggregate manifest-listed empirical sessions for v1-readiness tracking.

This script is the repeated-session complement to ``summarize_empirical_pilot``.
It reads a manifest of planned and available derived empirical-session reports,
validates the provenance/privacy boundary, and writes:

* ``docs/empirical_sessions_summary.json`` for release-readiness checks.
* ``output/figures/empirical_sessions_summary.png`` for documentation.

The current diagnostic v1 gate can be ready without public-dataset,
reference-device, or manual-annotation evidence. Those stronger validation lanes
are tracked separately as future scope.

Thin orchestrator: aggregation/validation lives in :mod:`itrace.empirical` and
the summary figure in :mod:`itrace.viz.empirical_sessions`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from itrace.empirical import (
    DEFAULT_SESSIONS_FIGURE,
    DEFAULT_SESSIONS_MANIFEST,
    DEFAULT_SESSIONS_SUMMARY,
    MANIFEST_KIND,
    write_json_atomic,
)
from itrace.empirical import (
    aggregate_empirical_sessions as core_aggregate_empirical_sessions,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / DEFAULT_SESSIONS_MANIFEST
DEFAULT_SUMMARY = ROOT / DEFAULT_SESSIONS_SUMMARY
DEFAULT_FIGURE = ROOT / DEFAULT_SESSIONS_FIGURE


def aggregate_empirical_sessions(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Return a validated empirical-session aggregate summary."""
    return core_aggregate_empirical_sessions(manifest_path=manifest_path, repo_root=repo_root)


def write_summary(summary: Mapping[str, object], out: Path = DEFAULT_SUMMARY) -> Path:
    """Write the aggregate empirical-session summary JSON."""
    return write_json_atomic(out, summary)


def generate_empirical_sessions_figure(
    summary: Mapping[str, object], out: Path = DEFAULT_FIGURE
) -> Path:
    """Render a compact v1-readiness figure for empirical-session intake."""
    from itrace.viz import empirical_sessions as _sessions

    return _sessions.generate_empirical_sessions_figure(summary, out)


def write_empirical_session_outputs(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    summary_out: Path = DEFAULT_SUMMARY,
    figure_out: Path = DEFAULT_FIGURE,
) -> dict[str, Path]:
    """Write aggregate empirical-session JSON and figure outputs."""
    from itrace.viz import empirical_sessions as _sessions

    return _sessions.write_empirical_session_outputs(
        manifest_path=manifest_path,
        summary_out=summary_out,
        figure_out=figure_out,
        repo_root=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate iTrace empirical-session manifests.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    if not args.manifest.exists():
        template = {
            "kind": MANIFEST_KIND,
            "version": 1,
            "sessions": [],
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
        print(f"missing manifest: {args.manifest}", file=sys.stderr)
        return 1
    paths = write_empirical_session_outputs(
        manifest_path=args.manifest,
        summary_out=args.summary_out,
        figure_out=args.figure_out,
    )
    for key, path in paths.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
