"""Summarize one derived live empirical pilot export.

The script reads the derived ``experiment_report.json`` written by the live HTML
experiment export and writes two downstream artifacts:

* ``docs/empirical_pilot_metrics.json`` for manuscript token hydration.
* ``output/figures/empirical_pilot_summary.png`` for README/manuscript figures.

No raw eye video is read or written; absent pilot data produces an explicit
``available=false`` metrics file so public prose stays honest before recording.

Thin orchestrator: metrics extraction lives in :mod:`itrace.empirical` and the
summary figure in :mod:`itrace.viz.empirical_pilot`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from itrace.empirical import (
    DEFAULT_PILOT_FIGURE,
    DEFAULT_PILOT_METRICS,
    DEFAULT_PILOT_REPORT,
    pilot_metrics_from_report,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / DEFAULT_PILOT_REPORT
DEFAULT_METRICS = ROOT / DEFAULT_PILOT_METRICS
DEFAULT_FIGURE = ROOT / DEFAULT_PILOT_FIGURE


def metrics_from_report(
    report: Mapping[str, object],
    *,
    source_report: Path,
    pilot_id: str = "local_pilot_001",
) -> dict[str, object]:
    """Extract manuscript-ready metrics from a derived experiment report."""
    return pilot_metrics_from_report(
        report, source_report=source_report, repo_root=ROOT, pilot_id=pilot_id
    )


def write_empirical_pilot_outputs(
    report_path: Path = DEFAULT_REPORT,
    metrics_out: Path = DEFAULT_METRICS,
    figure_out: Path = DEFAULT_FIGURE,
    *,
    pilot_id: str = "local_pilot_001",
) -> dict[str, Path]:
    """Write metrics JSON and summary figure for the empirical pilot."""
    from itrace.viz.empirical_pilot import (
        write_empirical_pilot_outputs as _write_empirical_pilot_outputs,
    )

    return _write_empirical_pilot_outputs(
        report_path=report_path,
        metrics_out=metrics_out,
        figure_out=figure_out,
        pilot_id=pilot_id,
        repo_root=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the iTrace empirical pilot export.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--pilot-id", default="local_pilot_001")
    args = parser.parse_args()

    paths = write_empirical_pilot_outputs(
        report_path=args.report,
        metrics_out=args.metrics_out,
        figure_out=args.figure_out,
        pilot_id=args.pilot_id,
    )
    for key, path in paths.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
