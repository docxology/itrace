from __future__ import annotations

import argparse
from pathlib import Path

from itrace.manuscript import (
    load_empirical_session_tokens as _session_tokens,
)
from itrace.manuscript import (
    load_empirical_tokens as _pilot_tokens,
)
from itrace.manuscript import (
    load_metric_tokens as _metric_tokens,
)
from itrace.manuscript import (
    render_manuscript,
    source_files,
)

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript"
DEFAULT_METRICS = ROOT / "docs" / "verification_metrics.json"
DEFAULT_EMPIRICAL_METRICS = ROOT / "docs" / "empirical_pilot_metrics.json"
DEFAULT_EMPIRICAL_SESSIONS = ROOT / "docs" / "empirical_sessions_summary.json"


def _source_files() -> list[Path]:
    return source_files(MANUSCRIPT)


def load_metric_tokens(path: Path = DEFAULT_METRICS) -> dict[str, str]:
    return _metric_tokens(path)


def load_empirical_tokens(path: Path = DEFAULT_EMPIRICAL_METRICS) -> dict[str, str]:
    return _pilot_tokens(path)


def load_empirical_session_tokens(path: Path = DEFAULT_EMPIRICAL_SESSIONS) -> dict[str, str]:
    return _session_tokens(path)


def render(values: dict[str, str], *, output_dir: Path = MANUSCRIPT) -> list[Path]:
    return render_manuscript(values, manuscript_dir=MANUSCRIPT, root=ROOT, output_dir=output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render audited iTrace manuscript sources.")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--empirical-json", type=Path, default=DEFAULT_EMPIRICAL_METRICS)
    parser.add_argument("--empirical-sessions-json", type=Path, default=DEFAULT_EMPIRICAL_SESSIONS)
    parser.add_argument("--output-dir", type=Path, default=MANUSCRIPT)
    parser.add_argument("--demo-amplitude")
    parser.add_argument("--test-count")
    parser.add_argument("--coverage-pct")
    args = parser.parse_args()
    values = load_metric_tokens(args.metrics_json)
    values.update(load_empirical_tokens(args.empirical_json))
    values.update(load_empirical_session_tokens(args.empirical_sessions_json))
    for key in ("demo_amplitude", "test_count", "coverage_pct"):
        value = getattr(args, key)
        if value is not None:
            values[key.upper()] = value
    for path in render(values, output_dir=args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
