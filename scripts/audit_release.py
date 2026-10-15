from __future__ import annotations

import argparse
from pathlib import Path

from itrace.release import release_issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check release metadata and publication artifacts."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        issues = release_issues(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Release audit failed: {type(exc).__name__}: {exc}")
        return 1
    for issue in issues:
        print(issue)
    if not issues:
        print("Release metadata and artifact integrity passed.")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
