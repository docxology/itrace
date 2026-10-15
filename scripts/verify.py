from __future__ import annotations

import argparse
from pathlib import Path

from itrace.verification import run_verification

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iTrace gates with retained command receipts.")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="New evidence directory; must not exist."
    )
    args = parser.parse_args()
    return run_verification(ROOT, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
