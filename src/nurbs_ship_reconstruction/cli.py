"""Command line entry point for the classic-hull benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import REVISION, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/classic_hulls"),
        help="classic hull dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / REVISION,
        help="new output directory; existing evidence is never overwritten",
    )
    parser.add_argument("--hull", action="append", help="run one hull; repeat for several")
    parser.add_argument(
        "--levels", type=int, default=33, help="number of draft waterline levels"
    )
    args = parser.parse_args()
    if args.levels < 8:
        parser.error("--levels must be at least 8")
    result = run_benchmark(
        dataset_dir=args.dataset,
        output_dir=args.output,
        selected=args.hull,
        level_count=args.levels,
    )
    print(f"completed {result['cases']} hulls: {result['output_dir']}")


if __name__ == "__main__":
    main()
