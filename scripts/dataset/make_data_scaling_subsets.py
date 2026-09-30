#!/usr/bin/env python3
"""Sample nested training subsets for the data-scaling experiment.

Subsets are nested: every smaller fraction is a subset of every larger one,
sampled once with a fixed seed from the shuffled record order. Each subset is
written with a SHA-256 manifest so training jobs can verify their input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    lines = [line for line in args.data.read_text().splitlines() if line.strip()]
    fractions = sorted(args.fractions)
    counts = [round(len(lines) * fraction / 100.0) for fraction in fractions]
    if any(count < 1 for count in counts) or counts[-1] >= len(lines):
        raise ValueError(f"fractions must map into [1, {len(lines)}): {counts}")

    order = list(range(len(lines)))
    random.Random(SEED).shuffle(order)
    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.data),
        "source_records": len(lines),
        "seed": SEED,
        "nested": True,
        "subsets": {},
    }
    for fraction, count in zip(fractions, counts):
        selected = sorted(order[:count])
        name = f"train_frac{fraction:g}"
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for index in selected:
                stream.write(lines[index] + "\n")
        manifest["subsets"][name] = {
            "fraction": fraction,
            "records": count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "subsets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest["subsets"], indent=2))


if __name__ == "__main__":
    main()
