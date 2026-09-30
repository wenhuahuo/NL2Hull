#!/usr/bin/env python3
"""Download the three public JSONLs once, byte-for-byte, at a fixed revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dataset.jevbench import BENCHMARK_COMMIT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    files = {}
    for tier in ("original", "easy", "hard"):
        filename = f"{tier}.jsonl"
        urls = [
            f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{BENCHMARK_COMMIT}/datasets/public/{filename}",
            f"https://githubfast.com/fstandhartinger/jevbench/raw/{BENCHMARK_COMMIT}/datasets/public/{filename}",
        ]
        data = None
        for url in urls:
            try:
                with urlopen(url, timeout=60) as response:
                    data = response.read()
                break
            except (OSError, URLError):
                continue
        if data is None:
            raise RuntimeError(f"unable to download {filename} from GitHub or githubfast.com")
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        if not records:
            raise ValueError(f"empty dataset: {url}")
        (args.output / filename).write_bytes(data)
        files[filename] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "records": len(records)}
    manifest = {"repository": "https://github.com/fstandhartinger/jevbench", "commit": BENCHMARK_COMMIT, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
