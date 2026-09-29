#!/usr/bin/env python3
"""Freeze a stratified, reproducible 5k subset of the existing Kev ship test split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

EXPECTED_TEST_SHA256 = "88f2797cf8a99b8863a591c0b8335dd962dfa289c5aaf01129cba3d579ea7c5e"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stratum(record: dict) -> tuple[str, str, str, bool]:
    questions = record["questions"]
    return (
        record["_meta"]["hull_id"],
        record["_meta"]["source_model"],
        questions["action_count"]["label"],
        any(q["label"] == "explicit" for qid, q in questions.items() if qid.startswith("mode_")),
    )


def freeze(source: Path, target: Path, manifest: Path, size: int = 5000, seed: int = 20260925) -> dict:
    if target.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite frozen subset or manifest")
    actual_sha = digest(source)
    if actual_sha != EXPECTED_TEST_SHA256:
        raise ValueError(f"unexpected test split SHA-256: {actual_sha}")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if size < 1 or size > len(records):
        raise ValueError("invalid subset size")
    keys = [record["_meta"]["task_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task_key in test split")
    buckets: dict[tuple[str, str, str, bool], list[dict]] = defaultdict(list)
    for record in records:
        buckets[stratum(record)].append(record)
    quotas = {key: (size * len(group)) // len(records) for key, group in buckets.items()}
    remaining = size - sum(quotas.values())
    # Largest-remainder apportionment, with a stable lexical tie-break.
    remainders = sorted(
        buckets,
        key=lambda key: (-(size * len(buckets[key]) % len(records)), key),
    )
    for key in remainders[:remaining]:
        quotas[key] += 1
    chosen_keys = set()
    for key, group in buckets.items():
        ranked = sorted(group, key=lambda record: hashlib.sha256(
            f"{seed}:{record['_meta']['task_key']}".encode()
        ).hexdigest())
        chosen_keys.update(record["_meta"]["task_key"] for record in ranked[:quotas[key]])
    selected = [record for record in records if record["_meta"]["task_key"] in chosen_keys]
    if len(selected) != size:
        raise AssertionError("subset count mismatch")
    target.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected))
    result = {
        "revision": "kev_ship_test_5k",
        "sampling": "stratified proportional largest-remainder; SHA-256 task-key rank within strata; original record order preserved",
        "strata": ["hull_id", "source_model", "action_count", "any_explicit_magnitude"],
        "seed": seed,
        "source": str(source),
        "source_sha256": actual_sha,
        "source_records": len(records),
        "source_questions": sum(len(record["questions"]) for record in records),
        "subset": str(target),
        "subset_sha256": digest(target),
        "subset_records": size,
        "subset_questions": sum(len(record["questions"]) for record in selected),
        "task_keys_sha256": hashlib.sha256("\n".join(record["_meta"]["task_key"] for record in selected).encode()).hexdigest(),
        "stratum_counts": [
            {"hull_id": key[0], "source_model": key[1], "action_count": key[2],
             "any_explicit_magnitude": key[3], "source_count": len(buckets[key]), "subset_count": quotas[key]}
            for key in sorted(buckets)
        ],
        "note": "Use this same frozen subset for every model; group-level uncertainty by structured sample_id, not by question.",
    }
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.source, args.target, args.manifest)
    print(json.dumps({key: result[key] for key in (
        "source_sha256", "subset_sha256", "source_records", "subset_records",
        "source_questions", "subset_questions", "task_keys_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
