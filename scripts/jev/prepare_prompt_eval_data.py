#!/usr/bin/env python3
"""Build prompt-evaluation records from the frozen ship language and action data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TRAIN_HULLS = {
    "DTC", "DTMB5415", "KCS", "KVLCC2", "containership", "firgate",
    "npl_round_bilge_4a_model", "npl_round_bilge_full_scale",
}
VALIDATION_HULLS = {"s_175_u_water", "series60"}
TEST_HULLS = {"wigley_hull", "work_boat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_hull(hull_id: str) -> str:
    if hull_id in TRAIN_HULLS:
        return "train"
    if hull_id in VALIDATION_HULLS:
        return "validation"
    if hull_id in TEST_HULLS:
        return "test"
    raise ValueError(f"unknown hull split: {hull_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source: dict[str, dict[str, Any]] = {}
    for line in args.structured.read_text().splitlines():
        record = json.loads(line)
        sample_id = record["sample_id"]
        if sample_id in source:
            raise ValueError(f"duplicate structured sample_id: {sample_id}")
        source[sample_id] = record

    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    by_turn: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_turn[sample["sample_id"]][sample["turn_index"]].append(sample)

    records: list[dict[str, Any]] = []
    for sample in samples:
        if split_for_hull(sample["hull_id"]) != args.split:
            continue
        record = source.get(sample["sample_id"])
        if record is None:
            raise ValueError(f"missing structured record: {sample['sample_id']}")
        if record["hull_id"] != sample["hull_id"]:
            raise ValueError(f"hull mismatch: {sample['task_key']}")
        action_indices = sample["action_indices"]
        actions = record["actions"]
        if any(index < 0 or index >= len(actions) for index in action_indices):
            raise ValueError(f"action index mismatch: {sample['task_key']}")
        previous_turns = []
        for turn_index in range(sample["turn_index"]):
            candidates = by_turn[sample["sample_id"]].get(turn_index, [])
            if candidates:
                previous_turns.append(sorted(candidates, key=lambda item: item["variant_index"])[0]["text"])
        records.append({
            "task_key": sample["task_key"],
            "sample_id": sample["sample_id"],
            "hull_id": sample["hull_id"],
            "turn_index": sample["turn_index"],
            "model_source": sample.get("model"),
            "augmentation_category": sample.get("augmentation_category"),
            "previous_turns": previous_turns,
            "text": sample["text"],
            "target_actions": [actions[index] for index in action_indices],
        })

    output_path = args.output / f"{args.split}.jsonl"
    output_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    manifest = {
        "revision": "kev_prompt_eval_dataset",
        "split": args.split,
        "record_count": len(records),
        "samples_sha256": sha256(args.samples),
        "structured_sha256": sha256(args.structured),
        "output_sha256": sha256(output_path),
        "hulls": sorted({record["hull_id"] for record in records}),
        "prompt_contract": "prompt-guided JSON action extraction; no Kev model or remote API",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
