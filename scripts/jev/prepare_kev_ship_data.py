"""Convert structured ship-language checkpoints into Kev labelled requests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

REGIONS = {
    "bow": "艏部",
    "stern": "艉部",
    "bulb": "球鼻艏",
    "midbody": "中体",
    "deck": "甲板",
    "bilge": "舭部",
    "global": "全船",
}
OPERATIONS = {
    "outward": "外扩",
    "inward": "内收",
    "upward": "上抬",
    "downward": "下压",
    "forward": "向艏移动",
    "aftward": "向艉移动",
    "increase_fullness": "增加丰满度",
    "decrease_fullness": "降低丰满度",
    "increase_flare": "增加外飘",
    "change_bulb_length": "改变球鼻艏长度",
    "increase_length": "增加全船长度",
    "decrease_length": "缩短全船长度",
    "increase_breadth": "增加全船宽度",
    "decrease_breadth": "减小全船宽度",
}
ACTION_COUNTS = {1: "one", 2: "two", 3: "three"}
MAGNITUDE_LEVELS = ["none", "slight", "small", "moderate", "large", "very_large"]

TRAIN_HULLS = {
    "DTC",
    "DTMB5415",
    "KCS",
    "KVLCC2",
    "containership",
    "firgate",
    "npl_round_bilge_4a_model",
    "npl_round_bilge_full_scale",
}
VALIDATION_HULLS = {"s_175_u_water", "series60"}
TEST_HULLS = {"wigley_hull", "work_boat"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _question(action: dict[str, Any], index: int) -> dict[str, Any]:
    constraints = action.get("constraints") or {}
    level = action["magnitude_level"]
    mode = "qualitative" if level is not None else "explicit"
    return {
        f"region_{index}": {
            "type": "choice",
            "instructions": f"第 {index} 个动作发生在哪个船体区域？",
            "criteria": {key: value for key, value in REGIONS.items()},
            "label": action["region"],
            "src": "ship_region",
        },
        f"operation_{index}": {
            "type": "choice",
            "instructions": f"第 {index} 个动作的变形操作是什么？",
            "criteria": {key: value for key, value in OPERATIONS.items()},
            "label": action["operation"],
            "src": "ship_operation",
        },
        f"mode_{index}": {
            "type": "choice",
            "instructions": f"第 {index} 个动作使用定性强度还是显式数值？",
            "criteria": {
                "qualitative": "使用略微、适度、明显等定性表达",
                "explicit": "包含变形量或作用范围等显式数值",
            },
            "label": mode,
            "src": "ship_magnitude_mode",
        },
        f"magnitude_{index}": {
            "type": "score",
            "instructions": f"第 {index} 个动作的定性变化程度是多少？显式数值动作选择 none。",
            "criteria": MAGNITUDE_LEVELS,
            "label": 0 if level is None else int(level),
            "src": "ship_magnitude_level",
        },
        f"preserve_displacement_{index}": {
            "type": "noul",
            "instructions": f"第 {index} 个动作是否要求保持排水量不变？",
            "criteria": {
                "true": "用户明确要求保持排水量或排水体积",
                "false": "用户没有提出保持排水量的要求",
            },
            "label": bool(constraints.get("preserve_displacement", False)),
            "src": "ship_constraint",
        },
        f"preserve_deck_line_{index}": {
            "type": "noul",
            "instructions": f"第 {index} 个动作是否要求保持甲板线不变？",
            "criteria": {
                "true": "用户明确要求保持甲板线、甲板边线或甲板型线",
                "false": "用户没有提出保持甲板线的要求",
            },
            "label": bool(constraints.get("preserve_deck_line", False)),
            "src": "ship_constraint",
        },
    }


def _convert_sample(sample: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    actions = [source["actions"][index] for index in sample["action_indices"]]
    questions: dict[str, dict[str, Any]] = {
        "action_count": {
            "type": "choice",
            "instructions": "当前输入包含几个船型变形动作？",
            "criteria": {key: f"包含 {count} 个动作" for count, key in ACTION_COUNTS.items()},
            "label": ACTION_COUNTS[len(actions)],
            "src": "ship_action_count",
        }
    }
    for index, action in enumerate(actions, start=1):
        questions.update(_question(action, index))
    return {
        "state": sample["text"],
        "questions": questions,
        "_meta": {
            "sample_id": sample["sample_id"],
            "task_key": sample["task_key"],
            "hull_id": sample["hull_id"],
            "interaction_type": sample["interaction_type"],
            "source_model": sample["model"],
            "variant_index": sample["variant_index"],
        },
    }


def _split(hull_id: str) -> str:
    if hull_id in TRAIN_HULLS:
        return "train"
    if hull_id in VALIDATION_HULLS:
        return "validation"
    if hull_id in TEST_HULLS:
        return "test"
    raise ValueError(f"unassigned hull: {hull_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = {
        record["sample_id"]: record
        for record in (json.loads(line) for line in args.structured.read_text().splitlines())
    }
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError("language checkpoint is empty")

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    seen_keys: set[str] = set()
    for sample in samples:
        key = sample["task_key"]
        if key in seen_keys:
            raise ValueError(f"duplicate task_key: {key}")
        seen_keys.add(key)
        if sample["sample_id"] not in source:
            raise ValueError(f"missing structured source: {sample['sample_id']}")
        split_records[_split(sample["hull_id"])].append(
            _convert_sample(sample, source[sample["sample_id"]])
        )

    for split, records in split_records.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "revision": output.name,
        "samples": str(args.samples),
        "samples_sha256": _digest(args.samples),
        "structured": str(args.structured),
        "structured_sha256": _digest(args.structured),
        "record_count": len(samples),
        "split_counts": {split: len(records) for split, records in split_records.items()},
        "hulls": {
            split: sorted({record["_meta"]["hull_id"] for record in records})
            for split, records in split_records.items()
        },
        "questions_per_split": {
            split: sum(len(record["questions"]) for record in records)
            for split, records in split_records.items()
        },
        "models": dict(Counter(sample["model"] for sample in samples)),
        "numeric_magnitude_note": (
            "Kev evaluates explicit numeric mode and typed categorical fields; it does not regress the numeric values or ranges."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
