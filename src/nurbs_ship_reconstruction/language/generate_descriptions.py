"""Generate Chinese descriptions for structured hull deformation actions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..agents.pi_runner import PI_MODEL, run_pi_text

REVISION = "v028_natural_language_action_dataset_100_qualitative"
SOURCE_DATASET = "outputs/v025_structured_action_dataset_100_all_hulls"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _source_digest(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _turn_prompt(
    record: dict[str, Any],
    turn_index: int,
    action_indices: list[int],
) -> str:
    actions = [record["actions"][index] for index in action_indices]
    context = ""
    if turn_index > 0:
        previous = record["turns"][turn_index - 1]
        previous_actions = [
            record["actions"][index] for index in previous["output_actions"]
        ]
        context = (
            "这是同一部位连续编辑的后续输入。前一轮动作如下，当前描述只表达本轮动作，"
            f"可以使用‘再’等衔接词：\n{_json(previous_actions)}\n"
        )
    if len(actions) > 1:
        instruction = "一次输入中包含多个部位的修改，请在一句话中完整表达所有动作。"
    else:
        instruction = "请只表达当前动作。"
    return f"""你正在为船型变形数据集生成一条中文用户输入描述。
{context}{instruction}

必须遵守：
1. 只返回一条自然、简洁的中文用户句子，不要返回 JSON、解释、标题、项目符号或引号。
2. 保留动作的区域、操作、相对变形大小和约束含义，不得增加输入 JSON 中没有的工程目标。
3. 对有内部强度编码的动作，只使用自然的模糊表达：1=略微，2=稍微，3=适度，4=明显，5=显著。禁止在输出中出现内部编码、数字强度、幅度、等级、级别、强度或档位等词语。
4. 如果 magnitude_level 为 null，必须准确表达 magnitude_value、longitudinal_extent 和 vertical_extent 中的数值；这种情况下可以使用“变形量”或“数值”。
5. constraints 中的 preserve_displacement、preserve_deck_line 等约束必须在句子中体现。
6. 不要引入米、毫米等物理单位；当前数据中的数值使用归一化参数。
7. 区域术语固定为：bow=艏部，stern=艉部，bulb=球鼻艏，midbody=中体，deck=甲板，bilge=舭部，global=全船。
8. 方向术语固定为：forward=向艏移动，aftward=向艉移动，upward=上抬，downward=下压，outward=外扩，inward=内收。
9. change_bulb_length 表示“按给定数值改变球鼻艏长度”，magnitude_value 是改变量，不能描述成最终长度。

船型：{record["hull_id"]}
交互类型：{record["interaction_type"]}
当前动作 JSON：
{_json(actions)}
"""


def _validate_description(
    description: str,
    actions: list[dict[str, Any]],
) -> str:
    value = " ".join(description.split())
    if not value:
        raise ValueError("generated description is empty")
    if value.startswith("```") or value.startswith("{") or value.startswith("["):
        raise ValueError("generated description is not plain natural language")
    forbidden = ("幅度", "等级", "级别", "强度", "档位")
    if any(term in value for term in forbidden):
        raise ValueError("generated description exposes internal intensity coding")
    if all(action["magnitude_level"] is None for action in actions):
        for action in actions:
            values = [
                action["magnitude_value"],
                *action["longitudinal_extent"],
                *action["vertical_extent"],
            ]
            if any(str(number) not in value for number in values):
                raise ValueError("explicit action values are missing from description")
    return value


def generate_language_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Generate one description per input turn from the stage-two records."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source_path = source_dir / "structured_actions.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    if len(records) != 100:
        raise ValueError(f"expected 100 source records, found {len(records)}")

    manifest = {
        "revision": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dir),
        "source_records": str(source_path),
        "source_sha256": _source_digest(records),
        "sample_count": len(records),
        "description_count_target": sum(len(record["turns"]) for record in records),
        "description_count_completed": 0,
        "provider_framework": "pi print mode",
        "model": PI_MODEL,
        "no_extensions": True,
        "no_tools": True,
        "job_id": None,
        "status": "running",
        "output_records": str(output_dir / "language_action_records.jsonl"),
        "failures": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    output_records = []
    for record_index, record in enumerate(records):
        enriched = dict(record)
        enriched["language"] = {
            "language": "zh-CN",
            "turns": [],
        }
        for turn_index, turn in enumerate(record["turns"]):
            try:
                prompt = _turn_prompt(record, turn_index, turn["output_actions"])
                response = _validate_description(
                    run_pi_text(prompt, timeout=timeout),
                    [record["actions"][index] for index in turn["output_actions"]],
                )
            except Exception as error:
                failure = {
                    "sample_id": record["sample_id"],
                    "turn_id": turn["turn_id"],
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
                manifest["failures"].append(failure)
                manifest["status"] = "failed"
                manifest["failure_reason"] = failure["failure_reason"]
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
                )
                raise
            enriched["language"]["turns"].append({
                "turn_id": turn["turn_id"],
                "action_indices": turn["output_actions"],
                "text": response,
            })
            manifest["description_count_completed"] += 1
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
        output_records.append(enriched)

    output_path = output_dir / "language_action_records.jsonl"
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records)
    )
    (output_dir / "language_action_records.json").write_text(
        json.dumps(output_records, indent=2, ensure_ascii=False) + "\n"
    )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "output_dir": str(output_dir),
        "sample_count": len(output_records),
        "description_count": manifest["description_count_completed"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_DATASET))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / REVISION,
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    result = generate_language_dataset(args.source, args.output, timeout=args.timeout)
    print(
        f"completed {result['description_count']} descriptions for "
        f"{result['sample_count']} records: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
