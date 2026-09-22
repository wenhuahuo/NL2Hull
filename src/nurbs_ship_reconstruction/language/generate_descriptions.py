"""Generate multilingual-model descriptions for structured hull actions."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..agents.pi_runner import DEFAULT_THINKING, run_pi_text

REVISION = "v037_natural_language_action_dataset_150000"
SMOKE_REVISION = f"{REVISION}_smoke_test"
SOURCE_DATASET = "outputs/v036_structured_action_dataset_30000"
DESCRIPTIONS_PER_TURN = 4
DEFAULT_TIMEOUT = 180
DEFAULT_WORKERS = 7
MAX_ATTEMPTS = 3
SEED = 20260922

MODELS = (
    "wokey/claude-opus-5",
    "wokey/kimi-k3",
    "wokey/glm-5.3",
    "wokey/grok-4.7",
    "deepseek/deepseek-flash",
    "openai-codex/gpt-5.6-sol",
    "xiaomi/mimo-v2.6-pro",
)


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
    variant_index: int,
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
这是同一结构化动作的第 {variant_index + 1} 种自然表达，请使用与其他表达不同但准确的句式。可以改变句子结构、动词、程度副词和衔接方式，不要把某个变化等级固定翻译成唯一词语；所有表达仍需保持相同的相对变化大小。

必须遵守：
1. 只返回一条自然、简洁的中文用户句子，不要返回 JSON、解释、标题、项目符号或引号。
2. 保留动作的区域、操作、相对变形大小和约束含义，不得增加输入 JSON 中没有的工程目标。
3. 对有内部强度编码的动作，只使用自然的模糊表达。可以在“略微、轻微、稍稍、稍微、小幅、适度、适当、较为、进一步、明显、较大、显著、大幅、充分”等表达中灵活选择，也可以调整语序和动词；禁止在输出中出现内部编码、数字强度、幅度、等级、级别、强度或档位等词语。表达的强弱必须与输入动作保持一致。
4. 如果 magnitude_level 为 null，必须准确表达 magnitude_value、longitudinal_extent 和 vertical_extent 中的数值；逐字符复制每个小数，不得四舍五入、截断或改写；这种情况下可以使用“变形量”或“数值”。
5. constraints 中的 preserve_displacement、preserve_deck_line 等约束必须在句子中体现。
6. 不要引入米、毫米等物理单位；当前数据中的数值使用归一化参数。
7. 区域术语固定为：bow=艏部，stern=艉部，bulb=球鼻艏，midbody=中体，deck=甲板，bilge=舭部，global=全船。
8. 方向术语固定为：forward=向艏移动，aftward=向艉移动，upward=上抬，downward=下压，outward=外扩，inward=内收。
9. 操作术语固定为：increase_fullness=增加丰满度，decrease_fullness=降低丰满度，increase_flare=增加外飘，increase_length=增加全船长度，decrease_length=缩短全船长度，increase_breadth=增加全船宽度，decrease_breadth=减小全船宽度。
10. change_bulb_length 表示“按给定数值改变球鼻艏长度”，magnitude_value 是改变量，不能描述成最终长度。

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


def _model_sequence(task_count: int, *, seed: int) -> list[str]:
    sequence = [MODELS[index % len(MODELS)] for index in range(task_count)]
    random.Random(seed).shuffle(sequence)
    return sequence


def _task_result(task: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    actions = [
        task["record"]["actions"][index]
        for index in task["action_indices"]
    ]
    prompt = task["prompt"]
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            text = _validate_description(
                run_pi_text(
                    prompt,
                    model=task["model"],
                    thinking=DEFAULT_THINKING,
                    timeout=timeout,
                ),
                actions,
            )
            break
        except Exception as error:
            last_error = error
            prompt = (
                task["prompt"]
                + "\n上一条回答未通过格式或语义校验。请重新生成，只返回一条符合全部要求的中文用户句子。"
                + "不要使用‘幅度、等级、级别、强度、档位’这些词，也不要解释校验过程。"
            )
    else:
        raise RuntimeError(
            f"failed after {MAX_ATTEMPTS} attempts: {type(last_error).__name__}: {last_error}"
        )
    return {
        "record_index": task["record_index"],
        "turn_index": task["turn_index"],
        "turn_id": task["turn_id"],
        "variant_index": task["variant_index"],
        "model": task["model"],
        "thinking": DEFAULT_THINKING,
        "attempts": attempt,
        "action_indices": task["action_indices"],
        "text": text,
    }


def _build_tasks(
    records: list[dict[str, Any]],
    *,
    model_sequence: list[str],
    variants_per_turn: int,
    start_task: int = 0,
) -> list[dict[str, Any]]:
    tasks = []
    task_index = start_task
    for record_index, record in enumerate(records):
        for turn_index, turn in enumerate(record["turns"]):
            for variant_index in range(variants_per_turn):
                model = model_sequence[task_index]
                tasks.append({
                    "record_index": record_index,
                    "record": record,
                    "turn_index": turn_index,
                    "turn_id": turn["turn_id"],
                    "variant_index": variant_index,
                    "model": model,
                    "action_indices": turn["output_actions"],
                    "prompt": _turn_prompt(
                        record,
                        turn_index,
                        turn["output_actions"],
                        variant_index,
                    ),
                })
                task_index += 1
    return tasks


def _write_json_array_item(file: Any, value: dict[str, Any], first: bool) -> bool:
    if not first:
        file.write(",\n")
    file.write(json.dumps(value, indent=2, ensure_ascii=False))
    return False


def _manifest(
    output_dir: Path,
    source_dir: Path,
    source_path: Path,
    records: list[dict[str, Any]],
    model_counts_target: dict[str, int],
) -> dict[str, Any]:
    turn_count = sum(len(record["turns"]) for record in records)
    return {
        "revision": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dir),
        "source_records": str(source_path),
        "source_sha256": _source_digest(records),
        "source_record_count": len(records),
        "source_turn_count": turn_count,
        "description_count_target": turn_count * DESCRIPTIONS_PER_TURN,
        "description_count_completed": 0,
        "provider_framework": "pi print mode",
        "models": list(MODELS),
        "model_counts_target": model_counts_target,
        "model_counts_completed": {model: 0 for model in MODELS},
        "thinking": DEFAULT_THINKING,
        "no_extensions": True,
        "no_skills": True,
        "no_prompt_templates": True,
        "no_themes": True,
        "no_context_files": True,
        "no_tools": True,
        "no_session": True,
        "workers": DEFAULT_WORKERS,
        "job_id": None,
        "status": "running",
        "outputs": {
            "records": str(output_dir / "language_action_records.jsonl"),
            "samples": str(output_dir / "language_samples.jsonl"),
        },
        "failures": [],
    }


def _load_records(source_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    source_path = source_dir / "structured_actions.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    if not records:
        raise ValueError(f"source dataset is empty: {source_path}")
    return source_path, records


def _run_generation(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int,
    workers: int,
    max_records: int | None = None,
    task_limit: int | None = None,
    variants_per_turn: int = DESCRIPTIONS_PER_TURN,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if workers < 1:
        raise ValueError("workers must be positive")
    source_path, all_records = _load_records(source_dir)
    records = all_records if max_records is None else all_records[:max_records]
    if task_limit is not None:
        if task_limit < 1:
            raise ValueError("task_limit must be positive")
        selected_records = []
        remaining = task_limit
        for source_record in records:
            if remaining <= 0:
                break
            turn_count = min(
                len(source_record["turns"]),
                (remaining + variants_per_turn - 1) // variants_per_turn,
            )
            if turn_count == len(source_record["turns"]):
                selected_records.append(source_record)
            else:
                selected_record = dict(source_record)
                selected_record["turns"] = source_record["turns"][:turn_count]
                selected_records.append(selected_record)
            remaining -= turn_count * variants_per_turn
        if remaining != 0:
            raise ValueError(
                f"could not select {task_limit} descriptions from source records"
            )
        records = selected_records
    if not records:
        raise ValueError("selected no source records")
    if variants_per_turn < 1:
        raise ValueError("variants_per_turn must be positive")
    task_count = sum(len(record["turns"]) for record in records) * variants_per_turn
    model_sequence = _model_sequence(task_count, seed=SEED)
    model_counts_target = dict(Counter(model_sequence))
    manifest = _manifest(
        output_dir, source_dir, source_path, records, model_counts_target
    )
    manifest["description_count_target"] = task_count
    manifest["variants_per_turn"] = variants_per_turn
    manifest["workers"] = workers
    if max_records is not None or task_limit is not None:
        manifest["smoke_test"] = True
        manifest["full_source_record_count"] = len(all_records)
        if task_limit is not None:
            manifest["requested_description_count"] = task_limit
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    tasks = _build_tasks(
        records,
        model_sequence=model_sequence,
        variants_per_turn=variants_per_turn,
    )
    results: dict[tuple[int, int, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_task_result, task, timeout=timeout): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failure = {
                    "record_index": task["record_index"],
                    "sample_id": task["record"]["sample_id"],
                    "turn_id": task["turn_id"],
                    "variant_index": task["variant_index"],
                    "model": task["model"],
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
                manifest["failures"].append(failure)
                manifest["status"] = "failed"
                manifest["failure_reason"] = failure["failure_reason"]
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
                )
                raise
            results[(result["record_index"], result["turn_index"], result["variant_index"])] = result
            manifest["description_count_completed"] += 1
            manifest["model_counts_completed"][result["model"]] += 1

    records_path = output_dir / "language_action_records.jsonl"
    samples_path = output_dir / "language_samples.jsonl"
    records_json_path = output_dir / "language_action_records.json"
    with (
        records_path.open("w") as records_file,
        samples_path.open("w") as samples_file,
        records_json_path.open("w") as records_json_file,
    ):
        records_json_file.write("[\n")
        first_json = True
        for record_index, record in enumerate(records):
            enriched = dict(record)
            language_turns = []
            for turn_index, turn in enumerate(record["turns"]):
                for variant_index in range(variants_per_turn):
                    result = results[(record_index, turn_index, variant_index)]
                    language_turns.append({
                        "turn_id": result["turn_id"],
                        "variant_index": variant_index,
                        "model": result["model"],
                        "thinking": result["thinking"],
                        "attempts": result["attempts"],
                        "action_indices": result["action_indices"],
                        "text": result["text"],
                    })
                    sample = {
                        "sample_id": record["sample_id"],
                        "hull_id": record["hull_id"],
                        "interaction_type": record["interaction_type"],
                        "turn_id": result["turn_id"],
                        "variant_index": variant_index,
                        "model": result["model"],
                        "thinking": result["thinking"],
                        "attempts": result["attempts"],
                        "action_indices": result["action_indices"],
                        "text": result["text"],
                    }
                    samples_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            enriched["language"] = {
                "language": "zh-CN",
                "turns": language_turns,
            }
            records_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            first_json = _write_json_array_item(records_json_file, enriched, first_json)
        records_json_file.write("]\n")

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "output_dir": str(output_dir),
        "source_record_count": len(records),
        "description_count": manifest["description_count_completed"],
        "model_counts": manifest["model_counts_completed"],
    }


def generate_language_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    """Generate four validated descriptions per source input turn."""
    return _run_generation(
        source_dir,
        output_dir,
        timeout=timeout,
        workers=workers,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_DATASET))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / REVISION,
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-count", type=int, default=7)
    args = parser.parse_args()
    output = args.output
    max_records = None
    task_limit = args.smoke_count if args.smoke_test else None
    if args.smoke_test and output.name == REVISION:
        output = output.with_name(SMOKE_REVISION)
    result = _run_generation(
        args.source,
        output,
        timeout=args.timeout,
        workers=args.workers,
        max_records=max_records,
        task_limit=task_limit,
        variants_per_turn=7 if args.smoke_test else DESCRIPTIONS_PER_TURN,
    )
    print(
        f"completed {result['description_count']} descriptions for "
        f"{result['source_record_count']} source records: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
