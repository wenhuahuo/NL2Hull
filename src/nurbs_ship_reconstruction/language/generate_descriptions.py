"""Generate validated Chinese descriptions for structured hull actions."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

from ..agents.pi_runner import DEFAULT_THINKING, run_pi_text

REVISION = "v038_stage3_language_150k"
SMOKE_REVISION = f"{REVISION}_smoke_test"
SOURCE_DATASET = "outputs/v036_structured_action_dataset_30000"
DESCRIPTIONS_PER_TURN = 4
DEFAULT_TIMEOUT = 180
DEFAULT_WORKERS = 7
DEFAULT_PROGRESS_EVERY = 10
TASK_CHUNK_SIZE = 256
MAX_ATTEMPTS = 3
SEED = 20260922

MODELS = (
    "wokey/claude-opus-5",
    "wokey/kimi-k3",
    "wokey/glm-5.3",
    "wokey/grok-4.7",
    "deepseek/deepseek-flash",
    "wokey/gpt-5.6-sol",
    "openrouter/xiaomi/mimo-v2.6-pro",
)


# A deterministic subset receives a short scenario so background wording is varied
# without changing the structured action target.
BACKGROUND_SCENARIOS = (
    "我正在进行船舶设计，当前使用的母型船是{hull_id}，我需要",
    "你是一名船舶设计专家，现在请",
    "我正在进行 AI4CFD 研究，需要生成一些船型数据，请",
    "基于{hull_id}母型船的方案优化，请",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _source_digest(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _task_key(sample_id: str, turn_id: int, variant_index: int) -> str:
    return f"{sample_id}/turn_{turn_id}/variant_{variant_index}"


def _background_instruction(
    record: dict[str, Any],
    turn_index: int,
    variant_index: int,
) -> str:
    digest = hashlib.sha256(
        f"{record['sample_id']}:{turn_index}:{variant_index}".encode()
    ).digest()
    selector = digest[0] % 10
    if selector >= len(BACKGROUND_SCENARIOS):
        return "本条直接表达设计修改，不添加额外背景。"
    scenario = BACKGROUND_SCENARIOS[selector].format(hull_id=record["hull_id"])
    return (
        f"本条请自然地以“{scenario}”开头或融入句子，背景只说明使用场景，"
        "不能增加结构化动作中没有的目标。"
    )


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
    background = _background_instruction(record, turn_index, variant_index)
    return f"""你正在为船型变形数据集生成一条中文用户输入描述。
{context}{instruction}
{background}
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
    for action in actions:
        if action["magnitude_level"] is not None:
            continue
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
            f"failed after {MAX_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )
    return {
        "task_key": task["task_key"],
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


def _iter_tasks(
    records: list[dict[str, Any]],
    *,
    model_sequence: list[str],
    variants_per_turn: int,
) -> Iterable[dict[str, Any]]:
    task_index = 0
    for record_index, record in enumerate(records):
        for turn_index, turn in enumerate(record["turns"]):
            for variant_index in range(variants_per_turn):
                model = model_sequence[task_index]
                yield {
                    "task_key": _task_key(
                        record["sample_id"], turn["turn_id"], variant_index
                    ),
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
                }
                task_index += 1


def _manifest_path_write(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_existing_samples(path: Path) -> tuple[set[str], Counter[str]]:
    completed: set[str] = set()
    model_counts: Counter[str] = Counter()
    if not path.exists():
        return completed, model_counts
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        sample = json.loads(line)
        key = sample["task_key"]
        if key in completed:
            raise ValueError(f"duplicate task_key in {path}:{line_number}: {key}")
        completed.add(key)
        model_counts[sample["model"]] += 1
    return completed, model_counts


def _select_records(
    all_records: list[dict[str, Any]],
    *,
    max_records: int | None,
    task_limit: int | None,
    variants_per_turn: int,
) -> list[dict[str, Any]]:
    records = all_records if max_records is None else all_records[:max_records]
    if task_limit is None:
        if not records:
            raise ValueError("selected no source records")
        return records
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
    return selected_records


def _base_manifest(
    output_dir: Path,
    source_dir: Path,
    source_path: Path,
    records: list[dict[str, Any]],
    model_counts_target: dict[str, int],
    *,
    variants_per_turn: int,
    workers: int,
    progress_every: int,
) -> dict[str, Any]:
    turn_count = sum(len(record["turns"]) for record in records)
    return {
        "revision": output_dir.name,
        "code_revision": os.environ.get("NURBS_CODE_COMMIT"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dir),
        "source_records": str(source_path),
        "source_sha256": _source_digest(records),
        "source_record_count": len(records),
        "source_turn_count": turn_count,
        "description_count_target": turn_count * variants_per_turn,
        "variants_per_turn": variants_per_turn,
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
        "workers": workers,
        "progress_every": progress_every,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "running",
        "outputs": {
            "checkpoint_samples": str(output_dir / "language_samples.jsonl"),
            "records": str(output_dir / "language_action_records.jsonl"),
        },
        "failures": [],
    }


def _load_records(source_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    source_path = source_dir / "structured_actions.jsonl"
    records = [json.loads(line) for line in source_path.read_text().splitlines()]
    if not records:
        raise ValueError(f"source dataset is empty: {source_path}")
    return source_path, records


def _sample_from_result(
    result: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_key": result["task_key"],
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "interaction_type": record["interaction_type"],
        "record_index": result["record_index"],
        "turn_index": result["turn_index"],
        "turn_id": result["turn_id"],
        "variant_index": result["variant_index"],
        "model": result["model"],
        "thinking": result["thinking"],
        "attempts": result["attempts"],
        "action_indices": result["action_indices"],
        "text": result["text"],
    }


def _finalize_records(
    output_dir: Path,
    records: list[dict[str, Any]],
    variants_per_turn: int,
) -> None:
    samples_path = output_dir / "language_samples.jsonl"
    samples: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(samples_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        sample = json.loads(line)
        key = sample["task_key"]
        if key in samples:
            raise ValueError(f"duplicate task_key at checkpoint line {line_number}: {key}")
        samples[key] = sample

    records_path = output_dir / "language_action_records.jsonl"
    records_json_path = output_dir / "language_action_records.json"
    with (
        records_path.open("w") as records_file,
        records_json_path.open("w") as records_json_file,
    ):
        records_json_file.write("[\n")
        first_json = True
        for record in records:
            enriched = dict(record)
            language_turns = []
            for turn in record["turns"]:
                for variant_index in range(variants_per_turn):
                    key = _task_key(record["sample_id"], turn["turn_id"], variant_index)
                    sample = samples[key]
                    language_turns.append({
                        "turn_id": sample["turn_id"],
                        "variant_index": variant_index,
                        "model": sample["model"],
                        "thinking": sample["thinking"],
                        "attempts": sample["attempts"],
                        "action_indices": sample["action_indices"],
                        "text": sample["text"],
                    })
            enriched["language"] = {"language": "zh-CN", "turns": language_turns}
            records_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            if not first_json:
                records_json_file.write(",\n")
            records_json_file.write(json.dumps(enriched, indent=2, ensure_ascii=False))
            first_json = False
        records_json_file.write("]\n")


def _run_generation(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int,
    workers: int,
    progress_every: int,
    max_records: int | None = None,
    task_limit: int | None = None,
    variants_per_turn: int = DESCRIPTIONS_PER_TURN,
    resume: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    if variants_per_turn < 1:
        raise ValueError("variants_per_turn must be positive")
    source_path, all_records = _load_records(source_dir)
    records = _select_records(
        all_records,
        max_records=max_records,
        task_limit=task_limit,
        variants_per_turn=variants_per_turn,
    )
    task_count = sum(len(record["turns"]) for record in records) * variants_per_turn
    model_sequence = _model_sequence(task_count, seed=SEED)
    model_counts_target = dict(Counter(model_sequence))
    manifest_path = output_dir / "run_manifest.json"
    samples_path = output_dir / "language_samples.jsonl"

    if output_dir.exists() and not resume:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if resume:
        if not output_dir.exists() or not manifest_path.exists():
            raise FileNotFoundError("resume requires an existing output and manifest")
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "source_sha256": _source_digest(records),
            "description_count_target": task_count,
            "models": list(MODELS),
            "variants_per_turn": variants_per_turn,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"resume mismatch for {key}: {manifest.get(key)!r}")
        completed, existing_model_counts = _load_existing_samples(samples_path)
        manifest["status"] = "running"
        manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["job_id"] = os.environ.get("SLURM_JOB_ID")
        manifest["description_count_completed"] = len(completed)
        manifest["model_counts_completed"] = {
            model: existing_model_counts.get(model, 0) for model in MODELS
        }
        _manifest_path_write(manifest_path, manifest)
    else:
        output_dir.mkdir(parents=True)
        manifest = _base_manifest(
            output_dir,
            source_dir,
            source_path,
            records,
            model_counts_target,
            variants_per_turn=variants_per_turn,
            workers=workers,
            progress_every=progress_every,
        )
        if max_records is not None or task_limit is not None:
            manifest["smoke_test"] = True
            manifest["full_source_record_count"] = len(all_records)
            if task_limit is not None:
                manifest["requested_description_count"] = task_limit
        _manifest_path_write(manifest_path, manifest)
        samples_path.touch()
        completed = set()

    total_completed = len(completed)
    started = time.monotonic()
    pending: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def process_batch(
        batch: list[dict[str, Any]],
        executor: ThreadPoolExecutor,
        samples_file: Any,
    ) -> None:
        nonlocal total_completed
        futures = {
            executor.submit(_task_result, task, timeout=timeout): task
            for task in batch
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append({
                    "task_key": task["task_key"],
                    "sample_id": task["record"]["sample_id"],
                    "turn_id": task["turn_id"],
                    "variant_index": task["variant_index"],
                    "model": task["model"],
                    "failure_reason": f"{type(error).__name__}: {error}",
                })
                continue
            sample = _sample_from_result(result, task["record"])
            samples_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples_file.flush()
            completed.add(result["task_key"])
            total_completed += 1
            manifest["description_count_completed"] = total_completed
            manifest["model_counts_completed"][result["model"]] += 1
            if (
                total_completed % progress_every == 0
                or total_completed == task_count
            ):
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = (total_completed - len(completed_before_run)) / elapsed
                remaining = task_count - total_completed
                eta = remaining / rate if rate > 0 else None
                eta_text = f" eta={eta / 3600:.2f}h" if eta is not None else ""
                print(
                    f"progress={total_completed}/{task_count} "
                    f"model={result['model']} rate={rate:.3f}/s{eta_text}",
                    flush=True,
                )
                _manifest_path_write(manifest_path, manifest)

    completed_before_run = set(completed)
    try:
        with samples_path.open("a") as samples_file, ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            for task in _iter_tasks(
                records,
                model_sequence=model_sequence,
                variants_per_turn=variants_per_turn,
            ):
                if task["task_key"] in completed:
                    continue
                pending.append(task)
                if len(pending) >= TASK_CHUNK_SIZE:
                    process_batch(pending, executor, samples_file)
                    pending = []
            if pending:
                process_batch(pending, executor, samples_file)
    except Exception:
        manifest["status"] = "failed"
        manifest["failure_reason"] = "unexpected generation failure"
        _manifest_path_write(manifest_path, manifest)
        raise

    if failures:
        manifest["status"] = "failed"
        manifest["failures"].extend(failures)
        manifest["failure_reason"] = failures[0]["failure_reason"]
        _manifest_path_write(manifest_path, manifest)
        raise RuntimeError(
            f"{len(failures)} generation tasks failed; use --resume after resolving the cause"
        )
    if total_completed != task_count:
        manifest["status"] = "failed"
        manifest["failure_reason"] = (
            f"checkpoint contains {total_completed} of {task_count} descriptions"
        )
        _manifest_path_write(manifest_path, manifest)
        raise RuntimeError(manifest["failure_reason"])

    _finalize_records(output_dir, records, variants_per_turn)
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _manifest_path_write(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "source_record_count": len(records),
        "description_count": total_completed,
        "model_counts": manifest["model_counts_completed"],
    }


def generate_language_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    workers: int = DEFAULT_WORKERS,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate four validated descriptions per source input turn."""
    return _run_generation(
        source_dir,
        output_dir,
        timeout=timeout,
        workers=workers,
        progress_every=progress_every,
        resume=resume,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_DATASET))
    parser.add_argument("--output", type=Path, default=Path("outputs") / REVISION)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-count", type=int, default=7)
    args = parser.parse_args()
    output = args.output
    if args.smoke_test and output.name == REVISION:
        output = output.with_name(SMOKE_REVISION)
    result = _run_generation(
        args.source,
        output,
        timeout=args.timeout,
        workers=args.workers,
        progress_every=args.progress_every,
        task_limit=args.smoke_count if args.smoke_test else None,
        variants_per_turn=7 if args.smoke_test else DESCRIPTIONS_PER_TURN,
        resume=args.resume,
    )
    print(
        f"completed {result['description_count']} descriptions for "
        f"{result['source_record_count']} source records: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
