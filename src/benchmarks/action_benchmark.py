"""Benchmark prompt-constrained LLMs and Jev on structured FFD actions."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.unified_action import aggregate_action_results, score_actions
from model_clients.jev import JEV_MODEL, JevClient
from model_clients.pi import PI_MODELS, run_pi_model

REVISION = "v033_action_model_benchmark"
SOURCE_DATASET = "outputs/v030_natural_language_action_dataset_100_qualitative"

REGIONS = ("bow", "stern", "bulb", "midbody", "deck", "bilge", "global")
OPERATIONS = (
    "outward",
    "inward",
    "upward",
    "downward",
    "forward",
    "aftward",
    "increase_fullness",
    "decrease_fullness",
    "increase_flare",
    "change_bulb_length",
    "increase_length",
    "decrease_length",
    "increase_breadth",
    "decrease_breadth",
)


def _digest(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_records(source_dir: Path) -> list[dict[str, Any]]:
    path = source_dir / "language_action_records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 100:
        raise ValueError(f"expected 100 records, found {len(records)}")
    return records


def _pi_prompt(record: dict[str, Any], turn_index: int) -> str:
    turn = record["language"]["turns"][turn_index]
    return f"""你是船型 NURBS/FFD 动作解析器。请把当前用户输入转换为严格 JSON。
当前用户输入：
{turn['text']}

只返回一个 JSON 对象，不要 Markdown、解释或额外文字：
{{"actions":[{{
  "region":"bow|stern|bulb|midbody|deck|bilge|global",
  "operation":"outward|inward|upward|downward|forward|aftward|increase_fullness|decrease_fullness|increase_flare|change_bulb_length|increase_length|decrease_length|increase_breadth|decrease_breadth"
}}]}}

规则：
- actions 数量必须等于当前输入表达的动作数量。
- 多个动作按当前输入的表达顺序输出。
- 只输出 region 和 operation，不输出其他字段。
"""


def _parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        value = "\n".join(line for line in lines if not line.strip().startswith("``"))
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response does not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON is not an object")
    return parsed


def _score_standard(
    predicted: dict[str, Any], target_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    return score_actions(predicted, target_actions)


def _criteria(items: tuple[str, ...]) -> dict[str, str]:
    return {item: item for item in items}


def _jev_questions(action_count: int) -> dict[str, Any]:
    return {
        "action_count": {
            "type": "choice",
            "instructions": "How many deformation actions are expressed in the current user input?",
            "criteria": {"one": "one action", "two": "two actions", "three": "three actions"},
        },
        **{
            key: {
                "type": "choice",
                "instructions": f"Which {field} is action slot {index} about?",
                "criteria": _criteria(values),
            }
            for index in range(1, action_count + 1)
            for field, values in (("region", REGIONS), ("operation", OPERATIONS))
            for key in (f"{field}_{index}",)
        },
    }


def _answer_choice(answers: dict[str, Any], name: str) -> Any:
    value = answers.get(name, {})
    return value.get("choice") if isinstance(value, dict) else None


def _jev_prediction(
    answers: dict[str, Any], target_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    count_names = {"one": 1, "two": 2, "three": 3}
    count = count_names.get(_answer_choice(answers, "action_count"), 0)
    return {
        "actions": [
            {
                "region": _answer_choice(answers, f"region_{index}"),
                "operation": _answer_choice(answers, f"operation_{index}"),
            }
            for index in range(1, count + 1)
        ]
    }


def _score_jev(
    predicted: dict[str, Any], target_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    return score_actions(predicted, target_actions)


def _aggregate(results: list[dict[str, Any]], *, jev: bool) -> dict[str, Any]:
    summary = aggregate_action_results(results)
    summary["ffd_accuracy"] = dict(summary)
    summary["failures"] = [result for result in results if result["status"] != "completed"]
    return summary


def _evaluate_turn(
    record: dict[str, Any],
    turn_index: int,
    spec: dict[str, Any],
    *,
    timeout: int,
    jev_client: JevClient | None,
) -> dict[str, Any]:
    turn = record["language"]["turns"][turn_index]
    target_actions = [
        record["actions"][index] for index in turn["action_indices"]
    ]
    result = {
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "turn_id": turn["turn_id"],
        "input_text": turn["text"],
        "target_actions": target_actions,
        "status": "running",
    }
    try:
        if spec["kind"] == "pi":
            raw = run_pi_model(
                _pi_prompt(record, turn_index), spec["model"], timeout=timeout
            )
            prediction = _parse_json(raw)
            score = _score_standard(prediction, target_actions)
            result.update({
                "status": "completed",
                "raw_response": raw,
                "prediction": prediction,
                "score": score,
            })
        else:
            if jev_client is None:
                raise RuntimeError("Jev client is not initialized")
            state = turn["text"]
            response = jev_client.decide(state, _jev_questions(len(target_actions)))
            prediction = _jev_prediction(response["answers"], target_actions)
            score = _score_jev(prediction, target_actions)
            result.update({
                "status": "completed",
                "raw_response": response,
                "prediction": prediction,
                "score": score,
            })
    except Exception as error:
        result.update({
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        })
    return result


def run_benchmark(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int = 180,
    workers: int = 1,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    records = _load_records(source_dir)
    model_specs = {
        "deepseek-flash": {"kind": "pi", "model": PI_MODELS["deepseek-flash"]},
        "grok-4.7": {"kind": "pi", "model": PI_MODELS["grok-4.7"]},
        "jev-latest": {"kind": "jev", "model": JEV_MODEL},
    }
    manifest = {
        "revision": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dir),
        "source_sha256": _digest(records),
        "sample_count": len(records),
        "turn_count": sum(len(record["language"]["turns"]) for record in records),
        "workers": workers,
        "job_id": None,
        "status": "running",
        "models": model_specs,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    jev_client = None
    results_by_model: dict[str, dict[str, Any]] = {}

    for model_name, spec in model_specs.items():
        if spec["kind"] == "jev":
            jev_client = JevClient(timeout=timeout)
        cases = [
            (record, turn_index)
            for record in records
            for turn_index in range(len(record["language"]["turns"]))
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_turn,
                    record,
                    turn_index,
                    spec,
                    timeout=timeout,
                    jev_client=jev_client,
                )
                for record, turn_index in cases
            ]
            results = [future.result() for future in futures]
        (output_dir / f"{model_name}_results.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results)
        )
        summary = _aggregate(results, jev=spec["kind"] == "jev")
        (output_dir / f"{model_name}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        results_by_model[model_name] = summary
        manifest["models"][model_name]["status"] = (
            "completed" if not summary["failures"] else "completed_with_failures"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(results_by_model, indent=2, ensure_ascii=False) + "\n"
    )
    manifest["status"] = (
        "completed"
        if all(not summary["failures"] for summary in results_by_model.values())
        else "completed_with_failures"
    )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "output_dir": str(output_dir),
        "models": list(results_by_model),
        "turn_count": manifest["turn_count"],
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
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_benchmark(
        args.source,
        args.output,
        timeout=args.timeout,
        workers=args.workers,
    )
    print(
        f"completed {result['turn_count']} turns for "
        f"{len(result['models'])} models: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
