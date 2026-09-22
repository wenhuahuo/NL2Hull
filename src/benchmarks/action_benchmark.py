"""Benchmark prompt-constrained LLMs and Jev on structured FFD actions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from model_clients.jev import JEV_MODEL, JevClient
from model_clients.pi import PI_MODELS, run_pi_model

REVISION = "v032_action_model_benchmark"
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


def _turn_context(record: dict[str, Any], turn_index: int) -> str:
    previous = record["language"]["turns"][:turn_index]
    if not previous:
        return ""
    return "此前同一轮对话中的用户输入：\n" + "\n".join(
        f"- {turn['text']}" for turn in previous
    ) + "\n"


def _pi_prompt(record: dict[str, Any], turn_index: int) -> str:
    turn = record["language"]["turns"][turn_index]
    return f"""你是船型 NURBS/FFD 动作解析器。请把用户输入转换为严格 JSON。
{_turn_context(record, turn_index)}当前用户输入：
{turn['text']}

只返回一个 JSON 对象，不要 Markdown、解释或额外文字：
{{"actions":[{{
  "region":"bow|stern|bulb|midbody|deck|bilge|global",
  "operation":"outward|inward|upward|downward|forward|aftward|increase_fullness|decrease_fullness|increase_flare|change_bulb_length|increase_length|decrease_length|increase_breadth|decrease_breadth",
  "magnitude_level":1,
  "magnitude_value":null,
  "longitudinal_extent":null,
  "vertical_extent":null,
  "symmetry":true,
  "constraints":{{}}
}}]}}

解析规则：
- 一次输入涉及多个部位时，actions 按用户表达顺序输出多个对象。
- 略微、稍微、适度、明显、显著分别对应 magnitude_level 1、2、3、4、5。
- 用户明确给出数值和范围时，magnitude_level 必须为 null，并原样填写 magnitude_value、longitudinal_extent、vertical_extent。
- 没有明确数值的范围字段填写 null；未提到的约束使用空对象。
- 未说明对称性时 symmetry 使用 true。
- 连续对话只解析当前用户输入，不重复输出此前动作。
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


def _number_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-8
    except (TypeError, ValueError):
        return left == right


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(right, list):
        return (
            isinstance(left, list)
            and len(left) == len(right)
            and all(_number_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(right, float):
        return _number_equal(left, right)
    return left == right


def _action_value(action: dict[str, Any], key: str) -> Any:
    defaults = {
        "magnitude_level": None,
        "magnitude_value": None,
        "longitudinal_extent": None,
        "vertical_extent": None,
        "symmetry": True,
        "constraints": {},
    }
    return action.get(key, defaults.get(key))


def _score_standard(
    predicted: dict[str, Any], target_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    predicted_actions = predicted.get("actions")
    if not isinstance(predicted_actions, list):
        raise ValueError("JSON object has no actions list")
    fields = (
        "region",
        "operation",
        "magnitude_level",
        "magnitude_value",
        "longitudinal_extent",
        "vertical_extent",
        "symmetry",
        "constraints",
    )
    count_correct = len(predicted_actions) == len(target_actions)
    field_totals = {field: 0 for field in fields}
    field_correct = {field: 0 for field in fields}
    action_exact = 0
    for index, target in enumerate(target_actions):
        if index >= len(predicted_actions) or not isinstance(predicted_actions[index], dict):
            continue
        predicted_action = predicted_actions[index]
        action_is_exact = True
        for field in fields:
            field_totals[field] += 1
            correct = _value_equal(
                _action_value(predicted_action, field),
                _action_value(target, field),
            )
            field_correct[field] += int(correct)
            action_is_exact &= correct
        action_exact += int(action_is_exact)
    turn_exact = count_correct and action_exact == len(target_actions)
    return {
        "action_count_correct": count_correct,
        "field_totals": field_totals,
        "field_correct": field_correct,
        "action_exact": action_exact,
        "turn_exact": turn_exact,
    }


def _criteria(items: tuple[str, ...]) -> dict[str, str]:
    return {item: item for item in items}


def _jev_questions() -> dict[str, Any]:
    questions: dict[str, Any] = {
        "action_count": {
            "type": "choice",
            "instructions": "How many deformation actions are expressed in the current user input?",
            "criteria": {"1": "one action", "2": "two actions", "3": "three actions"},
        }
    }
    for index in range(1, 4):
        questions[f"mode_{index}"] = {
            "type": "choice",
            "instructions": f"Is action slot {index} qualitative or an explicit numeric-range request?",
            "criteria": {
                "qualitative": "the user uses qualitative wording",
                "explicit": "the user gives explicit numeric value and ranges",
            },
        }
        questions[f"region_{index}"] = {
            "type": "choice",
            "instructions": f"Which hull region is action slot {index} about?",
            "criteria": _criteria(REGIONS),
        }
        questions[f"operation_{index}"] = {
            "type": "choice",
            "instructions": f"Which typed operation is action slot {index}?",
            "criteria": _criteria(OPERATIONS),
        }
        questions[f"magnitude_{index}"] = {
            "type": "score",
            "instructions": f"What qualitative intensity does action slot {index} express?",
            "criteria": ["none", "slight", "small", "moderate", "large", "very large"],
        }
        questions[f"preserve_displacement_{index}"] = {
            "type": "noul",
            "instructions": f"Does action slot {index} explicitly preserve displacement?",
            "criteria": {"true": "preserve displacement is requested", "false": "it is not requested"},
        }
        questions[f"preserve_deck_line_{index}"] = {
            "type": "noul",
            "instructions": f"Does action slot {index} explicitly preserve the deck line?",
            "criteria": {"true": "preserve deck line is requested", "false": "it is not requested"},
        }
    return questions


def _answer_choice(answers: dict[str, Any], name: str) -> Any:
    value = answers.get(name, {})
    return value.get("choice") if isinstance(value, dict) else None


def _answer_noul(answers: dict[str, Any], name: str) -> bool:
    value = answers.get(name, {})
    if not isinstance(value, dict):
        return False
    try:
        return float(value.get("noul", 0.0)) >= 0.5
    except (TypeError, ValueError):
        return False


def _answer_score(answers: dict[str, Any], name: str) -> int | None:
    value = answers.get(name, {})
    if not isinstance(value, dict):
        return None
    probabilities = value.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        try:
            return int(max(probabilities, key=lambda key: float(probabilities[key])))
        except (TypeError, ValueError):
            pass
    try:
        return round(float(value["score"]))
    except (KeyError, TypeError, ValueError):
        return None


def _jev_prediction(
    answers: dict[str, Any], target_actions: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    count_raw = _answer_choice(answers, "action_count")
    try:
        count = int(count_raw)
    except (TypeError, ValueError):
        count = 0
    actions = []
    for index in range(1, max(0, min(count, 3)) + 1):
        mode = _answer_choice(answers, f"mode_{index}")
        level = _answer_score(answers, f"magnitude_{index}")
        constraints = {}
        if _answer_noul(answers, f"preserve_displacement_{index}"):
            constraints["preserve_displacement"] = True
        if _answer_noul(answers, f"preserve_deck_line_{index}"):
            constraints["preserve_deck_line"] = True
        actions.append({
            "region": _answer_choice(answers, f"region_{index}"),
            "operation": _answer_choice(answers, f"operation_{index}"),
            "magnitude_mode": mode,
            "magnitude_level": None if mode == "explicit" else level,
            "magnitude_value": None,
            "longitudinal_extent": None,
            "vertical_extent": None,
            "symmetry": True,
            "constraints": constraints,
        })
    numeric_supported = all(
        action["magnitude_level"] is not None for action in target_actions
    )
    return {"actions": actions}, numeric_supported


def _score_jev(
    predicted: dict[str, Any],
    target_actions: list[dict[str, Any]],
    *,
    numeric_supported: bool,
) -> dict[str, Any]:
    predicted_actions = predicted["actions"]
    fields = (
        "region",
        "operation",
        "magnitude_mode",
        "magnitude_level",
        "constraints",
    )
    field_totals = {field: 0 for field in fields}
    field_correct = {field: 0 for field in fields}
    exact = 0
    for index, target in enumerate(target_actions):
        if index >= len(predicted_actions):
            continue
        candidate = predicted_actions[index]
        action_is_exact = True
        for field in fields:
            field_totals[field] += 1
            if field == "magnitude_mode":
                target_value = (
                    "qualitative"
                    if target["magnitude_level"] is not None
                    else "explicit"
                )
                correct = candidate.get(field) == target_value
            else:
                target_value = target[field]
                if field == "magnitude_level" and target_value is None:
                    correct = True
                else:
                    correct = _value_equal(candidate.get(field), target_value)
            field_correct[field] += int(correct)
            action_is_exact &= correct
        exact += int(action_is_exact)
    semantic_exact = len(predicted_actions) == len(target_actions) and exact == len(target_actions)
    return {
        "action_count_correct": len(predicted_actions) == len(target_actions),
        "field_totals": field_totals,
        "field_correct": field_correct,
        "action_exact": exact,
        "semantic_turn_exact": semantic_exact,
        "numeric_fields_supported": numeric_supported,
    }


def _aggregate(results: list[dict[str, Any]], *, jev: bool) -> dict[str, Any]:
    completed = [result for result in results if result["status"] == "completed"]
    failures = [result for result in results if result["status"] != "completed"]
    totals = defaultdict(int)
    correct = defaultdict(int)
    turn_exact_key = "semantic_turn_exact" if jev else "turn_exact"
    for result in completed:
        score = result["score"]
        totals["turns"] += 1
        correct["turn_exact"] += int(score.get(turn_exact_key, False))
        for field, value in score["field_totals"].items():
            totals[field] += value
            correct[field] += score["field_correct"][field]
    by_sample: dict[str, list[bool]] = defaultdict(list)
    for result in completed:
        by_sample[result["sample_id"]].append(
            bool(result["score"].get(turn_exact_key, False))
        )
    record_exact = sum(all(values) for values in by_sample.values())
    summary = {
        "completed_turns": len(completed),
        "failed_turns": len(failures),
        "turn_exact_rate": correct["turn_exact"] / totals["turns"] if totals["turns"] else 0.0,
        "record_exact_rate": record_exact / len(by_sample) if by_sample else 0.0,
        "field_accuracy": {
            field: correct[field] / totals[field] if totals[field] else 0.0
            for field in totals
            if field != "turns"
        },
        "failures": failures,
    }
    if jev:
        summary["numeric_explicit_value_evaluation"] = (
            "not supported by Jev typed decision outputs; semantic fields are scored separately"
        )
    return summary


def run_benchmark(
    source_dir: Path,
    output_dir: Path,
    *,
    timeout: int = 180,
) -> dict[str, Any]:
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
        "job_id": None,
        "status": "running",
        "models": model_specs,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    jev_client = None
    results_by_model: dict[str, dict[str, Any]] = {}

    for model_name, spec in model_specs.items():
        results = []
        if spec["kind"] == "jev":
            jev_client = JevClient(timeout=timeout)
        for record in records:
            for turn_index, turn in enumerate(record["language"]["turns"]):
                target_actions = [
                    record["actions"][index] for index in turn["action_indices"]
                ]
                base_result = {
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
                        base_result.update({
                            "status": "completed",
                            "raw_response": raw,
                            "prediction": prediction,
                            "score": score,
                        })
                    else:
                        state = (
                            f"hull_id={record['hull_id']}\n"
                            f"{_turn_context(record, turn_index)}"
                            f"current_user_input={turn['text']}"
                        )
                        response = jev_client.decide(state, _jev_questions())
                        prediction, numeric_supported = _jev_prediction(
                            response["answers"], target_actions
                        )
                        score = _score_jev(
                            prediction,
                            target_actions,
                            numeric_supported=numeric_supported,
                        )
                        base_result.update({
                            "status": "completed",
                            "raw_response": response,
                            "prediction": prediction,
                            "score": score,
                        })
                except Exception as error:
                    base_result.update({
                        "status": "failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    })
                results.append(base_result)
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
    args = parser.parse_args()
    result = run_benchmark(args.source, args.output, timeout=args.timeout)
    print(
        f"completed {result['turn_count']} turns for "
        f"{len(result['models'])} models: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
