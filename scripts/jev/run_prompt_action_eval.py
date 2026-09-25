#!/usr/bin/env python3
"""Evaluate a local chat model by prompting it to emit structured FFD actions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

FIELDS = (
    "region", "operation", "magnitude_level", "magnitude_value",
    "longitudinal_extent", "vertical_extent", "symmetry", "constraints",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prompt_for(record: dict[str, Any]) -> str:
    previous = ""
    if record["previous_turns"]:
        previous = "此前同一任务的历史输入：\n" + "\n".join(
            f"- {text}" for text in record["previous_turns"]
        ) + "\n"
    return f"""你是船舶设计专家和 NURBS/FFD 动作解析器。请把当前用户输入转换为严格 JSON。
{previous}当前用户输入：
{record['text']}

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

规则：
- 多个动作按用户表达顺序输出多个对象。
- 略微、稍微、适度、明显、显著分别对应 magnitude_level 1、2、3、4、5。
- 明确给出数值时 magnitude_level 必须为 null，并原样填写数值和范围。
- 没有明确数值的范围字段填写 null。
- 未提到的 constraints 使用空对象。
- constraints 只能使用 preserve_displacement 和 preserve_deck_line。
- 未说明对称性时 symmetry 使用 true。
- 只解析当前用户输入，不重复输出历史动作。
"""


def request_json(base_url: str, model: str, prompt: str, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }).encode()
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body["choices"][0]["message"]["content"].strip()


def parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = "\n".join(line for line in value.splitlines() if not line.strip().startswith("```"))
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    result = json.loads(value[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("response JSON is not an object")
    return result


def value_equal(left: Any, right: Any) -> bool:
    if isinstance(right, list):
        if not isinstance(left, list) or len(left) != len(right):
            return False
        return all(value_equal(a, b) for a, b in zip(left, right))
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return abs(float(left) - float(right)) <= 1e-8
        except (TypeError, ValueError):
            return False
    return left == right


def score(predicted: dict[str, Any], target: list[dict[str, Any]]) -> dict[str, Any]:
    actions = predicted.get("actions")
    if not isinstance(actions, list):
        raise ValueError("JSON object has no actions list")
    totals = {field: len(target) for field in FIELDS}
    correct = {field: 0 for field in FIELDS}
    exact = 0
    for index, expected in enumerate(target):
        if index >= len(actions) or not isinstance(actions[index], dict):
            continue
        candidate = actions[index]
        is_exact = True
        for field in FIELDS:
            ok = value_equal(candidate.get(field), expected.get(field))
            correct[field] += int(ok)
            is_exact &= ok
        exact += int(is_exact)
    return {
        "action_count_correct": len(actions) == len(target),
        "action_exact": exact,
        "turn_exact": len(actions) == len(target) and exact == len(target),
        "field_totals": totals,
        "field_correct": correct,
    }


def evaluate_one(record: dict[str, Any], base_url: str, model: str, timeout: int) -> dict[str, Any]:
    result = {
        "task_key": record["task_key"],
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "input_text": record["text"],
        "target_actions": record["target_actions"],
    }
    try:
        raw = request_json(base_url, model, prompt_for(record), timeout)
        prediction = parse_json(raw)
        result.update({"status": "completed", "raw_response": raw, "prediction": prediction,
                       "score": score(prediction, record["target_actions"])})
    except Exception as error:  # retain per-record failures for denominator auditing
        result.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    completed_keys = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            if line.strip():
                completed_keys.add(json.loads(line)["task_key"])
    pending = [record for record in records if record["task_key"] not in completed_keys]
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "model": args.model,
        "base_url": args.base_url,
        "workers": args.workers,
        "record_count": len(records),
        "resumed_records": len(completed_keys),
        "status": "running",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with result_path.open("a") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one, record, args.base_url, args.model, args.timeout)
                       for record in pending]
            for future in as_completed(futures):
                stream.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                stream.flush()

    results = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    totals = defaultdict(int)
    correct = defaultdict(int)
    completed = [result for result in results if result["status"] == "completed"]
    for result in completed:
        totals["turns"] += 1
        correct["turn_exact"] += int(result["score"]["turn_exact"])
        for field, value in result["score"]["field_totals"].items():
            totals[field] += value
            correct[field] += result["score"]["field_correct"][field]
    summary = {
        "record_count": len(records),
        "completed_records": len(completed),
        "failed_records": len(results) - len(completed),
        "turn_exact_rate": correct["turn_exact"] / totals["turns"] if totals["turns"] else 0.0,
        "field_accuracy": {field: correct[field] / totals[field] for field in FIELDS if totals[field]},
        "failures": [r for r in results if r["status"] != "completed"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest.update({"status": "completed", "completed_records": len(completed), "failed_records": len(results) - len(completed),
                     "completed_at": datetime.now(timezone.utc).isoformat()})
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
