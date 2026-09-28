#!/usr/bin/env python3
"""Evaluate a model through pi using the direct FFD action JSON protocol."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.unified_action import aggregate_action_results, score_actions, target_actions_from_kev
from nurbs_ship_reconstruction.agents.pi_runner import run_pi_text


def digest(path: Path) -> str:
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prompt_for(record: dict[str, Any]) -> str:
    return f"""你是船舶设计动作解析器。请把当前用户输入转换为严格 JSON。
当前用户输入：
{record['text']}

只返回一个 JSON 对象，不要 Markdown、解释或额外文字：
{{"actions":[{{
  "region":"bow|stern|bulb|midbody|deck|bilge|global",
  "operation":"outward|inward|upward|downward|forward|aftward|increase_fullness|decrease_fullness|increase_flare|change_bulb_length|increase_length|decrease_length|increase_breadth|decrease_breadth"
}}]}}

规则：
- actions 数量必须等于当前输入表达的动作数量。
- 多个动作按当前输入的表达顺序输出。
- 只输出 region 和 operation，不输出其他字段。
- 只解析当前用户输入，不使用或重复历史轮次。
"""


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


def evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    if "text" in record and "target_actions" in record:
        return record
    metadata = record["_meta"]
    return {
        "task_key": metadata["task_key"],
        "sample_id": metadata["sample_id"],
        "hull_id": metadata["hull_id"],
        "text": record["state"],
        "target_actions": target_actions_from_kev(record),
    }


def build_report(
    results: list[dict[str, Any]], records: list[dict[str, Any]], data: Path, model: str,
) -> dict[str, Any]:
    action_summary = aggregate_action_results(results)
    return {
        "coverage": {
            "requested_records": len(records),
            "evaluated_records": action_summary["completed_turns"],
            "rejected_records": action_summary["failed_turns"],
        },
        "ffd_accuracy": action_summary,
        "failures": [result for result in results if result["status"] != "completed"],
        "model": model,
        "data": str(data),
        "data_sha256": digest(data),
    }


def evaluate_one(
    record: dict[str, Any], model: str, provider: str | None, thinking: str, timeout: int,
) -> dict[str, Any]:
    result = {
        "task_key": record["task_key"],
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "input_text": record["text"],
        "target_actions": record["target_actions"],
    }
    try:
        raw = run_pi_text(
            prompt_for(record), model=model, provider=provider, thinking=thinking, timeout=timeout
        )
        prediction = parse_json(raw)
        result.update({
            "status": "completed",
            "raw_response": raw,
            "prediction": prediction,
            "score": score_actions(prediction, record["target_actions"]),
        })
    except Exception as error:
        result.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [
        evaluation_record(json.loads(line))
        for line in args.data.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise ValueError("empty evaluation dataset")

    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "model": args.model,
        "provider": args.provider,
        "thinking": args.thinking,
        "record_count": len(records),
        "prompt_contract": "pi direct FFD action JSON",
        "status": "running",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results = []
    with (args.output / "predictions.jsonl").open("w") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(
                evaluate_one, record, args.model, args.provider, args.thinking, args.timeout
            ) for record in records]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if len(results) % 50 == 0:
                    print(f"evaluated {len(results)}/{len(records)}", flush=True)

    report = build_report(results, records, args.data, args.model)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = report["ffd_accuracy"]
    manifest.update({
        "status": "completed",
        "completed_records": summary["completed_turns"],
        "failed_records": summary["failed_turns"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
