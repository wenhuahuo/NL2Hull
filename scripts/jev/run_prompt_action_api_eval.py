#!/usr/bin/env python3
"""Run the shared prompt/action scorer through a remote model via pi."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_prompt_action_eval import digest, parse_json, prompt_for, score

from benchmarks.unified_action import aggregate_action_results
from nurbs_ship_reconstruction.agents.pi_runner import run_pi_text


def evaluate_one(record: dict[str, Any], model: str, thinking: str, timeout: int) -> dict[str, Any]:
    result = {
        "task_key": record["task_key"],
        "sample_id": record["sample_id"],
        "hull_id": record["hull_id"],
        "input_text": record["text"],
        "target_actions": record["target_actions"],
    }
    try:
        raw = run_pi_text(prompt_for(record), model=model, thinking=thinking, timeout=timeout)
        prediction = parse_json(raw)
        result.update({"status": "completed", "raw_response": raw, "prediction": prediction,
                       "score": score(prediction, record["target_actions"])})
    except Exception as error:
        result.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        raise ValueError("limit and workers must be positive")
    records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()][:args.limit]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    result_path = args.output / "results.jsonl"
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "model": args.model,
        "thinking": args.thinking,
        "record_count": len(records),
        "prompt_contract": "shared prompt_for/parse_json/score via pi API model",
        "status": "running",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with result_path.open("w") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one, record, args.model, args.thinking, args.timeout)
                       for record in records]
            for future in as_completed(futures):
                stream.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                stream.flush()
    results = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    completed = [result for result in results if result["status"] == "completed"]
    action_summary = aggregate_action_results(results)
    summary = {
        "record_count": len(records),
        "completed_records": action_summary["completed_turns"],
        "failed_records": action_summary["failed_turns"],
        "turn_exact_rate": action_summary["turn_exact_rate"],
        "field_accuracy": action_summary["field_accuracy"],
        "ffd_accuracy": action_summary,
        "failures": [result for result in results if result["status"] != "completed"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest.update({"status": "completed", "completed_records": len(completed),
                     "failed_records": len(results) - len(completed),
                     "completed_at": datetime.now(timezone.utc).isoformat()})
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
