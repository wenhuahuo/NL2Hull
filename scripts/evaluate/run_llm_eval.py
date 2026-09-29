#!/usr/bin/env python3
"""Evaluate an API model on the exact Kev benchmark protocol through pi."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from model_clients.pi_runner import run_pi_text


def render(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}"
        if isinstance(item, (dict, list))
        else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def question_keys(question: dict[str, Any]) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "noul":
        return ["false", "true"]
    return [str(index) for index in range(len(question["criteria"]))]


def option_lines(question: dict[str, Any]) -> list[str]:
    if question["type"] == "choice":
        return [f"{key}: {render(value)}" for key, value in question["criteria"].items()]
    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        return [f"false: {render(criteria.get('false'))}", f"true: {render(criteria.get('true'))}"]
    return [f"{index}: {render(value)}" for index, value in enumerate(question["criteria"])]


def prompt_for(record: dict[str, Any]) -> str:
    sections = []
    for qid, question in record["questions"].items():
        sections.append(
            f"Question ID: {qid}\n"
            f"Type: {question['type']}\n"
            f"Instruction: {render(question.get('instructions'))}\n"
            "Options (keys must be preserved exactly):\n"
            + "\n".join(f"- {line}" for line in option_lines(question))
        )
    return f"""你是一个船舶设计决策模型。请根据状态和问题，输出每个问题的概率分布。

状态：
{render(record['state'])}

问题：
{chr(10).join(sections)}

只输出严格 JSON，不要 Markdown、解释或额外文字：
{{"answers": {{
  "question_id": {{"probabilities": {{"option_key": 0.5}}}}
}}}}

要求：
- 必须回答全部问题，question ID 必须完全匹配。
- 每个 probabilities 的 key 必须完全匹配该问题列出的 key。
- 概率为 0 到 1 之间的数字，并且每个问题的概率和为 1。
- choice 使用选项名称作为 key；noul 必须使用 false 和 true；score 使用 0、1、2 等字符串作为 key。
- 只能返回上述 JSON 对象。
"""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def benchmark_record(record: dict[str, Any], row: int) -> dict[str, Any]:
    """Add the Kev benchmark metadata expected by prediction_rows()."""
    normalized = dict(record)
    metadata = dict(record["_meta"])
    task_key = metadata["task_key"]
    metadata.update({
        "id": task_key,
        "group_id": metadata.get("sample_id", task_key),
        "source": metadata.get("source_model", "custom"),
        "variant": "clean",
        "row": row,
        "split": "custom",
    })
    normalized["_meta"] = metadata
    return normalized


def parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = "\n".join(line for line in value.splitlines() if not line.strip().startswith("```"))
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed


def probabilities(record: dict[str, Any], parsed: dict[str, Any]) -> dict[str, dict[str, float]]:
    answers = parsed.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("response has no answers object")
    result: dict[str, dict[str, float]] = {}
    for qid, question in record["questions"].items():
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            raise ValueError(f"missing answer: {qid}")
        keys = question_keys(question)
        raw = answer.get("probabilities")
        if question["type"] == "noul" and raw is None and "noul" in answer:
            value = float(answer["noul"])
            raw = {"false": 1.0 - value, "true": value}
        if not isinstance(raw, dict) or set(raw) != set(keys):
            raise ValueError(f"probability keys differ for {qid}")
        values = {key: float(raw[key]) for key in keys}
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError(f"probability outside [0,1] for {qid}")
        total = sum(values.values())
        if total <= 0:
            raise ValueError(f"probability sum is zero for {qid}")
        result[qid] = {key: value / total for key, value in values.items()}
    return result


def evaluate_one(record: dict[str, Any], model: str, provider: str | None, thinking: str, timeout: int, retries: int) -> dict[str, Any]:
    start = time.perf_counter()
    result = {"id": record["_meta"]["task_key"], "record": record}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = run_pi_text(prompt_for(record), model=model, provider=provider, thinking=thinking, timeout=timeout)
            parsed = parse_json(raw)
            probs = probabilities(record, parsed)
            result.update({"status": "completed", "raw_response": raw,
                           "prediction": {"probabilities": probs},
                           "latency_ms": 1000 * (time.perf_counter() - start),
                           "attempts": attempt + 1})
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** attempt)
    result.update({"status": "failed", "failure_reason": f"{type(last_error).__name__}: {last_error}",
                   "latency_ms": 1000 * (time.perf_counter() - start), "attempts": retries + 1})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--provider")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high"), default="low")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw_records = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not raw_records:
        raise ValueError("empty Kev test set")
    records = [benchmark_record(record, row) for row, record in enumerate(raw_records)]
    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "model": args.model,
        "provider": args.provider,
        "thinking": args.thinking,
        "record_count": len(records),
        "protocol": "Kev test split; kev.benchmark.prediction_rows and summarize",
        "status": "running",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result_items: list[dict[str, Any]] = []
    temporary_results = args.output / "results.tmp.jsonl"
    with temporary_results.open("w", encoding="utf-8") as temporary_stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one, record, args.model, args.provider, args.thinking, args.timeout, args.retries)
                       for record in records]
            for future in as_completed(futures):
                result = future.result()
                result_items.append(result)
                temporary_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                temporary_stream.flush()
                if len(result_items) % 50 == 0:
                    print(f"evaluated {len(result_items)}/{len(records)}", flush=True)

    # Import the upstream Kev metric implementation only on the cluster, where third_party/kev is available.
    from kev.benchmark import prediction_rows, summarize
    from kev.data import api_request
    from kev.suite import record_digest, write_json

    rows: list[dict[str, Any]] = []
    failures = []
    with (args.output / "predictions.jsonl").open("w") as stream:
        for item in result_items:
            if item["status"] != "completed":
                failures.append({key: item[key] for key in ("id", "status", "failure_reason", "attempts", "latency_ms")})
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue
            item_rows = prediction_rows(item["record"], item["prediction"])
            stream.write(json.dumps({"request_sha256": record_digest(api_request(item["record"])),
                                     "id": item["id"], "prediction": item["prediction"], "rows": item_rows},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            rows.extend(item_rows)
    write_json(args.output / "rows.json", rows)
    report = summarize(rows)
    report.update({
        "coverage": {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                     "evaluated_records": len(result_items) - len(failures),
                     "evaluated_questions": len(rows), "rejected_records": len(failures)},
        "failures": failures,
        "latency_ms": {"mean": sum(item["latency_ms"] for item in result_items) / len(result_items),
                       "p95": sorted(item["latency_ms"] for item in result_items)[max(0, int(len(result_items) * 0.95) - 1)]},
        "model": args.model,
        "provider": args.provider,
        "thinking": args.thinking,
        "data_sha256": digest(args.data),
    })
    write_json(args.output / "report.json", report)
    manifest.update({"status": "completed", "completed_records": len(result_items) - len(failures),
                     "failed_records": len(failures), "completed_at": datetime.now(timezone.utc).isoformat()})
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary_results.unlink(missing_ok=True)
    print(json.dumps({"coverage": report["coverage"], "clean": report["clean"],
                      "failed_records": len(failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
