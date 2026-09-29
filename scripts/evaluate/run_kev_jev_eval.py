#!/usr/bin/env python3
"""Score native OpenRouter Jev on the frozen Kev ship test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from benchmarks.unified_action import (
    actions_from_kev,
    aggregate_action_results,
    score_actions,
    target_actions_from_kev,
)
from model_clients.jev import JEV_MODEL, JEV_URL, JevClient


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probabilities(record: dict[str, Any], response: dict[str, Any]) -> dict[str, dict[str, float]]:
    from kev.api import question_keys

    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(record["questions"]):
        raise ValueError("Jev answer IDs do not match request IDs")
    result = {}
    for qid, q in record["questions"].items():
        answer = answers[qid]
        if not isinstance(answer, dict):
            raise ValueError(f"Jev answer is not an object for {qid}")
        if q["type"] == "noul":
            p = float(answer.get("noul", answer.get("probability")))
            dist = {"false": 1.0 - p, "true": p}
        else:
            dist = answer.get("probabilities")
        keys = question_keys(q["type"], q.get("criteria"))
        if not isinstance(dist, dict) or set(dist) != set(keys):
            raise ValueError(f"Jev probability keys mismatch for {qid}")
        values = {key: float(dist[key]) for key in keys}
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values.values()):
            raise ValueError(f"Jev invalid probability for {qid}")
        total = sum(values.values())
        if total <= 0 or abs(total - 1.0) > max(1e-5, len(keys) * 0.005 + 1e-8):
            raise ValueError(f"Jev probability sum mismatch for {qid}: {total}")
        result[qid] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    from kev.benchmark import prediction_rows, summarize
    from kev.data import api_request, load_records
    from kev.suite import record_digest, write_json

    records = load_records(args.data)[:args.limit]
    client = JevClient(timeout=args.timeout)
    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data), "data_sha256": digest(args.data), "record_count": len(records),
        "model": JEV_MODEL, "endpoint": JEV_URL,
        "protocol": "native OpenRouter Jev; same labelled requests and Kev benchmark metrics",
        "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    rows = []
    latencies = []
    action_results = []
    processed = 0
    failures = []
    with (args.output / "predictions.jsonl").open("w") as output:
        for record in records:
            started = time.perf_counter()
            target_actions = target_actions_from_kev(record)
            action_item = {"id": record["_meta"]["id"], "target_actions": target_actions}
            try:
                request = api_request(record)
                response = client.decide(request["state"], request["questions"])
                probs = probabilities(record, response)
                pred = {"probabilities": probs,
                        "latency_ms": (time.perf_counter() - started) * 1000}
                action_prediction = actions_from_kev(record, probs)
                action_item.update({"status": "completed", "score": score_actions(action_prediction, target_actions)})
                new_rows = prediction_rows(record, pred)
                output.write(json.dumps({
                    "request_sha256": record_digest(request), "id": record["_meta"]["id"],
                    "prediction": pred, "action_prediction": action_prediction,
                    "action_score": action_item["score"], "rows": new_rows,
                    "model": response.get("model"), "request_id": response.get("request_id"),
                }, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                rows.extend(new_rows)
                latencies.append(pred["latency_ms"])
                processed += 1
                if processed % 50 == 0:
                    print(f"evaluated {processed}/{len(records)}", flush=True)
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                action_item.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
                failures.append(action_item)
                output.write(json.dumps(action_item, ensure_ascii=False) + "\n")
                output.flush()
            action_results.append(action_item)
    write_json(args.output / "rows.json", rows)
    report = summarize(rows) if rows else {"clean": {}}
    report.update({
        "coverage": {"requested_records": len(records),
                     "requested_questions": sum(len(r["questions"]) for r in records),
                     "evaluated_records": processed, "evaluated_questions": len(rows),
                     "rejected_records": len(records) - processed},
        "ffd_accuracy": aggregate_action_results(action_results),
        "model": JEV_MODEL, "endpoint": JEV_URL, "split": "custom",
        "data": str(args.data), "data_sha256": digest(args.data),
        "failures": failures,
    })
    if rows and latencies:
        report["latency_ms"] = {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)],
        }
    write_json(args.output / "report.json", report)
    manifest.update({"status": "completed" if processed == len(records) else "incomplete",
                     "evaluated_records": processed, "failed_records": len(failures),
                     "finished_at": datetime.now(timezone.utc).isoformat()})
    write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "evaluated_records": processed,
                      "requested_records": len(records), "failed_records": len(failures)}, indent=2))
    if processed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
