#!/usr/bin/env python3
"""Evaluate a local Kev checkpoint with probability and common FFD metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.unified_action import (
    actions_from_kev,
    aggregate_action_results,
    score_actions,
    target_actions_from_kev,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from kev.benchmark import prediction_rows, summarize
    from kev.checkpoint import LoadOptions
    from kev.data import api_request, load_records
    from kev.predictors import LocalPredictor
    from kev.suite import CONTEXT, record_digest, write_json

    records = load_records(args.data)
    if not records:
        raise ValueError("empty Kev test set")
    predictor = LocalPredictor(args.run, args.device, LoadOptions.from_env(), context=CONTEXT)
    args.output.mkdir(parents=True)
    manifest = {
        "revision": args.output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "run": args.run,
        "record_count": len(records),
        "protocol": "Kev LocalPredictor; shared projected FFD action metrics",
        "status": "running",
    }
    write_json(args.output / "run_manifest.json", manifest)
    rows = []
    results = []
    with (args.output / "predictions.jsonl").open("w") as stream:
        for record in records:
            target_actions = target_actions_from_kev(record)
            item = {"id": record["_meta"]["id"], "target_actions": target_actions}
            try:
                prediction = predictor(record)
                item_rows = prediction_rows(record, prediction)
                action_prediction = actions_from_kev(record, prediction["probabilities"])
                item.update({
                    "status": "completed",
                    "prediction": prediction,
                    "action_prediction": action_prediction,
                    "score": score_actions(action_prediction, target_actions),
                })
                rows.extend(item_rows)
                stream.write(json.dumps({
                    "request_sha256": record_digest(api_request(record)),
                    **item,
                    "rows": item_rows,
                }, ensure_ascii=False, allow_nan=False) + "\n")
            except Exception as error:
                item.update({"status": "failed", "failure_reason": f"{type(error).__name__}: {error}"})
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(item)

    write_json(args.output / "rows.json", rows)
    report = summarize(rows) if rows else {"clean": {}}
    report.update({
        "coverage": {
            "requested_records": len(records),
            "evaluated_records": sum(item["status"] == "completed" for item in results),
            "rejected_records": sum(item["status"] != "completed" for item in results),
        },
        "ffd_accuracy": aggregate_action_results(results),
        "failures": [item for item in results if item["status"] != "completed"],
    })
    write_json(args.output / "report.json", report)
    manifest.update({
        "status": "completed",
        "completed_records": report["coverage"]["evaluated_records"],
        "failed_records": report["coverage"]["rejected_records"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({"coverage": report["coverage"], "ffd_accuracy": report["ffd_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
