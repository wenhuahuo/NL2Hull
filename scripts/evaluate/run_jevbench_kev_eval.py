#!/usr/bin/env python3
"""Run a local Kev checkpoint on one public JevBench JSONL tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
STRICT_TOL = 1e-3
RENORM_TOL = 2e-2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        task = json.loads(line)
        question = task["question"]
        if question["type"] not in {"noul", "choice", "score"}:
            raise ValueError(f"{path}:{line_number}: unsupported question type")
        if task["expected"] not in task["labels"]:
            raise ValueError(f"{path}:{line_number}: expected label is not in labels")
        expected_labels = {"no", "yes"} if question["type"] == "noul" else set(question_keys(question))
        if set(task["labels"]) != expected_labels:
            raise ValueError(f"{path}:{line_number}: labels do not match question criteria")
        tasks.append(task)
    if not tasks:
        raise ValueError(f"empty JevBench dataset: {path}")
    return tasks


def question_keys(question: dict[str, Any]) -> list[str]:
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(index) for index in range(len(question["criteria"]))]


def kev_record(task: dict[str, Any]) -> dict[str, Any]:
    question = task["question"]
    expected = task["expected"]
    if question["type"] == "noul":
        # Kev uses bool labels while JevBench uses the public-facing no/yes labels.
        expected = expected == "yes"
    elif question["type"] == "score":
        expected = int(expected)
    kev_question = {
        "type": question["type"],
        "instructions": question["instructions"],
        "label": expected,
        "src": f"jevbench_{task['family']}",
    }
    if "criteria" in question:
        kev_question["criteria"] = question["criteria"]
    return {
        "state": task["state"],
        "questions": {"decision": kev_question},
        "_meta": {
            "source": "jevbench",
            "variant": "clean",
            "id": task["id"],
            "group_id": task.get("group") or task["id"],
            "row": task["id"],
            "split": task["split"],
        },
    }


def jevbench_probabilities(task: dict[str, Any], probabilities: dict[str, float]) -> dict[str, float]:
    question = task["question"]
    if question["type"] != "noul":
        return probabilities
    if set(probabilities) != {"false", "true"}:
        raise ValueError("Kev noul probability keys differ")
    return {"no": float(probabilities["false"]), "yes": float(probabilities["true"])}


def validate_probabilities(probs: dict[str, float], labels: list[str]) -> tuple[dict[str, float], bool, bool]:
    if set(probs) != set(labels):
        raise ValueError("probability keys differ from JevBench labels")
    clean = {}
    for label in labels:
        value = probs[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid probability for {label}")
        clean[label] = float(value)
    total = sum(clean.values())
    if abs(total - 1.0) <= STRICT_TOL:
        return clean, True, False
    if abs(total - 1.0) <= RENORM_TOL and total > 0:
        return {label: value / total for label, value in clean.items()}, False, True
    raise ValueError(f"probabilities sum to {total}")


def ece(pairs: list[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    bins = [[0, 0.0, 0] for _ in range(10)]
    for confidence, correct in pairs:
        index = min(int(max(0.0, min(1.0, confidence)) * 10), 9)
        bins[index][0] += 1
        bins[index][1] += confidence
        bins[index][2] += int(correct)
    return sum((n / len(pairs)) * abs(correct / n - confidence / n) for n, confidence, correct in bins if n)


def summarize(tasks: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task, outcome in zip(tasks, outcomes):
        by_family[task["family"]].append(outcome)
    def metric(items: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [item for item in items if item["valid"]]
        attempted = len(items)
        correct = [item for item in items if item["correct"] is not None]
        calibration = [(item["confidence"], item["correct"]) for item in correct if item["valid"]]
        latencies = [item["latency_ms"] for item in items]
        return {
            "n": len(items),
            "n_valid": len(valid),
            "n_correct": sum(item["correct"] is True for item in correct),
            "accuracy": (sum(item["correct"] is True for item in correct) / len(correct)) if correct else None,
            "schema_validity": len(valid) / attempted if attempted else None,
            "schema_validity_strict": sum(item["strict_valid"] for item in items) / attempted if attempted else None,
            "brier": (sum(item["brier"] for item in valid if item["brier"] is not None) / len([item for item in valid if item["brier"] is not None])) if any(item["brier"] is not None for item in valid) else None,
            "ece": ece(calibration),
            "latency_ms": {
                "p50": statistics.median(latencies) if latencies else None,
                "p95": sorted(latencies)[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)] if latencies else None,
            },
        }
    return {
        "clean": metric(outcomes),
        "per_family": {family: metric(items) for family, items in sorted(by_family.items())},
        "tiers": {"public": metric(outcomes)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    tasks = load_tasks(args.data)
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    predictor = LocalPredictor(args.run, args.device, LoadOptions.from_env(), context=SERVING_CONTEXT)
    args.output.mkdir(parents=True)
    outcomes: list[dict[str, Any]] = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for task in tasks:
            started = time.perf_counter()
            outcome: dict[str, Any] = {
                "id": task["id"], "family": task["family"], "group": task.get("group"),
                "expected": task["expected"], "status": "failed", "valid": False,
                "strict_valid": False, "renormalized": False, "correct": False,
            }
            try:
                prediction = predictor(kev_record(task))
                raw = prediction["probabilities"]["decision"]
                probs = jevbench_probabilities(task, raw)
                clean, strict_valid, renormalized = validate_probabilities(probs, task["labels"])
                predicted = min(clean, key=lambda label: (-clean[label], label))
                expected = task["expected"]
                outcome.update({
                    "status": "completed", "valid": True, "strict_valid": strict_valid,
                    "renormalized": renormalized, "probabilities": clean,
                    "predicted": predicted, "correct": predicted == expected,
                    "confidence": max(clean.values()), "latency_ms": prediction["latency_ms"],
                    "input_tokens": prediction.get("input_tokens"),
                    "brier": sum((clean[label] - float(label == expected)) ** 2 for label in task["labels"]),
                })
            except Exception as error:
                outcome["failure_reason"] = f"{type(error).__name__}: {error}"
            outcome.setdefault("latency_ms", 1000 * (time.perf_counter() - started))
            stream.write(json.dumps(outcome, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            outcomes.append(outcome)

    report = summarize(tasks, outcomes)
    report.update({
        "benchmark": "jevbench-v1 public datasets",
        "benchmark_commit": BENCHMARK_COMMIT,
        "data": str(args.data), "data_sha256": sha256_file(args.data),
        "coverage": {
            "requested_records": len(tasks),
            "evaluated_records": sum(item["status"] == "completed" for item in outcomes),
            "rejected_records": sum(item["status"] != "completed" for item in outcomes),
        },
        "failures": [item for item in outcomes if item["status"] != "completed"],
    })
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "revision": args.output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run), "data": str(args.data), "record_count": len(tasks),
        "data_sha256": report["data_sha256"], "benchmark_commit": BENCHMARK_COMMIT,
        "protocol": "JevBench public tier; local Kev predictor; strict and rounded probability metrics",
        "context": "kev.suite.SERVING_CONTEXT",
        "status": "completed", "completed_records": report["coverage"]["evaluated_records"],
        "failed_records": report["coverage"]["rejected_records"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "clean": report["clean"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
