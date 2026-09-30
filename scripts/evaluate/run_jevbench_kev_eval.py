#!/usr/bin/env python3
"""Evaluate a ship-finetuned Kev checkpoint with the upstream JevBench scorer."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset.jevbench import BENCHMARK_COMMIT, jevbench_probabilities, kev_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--limit", type=int, help="Smoke only: first N items")
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.tasks import load_jsonl, sha256_file
    from kev.checkpoint import LoadOptions
    from kev.model import ContextOverflow
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT, write_json

    tasks = load_jsonl(str(args.data))
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("empty dataset or duplicate task IDs")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    requests = [kev_record(task) for task in tasks]
    args.output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), project / "src/dataset/jevbench.py",
                    project / "scripts/slurm/v046-kev-jevbench.sbatch"]
    manifest = {
        "revision": args.output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run), "data": str(args.data), "record_count": len(tasks),
        "limit": args.limit, "data_sha256": sha256_file(str(args.data)),
        "benchmark_commit": BENCHMARK_COMMIT, "code_revision": args.code_revision,
        "code_sha256": {str(path.relative_to(project)): sha256_file(str(path)) for path in source_files},
        "kev_source_sha256": {name: sha256_file(str(Path(__import__('kev').__file__).parent / name))
                              for name in ("api.py", "model.py", "predictors.py", "checkpoint.py")},
        "checkpoint_sha256": {name: sha256_file(str(args.run / name))
                              for name in ("adapter_model.safetensors", "head.pt", "training_config.json")},
        "context": SERVING_CONTEXT, "rotations": 1, "date_facts": False,
        "probability_source": "native", "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "protocol": "public-only; upstream JevBench score_task/summarize; no retries or truncation",
        "status": "running", "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(args.output / "run_manifest.json", manifest)
    predictor = LocalPredictor(args.run, args.device, LoadOptions.from_env(), context=SERVING_CONTEXT)
    import torch
    import transformers
    manifest.update(inference_temperature=predictor.temperature,
                    dependencies={"torch": torch.__version__, "transformers": transformers.__version__})
    write_json(args.output / "run_manifest.json", manifest)
    outcomes = []
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for task, request in zip(tasks, requests):
            started = time.perf_counter()
            prediction = None
            error = None
            try:
                prediction = predictor(request)
                probs = jevbench_probabilities(task, prediction)
                scored = score_task(probs, task)
            except ContextOverflow as overflow:
                error = f"ContextOverflow: {overflow}"
                probs = None
                scored = {"valid": False, "strict_valid": False, "renormalized": False,
                          "correct": False, "predicted": None}
            # Unexpected runtime/configuration errors abort; they are not silently
            # recast as model mistakes. Already written evidence remains intact.
            outcome = {
                "task_id": task.id, "family": task.family, "split": task.split, "group": task.group,
                "ok": error is None, "error": error, "probs_as_returned": probs,
                "latency_s": time.perf_counter() - started, "probs_source": "native",
                "model": str(args.run), "cost_usd": None, "cost_basis": "local_compute_unpriced",
                **scored,
            }
            stream.write(json.dumps({**outcome, "prediction": prediction}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            outcomes.append(outcome)
            if len(outcomes) % 10 == 0:
                print(f"evaluated {len(outcomes)}/{len(tasks)}", flush=True)

    report = summarize(tasks, outcomes)
    write_json(args.output / "report.json", report)
    manifest.update(status="completed", valid_records=report["clean"]["n_valid"],
                    rejected_records=sum(not row["ok"] for row in outcomes),
                    finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "run_manifest.json", manifest)
    print(json.dumps({key: report[key] for key in ("n_planned", "n_attempted", "n_valid", "accuracy", "brier_mean", "ece")}, indent=2))


if __name__ == "__main__":
    main()
