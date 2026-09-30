#!/usr/bin/env python3
"""End-to-end natural-language ship-form editing evaluation.

Phase ``predict`` samples KVLCC2 requests from the Kev ship dataset and runs a
local Kev checkpoint on them. Phase ``execute`` projects the predictions into
FFD actions, deforms the frozen hull state, rebuilds the hull, and checks
geometry and constraints. Both phases write into the same output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ("single", "multi_region_single_turn", "same_region_two_turns")
DEFAULT_SEED = 20260930
TURN_PATTERN = re.compile(r"^(.+)/turn_(\d+)/variant_(\d+)$")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sample_experiments(
    data: Path, hull_id: str, per_category: int, seed: int
) -> list[dict]:
    """Draw a deterministic per-category sample of KVLCC2 requests."""
    candidates: dict[str, list] = {category: [] for category in CATEGORIES}
    two_turns: dict[str, dict[int, dict]] = {}
    for record in _jsonl(data):
        meta = record["_meta"]
        if meta["hull_id"] != hull_id:
            continue
        category = meta["interaction_type"]
        if category in ("single", "multi_region_single_turn"):
            candidates[category].append(record)
        elif category == "same_region_two_turns":
            match = TURN_PATTERN.match(meta["task_key"])
            turn = int(match.group(2))
            entry = two_turns.setdefault(meta["sample_id"], {})
            if turn not in entry or meta["variant_index"] < entry[turn]["_meta"]["variant_index"]:
                entry[turn] = record
    rng = random.Random(seed)
    experiments = []
    for category in ("single", "multi_region_single_turn"):
        pool = candidates[category]
        if len(pool) < per_category:
            raise ValueError(f"only {len(pool)} {category} records for {hull_id}")
        for record in rng.sample(pool, per_category):
            experiments.append({
                "experiment_id": record["_meta"]["task_key"].replace("/", "__"),
                "category": category,
                "sample_id": record["_meta"]["sample_id"],
                "records": [record],
            })
    paired = {
        sample_id: turns
        for sample_id, turns in two_turns.items()
        if set(turns) == {0, 1}
    }
    if len(paired) < per_category:
        raise ValueError(f"only {len(paired)} paired two-turn samples for {hull_id}")
    for sample_id in rng.sample(sorted(paired), per_category):
        turns = paired[sample_id]
        experiments.append({
            "experiment_id": sample_id,
            "category": "same_region_two_turns",
            "sample_id": sample_id,
            "records": [turns[0], turns[1]],
        })
    return experiments


def run_predict(args: argparse.Namespace) -> None:
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = {
        "revision": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "run": args.run,
        "hull_id": args.hull,
        "per_category": args.per_category,
        "seed": args.seed,
        "categories": list(CATEGORIES),
        "explicit_range_note": (
            "explicit numeric requests are excluded: the decision interface does "
            "not predict continuous magnitudes or spatial extents"
        ),
        "job_id": None,
        "status": "running",
    }
    _write_json(output / "run_manifest.json", manifest)

    experiments = sample_experiments(args.data, args.hull, args.per_category, args.seed)
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import CONTEXT

    predictor = LocalPredictor(args.run, args.device, LoadOptions.from_env(), context=CONTEXT)
    completed = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for experiment in experiments:
            requests = []
            for record in experiment["records"]:
                item = {"record": record}
                try:
                    prediction = predictor(record)
                    item.update({
                        "status": "completed",
                        "probabilities": prediction["probabilities"],
                        "latency_ms": prediction["latency_ms"],
                        "input_tokens": prediction["input_tokens"],
                    })
                    completed += 1
                except Exception as error:
                    item.update({
                        "status": "failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    })
                requests.append(item)
            stream.write(json.dumps({
                "experiment_id": experiment["experiment_id"],
                "category": experiment["category"],
                "sample_id": experiment["sample_id"],
                "requests": requests,
            }, ensure_ascii=False) + "\n")
            stream.flush()
    manifest.update({
        "status": "completed",
        "experiment_count": len(experiments),
        "completed_requests": completed,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"experiments": len(experiments), "completed_requests": completed}))


def _level_labels(record: dict) -> list[int]:
    count = {"one": 1, "two": 2, "three": 3}[record["questions"]["action_count"]["label"]]
    return [int(record["questions"][f"magnitude_{index}"]["label"]) for index in range(1, count + 1)]


def run_execute(args: argparse.Namespace) -> None:
    output = args.output
    predictions_path = output / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"run the predict phase first: {predictions_path}")
    if (output / "experiments.jsonl").exists():
        raise FileExistsError(output / "experiments.jsonl")

    from benchmarks.unified_action import score_actions, target_actions_from_kev
    from end2end.actions import executable_actions
    from end2end.pipeline import run_turns
    from end2end.state import load_hull_state
    from nurbs_ship_reconstruction.core.geometry import skin_waterlines

    base_waterlines, _state = load_hull_state(args.hull_state)
    base_mesh = skin_waterlines(base_waterlines, samples=args.samples)
    meshes_dir = output / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    base_mesh.export(meshes_dir / "base.stl")

    experiments = []
    for experiment in _jsonl(predictions_path):
        start = time.perf_counter()
        turns = []
        predicted_actions = []
        target_actions = []
        turn_exact = True
        levels_correct = 0
        level_total = 0
        failure = None
        failure_reason = None
        for item in experiment["requests"]:
            record = item["record"]
            targets = target_actions_from_kev(record)
            target_actions.append(targets)
            level_total += len(targets)
            if item["status"] != "completed":
                failure = failure or "prediction_failed"
                failure_reason = failure_reason or item["failure_reason"]
                turn_exact = False
                continue
            try:
                actions = executable_actions(record, item["probabilities"])
            except Exception as error:
                failure = failure or "projection_failed"
                failure_reason = failure_reason or f"{type(error).__name__}: {error}"
                turn_exact = False
                continue
            pairs = [{"region": a.region, "operation": a.operation} for a in actions]
            score = score_actions({"actions": pairs}, targets)
            turn_exact = turn_exact and score["turn_exact"]
            levels = [action.magnitude_level for action in actions]
            levels_correct += sum(
                int(level == label) for level, label in zip(levels, _level_labels(record))
            )
            turns.append(actions)
            predicted_actions.append([asdict(action) for action in actions])

        result = {"status": "skipped", "geometry_valid": False, "constraints_satisfied": False}
        if failure is None:
            result = run_turns(base_waterlines, turns, samples=args.samples)
            if result["status"] != "completed":
                failure = "execution_failed"
            elif not result["geometry_valid"]:
                failure = "geometry_invalid"
            elif not result["constraints_satisfied"]:
                failure = "constraints_violated"
            elif not turn_exact:
                failure = "decision_mismatch"
        final_waterlines = result.pop("final_waterlines", None)
        if final_waterlines is not None:
            skin_waterlines(final_waterlines, samples=args.samples).export(
                meshes_dir / f"{experiment['experiment_id']}.stl"
            )
        e2e_success = failure is None and turn_exact
        experiments.append({
            "experiment_id": experiment["experiment_id"],
            "category": experiment["category"],
            "sample_id": experiment["sample_id"],
            "prompts": [item["record"]["state"] for item in experiment["requests"]],
            "predicted_actions": predicted_actions,
            "target_actions": target_actions,
            "turn_exact": turn_exact,
            "levels_correct": levels_correct,
            "level_total": level_total,
            "execution": result,
            "failure": failure,
            "failure_reason": failure_reason,
            "e2e_success": e2e_success,
            "execute_seconds": time.perf_counter() - start,
            "predict_latency_ms": [
                item.get("latency_ms") for item in experiment["requests"]
            ],
        })

    with (output / "experiments.jsonl").open("w", encoding="utf-8") as stream:
        for experiment in experiments:
            stream.write(json.dumps(experiment, ensure_ascii=False) + "\n")
    _write_json(output / "report.json", _aggregate(experiments))
    print(json.dumps(_aggregate(experiments), ensure_ascii=False, indent=2))


def _aggregate(experiments: list[dict]) -> dict:
    report = {}
    for category in CATEGORIES:
        rows = [e for e in experiments if e["category"] == category]
        if not rows:
            continue
        executed = [e for e in rows if e["execution"]["status"] == "completed"]
        constrained = [
            e for e in executed
            if e["execution"]["volume_errors"] or e["execution"]["deck_line_deviations"]
        ]
        latencies = [
            latency
            for e in rows
            for latency in e["predict_latency_ms"]
            if latency is not None
        ]
        latencies.sort()
        report[category] = {
            "experiments": len(rows),
            "e2e_success": sum(e["e2e_success"] for e in rows),
            "e2e_success_rate": sum(e["e2e_success"] for e in rows) / len(rows),
            "trajectory_exact_rate": sum(e["turn_exact"] for e in rows) / len(rows),
            "execution_success_rate": len(executed) / len(rows),
            "geometry_valid_rate": (
                sum(e["execution"]["geometry_valid"] for e in executed) / len(executed)
                if executed else None
            ),
            "constraints_satisfied_rate": (
                sum(e["execution"]["constraints_satisfied"] for e in constrained) / len(constrained)
                if constrained else None
            ),
            "constrained_experiments": len(constrained),
            "levels_correct": sum(e["levels_correct"] for e in rows),
            "level_total": sum(e["level_total"] for e in rows),
            "failures": {
                reason: sum(e["failure"] == reason for e in rows)
                for reason in (
                    "prediction_failed",
                    "projection_failed",
                    "execution_failed",
                    "geometry_invalid",
                    "constraints_violated",
                    "decision_mismatch",
                )
            },
            "predict_latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
            "predict_latency_ms_p95": (
                latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
                if latencies else None
            ),
            "execute_seconds_mean": (
                sum(e["execute_seconds"] for e in rows) / len(rows)
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    predict = subparsers.add_parser("predict", help="sample requests and run the Kev model")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--run", required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--hull", default="KVLCC2")
    predict.add_argument("--per-category", type=int, default=100)
    predict.add_argument("--seed", type=int, default=DEFAULT_SEED)
    predict.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    predict.set_defaults(func=run_predict)

    execute = subparsers.add_parser("execute", help="execute predicted actions and score geometry")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--hull-state", type=Path, required=True)
    execute.add_argument("--samples", type=int, default=48)
    execute.set_defaults(func=run_execute)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
