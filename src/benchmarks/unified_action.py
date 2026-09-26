"""Shared minimal FFD action projection and scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

ACTION_FIELDS = ("region", "operation")


def _value(action: dict[str, Any], field: str) -> Any:
    return action.get(field)


def score_actions(
    predicted: dict[str, Any], target_actions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score the common action-count/region/operation projection."""
    actions = predicted.get("actions")
    if not isinstance(actions, list):
        raise ValueError("prediction has no actions list")
    field_totals = {field: len(target_actions) for field in ACTION_FIELDS}
    field_correct = {field: 0 for field in ACTION_FIELDS}
    action_exact = 0
    for index, target in enumerate(target_actions):
        if index >= len(actions) or not isinstance(actions[index], dict):
            continue
        candidate = actions[index]
        if set(candidate) != set(ACTION_FIELDS):
            continue
        exact = True
        for field in ACTION_FIELDS:
            correct = _value(candidate, field) == _value(target, field)
            field_correct[field] += int(correct)
            exact &= correct
        action_exact += int(exact)
    return {
        "action_count_correct": len(actions) == len(target_actions),
        "field_totals": field_totals,
        "field_correct": field_correct,
        "action_exact": action_exact,
        "turn_exact": len(actions) == len(target_actions)
        and action_exact == len(target_actions),
    }


def failed_score(target_actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the all-wrong score used for failed or invalid predictions."""
    return {
        "action_count_correct": False,
        "field_totals": {field: len(target_actions) for field in ACTION_FIELDS},
        "field_correct": {field: 0 for field in ACTION_FIELDS},
        "action_exact": 0,
        "turn_exact": False,
    }


def aggregate_action_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate action scores with every requested record in the denominator."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    for result in results:
        score = result.get("score") if result.get("status") == "completed" else failed_score(result["target_actions"])
        totals["turns"] += 1
        correct["turn_exact"] += int(score["turn_exact"])
        for field, total in score["field_totals"].items():
            totals[field] += total
            correct[field] += score["field_correct"][field]
    return {
        "requested_turns": len(results),
        "completed_turns": sum(result.get("status") == "completed" for result in results),
        "failed_turns": sum(result.get("status") != "completed" for result in results),
        "turn_exact_rate": correct["turn_exact"] / totals["turns"] if totals["turns"] else 0.0,
        "field_accuracy": {
            field: correct[field] / totals[field] if totals[field] else 0.0
            for field in ACTION_FIELDS
        },
        "field_correct": {field: correct[field] for field in ACTION_FIELDS},
        "field_totals": {field: totals[field] for field in ACTION_FIELDS},
    }


def _argmax(distribution: dict[str, Any]) -> str:
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("empty probability distribution")
    return max(distribution, key=lambda key: float(distribution[key]))


def actions_from_kev(
    record: dict[str, Any], probabilities: dict[str, dict[str, float]]
) -> dict[str, list[dict[str, str]]]:
    """Project a Kev/Jev probability prediction to the common action JSON."""
    count_key = _argmax(probabilities["action_count"])
    count_names = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
    if count_key not in count_names:
        raise ValueError(f"unsupported action count key: {count_key}")
    count = count_names[count_key]
    actions = []
    for index in range(1, count + 1):
        actions.append({
            "region": _argmax(probabilities[f"region_{index}"]),
            "operation": _argmax(probabilities[f"operation_{index}"]),
        })
    return {"actions": actions}


def target_actions_from_kev(record: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the common action target from labelled Kev questions."""
    count_names = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}
    count_key = record["questions"]["action_count"]["label"]
    if count_key not in count_names:
        raise ValueError(f"unsupported target action count key: {count_key}")
    return [
        {
            "region": record["questions"][f"region_{index}"]["label"],
            "operation": record["questions"][f"operation_{index}"]["label"],
        }
        for index in range(1, count_names[count_key] + 1)
    ]


def validate_probabilities(
    record: dict[str, Any], prediction: dict[str, dict[str, Any]], *, tolerance: float = 0.005
) -> dict[str, dict[str, float]]:
    """Validate probability output consistently for native and prompted Kev paths."""
    from kev.api import question_keys

    if set(prediction) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    result = {}
    for qid, question in record["questions"].items():
        keys = question_keys(question["type"], question.get("criteria"))
        raw = prediction[qid]
        if set(raw) != set(keys):
            raise ValueError(f"probability keys differ for {qid}")
        values = {key: float(raw[key]) for key in keys}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"invalid probability for {qid}")
        total = sum(values.values())
        if total <= 0 or abs(total - 1.0) > max(1e-5, len(keys) * tolerance + 1e-8):
            raise ValueError(f"invalid probability sum for {qid}: {total}")
        result[qid] = {key: value / total for key, value in values.items()}
    return result
