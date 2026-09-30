"""Bridge public JevBench tasks to Kev's native typed-decision interface."""

from __future__ import annotations

from typing import Any

BENCHMARK_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"


def kev_record(task: Any) -> dict[str, Any]:
    """Keep state, instructions and option order; translate only label conventions."""
    question = {key: task.question[key] for key in ("type", "instructions", "criteria") if key in task.question}
    if question["type"] == "noul":
        if task.labels != ["no", "yes"]:
            raise ValueError(f"{task.id}: unexpected noul label order")
        label = task.expected == "yes"
    elif question["type"] == "score":
        if task.labels != [str(index) for index in range(len(question["criteria"]))]:
            raise ValueError(f"{task.id}: score labels must be level indices")
        label = task.expected if task.expected is not None else 0
    else:
        if set(question["criteria"]) != set(task.labels):
            raise ValueError(f"{task.id}: choice criteria differ from labels")
        label = task.expected if task.expected is not None else task.labels[0]
    # LocalPredictor materializes labelled records. The placeholder for an unlabelled
    # task never enters the model input or the upstream scorer's ground truth.
    question.update(label=label, src=f"jevbench_{task.family}")
    return {"state": task.state, "questions": {"decision": question}}


def jevbench_probabilities(task: Any, prediction: dict[str, Any]) -> dict[str, float]:
    if set(prediction["probabilities"]) != {"decision"}:
        raise ValueError("Kev answer IDs differ from request")
    probs = prediction["probabilities"]["decision"]
    if task.question["type"] != "noul":
        return probs
    if set(probs) != {"false", "true"}:
        raise ValueError("Kev noul probability keys differ")
    return {"no": probs["false"], "yes": probs["true"]}
