"""Compare Kev baseline and fine-tuned benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    "acc",
    "nll",
    "ece",
    "brier",
    "score_mae",
    "ranked_probability_score",
    "mean_conf",
    "coverage_at_0_9",
    "accuracy_at_0_9",
    "coverage_at_5pct_error",
    "coverage_at_1pct_error",
    "aurc",
    "error_rate_at_0_9",
    "confidence_bias",
)
LOWER_IS_BETTER = {
    "nll", "ece", "brier", "score_mae", "ranked_probability_score",
    "aurc", "error_rate_at_0_9",
}
HIGHER_IS_BETTER = {
    "acc", "coverage_at_0_9", "accuracy_at_0_9",
    "coverage_at_5pct_error", "coverage_at_1pct_error",
}


def _clean(report: dict[str, Any]) -> dict[str, Any]:
    clean = report.get("clean")
    if not isinstance(clean, dict):
        raise ValueError("benchmark report has no clean metrics")
    return {key: clean.get(key) for key in METRICS if key in clean}


def _delta(before: Any, after: Any) -> Any:
    if before is None or after is None:
        return None
    return after - before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    trained = json.loads(args.trained.read_text())
    before, after = _clean(baseline), _clean(trained)
    comparison = {
        "baseline_report": str(args.baseline),
        "trained_report": str(args.trained),
        "baseline_clean": before,
        "trained_clean": after,
        "delta_trained_minus_baseline": {
            key: _delta(before.get(key), after.get(key))
            for key in sorted(set(before) | set(after))
        },
        "improvement_trained_minus_baseline": {
            key: (
                -_delta(before.get(key), after.get(key))
                if key in LOWER_IS_BETTER
                else _delta(before.get(key), after.get(key))
                if key in HIGHER_IS_BETTER
                else None
            )
            for key in sorted(set(before) | set(after))
        },
        "baseline_coverage": baseline.get("coverage"),
        "trained_coverage": trained.get("coverage"),
        "tasks": {
            "baseline": baseline.get("tasks"),
            "trained": trained.get("tasks"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
