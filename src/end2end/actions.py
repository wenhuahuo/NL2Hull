"""Project Kev probability predictions into executable FFD actions."""

from __future__ import annotations

from typing import Any

from benchmarks.unified_action import actions_from_kev
from nurbs_ship_reconstruction.deformation.ffd import FFDAction

CONSTRAINT_QUESTIONS = ("preserve_displacement", "preserve_deck_line")


def _argmax(distribution: dict[str, Any]) -> str:
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("empty probability distribution")
    return max(distribution, key=lambda key: float(distribution[key]))


def executable_actions(
    record: dict[str, Any], probabilities: dict[str, dict[str, float]]
) -> list[FFDAction]:
    """Extend the shared count/region/operation projection with magnitude and constraints.

    The discrete magnitude level comes from the typed score question; explicit
    numeric magnitudes and spatial extents are outside the current decision
    interface, so actions always use the region default extents.
    """
    pairs = actions_from_kev(record, probabilities)["actions"]
    actions = []
    for index, pair in enumerate(pairs, start=1):
        level = int(_argmax(probabilities[f"magnitude_{index}"]))
        constraints = {}
        for name in CONSTRAINT_QUESTIONS:
            if _argmax(probabilities[f"{name}_{index}"]) == "true":
                constraints[name] = True
        actions.append(
            FFDAction(
                region=pair["region"],
                operation=pair["operation"],
                magnitude_level=level,
                constraints=constraints,
            )
        )
    return actions
