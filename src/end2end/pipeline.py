"""Execute predicted FFD action trajectories and measure geometric outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from nurbs_ship_reconstruction.core.geometry import Waterline, skin_waterlines
from nurbs_ship_reconstruction.deformation.ffd import FFDAction, apply_ffd_actions

VOLUME_TOLERANCE = 1e-3
DECK_LINE_TOLERANCE = 1e-9


def mesh_metrics(mesh: Any, original_volume: float) -> dict[str, Any]:
    return {
        "bounds": mesh.bounds.tolist(),
        "volume": float(mesh.volume),
        "volume_ratio": float(mesh.volume / original_volume),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
    }


def _controls(waterline: Waterline) -> np.ndarray:
    parts = []
    for curve in (waterline.after, waterline.fore):
        points = np.asarray(curve["control_points"], dtype=float)
        if points.shape[1] == 2:
            points = np.column_stack((points, np.full(len(points), waterline.z)))
        parts.append(points)
    return np.vstack(parts)


def _top_controls(waterlines: Sequence[Waterline]) -> np.ndarray:
    z_max = max(waterline.z for waterline in waterlines)
    return np.vstack(
        [_controls(waterline) for waterline in waterlines if waterline.z == z_max]
    )


def run_turns(
    base_waterlines: list[Waterline],
    turns: Sequence[Sequence[FFDAction]],
    *,
    samples: int = 48,
) -> dict[str, Any]:
    """Apply turn actions sequentially from the base state and check the lofted hull."""
    if not turns:
        raise ValueError("a trajectory needs at least one turn")
    base_mesh = skin_waterlines(base_waterlines, samples=samples)
    base_volume = float(base_mesh.volume)
    current = base_waterlines
    previous_volume = base_volume
    previous_top = _top_controls(current)
    turn_results = []
    mesh = None
    try:
        for turn_id, actions in enumerate(turns):
            current = apply_ffd_actions(current, actions)
            mesh = skin_waterlines(current, samples=samples)
            result: dict[str, Any] = {
                "turn_id": turn_id,
                "action_count": len(actions),
                "metrics": mesh_metrics(mesh, base_volume),
            }
            if any(action.constraints.get("preserve_displacement") for action in actions):
                result["volume_error"] = abs(float(mesh.volume) / previous_volume - 1.0)
            if any(action.constraints.get("preserve_deck_line") for action in actions):
                top = _top_controls(current)
                result["deck_line_deviation"] = float(np.abs(top - previous_top).max())
            previous_volume = float(mesh.volume)
            previous_top = _top_controls(current)
            turn_results.append(result)
    except Exception as error:
        return {
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "completed_turns": len(turn_results),
            "turns": turn_results,
        }

    geometry_valid = bool(
        mesh.is_watertight and mesh.is_volume and mesh.volume > 0.0
    )
    volume_errors = [t["volume_error"] for t in turn_results if "volume_error" in t]
    deck_deviations = [
        t["deck_line_deviation"] for t in turn_results if "deck_line_deviation" in t
    ]
    constraints_satisfied = all(
        error <= VOLUME_TOLERANCE for error in volume_errors
    ) and all(deviation <= DECK_LINE_TOLERANCE for deviation in deck_deviations)
    max_displacement = float(
        max(
            np.abs(_controls(after) - _controls(before)).max()
            for before, after in zip(base_waterlines, current)
        )
    )
    return {
        "status": "completed",
        "completed_turns": len(turn_results),
        "turns": turn_results,
        "final_metrics": mesh_metrics(mesh, base_volume),
        "max_control_displacement": max_displacement,
        "volume_errors": volume_errors,
        "deck_line_deviations": deck_deviations,
        "geometry_valid": geometry_valid,
        "constraints_satisfied": constraints_satisfied,
        "final_waterlines": current,
    }
