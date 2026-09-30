"""Load frozen hull waterline states for end-to-end editing evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nurbs_ship_reconstruction.core.geometry import Waterline, evaluate_waterline_3d


def _curve(mapping: dict[str, Any]) -> dict[str, Any]:
    curve = dict(mapping)
    curve["control_points"] = np.asarray(mapping["control_points"], dtype=float)
    curve["weights"] = np.asarray(mapping["weights"], dtype=float)
    return curve


def _refresh_derived(waterline: Waterline) -> None:
    """Recompute the sampled fields the same way as the FFD executor."""
    path = evaluate_waterline_3d(waterline, samples=128)
    path = np.vstack((path[:128], path[129:]))
    x = path[:, 0]
    width = np.maximum(path[:, 1], 0.0)
    area = float(np.trapezoid(width, x))
    if area <= 0:
        raise ValueError("waterline area is not positive")
    waterline.x = x
    waterline.width = width
    waterline.center_x = float((x[0] + x[-1]) / 2.0)
    waterline.area = area
    waterline.centroid_x = float(np.trapezoid(x * width, x) / area)
    waterline.centroid_y = float(np.trapezoid(width * width / 2.0, x) / area)


def load_hull_state(path: Path) -> tuple[list[Waterline], dict[str, Any]]:
    """Load a frozen ``hull_states/<hull>.json`` file into Waterline objects."""
    state = json.loads(Path(path).read_text())
    waterlines = []
    for entry in state["waterlines"]:
        waterline = Waterline(
            z=float(entry["z"]),
            center_x=0.0,
            x=np.zeros(0),
            width=np.zeros(0),
            after=_curve(entry["after"]),
            fore=_curve(entry["fore"]),
            area=0.0,
            centroid_x=0.0,
            centroid_y=0.0,
        )
        _refresh_derived(waterline)
        waterlines.append(waterline)
    return waterlines, state
