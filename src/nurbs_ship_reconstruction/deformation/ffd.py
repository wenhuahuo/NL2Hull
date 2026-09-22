"""Action-based deformation of fitted hull waterline NURBS parameters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import trimesh

from ..core.geometry import (
    HullInput,
    Waterline,
    _integral_parameters,
    evaluate_waterline,
    evaluate_waterline_3d,
    extract_waterline,
    skin_waterlines,
)

REGION_EXTENTS = {
    "stern": ((0.0, 0.35), (0.0, 1.0)),
    "bow": ((0.65, 1.0), (0.0, 1.0)),
    "bulb": ((0.65, 1.0), (0.0, 0.5)),
    "midbody": ((0.35, 0.65), (0.0, 1.0)),
    "deck": ((0.0, 1.0), (0.65, 1.0)),
    "bilge": ((0.0, 1.0), (0.0, 0.55)),
    "global": ((0.0, 1.0), (0.0, 1.0)),
}

MAGNITUDE_LEVELS = (0.0, 0.01, 0.02, 0.04, 0.08, 0.12)
DIRECTION_OPERATIONS = {
    "outward",
    "inward",
    "upward",
    "downward",
    "forward",
    "aftward",
}
SHAPE_OPERATIONS = {
    "increase_fullness",
    "decrease_fullness",
    "increase_flare",
    "change_bulb_length",
}
GLOBAL_OPERATIONS = {
    "increase_length",
    "decrease_length",
    "increase_breadth",
    "decrease_breadth",
}


@dataclass(frozen=True)
class FFDAction:
    """One typed deformation action on fitted NURBS waterlines.

    ``magnitude_value`` is a fraction of the relevant hull dimension.  For
    example, 0.04 means 4 percent of length, half-breadth, or depth,
    depending on the operation.  ``magnitude_level`` is the optional ordinal
    representation used by a Jev-like decision model.
    """

    region: str
    operation: str
    magnitude_level: int | None = None
    magnitude_value: float | None = None
    longitudinal_extent: tuple[float, float] | None = None
    vertical_extent: tuple[float, float] | None = None
    symmetry: bool = True
    constraints: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FFDAction":
        def extent(name: str) -> tuple[float, float] | None:
            raw = value.get(name)
            if raw is None:
                return None
            if len(raw) != 2:
                raise ValueError(f"{name} must contain two normalized values")
            return float(raw[0]), float(raw[1])

        return cls(
            region=str(value["region"]),
            operation=str(value["operation"]),
            magnitude_level=(
                None
                if value.get("magnitude_level") is None
                else int(value["magnitude_level"])
            ),
            magnitude_value=(
                None
                if value.get("magnitude_value") is None
                else float(value["magnitude_value"])
            ),
            longitudinal_extent=extent("longitudinal_extent"),
            vertical_extent=extent("vertical_extent"),
            symmetry=bool(value.get("symmetry", True)),
            constraints=dict(value.get("constraints", {})),
        )

    def magnitude_fraction(self) -> float:
        if self.magnitude_level is not None:
            if self.magnitude_level < 0 or self.magnitude_level >= len(MAGNITUDE_LEVELS):
                raise ValueError("magnitude_level must be between 0 and 5")
        if self.magnitude_value is not None:
            if self.magnitude_value < 0:
                raise ValueError("magnitude_value must be non-negative")
            return self.magnitude_value
        if self.magnitude_level is None:
            raise ValueError("one of magnitude_level or magnitude_value is required")
        return MAGNITUDE_LEVELS[self.magnitude_level]


def _validate_extent(name: str, extent: tuple[float, float]) -> tuple[float, float]:
    if len(extent) != 2 or extent[0] < 0.0 or extent[1] > 1.0 or extent[0] >= extent[1]:
        raise ValueError(f"{name} must be an increasing interval within [0, 1]")
    return float(extent[0]), float(extent[1])


def _validate_action(action: FFDAction) -> None:
    if action.region not in REGION_EXTENTS:
        raise ValueError(f"unknown deformation region: {action.region}")
    if action.operation not in DIRECTION_OPERATIONS | SHAPE_OPERATIONS | GLOBAL_OPERATIONS:
        raise ValueError(f"unknown deformation operation: {action.operation}")
    if action.longitudinal_extent is not None:
        _validate_extent("longitudinal_extent", action.longitudinal_extent)
    if action.vertical_extent is not None:
        _validate_extent("vertical_extent", action.vertical_extent)
    if not action.symmetry:
        raise ValueError("the fitted waterline representation only supports symmetry=true")
    if action.operation in GLOBAL_OPERATIONS and action.region != "global":
        raise ValueError("global length and breadth operations require region='global'")
    action.magnitude_fraction()


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _enforce_longitudinal_order(controls: np.ndarray) -> None:
    """Keep a half-waterline's NURBS control path longitudinally ordered."""
    direction = 1.0 if controls[-1, 0] >= controls[0, 0] else -1.0
    signed = direction * controls[:, 0]
    endpoint = signed[-1]
    signed = np.minimum(np.maximum.accumulate(signed), endpoint)
    controls[:, 0] = direction * signed


def _window(
    values: np.ndarray,
    extent: tuple[float, float],
    *,
    include_lower: bool = False,
    include_upper: bool = False,
) -> np.ndarray:
    lower, upper = extent
    if lower == 0.0 and upper == 1.0:
        return np.ones_like(values)
    width = upper - lower
    taper = min(width * 0.2, width / 2.0)
    left = _smoothstep((values - lower) / taper)
    right = _smoothstep((upper - values) / taper)
    result = left * right
    if include_lower:
        result[np.isclose(values, lower)] = 1.0
    if include_upper:
        result[np.isclose(values, upper)] = 1.0
    return result


def _state_bounds(waterlines: Sequence[Waterline]) -> tuple[float, float, float, float, float]:
    controls = [
        np.asarray(curve["control_points"], dtype=float)
        for waterline in waterlines
        for curve in (waterline.after, waterline.fore)
    ]
    points = np.vstack(controls)
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    z_values = []
    for waterline in waterlines:
        for curve in (waterline.after, waterline.fore):
            values = np.asarray(curve["control_points"], dtype=float)
            z_values.append(
                values[:, 2]
                if values.shape[1] == 3
                else np.full(len(values), waterline.z)
            )
    z_values = np.concatenate(z_values)
    z_min, z_max = float(z_values.min()), float(z_values.max())
    if x_max <= x_min or z_max <= z_min:
        raise ValueError("waterline parameters have degenerate x or z extent")
    return x_min, x_max, z_min, z_max, float(np.max(np.abs(points[:, 1])))


def _promote_controls(curve: dict, z: float) -> np.ndarray:
    controls = np.asarray(curve["control_points"], dtype=float)
    if controls.ndim != 2 or controls.shape[1] not in (2, 3):
        raise ValueError("waterline NURBS controls must be 2-D or 3-D")
    if controls.shape[1] == 2:
        return np.column_stack((controls, np.full(len(controls), z)))
    return controls.copy()


def _set_controls(curve: dict, controls: np.ndarray) -> None:
    curve["control_points"] = controls


def _action_extents(action: FFDAction) -> tuple[tuple[float, float], tuple[float, float]]:
    x_extent, z_extent = REGION_EXTENTS[action.region]
    if action.longitudinal_extent is not None:
        x_extent = action.longitudinal_extent
    if action.vertical_extent is not None:
        z_extent = action.vertical_extent
    return x_extent, z_extent


def _action_mask(
    controls: np.ndarray,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    action: FFDAction,
) -> np.ndarray:
    x_extent, z_extent = _action_extents(action)
    x_norm = (controls[:, 0] - x_min) / (x_max - x_min)
    z_norm = (controls[:, 2] - z_min) / (z_max - z_min)
    x_mask = _window(x_norm, x_extent)
    z_mask = _window(
        z_norm,
        z_extent,
        include_lower=z_extent[0] == 0.0,
        include_upper=z_extent[1] == 1.0,
    )
    mask = x_mask * z_mask
    if bool(action.constraints.get("preserve_deck_line", False)):
        mask *= 1.0 - np.clip(z_norm, 0.0, 1.0)
    return mask


def _volume(waterlines: Sequence[Waterline]) -> float:
    areas = []
    levels = []
    for waterline in waterlines:
        x, width = evaluate_waterline(waterline, samples=128)
        x = np.concatenate((x[:128], x[129:]))
        width = np.abs(np.concatenate((width[:128], width[129:])))
        area, _, _ = _integral_parameters(x, width)
        areas.append(area)
        levels.append(waterline.z)
    return float(np.trapezoid(areas, levels))


def _refresh_waterline(waterline: Waterline) -> None:
    path = evaluate_waterline_3d(waterline, samples=128)
    path = np.vstack((path[:128], path[129:]))
    if np.any(np.diff(path[:, 0]) < -1e-10):
        raise ValueError("FFD action produced a non-monotone waterline")
    if np.any(path[:, 1] < -1e-10):
        raise ValueError("FFD action produced negative half-breadth")
    x = path[:, 0]
    width = np.maximum(path[:, 1], 0.0)
    area, centroid_x, centroid_y = _integral_parameters(x, width)
    waterline.x = x
    waterline.width = width
    waterline.center_x = float((x[0] + x[-1]) / 2.0)
    waterline.area = area
    waterline.centroid_x = centroid_x
    waterline.centroid_y = centroid_y


def _curvature_proxy(controls: np.ndarray) -> np.ndarray:
    first = np.diff(controls[:, :2], axis=0)
    second = np.diff(first, axis=0)
    scale = max(float(np.ptp(controls[:, :2], axis=0).max()), 1e-12)
    return np.linalg.norm(second, axis=1) / scale


def _check_curvature(
    before: Sequence[tuple[np.ndarray, np.ndarray]],
    after: Sequence[tuple[np.ndarray, np.ndarray]],
    limit: object,
) -> None:
    threshold = float(limit)
    if threshold < 0.0:
        raise ValueError("max_curvature_change must be non-negative")
    change = []
    for (before_after, before_fore), (after_after, after_fore) in zip(before, after):
        for old, new in ((before_after, after_after), (before_fore, after_fore)):
            old_value = _curvature_proxy(old)
            new_value = _curvature_proxy(new)
            change.append(np.max(np.abs(new_value - old_value)))
    if max(change, default=0.0) > threshold:
        raise ValueError("FFD action exceeds max_curvature_change")


def _apply_action(waterlines: list[Waterline], action: FFDAction) -> list[Waterline]:
    _validate_action(action)
    value = action.magnitude_fraction()
    before_volume = _volume(waterlines) if action.constraints.get("preserve_displacement") else None
    before_controls = [
        (
            _promote_controls(waterline.after, waterline.z),
            _promote_controls(waterline.fore, waterline.z),
        )
        for waterline in waterlines
    ]
    x_min, x_max, z_min, z_max, half_breadth = _state_bounds(waterlines)
    length = x_max - x_min
    depth = z_max - z_min
    if half_breadth <= 0.0:
        raise ValueError("waterline parameters have no positive half-breadth")

    if action.operation in {"upward", "downward"} and action.constraints.get("preserve_draft"):
        raise ValueError("vertical FFD actions conflict with preserve_draft=true")

    for waterline in waterlines:
        for curve in (waterline.after, waterline.fore):
            controls = _promote_controls(curve, waterline.z)
            mask = _action_mask(controls, x_min, x_max, z_min, z_max, action)
            lateral_factor = np.clip(controls[:, 1] / half_breadth, 0.0, 1.0)
            if action.operation in {"outward", "increase_fullness", "increase_flare"}:
                controls[:, 1] += value * half_breadth * mask * lateral_factor
            elif action.operation in {"inward", "decrease_fullness"}:
                controls[:, 1] -= value * half_breadth * mask * lateral_factor
            elif action.operation == "upward":
                controls[:, 2] += value * depth * mask
            elif action.operation == "downward":
                controls[:, 2] -= value * depth * mask
            elif action.operation in {"forward", "aftward", "change_bulb_length"}:
                x_extent, z_extent = _action_extents(action)
                x_norm = (controls[:, 0] - x_min) / length
                z_norm = (controls[:, 2] - z_min) / depth
                z_mask = _window(
                    z_norm,
                    z_extent,
                    include_lower=z_extent[0] == 0.0,
                    include_upper=z_extent[1] == 1.0,
                )
                if action.constraints.get("preserve_deck_line"):
                    z_mask *= 1.0 - np.clip(z_norm, 0.0, 1.0)
                sign = 1.0 if action.operation in {"forward", "change_bulb_length"} else -1.0
                if action.region in {"bow", "bulb"}:
                    longitudinal = _smoothstep(
                        (x_norm - x_extent[0]) / (x_extent[1] - x_extent[0])
                    )
                    controls[:, 0] += sign * value * length * longitudinal * z_mask
                elif action.region == "stern":
                    longitudinal = _smoothstep(
                        (x_extent[1] - x_norm) / (x_extent[1] - x_extent[0])
                    )
                    controls[:, 0] += sign * value * length * longitudinal * z_mask
                else:
                    controls[:, 0] += sign * value * length * mask
            elif action.operation == "increase_length":
                center = (x_min + x_max) / 2.0
                controls[:, 0] = center + (controls[:, 0] - center) * (1.0 + value * mask)
            elif action.operation == "decrease_length":
                center = (x_min + x_max) / 2.0
                if value >= 1.0:
                    raise ValueError("decrease_length requires magnitude below 1")
                controls[:, 0] = center + (controls[:, 0] - center) * (1.0 - value * mask)
            elif action.operation == "increase_breadth":
                controls[:, 1] *= 1.0 + value * mask
            elif action.operation == "decrease_breadth":
                if value >= 1.0:
                    raise ValueError("decrease_breadth requires magnitude below 1")
                controls[:, 1] *= 1.0 - value * mask
            if action.operation in {"forward", "aftward", "change_bulb_length"}:
                _enforce_longitudinal_order(controls)
            _set_controls(curve, controls)
        _refresh_waterline(waterline)

    if before_volume is not None:
        after_volume = _volume(waterlines)
        if after_volume <= 0.0:
            raise ValueError("FFD action produced non-positive displacement")
        scale = before_volume / after_volume
        preserve_deck = bool(action.constraints.get("preserve_deck_line", False))
        for waterline in waterlines:
            for curve in (waterline.after, waterline.fore):
                controls = _promote_controls(curve, waterline.z)
                if preserve_deck:
                    z_norm = np.clip((controls[:, 2] - z_min) / depth, 0.0, 1.0)
                    adjustable = z_norm < 1.0 - 1e-10
                    controls[adjustable, 1] *= scale
                else:
                    controls[:, 1] *= scale
                _set_controls(curve, controls)
            _refresh_waterline(waterline)

    if action.constraints.get("max_curvature_change") is not None:
        after_controls = [
            (
                _promote_controls(waterline.after, waterline.z),
                _promote_controls(waterline.fore, waterline.z),
            )
            for waterline in waterlines
        ]
        _check_curvature(
            before_controls,
            after_controls,
            action.constraints["max_curvature_change"],
        )
    return waterlines


def apply_ffd_actions(
    waterlines: Sequence[Waterline],
    actions: Sequence[FFDAction | Mapping[str, object]],
) -> list[Waterline]:
    """Apply a sequence of local or global NURBS actions in order.

    A sequence represents continuous editing.  The input waterlines are
    copied before the first action.
    """

    result = deepcopy(list(waterlines))
    for value in actions:
        action = value if isinstance(value, FFDAction) else FFDAction.from_mapping(value)
        result = _apply_action(result, action)
    return result


def deform_hull(
    hull: HullInput,
    actions: Sequence[FFDAction | Mapping[str, object]],
    *,
    level_count: int = 33,
    bins: int = 512,
    samples: int = 96,
    profile: object | None = None,
) -> tuple[trimesh.Trimesh, list[Waterline]]:
    """Extract NURBS from an STL, deform the parameters, and loft a new STL mesh."""

    if level_count < 2:
        raise ValueError("level_count must be at least 2")
    levels = np.linspace(0.0, hull.depth, level_count)
    waterlines = [
        extract_waterline(hull, float(z), bins=bins, profile=profile) for z in levels
    ]
    deformed = apply_ffd_actions(waterlines, actions)
    mesh = skin_waterlines(deformed, samples=samples)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError("FFD loft did not produce a valid watertight solid")
    return mesh, deformed
