"""Extract and fit the paper's stem/stern NURBS contours from the STL center plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import HullInput
from .nurbs import evaluate

CENTER_PLANE_Y = 1e-4
STEM_X_MIN = 0.65
STERN_X_MAX = 0.35
# Paper 2.2: stem has 14 controls with two double vertices; stern has 19
# controls with two triples and one double vertex.
STEM_COINCIDENT = ((4, 5), (9, 10))
STERN_COINCIDENT = ((3, 4, 5), (6, 7, 8), (11, 12), (15, 16, 17))


@dataclass
class ProfileFit:
    raw_xz: np.ndarray
    stem: np.ndarray
    stern: np.ndarray
    stem_nurbs: dict
    stern_nurbs: dict


def _polylines(mesh, min_points: int = 8) -> list[np.ndarray]:
    section = mesh.section(
        plane_normal=[0.0, 1.0, 0.0], plane_origin=[0.0, CENTER_PLANE_Y, 0.0]
    )
    if section is None:
        raise ValueError("STL center plane has no intersection")
    curves = []
    for entity in section.entities:
        if len(entity.points) < min_points:
            continue
        points = np.asarray(section.vertices[entity.points])[:, [0, 2]]
        keep = np.r_[True, np.hypot(*np.diff(points, axis=0).T) > 1e-8]
        curves.append(points[keep])
    if not curves:
        raise ValueError("STL center plane has no usable polylines")
    curves.sort(key=len, reverse=True)
    return curves


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


def _range_segments(
    points: np.ndarray, x_min: float | None, x_max: float | None
) -> list[np.ndarray]:
    mask = np.ones(len(points), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    indices = np.flatnonzero(mask)
    if len(indices) < 4:
        return []
    runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    if np.linalg.norm(points[0] - points[-1]) <= 1e-8 and len(runs) > 1:
        if runs[0][0] == 0 and runs[-1][-1] == len(points) - 1:
            runs = [np.r_[runs[-1], runs[0][1:]], *runs[1:-1]]
    return [points[run] for run in runs if len(run) >= 4]


def _stitch_segments(segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        raise ValueError("profile range has too few samples")
    start = max(range(len(segments)), key=lambda index: _path_length(segments[index]))
    chain = segments[start].copy()
    remaining = [segment for index, segment in enumerate(segments) if index != start]
    while remaining:
        best: tuple[float, int, str, bool] | None = None
        for index, segment in enumerate(remaining):
            candidates = (
                (np.linalg.norm(chain[-1] - segment[0]), "append", False),
                (np.linalg.norm(chain[-1] - segment[-1]), "append", True),
                (np.linalg.norm(chain[0] - segment[-1]), "prepend", False),
                (np.linalg.norm(chain[0] - segment[0]), "prepend", True),
            )
            for distance, side, reverse in candidates:
                candidate = (float(distance), index, side, reverse)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        _, index, side, reverse = best
        segment = remaining.pop(index)
        if reverse:
            segment = segment[::-1]
        if side == "append":
            chain = np.vstack([chain, segment])
        else:
            chain = np.vstack([segment, chain])
    if chain[0, 1] > chain[-1, 1]:
        chain = chain[::-1]
    return chain


def _range_contour(
    curves: list[np.ndarray], x_min: float | None, x_max: float | None
) -> np.ndarray:
    segments = [
        segment
        for curve in curves
        for segment in _range_segments(curve, x_min, x_max)
    ]
    return _stitch_segments(segments)


def extract_stem_stern(hull: HullInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complete ordered center-plane contours within fore/aft x ranges."""
    curves = _polylines(hull.mesh)
    raw = np.vstack(curves)
    stem = _range_contour(curves, STEM_X_MIN, None)
    stern = _range_contour(curves, None, STERN_X_MAX)
    return raw, stem, stern


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    chord = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    if chord[-1] <= 0:
        raise ValueError("degenerate profile contour")
    grid = np.linspace(0.0, chord[-1], count)
    return np.column_stack(
        [np.interp(grid, chord, points[:, 0]), np.interp(grid, chord, points[:, 1])]
    )


def _groups(control_count: int, coincident: tuple[tuple[int, ...], ...]) -> list[int]:
    parent = list(range(control_count))
    for group in coincident:
        root = group[0]
        for index in group[1:]:
            parent[index] = root
    return parent


def _fit_contour(points: np.ndarray, control_count: int, coincident: tuple[tuple[int, ...], ...]) -> dict:
    target = _resample(points, 80)
    chord = np.r_[0.0, np.cumsum(np.hypot(*np.diff(target, axis=0).T))]
    parameter = chord / chord[-1]
    parent = _groups(control_count, coincident)
    free = [i for i in range(control_count) if parent[i] == i and i not in (0, control_count - 1)]
    seeds = _resample(target, control_count)
    for group in coincident:
        seeds[list(group)] = seeds[list(group)].mean(axis=0)
    seeds[0], seeds[-1] = target[0], target[-1]
    initial = seeds[free].ravel()
    x_min, x_max = float(target[:, 0].min()), float(target[:, 0].max())
    z_min, z_max = float(target[:, 1].min()), float(target[:, 1].max())
    if x_max <= x_min:
        x_min, x_max = x_min - 1e-6, x_max + 1e-6
    if z_max <= z_min:
        z_min, z_max = z_min - 1e-6, z_max + 1e-6
    lower = np.tile([x_min, z_min], len(free))
    upper = np.tile([x_max, z_max], len(free))

    def controls(vector: np.ndarray) -> np.ndarray:
        points_ctrl = seeds.copy()
        points_ctrl[free] = vector.reshape(len(free), 2)
        for index, root in enumerate(parent):
            points_ctrl[index] = points_ctrl[root]
        points_ctrl[0], points_ctrl[-1] = target[0], target[-1]
        return points_ctrl

    def residual(vector: np.ndarray) -> np.ndarray:
        fitted = evaluate(parameter, controls(vector))
        return (fitted - target).ravel()

    result = least_squares(
        residual,
        np.clip(initial, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=400,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    if not result.success:
        raise RuntimeError(f"profile NURBS fitting failed: {result.message}")
    control = controls(result.x)
    fitted = evaluate(parameter, control)
    return {
        "control_points": control,
        "weights": np.ones(control_count),
        "parameter": parameter,
        "fit_rmse": float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1)))),
        "fit_max_error": float(np.max(np.linalg.norm(fitted - target, axis=1))),
        "x_keel": float(control[0, 0]),
        "z_keel": float(control[0, 1]),
        "x_deck": float(control[-1, 0]),
        "z_deck": float(control[-1, 1]),
        "x_max": float(np.max(control[:, 0])),
        "x_min": float(np.min(control[:, 0])),
        "z_at_x_max": float(control[np.argmax(control[:, 0]), 1]),
        "z_at_x_min": float(control[np.argmin(control[:, 0]), 1]),
    }


def fit_profile(hull: HullInput) -> ProfileFit:
    raw, stem, stern = extract_stem_stern(hull)
    return ProfileFit(
        raw_xz=raw,
        stem=stem,
        stern=stern,
        stem_nurbs=_fit_contour(stem, 14, STEM_COINCIDENT),
        stern_nurbs=_fit_contour(stern, 19, STERN_COINCIDENT),
    )


def profile_record(fit: ProfileFit) -> dict:
    def pack(name: str, points: np.ndarray, nurbs: dict) -> dict:
        return {
            "extracted": points.tolist(),
            "control_points": nurbs["control_points"].tolist(),
            "weights": nurbs["weights"].tolist(),
            "fit_rmse": nurbs["fit_rmse"],
            "fit_max_error": nurbs["fit_max_error"],
            "x_keel": nurbs["x_keel"],
            "x_deck": nurbs["x_deck"],
            "extent": {
                "x_min": nurbs["x_min"],
                "x_max": nurbs["x_max"],
                "z_at_x_max": nurbs["z_at_x_max"],
                "z_at_x_min": nurbs["z_at_x_min"],
            },
        }

    return {
        "center_plane_y": CENTER_PLANE_Y,
        "stem_model": "degree-3 clamped NURBS; 14 controls; V5=V6, V10=V11",
        "stern_model": "degree-3 clamped NURBS; 19 controls; V4-6, V7-9, V12-13, V16-18 coincident",
        "stem": pack("stem", fit.stem, fit.stem_nurbs),
        "stern": pack("stern", fit.stern, fit.stern_nurbs),
    }
