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


def _arc(poly: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(poly)
    forward = [i0]
    i = i0
    while i != i1:
        i = (i + 1) % n
        forward.append(i)
        if len(forward) > n:
            break
    backward = [i0]
    i = i0
    while i != i1:
        i = (i - 1) % n
        backward.append(i)
        if len(backward) > n:
            break
    return poly[np.asarray(forward)], poly[np.asarray(backward)]


def _choose_arc(first: np.ndarray, second: np.ndarray, high_x: bool) -> np.ndarray:
    def key(points: np.ndarray) -> tuple[float, float]:
        mean_x = float(points[:, 0].mean())
        return (mean_x if high_x else -mean_x, -_path_length(points))

    return first if key(first) >= key(second) else second


def _extreme_index(poly: np.ndarray, mask: np.ndarray, take_max: bool) -> int:
    subset = poly[mask]
    if len(subset) == 0:
        raise ValueError("profile band is empty")
    local = int(np.argmax(subset[:, 0]) if take_max else np.argmin(subset[:, 0]))
    return int(np.flatnonzero(mask)[local])


def _split_loop(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = poly[:, 1]
    span = max(float(np.ptp(z)), 1e-12)
    keel = z <= z.min() + 0.12 * span
    deck = z >= z.min() + 0.55 * span
    keel_bow = _extreme_index(poly, keel, True)
    keel_stern = _extreme_index(poly, keel, False)
    deck_bow = _extreme_index(poly, deck, True)
    deck_stern = _extreme_index(poly, deck, False)
    stem = _choose_arc(*_arc(poly, keel_bow, deck_bow), high_x=True)
    stern = _choose_arc(*_arc(poly, keel_stern, deck_stern), high_x=False)
    return stem, stern


def _trim(points: np.ndarray, x_min: float | None, x_max: float | None) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if mask.sum() < 4:
        raise ValueError("trimmed profile contour has too few samples")
    # Keep the longest contiguous run so a loop fragment does not jump.
    runs = np.split(np.flatnonzero(mask), np.where(np.diff(np.flatnonzero(mask)) > 1)[0] + 1)
    keep = max(runs, key=len)
    curve = points[keep]
    if curve[0, 1] > curve[-1, 1]:
        curve = curve[::-1]
    return curve


def extract_stem_stern(hull: HullInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw center-plane points plus ordered stem and stern polylines in x-z."""
    curves = _polylines(hull.mesh)
    raw = np.vstack(curves)
    longest = curves[0]
    is_loop = longest[:, 0].min() < 0.3 and longest[:, 0].max() > 0.7
    if is_loop:
        stem, stern = _split_loop(longest)
    else:
        stem = max(curves[:4], key=lambda points: float(points[:, 0].max()))
        stern = min(curves[:4], key=lambda points: float(points[:, 0].min()))
    return raw, _trim(stem, STEM_X_MIN, None), _trim(stern, None, STERN_X_MAX)


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
