"""Geometry extraction, NURBS fitting, skinning, and validation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import trimesh
import yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .nurbs import evaluate


@dataclass
class HullInput:
    hull_id: str
    stl_path: Path
    mesh: trimesh.Trimesh
    draft: float
    depth: float
    centerline_y: float
    source_bounds: np.ndarray
    source_scale: float
    source_sha256: str


@dataclass
class Waterline:
    z: float
    center_x: float
    x: np.ndarray
    width: np.ndarray
    after: dict
    fore: dict
    area: float
    centroid_x: float
    centroid_y: float


def _source_stl(hull_dir: Path, metadata: dict) -> Path:
    declared = metadata.get("source", {}).get("stl_file")
    if declared:
        path = hull_dir / declared
        if path.exists():
            return path
    candidates = sorted(
        path
        for path in hull_dir.glob("*.stl")
        if path.name != "canonical.stl"
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one source STL in {hull_dir}, found {candidates}"
        )
    return candidates[0]


def load_hull(hull_dir: Path) -> HullInput:
    with (hull_dir / "benchmark.yaml").open() as stream:
        benchmark = yaml.safe_load(stream)
    with (hull_dir / "metadata.yaml").open() as stream:
        metadata = yaml.safe_load(stream)
    source_path = _source_stl(hull_dir, metadata)
    source_mesh = trimesh.load_mesh(source_path, process=False)
    if not isinstance(source_mesh, trimesh.Trimesh):
        raise TypeError(f"{source_path} is not a triangular STL mesh")
    source_bounds = source_mesh.bounds
    length = float(source_bounds[1, 0] - source_bounds[0, 0])
    if length <= 0:
        raise ValueError(f"source STL has no longitudinal extent: {source_path}")

    # This follows the dataset's length normalization, but is computed from the
    # STL itself so the validation input remains the original STL rather than
    # the STEP-derived canonical file.
    vertices = source_mesh.vertices.astype(float, copy=True)
    vertices[:, 0] = (vertices[:, 0] - source_bounds[0, 0]) / length
    transverse_min, transverse_max = source_bounds[:, 1]
    if transverse_min < 0 < transverse_max:
        centerline_y = (transverse_min + transverse_max) / 2.0
    elif transverse_max <= 0:
        centerline_y = transverse_max
    else:
        centerline_y = transverse_min
    vertices[:, 1] = (vertices[:, 1] - centerline_y) / length
    vertices[:, 2] = (vertices[:, 2] - source_bounds[0, 2]) / length
    mesh = trimesh.Trimesh(vertices=vertices, faces=source_mesh.faces, process=False)

    depth = float(mesh.bounds[1, 2])
    draft_fraction = float(metadata["annotation"]["draft_fraction_of_depth"])
    # KCS and KVLCC2 STL files are independent validation meshes whose scales
    # differ from the reviewed STEP files; the reviewed draft fraction remains
    # the applicable vertical design parameter.
    draft = depth * draft_fraction
    return HullInput(
        hull_id=hull_dir.name,
        stl_path=source_path,
        mesh=mesh,
        draft=draft,
        depth=depth,
        centerline_y=centerline_y,
        source_bounds=source_bounds,
        source_scale=length,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )


def _section_width(section: trimesh.path.Path3D | None, bins: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if section is None or len(section.vertices) < 4:
        raise ValueError("STL waterline has no usable plane intersection")
    points = np.asarray(section.vertices)
    x = points[:, 0]
    width = np.abs(points[:, 1])
    indices = np.floor(np.clip(x, 0.0, 1.0) * bins).astype(int)
    indices = np.minimum(indices, bins - 1)
    values = np.full(bins, -np.inf)
    np.maximum.at(values, indices, width)
    valid = np.isfinite(values)
    if valid.sum() < 4:
        raise ValueError("STL waterline has too few longitudinal samples")
    centers = (np.arange(bins) + 0.5) / bins
    x_valid = centers[valid]
    width_valid = values[valid]
    order = np.argsort(x_valid)
    x_valid, width_valid = x_valid[order], width_valid[order]
    # Plane intersections end slightly inside the bow and stern. The paper's
    # curve model has zero-width endpoints, so add those endpoints explicitly.
    x_valid = np.r_[x_valid[0], x_valid, x_valid[-1]]
    width_valid = np.r_[0.0, width_valid, 0.0]
    unique = np.r_[True, np.diff(x_valid) > 1e-10]
    return x_valid[unique], width_valid[unique]


def _resample_half(
    x: np.ndarray,
    width: np.ndarray,
    start: float,
    end: float,
    samples: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = sorted((start, end))
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 4:
        raise ValueError(f"waterline half-body has too few samples in [{lo}, {hi}]")
    x_part = x[mask]
    width_part = width[mask]
    order = np.argsort(x_part)
    x_part, width_part = x_part[order], width_part[order]
    x_part = np.r_[lo, x_part, hi]
    width_part = np.r_[0.0 if lo != start else width_part[0], width_part, width_part[-1]]
    unique = np.r_[True, np.diff(x_part) > 1e-10]
    x_part, width_part = x_part[unique], width_part[unique]
    grid = np.linspace(start, end, samples)
    return grid, np.interp(grid, x_part, width_part)


def _radius_estimate(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    points = np.c_[x[:3], y[:3]]
    a, b, c = points
    twice_area = np.cross(b - a, c - a)
    if abs(twice_area) < 1e-12:
        return None
    lengths = np.linalg.norm([a - b, b - c, c - a], axis=1)
    return float(np.prod(lengths) / (2.0 * abs(twice_area)))


def _fit_half(x_target: np.ndarray, y_target: np.ndarray) -> dict:
    # Controls are ordered from the outside end toward midships. P4-P6 are
    # coincident, as in the paper's straight-section constraint.
    points = np.c_[x_target, y_target]
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    u_target = np.r_[0.0, np.cumsum(distances)]
    if u_target[-1] <= 0:
        raise ValueError("degenerate waterline half-body")
    u_target /= u_target[-1]
    u_initial = np.linspace(0.0, 1.0, 8)
    initial_controls = np.column_stack(
        [
            np.interp(u_initial, u_target, x_target),
            np.interp(u_initial, u_target, y_target),
        ]
    )
    p0, p7 = points[0], points[-1]
    flat = initial_controls[5]
    p3 = 0.5 * (flat + p7)
    initial_vector = np.r_[
        initial_controls[1], initial_controls[2], p3, flat, [1.0, 1.0, 1.0]
    ]
    x_min, x_max = sorted((float(x_target.min()), float(x_target.max())))
    y_max = max(float(y_target.max()), 1e-8)

    def controls(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1, p2, p3_free, flat_point = vector[:2], vector[2:4], vector[4:6], vector[6:8]
        control = np.array([p0, p1, p2, p3_free, flat_point, flat_point, flat_point, p7])
        weights = np.r_[1.0, vector[8:11], 1.0, 1.0, 1.0, 1.0]
        return control, weights

    def residual(vector: np.ndarray) -> np.ndarray:
        control, weights = controls(vector)
        fitted = evaluate(u_target, control, weights)
        return (fitted - points).ravel()

    lower = np.array([
        x_min, 0.0, x_min, 0.0, x_min, 0.0, x_min, 0.0, 0.05, 0.05, 0.05
    ])
    upper = np.array([
        x_max, y_max * 1.5, x_max, y_max * 1.5, x_max, y_max * 1.5,
        x_max, y_max * 1.5, 50.0, 50.0, 50.0
    ])
    result = least_squares(
        residual,
        np.clip(initial_vector, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=500,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    if not result.success:
        raise RuntimeError(f"NURBS fitting failed: {result.message}")
    control, weights = controls(result.x)
    fitted = evaluate(u_target, control, weights)
    return {
        "control_points": control,
        "weights": weights,
        "parameter": u_target,
        "fit_rmse": float(np.sqrt(np.mean(np.sum((fitted - points) ** 2, axis=1)))),
        "fit_max_error": float(np.max(np.linalg.norm(fitted - points, axis=1))),
    }


def _integral_parameters(x: np.ndarray, width: np.ndarray) -> tuple[float, float, float]:
    area = float(np.trapezoid(width, x))
    if area <= 0:
        raise ValueError("waterline area is not positive")
    centroid_x = float(np.trapezoid(x * width, x) / area)
    centroid_y = float(np.trapezoid(width * width / 2.0, x) / area)
    return area, centroid_x, centroid_y


def extract_waterline(hull: HullInput, z: float, bins: int = 512) -> Waterline:
    # Avoid exact coplanar numerical cases while retaining the requested z in
    # the artifact metadata.
    section_z = z
    if z <= hull.depth / 32.0:
        # A triangulated STL can touch the baseline at only a handful of
        # vertices. Use the first resolvable slice for that boundary row while
        # retaining z=0 in the recorded parameter set.
        section_z = hull.depth / 32.0
    elif z >= hull.depth * (31.0 / 32.0):
        # The deck plane of a triangulated STL is often coplanar with only a
        # handful of faces. Mirror the baseline offset so the top row still
        # cuts a resolvable hull section.
        section_z = hull.depth * (31.0 / 32.0)
    section = hull.mesh.section(
        plane_normal=[0.0, 0.0, 1.0], plane_origin=[0.0, 0.0, section_z]
    )
    x, width = _section_width(section, bins=bins)
    center = float((x[0] + x[-1]) / 2.0)
    after_x, after_y = _resample_half(x, width, float(x[0]), center)
    fore_x, fore_y = _resample_half(x, width, center, float(x[-1]))
    after = _fit_half(after_x, after_y)
    fore = _fit_half(fore_x[::-1], fore_y[::-1])
    area, centroid_x, centroid_y = _integral_parameters(x, width)
    return Waterline(
        z=float(z),
        center_x=center,
        x=x,
        width=width,
        after=after,
        fore=fore,
        area=area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )


def waterline_record(waterline: Waterline) -> dict:
    def serialize_half(half: dict) -> dict:
        controls = half["control_points"]
        outside, center = controls[0], controls[-1]
        flat = controls[4]
        entrance = np.diff(controls[:2], axis=0)[0]
        angle = np.degrees(np.arctan2(abs(entrance[1]), abs(entrance[0])))
        return {
            "control_points": controls.tolist(),
            "weights": half["weights"].tolist(),
            "fit_rmse": half["fit_rmse"],
            "fit_max_error": half["fit_max_error"],
            "length": float(abs(center[0] - outside[0])),
            "beam": float(center[1]),
            "flat_start_x": float(flat[0]),
            "flat_length": float(abs(center[0] - flat[0])),
            "entrance_angle_deg": float(angle),
            "radius_estimate": _radius_estimate(controls[:3, 0], controls[:3, 1]),
        }

    return {
        "z": waterline.z,
        "center_x": waterline.center_x,
        "area": waterline.area,
        "centroid_x": waterline.centroid_x,
        "centroid_y": waterline.centroid_y,
        "raw_waterline": np.c_[waterline.x, waterline.width].tolist(),
        "afterbody": serialize_half(waterline.after),
        "forebody": serialize_half(waterline.fore),
    }


def evaluate_waterline(waterline: Waterline, samples: int = 96) -> tuple[np.ndarray, np.ndarray]:
    after_u = np.linspace(0.0, 1.0, samples)
    fore_u = np.linspace(0.0, 1.0, samples)
    after = evaluate(after_u, waterline.after["control_points"], waterline.after["weights"])
    fore = evaluate(fore_u, waterline.fore["control_points"], waterline.fore["weights"])
    # Both curves are outside-to-midships. Return stern-to-bow ordering.
    path = np.vstack([after, fore[::-1]])
    return path[:, 0], path[:, 1]


def skin_waterlines(waterlines: list[Waterline], samples: int = 96) -> trimesh.Trimesh:
    rows = []
    for waterline in waterlines:
        x, width = evaluate_waterline(waterline, samples=samples)
        positive = np.c_[x, width, np.full_like(x, waterline.z)]
        negative = np.c_[x, -width, np.full_like(x, waterline.z)]
        rows.append((positive, negative))

    vertices = np.vstack([part for positive, negative in rows for part in (positive, negative)])
    row_width = 2 * (2 * samples - 1)
    faces: list[list[int]] = []
    for row in range(len(rows) - 1):
        start_a = row * row_width
        start_b = (row + 1) * row_width
        for column in range(row_width - 1):
            a, b = start_a + column, start_a + column + 1
            c, d = start_b + column, start_b + column + 1
            faces.extend([[a, b, d], [a, d, c]])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _sample_vertices(vertices: np.ndarray, maximum: int = 50000) -> np.ndarray:
    if len(vertices) <= maximum:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, maximum, dtype=int)
    return vertices[indices]


def compare_meshes(
    source: trimesh.Trimesh,
    reconstruction: trimesh.Trimesh,
    waterlines: list[Waterline],
) -> dict:
    source_points = source.vertices[source.vertices[:, 2] <= waterlines[-1].z + 1e-8]
    source_points = _sample_vertices(source_points)
    reconstructed_points = _sample_vertices(reconstruction.vertices)
    source_tree = cKDTree(source_points)
    reconstruction_tree = cKDTree(reconstructed_points)
    source_distances = reconstruction_tree.query(source_points, workers=-1)[0]
    reconstruction_distances = source_tree.query(reconstructed_points, workers=-1)[0]
    all_distances = np.r_[source_distances, reconstruction_distances]

    width_errors = []
    area_relative_errors = []
    centroid_x_errors = []
    centroid_y_errors = []
    for waterline in waterlines:
        x_fit, width_fit = evaluate_waterline(waterline, samples=128)
        width_source = np.interp(x_fit, waterline.x, waterline.width)
        width_errors.extend(width_fit - width_source)
        area_fit, cgx_fit, cgy_fit = _integral_parameters(x_fit, width_fit)
        area_relative_errors.append(abs(area_fit - waterline.area) / waterline.area)
        centroid_x_errors.append(abs(cgx_fit - waterline.centroid_x))
        centroid_y_errors.append(abs(cgy_fit - waterline.centroid_y))

    return {
        "source_vertex_count_compared": int(len(source_points)),
        "reconstruction_vertex_count_compared": int(len(reconstructed_points)),
        "surface_vertex_chamfer_rmse": float(np.sqrt(np.mean(all_distances**2))),
        "surface_vertex_chamfer_mean": float(np.mean(all_distances)),
        "surface_vertex_hausdorff95": float(np.percentile(all_distances, 95)),
        "surface_vertex_max": float(np.max(all_distances)),
        "waterline_width_rmse": float(np.sqrt(np.mean(np.asarray(width_errors) ** 2))),
        "waterline_width_max": float(np.max(np.abs(width_errors))),
        "waterline_area_relative_error_mean": float(np.mean(area_relative_errors)),
        "waterline_area_relative_error_max": float(np.max(area_relative_errors)),
        "waterline_centroid_x_error_mean": float(np.mean(centroid_x_errors)),
        "waterline_centroid_y_error_mean": float(np.mean(centroid_y_errors)),
    }
