"""Render KCS-based assets used by the editable NL2Hull framework figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nurbs_ship_reconstruction.core.geometry import extract_waterline, load_hull, skin_waterlines
from nurbs_ship_reconstruction.core.nurbs import evaluate
from nurbs_ship_reconstruction.core.profile import fit_profile
from nurbs_ship_reconstruction.deformation.ffd import FFDAction, apply_ffd_actions


BLUE = "#2b80d7"
LIGHT_BLUE = "#cfe7ff"
ORANGE = "#f28c28"
GREEN = "#55a868"


def _normalise(vertices: np.ndarray) -> np.ndarray:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    result = (vertices - lower) / (upper[0] - lower[0])
    result[:, 1] = (result[:, 1] - result[:, 1].mean()) * 1.5
    result[:, 2] *= 2.3
    return result


def _mesh_for_plot(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = _normalise(np.asarray(mesh.vertices, dtype=float))
    faces = np.asarray(mesh.faces, dtype=int)
    return vertices, faces[::2]


def _cuboid(ax, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, color: str) -> None:
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        ax.plot(*corners[[start, end]].T, color=color, linewidth=1.4, alpha=0.95)


def _control_points(waterlines) -> np.ndarray:
    controls = []
    for waterline in waterlines:
        for curve in (waterline.after, waterline.fore):
            points = np.asarray(curve["control_points"], dtype=float)
            if points.shape[1] == 2:
                points = np.column_stack((points, np.full(len(points), waterline.z)))
            controls.append(points)
    return _normalise(np.vstack(controls))


def _render(mesh: trimesh.Trimesh, output: Path, *, waterlines=None, highlight: bool = False,
            operation: bool = False, parameters: bool = False, edited: bool = False) -> None:
    vertices, faces = _mesh_for_plot(mesh)
    fig = plt.figure(figsize=(5.6, 2.9), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        vertices[faces],
        facecolors=LIGHT_BLUE if not edited else "#d9efff",
        edgecolors=BLUE,
        linewidths=0.22,
        alpha=0.18,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.24, 0.24)
    ax.set_zlim(-0.02, 0.25)
    ax.set_box_aspect((1.0, 0.38, 0.25))
    ax.view_init(elev=18, azim=-116)
    ax.set_axis_off()

    box = (0.83, 1.02, -0.112, 0.112, 0.005, 0.099)
    if highlight or operation or parameters:
        _cuboid(ax, *box, ORANGE if not parameters else GREEN)
    if waterlines is not None:
        points = _control_points(waterlines)
        ax.scatter(*points.T, s=3.2, color=BLUE, depthshade=False, alpha=0.8)
        if highlight or operation:
            selected = points[(points[:, 0] >= 0.83) & (points[:, 2] <= 0.099)]
            ax.scatter(*selected.T, s=7, color=ORANGE, depthshade=False)
    if parameters:
        points = _control_points(waterlines) if waterlines is not None else np.empty((0, 3))
        selected = points[(points[:, 0] >= 0.83) & (points[:, 2] <= 0.099)]
        ax.scatter(*selected.T, s=8, color=GREEN, depthshade=False)
        ax.text2D(0.67, 0.18, "FFD parameters", transform=ax.transAxes, color=GREEN, fontsize=7)
    if operation:
        ax.quiver(0.91, 0.0, 0.064, 0.0, 0.14, 0.042, color=ORANGE, linewidth=2.4, arrow_length_ratio=0.23)
        ax.quiver(0.94, 0.0, 0.058, 0.10, 0.0, 0.0, color=ORANGE, linewidth=2.4, arrow_length_ratio=0.23)
        ax.text2D(0.78, 0.68, "↑", transform=ax.transAxes, color=ORANGE, fontsize=18, weight="bold")
        ax.text2D(0.83, 0.54, "→", transform=ax.transAxes, color=ORANGE, fontsize=18, weight="bold")
    fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _curve(points: np.ndarray, nurbs: dict) -> np.ndarray:
    parameters = np.linspace(0.0, 1.0, 160)
    return evaluate(
        parameters,
        np.asarray(nurbs["control_points"]),
        np.asarray(nurbs["weights"]),
        knots=np.asarray(nurbs["knots"]),
    )


def _section(mesh: trimesh.Trimesh, x: float) -> np.ndarray | None:
    path = mesh.section(plane_normal=[1.0, 0.0, 0.0], plane_origin=[x, 0.0, 0.0])
    if path is None or len(path.entities) == 0:
        return None
    entity = max(path.entities, key=lambda item: len(item.points))
    points = np.asarray(path.vertices[entity.points])
    if len(points) < 4:
        return None
    return points[:, [1, 2]]


def _render_nurbs_profiles(hull, reconstructed, profile, output: Path) -> None:
    """Render the actual KCS surface, fitted center-plane profiles, and sections."""
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.0), dpi=180,
                             gridspec_kw={"width_ratios": [2.7, 1.0, 1.0, 1.0]})
    vertices, faces = _mesh_for_plot(reconstructed)
    triangles = vertices[faces][:, :, [0, 2]]
    axes[0].add_collection(PolyCollection(triangles, facecolors="none", edgecolors=BLUE, linewidths=0.16, alpha=0.75))
    for curve in (profile.stem, profile.stern):
        axes[0].plot(curve[:, 0], curve[:, 1] * 2.3, color=ORANGE, linewidth=0.7, alpha=0.8)
    for curve in (profile.stem_nurbs, profile.stern_nurbs):
        fitted = _curve(profile.stem if curve is profile.stem_nurbs else profile.stern, curve)
        axes[0].plot(fitted[:, 0], fitted[:, 1] * 2.3, color=BLUE, linewidth=1.0)
    axes[0].set_title("KCS hull + NURBS waterlines", fontsize=7)
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.02, 0.25)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    sections = ((0.88, "Bow profile"), (0.50, "Midship profile"), (0.12, "Stern profile"))
    for ax, (x, title) in zip(axes[1:], sections):
        values = _section(hull.mesh, x)
        if values is not None:
            ax.plot(values[:, 0] * 1.5, values[:, 1] * 2.3, color=BLUE, linewidth=0.7)
            ax.scatter(values[::max(1, len(values) // 8), 0] * 1.5,
                       values[::max(1, len(values) // 8), 1] * 2.3, s=5, color=BLUE)
        ax.set_title(title, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _make_meshes(dataset_dir: Path):
    hull = load_hull(dataset_dir / "KCS")
    profile = fit_profile(hull)
    levels = np.linspace(0.0, hull.depth, 33)
    waterlines = [extract_waterline(hull, float(z), profile=profile) for z in levels]
    reconstructed = skin_waterlines(waterlines, samples=96)
    edited_waterlines = apply_ffd_actions(
        waterlines, [FFDAction("bulb", "outward", magnitude_level=5)]
    )
    edited = skin_waterlines(edited_waterlines, samples=96)
    return hull, profile, waterlines, reconstructed, edited


def generate(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hull, profile, waterlines, reconstructed, edited = _make_meshes(dataset_dir)
    _render(reconstructed, output_dir / "kcs_engine.png", waterlines=waterlines, highlight=True)
    _render(reconstructed, output_dir / "kcs_region.png", waterlines=waterlines, highlight=True)
    _render(reconstructed, output_dir / "kcs_operation.png", waterlines=waterlines, highlight=True, operation=True)
    _render(reconstructed, output_dir / "kcs_parameters.png", waterlines=waterlines, parameters=True)
    _render(edited, output_dir / "kcs_edited_hull.png", edited=True)
    _render(reconstructed, output_dir / "kcs_wireframe.png", waterlines=waterlines)
    _render_nurbs_profiles(hull, reconstructed, profile, output_dir / "kcs_nurbs_profiles.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/classic_hulls"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.dataset, args.output)


if __name__ == "__main__":
    main()
