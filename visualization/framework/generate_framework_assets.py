"""Render KCS hull assets used by the editable NL2Hull framework figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nurbs_ship_reconstruction.core.geometry import extract_waterline, load_hull, skin_waterlines
from nurbs_ship_reconstruction.core.profile import fit_profile
from nurbs_ship_reconstruction.deformation.ffd import FFDAction, apply_ffd_actions


BLUE = "#2b80d7"
LIGHT_BLUE = "#cfe7ff"
ORANGE = "#f28c28"
GREEN = "#55a868"


def _normalise(vertices: np.ndarray) -> np.ndarray:
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    scale = upper[0] - lower[0]
    result = (vertices - lower) / scale
    result[:, 1] = (result[:, 1] - result[:, 1].mean()) * 1.5
    result[:, 2] *= 2.3
    return result


def _mesh_for_plot(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = _normalise(np.asarray(mesh.vertices, dtype=float))
    faces = np.asarray(mesh.faces, dtype=int)
    # The edge mesh is deliberately thinned so the small embedded panels remain legible.
    return vertices, faces[::2]


def _cuboid(ax, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, color: str) -> None:
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        ax.plot(*corners[[start, end]].T, color=color, linewidth=1.4, alpha=0.95)


def _render(
    mesh: trimesh.Trimesh,
    output: Path,
    *,
    highlight: bool = False,
    operation: bool = False,
    parameters: bool = False,
    edited: bool = False,
) -> None:
    vertices, faces = _mesh_for_plot(mesh)
    fig = plt.figure(figsize=(5.6, 2.9), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        vertices[faces],
        facecolors=LIGHT_BLUE if not edited else "#d9efff",
        edgecolors=BLUE,
        linewidths=0.22,
        alpha=0.42,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.24, 0.24)
    ax.set_zlim(-0.02, 0.25)
    ax.set_box_aspect((1.0, 0.38, 0.25))
    ax.view_init(elev=18, azim=-116)
    ax.set_axis_off()

    # The KCS bulb is enclosed by a real 3-D selection box, not a 2-D annotation.
    box = (0.83, 1.02, -0.112, 0.112, 0.005, 0.099)
    if highlight or operation or parameters:
        _cuboid(ax, *box, ORANGE if not parameters else GREEN)
    if parameters:
        rng = np.random.default_rng(7)
        points = np.column_stack(
            [
                rng.uniform(0.84, 1.0, 16),
                rng.uniform(-0.09, 0.09, 16),
                rng.uniform(0.014, 0.092, 16),
            ]
        )
        ax.scatter(*points.T, s=7, color=GREEN, depthshade=False)
    if operation:
        ax.quiver(0.91, 0.0, 0.064, 0.0, 0.14, 0.042, color=ORANGE, linewidth=2.4, arrow_length_ratio=0.23)
        ax.quiver(0.94, 0.0, 0.058, 0.10, 0.0, 0.0, color=ORANGE, linewidth=2.4, arrow_length_ratio=0.23)
        ax.text2D(0.78, 0.68, "↑", transform=ax.transAxes, color=ORANGE, fontsize=18, weight="bold")
        ax.text2D(0.83, 0.54, "→", transform=ax.transAxes, color=ORANGE, fontsize=18, weight="bold")
    fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _make_edited_mesh(dataset_dir: Path) -> trimesh.Trimesh:
    hull = load_hull(dataset_dir / "KCS")
    profile = fit_profile(hull)
    levels = np.linspace(0.0, hull.depth, 33)
    waterlines = [extract_waterline(hull, float(z), profile=profile) for z in levels]
    edited = apply_ffd_actions(
        waterlines,
        [FFDAction("bulb", "outward", magnitude_level=5)],
    )
    return skin_waterlines(edited, samples=96)


def generate(dataset_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hull = load_hull(dataset_dir / "KCS")
    edited = _make_edited_mesh(dataset_dir)
    _render(hull.mesh, output_dir / "kcs_engine.png", highlight=True)
    _render(hull.mesh, output_dir / "kcs_region.png", highlight=True)
    _render(hull.mesh, output_dir / "kcs_operation.png", highlight=True, operation=True)
    _render(hull.mesh, output_dir / "kcs_parameters.png", parameters=True)
    _render(edited, output_dir / "kcs_edited_hull.png", edited=True)
    _render(hull.mesh, output_dir / "kcs_wireframe.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/classic_hulls"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.dataset, args.output)


if __name__ == "__main__":
    main()
