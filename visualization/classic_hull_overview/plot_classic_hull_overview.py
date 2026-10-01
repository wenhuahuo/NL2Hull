#!/usr/bin/env python3
"""Generate Figure 1: an overview of the 12 source hull geometries.

The figure uses the original STL files, normalizes each hull to unit length for
comparison, and shows orthographic side and front projections. VTK is used only
for mesh reading/decimation; matplotlib performs all visual rendering and export.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


HULLS = [
    {
        "id": "DTC",
        "label": "DTC",
        "source": "DTC/DTC-scaled.stl",
    },
    {
        "id": "DTMB5415",
        "label": "DTMB 5415",
        "source": "DTMB5415/DTMB5415.stl",
    },
    {
        "id": "KCS",
        "label": "KCS",
        "source": "KCS/kcs.stl",
    },
    {
        "id": "KVLCC2",
        "label": "KVLCC2",
        "source": "KVLCC2/kvlcc2.stl",
    },
    {
        "id": "containership",
        "label": "Container ship hull",
        "source": "containership/containership.stl",
    },
    {
        "id": "firgate",
        "label": "Frigate hull",
        "source": "firgate/firgate.stl",
    },
    {
        "id": "npl_round_bilge_4a_model",
        "label": "NPL Round Bilge 4a",
        "source": "npl_round_bilge_4a_model/NPL Round Bilge 4a Model.stl",
    },
    {
        "id": "npl_round_bilge_full_scale",
        "label": "NPL Round Bilge",
        "source": "npl_round_bilge_full_scale/NPL Round BilGe Full Scale.stl",
    },
    {
        "id": "s_175_u_water",
        "label": "S-175 container ship",
        "source": "s_175_u_water/s-175-u-water.stl",
    },
    {
        "id": "series60",
        "label": "Series 60 hull",
        "source": "series60/series60.stl",
    },
    {
        "id": "wigley_hull",
        "label": "Wigley hull",
        "source": "wigley_hull/wigley hull.stl",
    },
    {
        "id": "work_boat",
        "label": "Workboat hull",
        "source": "work_boat/work boat.stl",
    },
]


def read_generation_reference_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact surface from the reviewed section reference for S-175.

    The S-175 source STL contains several shells and aggressive triangle
    decimation collapsed its longitudinal extent. The repository's reviewed VTP
    reference contains ordered 32-point sections, so it is used to reconstruct
    a stable overview surface for this figure only.
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    part_array = poly.GetCellData().GetArray("part_id")
    kind_array = poly.GetCellData().GetArray("curve_kind")
    station_array = poly.GetCellData().GetArray("station_fraction")

    sections: dict[int, list[tuple[float, np.ndarray]]] = {}
    for cell_id in range(poly.GetNumberOfCells()):
        if int(kind_array.GetTuple1(cell_id)) != 0:
            continue
        part = int(part_array.GetTuple1(cell_id))
        station = float(station_array.GetTuple1(cell_id))
        cell = poly.GetCell(cell_id)
        points = np.asarray(
            [poly.GetPoint(cell.GetPointIds().GetId(j)) for j in range(cell.GetNumberOfPoints())],
            dtype=float,
        )
        sections.setdefault(part, []).append((station, points))

    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for part_sections in sections.values():
        part_sections.sort(key=lambda item: item[0])
        if len(part_sections) < 2:
            continue
        n_points = len(part_sections[0][1])
        for station, points in part_sections:
            if len(points) != n_points:
                raise ValueError(f"inconsistent section size at station {station}")

        for sign in (1.0, -1.0):
            offset = len(vertices)
            for _, points in part_sections:
                mirrored = points.copy()
                mirrored[:, 1] *= sign
                vertices.extend(mirrored)
            for section_index in range(len(part_sections) - 1):
                start = offset + section_index * n_points
                next_start = start + n_points
                for point_index in range(n_points - 1):
                    a = start + point_index
                    b = start + point_index + 1
                    c = next_start + point_index
                    d = next_start + point_index + 1
                    faces.extend(((a, c, b), (b, c, d)))

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def read_decimated_stl(path: Path, target_faces: int = 4500) -> tuple[np.ndarray, np.ndarray]:
    """Read an STL and return vertices and triangular faces after decimation."""
    if path.name == "s-175-u-water.stl":
        return read_generation_reference_surface(path.parent / "generation_reference.vtp")

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(reader.GetOutputPort())
    triangulate.Update()
    source = triangulate.GetOutput()
    n_faces = source.GetNumberOfPolys()

    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputConnection(triangulate.GetOutputPort())
    if n_faces > target_faces:
        reduction = 1.0 - target_faces / float(n_faces)
        decimator.SetTargetReduction(min(0.9995, max(0.0, reduction)))
    decimator.Update()
    mesh = decimator.GetOutput()

    vertices = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    raw_polys = vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)
    faces = raw_polys[:, 1:4].astype(np.int64)
    return vertices, faces


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Normalize x to unit length while preserving hull aspect ratios."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    length = hi[0] - lo[0]
    if length <= 0:
        raise ValueError("Hull has zero longitudinal extent")
    out = vertices.copy()
    out[:, 0] = (vertices[:, 0] - lo[0]) / length
    out[:, 1] = (vertices[:, 1] - 0.5 * (lo[1] + hi[1])) / length
    out[:, 2] = (vertices[:, 2] - lo[2]) / length
    return out


def projected_polygons(
    vertices: np.ndarray, faces: np.ndarray, view: str
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return projected triangles, grayscale shading, and depth order."""
    tri = vertices[faces]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    normals = np.cross(edge_a, edge_b)
    norm = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norm[:, None], 1e-12)

    if view == "side":
        projected = tri[:, :, (0, 2)]
        illumination = np.abs(normals[:, 1])
        depth = tri[:, :, 1].mean(axis=1)
    elif view == "front":
        projected = tri[:, :, (1, 2)]
        illumination = np.abs(normals[:, 0])
        depth = tri[:, :, 0].mean(axis=1)
    else:
        raise ValueError(f"Unknown view: {view}")

    order = np.argsort(depth)
    projected = projected[order]
    # Use a bright neutral gray and use illumination only to reveal curvature.
    values = 0.68 + 0.25 * illumination[order]
    return [polygon for polygon in projected], values, order


def draw_view(ax, vertices: np.ndarray, faces: np.ndarray, view: str) -> None:
    polygons, values, _ = projected_polygons(vertices, faces, view)
    colors = np.column_stack([values, values, values, np.full(values.shape, 1.0)])
    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        closed=True,
        rasterized=True,
    )
    ax.add_collection(collection)

    if view == "side":
        x = vertices[:, 0]
        z = vertices[:, 2]
        x_margin = max(0.025, 0.045 * (x.max() - x.min()))
        z_margin = max(0.006, 0.08 * (z.max() - z.min()))
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(z.min() - z_margin, z.max() + z_margin)
    else:
        y = vertices[:, 1]
        z = vertices[:, 2]
        y_margin = max(0.008, 0.12 * (y.max() - y.min()))
        z_margin = max(0.006, 0.08 * (z.max() - z.min()))
        ax.set_xlim(y.min() - y_margin, y.max() + y_margin)
        ax.set_ylim(z.min() - z_margin, z.max() + z_margin)

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.text(
        0.98,
        0.97,
        "side" if view == "side" else "front",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#3f4850",
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
        zorder=10,
    )


def build_figure(repo_root: Path, target_faces: int = 4500) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(7.25, 6.9), constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=3,
        ncols=4,
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.955,
        wspace=0.075,
        hspace=0.22,
    )

    for index, spec in enumerate(HULLS):
        row, col = divmod(index, 4)
        cell = outer[row, col].subgridspec(
            2, 1, height_ratios=[1.45, 1.0], hspace=0.04
        )
        source = repo_root / "datasets" / "classic_hulls" / spec["source"]
        vertices, faces = read_decimated_stl(source, target_faces=target_faces)
        vertices = normalize_vertices(vertices)

        title_ax = fig.add_subplot(cell[0])
        title_ax.set_axis_off()
        title_ax.text(
            0.5,
            1.08,
            spec["label"],
            transform=title_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color="#26313a",
        )
        draw_view(title_ax, vertices, faces, "side")

        front_ax = fig.add_subplot(cell[1])
        draw_view(front_ax, vertices, faces, "front")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing datasets/classic_hulls",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for exported figure files",
    )
    parser.add_argument("--target-faces", type=int, default=4500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(args.repo_root, target_faces=args.target_faces)
    stem = args.output_dir / "figure1_classic_hull_overview"
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {stem}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
