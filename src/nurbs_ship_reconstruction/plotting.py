"""Compact diagnostic plots for a reconstruction run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .geometry import HullInput, Waterline, evaluate_waterline
from .nurbs import evaluate
from .profile import ProfileFit


def _gray_poly(ax, mesh, axis_a: int, axis_b: int, view_axis: int) -> None:
    faces = np.asarray(mesh.faces)
    depth = mesh.vertices[faces][:, :, view_axis].mean(axis=1)
    order = np.argsort(depth)
    triangles = mesh.vertices[faces[order]][:, :, [axis_a, axis_b]]
    view = np.zeros(3)
    view[view_axis] = 1.0
    shade = 0.40 + 0.45 * np.abs(mesh.face_normals[order] @ view)
    ax.add_collection(
        PolyCollection(
            triangles,
            facecolors=np.c_[shade, shade, shade],
            edgecolors="none",
            rasterized=True,
        )
    )


def _gray_mesh(ax, mesh, axis_a: int, axis_b: int, view_axis: int) -> None:
    _gray_poly(ax, mesh, axis_a, axis_b, view_axis)


def _side_view(ax, mesh, title: str, draft: float, limits) -> None:
    _gray_mesh(ax, mesh, 0, 2, 1)
    draft_line = ax.axhline(draft, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / L")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal")
    ax.legend([draft_line], ["design draft"], fontsize=8, loc="upper right")


def plot_comparison(
    hull: HullInput,
    waterlines: list[Waterline],
    reconstruction,
    metrics: dict,
    output_path: Path,
) -> None:
    source = hull.mesh
    rebuilt = reconstruction
    x_limits = (
        min(source.bounds[0, 0], rebuilt.bounds[0, 0]),
        max(source.bounds[1, 0], rebuilt.bounds[1, 0]),
    )
    y_limits = (
        min(source.bounds[0, 1], rebuilt.bounds[0, 1]),
        max(source.bounds[1, 1], rebuilt.bounds[1, 1]),
    )
    z_limits = (
        min(source.bounds[0, 2], rebuilt.bounds[0, 2]),
        max(source.bounds[1, 2], rebuilt.bounds[1, 2]),
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    ax = axes[0, 0]
    _gray_mesh(ax, source, 0, 1, 2)
    ax.set_title("Original STL plan view")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    _gray_mesh(ax, rebuilt, 0, 1, 2)
    ax.set_title("Lofted gray-mesh plan view")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")

    _side_view(
        axes[1, 0],
        source,
        "Original STL side view",
        hull.draft,
        (x_limits, z_limits),
    )
    _side_view(
        axes[1, 1],
        rebuilt,
        "Lofted gray-mesh side view",
        hull.draft,
        (x_limits, z_limits),
    )

    ax = axes[2, 0]
    selected = np.linspace(0, len(waterlines) - 1, min(6, len(waterlines)), dtype=int)
    for index in selected:
        x, width = evaluate_waterline(waterlines[index], samples=128)
        ax.plot(x, width, label=f"z={waterlines[index].z:.3f}")
        ax.plot(waterlines[index].x, waterlines[index].width, "k.", ms=1.2, alpha=0.35)
    ax.set_title("Waterline extraction and NURBS fit")
    ax.set_xlabel("x / L")
    ax.set_ylabel("half breadth / L")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2, 1]
    names = ["surface\nRMSE", "surface\n95%", "width\nRMSE"]
    values = [
        metrics["surface_vertex_chamfer_rmse"],
        metrics["surface_vertex_hausdorff95"],
        metrics["waterline_width_rmse"],
    ]
    ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_title("Error summary")
    ax.set_ylabel("normalized length")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{hull.hull_id}: STL → vertical NURBS parameterization → lofted hull")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(rows: list[dict], output_path: Path) -> None:
    names = [row["hull_id"] for row in rows]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for offset, key, label in [
        (0.0, "surface_vertex_chamfer_rmse", "surface RMSE"),
        (width, "surface_vertex_hausdorff95", "surface 95%"),
        (2 * width, "waterline_width_rmse", "waterline RMSE"),
    ]:
        ax.bar(x + offset, [row[key] for row in rows], width, label=label)
    ax.set_xticks(x + width, names, rotation=35, ha="right")
    ax.set_ylabel("normalized length")
    ax.set_title("Classic hull validation summary")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profile(hull: HullInput, fit: ProfileFit, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, name, points, nurbs, color in [
        (axes[0], "Stem contour", fit.stem, fit.stem_nurbs, "tab:red"),
        (axes[1], "Stern contour", fit.stern, fit.stern_nurbs, "tab:blue"),
    ]:
        ax.plot(fit.raw_xz[:, 0], fit.raw_xz[:, 1], ".", ms=0.6, alpha=0.12, color="0.4")
        ax.plot(points[:, 0], points[:, 1], "k.", ms=2.5, label="extracted")
        curve = evaluate(
            np.linspace(0.0, 1.0, 160),
            nurbs["control_points"],
            nurbs["weights"],
            knots=nurbs["knots"],
        )
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color=color,
            lw=1.8,
            label=nurbs.get("model", "NURBS fit"),
        )
        ax.plot(
            nurbs["control_points"][:, 0],
            nurbs["control_points"][:, 1],
            "o--",
            color=color,
            ms=4,
            alpha=0.7,
            label="controls",
        )
        ax.axhline(hull.draft, color="tab:red", ls="--", lw=0.7, label="design draft")
        pad = 0.02
        ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
        ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
        ax.set_title(f"{name}  RMSE={nurbs['fit_rmse']:.4f} L")
        ax.set_xlabel("x / L")
        ax.set_ylabel("z / L")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{hull.hull_id}: center-plane stem/stern fixed-dimensional fit")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
