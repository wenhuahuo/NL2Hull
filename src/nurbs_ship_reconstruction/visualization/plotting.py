"""Compact diagnostic plots for a reconstruction run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from ..core.geometry import HullInput, Waterline, evaluate_waterline
from ..core.profile import (
    ProfileFit,
    evaluate_profile_parameters,
    profile_parameter_vector,
)


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


def _colored_mesh(
    ax,
    mesh,
    axis_a: int,
    axis_b: int,
    view_axis: int,
    color: str,
    alpha: float,
) -> None:
    faces = np.asarray(mesh.faces)
    depth = mesh.vertices[faces][:, :, view_axis].mean(axis=1)
    triangles = mesh.vertices[faces[np.argsort(depth)]][:, :, [axis_a, axis_b]]
    ax.add_collection(
        PolyCollection(
            triangles,
            facecolors=color,
            edgecolors="none",
            alpha=alpha,
            rasterized=True,
        )
    )


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
        metrics["vertex_nn_rmse"],
        metrics["vertex_nn_p95"],
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


def plot_ffd_comparison(
    original,
    deformed,
    case_name: str,
    output_path: Path,
) -> None:
    """Plot original NURBS loft and one deformed loft with shared limits."""
    bounds = np.vstack((original.bounds, deformed.bounds))
    limits = []
    for axis in range(3):
        lower, upper = float(bounds[:, axis].min()), float(bounds[:, axis].max())
        pad = max((upper - lower) * 0.04, 1e-4)
        limits.append((lower - pad, upper + pad))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    views = [
        (0, 1, 2, "Plan view", "y / L", limits[0], limits[1]),
        (0, 2, 1, "Side view", "z / L", limits[0], limits[2]),
    ]
    for row, (axis_a, axis_b, view_axis, title, ylabel, xlim, ylim) in enumerate(views):
        original_ax = axes[row, 0]
        _gray_mesh(original_ax, original, axis_a, axis_b, view_axis)
        original_ax.set_title(f"Original fitted NURBS — {title}")

        changed_ax = axes[row, 1]
        _colored_mesh(
            changed_ax,
            original,
            axis_a,
            axis_b,
            view_axis,
            color="0.65",
            alpha=0.30,
        )
        _colored_mesh(
            changed_ax,
            deformed,
            axis_a,
            axis_b,
            view_axis,
            color="tab:blue",
            alpha=0.62,
        )
        changed_ax.set_title(f"Deformed (blue) over original — {title}")
        for ax in (original_ax, changed_ax):
            ax.set_xlabel("x / L")
            ax.set_ylabel(ylabel)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

    fig.suptitle(case_name)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ffd_gallery(
    original,
    cases: list[tuple[str, str, object]],
    output_path: Path,
) -> None:
    """Plot one informative original/deformed overlay for every FFD case."""
    columns = 4
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.2 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for ax, (name, view, deformed) in zip(axes, cases):
        if view == "plan":
            axis_a, axis_b, view_axis = 0, 1, 2
            ylabel = "y / L"
        else:
            axis_a, axis_b, view_axis = 0, 2, 1
            ylabel = "z / L"
        bounds = np.vstack((original.bounds, deformed.bounds))
        xlim = (float(bounds[:, 0].min()), float(bounds[:, 0].max()))
        ylim = (
            float(bounds[:, axis_b].min()),
            float(bounds[:, axis_b].max()),
        )
        _colored_mesh(ax, original, axis_a, axis_b, view_axis, "0.65", 0.28)
        _colored_mesh(ax, deformed, axis_a, axis_b, view_axis, "tab:blue", 0.62)
        ax.set_title(f"{name}\n{view}", fontsize=9)
        ax.set_xlabel("x / L", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    for ax in axes[len(cases) :]:
        ax.axis("off")
    fig.suptitle("FFD action gallery: original gray, deformed blue")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary(rows: list[dict], output_path: Path) -> None:
    names = [row["hull_id"] for row in rows]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for offset, key, label in [
        (0.0, "vertex_nn_rmse", "vertex NN RMSE"),
        (width, "vertex_nn_p95", "vertex NN 95%"),
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
    generated_stem, generated_stern = evaluate_profile_parameters(
        profile_parameter_vector(fit), samples=160
    )
    for ax, name, points, nurbs, generated, color in [
        (axes[0], "Stem contour", fit.stem, fit.stem_nurbs, generated_stem, "tab:red"),
        (axes[1], "Stern contour", fit.stern, fit.stern_nurbs, generated_stern, "tab:blue"),
    ]:
        ax.plot(fit.raw_xz[:, 0], fit.raw_xz[:, 1], ".", ms=0.6, alpha=0.12, color="0.4")
        ax.plot(points[:, 0], points[:, 1], "k.", ms=2.5, label="extracted")
        curve = generated
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
