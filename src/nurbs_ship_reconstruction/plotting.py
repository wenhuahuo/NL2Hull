"""Compact diagnostic plots for a reconstruction run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import HullInput, Waterline, evaluate_waterline


def plot_comparison(
    hull: HullInput,
    waterlines: list[Waterline],
    reconstruction,
    metrics: dict,
    output_path: Path,
) -> None:
    source = hull.mesh.vertices[hull.mesh.vertices[:, 2] <= hull.draft + 1e-8]
    if len(source) > 12000:
        source = source[np.linspace(0, len(source) - 1, 12000, dtype=int)]
    rebuilt = reconstruction.vertices
    if len(rebuilt) > 12000:
        rebuilt = rebuilt[np.linspace(0, len(rebuilt) - 1, 12000, dtype=int)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(source[:, 0], source[:, 1], s=0.15, alpha=0.25, label="source STL")
    ax.scatter(rebuilt[:, 0], rebuilt[:, 1], s=0.15, alpha=0.25, label="NURBS skin")
    ax.set_title("Plan view (immersed)")
    ax.set_xlabel("x / L")
    ax.set_ylabel("y / L")
    ax.legend(markerscale=10)
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.scatter(source[:, 0], source[:, 2], s=0.15, alpha=0.25, label="source STL")
    ax.scatter(rebuilt[:, 0], rebuilt[:, 2], s=0.15, alpha=0.25, label="NURBS skin")
    ax.axhline(hull.draft, color="tab:red", linestyle="--", linewidth=0.8, label="design draft")
    ax.set_title("Profile view (immersed)")
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / L")
    ax.legend(markerscale=10)

    ax = axes[1, 0]
    selected = np.linspace(0, len(waterlines) - 1, min(6, len(waterlines)), dtype=int)
    for index in selected:
        x, width = evaluate_waterline(waterlines[index], samples=128)
        ax.plot(x, width, label=f"z={waterlines[index].z:.3f}")
        ax.plot(waterlines[index].x, waterlines[index].width, "k.", ms=1.2, alpha=0.35)
    ax.set_title("Waterline extraction and NURBS fit")
    ax.set_xlabel("x / L")
    ax.set_ylabel("half breadth / L")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
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

    fig.suptitle(f"{hull.hull_id}: STL → vertical NURBS parameterization → skin")
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
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
