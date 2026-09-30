"""Triptych visualization for end-to-end natural-language editing experiments.

One figure per interaction category: the original hull on the left, the
natural-language request on a central arrow, and the deformed hull on the
right. Long prompts are truncated with an ellipsis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import font_manager

from visualization.reconstruction.plotting import _colored_mesh, _gray_mesh

CATEGORY_TITLES = {
    "single": "Single edit",
    "multi_region_single_turn": "Multi-action edit",
    "same_region_two_turns": "Continuous two-turn edit",
}
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)
PROMPT_LIMIT = 42
PROMPT_WRAP = 14
VIEWS = (
    (0, 1, 2, "y / L", "plan view"),
    (0, 2, 1, "z / L", "side view"),
)


def _cjk_font() -> font_manager.FontProperties:
    for candidate in CJK_FONT_CANDIDATES:
        if Path(candidate).exists():
            return font_manager.FontProperties(fname=candidate)
    raise RuntimeError(
        "no CJK font found; expected one of: " + ", ".join(CJK_FONT_CANDIDATES)
    )


def _prompt_text(prompts: list[str]) -> str:
    lines = []
    for index, prompt in enumerate(prompts, start=1):
        truncated = prompt if len(prompt) <= PROMPT_LIMIT else prompt[: PROMPT_LIMIT - 1] + "…"
        if len(prompts) > 1:
            truncated = f"{'①②③'[index - 1]} {truncated}"
        lines.extend(
            truncated[start : start + PROMPT_WRAP]
            for start in range(0, len(truncated), PROMPT_WRAP)
        )
    return "\n".join(lines)


def select_representative(experiments: list[dict]) -> dict:
    """Pick the successful experiment with the largest control-point change."""
    successful = [e for e in experiments if e["e2e_success"]]
    if not successful:
        raise RuntimeError("no successful experiment available for visualization")
    return max(
        successful,
        key=lambda e: e["execution"]["max_control_displacement"],
    )


def _sections(mesh: trimesh.Trimesh, origin, normal) -> list[np.ndarray]:
    """Return the polylines of one planar mesh section."""
    path = mesh.section(plane_origin=origin, plane_normal=normal)
    if path is None:
        return []
    return [np.asarray(points) for points in path.discrete]


def _overlay_original_sections(
    ax, original: trimesh.Trimesh, view: str, *, levels: int = 6
) -> None:
    """Draw thin original-hull section lines over a deformed-hull panel."""
    z_min, z_max = original.bounds[:, 2]
    if view == "plan":
        for z in np.linspace(z_min, z_max, levels + 2)[1:-1]:
            for points in _sections(original, [0.0, 0.0, z], [0.0, 0.0, 1.0]):
                ax.plot(points[:, 0], points[:, 1], color="tab:red", lw=0.8, alpha=0.85)
    else:
        for points in _sections(original, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
            ax.plot(points[:, 0], points[:, 2], color="tab:red", lw=0.9, alpha=0.9)


def plot_triptych(
    original: trimesh.Trimesh,
    deformed: trimesh.Trimesh,
    prompts: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Render original/deformed hulls in plan and side views around a prompt arrow."""
    cjk = _cjk_font()
    bounds = np.vstack((original.bounds, deformed.bounds))

    fig = plt.figure(figsize=(13, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[2.0, 1.2, 2.0])
    for row, (axis_a, axis_b, view_axis, ylabel, view_name) in enumerate(VIEWS):
        limits = []
        for axis in (axis_a, axis_b):
            lower = float(bounds[:, axis].min())
            upper = float(bounds[:, axis].max())
            pad = max((upper - lower) * 0.04, 1e-4)
            limits.append((lower - pad, upper + pad))
        ax_original = fig.add_subplot(grid[row, 0])
        ax_deformed = fig.add_subplot(grid[row, 2])
        _gray_mesh(ax_original, original, axis_a, axis_b, view_axis)
        ax_original.set_title(f"Original hull — {view_name}")
        _colored_mesh(ax_deformed, deformed, axis_a, axis_b, view_axis, "tab:blue", 0.55)
        _overlay_original_sections(ax_deformed, original, "plan" if view_name == "plan view" else "side")
        ax_deformed.set_title(f"Modified vs original — {view_name}")
        for ax in (ax_original, ax_deformed):
            ax.set_xlabel("x / L")
            ax.set_ylabel(ylabel)
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

    ax_arrow = fig.add_subplot(grid[:, 1])
    ax_arrow.axis("off")
    ax_arrow.annotate(
        "",
        xy=(0.95, 0.35),
        xytext=(0.05, 0.35),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "0.25"},
    )
    ax_arrow.text(
        0.5,
        0.45,
        _prompt_text(prompts),
        ha="center",
        va="bottom",
        fontsize=10,
        fontproperties=cjk,
    )
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(results: Path, output: Path | None = None) -> dict:
    experiments_path = results / "experiments.jsonl"
    if not experiments_path.exists():
        raise FileNotFoundError(experiments_path)
    experiments = [
        json.loads(line) for line in experiments_path.read_text().splitlines() if line.strip()
    ]
    output = output or results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    original = trimesh.load_mesh(results / "meshes" / "base.stl", process=False)
    written = []
    for category, title in CATEGORY_TITLES.items():
        rows = [e for e in experiments if e["category"] == category]
        if not rows:
            continue
        chosen = select_representative(rows)
        mesh_path = results / "meshes" / f"{chosen['experiment_id']}.stl"
        deformed = trimesh.load_mesh(mesh_path, process=False)
        figure = output / f"end2end_{category}.png"
        plot_triptych(
            original,
            deformed,
            chosen["prompts"],
            f"{title} — KVLCC2, Kev-0.8B",
            figure,
        )
        written.append({"category": category, "experiment_id": chosen["experiment_id"], "figure": str(figure)})
    (output / "figures_manifest.json").write_text(
        json.dumps(written, indent=2, ensure_ascii=False) + "\n"
    )
    return {"figures": written}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = run(args.results, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
