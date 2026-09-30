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

from visualization.reconstruction.plotting import _gray_mesh

CATEGORY_TITLES = {
    "single": "Single edit",
    "multi_region_single_turn": "Multi-action edit",
    "same_region_two_turns": "Continuous two-turn edit",
}
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
PROMPT_LIMIT = 42
PROMPT_WRAP = 14
VERTICAL_OPERATIONS = {"upward", "downward"}


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


def _view(experiment: dict) -> str:
    for turn in experiment["predicted_actions"]:
        for action in turn or []:
            if action["operation"] in VERTICAL_OPERATIONS:
                return "side"
    return "plan"


def select_representative(experiments: list[dict]) -> dict:
    """Pick the successful experiment with the largest control-point change."""
    successful = [e for e in experiments if e["e2e_success"]]
    if not successful:
        raise RuntimeError("no successful experiment available for visualization")
    return max(
        successful,
        key=lambda e: e["execution"]["max_control_displacement"],
    )


def plot_triptych(
    original: trimesh.Trimesh,
    deformed: trimesh.Trimesh,
    prompts: list[str],
    view: str,
    title: str,
    output_path: Path,
) -> None:
    cjk = _cjk_font()
    bounds = np.vstack((original.bounds, deformed.bounds))
    axis_a, axis_b, view_axis = (0, 1, 2) if view == "plan" else (0, 2, 1)
    ylabel = "y / L" if view == "plan" else "z / L"
    limits = []
    for axis in (axis_a, axis_b):
        lower, upper = float(bounds[:, axis].min()), float(bounds[:, axis].max())
        pad = max((upper - lower) * 0.04, 1e-4)
        limits.append((lower - pad, upper + pad))

    fig = plt.figure(figsize=(13, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.2, 2.0])
    ax_original = fig.add_subplot(grid[0, 0])
    ax_arrow = fig.add_subplot(grid[0, 1])
    ax_deformed = fig.add_subplot(grid[0, 2])

    for ax, mesh, name in (
        (ax_original, original, "Original hull"),
        (ax_deformed, deformed, "Modified hull"),
    ):
        _gray_mesh(ax, mesh, axis_a, axis_b, view_axis)
        ax.set_title(name)
        ax.set_xlabel("x / L")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    ax_arrow.axis("off")
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
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
            _view(chosen),
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
