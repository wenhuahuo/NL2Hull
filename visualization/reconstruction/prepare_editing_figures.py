#!/usr/bin/env python3
"""Regenerate compact Figure 4/5 panels from stored meshes, without re-evaluation.

All cases/experiment IDs are preserved. Each case occupies a full-width row in
LaTeX. Column/view semantics belong in the caption, not repeated plot titles.
English prompts are display translations of the existing evaluated requests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from visualization.reconstruction.plotting import _colored_mesh, _gray_mesh
from visualization.end2end_test_vis.plot_end2end import (
    _english_prompt,
    _overlay_original_sections,
)

FFD_RUN = ROOT / "outputs/v022_ffd_action_visualization_bulb_fix"
E2E_RUN = ROOT / "outputs/end2end_kvlcc2_kev08"
FFD_CASES = ("local_bow_outward", "local_bulb_forward", "global_increase_breadth")
DPI = 300


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_view(ax, bounds, indices) -> None:
    for setter, index in zip((ax.set_xlim, ax.set_ylim), indices):
        lo, hi = float(bounds[:, index].min()), float(bounds[:, index].max())
        pad = max((hi - lo) * .04, 1e-4)
        setter(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x / L", labelpad=1)
    ax.set_ylabel("y / L" if indices[1] == 1 else "z / L", labelpad=1)
    ax.tick_params(length=2, pad=1)
    ax.grid(alpha=.12, linewidth=.4)


def render_case(original, changed, output: Path, prompts=None) -> None:
    """Two views in each hull column; optional full instruction on centre arrow."""
    bounds = np.vstack((original.bounds, changed.bounds))
    end_to_end = prompts is not None
    fig = plt.figure(figsize=(7.1, 2.25 if end_to_end else 1.9), layout="constrained")
    fig.get_layout_engine().set(w_pad=.03, h_pad=.025, wspace=.04, hspace=.06)
    grid = fig.add_gridspec(
        2, 3 if end_to_end else 2,
        width_ratios=[2, 1.3, 2] if end_to_end else [1, 1],
        height_ratios=[1.3, 1],
    )
    for row, (axis_b, view_axis, name) in enumerate(((1, 2, "plan"), (2, 1, "side"))):
        left = fig.add_subplot(grid[row, 0])
        right = fig.add_subplot(grid[row, -1])
        _gray_mesh(left, original, 0, axis_b, view_axis)
        if end_to_end:
            _colored_mesh(right, changed, 0, axis_b, view_axis, "tab:blue", .55)
            _overlay_original_sections(right, original, name)
        else:
            _colored_mesh(right, original, 0, axis_b, view_axis, "0.65", .30)
            _colored_mesh(right, changed, 0, axis_b, view_axis, "tab:blue", .62)
        for ax in (left, right):
            set_view(ax, bounds, (0, axis_b))

    if end_to_end:
        arrow = fig.add_subplot(grid[:, 1])
        arrow.set_axis_off()
        lines = []
        for index, prompt in enumerate(prompts, 1):
            text = _english_prompt(prompt)
            if len(prompts) > 1:
                text = f"{index}. {text}"
            lines.extend(textwrap.wrap(text, width=30))
        arrow.text(.5, .55, "\n".join(lines), ha="center", va="center",
                   fontsize=6.5, linespacing=1.1, transform=arrow.transAxes)
        arrow.annotate("", xy=(.98, .12), xytext=(.02, .12),
                       xycoords="axes fraction",
                       arrowprops={"arrowstyle": "-|>", "lw": .9, "color": ".3"})
    fig.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=.02,
                pil_kwargs={"optimize": True})
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7,
                         "axes.linewidth": .5, "legend.fontsize": 6})
    ffd_output = FFD_RUN / "figure4_clean"
    e2e_output = E2E_RUN / "figure5_clean"
    ffd_output.mkdir(exist_ok=True)
    e2e_output.mkdir(exist_ok=True)
    entries = []
    base_path = FFD_RUN / "original_fitted_nurbs_hull.stl"
    original = trimesh.load_mesh(base_path, process=False)
    for case in FFD_CASES:
        source = FFD_RUN / "cases" / case / "deformed_hull.stl"
        render_case(original, trimesh.load_mesh(source, process=False), ffd_output / f"{case}.png")
        entries.append({"case": case, "mesh_sha256": sha256(source),
                        "base_sha256": sha256(base_path)})
    (ffd_output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")

    experiments = {e["experiment_id"]: e for e in (
        json.loads(line) for line in (E2E_RUN / "experiments.jsonl").read_text().splitlines()
        if line.strip()
    )}
    selected = json.loads((E2E_RUN / "figures/figures_manifest.json").read_text())
    original_path = E2E_RUN / "meshes/base.stl"
    original = trimesh.load_mesh(original_path, process=False)
    entries = []
    for entry in selected:
        if Path(entry["figure"]).stem.endswith("_alt"):
            continue
        experiment = experiments[entry["experiment_id"]]
        source = E2E_RUN / "meshes" / f"{entry['experiment_id']}.stl"
        render_case(original, trimesh.load_mesh(source, process=False),
                    e2e_output / Path(entry["figure"]).name, experiment["prompts"])
        entries.append({"category": entry["category"],
                        "experiment_id": entry["experiment_id"],
                        "mesh_sha256": sha256(source), "base_sha256": sha256(original_path),
                        "prompts": experiment["prompts"]})
    (e2e_output / "manifest.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print("Regenerated three Figure 4 panels and three Figure 5 panels at 300 dpi.")


if __name__ == "__main__":
    main()
