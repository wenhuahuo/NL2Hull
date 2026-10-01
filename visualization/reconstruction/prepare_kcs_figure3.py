#!/usr/bin/env python3
"""Prepare clean, compact assets for Figure 3 (KCS reconstruction).

The profile and reconstruction panels are regenerated from the existing KCS
benchmark evidence with plot titles disabled. The closure diagnostic has no
reproducible plotting source in this repository, so its top title strip is
cropped without altering the plotted content.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nurbs_ship_reconstruction.core.geometry import extract_waterline, load_hull
from nurbs_ship_reconstruction.core.profile import fit_profile
from visualization.reconstruction.plotting import plot_comparison, plot_profile


SOURCE_RUN = ROOT / "outputs" / "v020_watertight_loft"
OUTPUT_DIR = SOURCE_RUN / "figure3_clean"


def main() -> None:
    dataset_dir = ROOT / "datasets" / "classic_hulls"
    hull = load_hull(dataset_dir / "KCS")
    profile = fit_profile(hull)
    waterlines = [
        extract_waterline(hull, float(z), profile=profile)
        for z in np.linspace(0.0, hull.depth, 33)
    ]
    reconstruction = trimesh.load_mesh(
        SOURCE_RUN / "hulls" / "KCS" / "reconstructed_hull.stl",
        process=False,
    )
    stored_metrics = json.loads(
        (SOURCE_RUN / "hulls" / "KCS" / "metrics.json").read_text()
    )
    plot_metrics = {
        "vertex_nn_rmse": stored_metrics["surface_vertex_chamfer_rmse"],
        "vertex_nn_p95": stored_metrics["surface_vertex_hausdorff95"],
        "waterline_width_rmse": stored_metrics["waterline_width_rmse"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_profile(
        hull,
        profile,
        OUTPUT_DIR / "profile_stem_stern.png",
        show_titles=False,
    )
    plot_comparison(
        hull,
        waterlines,
        reconstruction,
        plot_metrics,
        OUTPUT_DIR / "comparison.png",
        show_titles=False,
    )

    closure_source = Image.open(SOURCE_RUN / "closure_details.png").convert("RGB")
    # The source diagnostic has a short title strip above the first axes. Keep
    # the plotted panels and axes intact while removing that nonessential title.
    closure_clean = closure_source.crop((0, 32, closure_source.width, closure_source.height))
    closure_clean.save(OUTPUT_DIR / "closure_details.png", optimize=True)

    print(f"Wrote clean Figure 3 assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
