#!/usr/bin/env python3
"""Prepare clean, compact assets for Figure 3 (KCS reconstruction).

The profile and reconstruction panels are regenerated from the existing KCS
benchmark evidence with plot titles disabled. The closure diagnostic has no
reproducible plotting source in this repository, so only its title bands are
relabelled without altering the plotted content or interpolating source pixels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.font_manager import findfont
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nurbs_ship_reconstruction.core.geometry import extract_waterline, load_hull
from nurbs_ship_reconstruction.core.profile import fit_profile
from visualization.reconstruction.plotting import plot_comparison, plot_profile


SOURCE_RUN = ROOT / "outputs" / "v020_watertight_loft"
OUTPUT_DIR = SOURCE_RUN / "figure3_clean"


def relabel_closure(source_path: Path, output_path: Path) -> None:
    """Replace title-band text at native resolution; preserve all axes and meshes.

    Each entry gives the four source axes' top edges. Title bands end three
    pixels above the axes, so the plotted content is never covered. The source
    contains six hulls and four views per hull, with no overarching figure title.
    """
    image = Image.open(source_path).convert("RGB")
    if image.size != (1560, 1716):
        raise ValueError("Closure title-band coordinates require the original 1560x1716 image")
    rows = [
        ("KRISO Container Ship (KCS)", [125, 27, 35, 35]),
        ("DTMB 5415", [411, 313, 323, 323]),
        ("KRISO Very Large Crude Carrier 2 (KVLCC2)", [689, 635, 599, 599]),
        ("Generic workboat hull", [965, 885, 885, 885]),
        ("Wigley hull", [1276, 1171, 1207, 1207]),
        ("Generic container-ship hull", [1552, 1467, 1457, 1457]),
    ]
    font = ImageFont.truetype(findfont("DejaVu Sans"), 12)
    draw = ImageDraw.Draw(image)
    for name, tops in rows:
        for column, (view, top) in enumerate(zip(
            ["deck", "bow end", "stern plan", "bow plan"], tops
        )):
            left = column * 390
            draw.rectangle((left, max(0, top - 26), left + 389, top - 3), fill="white")
            draw.text((left + 195, top - 16), f"{name} {view}",
                      anchor="mm", font=font, fill="black")
    # No artificial upsampling: native resolution exceeds 400 ppi at print size.
    image.save(output_path, optimize=True, dpi=(300, 300))


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
        dpi=300,
        compact=True,
    )
    plot_comparison(
        hull,
        waterlines,
        reconstruction,
        plot_metrics,
        OUTPUT_DIR / "comparison.png",
        show_titles=False,
        dpi=300,
        compact=True,
    )

    relabel_closure(
        SOURCE_RUN / "closure_details.png", OUTPUT_DIR / "closure_details.png"
    )

    print(f"Wrote clean Figure 3 assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
