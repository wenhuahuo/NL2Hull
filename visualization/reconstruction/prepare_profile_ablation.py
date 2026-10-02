#!/usr/bin/env python3
"""Render Figure 7 from archived KCS profile fits; do not refit any variant.

Uniform clamped knots are used only for the two historical runs that recorded
uniform-knot models without serializing knots. Recomputed contour RMSE must
match the recorded value before a panel is exported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nurbs_ship_reconstruction.core.nurbs import evaluate
from nurbs_ship_reconstruction.core.profile import _resample

VARIANTS = (
    ("baseline", "v014_fixed_profile_parameters"),
    ("controls32", "v015_fixed_profile_32_controls"),
    ("deck_knot", "v016_fixed_profile_deck_corner_knot"),
    ("bulb_knots", "v017_fixed_profile_bulb_local_knots"),
    ("combined", "v018b_fixed_profile_combined_features"),
)
OUTPUT = ROOT / "outputs/profile_ablation_figure7"
DPI = 300


def checked_curve(contour: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(contour["extracted"], dtype=float)
    controls = np.asarray(contour["control_points"], dtype=float)
    knots = np.asarray(contour["knots"], dtype=float) if "knots" in contour else None
    if knots is None and "clamped B-spline" not in contour["model"]:
        raise ValueError("Missing archived knot vector for a non-uniform model")
    weights = np.asarray(contour["weights"], dtype=float)
    target = _resample(points, 80)
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(target, axis=0), axis=1))]
    predicted = evaluate(chord / chord[-1], controls, weights, knots=knots)
    rmse = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    if not np.isclose(rmse, contour["fit_rmse"], rtol=1e-8, atol=1e-12):
        raise ValueError(f"Archived fit mismatch: recomputed {rmse}, recorded {contour['fit_rmse']}")
    curve = evaluate(np.linspace(0., 1., 240), controls, weights, knots=knots)
    return points, controls, curve, rmse


def render(record: dict, output: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.45), layout="constrained")
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.025)
    errors = {}
    for ax, name, color in zip(axes, ("stem", "stern"), ("tab:red", "tab:blue")):
        points, controls, curve, rmse = checked_curve(record["profile"][name])
        ax.plot(points[:, 0], points[:, 1], "k.", ms=1.5, label="Extracted contour")
        ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1., label="NURBS fit")
        ax.plot(controls[:, 0], controls[:, 1], "o--", color=color,
                ms=2.5, lw=.6, alpha=.65, label="Control points")
        ax.axhline(record["draft"], color="tab:red", ls="--", lw=.5, label="Design draft")
        ax.set_xlim(points[:, 0].min() - .012, points[:, 0].max() + .012)
        ax.set_ylim(points[:, 1].min() - .012, points[:, 1].max() + .012)
        ax.set_aspect("equal")
        ax.set_xlabel("x / L", labelpad=1)
        ax.set_ylabel("z / L", labelpad=1)
        ax.tick_params(length=2, pad=1)
        ax.legend(loc="upper left" if name == "stem" else "upper right",
                  fontsize=5, framealpha=.9, borderpad=.3, labelspacing=.2,
                  handlelength=1.3)
        errors[name + "_rmse"] = rmse
    fig.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=.02,
                pil_kwargs={"optimize": True})
    plt.close(fig)
    return errors


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.linewidth": .5})
    OUTPUT.mkdir(exist_ok=True)
    entries = []
    for name, run in VARIANTS:
        source = ROOT / "outputs" / run / "hulls/KCS/source_parameters.json"
        record = json.loads(source.read_text())
        errors = render(record, OUTPUT / f"{name}.png")
        entries.append({"variant": name, "source_run": run,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        **errors})
    (OUTPUT / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print("Rendered five archived profile variants at 300 dpi; all contour RMSE checks passed.")


if __name__ == "__main__":
    main()
