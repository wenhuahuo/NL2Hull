"""Generate long-lived visual evidence for the NURBS FFD action interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ..core.geometry import extract_waterline, load_hull, skin_waterlines
from ..core.profile import fit_profile
from ..deformation.ffd import FFDAction, apply_ffd_actions
from .plotting import plot_ffd_comparison, plot_ffd_gallery

REVISION = "v021_ffd_action_visualization"


def ffd_visualization_cases() -> list[tuple[str, str, FFDAction]]:
    """Return local, global, and shape actions used by the visual audit."""
    cases: list[tuple[str, str, FFDAction]] = []
    view_by_direction = {
        "outward": "plan",
        "inward": "plan",
        "upward": "side",
        "downward": "side",
        "forward": "side",
        "aftward": "side",
    }
    for region in ("bow", "stern", "bulb"):
        for operation, view in view_by_direction.items():
            magnitude_level = 4 if operation in {"forward", "aftward"} else 5
            cases.append(
                (
                    f"local_{region}_{operation}",
                    view,
                    FFDAction(region, operation, magnitude_level=magnitude_level),
                )
            )
    cases.extend(
        [
            (
                "local_midbody_outward",
                "plan",
                FFDAction("midbody", "outward", magnitude_level=5),
            ),
            (
                "local_deck_upward",
                "side",
                FFDAction("deck", "upward", magnitude_level=5),
            ),
            (
                "local_bilge_outward",
                "plan",
                FFDAction("bilge", "outward", magnitude_level=5),
            ),
            (
                "global_increase_length",
                "side",
                FFDAction("global", "increase_length", magnitude_level=5),
            ),
            (
                "global_decrease_length",
                "side",
                FFDAction("global", "decrease_length", magnitude_level=5),
            ),
            (
                "global_increase_breadth",
                "plan",
                FFDAction("global", "increase_breadth", magnitude_level=5),
            ),
            (
                "global_decrease_breadth",
                "plan",
                FFDAction("global", "decrease_breadth", magnitude_level=5),
            ),
            (
                "shape_bow_increase_fullness",
                "plan",
                FFDAction("bow", "increase_fullness", magnitude_level=5),
            ),
            (
                "shape_stern_decrease_fullness",
                "plan",
                FFDAction("stern", "decrease_fullness", magnitude_level=5),
            ),
            (
                "shape_bow_increase_flare",
                "plan",
                FFDAction("bow", "increase_flare", magnitude_level=5),
            ),
            (
                "shape_bulb_change_length",
                "side",
                FFDAction("bulb", "change_bulb_length", magnitude_level=5),
            ),
        ]
    )
    return cases


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_ffd_visualization(
    dataset_dir: Path,
    hull_id: str,
    output_dir: Path,
    *,
    level_count: int = 33,
    samples: int = 96,
) -> dict:
    """Generate one comparison per action and a complete gallery."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()
    hull = load_hull(dataset_dir / hull_id)
    case_definitions = ffd_visualization_cases()
    manifest = {
        "revision": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_stl": str(hull.stl_path),
        "input_sha256": hull.source_sha256,
        "hull_id": hull_id,
        "level_count": level_count,
        "samples_per_half_waterline": samples,
        "job_id": None,
        "status": "running",
        "original_loft": str(output_dir / "original_fitted_nurbs_hull.stl"),
        "gallery": str(output_dir / "ffd_action_gallery.png"),
        "cases": [
            {
                "case_id": case_id,
                "action": asdict(action),
                "preferred_view": view,
                "output_dir": str(cases_dir / case_id),
                "status": "pending",
                "failure_reason": None,
            }
            for case_id, view, action in case_definitions
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)

    profile = fit_profile(hull)
    levels = np.linspace(0.0, hull.depth, level_count)
    original_waterlines = [
        extract_waterline(hull, float(z), profile=profile) for z in levels
    ]
    original_mesh = skin_waterlines(original_waterlines, samples=samples)
    original_mesh.export(output_dir / "original_fitted_nurbs_hull.stl")

    gallery_cases = []
    for index, (case_id, view, action) in enumerate(case_definitions):
        record = manifest["cases"][index]
        case_dir = cases_dir / case_id
        case_dir.mkdir()
        record["status"] = "running"
        _write_json(manifest_path, manifest)
        try:
            deformed_waterlines = apply_ffd_actions(original_waterlines, [action])
            deformed_mesh = skin_waterlines(deformed_waterlines, samples=samples)
            if not deformed_mesh.is_watertight or not deformed_mesh.is_volume:
                raise ValueError("deformed loft is not a valid watertight solid")
            deformed_mesh.export(case_dir / "deformed_hull.stl")
            _write_json(case_dir / "action.json", asdict(action))
            plot_ffd_comparison(
                original_mesh,
                deformed_mesh,
                case_id,
                case_dir / "comparison.png",
            )
            record["metrics"] = {
                "original_bounds": original_mesh.bounds.tolist(),
                "deformed_bounds": deformed_mesh.bounds.tolist(),
                "original_volume": float(original_mesh.volume),
                "deformed_volume": float(deformed_mesh.volume),
                "volume_ratio": float(deformed_mesh.volume / original_mesh.volume),
                "watertight": bool(deformed_mesh.is_watertight),
                "is_volume": bool(deformed_mesh.is_volume),
            }
            record["status"] = "completed"
            gallery_cases.append((case_id, view, deformed_mesh))
        except Exception as error:
            record["status"] = "failed"
            record["failure_reason"] = f"{type(error).__name__}: {error}"
            manifest["status"] = "failed"
            _write_json(manifest_path, manifest)
            raise
        finally:
            _write_json(manifest_path, manifest)

    plot_ffd_gallery(original_mesh, gallery_cases, output_dir / "ffd_action_gallery.png")
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "hull_id": hull_id,
        "cases": len(case_definitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/classic_hulls"),
    )
    parser.add_argument("--hull", default="KVLCC2")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / REVISION,
    )
    parser.add_argument("--levels", type=int, default=33)
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args()
    result = run_ffd_visualization(
        args.dataset,
        args.hull,
        args.output,
        level_count=args.levels,
        samples=args.samples,
    )
    print(
        f"completed {result['cases']} FFD visualizations for "
        f"{result['hull_id']}: {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
