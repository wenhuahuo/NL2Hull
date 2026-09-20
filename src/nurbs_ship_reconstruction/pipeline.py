"""Run and persist one reproducible classic-hull benchmark."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .geometry import (
    compare_meshes,
    extract_waterline,
    load_hull,
    skin_waterlines,
    waterline_record,
)
from .plotting import plot_comparison, plot_summary

REVISION = "v010_classic_hulls_nurbs_skinning"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _case_dirs(dataset_dir: Path, selected: list[str] | None) -> list[Path]:
    names = selected or []
    if not names:
        manifest = yaml.safe_load((dataset_dir / "manifest.yaml").read_text())
        names = manifest["hulls"]
    dirs = [dataset_dir / name for name in names]
    missing = [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"unknown hull directories: {missing}")
    return dirs


def _new_run_manifest(
    case_dirs: list[Path], dataset_dir: Path, output_dir: Path, revision: str
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "revision": revision,
        "created_at": now,
        "dataset": str(dataset_dir),
        "output": str(output_dir),
        "job_id": None,
        "cases": [
            {
                "hull_id": path.name,
                "input_stl": None,
                "output_dir": str(output_dir / "hulls" / path.name),
                "status": "pending",
                "failure_reason": None,
            }
            for path in case_dirs
        ],
    }


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    selected: list[str] | None = None,
    overwrite: bool = False,
    level_count: int = 33,
) -> dict:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output_dir}; use a new revision or --overwrite"
        )
    if output_dir.exists() and overwrite:
        # Deliberately do not delete existing evidence. The caller must choose a
        # new revision instead of silently clearing a previous run.
        raise FileExistsError(
            f"refusing to overwrite existing run evidence: {output_dir}; choose a new revision"
        )
    output_dir.mkdir(parents=True)
    case_dirs = _case_dirs(dataset_dir, selected)
    revision = output_dir.name
    manifest = _new_run_manifest(case_dirs, dataset_dir, output_dir, revision)
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    summary_rows: list[dict] = []

    for case_index, case_dir in enumerate(case_dirs):
        case_record = manifest["cases"][case_index]
        case_output = output_dir / "hulls" / case_dir.name
        case_output.mkdir(parents=True)
        case_record["status"] = "running"
        _write_json(manifest_path, manifest)
        try:
            hull = load_hull(case_dir)
            case_record["input_stl"] = str(hull.stl_path)
            levels = np.linspace(0.0, hull.depth, level_count)
            waterlines = [extract_waterline(hull, float(z)) for z in levels]
            reconstruction = skin_waterlines(waterlines)
            metrics = compare_meshes(hull.mesh, reconstruction, waterlines)
            records = [waterline_record(waterline) for waterline in waterlines]
            parameters = {
                "revision": revision,
                "hull_id": hull.hull_id,
                "input_stl": str(hull.stl_path),
                "input_sha256": hull.source_sha256,
                "source_bounds": hull.source_bounds.tolist(),
                "source_length_scale": hull.source_scale,
                "source_centerline_y": hull.centerline_y,
                "normalized_bounds": hull.mesh.bounds.tolist(),
                "coordinate_frame": {
                    "x": "aft=0, bow=1 from source STL bbox",
                    "y": "centerline from source STL bbox; half breadth is abs(y)",
                    "z": "baseline=0 from source STL bbox",
                    "unit": "source-length normalized by L",
                },
                "depth": hull.depth,
                "draft": hull.draft,
                "draft_fraction": hull.draft / hull.depth,
                "level_count": level_count,
                "control_model": "degree-3 clamped NURBS; 8 controls; P4=P5=P6",
                "waterlines": records,
            }
            reconstruction_path = case_output / "reconstructed_hull.stl"
            reconstruction.export(reconstruction_path)
            _write_json(case_output / "source_parameters.json", parameters)
            _write_json(case_output / "metrics.json", metrics)
            plot_comparison(
                hull,
                waterlines,
                reconstruction,
                metrics,
                case_output / "comparison.png",
            )
            summary_rows.append({"hull_id": hull.hull_id, **metrics})
            case_record["status"] = "completed"
        except Exception as error:
            case_record["status"] = "failed"
            case_record["failure_reason"] = f"{type(error).__name__}: {error}"
            _write_json(manifest_path, manifest)
            raise
        finally:
            _write_json(manifest_path, manifest)

    summary_rows.sort(key=lambda row: row["hull_id"])
    _write_json(output_dir / "benchmark_summary.json", summary_rows)
    fieldnames = ["hull_id", *[key for key in summary_rows[0] if key != "hull_id"]]
    with (output_dir / "benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_summary(summary_rows, output_dir / "benchmark_summary.png")
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return {"output_dir": str(output_dir), "cases": len(summary_rows)}
