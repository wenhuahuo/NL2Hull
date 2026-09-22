"""Generate reproducible hull-conditioned structured FFD action data."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.geometry import extract_waterline, load_hull, skin_waterlines
from ..core.profile import fit_profile
from ..deformation.ffd import FFDAction, REGION_EXTENTS, apply_ffd_actions

REVISION = "v024_structured_action_dataset_100"
SEED = 20260922
LEVEL_COUNT = 17
SAMPLES_PER_WATERLINE = 48
SAMPLE_COUNT = 100


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _curve_state(curve: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in curve.items()}


def _waterline_state(waterline: Any) -> dict[str, Any]:
    return {
        "z": float(waterline.z),
        "after": _curve_state(waterline.after),
        "fore": _curve_state(waterline.fore),
    }


def _state_digest(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _action_payload(action: FFDAction) -> dict[str, Any]:
    return asdict(action)


def _available_regions(has_bulb: bool) -> list[str]:
    regions = ["bow", "stern", "midbody", "deck", "bilge", "global"]
    if has_bulb:
        regions.insert(2, "bulb")
    return regions


def _operations(region: str) -> list[str]:
    if region == "global":
        return [
            "increase_length",
            "decrease_length",
            "increase_breadth",
            "decrease_breadth",
        ]
    operations = [
        "outward",
        "inward",
        "upward",
        "downward",
        "forward",
        "aftward",
        "increase_fullness",
        "decrease_fullness",
    ]
    if region in {"bow", "stern"}:
        operations.append("increase_flare")
    if region == "bulb":
        operations.append("change_bulb_length")
    return operations


def _constraints(rng: np.random.Generator, operation: str) -> dict[str, object]:
    if operation in {"upward", "downward"}:
        return {"preserve_deck_line": True} if rng.random() < 0.25 else {}
    if operation in {
        "outward",
        "inward",
        "increase_fullness",
        "decrease_fullness",
        "increase_breadth",
        "decrease_breadth",
    } and rng.random() < 0.35:
        return {"preserve_displacement": True}
    return {}


def _level_action(
    rng: np.random.Generator,
    region: str,
    *,
    max_level: int = 5,
    operation: str | None = None,
) -> FFDAction:
    selected_operation = operation or rng.choice(_operations(region))
    return FFDAction(
        region=region,
        operation=selected_operation,
        magnitude_level=int(rng.integers(1, max_level + 1)),
        constraints=_constraints(rng, selected_operation),
    )


def _subextent(
    rng: np.random.Generator, extent: tuple[float, float]
) -> tuple[float, float]:
    lower, upper = extent
    width = upper - lower
    start = lower + float(rng.uniform(0.0, width * 0.35))
    end = upper - float(rng.uniform(0.0, width * 0.25))
    if end - start < width * 0.35:
        start, end = lower, upper
    return round(start, 5), round(end, 5)


def _explicit_action(rng: np.random.Generator, region: str) -> FFDAction:
    operation = rng.choice(_operations(region))
    x_extent, z_extent = REGION_EXTENTS[region]
    return FFDAction(
        region=region,
        operation=operation,
        magnitude_value=round(float(rng.uniform(0.005, 0.05)), 5),
        longitudinal_extent=_subextent(rng, x_extent),
        vertical_extent=_subextent(rng, z_extent),
        constraints=_constraints(rng, operation),
    )


def _metrics(mesh: Any, original_volume: float) -> dict[str, Any]:
    return {
        "bounds": mesh.bounds.tolist(),
        "volume": float(mesh.volume),
        "volume_ratio": float(mesh.volume / original_volume),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
    }


def _prepare_hull_state(dataset_dir: Path, hull_id: str) -> dict[str, Any]:
    hull = load_hull(dataset_dir / hull_id)
    profile = fit_profile(hull)
    levels = np.linspace(0.0, hull.depth, LEVEL_COUNT)
    waterlines = [
        extract_waterline(hull, float(z), profile=profile) for z in levels
    ]
    mesh = skin_waterlines(waterlines, samples=SAMPLES_PER_WATERLINE)
    metadata = yaml.safe_load((dataset_dir / hull_id / "metadata.yaml").read_text())
    annotation = metadata["annotation"]
    state = {
        "hull_id": hull_id,
        "source_stl": str(hull.stl_path),
        "source_sha256": hull.source_sha256,
        "has_bulb": bool(annotation["has_bulb"]),
        "length": 1.0,
        "depth": float(hull.depth),
        "draft": float(hull.draft),
        "waterlines": [_waterline_state(waterline) for waterline in waterlines],
    }
    state["state_sha256"] = _state_digest(state)
    return {
        "hull": hull,
        "waterlines": waterlines,
        "mesh": mesh,
        "state": state,
    }


def _build_candidate(
    index: int,
    hull_data: dict[str, Any],
    sample_type: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    has_bulb = hull_data["state"]["has_bulb"]
    regions = _available_regions(has_bulb)
    non_global = [region for region in regions if region != "global"]

    if sample_type == "single":
        actions = [_level_action(rng, rng.choice(regions))]
        turns = [{"turn_id": 0, "input_count": 1, "output_actions": [0]}]
    elif sample_type == "multi_region_single_turn":
        selected = list(rng.choice(non_global, size=2, replace=False))
        if rng.random() < 0.35:
            selected.append(str(rng.choice([r for r in regions if r not in selected])))
        actions = [_level_action(rng, region, max_level=4) for region in selected]
        turns = [{
            "turn_id": 0,
            "input_count": 1,
            "output_actions": list(range(len(actions))),
        }]
    elif sample_type == "same_region_two_turns":
        region = str(rng.choice(non_global))
        operation = str(rng.choice(_operations(region)))
        first = _level_action(rng, region, max_level=3, operation=operation)
        if rng.random() < 0.3:
            opposite = {
                "outward": "inward",
                "inward": "outward",
                "upward": "downward",
                "downward": "upward",
                "forward": "aftward",
                "aftward": "forward",
            }.get(operation, operation)
            second_operation = opposite
        else:
            second_operation = operation
        second = _level_action(rng, region, max_level=3, operation=second_operation)
        actions = [first, second]
        turns = [
            {"turn_id": 0, "input_count": 1, "output_actions": [0]},
            {"turn_id": 1, "input_count": 1, "output_actions": [1]},
        ]
    elif sample_type == "explicit_range":
        actions = [_explicit_action(rng, str(rng.choice(non_global)))]
        turns = [{"turn_id": 0, "input_count": 1, "output_actions": [0]}]
    else:
        raise ValueError(f"unknown sample type: {sample_type}")

    return {
        "sample_id": f"action_{index:04d}",
        "interaction_type": sample_type,
        "hull_id": hull_data["state"]["hull_id"],
        "base_state_ref": f"hull_states/{hull_data['state']['hull_id']}.json",
        "base_state_sha256": hull_data["state"]["state_sha256"],
        "turns": turns,
        "actions": [_action_payload(action) for action in actions],
    }


def _execute_candidate(
    candidate: dict[str, Any],
    hull_data: dict[str, Any],
) -> dict[str, Any]:
    baseline = hull_data["waterlines"]
    original_mesh = hull_data["mesh"]
    current = baseline
    action_objects = [FFDAction.from_mapping(action) for action in candidate["actions"]]
    outputs = []
    action_cursor = 0
    for turn in candidate["turns"]:
        turn_actions = [
            action_objects[index] for index in turn["output_actions"]
        ]
        current = apply_ffd_actions(current, turn_actions)
        mesh = skin_waterlines(current, samples=SAMPLES_PER_WATERLINE)
        if not mesh.is_watertight or not mesh.is_volume:
            raise ValueError("deformed loft is not a valid watertight solid")
        outputs.append({
            "turn_id": turn["turn_id"],
            "action_indices": turn["output_actions"],
            "metrics": _metrics(mesh, original_mesh.volume),
        })
        action_cursor += len(turn_actions)
    candidate["execution"] = {
        "status": "completed",
        "baseline_metrics": _metrics(original_mesh, original_mesh.volume),
        "turn_results": outputs,
        "final_action_count": action_cursor,
    }
    return candidate


def _check_hull_action_support(hull_data: dict[str, Any]) -> str | None:
    """Return the first unsupported level-5 operation, if any."""
    for region in _available_regions(hull_data["state"]["has_bulb"]):
        for operation in _operations(region):
            try:
                apply_ffd_actions(
                    hull_data["waterlines"],
                    [FFDAction(region, operation, magnitude_level=5)],
                )
            except Exception as error:
                return f"{region}/{operation}: {type(error).__name__}: {error}"
    return None


def generate_action_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Generate 100 valid, hull-conditioned structured action samples."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "hull_states").mkdir()

    manifest_data = yaml.safe_load((dataset_dir / "manifest.yaml").read_text())
    source_hull_ids = list(manifest_data["hulls"])
    all_hull_data = {}
    excluded_hulls = {}
    for hull_id in source_hull_ids:
        prepared = _prepare_hull_state(dataset_dir, hull_id)
        all_hull_data[hull_id] = prepared
        unsupported = _check_hull_action_support(prepared)
        if unsupported is not None:
            excluded_hulls[hull_id] = unsupported
        else:
            (output_dir / "hull_states" / f"{hull_id}.json").write_text(
                json.dumps(prepared["state"], indent=2, ensure_ascii=False) + "\n"
            )
    hull_ids = [hull_id for hull_id in source_hull_ids if hull_id not in excluded_hulls]
    if not hull_ids:
        raise RuntimeError("no hull supports the complete structured action space")
    hull_data = {hull_id: all_hull_data[hull_id] for hull_id in hull_ids}

    manifest = {
        "revision": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "hull_manifest": str(dataset_dir / "manifest.yaml"),
        "source_hull_ids": source_hull_ids,
        "eligible_hull_ids": hull_ids,
        "excluded_hulls": excluded_hulls,
        "sample_count_target": SAMPLE_COUNT,
        "sample_count_completed": 0,
        "seed": seed,
        "level_count": LEVEL_COUNT,
        "samples_per_waterline": SAMPLES_PER_WATERLINE,
        "job_id": None,
        "status": "running",
        "outputs": {
            "records": str(output_dir / "structured_actions.jsonl"),
            "hull_states": str(output_dir / "hull_states"),
        },
        "sample_type_targets": {
            "single": 30,
            "multi_region_single_turn": 25,
            "same_region_two_turns": 25,
            "explicit_range": 20,
        },
        "rejected_candidates": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    rng = np.random.default_rng(seed)
    sample_types = (
        ["single"] * 30
        + ["multi_region_single_turn"] * 25
        + ["same_region_two_turns"] * 25
        + ["explicit_range"] * 20
    )
    rng.shuffle(sample_types)
    records_path = output_dir / "structured_actions.jsonl"
    records = []
    for index, sample_type in enumerate(sample_types, start=1):
        hull_id = hull_ids[(index - 1) % len(hull_ids)]
        for attempt in range(50):
            try:
                candidate = _build_candidate(
                    index, hull_data[hull_id], sample_type, rng
                )
                record = _execute_candidate(candidate, hull_data[hull_id])
                records.append(record)
                break
            except Exception as error:
                manifest["rejected_candidates"].append({
                    "sample_id": f"action_{index:04d}",
                    "hull_id": hull_id,
                    "sample_type": sample_type,
                    "attempt": attempt + 1,
                    "failure_reason": f"{type(error).__name__}: {error}",
                })
        else:
            manifest["status"] = "failed"
            manifest["failure_reason"] = f"could not generate sample action_{index:04d}"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            )
            raise RuntimeError(manifest["failure_reason"])
        manifest["sample_count_completed"] = len(records)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    (output_dir / "structured_actions.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {
        "output_dir": str(output_dir),
        "sample_count": len(records),
        "rejected_candidates": len(manifest["rejected_candidates"]),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/classic_hulls"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / REVISION,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = generate_action_dataset(args.dataset, args.output, seed=args.seed)
    print(
        f"completed {result['sample_count']} structured action samples: "
        f"{result['output_dir']}"
    )


if __name__ == "__main__":
    main()
