import json

import numpy as np

from end2end.actions import executable_actions
from end2end.pipeline import run_turns
from end2end.state import load_hull_state
from nurbs_ship_reconstruction.core.geometry import Waterline, skin_waterlines
from nurbs_ship_reconstruction.deformation.ffd import FFDAction, apply_ffd_actions


def _waterline(z: float) -> Waterline:
    after = np.array(
        [[0.0, 0.0], [0.12, 0.04], [0.25, 0.08], [0.32, 0.10],
         [0.4, 0.10], [0.4, 0.10], [0.4, 0.10], [0.5, 0.10]],
        dtype=float,
    )
    fore = np.array(
        [[1.0, 0.0], [0.88, 0.04], [0.75, 0.08], [0.68, 0.10],
         [0.6, 0.10], [0.6, 0.10], [0.6, 0.10], [0.5, 0.10]],
        dtype=float,
    )
    return Waterline(
        z=z,
        center_x=0.5,
        x=np.array([0.0, 0.5, 1.0]),
        width=np.array([0.0, 0.1, 0.0]),
        after={"control_points": after, "weights": np.ones(8)},
        fore={"control_points": fore, "weights": np.ones(8)},
        area=0.1,
        centroid_x=0.5,
        centroid_y=0.05,
    )


def _state() -> list[Waterline]:
    return [_waterline(z) for z in (0.0, 0.5, 1.0)]


def _kev_record() -> dict:
    return {
        "state": "把艏部明显外扩并保持排水量。",
        "questions": {
            "action_count": {
                "type": "choice",
                "instructions": "q",
                "criteria": {"one": "1", "two": "2", "three": "3"},
                "label": "one",
            },
            "region_1": {
                "type": "choice",
                "instructions": "q",
                "criteria": {"bow": "b", "stern": "s"},
                "label": "bow",
            },
            "operation_1": {
                "type": "choice",
                "instructions": "q",
                "criteria": {"outward": "o", "inward": "i"},
                "label": "outward",
            },
            "magnitude_1": {
                "type": "score",
                "instructions": "q",
                "criteria": ["none", "slight", "small", "moderate", "large", "very_large"],
                "label": 4,
            },
            "preserve_displacement_1": {
                "type": "noul",
                "instructions": "q",
                "criteria": {},
                "label": True,
            },
            "preserve_deck_line_1": {
                "type": "noul",
                "instructions": "q",
                "criteria": {},
                "label": False,
            },
        },
    }


def test_executable_actions_include_level_and_constraints():
    record = _kev_record()
    probabilities = {
        "action_count": {"one": 0.9, "two": 0.05, "three": 0.05},
        "region_1": {"bow": 0.8, "stern": 0.2},
        "operation_1": {"outward": 0.7, "inward": 0.3},
        "magnitude_1": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.2, "4": 0.6, "5": 0.1},
        "preserve_displacement_1": {"false": 0.1, "true": 0.9},
        "preserve_deck_line_1": {"false": 0.8, "true": 0.2},
    }
    actions = executable_actions(record, probabilities)
    assert actions == [
        FFDAction(
            region="bow",
            operation="outward",
            magnitude_level=4,
            constraints={"preserve_displacement": True},
        )
    ]


def test_load_hull_state_roundtrip(tmp_path):
    original = _state()
    state = {
        "hull_id": "synthetic",
        "waterlines": [
            {
                "z": waterline.z,
                "after": {
                    "control_points": np.asarray(waterline.after["control_points"]).tolist(),
                    "weights": np.asarray(waterline.after["weights"]).tolist(),
                },
                "fore": {
                    "control_points": np.asarray(waterline.fore["control_points"]).tolist(),
                    "weights": np.asarray(waterline.fore["weights"]).tolist(),
                },
            }
            for waterline in original
        ],
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(state))
    loaded, _ = load_hull_state(path)
    changed = apply_ffd_actions(loaded, [FFDAction("bow", "outward", magnitude_level=3)])
    mesh = skin_waterlines(changed, samples=48)
    assert mesh.is_watertight
    assert loaded[1].area > 0.0
    assert loaded[1].center_x > 0.0


def test_run_turns_reports_metrics_and_displacement():
    result = run_turns(
        _state(),
        [[FFDAction("bow", "outward", magnitude_level=4)]],
        samples=48,
    )
    assert result["status"] == "completed"
    assert result["geometry_valid"]
    assert result["constraints_satisfied"]
    assert result["max_control_displacement"] > 0.0
    assert result["final_metrics"]["volume_ratio"] > 1.0


def test_run_turns_checks_requested_constraints():
    result = run_turns(
        _state(),
        [[FFDAction(
            "global",
            "increase_breadth",
            magnitude_level=3,
            constraints={"preserve_displacement": True},
        )]],
        samples=48,
    )
    assert result["status"] == "completed"
    assert result["volume_errors"]
    assert max(result["volume_errors"]) <= 1e-3
    assert result["constraints_satisfied"]


def test_run_turns_reports_executor_failure():
    result = run_turns(
        _state(),
        [[FFDAction(
            "bulb",
            "upward",
            magnitude_level=2,
            constraints={"preserve_draft": True},
        )]],
        samples=48,
    )
    assert result["status"] == "failed"
    assert "preserve_draft" in result["failure_reason"]
    assert result["completed_turns"] == 0
