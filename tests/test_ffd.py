import numpy as np
import pytest

from nurbs_ship_reconstruction.core.geometry import Waterline, evaluate_waterline_3d, skin_waterlines
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


def test_local_bow_outward_only_changes_bow_parameters():
    original = _state()
    changed = apply_ffd_actions(
        original,
        [FFDAction("bow", "outward", magnitude_value=0.1)],
    )
    np.testing.assert_allclose(
        changed[1].after["control_points"][:, :2],
        original[1].after["control_points"],
    )
    assert changed[1].fore["control_points"][1, 1] > original[1].fore["control_points"][1, 1]
    np.testing.assert_allclose(
        changed[1].after["control_points"][:3, 1],
        original[1].after["control_points"][:3, 1],
    )


def test_global_length_and_breadth_change_nurbs_parameters():
    original = _state()
    changed = apply_ffd_actions(
        original,
        [
            FFDAction("global", "increase_length", magnitude_value=0.1),
            FFDAction("global", "increase_breadth", magnitude_value=0.1),
        ],
    )
    assert changed[1].after["control_points"][0, 0] < original[1].after["control_points"][0, 0]
    assert changed[1].fore["control_points"][0, 0] > original[1].fore["control_points"][0, 0]
    assert changed[1].after["control_points"][4, 1] > original[1].after["control_points"][4, 1]


def test_vertical_action_promotes_curves_to_3d_and_lofts():
    changed = apply_ffd_actions(
        _state(),
        [FFDAction("bulb", "upward", magnitude_value=0.1)],
    )
    controls = changed[0].fore["control_points"]
    assert controls.shape[1] == 3
    assert controls[1, 2] > 0.0
    mesh = skin_waterlines(changed, samples=96)
    assert mesh.is_watertight
    assert np.ptp(mesh.vertices[:, 2]) > 1.0


def test_preserve_displacement_restores_the_lofted_volume():
    original = _state()
    baseline = skin_waterlines(original, samples=96).volume
    changed = apply_ffd_actions(
        original,
        [
            FFDAction(
                "global",
                "increase_breadth",
                magnitude_value=0.2,
                constraints={"preserve_displacement": True},
            )
        ],
    )
    result = skin_waterlines(changed, samples=96).volume
    np.testing.assert_allclose(result, baseline, rtol=1e-6, atol=1e-8)


def test_preserve_draft_rejects_vertical_action():
    with pytest.raises(ValueError, match="preserve_draft"):
        apply_ffd_actions(
            _state(),
            [
                FFDAction(
                    "bulb",
                    "upward",
                    magnitude_value=0.1,
                    constraints={"preserve_draft": True},
                )
            ],
        )


def test_mapping_action_accepts_exact_magnitude():
    changed = apply_ffd_actions(
        _state(),
        [{"region": "stern", "operation": "forward", "magnitude_value": 0.02}],
    )
    assert changed[1].after["control_points"][1, 0] > 0.0
    assert evaluate_waterline_3d(changed[1]).shape[1] == 3


def test_bow_longitudinal_actions_remain_ordered_at_large_magnitude():
    for operation in ("forward", "aftward"):
        changed = apply_ffd_actions(
            _state(),
            [FFDAction("bow", operation, magnitude_level=5)],
        )
        for waterline in changed:
            path = evaluate_waterline_3d(waterline)
            assert np.all(np.diff(path[:, 0]) >= -1e-10)
