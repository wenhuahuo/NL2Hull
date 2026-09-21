import numpy as np

from nurbs_ship_reconstruction.nurbs import evaluate
from nurbs_ship_reconstruction.profile import (
    STEM_COINCIDENT,
    STERN_COINCIDENT,
    _fit_contour,
    _groups,
    _range_contour,
)


def test_stem_coincident_groups_collapse_duplicates():
    parent = _groups(14, STEM_COINCIDENT)
    assert parent[5] == 4
    assert parent[10] == 9
    assert parent[0] == 0
    assert parent[13] == 13


def test_vertical_stem_nurbs_recovers_endpoints_and_doubles():
    z = np.linspace(0.0, 0.12, 40)
    points = np.column_stack([np.ones_like(z), z])
    fit = _fit_contour(points, 14, STEM_COINCIDENT)
    recovered = evaluate(np.linspace(0.0, 1.0, 20), fit["control_points"])
    np.testing.assert_allclose(recovered[0], points[0], atol=1e-5)
    np.testing.assert_allclose(recovered[-1], points[-1], atol=1e-5)
    np.testing.assert_allclose(fit["control_points"][4], fit["control_points"][5])
    np.testing.assert_allclose(fit["control_points"][9], fit["control_points"][10])
    assert fit["fit_rmse"] < 0.01


def test_range_contour_stitches_fore_and_aft_segments():
    lower = np.column_stack([np.linspace(0.65, 1.0, 8), np.zeros(8)])
    upper = np.column_stack([np.linspace(1.0, 0.65, 8), np.ones(8)])
    contour = _range_contour([lower, upper], 0.65, None)
    assert contour[0, 1] == 0.0
    assert contour[-1, 1] == 1.0
    assert contour[:, 0].min() >= 0.65
    assert len(contour) == len(lower) + len(upper)


def test_stern_triples_are_coincident_after_fit():
    z = np.linspace(0.0, 0.1, 40)
    x = 0.02 + 0.01 * (z / 0.1) ** 2
    points = np.column_stack([x, z])
    fit = _fit_contour(points, 19, STERN_COINCIDENT)
    controls = fit["control_points"]
    np.testing.assert_allclose(controls[3], controls[4])
    np.testing.assert_allclose(controls[4], controls[5])
    np.testing.assert_allclose(controls[15], controls[17])
    assert fit["fit_rmse"] < 0.01
