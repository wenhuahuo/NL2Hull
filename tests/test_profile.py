import numpy as np

from nurbs_ship_reconstruction.nurbs import evaluate
from nurbs_ship_reconstruction.profile import (
    PROFILE_CONTROL_COUNT,
    ProfileFit,
    STEM_COINCIDENT,
    STERN_COINCIDENT,
    _fit_contour,
    _groups,
    _combined_feature_knots,
    _range_contour,
    evaluate_profile_parameters,
    profile_parameter_vector,
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


def test_fixed_profile_contour_has_shared_parameter_dimension():
    z = np.linspace(0.0, 0.1, 60)
    points = np.column_stack([0.9 + 0.05 * z, z])
    fit = _fit_contour(points, PROFILE_CONTROL_COUNT, ())
    assert len(fit["control_points"]) == PROFILE_CONTROL_COUNT
    assert len(fit["weights"]) == PROFILE_CONTROL_COUNT
    assert fit["model"] == "fixed degree-3 clamped B-spline; 32 controls"


def test_profile_parameter_vector_round_trips_fixed_controls():
    z = np.linspace(0.0, 0.1, 60)
    points = np.column_stack([0.9 + 0.05 * z, z])
    stem = _fit_contour(points, PROFILE_CONTROL_COUNT, ())
    stern = _fit_contour(points[:, [0, 1]] * [0.1, 1.0], PROFILE_CONTROL_COUNT, ())
    fit = ProfileFit(
        raw_xz=np.empty((0, 2)),
        stem=points,
        stern=points,
        stem_nurbs=stem,
        stern_nurbs=stern,
    )
    vector = profile_parameter_vector(fit)
    stem_curve, stern_curve = evaluate_profile_parameters(vector, samples=11)
    np.testing.assert_allclose(stem_curve[0], stem["control_points"][0])
    np.testing.assert_allclose(stern_curve[-1], stern["control_points"][-1])
    assert len(vector) == 4 * PROFILE_CONTROL_COUNT


def test_combined_feature_fit_preserves_custom_model_label():
    z = np.linspace(0.0, 0.1, 60)
    points = np.column_stack([0.9 + 0.05 * z, z])
    fit = _fit_contour(
        points,
        PROFILE_CONTROL_COUNT,
        (),
        knots=_combined_feature_knots(PROFILE_CONTROL_COUNT, 0.44, 0.60),
        model="combined feature model",
    )
    assert fit["model"] == "combined feature model"


def test_combined_feature_knots_keep_control_dimension_and_features():
    knots = _combined_feature_knots(PROFILE_CONTROL_COUNT, 0.44, 0.60)
    assert len(knots) == PROFILE_CONTROL_COUNT + 4
    assert np.sum(np.isclose(knots, 0.60)) == 3
    assert np.sum((knots >= 0.34) & (knots <= 0.54)) >= 8


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
