import numpy as np

from nurbs_ship_reconstruction.geometry import _fit_half, _integral_parameters
from nurbs_ship_reconstruction.nurbs import evaluate


def test_waterline_area_and_centroid_of_a_rectangle():
    x = np.array([0.0, 0.5])
    width = np.array([0.2, 0.2])
    area, centroid_x, centroid_y = _integral_parameters(x, width)
    np.testing.assert_allclose([area, centroid_x, centroid_y], [0.1, 0.25, 0.1])


def test_half_body_nurbs_recovers_a_simple_fair_curve():
    x = np.linspace(0.9, 0.5, 48)
    y = 0.08 * (1.0 - ((x - 0.5) / 0.4) ** 2)
    y[-1] = y.max()
    fit = _fit_half(x, y)
    recovered = evaluate(fit["parameter"], fit["control_points"], fit["weights"])
    assert fit["fit_rmse"] < 0.02
    np.testing.assert_allclose(fit["control_points"][4:7], np.repeat(fit["control_points"][4:5], 3, axis=0))
    np.testing.assert_allclose(recovered[0], [x[0], y[0]], atol=1e-6)
    np.testing.assert_allclose(recovered[-1], [x[-1], y[-1]], atol=1e-6)


def test_half_body_nurbs_controls_remain_longitudinally_ordered():
    x = np.linspace(0.92, 0.5, 48)
    y = 0.06 * (1.0 - ((x - 0.5) / 0.42) ** 2)
    fit = _fit_half(x, y)
    assert np.all(np.diff(fit["control_points"][:, 0]) <= 1e-12)
