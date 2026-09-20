import numpy as np

from nurbs_ship_reconstruction.nurbs import basis_matrix, evaluate


def test_clamped_basis_partitions_unity_including_endpoints():
    parameters = np.linspace(0, 1, 101)
    basis = basis_matrix(parameters, control_count=8, degree=3)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0)
    np.testing.assert_allclose(basis[0], [1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(basis[-1], [0, 0, 0, 0, 0, 0, 0, 1])


def test_nurbs_interpolates_clamped_endpoints():
    controls = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    values = evaluate(np.array([0.0, 1.0]), controls)
    np.testing.assert_allclose(values, [controls[0], controls[-1]])
