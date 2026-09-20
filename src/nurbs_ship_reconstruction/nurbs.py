"""Small, explicit NURBS implementation used by the reconstruction pipeline."""

from __future__ import annotations

import numpy as np


def clamped_knots(control_count: int, degree: int = 3) -> np.ndarray:
    """Return a clamped uniform knot vector on [0, 1]."""
    if control_count <= degree:
        raise ValueError("control_count must be greater than degree")
    interior_count = control_count - degree - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    return np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]


def basis_matrix(parameters: np.ndarray, control_count: int, degree: int = 3) -> np.ndarray:
    """Evaluate all B-spline basis functions at *parameters*."""
    u = np.asarray(parameters, dtype=float).ravel()
    knots = clamped_knots(control_count, degree)
    basis = np.zeros((u.size, control_count), dtype=float)

    for i in range(control_count):
        right = (knots[i] <= u) & (u < knots[i + 1])
        basis[right, i] = 1.0
    basis[u == 1.0, -1] = 1.0

    for order in range(1, degree + 1):
        current = np.zeros_like(basis)
        for i in range(control_count):
            left_denominator = knots[i + order] - knots[i]
            right_denominator = knots[i + order + 1] - knots[i + 1]
            if left_denominator:
                current[:, i] += (u - knots[i]) / left_denominator * basis[:, i]
            if right_denominator and i + 1 < control_count:
                current[:, i] += (
                    knots[i + order + 1] - u
                ) / right_denominator * basis[:, i + 1]
        basis = current
    # Cox-de Boor's half-open base interval excludes u=1; the clamped
    # endpoint belongs to the final basis function.
    endpoint = u == 1.0
    if np.any(endpoint):
        basis[endpoint] = 0.0
        basis[endpoint, -1] = 1.0
    return basis


def evaluate(
    parameters: np.ndarray,
    control_points: np.ndarray,
    weights: np.ndarray | None = None,
    degree: int = 3,
) -> np.ndarray:
    """Evaluate a 2-D or 3-D NURBS curve."""
    points = np.asarray(control_points, dtype=float)
    if points.ndim != 2:
        raise ValueError("control_points must be a 2-D array")
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(points) or np.any(weights <= 0):
        raise ValueError("weights must be positive and match control_points")

    basis = basis_matrix(parameters, len(points), degree)
    weighted_basis = basis * weights[None, :]
    denominator = weighted_basis.sum(axis=1)
    return weighted_basis @ points / denominator[:, None]
