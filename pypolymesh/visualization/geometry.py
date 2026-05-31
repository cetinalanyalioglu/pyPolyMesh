from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.typing import NDArray

import numpy as np


@dataclass(frozen=True)
class BestFitPlane:
    origin: NDArray
    normal: NDArray
    basis_u: NDArray
    basis_v: NDArray


def _as_point_array(points: ArrayLike) -> NDArray:

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("Expected points with shape (n_points, 3).")

    if point_array.shape[0] < 3:
        raise ValueError("At least three points are required.")

    return point_array


def _orient_normal(normal: NDArray) -> NDArray:

    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] < 0.0:
        return -normal
    return normal


def fit_best_plane(points: ArrayLike, collinearity_rtol: float = 1.0e-12) -> BestFitPlane:
    """Fit a best-fit plane to a 3D polygon point set."""

    point_array = _as_point_array(points)
    origin = np.mean(point_array, axis=0)
    centered = point_array - origin

    _, singular_values, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    scale = singular_values[0] if singular_values.size > 0 else 0.0
    if singular_values.size < 2 or singular_values[1] <= collinearity_rtol * max(scale, 1.0):
        raise ValueError("Points are collinear or nearly collinear; cannot define a stable plane.")

    basis_u = right_singular_vectors[0, :]
    basis_u = basis_u / np.linalg.norm(basis_u)

    normal = _orient_normal(right_singular_vectors[-1, :] / np.linalg.norm(right_singular_vectors[-1, :]))
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)

    return BestFitPlane(origin=origin, normal=normal, basis_u=basis_u, basis_v=basis_v)


def project_points_to_plane(points: ArrayLike, plane: BestFitPlane) -> NDArray:
    """Project 3D points to local 2D coordinates on the given plane."""

    point_array = _as_point_array(points)
    centered = point_array - plane.origin
    return np.column_stack((centered @ plane.basis_u, centered @ plane.basis_v))

