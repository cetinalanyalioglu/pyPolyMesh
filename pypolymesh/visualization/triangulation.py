from dataclasses import dataclass
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import Optional

import numpy as np

from .geometry import BestFitPlane
from .geometry import _as_point_array
from .geometry import fit_best_plane
from .geometry import project_points_to_plane

_DEFAULT_TRIANGULATION_STRATEGY = "best_fit_projection"
_VALID_TRIANGULATION_STRATEGIES = {"fan", "best_fit_projection"}


@dataclass(frozen=True)
class TriangulatedFace:
    points: NDArray
    projected_points: Optional[NDArray]
    triangles: NDArray
    strategy: str
    plane: Optional[BestFitPlane]


def _normalize_strategy(strategy: Optional[str]) -> str:

    selected = _DEFAULT_TRIANGULATION_STRATEGY if strategy is None else strategy
    if selected not in _VALID_TRIANGULATION_STRATEGIES:
        raise ValueError(
            f'"strategy" must be one of {sorted(_VALID_TRIANGULATION_STRATEGIES)}, received "{selected}".'
        )
    return selected


def fan_triangulation_indices(n_points: int) -> NDArray:
    """Triangulate an ordered polygon with a simple vertex fan."""

    if n_points < 3:
        raise ValueError("At least three points are required to triangulate a polygon.")

    return np.array([(0, idx, idx + 1) for idx in range(1, n_points - 1)], dtype=np.int64)


def _polygon_signed_area(points_2d: NDArray) -> float:

    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _is_convex_vertex(a: NDArray, b: NDArray, c: NDArray, eps: float) -> bool:

    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > eps


def _point_in_triangle(point: NDArray, a: NDArray, b: NDArray, c: NDArray, eps: float) -> bool:

    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if np.abs(denominator) <= eps:
        return False

    alpha = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denominator
    beta = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denominator
    gamma = 1.0 - alpha - beta

    return (alpha >= -eps) and (beta >= -eps) and (gamma >= -eps)


def earclip_triangulation_2d(points_2d: ArrayLike, eps: float = 1.0e-14) -> NDArray:
    """Triangulate a simple 2D polygon using ear clipping."""

    polygon = np.asarray(points_2d, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Expected 2D polygon coordinates with shape (n_points, 2).")

    n_points = polygon.shape[0]
    if n_points < 3:
        raise ValueError("At least three points are required to triangulate a polygon.")
    if n_points == 3:
        return np.array([[0, 1, 2]], dtype=np.int64)

    working = list(range(n_points))
    if _polygon_signed_area(polygon) < 0.0:
        working.reverse()

    triangles = []
    guard_limit = n_points * n_points
    guard = 0

    while len(working) > 3:
        ear_found = False
        m = len(working)
        for index in range(m):
            prev_idx = working[(index - 1) % m]
            curr_idx = working[index]
            next_idx = working[(index + 1) % m]

            a = polygon[prev_idx, :]
            b = polygon[curr_idx, :]
            c = polygon[next_idx, :]

            if not _is_convex_vertex(a, b, c, eps):
                continue

            if any(
                _point_in_triangle(polygon[other_idx, :], a, b, c, eps)
                for other_idx in working
                if other_idx not in (prev_idx, curr_idx, next_idx)
            ):
                continue

            triangles.append((prev_idx, curr_idx, next_idx))
            del working[index]
            ear_found = True
            break

        guard += 1
        if ear_found:
            continue
        if guard > guard_limit:
            raise ValueError("Failed to triangulate polygon. It may be self-intersecting or degenerate.")
        raise ValueError("Failed to triangulate polygon. It may be self-intersecting or degenerate.")

    triangles.append(tuple(working))
    return np.asarray(triangles, dtype=np.int64)


def triangulate_face(points: ArrayLike, strategy: Optional[str] = None) -> TriangulatedFace:
    """Triangulate an ordered 3D polygon face."""

    point_array = _as_point_array(points)
    selected_strategy = _normalize_strategy(strategy)

    if selected_strategy == "fan":
        triangles = fan_triangulation_indices(point_array.shape[0])
        return TriangulatedFace(
            points=point_array,
            projected_points=None,
            triangles=triangles,
            strategy=selected_strategy,
            plane=None,
        )

    plane = fit_best_plane(point_array)
    projected_points = project_points_to_plane(point_array, plane)
    triangles = earclip_triangulation_2d(projected_points)

    return TriangulatedFace(
        points=point_array,
        projected_points=projected_points,
        triangles=triangles,
        strategy=selected_strategy,
        plane=plane,
    )

