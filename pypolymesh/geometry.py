from numpy.typing import NDArray
from typing import Tuple

import numba as nb
import numpy as np
import time


def compute_face_areas_and_centroids(
    points: NDArray, face_point_indices: NDArray, face_point_list: NDArray, verbose=1
) -> Tuple[NDArray, NDArray]:
    """Computes the area and centroid of each face in a mesh.

    Parameters
    ----------
    points : NDArray
        Array of points in 3D space, shape (n_points, 3).
    face_point_indices : NDArray
        Face point index pointer array, shape (n_faces + 1).
    face_point_list : NDArray
        Face point list array, shape (n_face_points).
    verbose : int, optional
        Level of verbosity, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        Array of face centroids and face area vectors, both of shape (n_faces, 3). The dtype is taken from points.
    """

    if verbose > 0:
        tic = time.perf_counter()
        print("Computing face areas and centroids ...", end=" ")

    face_centroid, face_area_vector = _compute_face_areas_and_centroids_jit(points, face_point_indices, face_point_list)

    if verbose > 0:
        toc = time.perf_counter()
        print(f"done in {toc - tic:.4f} seconds.", flush=True)

    if verbose > 1:
        print(f"Minimum face area: {np.min(np.linalg.norm(face_area_vector, axis=1)):.15e}")
        print(f"Maximum face area: {np.max(np.linalg.norm(face_area_vector, axis=1)):.15e}")

    return face_centroid, face_area_vector


@nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
def triangle_centroid(points: NDArray) -> NDArray:

    return np.sum(points, axis=0) / 3.0


@nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
def triangle_area_vector(points: NDArray) -> NDArray:

    return np.cross(points[1, :] - points[0, :], points[2, :] - points[1, :]) / 2.0


@nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
def polygon_area_and_centroid_openfoam(points: NDArray) -> Tuple[NDArray, NDArray]:
    """Compute the area and centroid of a polygon in 3D space (as in OpenFOAM)."""

    n_points = points.shape[0]

    dtype = points.dtype

    # Compute the bounding box center (e.g. geometric center)
    bounding_box_center = np.sum(points, axis=0) / n_points

    sum_tri_area_mag = 0.0
    sum_tri_area_mom = np.zeros(3, dtype=dtype)
    sum_tri_area_vec = np.zeros(3, dtype=dtype)

    for i in range(n_points):

        this_point = points[i, :]
        next_point = points[(i + 1) % n_points, :]

        tri_centroid = this_point + next_point + bounding_box_center
        tri_area_vec = np.cross(next_point - this_point, bounding_box_center - this_point)
        tri_area_mag = np.linalg.norm(tri_area_vec)

        sum_tri_area_mag += tri_area_mag
        sum_tri_area_vec += tri_area_vec
        sum_tri_area_mom += tri_centroid * tri_area_mag

    # Avoid division by zero in case of degenerate face
    if sum_tri_area_mag < np.finfo(dtype).tiny:
        centroid = bounding_box_center
        area_vector = np.zeros(3, dtype=dtype)
    else:
        centroid = sum_tri_area_mom / sum_tri_area_mag / 3.0
        area_vector = sum_tri_area_vec / 2.0

    return centroid, area_vector


@nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
def polygon_area_and_centroid_precise(points: NDArray) -> Tuple[NDArray, NDArray]:
    """Compute the area and centroid of a polygon in 3D space (as in PRECISE)."""

    # CA: I have no idea why this would be preferred over the OpenFOAM method, difference between these methods
    # only arise when the face is not planar. In this case result depends on which vertex you choose as the anchor.
    # Using the anchor point, one then successively builds triangles with the next two points to perform a cumulative
    # computation of centroid and area vector. Here they choose arbitrarily the "first" vertex, for god knows why,
    # OpenFOAM uses the geometric center of the polygon.

    n_points = points.shape[0]

    dtype = points.dtype

    sum_tri_area_mag = 0.0
    sum_tri_area_mom = np.zeros(3, dtype=dtype)
    sum_tri_area_vec = np.zeros(3, dtype=dtype)

    # Use the first point as anchor to build triangles
    anchor = points[0, :]

    for i in range(1, n_points - 1):

        this_point = points[i, :]
        next_point = points[i + 1, :]

        tri_centroid = this_point + next_point + anchor
        tri_area_vec = np.cross(this_point - anchor, next_point - this_point)
        tri_area_mag = np.linalg.norm(tri_area_vec)

        sum_tri_area_mag += tri_area_mag
        sum_tri_area_vec += tri_area_vec
        sum_tri_area_mom += tri_centroid * tri_area_mag

    # Avoid division by zero in case of degenerate face
    if sum_tri_area_mag < np.finfo(dtype).tiny:
        centroid = np.sum(points, axis=0) / n_points
        area_vector = np.zeros(3, dtype=dtype)
    else:
        centroid = sum_tri_area_mom / sum_tri_area_mag / 3.0
        area_vector = sum_tri_area_vec / 2.0

    return centroid, area_vector


# CA: With nb.prange, this is about 3 times faster than FORTRAN in my workstation (Ryzen 7950X3D)
@nb.jit(nopython=True, fastmath=True, parallel=True, boundscheck=False)
def _compute_face_areas_and_centroids_jit(
    points: NDArray, face_point_indices: NDArray, face_point_list: NDArray
) -> Tuple[NDArray, NDArray]:

    n_faces = face_point_indices.size - 1

    face_centroid = np.empty((n_faces, 3), dtype=points.dtype)
    face_area_vector = np.empty((n_faces, 3), dtype=points.dtype)

    for k in nb.prange(n_faces):

        face_points = points[face_point_list[face_point_indices[k] : face_point_indices[k + 1]], :]

        if face_points.size == 3:
            face_centroid[k, :] = triangle_centroid(face_points)
            face_area_vector[k, :] = triangle_area_vector(face_points)
        else:
            face_centroid[k, :], face_area_vector[k, :] = polygon_area_and_centroid_openfoam(face_points)

    return face_centroid, face_area_vector
