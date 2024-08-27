from numpy.typing import NDArray
from typing import Tuple

import numba as nb
import numpy as np
import time

MAX_POINTS_PER_CELL = 50  # Used to infer if we need to re-allocate during read
POINTS_PER_CELL = 9  # Has no restrictive effect, only used to estimate initial memory requirement


def build_cell_face_list(face_owner: NDArray, face_neighbour: NDArray, verbose=1) -> Tuple[NDArray, NDArray]:
    """Assembles the compact cell-face list.

    Parameters
    ----------
    face_owner : NDArray
        Face owner cell index array
    face_neighbour : NDArray
        Face neighbour cell index array
    verbose : int, optional
        Display information, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        2-tuple consisting of cell face index pointer array and the cell face index list arrays

    Notes
    -----
    The ordering of faces within a cell definition is arbitrary.
    """

    if verbose > 0:
        tic = time.perf_counter()
        print("Building cell face list ...", end=" ")

    cell_face_indices, cell_face_list = _build_cell_face_list_jit(face_owner, face_neighbour)

    if verbose > 0:
        toc = time.perf_counter()
        print(f"done in {toc - tic:0.4f} seconds.", flush=True)

    return cell_face_indices, cell_face_list


def build_cell_point_list(
    face_point_indices: NDArray,
    face_point_list: NDArray,
    cell_face_indices: NDArray,
    cell_face_list: NDArray,
    verbose=1,
) -> Tuple[NDArray, NDArray]:
    """Assembles the compact cell-point list.

    Parameters
    ----------
    face_point_indices : NDArray
        Face point index pointer array
    face_point_list : NDArray
        Face point index list array
    cell_face_indices : NDArray
        Cell face index pointer array
    cell_face_list : NDArray
        Cell face index list array
    verbose : int, optional
        Display information, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        2-tuple consisting of cell point index pointer array and the cell face point list arrays

    Note
    ----
    The ordering of points within a cell definition is arbitrary.
    """

    if verbose > 0:
        tic = time.perf_counter()
        print("Building cell point list ...", end=" ")

    cell_point_indices, cell_point_list = _build_cell_point_list_jit(
        face_point_indices, face_point_list, cell_face_indices, cell_face_list
    )

    if verbose > 0:
        toc = time.perf_counter()
        print(f"done in {toc - tic:0.4f} seconds.", flush=True)

    return cell_point_indices, cell_point_list


@nb.jit(nopython=True, fastmath=True, parallel=True, boundscheck=False)
def _build_cell_face_list_jit(face_owner: NDArray, face_neighbour: NDArray) -> Tuple[NDArray, NDArray]:

    # Infer the dtype for labels
    dtype = face_owner.dtype

    # Compute the number of cells and faces
    n_cells = max(np.max(face_owner), np.max(face_neighbour)) + 1
    n_faces = face_owner.size
    n_internal_faces = face_neighbour[face_neighbour > -1].size

    # Allocate output arrays
    cell_face_indices = np.zeros(n_cells + 1, dtype=dtype)

    # Figure out maximum number of faces in a cell for the whole grid
    for k in nb.prange(n_internal_faces):

        cell_face_indices[face_owner[k]] += 1
        cell_face_indices[face_neighbour[k]] += 1

    for k in nb.prange(n_internal_faces, n_faces):

        cell_face_indices[face_owner[k]] += 1

    # Get max. faces per cell and reset the cell face index pointer array
    max_faces_per_cell = np.max(cell_face_indices)
    cell_face_indices[:] = 0

    # Now create a large array where we can hold all cell faces
    cell_faces = np.full((n_cells, max_faces_per_cell), -1, dtype=dtype)

    for k in nb.prange(n_internal_faces):

        n = face_owner[k]
        cell_faces[n, cell_face_indices[n + 1]] = k  # Face count for cell "n" is stored at index "n+1"
        cell_face_indices[n + 1] += 1  # Increment the face count

        n = face_neighbour[k]
        cell_faces[n, cell_face_indices[n + 1]] = k
        cell_face_indices[n + 1] += 1

    for k in nb.prange(n_internal_faces, n_faces):

        n = face_owner[k]
        cell_faces[n, cell_face_indices[n + 1]] = k
        cell_face_indices[n + 1] += 1

    # Flatten the cell face list and reduce to size
    cell_face_list = cell_faces.flatten()
    cell_face_list = cell_face_list[cell_face_list >= 0]

    cell_face_indices = np.cumsum(cell_face_indices)

    return cell_face_indices, cell_face_list


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def _build_cell_point_list_jit(
    face_point_indices: NDArray,
    face_point_list: NDArray,
    cell_face_indices: NDArray,
    cell_face_list: NDArray,
) -> Tuple[NDArray, NDArray]:

    # Infer dtype from input
    dtype = face_point_indices.dtype

    # Initialize counter for unique points referred by cells
    total_count = 0

    # Evaluate number of cells
    n_cells = cell_face_indices.size - 1

    # Allocate index pointer array
    cell_point_indices = np.empty(n_cells + 1, dtype=dtype)
    cell_point_indices[0] = 0

    # Start by assuming "POINTS_PER_CELL" unique points per cell is enough to represent all cells in this grid
    cell_point_list = np.empty(POINTS_PER_CELL * n_cells, dtype=dtype)

    for cell_idx in range(n_cells):

        # Check size and reallocate if necessary
        if total_count + MAX_POINTS_PER_CELL > cell_point_list.size:
            cell_point_list = np.concatenate((cell_point_list, np.empty(n_cells, dtype=dtype)))

        # Get indices of faces of this cell
        cell_faces = cell_face_list[cell_face_indices[cell_idx] : cell_face_indices[cell_idx + 1]]

        # Initialize counter of processed points (including duplicates)
        count = 0

        for k in range(cell_faces.size):

            # Get face index
            face_idx = cell_faces[k]

            # Get point indices for this face
            face_points = face_point_list[face_point_indices[face_idx] : face_point_indices[face_idx + 1]]

            n_face_points = face_points.size

            start = total_count + count
            end = start + n_face_points

            cell_point_list[start:end] = face_points

            count += n_face_points

        # Remove duplicated point indices
        cell_points = np.unique(cell_point_list[total_count:end])
        n_cell_points = cell_points.size

        # Store points of this cell in the cell point list array
        cell_point_list[total_count : total_count + n_cell_points] = cell_points

        # Store the point count for this cell in cell point index pointer array
        cell_point_indices[cell_idx + 1] = n_cell_points

        # Increment the total counter
        total_count += n_cell_points

    cell_point_indices = np.cumsum(cell_point_indices)

    return cell_point_indices, cell_point_list
