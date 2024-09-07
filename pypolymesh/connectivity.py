from .elements import HEXAHEDRON
from .elements import PENTAHEDRON
from .elements import POLYHEDRON
from .elements import PYRAMID
from .elements import TETRAHEDRON
from numpy.typing import NDArray
from typing import Tuple

import numba as nb
import numpy as np
import time

MAX_POINTS_PER_CELL = 50  # Used to infer if we need to re-allocate during read
POINTS_PER_CELL = 9  # Has no restrictive effect, only used to estimate initial memory requirement


def build_cell_face_list(face_owner: NDArray, face_neighbour: NDArray, verbose=1) -> Tuple[NDArray, NDArray]:
    """Computes the compact cell-face list.

    The compact cell-face list contains all information regarding the faces of each cell in the grid.
    It is formed by 2 arrays: the cell face index pointer array and the cell face index list array.
    The cell face index list array is a one dimensional array which contains the face indices of each cell,
    ordered from cell index 0 to the highest cell index. The cell face index pointer array is also a one
    dimensional array, but it contains the starting index in the cell face list array for each cell.

    >>> # Consider a grid formed by 2 tetrahedrons,
    >>> face_owner = np.array([0, 0, 0, 0, 1, 1, 1])
    >>> face_neighbour = np.array([1, -1, -1, -1, -1, -1, -1])
    >>> # Compute the cell-face list
    >>> cell_face_indices, cell_face_list = build_cell_face_list(face_owner, face_neighbour, verbose=0)
    >>> cell_face_indices
    (array([0, 4, 8])
    >>> cell_face_list
    array([0, 1, 2, 3, 0, 4, 5, 6])
    >>> # Retrieve the indices of cell 0
    >>> cell_face_list[cell_face_indices[0]:cell_face_indices[1]]
    array([0, 1, 2, 3])
    >>> # Retrieve the indices of cell 1
    >>> cell_face_list[cell_face_indices[1]:cell_face_indices[2]]
    array([0, 4, 5, 6])

    Notes
    -----
    The ordering of faces within a cell definition is arbitrary.

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
        2-tuple consisting of cell face index pointer array and the cell face index list arrays. The dtype of returned
        arrays is inferred from the input arrays.
    """

    if face_owner.dtype != face_neighbour.dtype:
        raise ValueError("Face owner and neighbour arrays must have the same dtype.")

    if verbose > 0:
        tic = time.perf_counter()
        print("Building cell face list ...", end=" ")

    cell_face_indices, cell_face_list = __build_cell_face_list_jit(face_owner, face_neighbour)

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
    """Assembles the cell-point list.

    The cell-point list contains the (unique) points referred by each cell in the grid. It is formed by 2 arrays: the
    cell point index pointer array and the cell point list array. The cell point list array is a one dimensional array
    which contains the point indices of each cell, ordered from cell index 0 to the highest cell index. The cell point
    index pointer array is also a one dimensional array, but it contains the starting index in the cell point list array
    for each cell.

    Note that this routine does not care about the ordering of points within a cell, therefore for standard elements
    (e.g. tetrahedron, hexahedron) the ordering of points won't match the VTK or other standard conventions.

    Parameters
    ----------
    face_point_indices : NDArray
        Face point index pointer array, shape (n_faces + 1)
    face_point_list : NDArray
        Face point index list array, shape (n_face_points)
    cell_face_indices : NDArray
        Cell face index pointer array, shape (n_cells + 1)
    cell_face_list : NDArray
        Cell face index list array, shape (n_cell_faces)
    verbose : int, optional
        Display information, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray]
        2-tuple consisting of cell point index pointer array and the cell point list arrays. The dtype of returned
        arrays is inferred from the input arrays.
    """

    if not all(arr.dtype == face_point_indices.dtype for arr in [face_point_list, cell_face_indices, cell_face_list]):
        raise ValueError("All input arrays must have the same dtype.")

    if verbose > 0:
        tic = time.perf_counter()
        print("Building cell point list ...", end=" ")

    cell_point_indices, cell_point_list = __build_cell_point_list_jit(
        face_point_indices, face_point_list, cell_face_indices, cell_face_list
    )

    if verbose > 0:
        toc = time.perf_counter()
        print(f"done in {toc - tic:0.4f} seconds.", flush=True)

    return cell_point_indices, cell_point_list


def build_ordered_cell_point_list(
    points: NDArray,
    face_point_indices: NDArray,
    face_point_list: NDArray,
    face_centroids: NDArray,
    face_area_vectors: NDArray,
    cell_face_indices: NDArray,
    cell_face_list: NDArray,
    verbose=1,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Assembles the ordered cell-point list.

    The output is similar to ```build_cell_point_list``` but the points are ordered in a specific way for each
    non-polyhedral cell type following the VTK convention. With this type of definition, each non-polyhedral cell
    is completely defined by its ordered set of points, e.g. face information is no longer necessary. For polyhedral
    cells, the point list contains the unique points referred by the cell arranged in an arbitrary order.

    The VTK polyhedron definition is
    ``` [n_items, n_faces, n_face_0_points, face_0_point_1, face_0_point_2, ..., n_face_1_points, ...]```
     where ```n_items``` refers to the length of above list (not including ```n_items```), and faces are expected
     to have a counter-clowise orientation when viewed from outside the cell. This routine is intended to contain
     only point indices in the output, and **does not** return the VTK definition for polyhedrons. Instead, each
     polyhedral is represented as ```[point_1, point_2, point_3, ...]```, where the order of points is arbitrary.
     These points are unique, e.g. the duplications due to sharing among faces are removed.

    To use this routine,
    1. The cell face list must be computed first using ```build_cell_face_list```.
    2. The face centroids and area vectors should be computed using ```compute_face_centroids_and_area_vectors```.

    Parameters
    ----------
    points : NDArray
        Point coordinates array, shape (n_points, 3)
    face_point_indices : NDArray
        Face point index pointer array, shape (n_faces + 1)
    face_point_list : NDArray
        Face point index list array, shape (n_face_points)
    face_centroids : NDArray
        Face centroids array, shape (n_faces, 3)
    face_area_vectors : NDArray
        Face area vectors array, shape (n_faces, 3)
    cell_face_indices : NDArray
        Cell face index pointer array, shape (n_cells + 1)
    cell_face_list : NDArray
        Cell face index list array, shape (n_cell_faces)
    verbose : int, optional
        Display information, by default 1

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        3-tuple consisting of cell types, cell point index pointer array and the cell face point list arrays
    """

    if not all(arr.dtype == points.dtype for arr in [face_centroids, face_area_vectors]):
        raise ValueError("Input arrays have non-uniform dtypes for floating point values.")

    if not all(arr.dtype == face_point_indices.dtype for arr in [face_point_list, cell_face_indices, cell_face_list]):
        raise ValueError("Input arrays have non-uniform dtypes for integers.")

    if verbose > 0:
        tic = time.perf_counter()
        print("Building ordered cell point list ...", end=" ")

    cell_types, cell_point_indices, cell_point_list = __build_ordered_cell_point_list_jit(
        points,
        face_point_indices,
        face_point_list,
        face_centroids,
        face_area_vectors,
        cell_face_indices,
        cell_face_list,
    )

    if verbose > 0:
        toc = time.perf_counter()
        print(f"done in {toc - tic:0.4f} seconds.", flush=True)

    return cell_types, cell_point_indices, cell_point_list


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def __build_cell_face_list_jit(face_owner: NDArray, face_neighbour: NDArray) -> Tuple[NDArray, NDArray]:

    # Infer the dtype for labels
    dtype = face_owner.dtype

    # Compute the number of cells and faces
    n_cells = max(np.max(face_owner), np.max(face_neighbour)) + 1
    n_faces = face_owner.size
    n_internal_faces = face_neighbour[face_neighbour > -1].size

    # Allocate output arrays
    cell_face_indices = np.zeros(n_cells + 1, dtype=dtype)

    # Figure out maximum number of faces in a cell for the whole grid
    for k in range(n_internal_faces):

        cell_face_indices[face_owner[k]] += 1
        cell_face_indices[face_neighbour[k]] += 1

    for k in range(n_internal_faces, n_faces):

        cell_face_indices[face_owner[k]] += 1

    # Get max. faces per cell and reset the cell face index pointer array
    max_faces_per_cell = np.max(cell_face_indices)
    cell_face_indices[:] = 0

    # Now create a large array where we can hold all cell faces
    cell_faces = np.full((n_cells, max_faces_per_cell), -1, dtype=dtype)

    for k in range(n_internal_faces):

        n = face_owner[k]
        cell_faces[n, cell_face_indices[n + 1]] = k  # Face count for cell "n" is stored at index "n+1"
        cell_face_indices[n + 1] += 1  # Increment the face count

        n = face_neighbour[k]
        cell_faces[n, cell_face_indices[n + 1]] = k
        cell_face_indices[n + 1] += 1

    for k in range(n_internal_faces, n_faces):

        n = face_owner[k]
        cell_faces[n, cell_face_indices[n + 1]] = k
        cell_face_indices[n + 1] += 1

    # Flatten the cell face list and reduce to size
    cell_face_list = cell_faces.flatten()
    cell_face_list = cell_face_list[cell_face_list >= 0]

    cell_face_indices = np.cumsum(cell_face_indices)

    return cell_face_indices, cell_face_list


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def __build_cell_point_list_jit(
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

    # Remove the unused part of the cell point list
    cell_point_list = cell_point_list[:total_count]

    cell_point_indices = np.cumsum(cell_point_indices)

    return cell_point_indices, cell_point_list


# Around 5 seconds for 18 million cells including jit compilation
@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def __build_ordered_cell_point_list_jit(
    points: NDArray,
    face_point_indices: NDArray,
    face_point_list: NDArray,
    face_centroids: NDArray,
    face_area_vectors: NDArray,
    cell_face_indices: NDArray,
    cell_face_list: NDArray,
) -> Tuple[NDArray, NDArray]:

    # Infer dtype from input (ensure the input arrays have uniform dtypes before calling)
    dtype = face_point_indices.dtype

    # Initialize counter for unique points referred by cells
    total_count = 0

    # Evaluate number of cells
    n_cells = cell_face_indices.size - 1

    # Allocate index pointer array
    cell_point_indices = np.empty(n_cells + 1, dtype=dtype)
    cell_point_indices[0] = 0

    # Allocate cell type array
    cell_types = np.empty(n_cells, dtype=dtype)

    # Start by assuming "POINTS_PER_CELL" unique points per cell is enough to represent all cells in this grid
    cell_point_list = np.full(POINTS_PER_CELL * n_cells, -1, dtype=dtype)

    # Loop over all cells
    for cell_idx in range(n_cells):

        # Check size and reallocate if necessary
        if total_count + MAX_POINTS_PER_CELL > cell_point_list.size:
            cell_point_list = np.concatenate((cell_point_list, np.full(n_cells, -1, dtype=dtype)))

        # Get indices of faces of this cell
        cell_faces = cell_face_list[cell_face_indices[cell_idx] : cell_face_indices[cell_idx + 1]]

        n_quads = 0  # Number of quadrilaterals
        n_tris = 0  # Number of triangles
        n_polys = 0  # Number of polygons

        idx_quad = -1  # Index of an arbitrary quadrilateral face in this cell
        idx_tris = np.full(2, 1, dtype=dtype)  # Indices of two triangles in this cell

        # First count the face types for this cell
        for k in range(cell_faces.size):

            # Get face index
            face_idx = cell_faces[k]

            # Get point indices for this face
            face_points = face_point_list[face_point_indices[face_idx] : face_point_indices[face_idx + 1]]

            # Determine the type of face and accumulate the counts of each face type
            if face_points.size == 3:
                # We need to save index of only 2 triangle faces
                idx_tris[min(n_tris, 1)] = face_idx
                n_tris += 1
            elif face_points.size == 4:
                # We need to save index of only one quadrilateral face
                idx_quad = face_idx
                n_quads += 1
            else:
                n_polys += 1

        # Tetrahedron
        if n_quads == 0 and n_tris == 4:
            cell_type = TETRAHEDRON
            # Pick any face as the base
            tri_1 = face_point_list[face_point_indices[idx_tris[0]] : face_point_indices[idx_tris[0] + 1]]
            # Find the tip point, which is not in the base
            tri_2 = face_point_list[face_point_indices[idx_tris[1]] : face_point_indices[idx_tris[1] + 1]]
            for point in tri_2:
                if point not in tri_1:
                    tip = point
                    break
            # The correct orientation is such that normal vector of the base points towards interior
            v1 = points[tip] - face_centroids[idx_tris[0], :]  # Vector from base centroid to tip point
            v2 = face_area_vectors[idx_tris[0], :]
            if np.dot(v1, v2) < 0:
                tri_1 = tri_1[::-1]
            # Append to the cell point list
            cell_point_list[total_count : total_count + 3] = tri_1
            cell_point_list[total_count + 3] = tip
            # Fill the index pointer array
            cell_point_indices[cell_idx + 1] = 4
            # Increment the total counter
            total_count += 4

        # Pyramid
        elif n_quads == 1 and n_tris == 4 and n_polys == 0:
            cell_type = PYRAMID
            # The quadrilateral face is the base of the pyramid
            quad = face_point_list[face_point_indices[idx_quad] : face_point_indices[idx_quad + 1]]
            # Find the tip point, which is not in the base
            for point in face_point_list[face_point_indices[idx_tris[0]] : face_point_indices[idx_tris[0] + 1]]:
                if point not in quad:
                    tip = point
                    break
            # The correct orientation is such that normal vector of the base points towards interior
            v1 = points[tip] - face_centroids[idx_quad, :]  # Vector from base centroid to tip point
            v2 = face_area_vectors[idx_quad, :]
            if np.dot(v1, v2) < 0:
                quad = quad[::-1]
            # Append to the cell point list
            cell_point_list[total_count : total_count + 4] = quad
            cell_point_list[total_count + 4] = tip
            # Fill the index pointer array
            cell_point_indices[cell_idx + 1] = 5
            # Increment the total counter
            total_count += 5

        # Pentahedron, also referred as wedge or triangular prism
        elif n_quads == 3 and n_tris == 2 and n_polys == 0:
            cell_type = PENTAHEDRON
            # Pick one of the triangular faces as the base
            tri_1 = face_point_list[face_point_indices[idx_tris[0]] : face_point_indices[idx_tris[0] + 1]]
            # Pick the other triangular face as the top
            tri_2 = face_point_list[face_point_indices[idx_tris[1]] : face_point_indices[idx_tris[1] + 1]]
            # The normal vector of the first triangle should point towards the interior of the cell
            v1 = face_centroids[idx_tris[1], :] - face_centroids[idx_tris[0], :]  # Base to top
            v2 = face_area_vectors[idx_tris[0], :]
            if np.dot(v1, v2) < 0:
                tri_1 = tri_1[::-1]
            # The normal vectors of both triangles should point to the same direction
            if np.dot(v2, face_area_vectors[idx_tris[1], :]) < 0:
                # Reverse the order of the second triangle if the normal vectors point to opposite directions
                tri_2 = tri_2[::-1]
            # Append to the cell point list
            cell_point_list[total_count : total_count + 3] = tri_1
            cell_point_list[total_count + 3 : total_count + 6] = tri_2
            # Fill the index pointer array
            cell_point_indices[cell_idx + 1] = 6
            # Increment the total counter
            total_count += 6

        # Hexahedron
        elif n_quads == 6 and n_polys == 0:
            cell_type = HEXAHEDRON
            # Pick any face as base
            quad_1 = face_point_list[face_point_indices[cell_faces[0]] : face_point_indices[cell_faces[0] + 1]]
            # Find the top face (it should have no common points with the base)
            for face_idx in cell_faces:
                quad_2 = face_point_list[face_point_indices[face_idx] : face_point_indices[face_idx + 1]]
                idx_quad_2 = face_idx
                found_common_point = False
                for point in quad_2:
                    if point in quad_1:
                        found_common_point = True
                        break
                if not found_common_point:
                    break
            # The normal vector of the base should point towards the interior of the cell
            v1 = face_centroids[idx_quad_2, :] - face_centroids[cell_faces[0], :]
            v2 = face_area_vectors[cell_faces[0], :]
            if np.dot(v1, v2) < 0:
                quad_1 = quad_1[::-1]
            # The normal vectors of both the base andd top should point to the same direction
            if np.dot(face_area_vectors[cell_faces[0], :], face_area_vectors[idx_quad_2, :]) < 0:
                # Reverse the order of the top if the normal vectors point to opposite directions
                quad_2 = quad_2[::-1]
            # Append to the cell point list
            cell_point_list[total_count : total_count + 4] = quad_1
            cell_point_list[total_count + 4 : total_count + 8] = quad_2
            # Fill the index pointer array
            cell_point_indices[cell_idx + 1] = 8
            # Increment the total counter
            total_count += 8

        # For all other cases, we categorize the cell as a polyhedron
        else:
            cell_type = POLYHEDRON
            # In contrast to other element types, the point order does not matter here. We only need to store the
            # indices of unique points forming this cell. Note that this is not same as the VTK polyhedron definition.
            count = 0
            for face_idx in cell_faces:
                face_points = face_point_list[face_point_indices[face_idx] : face_point_indices[face_idx + 1]]
                for point in face_points:
                    if point not in cell_point_list[total_count : total_count + count]:
                        cell_point_list[total_count + count] = point
                        count += 1
            # Fill the index pointer array
            cell_point_indices[cell_idx + 1] = count
            # Increment the total counter
            total_count += count

        # Assign the cell type for this cell
        cell_types[cell_idx] = cell_type

    # Remove the unused portion of the cell point list array
    cell_point_list = cell_point_list[:total_count]

    # Build the cell point index pointer array from cumulative sum of individual cell point counts
    cell_point_indices = np.cumsum(cell_point_indices)

    return cell_types, cell_point_indices, cell_point_list
