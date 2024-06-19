from .utils import read_faces
from .utils import read_scalar_field
from .utils import read_vector_field
from .utils import recursive_dictionary_parser
from .utils import recursive_dictionary_writer
from .utils import write_faces
from .utils import write_scalar_field
from .utils import write_vector_field
from collections import OrderedDict
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from numpy.typing import NDArray
from pathlib import Path
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import collections
import numba as nb
import numpy as np
import os
import warnings


class PolyMesh:

    def __init__(self, path: str, verbose=1, **kwargs) -> None:

        self.__verbose = verbose
        self.__byteorder = "little"
        self.__dtype_int = np.int32
        self.__dtype_float = np.float64
        self.__points: NDArray
        self.__face_point_indices: NDArray
        self.__face_point_list: NDArray
        self.__face_owner: NDArray
        self.__face_neighbour: NDArray
        self.__boundary: "OrderedDict"
        self.__deleted_faces: List[int]
        self.__deleted_cells: List[int]
        self.__deleted_points: List[int]
        self.__face_offset_log: NDArray
        self.__cell_offset_log: NDArray
        self.__point_offset_log = NDArray

        # Read the mesh
        self._read(path=path)

        # Initialize offset log to keep track of modifications
        self.reset_offsets()

    @property
    def boundary(self) -> "OrderedDict[str, OrderedDict[str, Any]]":
        """Returns boundary information.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Boundary dictionary, keys are boundary names and values are sub-dictionaries
        """

        return self.__boundary

    @property
    def dtype_float(self) -> DTypeLike:

        return self.__dtype_float

    @dtype_float.setter
    def dtype_float(self, value: DTypeLike) -> None:

        if np.dtype(value).itemsize < self.dtype_float.itemsize:
            warnings.warn(
                f"Switched to a dtype of lower precision ({str(self.dtype_float)} to {str(value)})"
            )

        self.__dtype_float = np.dtype(value)

    @property
    def dtype_int(self) -> DTypeLike:

        return self.__dtype_int

    @dtype_int.setter
    def dtype_int(self, value: DTypeLike) -> None:

        if np.dtype(value).itemsize < self.dtype_int.itemsize:
            warnings.warn(
                f"Switched to a dtype of lower precision ({str(self.dtype_int)} to {str(value)})"
            )

        self.__dtype_int = np.dtype(value)

    @property
    def face_owner(self) -> NDArray:
        """Returns the face owner list, e.g. "constant/owner".

        Returns
        -------
        NDArray
            Face owner array
        """

        return self.__face_owner

    @property
    def face_neighbour(self) -> NDArray:
        """Returns the face neighbour list, e.g. "constant/neighbour".

        Returns
        -------
        NDArray
            Face neighbour array
        """

        return self.__face_neighbour

    @property
    def face_point_indices(self) -> NDArray:
        """Returns the face point index list.

        Returns
        -------
        NDArray
            Face point index array

        Notes
        -----
        The face point index array contains indices pointing to the start and end indices in the
        face point list array. It is formed by performing a cumulative summation of the count of
        ponts over all faces, yield a size of N+1 for N faces.

        Consider a domain formed by 3 faces, having [4, 3, 5] points respectively. For this case,
        the face index array would be as follows:

        >>> np.insert(np.cumsum(np.array([4, 3, 5])), 0, 0)
        array([ 0,  4,  7, 12])
        """

        return self.__face_point_indices

    @property
    def face_point_list(self) -> NDArray:
        """Returns the face point list.

        Returns
        -------
        NDArray
            Face point array

        Notes
        -----
        Face point array contains ordered indices of points forming the faces in the domain, in
        other words definitions of the faces. It is formed by concatenating ordered point lists for
        all faces.

        Consider a domain formed by 3 faces, defined as [0, 1, 2, 5], [1, 3, 0], [2, 1, 4, 0]
        respectively. For this case the the face point list would be as follows:

        >>> np.concatenate((np.array([0, 1, 2, 5]), np.array([1, 3, 0]), np.array([2, 1, 4, 0])))
        array([0, 1, 2, 5, 1, 3, 0, 2, 1, 4, 0])

        together with the face point index array, the face definitions are complete.

        See also
        --------
        PolyMesh.face_point_indices
        """

        return self.__face_point_list

    @property
    def n_boundaries(self) -> int:
        """Number of defined boundaries.

        Returns
        -------
        int
            Number of boundaries
        """

        return len(self.__boundary)

    @property
    def n_cells(self) -> int:
        """Returns the number of cells.

        Note that the cell count is not stored statically, but computed from the face owner/neigbour
        lists once called.

        Returns
        -------
        int
            Number of cells in the mesh
        """

        return np.max([np.max(self.__face_owner), np.max(self.__face_neighbour)]) + 1

    @property
    def n_faces(self) -> int:
        """Returns the number of faces.

        Note that the face count is not stored statically, but computed from the face
        owner/neighbour lists once called.

        Returns
        -------
        int
            Number of faces in the mesh
        """

        return self.__face_owner.size

    @property
    def n_internal_faces(self) -> int:
        """Returns the number of internal faces.

        Returns
        -------
        int
            Number of internal faces in the mesh.
        """

        return np.argmax(self.__face_neighbour == -1)

    @property
    def n_points(self) -> int:
        """Returns the number of points.

        Returns
        -------
        int
            Number of points in the mesh.
        """

        return self.__points.shape[0]

    @property
    def points(self) -> NDArray:
        """Returns the 2d array containing the point coordinates in three-dimensional space.

        The coordinate arrays has a shape of ``(npts, 3)``.

        Returns
        -------
        NDArray
            Point coordinates
        """

        return self.__points

    @property
    def verbose(self) -> int:
        """Level of verbosity, higher numbers print more information to screen, 0 disables printing.

        Can be set as follows,

        >>> # mesh.verbose = 0  # disable
        >>> # mesh.verbose = 1  # brief
        >>> # mesh.verbose = 2  # detailed
        >>> # mesh.verbose = 3  # very detailed

        Returns
        -------
        int
            Level of verbosity
        """

        return self.__verbose

    @verbose.setter
    def verbose(self, value: int) -> None:

        if not isinstance(value, int):
            raise TypeError

        self.__verbose = value

    # TODO: Documentation and information messages
    def boundary_align_nodes(
        self,
        source_boundary: str,
        target_boundary: str,
        transformation_matrix: NDArray,
        maximum_displacement=np.Inf,
        dry_run=False,
        return_pairs=False,
        verbose=1,
    ) -> Union[None, Tuple[NDArray, NDArray]]:

        if verbose > 2:
            print(f">>> PolyMesh.align_nodes({locals()})")

        # Indices of point on source and target boundaries
        idx_source = self.boundary_points(source_boundary)
        idx_target = self.boundary_points(target_boundary)

        # Corresponding arrays of points
        points_source = self.points[idx_source]
        points_target = self.points[idx_target]

        if points_source.shape[0] != points_target.shape[0]:
            print(
                "Warning: source and target point sets have different number of points"
                f" ({idx_source.size}, {idx_target.size})"
            )

        # Apply the transformation
        points_target_new = PolyMesh.apply_transformation(points_source, transformation_matrix)

        # Compute the point pairs
        pair_distance, pair_index = PolyMesh.find_closest_points(
            points_target_new, points_target
        )  # indexing of "pair_distance" is identical to idx_source
        # Ensure returned arrays have the correct dtype
        pair_distance = pair_distance.astype(self.dtype_float)
        pair_index = pair_index.astype(self.dtype_int)

        # Mask pairs that have a minimum distance less than the specified maximum displacement
        mask = pair_distance < maximum_displacement
        if mask.sum() != mask.size:
            print(
                f"Warning: {mask.size - mask.sum()} pairs exceed the specified maximum displacement"
                f" of {maximum_displacement}"
            )

        if verbose > 0:
            print(17 * " ", "min             max             mean")
            print(17 * " ", "----------------------------------------------")
            print(
                "Distance to pair:"
                f" {np.min(pair_distance):<15.8e} {np.max(pair_distance):<15.8e}"
                f" {np.mean(pair_distance):<15.8e} (before"
                f" alignment) dry_run={dry_run}"
            )

        # Modify the points if this is not a dry run
        if not dry_run:
            self.__points[idx_target[pair_index[mask]], :] = points_target_new[mask]
            if verbose > 0:
                # Compute distances after alignment with a recursive dry run
                _pair_distance, _ = self.boundary_align_nodes(
                    source_boundary,
                    target_boundary,
                    transformation_matrix,
                    dry_run=True,
                    return_pairs=True,
                    verbose=0,
                )
                print(
                    17 * " ",
                    f"{np.min(_pair_distance):<15.8e} {np.max(_pair_distance):<15.8e} "
                    f"{np.mean(_pair_distance):<15.8e} (after"
                    " alignment)",
                )

        if verbose > 2:
            print("<<< PolyMesh.align_nodes\n")

        # mapping array has shape (pair idx, point idx in source/target)
        return pair_distance, (
            (np.vstack((idx_source, idx_target[pair_index])).T) if return_pairs else None
        )

    # TODO: Documentation
    def boundary_add_face(
        self,
        boundary: str,
        faces: Union[int, ArrayLike],
        create=True,
        indexing="current",
        verbose=1,
    ) -> None:

        if boundary not in self.boundary:
            if not create:
                raise KeyError(f"Boundary {boundary} does not exist")
            self.boundary_create(boundary, verbose=verbose)

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _faces = faces
        elif indexing == "original":
            _faces = tuple(self.face_current_index(faces))
        else:
            raise ValueError

        if not np.all(self.face_is_boundary(_faces)):
            raise Exception("At least one face within faces is not a boundary face")

        # Ensure faces is an iterable sequence
        _faces = np.atleast_1d(_faces)
        _faces = _faces[
            np.argsort(_faces)[::-1]
        ]  # decreasing order of deleting produces no offsets

        # Store the face definitions and owners before deleting them
        definitions = self.face_points(_faces)
        owners = self.face_owner[_faces]

        # Delete faces first
        for face in _faces:
            self._del_face(face, indexing="current", verbose=verbose)

        # Add faces to the requested boundary
        for definition, owner in zip(definitions, owners):
            self._add_face_to_boundary(boundary, definition, owner, verbose=verbose)

    def boundary_by_id(self, id: int) -> str:
        """Returns the name of the boundary with given index.

        Negative indices can be used for reverse indexing.

        Parameters
        ----------
        id : int
            Boundary index

        Returns
        -------
        str
            Boundary name
        """

        if id < 0:
            id = self.n_boundaries + id

        try:
            return list(self.boundary.keys())[id]
        except IndexError:
            raise Exception("There is no boundary with id {id}")

    def boundary_face_bounds(self, boundary: str) -> Tuple[int, int]:
        """Returns the start and end face of a given boundary.

        For bounds of ``(4, 7)``, the boundary starts inclusively from face ``4`` and ends at face
        ``7``, e.g. formed by faces ``[4, 5, 6]``.

        Parameters
        ----------
        boundary : str
            Boundary name

        Returns
        -------
        Tuple[int, int]
            Face bounds, (start_face, end_face)
        """

        return (
            self.boundary[boundary]["startFace"],
            self.boundary[boundary]["startFace"] + self.boundary[boundary]["nFaces"],
        )

    def boundary_faces(self, boundary: str) -> NDArray:
        """Returns the index of faces on a given boundary.

        Parameters
        ----------
        boundary : str
            Boundary name

        Returns
        -------
        NDArray
            Face indices
        """

        return np.arange(
            self.boundary[boundary]["startFace"],
            self.boundary[boundary]["startFace"] + self.boundary[boundary]["nFaces"],
        )

    def boundary_cells(self, boundary: str) -> NDArray:
        """Returns the index of cells adjacent to a given boundary.

        Parameters
        ----------
        boundary : str
            Boundary name

        Returns
        -------
        NDArray
            Cell indices
        """

        return self.face_owner[self.boundary_faces(boundary)]

    def boundary_create(self, name: str, type="patch", verbose=1) -> None:
        """Creates an empty boundary with given name.

        Parameters
        ----------
        name : str
            Name of the boundary
        type : str, optional
            Type of the boundary, by default "patch"
        verbose : int, optional
            Level of verbosity, by default 1
        """

        if name in self.boundary:
            raise Exception(f'Boundary "{name}" already exists')

        self.__boundary[name] = OrderedDict({
            "type": type,
            "startFace": self.boundary_face_bounds(self.boundary_by_id(-1))[1],
            "nFaces": 0,
        })

        # The info attribute is used by "recursive_dictionary_writer" while writing the mesh
        self.__boundary[name].info = {"end": "}", "type": "named_dict"}

        if verbose > 0:
            print(
                f'Created empty boundary "{name}" starting from face'
                f' {self.boundary[name]["startFace"]}'
            )

    def boundary_delete_empty(self, verbose=1) -> None:
        """Removes empty boundaries (those with zero faces) from the boundary dictionary."""

        for k in list(self.boundary.keys()):
            if self.boundary[k]["nFaces"] == 0:
                if verbose > 0:
                    print(f'Deleted empty boundary "{k}"')
                del self.__boundary[k]

    def boundary_points(self, boundary: str) -> NDArray:
        """Returns the indices of points lying on a given boundary.

        Parameters
        ----------
        boundary : str
            Boundary name

        Returns
        -------
        NDArray
            Point indices
        """

        return np.unique(np.concatenate(self.face_points(self.boundary_faces(boundary))))

    # TODO: Documentation and typing
    def cell_near_boundary(self, cell: Union[int, ArrayLike]) -> Union[int, NDArray]:

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            [self.cell_near_boundary(_cell) for _cell in cell]
        else:
            return len(
                set(
                    boundary
                    for boundary in self.face_boundary(self.cell_faces(cell))
                    if boundary is not None
                )
            )

    def cell_current_index(self, cell: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Returns the current index of a given cell or sequence of cells.

        Parameters
        ----------
        cell : Union[int, ArrayLike]
            Original cell index (e.g. before calling PolyMesh.reset_offset_log)

        Returns
        -------
        Union[int, NDArray]
            Current cell index or sequence of current cell indices
        """

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.cell_current_index(_cell) for _cell in cell], dtype=self.dtype_int
            )
        else:
            if cell in self.__deleted_cells:
                raise Exception(f"Attempted to refer to a deleted cell with index {cell}")
            return cell + self.__cell_offset_log[cell]

    def cell_original_index(self, cell: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Returns the original index of a given cell or sequence of cells.

        Parameters
        ----------
        cell : Union[int, ArrayLike]
            Current cell index (e.g. after calling PolyMesh.reset_offset_log)

        Returns
        -------
        Union[int, NDArray]
            Original cell index or sequence of original cell indices
        """

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.cell_original_index(_cell) for _cell in cell], dtype=self.dtype_int
            )
        else:
            return np.argmax(
                np.arange(self.__cell_offset_log.size) + self.__cell_offset_log == cell
            )

    def cell_faces(self, cell: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the indices of faces forming a given cell or a sequence of cells.

        Parameters
        ----------
        cell : Union[int, ArrayLike]
            Cell index or a sequence of cell indices

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Face indices, array for a single cell, list of arrays for a sequence of cells
        """

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            return [self.cell_faces(_cell) for _cell in cell]
        else:
            return np.union1d(
                np.where(self.face_owner == cell), np.where(self.face_neighbour == cell)
            )

        # return (
        #     [
        #         np.union1d(np.where(self.face_owner == n), np.where(self.face_neighbour == n))
        #         for n in cell
        #     ]
        #     if isinstance(cell, (collections.abc.Sequence, np.ndarray))
        #     else np.union1d(
        #         np.where(self.face_owner == cell), np.where(self.face_neighbour == cell)
        #     )
        # )

    def cell_neighbour(self, cell: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the indices of neighbour cells for a given cell.

        Parameters
        ----------
        cell : Union[int, ArrayLike]
            Cell index

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Indices of neighbour cells
        """

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            return [self.cell_neighbour(_cell) for _cell in cell]
        else:
            result = np.hstack(self.face_cells(self.cell_faces(cell)))
            return result[result != cell]

    def cell_points(self, cell: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the indices of points forming a given cell or a sequence of cells.

        Parameters
        ----------
        cell : Union[int, ArrayLike]
            Cell index or a sequence of cell indices

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Point indices, array for a single cell, list of arrays for a sequence of cells
        """

        if isinstance(cell, (collections.abc.Sequence, np.ndarray)):
            return [self.cell_points(_cell) for _cell in cell]
        else:
            return np.unique(np.concatenate(self.face_points(self.cell_faces(cell))))

    def cells_shared_face(self, cells: Tuple[int, int]) -> Union[int, None, NDArray]:
        """Returns the index of the shared face between two neighboring cells.

        Parameters
        ----------
        cells : Tuple[int, int]
            Pair of cells

        Returns
        -------
        Union[int, None]
            Index of shared face if cells are neighbour, None otherwise
        """

        result = np.intersect1d(self.cell_faces(cells[0]), self.cell_faces(cells[1]))

        if result.size == 0:
            return None

        return result.item() if result.size == 1 else result

    def edge_length(self, edge: ArrayLike) -> Union[float, NDArray, List]:
        """Returns the length of an edge or a sequence of edges.

        An edge is defined by a sequence of length 2 formed by indices of points corresponding to
        start and end points of the edge.

        Parameters
        ----------
        edge : Union[NDArray, Sequence[NDArray]]
            Edge or sequence of edges, arrays or sequence of arrays having a length of 2 in the last
            axis are supported, e.g. shape (..., 2)

        Returns
        -------
        Union[float, NDArray]
            Edge length or array of edge lengths. If an array is returned, it has the shape of input
            array with last axis removed.
        """

        if isinstance(edge, collections.abc.Sequence):
            return [self.edge_length(_edge) for _edge in edge]
        else:
            return np.linalg.norm(self.points[edge[..., 1]] - self.points[edge[..., 0]], axis=-1)

    def face_mean_edge_length(
        self, face: Union[int, ArrayLike]
    ) -> Union[float, NDArray, List[NDArray]]:
        """Returns the average edge length for a given face or sequence of faces.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices

        Returns
        -------
        Union[float, NDArray, List[NDArray]]
            Average edge length
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.face_mean_edge_length(_face) for _face in face], dtype=self.dtype_float
            )
        else:
            return np.mean(self.edge_length(self.face_edges(face)))

    def face_boundary(self, face: int) -> Union[str, None]:
        """Returns the name of the boundary given face belongs to, or None if face is internal.

        Parameters
        ----------
        face : int
            Face index

        Returns
        -------
        Union[str, None]
            Boundary name or None if face is an internal face
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return [self.face_boundary(_face) for _face in face]
        else:
            for k, v in self.boundary.items():
                if (face >= v["startFace"]) and (face < v["startFace"] + v["nFaces"]):
                    return k
            return None

    def face_cells(
        self, face: Union[int, ArrayLike]
    ) -> Union[int, List[int], NDArray, List[NDArray]]:
        """Returns the index of cells adjacent to the given face.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Index or indices of adjacent cells
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return [self.face_cells(_face) for _face in face]
        else:
            return (
                np.union1d(self.face_owner[face], self.face_neighbour[face])
                if self.face_is_internal(face)
                else self.face_owner[face]
            )

    def face_current_index(self, face: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Returns the current index of a given face or sequence of faces.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Original face index (e.g. before calling PolyMesh.reset_offset_log)

        Returns
        -------
        Union[int, NDArray]
            Current face index or sequence of current face indices
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.face_current_index(_face) for _face in face], dtype=self.dtype_int
            )
        else:
            if face in self.__deleted_faces:
                raise Exception(f"Attempted to refer to a deleted face with index {face}")
            return face + self.__face_offset_log[face]

    def face_original_index(self, face: Union[int, ArrayLike]) -> Union[int, NDArray]:
        """Returns the original index of a given face or sequence of faces.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Current face index (e.g. after calling PolyMesh.reset_offset_log)

        Returns
        -------
        Union[int, NDArray]
            Original face index or sequence of original face indices
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.face_original_index(_face) for _face in face], dtype=self.dtype_int
            )
        else:
            return np.argmax(
                np.arange(self.__face_offset_log.size) + self.__face_offset_log == face
            )

    def face_edges(self, face: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the ordered sequence of edges forming a face.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices.

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Face edge array of shape ``(ne, 2)`` or list of face edge arrays

        Notes
        -----
        An edge is defined by a sequence of 2 integers corresponding to the start and end point
        indices. For a face defined by ordered points ``[1, 2, 3, 4]`` the face edge array is
        ``array([1, 2], [2, 3], [3, 4], [4, 1])``.
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return [self.face_edges(_face) for _face in face]
        else:
            face_points = self.face_points(face)
            return np.array(
                [
                    (face_points[m % face_points.size], face_points[(m + 1) % face_points.size])
                    for m in range(face_points.size)
                ],
                dtype=self.dtype_int,
            )

    def face_points(self, face: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the ordered points forming a face, e.g. the definition of a face.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Point list for a single face is given, list of arrays for a sequence of faces
        """

        if isinstance(face, (collections.abc.Sequence, np.ndarray)):
            return [self.face_points(_face) for _face in face]
        else:
            return self.__face_point_list[
                self.__face_point_indices[face] : self.__face_point_indices[face + 1]
            ]

    def faces_internal(self) -> NDArray:
        """Returns the indices of internal faces.

        Returns
        -------
        NDArray
            Internal face index array
        """

        return np.arange(self.n_internal_faces)

    def faces_boundary(self) -> NDArray:
        """Returns the indices of all boundary faces.

        Returns
        -------
        NDArray
            Boundary face index array
        """

        return np.arange(self.n_internal_faces, self.n_faces)

    def face_is_boundary(self, face: Union[int, ArrayLike]) -> Union[bool, NDArray]:
        """Returns True if face is a boundary, False otherwise.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices

        Returns
        -------
        Union[bool, NDArray]
            True/False for a single face or a bool array for a sequence of face indices
        """

        if isinstance(face, collections.abc.Sequence):
            return [self.face_is_boundary(_face) for _face in face]
        else:
            return face >= self.n_internal_faces

    def face_in_boundary(self, face: Union[int, ArrayLike], boundary: str) -> Union[bool, NDArray]:
        """Returns True if face is on the given boundary, False otherwise.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices
        boundary : str
            Boundary name

        Returns
        -------
        Union[bool, NDArray]
            True/False for a single face or a bool array for a sequence of face indices
        """
        if isinstance(face, collections.abc.Sequence):
            return [self.face_in_boundary(_face) for _face in face]
        else:
            bounds = self.boundary_face_bounds(boundary)
            return (face >= bounds[0]) & (face < bounds[1])

    def face_is_internal(self, face: Union[int, ArrayLike]) -> Union[bool, NDArray]:
        """Returns True if face is an internal face, False otherwise.

        Parameters
        ----------
        face : Union[int, ArrayLike]
            Face index or a sequence of face indices

        Returns
        -------
        Union[bool, NDArray]
            True/False for a single face or a bool array for a sequence of face indices
        """

        if isinstance(face, collections.abc.Sequence):
            return [self.face_is_internal(_face) for _face in face]
        else:
            return face < self.n_internal_faces

    def faces_shared_points(self, faces: Tuple[int, int]) -> NDArray:
        """Returns the shared points between two faces.

        Parameters
        ----------
        faces : Tuple[int, int]
            Pair of faces

        Returns
        -------
        NDArray
            Array containing indices of shared points, empty array if there are no shared points
        """

        return np.intersect1d(self.face_points(faces[0]), self.face_points(faces[1]))

    def point_cells(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the cells that contains the given point in their definition.

        If a sequence of points is given, same operation will be conducted on the sequence. In other
        words, the operation will NOT filter cells referring to all points in the sequence at the
        same time.

        Parameters
        ----------
        point : Union[int, ArrayLike]
            Point index or a sequence of point indices

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Array containing cell indices for a single point, or list of arrays for a sequence of
            points

        Note
        ----
        This operation is not based on a lookup table, computation is carried out on-demand with
        a jitted routine. It is not a cheap computation, be aware that for large inputs large
        computational times may be required.
        """

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return [self.point_cells(_point) for _point in point]
        else:
            return PolyMesh.__parents_of_point(
                point,
                self.__face_point_indices,
                self.__face_point_list,
                self.__face_owner,
                self.__face_neighbour,
            )[1]

    def point_current_index(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the current index of a given point or sequence of points.

        Parameters
        ----------
        point : Union[int, ArrayLike]
            Original point index (e.g. before calling PolyMesh.reset_offset_log)

        Returns
        -------
        Union[int, NDArray]
            Current point index or sequence of current point indices
        """

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.point_current_index(_point) for _point in point], dtype=self.dtype_int
            )
        else:
            if point in self.__deleted_points:
                raise Exception(f"Attempted to refer to a deleted face with index {point}")
            return point + self.__point_offset_log[point]

    def point_original_index(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the original index of a given point or sequence of points.

        Parameters
        ----------
        point : Union[int, ArrayLike]
            Current point index

        Returns
        -------
        Union[int, NDArray]
            Original point index or sequence of original point indices
        """

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return np.array(
                [self.point_original_index(_point) for _point in point], dtype=self.dtype_int
            )
        else:
            return np.argmax(
                np.arange(self.__point_offset_log.size) + self.__point_offset_log == point
            )

    def point_faces(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:
        """Returns the faces that contains the given point in their definition.

        If a sequence of points is given, same operation will be conducted on the sequence. In other
        words, the operation will NOT filter faces referring to all points in the sequence at the
        same time.

        Parameters
        ----------
        point : Union[int, ArrayLike]
            Point index or a sequence of point indices

        Returns
        -------
        Union[NDArray, List[NDArray]]
            Array containing face indices for a single point, or list of arrays for a sequence of
            points

        Note
        ----
        This operation is not based on a lookup table, but computation is carried out on-demand with
        a jitted routine. It is not a cheap computation, be aware that for large inputs large
        computational times may be required.
        """

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return [self.point_faces(_point) for _point in point]
        else:
            return PolyMesh.__parents_of_point(
                point,
                self.__face_point_indices,
                self.__face_point_list,
                self.__face_owner,
                self.__face_neighbour,
            )[0]

    # TODO: Type hinting
    def point_boundary(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return [self.point_boundary(_point) for _point in point]
        else:
            boundaries = set(self.face_boundary(self.point_faces(point)))
            if None in boundaries:
                boundaries.remove(None)
            return boundaries

    def point_is_boundary(self, point: Union[int, ArrayLike]) -> Union[NDArray, List[NDArray]]:

        if isinstance(point, (collections.abc.Sequence, np.ndarray)):
            return [self.point_is_boundary(_point) for _point in point]
        else:
            return np.any(self.face_is_boundary(self.point_faces(point)))

    def write(self, path: str, **kwargs) -> None:
        """Writes the current state of this ``PolyMesh`` instance in OpenFOAM format.

        Parameters
        ----------
        path : str
            Path to write, will create "constant/polyMesh" under the given path

        Keyword arguments
        -----------------
        mode : str
            Writing mode, either "binary" or "ascii". Default is "binary".
        ascii_float_precision: int
            Floating point number precision for ascii output. Default is 18.
        verbose: int
            Level of verbosity, higher numbers print more information. Default is 1.
        version: str
            Value of "version" key to be written in the "FoamFile" dictionary. Default is "2.0".
        byteorder: str
            Byte order for binary output. Default is "little".
        encoding: str
            Encoding to use for ascii output. Default is "utf-8".
        """

        settings = {
            "mode": "binary",
            "ascii_float_precision": 18,
            "verbose": 1,
            "version": "2.0",
            "byteorder": "little",
            "encoding": "utf-8",
        }

        for k, v in kwargs.items():
            if k not in settings:
                raise ValueError(
                    f'"{k}" is not a valid keyword argument, valid options are'
                    f" {list(settings.keys())}"
                )
            if type(k) is not type(settings[k]):
                raise TypeError(
                    f'Incorrect type for "{k}", expected {type(settings[k])} found {type(k)}'
                )
            settings[k] = v

        _path = Path(path)
        _path = _path.joinpath("constant", "polyMesh")
        _path.mkdir(parents=True, exist_ok=True)

        # Points
        header = {
            "version": settings["version"],
            "format": settings["mode"],
            "class": "vectorField",
            "object": "points",
        }
        write_vector_field(
            _path.joinpath("points").as_posix(),
            self.__points,
            header,
            byteorder=settings["byteorder"],
            encoding=settings["encoding"],
            format=f'%.{settings["ascii_float_precision"]}f',
            verbose=settings["verbose"],
        )

        # Faces
        header = {
            "version": settings["version"],
            "format": settings["mode"],
            "class": "faceList" if settings["mode"] == "ascii" else "faceCompactList",
            "object": "faces",
        }
        write_faces(
            _path.joinpath("faces").as_posix(),
            self.__face_point_indices,
            self.__face_point_list,
            header,
            byteorder=settings["byteorder"],
            encoding=settings["encoding"],
            verbose=settings["verbose"],
        )

        # Face owner
        header = {
            "version": settings["version"],
            "format": settings["mode"],
            "class": "labelList",
            "object": "owner",
        }
        write_scalar_field(
            _path.joinpath("owner").as_posix(),
            self.__face_owner,
            header,
            byteorder=settings["byteorder"],
            encoding=settings["encoding"],
            format="%d",
            verbose=settings["verbose"],
        )

        # Face owner
        header = {
            "version": settings["version"],
            "format": settings["mode"],
            "class": "labelList",
            "object": "neighbour",
        }
        write_scalar_field(
            _path.joinpath("neighbour").as_posix(),
            self.__face_neighbour,
            header,
            byteorder=settings["byteorder"],
            encoding=settings["encoding"],
            format="%d",
            verbose=settings["verbose"],
        )

        # Boundary
        header = collections.OrderedDict({
            "version": settings["version"],
            "format": settings["mode"],
            "class": "polyBoundaryMesh",
            "object": "boundary",
        })
        header.info = {}  # this attribute is required for "recursive_dictionary_writer"
        header.info.__setitem__("type", "named_dict")
        header.info.__setitem__("end", "}")
        recursive_dictionary_writer(
            _path.joinpath("boundary").as_posix(),
            {"FoamFile": header, "boundary": self.__boundary},
            verbose=settings["verbose"],
        )

    # TODO: This is replaced by "face_current_index" and "face_original_index"
    def _face_offset_log(self, face: int) -> int:
        """Returns the current offset value for a given face index from "before" state.

        For internal use during mesh modification.

        Parameters
        ----------
        face : int
            Face index corresponding to the "before" state

        Returns
        -------
        int
            Face index corresponding to the "current" state

        Raises
        ------
        Exception
            Face was deleted
        """

        if face in self.__deleted_faces:
            raise Exception(f"Attempted to refer to a deleted face with index {face}")

        return self.__face_offset_log[face]

    # TODO: Keyword arguments
    def _read(self, path: str) -> None:
        """Reads the polyMesh and populates internal containers (for internal use)."""

        # Points
        self.__points = read_vector_field(
            os.path.join(path, "points"),
            dtype=self.dtype_float,
            byteorder=self.__byteorder,
            verbose=self.verbose,
        )
        # Face definitions
        self.__face_point_indices, self.__face_point_list = read_faces(
            os.path.join(path, "faces"),
            dtype=self.dtype_int,
            byteorder=self.__byteorder,
            verbose=self.verbose,
        )
        # Face owner list
        self.__face_owner = read_scalar_field(
            os.path.join(path, "owner"),
            dtype=self.dtype_int,
            byteorder=self.__byteorder,
            verbose=self.verbose,
        )
        # Face neighbour list
        self.__face_neighbour = read_scalar_field(
            os.path.join(path, "neighbour"),
            dtype=self.dtype_int,
            byteorder=self.__byteorder,
            verbose=self.verbose,
        )
        # Bounary dictionary
        self.__boundary = recursive_dictionary_parser(
            os.path.join(path, "boundary"), verbose=self.verbose
        )["boundary"]

    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
    def __build_cell_face_list(
        face_owner: NDArray, face_neighbour: NDArray, maximum_cell_faces: int = 20
    ) -> Tuple[NDArray, NDArray]:
        """Computes cell definitions in terms of faces (similar to "compactFaceList").

        This routine can be used to compute cell definitions in advance to obtain cell indices from
        points and faces without performing expensive computations.

        Parameters
        ----------
        face_owner : NDArray
            Face owner array
        face_neighbour : NDArray
            Face neighbour array
        maximum_cell_faces : int, optional
            Maximum number of faces a cell can have, by default 20. This is only used to
            pre-allocate a safe amount of memory during operations.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Cell face index array and cell face arrays
        """

        dtype = face_owner.dtype
        # Compute number of cells
        n_cells = max(np.max(face_owner), np.max(face_neighbour)) + 1
        n_faces = face_owner.size
        # Initialize arrays, overallocate cell faces to reduce later
        cell_face_indices = np.zeros(
            n_cells, dtype=dtype
        )  # This acts as cell face count in below loops
        cell_face_list = np.full((n_cells, maximum_cell_faces), -1, dtype=dtype)
        # Find the start index of boundary faces
        n_internal_faces = np.argmax(face_neighbour == -1)
        # First loop within the range of internal faces - common for owner and neighbour
        for face in range(n_internal_faces):
            cell1 = face_owner[face]
            cell2 = face_neighbour[face]
            cell_face_indices[cell1] += 1
            cell_face_indices[cell2] += 1
            cell_face_list[cell1, cell_face_indices[cell1] - 1] = face
            cell_face_list[cell2, cell_face_indices[cell2] - 1] = face
        # Second loop - only for owner
        for face in range(n_internal_faces, n_faces):
            cell1 = face_owner[face]
            cell_face_indices[cell1] += 1
            cell_face_list[cell1, cell_face_indices[cell1] - 1] = face
        # Reduce
        cell_face_list = cell_face_list.flatten()
        cell_face_list = cell_face_list[cell_face_list >= 0]
        cell_face_indices = np.cumsum(cell_face_indices)
        cell_face_indices = np.concatenate((np.array([0], dtype=dtype), cell_face_indices))

        return cell_face_indices, cell_face_list

    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False)
    def __parents_of_point(
        point: int,
        face_point_indices: NDArray,
        face_point_list: NDArray,
        face_owner: NDArray,
        face_neighbour: NDArray,
    ) -> Tuple[NDArray, NDArray]:

        n_faces = face_point_indices.size - 1
        n_internal_faces = np.argmax(face_neighbour == -1)

        nearby_faces = []
        nearby_cells = []

        for face in range(n_internal_faces):
            face_points = face_point_list[face_point_indices[face] : face_point_indices[face + 1]]
            if point in face_points:
                nearby_faces.append(face)
                nearby_cells.append(face_owner[face])
                nearby_cells.append(face_neighbour[face])
        for face in range(n_internal_faces, n_faces):
            face_points = face_point_list[face_point_indices[face] : face_point_indices[face + 1]]
            if point in face_points:
                nearby_faces.append(face)
                nearby_cells.append(face_owner[face])

        return np.unique(np.array(nearby_faces, dtype=face_owner.dtype)), np.unique(
            np.array(nearby_cells, dtype=face_owner.dtype)
        )

    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True, boundscheck=False)
    def find_closest_points(
        source=np.array([[]], dtype=np.float64), target=np.array([[]], dtype=np.float64)
    ) -> Tuple[NDArray, NDArray]:
        """Finds the nearest points (e.g. pairs) in the source array to the target array.

        Parameters
        ----------
        source : _type_, optional
            Array containing "source" points, shape (:, 3)
        target : _type_, optional
            Array containing "target" points, shape (:, 3)

        Returns
        -------
        Tuple[NDArray, NDArray]
            Distance and pair index arrays. The indices of these arrays are same as the "source"
            points array, and pair index contains the index of the corresponding point in the
            "target" points array.
        """

        delta = np.zeros(source.shape[0], dtype=source.dtype)
        distance = np.zeros(source.shape[0], dtype=source.dtype)
        index = np.zeros(source.shape[0], dtype=np.int64)  # casted later to selected dtype

        # For each point in source array, compute the distance to all points in target array and
        # store the value and index of the minimum distance
        for n in nb.prange(source.shape[0]):
            delta = np.sqrt(
                (target[:, 0] - source[n, 0]) * (target[:, 0] - source[n, 0])
                + (target[:, 1] - source[n, 1]) * (target[:, 1] - source[n, 1])
                + (target[:, 2] - source[n, 2]) * (target[:, 2] - source[n, 2])
            )
            index[n] = np.argmin(delta)
            distance[n] = delta[index[n]]

        return distance, index

    @staticmethod
    def apply_transformation(points: NDArray, transformation_matrix: NDArray) -> NDArray:
        """Apply given transformation on the given points.

        The transformation matrix should be a 4x4 affine transformation matrix which can contain
        rotation and translation simultaneously. See ``transformations`` module for obtaining
        such matrices.

        Parameters
        ----------
        points : NDArray
            Array of point coordinates, shape (npts,3)
        transformation_matrix : NDArray
            Transformation matrix, shape (4,4)

        Returns
        -------
        NDArray
            Coordinates of transformed points, shape (npts,3)
        """

        if (points.ndim != 2) or (points.shape[-1] != 3):
            raise ValueError("Invalid shape for points array")

        if transformation_matrix.shape != (4, 4):
            raise ValueError("Invalid shape for transformation matrix")

        return (transformation_matrix @ np.vstack((points.T, np.ones(points.shape[0])))).T[:, :3]

    def merge_cells(
        self,
        cells: Tuple[int, int],
        indexing="current",
        merge_internal_faces=True,
        merge_boundary_faces=True,
        verbose=1,
    ) -> None:

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _cells = cells
        elif indexing == "original":
            _cells = tuple(self.cell_current_index(cells))
        else:
            raise ValueError

        # Find the internal face to delete
        face = self.cells_shared_face(_cells)

        if face is None:
            raise ValueError(
                f"Given cells {_cells}({self.cell_original_index(_cells)}) are not neighbours"
            )

        # Arbitrarily select the smaller cell index to keep
        merged_cell = min(_cells)
        deleted_cell = max(_cells)

        # Check boundary adjacency
        boundaries_merged_cell = set(self.face_boundary(self.cell_faces(merged_cell)))
        boundaries_merged_cell.remove(None)
        boundaries_deleted_cell = set(self.face_boundary(self.cell_faces(deleted_cell)))
        boundaries_deleted_cell.remove(None)

        common_boundaries = boundaries_merged_cell.intersection(boundaries_deleted_cell)

        if verbose > 0:
            print(f"Merging cells {_cells}({self.cell_original_index(_cells)}) ...")
            if len(common_boundaries) > 0:
                print(f"Both cells are adjacent to boundary(s) {common_boundaries}")
        if verbose > 1:
            print(f"Using index {merged_cell} for the merged cell and deleting cell {deleted_cell}")
            print(f"Deleting internal face {face} with definition {self.face_points(face)}")

        # Replace references to deleted cell with the merged cell
        self._replace_cell((deleted_cell, merged_cell), indexing="current", verbose=verbose)

        # Delete the internal face
        self._del_face(face, indexing="current", verbose=verbose)

        # Check if the merged cell has more than one face on the same boundary
        cell_boundaries_list = self.face_boundary(self.cell_faces(merged_cell))
        cell_boundaries_set = set(cell_boundaries_list)
        try:
            cell_boundaries_set.remove(None)  # remove entries due to internal faces
        except KeyError:
            raise RuntimeError(f"Cell {merged_cell} has no internal faces")
        duplicate_faces = []
        for boundary in cell_boundaries_set:
            count = cell_boundaries_list.count(boundary)
            # Cell has one face on a given boundary, do nothing
            if count == 1:
                continue
            # Cell has two faces on a given boundary, these faces need to be merged
            if count == 2:
                print(
                    f"Cell {merged_cell}({self.cell_original_index(merged_cell)}) has duplicate"
                    f' faces on boundary "{boundary}"'
                )
                for _face in self.cell_faces(merged_cell):
                    if self.face_boundary(_face) == boundary:
                        duplicate_faces.append(_face)
                if len(duplicate_faces) != 2:
                    raise Exception(
                        f"Cell {merged_cell}({self.cell_original_index(merged_cell)} has duplicate"
                        " faces on more than one boundary"
                    )
            # Cell has more than two faces on a given boundary -- does not make sense
            else:
                raise Exception(
                    f"Cell {merged_cell}({self.cell_original_index(merged_cell)} has {count} faces"
                    f" on boundary {boundary}"
                )
        # Merge pair of boundary faces if permitted
        if merge_boundary_faces and len(duplicate_faces) != 0:
            self._merge_faces(tuple(duplicate_faces), indexing="current", verbose=verbose)

        # Check if the merged cell has "duplicated neighbours" and merge if permitted
        unique_neighbours, count = np.unique(self.cell_neighbour(merged_cell), return_counts=True)
        duplicate_neighbours = unique_neighbours[count > 1]
        if verbose > 1:
            print(f"Found {duplicate_neighbours.size} duplicate neighbours")
        for duplicate_neighbour in duplicate_neighbours:
            faces = self.cells_shared_face((merged_cell, duplicate_neighbour))
            if merge_internal_faces:
                if verbose > 0:
                    print(f"Merging duplicate faces {faces}({self.face_original_index(faces)})")
                self._merge_faces(
                    tuple(faces),
                    indexing="current",
                    verbose=verbose,
                )

        if verbose > 0:
            print("Done.")

    def delete_internal_face(self, face_to_delete: int, verbose=1) -> None:
        """Delete an internal face, leading to merger of adjacent cells.

        This does the following operations:

        1. Selects the cell with the smaller index as the index of the newly formed, merged cell,
        2. Removes all reference to the deleted cell in the face owner/neighbour lists,
        3. Removes the deleted face from face point index and face point list arrays,
        4. Removes the deleted face from face owner/neighbour lists,
        5. Decrements "startFace" for all boundaries,
        6. Updates the cell indices in face owner/neighbour lists to accomodate deleted cell,
        7. Updates the face and cell book keeping.

        Parameters
        ----------
        face_to_delete : int
            Index of face to delete
        verbose : int, optional
            Level of verbosity, by default 1
        """

        offset = self._face_offset_log(face_to_delete)
        face = face_to_delete + offset

        # Verify face to delete is an internal face
        if face_to_delete >= self.n_internal_faces - 1:
            raise ValueError

        if verbose > 0:
            print(f"Deleting internal face {face} with definition {self.face_points(face)}")

        # With this operation two internal cells are merged, we arbitrarily select the smaller cell
        # index to keep
        merged_cell = min(self.face_owner[face], self.face_neighbour[face])
        deleted_cell = max(self.face_owner[face], self.face_neighbour[face])

        if verbose > 1:
            print(
                f"Merging cells {self.face_owner[face]} and {self.face_neighbour[face]}, using"
                f" {merged_cell} as the merged cell index"
            )

        # Replace all reference to the deleted cell in the owner/neighbour cell lists
        indices = np.where(self.__face_owner == deleted_cell)
        self.__face_owner[indices] = merged_cell
        if verbose > 1:
            print(
                f"Updated {len(indices)} reference(s) to cell {deleted_cell} in the owner cell list"
            )

        indices = np.where(self.__face_neighbour == deleted_cell)
        self.__face_neighbour[indices] = merged_cell
        if verbose > 1:
            print(
                f"Updated {len(indices)} reference(s) to cell {deleted_cell} in the neighbour cell"
                " list"
            )

        # Remove all information coresponding to the index of this face
        self.__face_point_list = np.delete(
            self.__face_point_list,
            np.arange(self.__face_point_indices[face], self.__face_point_indices[face + 1]),
        )
        self.__face_point_indices = np.insert(
            np.cumsum(np.delete(np.diff(self.__face_point_indices), face)), 0, 0
        )
        self.__face_owner = np.delete(self.__face_owner, face)
        self.__face_neighbour = np.delete(self.__face_neighbour, face)

        if verbose > 1:
            print(
                f"Removed face {face} from face owner/neighbour lists, updated face point list and"
                " face point indices"
            )

        # Start face of all boundaries should offset one face
        for boundary in self.__boundary.keys():
            self.__boundary[boundary]["startFace"] -= 1

        if verbose > 1:
            print("Updated start indices of boundaries")

        # Update cell indices
        self.__face_owner[self.__face_owner > deleted_cell] -= 1
        self.__face_neighbour[self.__face_neighbour > deleted_cell] -= 1

        if verbose > 1:
            print("Updated cell indices in owner/neighbour lists")

        # Update book keeping
        self.__deleted_faces.append(face - offset)
        self.__deleted_cells.append(deleted_cell)
        self.__face_offset_log[face - offset + 1 :] -= 1

        if verbose > 0:
            print("Done.")

        return

    def _replace_face(self, face: int, points: NDArray, indexing="current", verbose=1) -> None:

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _face = face
        elif indexing == "original":
            _face = self.face_current_index(face)
        else:
            raise ValueError

        if verbose > 0:
            print(
                f"Replacing definition of face {_face}({self.face_original_index(_face)}) with"
                f" {points} ...",
                end=" ",
            )

        indices = np.arange(self.__face_point_indices[_face], self.__face_point_indices[_face + 1])
        # offset = len(points) - self.face_points(face).size

        self.__face_point_indices = np.concatenate((
            self.__face_point_indices[: _face + 1],
            self.__face_point_indices[_face + 1 :] + len(points) - self.face_points(_face).size,
        ))
        self.__face_point_list = np.insert(
            np.delete(self.__face_point_list, indices), self.__face_point_indices[_face], points
        )

        if verbose > 0:
            print("done", flush=True)

    def _replace_cell(self, cells: Tuple[int, int], indexing="current", verbose=1) -> None:

        if verbose > 2:
            print(f">>> PolyMesh._delete_cell {locals()}")

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            cell_old, cell_new = cells
        elif indexing == "original":
            cell_old, cell_new = tuple(self.cell_current_index(cells))
        else:
            raise ValueError

        if verbose > 0:
            print(
                f"Replacing cell {cell_old}({self.cell_original_index(cell_old)}) with"
                f" {cell_new}({self.cell_original_index(cell_new)}) ..."
            )

        # Replace all reference to the deleted cell in the face owner list
        indices = np.where(self.__face_owner == cell_old)
        self.__face_owner[indices] = cell_new
        if verbose > 1:
            print(f"Updated {len(indices)} reference(s) to cell {cell_old} in the owner cell list")
        # Replace all reference to the deleted cell in the face neighbour list
        indices = np.where(self.__face_neighbour == cell_old)
        self.__face_neighbour[indices] = cell_new
        if verbose > 1:
            print(
                f"Updated {len(indices)} reference(s) to cell {cell_old} in the neighbour cell list"
            )
        # Update cell indices
        self.__face_owner[self.__face_owner > cell_old] -= 1
        self.__face_neighbour[self.__face_neighbour > cell_old] -= 1
        # Update book keeping
        self.__deleted_cells.append(self.cell_original_index(cell_old))
        self.__cell_offset_log[self.cell_original_index(cell_old) + 1 :] -= 1

    def _del_face(self, face: int, indexing="current", verbose=1) -> None:

        if verbose > 2:
            print(f">>> PolyMesh._delete_face {locals()}")

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _face = face
        elif indexing == "original":
            _face = tuple(self.face_current_index(face))
        else:
            raise ValueError

        internal_face = self.face_is_internal(_face)

        if internal_face:
            if verbose > 0:
                print(f"Removing references to internal face {_face} ...", end=" ")
        else:
            if verbose > 0:
                print(f"Removing references to boundary face {_face} ...")

        # Store the point count before deleting
        size = self.face_points(_face).size
        # Remove from face point list array
        self.__face_point_list = np.delete(
            self.__face_point_list,
            np.arange(self.__face_point_indices[_face], self.__face_point_indices[_face + 1]),
        )
        # Remove from face point index array
        # self.__face_point_indices = np.delete(self.__face_point_indices, _face + 1)
        # self.__face_point_indices[_face:] -= size
        self.__face_point_indices = np.insert(
            np.cumsum(np.delete(np.diff(self.__face_point_indices), _face), dtype=self.dtype_int),
            0,
            0,
        )
        # Remove the face from face owner/neighbour lists
        self.__face_owner = np.delete(self.__face_owner, _face)
        self.__face_neighbour = np.delete(self.__face_neighbour, _face)

        # Handle boundaries
        if internal_face:
            # Decrement start face of all boundaries
            for boundary in self.__boundary.keys():
                self.__boundary[boundary]["startFace"] -= 1
        else:
            # Decrement the face count of the boundary this face belongs to
            self.__boundary[self.face_boundary(_face)]["nFaces"] -= 1
            # Decrement the start face of boundaries coming after this boundary
            for boundary in self.boundary.keys():
                if self.boundary[boundary]["startFace"] >= _face:
                    self.__boundary[boundary]["startFace"] -= 1

        # Update book keeping
        self.__deleted_faces.append(self.face_original_index(_face))
        self.__face_offset_log[self.face_original_index(_face) + 1 :] -= 1

        if verbose > 0:
            print("Done.", flush=True)

        if verbose > 2:
            print("<<< PolyMesh._delete_face")

    def _merge_faces(self, faces: Tuple[int, int], indexing="current", verbose=1) -> None:

        if len(faces) != 2:
            raise ValueError(f"Expected a pair of faces, got {len(faces)} elements")

        if len(set(faces)) == 1:
            raise ValueError("Can not merge a face with itself")

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _faces = faces
        elif indexing == "original":
            _faces = tuple(self.face_current_index(faces))
        else:
            raise ValueError

        # Check if faces are sharing an edge
        shared_points = self.faces_shared_points(_faces)
        if len(shared_points) == 0:
            raise Exception(
                f"Faces {_faces}({self.face_original_index(_faces)}) do not have a shared edge"
            )

        # Check if both faces are pointing to same cells
        faces_cells = self.face_cells(_faces)
        if set(np.atleast_1d(faces_cells[0]).tolist()) != set(
            np.atleast_1d(faces_cells[1]).tolist()
        ):
            raise Exception(
                f"Faces {_faces}({self.face_original_index(_faces)}) are not associated with the"
                f" same cells ({faces_cells})"
            )

        # Check if this is a valid topology
        if len(shared_points) != 2:
            raise RuntimeError(
                f"Faces {_faces}({self.face_original_index(_faces)}) have more than 2 shared points"
            )

        # Check if it makes sense to merge these faces
        if np.all(self.face_is_internal(_faces)):
            internal_faces = True
        elif np.all(self.face_is_boundary(_faces)):
            internal_faces = False
        else:
            raise ValueError("Can not merge an internal face with a boundary face")

        # Arbitrarily select the smaller face index to keep
        merged_face = min(_faces)
        deleted_face = max(_faces)

        if verbose > 0:
            if internal_faces:
                print(f"Merging internal faces {_faces} ...")
            else:
                print(f"Merging boundary faces {_faces} ...")
        if verbose > 1:
            print(f"Using index {merged_face} for the merged face and deleting face {deleted_face}")

        # Check if we can delete any point
        points_to_delete = []
        point_boundaries = self.point_boundary(shared_points)
        if (len(point_boundaries[0]) == 0) and (len(point_boundaries[1]) >= 1):
            points_to_delete.append(shared_points[1])
            if verbose > 0:
                print(f"Point {shared_points[1]} on the shared edge is marked for deletion")
        elif (len(point_boundaries[1]) == 0) and (len(point_boundaries[0]) >= 1):
            points_to_delete.append(shared_points[0])
            if verbose > 0:
                print(f"Point {shared_points[0]} on the shared edge is marked for deletion")

        # If we are merging internal faces, and this does not result with a point deletion, stop
        # (this often results in highly concave faces, and requires further checks)
        if not points_to_delete and internal_faces:
            if verbose > 0:
                print("Cancelling merger as this does not result in a free point")
            return

        # Compute definition of the merged face
        merged_face_points = self.compute_merged_face(_faces, points_to_delete, verbose=verbose)
        if verbose > 1:
            print(f"Ordered point sequence for the merged face is {merged_face_points}")

        # Update the definition of the merged face
        self._replace_face(merged_face, merged_face_points, indexing="current", verbose=verbose)

        # Remove the deleted face
        self._del_face(deleted_face, indexing="current", verbose=verbose)

        # Delete the marked point
        if points_to_delete:
            self._del_point(
                points_to_delete[0], replace_faces=True, indexing="current", verbose=verbose
            )

        if verbose > 0:
            print("Done.")

    def _del_point(self, point: int, replace_faces=True, indexing="current", verbose=1):

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _point = point
        elif indexing == "original":
            _point = self.point_current_index(point)
        else:
            raise ValueError

        if verbose > 0:
            print(f"Deleting point {_point}({self.point_original_index(point)}) ...")

        # Check if there is still any reference to this point
        if np.any(self.__face_point_list == _point):
            if not replace_faces:
                raise Exception(
                    f"Can not delete point {_point}({self.point_original_index(_point)}), it is"
                    " still referenced -- set replace_faces=True to override"
                )
            # Find which faces still refer to this point
            faces = self.point_faces(_point)
            # Update point index list
            for face in faces:
                self.__face_point_indices[face + 1 :] -= 1
            # Update point list
            self.__face_point_list = self.__face_point_list[self.__face_point_list != _point]
            if verbose > 0:
                print(f"Modified definition of {len(faces)} faces")

        # Remove this point from the point coordinate array
        self.__points = np.delete(self.__points, _point, axis=0)

        # Modify the index of other points
        self.__face_point_list[self.__face_point_list > _point] -= 1

        self.__deleted_faces.append(self.point_original_index(_point))
        self.__point_offset_log[self.point_original_index(_point) + 1 :] -= 1

        if verbose > 0:
            print("Done.")

    def _add_face_to_boundary(self, boundary: str, points: NDArray, owner: int, verbose=1) -> None:

        # Two cases to be handled differently: we insert face to the end of face list or else
        if boundary == self.boundary_by_id(-1):
            if verbose > 0:
                print(
                    f'Adding face with definition {points} to boundary "{boundary}" at the end of'
                    " face list"
                )
            # Append the face definition in face point list
            self.__face_point_list = np.append(self.__face_point_list, points)
            # Append point count into the face index list
            self.__face_point_indices = np.append(
                self.__face_point_indices,
                self.__face_point_indices[-1] + np.array(len(points), dtype=np.int32),
            )
            # Append into face owner/neighbour lists
            self.__face_owner = np.append(self.__face_owner, owner)
            self.__face_neighbour = np.append(self.__face_neighbour, np.array(-1, dtype=np.int32))
        else:
            # Insert to the last position in the given boundary
            index = np.max(self.boundary_faces(boundary))
            if verbose > 0:
                print(
                    f'Adding face with definition {points} to boundary "{boundary}" with face index'
                    f" {index}"
                )
            # Insert the face definition in face point list
            self.__face_point_list = np.insert(
                self.__face_point_list, self.__face_point_indices[index], points
            )
            # Insert point count into the face index list and increment affected cumulative counts
            self.__face_point_indices = np.insert(
                self.__face_point_indices, index + 1, self.__face_point_indices[index] + len(points)
            )
            self.__face_point_indices[index + 2 :] += len(points)
            # Insert into face owner/neighbour lists
            self.__face_owner = np.insert(self.__face_owner, index, owner)
            self.__face_neighbour = np.insert(self.__face_neighbour, index, -1)
            # Update book keeping
            self.__face_offset_log[index:] += 1

        # Modify boundary dictionary
        self.__boundary[boundary]["nFaces"] += 1
        for k in self.boundary.keys():
            if self.boundary[k]["startFace"] > self.boundary[boundary]["startFace"]:
                self.__boundary[k]["startFace"] += 1

    def compute_merged_face(
        self, faces: Tuple[int, int], points_to_delete: Sequence[int], indexing="current", verbose=1
    ) -> NDArray:
        """Computes the definition (ordered point sequence) resulting from merger of two faces.

        Parameters
        ----------
        faces : Tuple[int, int]
            Face pair
        points_to_delete : Sequence[int]
            Point(s) on the common edge between faces to delete, provide empty list to omit
        verbose : int, optional
            Level of verbosity, by default 1

        Returns
        -------
        NDArray
            Definition of merged face
        """

        if len(faces) != 2:
            raise ValueError("Can only merge two faces at a time")

        if len(points_to_delete) > 2:
            raise ValueError("Can not delete more than 2 points")

        # Ensure the indices we are going to work with refer to the "current" state
        if indexing == "current":
            _faces = faces
        elif indexing == "original":
            _faces = self.face_current_index(faces)
        else:
            raise ValueError

        # Compute the shared points between given faces
        common_points = self.faces_shared_points(_faces).tolist()
        if len(common_points) != 2:
            raise Exception(
                f"Given faces {_faces} have {len(common_points)} shared points (required 2)"
            )
        # Verify points to delete are contained within the common points
        for point in points_to_delete:
            if point not in common_points:
                raise Exception(
                    f"Given point to delete {point} is not a shared point between faces {_faces}"
                )
        if verbose > 1:
            print(f"Common points between faces {_faces} are {common_points}")
        # Label given faces arbitrarily and choose the last common point in "face 1" as anchor
        face_1 = self.face_points(_faces[0])
        face_2 = self.face_points(_faces[1])
        if verbose > 2:
            print(f"Selected face {_faces[0]} as face_1")
        for i in range(face_1.size):
            next = (i + 1) % face_1.size
            if face_1[i] in common_points and face_1[next] in common_points:
                anchor = face_1[next]
                break
        if verbose > 2:
            print(f"Using point {anchor} as anchor")
        # Assemble the merged face
        merged_face = []
        idx_anchor_face_1 = np.argwhere(face_1 == anchor)[0][0]
        idx_anchor_face_2 = np.argwhere(face_2 == anchor)[0][0]
        if verbose > 2:
            print(f"Index of anchor point in face_1 is {idx_anchor_face_1}")
            print(f"Index of anchor point in face_2 is {idx_anchor_face_2}")
        for i in range(idx_anchor_face_1, idx_anchor_face_1 + face_1.size):
            merged_face.insert(-1, face_1[i % face_1.size])
        for i in range(
            idx_anchor_face_2 + len(common_points),
            idx_anchor_face_2 + len(common_points) + face_2.size,
        ):
            merged_face.insert(len(merged_face), face_2[i % face_2.size])
        if verbose > 2:
            print(f"Initial point ordering after attaching point loops {merged_face}")
        # Remove points to delete in the new face definition
        for point in points_to_delete:
            while point in merged_face:
                merged_face.remove(point)
            common_points.remove(point)
        if verbose > 2:
            print(f"Point ordering after removing points to delete {merged_face}")
        # At this stage the common point(s) to be kept have 2 occurences, we need to remove one
        for point in common_points:
            if point == anchor:
                merged_face.remove(point)
            else:
                merged_face.reverse()
                merged_face.remove(point)
                merged_face.reverse()
        if verbose > 1:
            print(f"Final point ordering after removing extra common point {merged_face}")

        return np.array(merged_face, dtype=self.__face_point_list.dtype)

    def reset_offsets(self) -> None:

        self.__face_offset_log = np.zeros(self.n_faces, dtype=self.__face_owner.dtype)
        self.__cell_offset_log = np.zeros(self.n_cells, dtype=self.__face_owner.dtype)
        self.__point_offset_log = np.zeros(self.n_points, dtype=self.__face_owner.dtype)

        self.__deleted_faces = []
        self.__deleted_cells = []
        self.__deleted_points = []
