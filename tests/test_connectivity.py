import unittest

import numpy as np

from pypolymesh.connectivity import build_cell_face_list
from pypolymesh.connectivity import build_ordered_cell_point_list
from pypolymesh.connectivity import build_cell_point_list
from pypolymesh.elements import TETRAHEDRON
from pypolymesh.geometry import compute_face_areas_and_centroids


class ConnectivityTests(unittest.TestCase):
    def test_build_cell_face_list_uses_requested_dtype(self) -> None:
        face_owner = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        face_neighbour = np.array([1, -1, -1, -1, -1, -1, -1], dtype=np.int64)

        cell_face_indices, cell_face_list = build_cell_face_list(
            face_owner,
            face_neighbour,
            verbose=0,
            dtype=np.int32,
        )

        self.assertEqual(cell_face_indices.dtype, np.dtype(np.int32))
        self.assertEqual(cell_face_list.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(cell_face_indices, np.array([0, 4, 8], dtype=np.int32))
        np.testing.assert_array_equal(cell_face_list, np.array([0, 1, 2, 3, 0, 4, 5, 6], dtype=np.int32))

    def test_build_cell_face_list_allows_unsigned_output_dtype(self) -> None:
        face_owner = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        face_neighbour = np.array([1, -1, -1, -1, -1, -1, -1], dtype=np.int64)

        cell_face_indices, cell_face_list = build_cell_face_list(
            face_owner,
            face_neighbour,
            verbose=0,
            dtype=np.uint32,
        )

        self.assertEqual(cell_face_indices.dtype, np.dtype(np.uint32))
        self.assertEqual(cell_face_list.dtype, np.dtype(np.uint32))
        np.testing.assert_array_equal(cell_face_indices, np.array([0, 4, 8], dtype=np.uint32))
        np.testing.assert_array_equal(cell_face_list, np.array([0, 1, 2, 3, 0, 4, 5, 6], dtype=np.uint32))

    def test_build_cell_point_list_uses_requested_output_dtype(self) -> None:
        face_owner = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        face_neighbour = np.array([1, -1, -1, -1, -1, -1, -1], dtype=np.int64)
        face_point_indices = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=np.int64)
        face_point_list = np.array(
            [
                0,
                1,
                2,
                0,
                3,
                1,
                1,
                3,
                2,
                2,
                3,
                0,
                0,
                1,
                4,
                1,
                2,
                4,
                2,
                0,
                4,
            ],
            dtype=np.int64,
        )

        cell_face_indices, cell_face_list = build_cell_face_list(face_owner, face_neighbour, verbose=0)
        cell_point_indices, cell_point_list = build_cell_point_list(
            face_point_indices,
            face_point_list,
            cell_face_indices,
            cell_face_list,
            verbose=0,
            dtype=np.uint32,
        )

        self.assertEqual(cell_point_indices.dtype, np.dtype(np.uint32))
        self.assertEqual(cell_point_list.dtype, np.dtype(np.uint32))
        np.testing.assert_array_equal(cell_point_indices, np.array([0, 4, 8], dtype=np.uint32))
        np.testing.assert_array_equal(cell_point_list, np.array([0, 1, 2, 3, 0, 1, 2, 4], dtype=np.uint32))

    def test_build_cell_point_list_handles_single_tetrahedron(self) -> None:
        face_owner = np.array([0, 0, 0, 0], dtype=np.int64)
        face_neighbour = np.array([-1, -1, -1, -1], dtype=np.int64)
        face_point_indices = np.array([0, 3, 6, 9, 12], dtype=np.int64)
        face_point_list = np.array(
            [
                0,
                2,
                1,
                0,
                1,
                3,
                1,
                2,
                3,
                2,
                0,
                3,
            ],
            dtype=np.int64,
        )

        cell_face_indices, cell_face_list = build_cell_face_list(face_owner, face_neighbour, verbose=0)
        cell_point_indices, cell_point_list = build_cell_point_list(
            face_point_indices,
            face_point_list,
            cell_face_indices,
            cell_face_list,
            verbose=0,
            dtype=np.int32,
        )

        np.testing.assert_array_equal(cell_point_indices, np.array([0, 4], dtype=np.int32))
        np.testing.assert_array_equal(np.sort(cell_point_list), np.array([0, 1, 2, 3], dtype=np.int32))

    def test_build_ordered_cell_point_list_uses_requested_output_dtype(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        face_point_indices = np.array([0, 3, 6, 9, 12], dtype=np.int64)
        face_point_list = np.array(
            [
                0,
                2,
                1,
                0,
                1,
                3,
                1,
                2,
                3,
                2,
                0,
                3,
            ],
            dtype=np.int64,
        )
        face_owner = np.array([0, 0, 0, 0], dtype=np.int64)
        face_neighbour = np.array([-1, -1, -1, -1], dtype=np.int64)

        face_centroids, face_area_vectors = compute_face_areas_and_centroids(
            points,
            face_point_indices,
            face_point_list,
            verbose=0,
        )
        cell_face_indices, cell_face_list = build_cell_face_list(face_owner, face_neighbour, verbose=0)
        cell_types, cell_point_indices, cell_point_list = build_ordered_cell_point_list(
            points,
            face_point_indices,
            face_point_list,
            face_centroids,
            face_area_vectors,
            cell_face_indices,
            cell_face_list,
            verbose=0,
            dtype=np.uint8,
        )

        self.assertEqual(cell_types.dtype, np.dtype(np.uint8))
        self.assertEqual(cell_point_indices.dtype, np.dtype(np.uint8))
        self.assertEqual(cell_point_list.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(cell_types, np.array([TETRAHEDRON], dtype=np.uint8))
        np.testing.assert_array_equal(cell_point_indices, np.array([0, 4], dtype=np.uint8))
        np.testing.assert_array_equal(np.sort(cell_point_list), np.array([0, 1, 2, 3], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
