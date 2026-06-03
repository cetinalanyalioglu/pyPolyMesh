import unittest

import numpy as np

from pypolymesh import PolyMesh


class PolyMeshScaleTests(unittest.TestCase):
    def test_scale_updates_points_component_wise(self) -> None:
        mesh = PolyMesh.__new__(PolyMesh)
        mesh._PolyMesh__dtype_float = np.dtype(np.float64)
        mesh._PolyMesh__points = np.array(
            [
                [1.0, 2.0, 3.0],
                [-4.0, 5.0, -6.0],
            ],
            dtype=np.float64,
        )
        mesh._PolyMesh__face_owner = np.array([0, 1], dtype=np.int32)
        mesh._PolyMesh__face_neighbour = np.array([-1, -1], dtype=np.int32)

        mesh.scale((2.0, 0.5, -1.0))

        np.testing.assert_allclose(
            mesh.points,
            np.array(
                [
                    [2.0, 1.0, -3.0],
                    [-8.0, 2.5, 6.0],
                ]
            ),
        )
        np.testing.assert_array_equal(mesh.face_owner, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(mesh.face_neighbour, np.array([-1, -1], dtype=np.int32))

    def test_scale_rejects_invalid_factor_shape(self) -> None:
        mesh = PolyMesh.__new__(PolyMesh)
        mesh._PolyMesh__dtype_float = np.dtype(np.float64)
        mesh._PolyMesh__points = np.zeros((1, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, 'shape \\(3,\\)'):
            mesh.scale((1.0, 2.0))

    def test_scale_rejects_non_finite_factors(self) -> None:
        mesh = PolyMesh.__new__(PolyMesh)
        mesh._PolyMesh__dtype_float = np.dtype(np.float64)
        mesh._PolyMesh__points = np.zeros((1, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "finite values"):
            mesh.scale((1.0, np.inf, 1.0))


if __name__ == "__main__":
    unittest.main()
