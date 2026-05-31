import io
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import transformations as tf

from pypolymesh.diagnostics import build_cyclic_transform
from pypolymesh.diagnostics import check_cyclic_pair
from pypolymesh.diagnostics import print_cyclic_report


class CyclicDiagnosticsTests(unittest.TestCase):
    def test_build_cyclic_transform_requires_rotation_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotation_axis is required"):
            build_cyclic_transform(rotation_angle=0.1)

    def test_build_cyclic_transform_rejects_zero_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "rotation_axis must be non-zero"):
            build_cyclic_transform(rotation_angle=0.1, rotation_axis=(0.0, 0.0, 0.0))

    def test_build_cyclic_transform_requires_at_least_one_operation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Provide at least one"):
            build_cyclic_transform()

    def test_check_cyclic_pair_passes_for_matching_translated_patches(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        mesh = MagicMock()
        mesh.boundary = {
            "source": {"nFaces": 1},
            "target": {"nFaces": 1},
        }
        mesh.boundary_points = lambda name: (
            np.array([0, 1, 2, 3], dtype=np.int64)
            if name == "source"
            else np.array([4, 5, 6, 7], dtype=np.int64)
        )
        mesh.points = points
        mesh.dtype_float = np.float64
        mesh.dtype_int = np.int64

        transform = build_cyclic_transform(translation=(0.0, 0.0, 1.0))
        report = check_cyclic_pair(
            mesh,
            "source",
            "target",
            transform,
            tolerance=1.0e-12,
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.failure_reasons, [])
        self.assertAlmostEqual(float(report.pair_distances.max()), 0.0, places=12)
        self.assertEqual(report.n_duplicate_target_matches, 0)

    def test_check_cyclic_pair_fails_without_tolerance(self) -> None:
        mesh = MagicMock()
        mesh.boundary = {
            "source": {"nFaces": 1},
            "target": {"nFaces": 1},
        }
        mesh.boundary_points = lambda name: np.array([0, 1, 2, 3], dtype=np.int64)
        mesh.points = np.eye(4, 3, dtype=np.float64)
        mesh.dtype_float = np.float64
        mesh.dtype_int = np.int64

        report = check_cyclic_pair(
            mesh,
            "source",
            "target",
            tf.translation_matrix((0.0, 0.0, 0.0)),
        )

        self.assertFalse(report.passed)
        self.assertIn("distance tolerance not specified", report.failure_reasons[0])

    def test_check_cyclic_pair_fails_on_face_count_mismatch(self) -> None:
        mesh = MagicMock()
        mesh.boundary = {
            "source": {"nFaces": 2},
            "target": {"nFaces": 1},
        }
        mesh.boundary_points = lambda name: np.array([0, 1, 2, 3], dtype=np.int64)
        mesh.points = np.eye(4, 3, dtype=np.float64)
        mesh.dtype_float = np.float64
        mesh.dtype_int = np.int64

        report = check_cyclic_pair(
            mesh,
            "source",
            "target",
            tf.translation_matrix((0.0, 0.0, 0.0)),
            tolerance=1.0,
        )

        self.assertFalse(report.passed)
        self.assertTrue(any("face count mismatch" in reason for reason in report.failure_reasons))

    def test_print_cyclic_report_shows_pass_banner(self) -> None:
        mesh = MagicMock()
        mesh.boundary = {
            "source": {"nFaces": 1},
            "target": {"nFaces": 1},
        }
        mesh.boundary_points = lambda name: np.array([0, 1, 2, 3], dtype=np.int64)
        mesh.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        mesh.dtype_float = np.float64
        mesh.dtype_int = np.int64

        report = check_cyclic_pair(
            mesh,
            "source",
            "target",
            build_cyclic_transform(translation=(0.0, 0.0, 0.0)),
            tolerance=1.0e-12,
        )

        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            print_cyclic_report(report)

        output = buffer.getvalue()
        self.assertIn("RESULT: PASS", output)
        self.assertNotIn("RESULT: FAIL", output)


if __name__ == "__main__":
    unittest.main()
