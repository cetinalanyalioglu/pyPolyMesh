import unittest

import numpy as np

from pypolymesh.visualization import earclip_triangulation_2d
from pypolymesh.visualization import fan_triangulation_indices
from pypolymesh.visualization import fit_best_plane
from pypolymesh.visualization import plot_polygon_surface_plotly
from pypolymesh.visualization import project_points_to_plane
from pypolymesh.visualization import triangulate_face

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised when the optional dependency is missing
    go = None


class VisualizationTests(unittest.TestCase):
    def test_fit_best_plane_handles_axis_aligned_face(self) -> None:

        points = np.array(
            [
                [0.17659999, 0.04824729, 0.24255548],
                [0.1766, 0.0485804, 0.24423018],
                [0.17659999, 0.04660113, 0.24317948],
                [0.17659999, 0.04706211, 0.24251163],
            ]
        )

        plane = fit_best_plane(points)

        self.assertAlmostEqual(abs(plane.normal[0]), 1.0, places=8)
        self.assertAlmostEqual(np.dot(plane.normal, plane.basis_u), 0.0, places=8)
        self.assertAlmostEqual(np.dot(plane.normal, plane.basis_v), 0.0, places=8)

    def test_projection_to_best_fit_plane_returns_2d_polygon(self) -> None:

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        projected = project_points_to_plane(points, fit_best_plane(points))

        self.assertEqual(projected.shape, (4, 2))
        self.assertGreater(np.ptp(projected[:, 0]), 0.0)
        self.assertGreater(np.ptp(projected[:, 1]), 0.0)

    def test_fan_triangulation_builds_triangle_fan(self) -> None:

        triangles = fan_triangulation_indices(5)

        np.testing.assert_array_equal(triangles, np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64))

    def test_earclip_triangulation_handles_concave_polygon(self) -> None:

        polygon = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [1.0, 0.25],
                [0.0, 1.0],
            ]
        )

        triangles = earclip_triangulation_2d(polygon)

        self.assertEqual(triangles.shape, (3, 3))
        self.assertEqual(np.unique(triangles).size, 5)

    def test_triangulate_face_uses_best_fit_projection_by_default(self) -> None:

        points = np.array(
            [
                [0.17659999, 0.04824729, 0.24255548],
                [0.1766, 0.0485804, 0.24423018],
                [0.17659999, 0.04660113, 0.24317948],
                [0.17659999, 0.04706211, 0.24251163],
            ]
        )

        triangulation = triangulate_face(points)

        self.assertEqual(triangulation.strategy, "best_fit_projection")
        self.assertEqual(triangulation.triangles.shape, (2, 3))
        self.assertEqual(triangulation.projected_points.shape, (4, 2))
        self.assertIsNotNone(triangulation.plane)

    def test_triangulate_face_accepts_fan_strategy(self) -> None:

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        triangulation = triangulate_face(points, strategy="fan")

        np.testing.assert_array_equal(triangulation.triangles, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
        self.assertIsNone(triangulation.projected_points)
        self.assertIsNone(triangulation.plane)

    @unittest.skipIf(go is None, "plotly is not installed")
    def test_plot_polygon_surface_plotly_returns_interactive_figure(self) -> None:

        points = np.array(
            [
                [0.17659999, 0.04824729, 0.24255548],
                [0.1766, 0.0485804, 0.24423018],
                [0.17659999, 0.04660113, 0.24317948],
                [0.17659999, 0.04706211, 0.24251163],
            ]
        )

        fig = plot_polygon_surface_plotly(points, show=False, show_edges=True, show_points=True, label="test face")

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(len(fig.data[0]["i"]), 2)
        self.assertEqual(fig.layout.scene.camera.projection.type, "orthographic")
        self.assertEqual(fig.layout.scene.dragmode, "turntable")

    @unittest.skipIf(go is None, "plotly is not installed")
    def test_plot_polygon_surface_plotly_can_overlay_triangulation(self) -> None:

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        fig = plot_polygon_surface_plotly(
            points,
            show=False,
            show_edges=True,
            show_triangulation=True,
            show_points=False,
        )

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[1].type, "scatter3d")
        self.assertEqual(fig.data[2].type, "scatter3d")


if __name__ == "__main__":
    unittest.main()
