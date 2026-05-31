from numpy.typing import ArrayLike
from numpy.typing import NDArray
from typing import Any
from typing import List
from typing import Optional

import numpy as np

from .triangulation import TriangulatedFace
from .triangulation import triangulate_face


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            'Plotly is required for visualization primitives. Install it with `pip install plotly` or enable the '
            '"plotting" extra.'
        ) from exc

    return go


def _compute_axis_ranges(points: NDArray, axis_mode: str = "tight") -> List[List[float]]:

    if axis_mode not in {"tight", "equal"}:
        raise ValueError(f'"axis_mode" must be one of ["equal", "tight"], received "{axis_mode}".')

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    spans = maximum - minimum

    if axis_mode == "equal":
        span = float(np.max(spans))
        if span <= 0.0:
            span = 1.0
        half_width = 0.55 * span
        center = 0.5 * (minimum + maximum)
        return [[float(center[k] - half_width), float(center[k] + half_width)] for k in range(3)]

    global_span = float(np.max(spans))
    if global_span <= 0.0:
        global_span = 1.0

    ranges = []
    for axis in range(3):
        axis_span = float(spans[axis])
        center = 0.5 * float(minimum[axis] + maximum[axis])
        half_width = 0.55 * axis_span if axis_span > 0.0 else 0.05 * global_span
        ranges.append([center - half_width, center + half_width])

    return ranges


def _aspectratio_from_ranges(ranges: List[List[float]]) -> dict:

    widths = np.array([axis_range[1] - axis_range[0] for axis_range in ranges], dtype=np.float64)
    max_width = float(np.max(widths))
    if max_width <= 0.0:
        return {"x": 1.0, "y": 1.0, "z": 1.0}

    normalized = np.maximum(widths / max_width, 1.0e-3)
    return {"x": float(normalized[0]), "y": float(normalized[1]), "z": float(normalized[2])}


def _axis_layout(title: str, value_range: List[float], show_axes: bool) -> dict:

    if show_axes:
        return {"title": title, "range": value_range, "autorange": False}

    return {
        "title": "",
        "range": value_range,
        "autorange": False,
        "visible": False,
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
        "showticklabels": False,
    }


def _camera_for_view(view: str, projection: str) -> dict:

    if projection not in {"orthographic", "perspective"}:
        raise ValueError(f'"projection" must be one of ["orthographic", "perspective"], received "{projection}".')
    if view not in {"iso", "xy", "xz", "yz"}:
        raise ValueError(f'"view" must be one of ["iso", "xy", "xz", "yz"], received "{view}".')

    if view == "xy":
        eye = {"x": 0.0, "y": 0.0, "z": 2.5}
        up = {"x": 0.0, "y": 1.0, "z": 0.0}
    elif view == "xz":
        eye = {"x": 0.0, "y": -2.5, "z": 0.0}
        up = {"x": 0.0, "y": 0.0, "z": 1.0}
    elif view == "yz":
        eye = {"x": 2.5, "y": 0.0, "z": 0.0}
        up = {"x": 0.0, "y": 0.0, "z": 1.0}
    else:
        eye = {"x": 1.6, "y": 1.6, "z": 1.25}
        up = {"x": 0.0, "y": 0.0, "z": 1.0}

    return {
        "projection": {"type": projection},
        "eye": eye,
        "up": up,
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def _mesh_lighting() -> dict:

    return {
        "ambient": 0.85,
        "diffuse": 0.5,
        "specular": 0.05,
        "roughness": 1.0,
        "facenormalsepsilon": 0.0,
        "vertexnormalsepsilon": 0.0,
    }


def _build_triangulation_trace(
    go: Any,
    vertices: NDArray,
    triangles: NDArray,
    color: str,
    width: float,
):

    unique_edges = set()
    for a, b, c in triangles:
        unique_edges.add(tuple(sorted((int(a), int(b)))))
        unique_edges.add(tuple(sorted((int(b), int(c)))))
        unique_edges.add(tuple(sorted((int(c), int(a)))))

    x = []
    y = []
    z = []
    for start, end in sorted(unique_edges):
        x.extend([float(vertices[start, 0]), float(vertices[end, 0]), None])
        y.extend([float(vertices[start, 1]), float(vertices[end, 1]), None])
        z.extend([float(vertices[start, 2]), float(vertices[end, 2]), None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": color, "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def _hover_text(label: Optional[str], triangulation: TriangulatedFace) -> List[str]:

    base = (
        f"{label}<br>" if label else ""
    ) + f"n_points: {triangulation.points.shape[0]}<br>n_triangles: {triangulation.triangles.shape[0]}<br>strategy: {triangulation.strategy}"
    return [base] * triangulation.points.shape[0]


def plot_triangle_mesh_plotly(
    vertices: ArrayLike,
    triangles: ArrayLike,
    *,
    boundary_points: Optional[ArrayLike] = None,
    color: str = "#5DA5DA",
    opacity: float = 0.9,
    show_edges: bool = True,
    edge_color: str = "#202020",
    edge_width: float = 3.0,
    show_triangulation: bool = False,
    triangulation_edge_color: str = "#C44E52",
    triangulation_edge_width: float = 2.0,
    show_points: bool = False,
    point_color: str = "#202020",
    point_size: float = 4.0,
    title: Optional[str] = None,
    hover_label: Optional[str] = None,
    show_axes: bool = True,
    axis_mode: str = "tight",
    projection: str = "orthographic",
    dragmode: str = "turntable",
    view: str = "iso",
    show: bool = True,
):
    """Render a triangulated polygon surface with Plotly."""

    if dragmode not in {"turntable", "orbit"}:
        raise ValueError(f'"dragmode" must be one of ["turntable", "orbit"], received "{dragmode}".')

    go = _require_plotly()

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)

    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("Expected vertices with shape (n_vertices, 3).")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError("Expected triangles with shape (n_triangles, 3).")

    text = [hover_label] * vertex_array.shape[0] if hover_label is not None else None

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertex_array[:, 0],
            y=vertex_array[:, 1],
            z=vertex_array[:, 2],
            i=triangle_array[:, 0],
            j=triangle_array[:, 1],
            k=triangle_array[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
            text=text,
            hovertemplate="%{text}<extra></extra>" if text is not None else None,
            lighting=_mesh_lighting(),
            showlegend=False,
        )
    )

    edge_points = vertex_array if boundary_points is None else np.asarray(boundary_points, dtype=np.float64)
    if show_edges:
        closed = np.vstack((edge_points, edge_points[0, :]))
        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                line={"color": edge_color, "width": edge_width},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if show_triangulation:
        fig.add_trace(
            _build_triangulation_trace(
                go,
                vertex_array,
                triangle_array,
                triangulation_edge_color,
                triangulation_edge_width,
            )
        )

    if show_points:
        fig.add_trace(
            go.Scatter3d(
                x=edge_points[:, 0],
                y=edge_points[:, 1],
                z=edge_points[:, 2],
                mode="markers",
                marker={"size": point_size, "color": point_color},
                text=[f"point: {idx}" for idx in range(edge_points.shape[0])],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    ranges = _compute_axis_ranges(edge_points, axis_mode=axis_mode)
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "b": 0, "t": 40 if title else 0},
        scene={
            "aspectmode": "manual",
            "aspectratio": _aspectratio_from_ranges(ranges),
            "dragmode": dragmode,
            "camera": _camera_for_view(view, projection),
            "xaxis": _axis_layout("x", ranges[0], show_axes),
            "yaxis": _axis_layout("y", ranges[1], show_axes),
            "zaxis": _axis_layout("z", ranges[2], show_axes),
        },
    )

    if show:
        fig.show()

    return fig


def plot_polygon_surface_plotly(
    points: ArrayLike,
    *,
    triangulation_strategy: Optional[str] = None,
    color: str = "#5DA5DA",
    opacity: float = 0.9,
    show_edges: bool = True,
    edge_color: str = "#202020",
    edge_width: float = 3.0,
    show_triangulation: bool = False,
    triangulation_edge_color: str = "#C44E52",
    triangulation_edge_width: float = 2.0,
    show_points: bool = False,
    point_color: str = "#202020",
    point_size: float = 4.0,
    title: Optional[str] = None,
    label: Optional[str] = None,
    show_axes: bool = True,
    axis_mode: str = "tight",
    projection: str = "orthographic",
    dragmode: str = "turntable",
    view: str = "iso",
    show: bool = True,
):
    """Triangulate and render a single polygonal face with Plotly."""

    triangulation = triangulate_face(points, strategy=triangulation_strategy)
    return plot_triangle_mesh_plotly(
        triangulation.points,
        triangulation.triangles,
        boundary_points=triangulation.points,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        edge_width=edge_width,
        show_triangulation=show_triangulation,
        triangulation_edge_color=triangulation_edge_color,
        triangulation_edge_width=triangulation_edge_width,
        show_points=show_points,
        point_color=point_color,
        point_size=point_size,
        title=title,
        hover_label=_hover_text(label, triangulation)[0],
        show_axes=show_axes,
        axis_mode=axis_mode,
        projection=projection,
        dragmode=dragmode,
        view=view,
        show=show,
    )

