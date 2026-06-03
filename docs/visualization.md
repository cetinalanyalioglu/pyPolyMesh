# Visualization (in development)

> **Status:** The `pypolymesh.visualization` package is under active development. APIs may change, notebook integration is still being refined, and the module is **not** recommended for production or general end-user plotting yet. Install the optional extra only if you are experimenting with the codebase.

```bash
pip install -e ".[plotting]"
```

Requires Plotly 6.x.

## Intended scope

The visualization layer provides **low-level primitives** for inspecting individual faces or small selections — not for rendering full meshes at scale. Design goals:

- Triangulate a single polygonal face for display
- Optional Plotly 3D figures with local axis scaling
- Debug cyclic faces, warped polygons, or mesh edits in notebooks

There is **no** `PolyMesh.plot_faces()`-style high-level API on the mesh class at this time.

## Modules

| Module | Role |
|--------|------|
| `pypolymesh.visualization.geometry` | Best-fit plane, 2D projection |
| `pypolymesh.visualization.triangulation` | Fan or ear-clip triangulation |
| `pypolymesh.visualization.plotly` | `plot_polygon_surface_plotly`, `plot_triangle_mesh_plotly` |

## Triangulation strategies

```python
from pypolymesh.visualization import triangulate_face

result = triangulate_face(points_3d, strategy="best_fit_projection")  # default
result = triangulate_face(points_3d, strategy="fan")
```

| Strategy | Behavior |
|----------|----------|
| `best_fit_projection` | Fit plane → project to 2D → ear clipping → map triangles back to 3D vertices |
| `fan` | Fan from first vertex in 3D order; fast, best for nearly planar convex faces |

## Plotly usage (experimental)

```python
from pypolymesh.visualization import plot_polygon_surface_plotly

fig = plot_polygon_surface_plotly(
    points_3d,
    show=False,           # prefer in notebooks to avoid double display
    show_triangulation=True,
    axis_mode="tight",
)
fig.show()
```

### Known notebook issues

- Returning a figure as the last cell expression **and** calling `show=True` can render twice; use `show=False` and display explicitly.
- Large numbers of traces (e.g. one trace per triangle) can freeze the notebook; the current implementation uses one `Mesh3d` per face.

## When to use something else

For production visualization of full cases, use ParaView, OpenFOAM native tools, or export mesh data to VTK and view externally until this module stabilizes.

## Related documentation

- [Getting started](getting-started.md)
- [Diagnostics](diagnostics.md) — often used together when debugging periodic patches
