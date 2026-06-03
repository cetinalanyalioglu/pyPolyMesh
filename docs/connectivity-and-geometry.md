# Connectivity and geometry

This guide covers building compact connectivity arrays and computing face metrics. It expands on the workflow previously shown only in the README.

## Cell–face list

Build a compact CSR-style cell–face structure from owner/neighbour arrays:

```python
from pypolymesh import PolyMesh
from pypolymesh.connectivity import build_cell_face_list

mesh = PolyMesh("path/to/case")

cell_face_indices, cell_face_list = build_cell_face_list(
    mesh.face_owner,
    mesh.face_neighbour,
    verbose=1,
    dtype=None,  # optional: request int32, uint32, etc.
)

# Faces of cell 1263 (0-based index)
cell_id = 1263
cell_faces = cell_face_list[cell_face_indices[cell_id] : cell_face_indices[cell_id + 1]]
```

The same face list for one cell is available as `mesh.cell_faces(1263)`, but that recomputes adjacency each time and is better for occasional queries.

## Face → points

Face definitions are stored in compact form at read time:

```python
face_id = 124
face_points = mesh.face_point_list[
    mesh.face_point_indices[face_id] : mesh.face_point_indices[face_id + 1]
]
# equivalent to:
face_points = mesh.face_points(124)
```

For a valid OpenFOAM mesh, face point order is counter-clockwise when viewed from the owner cell.

## Face areas and centroids

Uses the same approach as OpenFOAM’s face area/centroid calculation:

```python
from pypolymesh.geometry import compute_face_areas_and_centroids

face_centroids, face_area_vectors = compute_face_areas_and_centroids(
    mesh.points,
    mesh.face_point_indices,
    mesh.face_point_list,
)
```

## Cell–point list (unordered)

```python
from pypolymesh.connectivity import build_cell_point_list

cell_point_indices, cell_point_list = build_cell_point_list(
    mesh.face_point_indices,
    mesh.face_point_list,
    cell_face_indices,
    cell_face_list,
)

cell_id = 1263
cell_points = cell_point_list[cell_point_indices[cell_id] : cell_point_indices[cell_id + 1]]
```

Or for one cell: `mesh.cell_points(1263)`.

## Ordered cell–point list (VTK-style)

For standard element types (tetra, pyramid, prism, hex), you can build VTK-ordered point rings:

```python
from pypolymesh.connectivity import build_ordered_cell_point_list

cell_types, cell_point_indices, cell_point_list = build_ordered_cell_point_list(
    mesh.points,
    mesh.face_point_indices,
    mesh.face_point_list,
    face_centroids,
    face_area_vectors,
    cell_face_indices,
    cell_face_list,
)
```

Polyhedral cells are represented separately in the `cell_types` output; see `pypolymesh.elements` and the connectivity module docstrings.

## Output integer dtype

`build_cell_face_list` and `build_cell_point_list` accept an optional `dtype` for the returned index arrays. Values must fit in the requested integer type; otherwise a `ValueError` is raised.

## Related modules

| Module | Role |
|--------|------|
| `pypolymesh.connectivity` | Cell–face, cell–point, ordered cell–point builders |
| `pypolymesh.geometry` | Face areas, centroids |
| `pypolymesh.elements` | Cell type constants |
