# Getting started

## Install

From the repository root:

```bash
pip install .
```

For local development:

```bash
pip install -e .
```

## Loading a mesh

`PolyMesh` expects a path to the **case** directory (the parent of `constant/polyMesh`), not the `polyMesh` folder itself:

```python
from pypolymesh import PolyMesh

mesh = PolyMesh("path/to/case", verbose=1)
```

On read, the library loads:

| File | Role |
|------|------|
| `constant/polyMesh/points` | Node coordinates `(n_points, 3)` |
| `constant/polyMesh/faces` | Face definitions (compact point lists) |
| `constant/polyMesh/owner` | Owner cell per face |
| `constant/polyMesh/neighbour` | Neighbour cell per face (`-1` on boundaries) |
| `constant/polyMesh/boundary` | Boundary patches (`startFace`, `nFaces`, `type`, …) |

Default dtypes are `float64` for coordinates and `int32` for topology. You can change them via `mesh.dtype_float` and `mesh.dtype_int` before reading if you construct the mesh differently; the reader uses the instance dtypes set at construction time.

## Basic inspection

```python
mesh.n_points
mesh.n_cells
mesh.n_faces
mesh.n_internal_faces
mesh.n_boundaries

mesh.boundary          # OrderedDict of patch name -> patch dict
mesh.points            # (n_points, 3)
mesh.face_owner
mesh.face_neighbour
mesh.face_point_indices
mesh.face_point_list
```

Per-entity helpers (single index or sequences):

```python
mesh.face_points(124)
mesh.cell_faces(1263)
mesh.cell_points(1263)
mesh.boundary_faces("wall")
mesh.boundary_points("wall")
```

For a **single** cell or face, these helpers evaluate connectivity on demand. For many cells, prefer building compact lists once (see [Connectivity and geometry](connectivity-and-geometry.md)).

## Writing a mesh

```python
mesh.write("path/to/output_case")
```

Keyword arguments (see `PolyMesh.write` docstring):

| Option | Default | Meaning |
|--------|---------|---------|
| `mode` | `"binary"` | `"binary"` or `"ascii"` |
| `ascii_float_precision` | `18` | ASCII float formatting |
| `verbose` | `1` | Log level while writing |
| `byteorder` | `"little"` | Binary byte order |

The writer creates `constant/polyMesh/` under the given path.

## Scaling coordinates

`PolyMesh.scale` multiplies coordinates component-wise in place. Topology (face lists, owner, neighbour, boundary face ranges) is index-based and **does not** need separate scaling:

```python
mesh.scale((2.0, 1.0, 1.0))  # stretch x only
```

## Verbosity

```python
mesh.verbose = 0   # quiet read/write
mesh.verbose = 1   # brief messages (default)
```

## Next steps

- [Connectivity and geometry](connectivity-and-geometry.md) — bulk connectivity and face metrics
- [Mesh modification](mesh-modification.md) — merging, boundaries, index logs
- [Diagnostics](diagnostics.md) — cyclic patch pair checks
- [Visualization](visualization.md) — experimental Plotly helpers
