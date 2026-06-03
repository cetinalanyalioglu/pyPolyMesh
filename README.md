# pyPolyMesh

A Python library to read, write, and modify OpenFOAM `polyMesh` meshes in memory. It focuses on mesh topology and geometry: boundary inspection, connectivity builders, in-place mesh edits, scaling, and diagnostics for manually specified cyclic patch pairs.

Detailed guides live under [`docs/`](docs/).

## Features

- Read and write ASCII or binary `constant/polyMesh` (points, faces, owner, neighbour, boundary)
- Query mesh topology via `PolyMesh` (cells, faces, boundaries, adjacency)
- Build compact cell–face and cell–point connectivity arrays
- Compute geometrial properties (face areas and centroids)
- Modify meshes in place (merge cells, boundary edits, point scaling, and more)
- Diagnostics (currently only checking of cyclic/periodic patch pairs)([diagnostics](docs/diagnostics.md))

## Requirements

- Python 3.8+
- NumPy, Numba
- `transformations` (cyclic diagnostics)

## Installation

Clone the repository and install from the project root:

```bash
pip install .
```

Editable install while developing:

```bash
pip install -e .
```
## Quick start

Point `PolyMesh` at a case directory (the folder that contains `constant/polyMesh`):

```python
from pypolymesh import PolyMesh

# Can read and auto-detect both binary and ascii, by default expects int32 and float64
mesh = PolyMesh("path/to/case")

# Display some information - many intuitive methods under PolyMeesh class
print(mesh.n_points, mesh.n_cells, mesh.n_faces, mesh.n_boundaries)
print(list(mesh.boundary.keys())[:5])

# Inspect one boundary patch
patch = "inlet"
print(mesh.boundary[patch])
face_ids = mesh.boundary_faces(patch)
print("first face points:", mesh.face_points(face_ids[0]))

# Scale coordinates (topology arrays are unchanged)
mesh.scale((0.001, 0.001, 0.001))  # mm -> m, for example

# Write mesh back under a new case path in ASCII Format
mesh.write("path/to/output_case", mode="ascii")
```

See [Getting started](docs/getting-started.md) for dtypes, verbosity, and write options.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting started](docs/getting-started.md) | Install, load/save, core properties |
| [Connectivity and geometry](docs/connectivity-and-geometry.md) | Cell–face/point lists, areas, ordered VTK-style points |
| [Mesh modification](docs/mesh-modification.md) | Edits, scaling, current vs original indices |
| [Diagnostics](docs/diagnostics.md) | Cyclic / periodic patch pair checks |
| [Visualization](docs/visualization.md) | Plotly primitives (**in development**) |

API details are also available in module and class docstrings.

## Development

Run tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Status and limitations
- **Mesh modification features are experimental**
- The `pypolymesh.visualization` package is under active development and is not yet a supported end-user plotting API.
