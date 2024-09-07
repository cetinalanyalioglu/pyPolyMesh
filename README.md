# pyPolyMesh

This is a small library that can read, write and modify meshes in OpenFOAM ```polyMesh``` format.
Currently only documentation is within docstrings.

## Installation
1. Clone this repository
2. Run ```pip install .``` in the cloned repository folder

Alternatively, use ```pip install - e .``` for an interactive install if you are planning to work on the source code.
This way the changes are going to be reflected without a need to reinstall the package.

## Incomplete example
```python
from pypolymesh import PolyMesh
from pypolymesh.geometry import compute_face_areas_and_centroids
from pypolymesh.connectivity import build_cell_face_list, build_cell_point_list, build_ordered_cell_point_list

# Read an OpenFOAM mesh (ascii/binary, by default expects int32 and float64)
mesh = PolyMesh("path/to/case/folder")

# Build cell face list
cell_face_indices, cell_face_list = build_cell_face_list(mesh.face_owner, mesh.face_neighbour)

# Extract face indices of 1264'th cell
cell_faces = cell_face_list[cell_face_indices[1263] : cell_face_indices[1264]]

# Above information is also available using PolyMesh.cell_faces
# Instead of building the cell-face list, this will do a local evaluation (expensive for multiple requests)
cell_faces = mesh.cell_faces(1263)

# The face -> point connectivity is automatically built during read.
# Extract the ordered point sequence defining face 124
face_points = mesh.face_point_list[mesh.face_point_indices[124] : mesh.face_point_indices[125]]
# is identical to
face_points = mesh.face_points(124)
# For a valid mesh the point sequence follows a counter-clockwise orientation.

# Compute face areas and centroids (uses the exact same method in OpenFOAM)
face_centroids, face_area_vectors = compute_face_areas_and_centroids(mesh.points, mesh.face_point_indices, 
    mesh.face_point_list)

# Build non-ordered cell-point list
cell_point_indices, cell_point_list = build_cell_point_list(mesh.face_point_indices, mesh.face_point_list,
    cell_face_indices, cell_face_list)

# Extract points referered by cell 1263
cell_points = cell_point_list[cell_point_indices[1263] : cell_point_indices[1264]]
# Above information is also available using PolyMesh.cell_points
# Instead of building the cell-point list, this will do a local evaluation (expensive for multiple requests)
cell_points = mesh.cell_points(1263)

# Build the ordered cell-pont list
# The ordered cell point list follows the VTK convention for the point ordering of standard elements (non-polyhedra)
cell_types, cell_point_indices, cell_point_list = build_ordered_cell_point_list(mesh.points,
    mesh.face_point_indices, mesh.face_point_list, face_centroids, face_area_vectors, cell_face_indices,
    cell_face_list)

```