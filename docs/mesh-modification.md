# Mesh modification

`PolyMesh` supports in-place topology and geometry edits: merging cells, moving boundary faces, deleting entities, and scaling coordinates. After edits, use `mesh.write` to export an updated `polyMesh`.

## Current vs original indices

Modification routines maintain **offset logs** for faces, cells, and points. Many methods accept `indexing="current"` (default) or `indexing="original"`:

| Mode | Meaning |
|------|---------|
| `current` | Index in the mesh **as it exists now**, after prior edits |
| `original` | Index from the mesh **at load time**, before edits |

Helpers:

```python
mesh.face_current_index(face)
mesh.face_original_index(face)
mesh.cell_current_index(cell)
mesh.cell_original_index(cell)
```

Deleted entities are recorded internally; referring to a deleted index raises an error.

Reset bookkeeping after a batch of edits if your workflow requires it:

```python
mesh.reset_offsets()
```

## Scaling (geometry only)

`scale` updates node coordinates in place:

```python
mesh.scale((sx, sy, sz))
```

Face/cell connectivity and boundary `startFace` / `nFaces` are unchanged because they store **indices**, not coordinates.

## Merging cells

Merge two adjacent cells (deletes the shared internal face):

```python
mesh.merge_cells((cell_a, cell_b), indexing="current")
```

Options include `merge_internal_faces` and `merge_boundary_faces`; see the `merge_cells` docstring.

## Boundaries

Common operations:

```python
mesh.boundary_create("newPatch", type="patch")
mesh.boundary_add_face("newPatch", face_ids, indexing="current")
mesh.boundary_delete_empty()
mesh.boundary_faces("wall")
mesh.boundary_points("wall")
```

`boundary_add_face` moves existing boundary faces onto a patch (and may delete/re-add faces internally).

## Aligning boundary nodes (cyclic-style)

`boundary_align_nodes` maps points on a source patch onto a target patch using a 4×4 affine matrix (see `transformations` for building matrices). Useful after mesh edits or for manual cyclic repair:

```python
import transformations as tf

matrix = tf.rotation_matrix(angle, axis, point=origin)
mesh.boundary_align_nodes(
    "left",
    "left_shadow",
    matrix,
    maximum_displacement=1e-4,
    dry_run=False,
)
```

## Internal face deletion

```python
mesh.delete_internal_face(face_index)
```

## Lower-level routines

Methods prefixed with `_` (`_del_face`, `_merge_faces`, `_replace_face`, …) are intended for internal use. Prefer the public workflows above unless you are extending the library.

## Writing after edits

```python
mesh.write("path/to/modified_case", mode="binary", verbose=1)
```

Always validate modified meshes in OpenFOAM or with your own checks (e.g. [cyclic diagnostics](diagnostics.md)) before running solvers.
