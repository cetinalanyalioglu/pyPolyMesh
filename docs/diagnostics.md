# Diagnostics

The `pypolymesh.diagnostics` package provides mesh consistency checks that are separate from core I/O.

## Cyclic / periodic patch pairs

Module: `pypolymesh.diagnostics`.

OpenFOAM stores cyclic metadata in the `boundary` file (`neighbourPatch`, `transform`, …). **This library does not parse those entries automatically.** You supply:

1. Source and target patch **names**
2. A 4×4 affine matrix mapping the source patch onto the target
3. A distance **tolerance** (required for a pass/fail result)

### Build a transform

Using the `transformations` package:

```python
from pypolymesh.diagnostics import build_cyclic_transform

matrix = build_cyclic_transform(
    rotation_angle=0.392699,  # radians
    rotation_axis=(0.0, 0.0, 1.0),
    rotation_origin=(0.0, 0.0, 0.0),
)
```

Or pass any `(4, 4)` matrix from `transformations` directly.

### Run the check

```python
from pypolymesh import PolyMesh
from pypolymesh.diagnostics import check_cyclic_pair
from pypolymesh.diagnostics import print_cyclic_report

mesh = PolyMesh("path/to/case")

report = check_cyclic_pair(
    mesh,
    "left",
    "left_shadow",
    matrix,
    tolerance=1e-4,
)

print_cyclic_report(report)
print(report.passed)
```

### What the report contains

| Field / output | Meaning |
|----------------|---------|
| Face counts | Should match for a valid pair |
| Point counts | Unique nodes on each patch; usually should match |
| Pair distances | After transforming source points, distance to nearest target point |
| Duplicate target matches | More than one source point maps to the same target — likely invalid |
| Tolerance exceedance | Count of source points farther than `tolerance` from their match |
| `RESULT: PASS` / `FAIL` | Overall banner with failure reasons |

Pass criteria (all required):

- Equal face counts on both patches
- Equal point counts on both patches
- No duplicate target matches
- Every pair distance ≤ `tolerance`

If `tolerance` is omitted, the geometric check is skipped and the report **fails** with an explicit reason.

### Notes

- Wrong rotation axis, angle, or origin will show up as large pair distances.
- Slightly mismatched face counts (e.g. off-by-one after mesh editing) are reported as topology failures.
- This is a **geometric** consistency tool, not a full replacement for `checkMesh`.
