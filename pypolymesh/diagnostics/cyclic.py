"""Diagnostics for manually specified cyclic / periodic boundary patch pairs.

The routines in this module check whether two named boundary patches are
geometrically consistent under a user-supplied affine transform. Patch names and
the transform matrix must be provided explicitly; OpenFOAM boundary dictionaries
are not parsed for cyclic metadata.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import transformations as tf
from numpy.typing import NDArray

from pypolymesh.polymesh import PolyMesh

Vector3 = Tuple[float, float, float]

__all__ = [
    "CyclicPairReport",
    "build_cyclic_transform",
    "check_cyclic_pair",
    "print_cyclic_report",
]


def build_cyclic_transform(
    *,
    rotation_angle: Optional[float] = None,
    rotation_axis: Optional[Vector3] = None,
    rotation_origin: Vector3 = (0.0, 0.0, 0.0),
    translation: Optional[Vector3] = None,
) -> NDArray:
    """Build a 4x4 affine matrix for a cyclic pair using ``transformations``.

    Supply either a rotation (``rotation_angle`` + ``rotation_axis``) or a
    ``translation`` vector, or both. Matrices are composed in the same spirit
    as OpenFOAM cyclic patches: rotation about ``rotation_origin``, then
    translation.

    Parameters
    ----------
    rotation_angle : float, optional
        Rotation angle in radians.
    rotation_axis : tuple of float, optional
        Axis direction for the rotation. Required when ``rotation_angle`` is
        given.
    rotation_origin : tuple of float, optional
        Point about which the rotation is applied.
    translation : tuple of float, optional
        Translation vector applied after rotation.

    Returns
    -------
    NDArray
        Affine transformation matrix with shape ``(4, 4)``.
    """

    matrices = []

    if rotation_angle is not None:
        if rotation_axis is None:
            raise ValueError("rotation_axis is required when rotation_angle is given")
        axis = np.asarray(rotation_axis, dtype=np.float64)
        if np.linalg.norm(axis) == 0.0:
            raise ValueError("rotation_axis must be non-zero")
        matrices.append(tf.rotation_matrix(rotation_angle, axis, point=rotation_origin))

    if translation is not None:
        matrices.append(tf.translation_matrix(translation))

    if not matrices:
        raise ValueError("Provide at least one of rotation_angle or translation")

    return tf.concatenate_matrices(*matrices)


@dataclass
class CyclicPairReport:
    """Result of checking one cyclic / periodic boundary patch pair.

    Attributes
    ----------
    source_boundary, target_boundary : str
        Names of the patch pair supplied to :func:`check_cyclic_pair`. The
        transform is assumed to map coordinates on ``source_boundary`` onto the
        coordinate frame of ``target_boundary``.
    transform_matrix : NDArray
        The 4x4 affine matrix used for the check.
    n_faces_source, n_faces_target : int
        Number of faces on each patch. For a consistent cyclic pair these counts
        should match.
    n_points_source, n_points_target : int
        Number of unique mesh points referenced by the faces on each patch.
        Matching cyclic pairs normally use the same count on both sides.
    source_point_indices, target_point_indices : NDArray
        Mesh point indices belonging to each patch. These arrays define the
        ordering used for the distance statistics below.
    pair_distances : NDArray
        For each source-boundary point, the Euclidean distance to the nearest
        point on the target boundary after applying ``transform_matrix``. Small
        values indicate good geometric agreement; large values usually mean the
        wrong transform, a broken mesh, or a non-cyclic patch pairing.
    pair_target_indices : NDArray
        For each source-boundary point, the index into ``target_point_indices``
        of the nearest target point. When several source points map to the same
        target point, the pairing is ambiguous and the cyclic link is likely
        invalid.
    tolerance : float, optional
        Distance threshold supplied to :func:`check_cyclic_pair`. When set,
        points with ``pair_distances`` greater than this value are counted as
        failures.
    failure_reasons : list of str
        Human-readable reasons why :attr:`passed` is ``False``. Empty when the
        check passes.
    passed : bool
        Overall pass/fail flag. See :func:`check_cyclic_pair` for the criteria.
    """

    source_boundary: str
    target_boundary: str
    transform_matrix: NDArray
    n_faces_source: int
    n_faces_target: int
    n_points_source: int
    n_points_target: int
    source_point_indices: NDArray
    target_point_indices: NDArray
    pair_distances: NDArray
    pair_target_indices: NDArray
    tolerance: Optional[float] = None
    failure_reasons: List[str] = field(default_factory=list)
    passed: bool = False

    @property
    def n_exceeding_tolerance(self) -> Optional[int]:
        """Number of source points whose pair distance exceeds ``tolerance``."""

        if self.tolerance is None:
            return None
        return int(np.count_nonzero(self.pair_distances > self.tolerance))

    @property
    def n_duplicate_target_matches(self) -> int:
        """Number of target points matched by more than one source point."""

        _, counts = np.unique(self.pair_target_indices, return_counts=True)
        return int(np.count_nonzero(counts > 1))

    @property
    def geometry_checked(self) -> bool:
        """Whether a distance tolerance was supplied for the geometric check."""

        return self.tolerance is not None


def _evaluate_cyclic_pair(report: CyclicPairReport) -> Tuple[bool, List[str]]:
    """Return pass/fail status and a list of failure reasons."""

    reasons: List[str] = []

    if report.n_faces_source != report.n_faces_target:
        reasons.append(
            "face count mismatch "
            f"({report.n_faces_source} on {report.source_boundary}, "
            f"{report.n_faces_target} on {report.target_boundary})"
        )

    if report.n_points_source != report.n_points_target:
        reasons.append(
            "point count mismatch "
            f"({report.n_points_source} on {report.source_boundary}, "
            f"{report.n_points_target} on {report.target_boundary})"
        )

    if report.n_duplicate_target_matches > 0:
        reasons.append(
            f"{report.n_duplicate_target_matches} target points matched by multiple source points"
        )

    if report.tolerance is None:
        reasons.append("distance tolerance not specified; geometric check skipped")
    elif report.n_exceeding_tolerance:
        reasons.append(
            f"{report.n_exceeding_tolerance}/{report.pair_distances.size} source points "
            f"exceed tolerance ({report.tolerance:.6e})"
        )

    return len(reasons) == 0, reasons


def check_cyclic_pair(
    mesh: PolyMesh,
    source_boundary: str,
    target_boundary: str,
    transform_matrix: NDArray,
    *,
    tolerance: Optional[float] = None,
) -> CyclicPairReport:
    """Check geometric consistency of a manually specified cyclic patch pair.

    Each unique point on ``source_boundary`` is transformed with
    ``transform_matrix`` and paired with the nearest point on
    ``target_boundary``. The returned :class:`CyclicPairReport` summarizes
    topology, pairing quality, and distance statistics.

    Pass / fail criteria
    --------------------
    The report passes when all of the following hold:

    * face counts on the two patches are equal
    * unique point counts on the two patches are equal
    * no target point is the nearest match for more than one source point
    * ``tolerance`` is provided and every ``pair_distances`` value is less than
      or equal to ``tolerance``

    If ``tolerance`` is omitted, the geometric distance check is skipped and the
    report fails with reason ``"distance tolerance not specified"``.

    Parameters
    ----------
    mesh : PolyMesh
        Mesh containing both boundary patches.
    source_boundary : str
        Name of the patch whose points are transformed.
    target_boundary : str
        Name of the partner patch used as the matching target.
    transform_matrix : NDArray
        4x4 affine matrix mapping ``source_boundary`` onto ``target_boundary``.
    tolerance : float, optional
        Maximum allowed pair distance. Required for an overall pass.

    Returns
    -------
    CyclicPairReport
        Structured report including ``passed`` and ``failure_reasons``.
    """

    transform_matrix = np.asarray(transform_matrix, dtype=np.float64)
    if transform_matrix.shape != (4, 4):
        raise ValueError("transform_matrix must have shape (4, 4)")

    if source_boundary not in mesh.boundary:
        raise KeyError(f"Unknown boundary: {source_boundary}")
    if target_boundary not in mesh.boundary:
        raise KeyError(f"Unknown boundary: {target_boundary}")

    idx_source = mesh.boundary_points(source_boundary)
    idx_target = mesh.boundary_points(target_boundary)

    points_source = mesh.points[idx_source]
    points_target = mesh.points[idx_target]

    points_source_transformed = PolyMesh.apply_transformation(points_source, transform_matrix)
    pair_distances, pair_target_indices = PolyMesh.find_closest_points(
        points_source_transformed,
        points_target,
    )

    report = CyclicPairReport(
        source_boundary=source_boundary,
        target_boundary=target_boundary,
        transform_matrix=transform_matrix,
        n_faces_source=int(mesh.boundary[source_boundary]["nFaces"]),
        n_faces_target=int(mesh.boundary[target_boundary]["nFaces"]),
        n_points_source=int(idx_source.size),
        n_points_target=int(idx_target.size),
        source_point_indices=idx_source,
        target_point_indices=idx_target,
        pair_distances=pair_distances.astype(mesh.dtype_float),
        pair_target_indices=pair_target_indices.astype(mesh.dtype_int),
        tolerance=tolerance,
    )
    report.passed, report.failure_reasons = _evaluate_cyclic_pair(report)
    return report


def print_cyclic_report(report: CyclicPairReport) -> None:
    """Print a human-readable summary of a :class:`CyclicPairReport`.

    The printed fields correspond to the attributes documented on
    :class:`CyclicPairReport`. A prominent pass/fail banner is printed at the
    end.
    """

    print(f"Source patch : {report.source_boundary}")
    print(f"Target patch : {report.target_boundary}")
    print(f"Faces        : {report.n_faces_source} (source) vs {report.n_faces_target} (target)")
    print(f"Points       : {report.n_points_source} (source) vs {report.n_points_target} (target)")
    print(
        "Pair distance: "
        f"min={report.pair_distances.min():.6e}  "
        f"max={report.pair_distances.max():.6e}  "
        f"mean={report.pair_distances.mean():.6e}  "
        f"p95={np.percentile(report.pair_distances, 95):.6e}"
    )
    print(f"Duplicate target matches: {report.n_duplicate_target_matches}")
    if report.tolerance is not None:
        print(
            f"Exceeding tolerance ({report.tolerance:.6e}): "
            f"{report.n_exceeding_tolerance}/{report.pair_distances.size}"
        )
    else:
        print("Exceeding tolerance: not evaluated (tolerance not specified)")

    print("=" * 50)
    if report.passed:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for reason in report.failure_reasons:
            print(f"  - {reason}")
    print("=" * 50)
