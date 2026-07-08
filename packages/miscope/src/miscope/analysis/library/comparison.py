"""Trajectory comparison primitives.

Per REQ_106 / REQ_109 layering rule: comparison primitives take pre-materialized
trajectories (typically 2D/3D PCA projections) and return typed results
describing how the trajectories relate. Coordinate-frame materialization is
the caller's responsibility — never re-derived inside a comparison.

Procrustes alignment:
- procrustes_align: pairwise alignment via translation + uniform scale +
  rotation/reflection; reports residual disparity and the standardized arrays.
- compute_procrustes_disparity_matrix: pairwise N×N disparity matrix over a
  collection of trajectories.
"""

from typing import NamedTuple

import numpy as np
from scipy.spatial import procrustes as _scipy_procrustes


class ProcrustesResult(NamedTuple):
    """Procrustes alignment of two trajectories.

    scipy.spatial.procrustes standardizes both inputs to zero mean and unit
    Frobenius norm, then finds the orthogonal transformation (rotation +
    reflection) of ``b`` that minimizes the sum of squared point distances
    against ``a``. ``standardized_a`` is centered + unit-Frobenius only;
    ``aligned_b`` is centered + unit-Frobenius + rotated.

    Fields:
        disparity: Sum of squared element-wise differences between
            ``standardized_a`` and ``aligned_b``. Bounded in [0, 1] for the
            scipy convention (both arrays unit-norm). 0 = identical shape up
            to translation, scale, rotation, reflection.
        standardized_a: ``a`` after centering and unit-Frobenius scaling, no
            rotation. Shape matches input.
        aligned_b: ``b`` after centering, unit-Frobenius scaling, and the
            rotation/reflection that minimizes residual against
            ``standardized_a``. Shape matches input.
        n_points: Number of rows in the inputs.
        n_features: Number of columns in the inputs.
    """

    disparity: float
    standardized_a: np.ndarray
    aligned_b: np.ndarray
    n_points: int
    n_features: int


def procrustes_align(a: np.ndarray, b: np.ndarray) -> ProcrustesResult:
    """Align two trajectories via Procrustes.

    Removes translation (centering), uniform scale (unit Frobenius norm),
    and rotation/reflection (orthogonal alignment of ``b`` to ``a``). Wraps
    ``scipy.spatial.procrustes`` for bit-exact compatibility with existing
    notebook-level uses.

    Convention: rows are points (e.g. epochs along a trajectory), columns
    are features (e.g. PC dimensions). Both arrays must share the same
    shape.

    Args:
        a: First trajectory, shape (n_points, n_features).
        b: Second trajectory, shape (n_points, n_features).

    Returns:
        ProcrustesResult with disparity scalar and standardized arrays.

    Raises:
        ValueError: If shapes mismatch, ndim != 2, or scipy rejects the
            input (e.g. fewer than 2 unique points).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError(
            f"procrustes_align requires 2D arrays; got a.ndim={a_arr.ndim}, b.ndim={b_arr.ndim}"
        )
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"procrustes_align requires matching shapes; got a.shape={a_arr.shape}, "
            f"b.shape={b_arr.shape}"
        )

    standardized_a, aligned_b, disparity = _scipy_procrustes(a_arr, b_arr)
    n_points, n_features = a_arr.shape
    return ProcrustesResult(
        disparity=float(disparity),
        standardized_a=standardized_a,
        aligned_b=aligned_b,
        n_points=n_points,
        n_features=n_features,
    )


def compute_procrustes_disparity_matrix(
    trajectories: list[np.ndarray],
) -> np.ndarray:
    """Pairwise Procrustes disparity matrix over a collection of trajectories.

    Computes the upper triangle and mirrors. Diagonal is zero by construction.
    Pairs that fail to align (e.g. degenerate input) are written as NaN
    instead of raising, so a single bad trajectory does not poison the whole
    matrix.

    Args:
        trajectories: List of arrays, each shape (n_points, n_features). All
            must share the same shape (procrustes itself enforces this per
            pair).

    Returns:
        Symmetric (n, n) array of disparities. ``D[i, i] = 0``.
    """
    n = len(trajectories)
    disparity_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                result = procrustes_align(trajectories[i], trajectories[j])
                disparity = result.disparity
            except ValueError:
                disparity = float("nan")
            disparity_matrix[i, j] = disparity
            disparity_matrix[j, i] = disparity
    return disparity_matrix
