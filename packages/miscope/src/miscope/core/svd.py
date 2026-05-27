"""Canonical SVD result type.

Companion to :class:`miscope.core.pca.PCAResult` for **raw** (non-centered)
singular value decompositions of arbitrary matrices — e.g. weight matrices
where the matrix itself is the object of study, not a sample distribution.

The left/right singular vectors are arbitrary up to per-component sign flip
(a property of SVD). Consumers handle sign flips downstream and should not
assume the SVD's choice is canonical.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SVDResult:
    """Result of a non-centered SVD computation.

    Attributes:
        singular_values: (k,) singular values, descending order.
        left_vectors: (m, k) left singular vectors as columns (``U``).
        right_vectors: (k, n) right singular vectors as rows (``Vt``).
        rank: count of singular values above numerical tolerance.
    """

    singular_values: np.ndarray
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    rank: int
