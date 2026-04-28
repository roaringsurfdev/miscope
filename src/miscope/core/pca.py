"""Canonical PCA result type.

All PCA primitives in miscope return :class:`PCAResult`. Fields cover the
full spectrum (singular values, eigenvalues), the basis, the projections,
and derived metrics (participation ratio, rank, spread). Consumers read
fields directly rather than re-deriving from singular values.

The basis is arbitrary up to per-component sign flip (a property of SVD).
Consumers handle sign flips downstream and should not assume the SVD's
choice is canonical.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    """Result of a PCA computation.

    Attributes:
        singular_values: (k,) singular values from SVD, descending order.
        eigenvalues: (k,) sample-covariance eigenvalues = ``singular_values**2 / (n-1)``.
        basis_vectors: (k, d) principal components in row form.
        projections: (n, k) data projected onto the basis.
        explained_variance: (k,) sklearn-compatible alias for ``eigenvalues``.
        explained_variance_ratio: (k,) ``eigenvalues / sum(eigenvalues)``.
        participation_ratio: ``(sum λ)^2 / sum(λ^2)`` — effective rank.
        rank: count of singular values above numerical tolerance.
        spread: RMS spread = ``sqrt(sum(eigenvalues))``.
        center: (d,) mean subtracted from input before SVD.
    """

    singular_values: np.ndarray
    eigenvalues: np.ndarray
    basis_vectors: np.ndarray
    projections: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    participation_ratio: float
    rank: int
    spread: float
    center: np.ndarray
