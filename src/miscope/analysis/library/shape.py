"""Shape characterization primitives.

Per REQ_106 / REQ_109 layering rule: characterizations are *measures*, not
*derivations*. Each function takes an already-projected ndarray (typically
a 2D PCA projection) and returns a typed result describing the shape. The
projection itself is the caller's responsibility, materialized once via the
canonical :mod:`miscope.analysis.library.pca` primitive — never re-derived
inside a measure.

Functions:
- characterize_circularity: how well projected points lie on a circle.
- characterize_fourier_alignment: angular ordering matches residue-class ordering.
"""

import numpy as np


def characterize_circularity(projection_2d: np.ndarray, var_explained: float) -> float:
    """Score how well 2D-projected points lie on a circle.

    Fits a circle via the algebraic Kåsa method, then weights the residual-
    based fit score by two factors:

    - ``var_explained`` (caller-supplied): how much of the underlying total
      variance the 2D projection captures. Discounts cases where the data
      is high-dimensional and the 2D projection is just an arbitrary slice.
    - *balance* (computed from the projection): how comparable the variances
      along the two PCs are. Discounts cases where the projection is
      essentially 1D (collinear) — for a true circle the two PCs have
      similar variance; for a line, the second is near zero.

    Score of 1.0 = perfect circle; 0.0 = no circular structure. Clamped to [0, 1].

    Args:
        projection_2d: ``(n, 2)`` data projected onto top-2 principal components.
        var_explained: Fraction of total variance captured by the 2D projection
            (typically ``sum(explained_variance_ratio[:2])``).

    Returns:
        Circularity score in ``[0, 1]``.
    """
    var_x = float(np.var(projection_2d[:, 0]))
    var_y = float(np.var(projection_2d[:, 1]))
    variance = var_x + var_y
    if variance < 1e-12 or var_x < 1e-12:
        return 0.0
    balance = var_y / var_x

    cx, cy, radius = _kasa_circle_fit(projection_2d)
    distances = np.sqrt((projection_2d[:, 0] - cx) ** 2 + (projection_2d[:, 1] - cy) ** 2)
    residuals = distances - radius
    msr = float(np.mean(residuals**2))
    raw_score = 1.0 - msr / variance
    score = raw_score * var_explained * balance
    return float(np.clip(score, 0.0, 1.0))


def characterize_fourier_alignment(projection_2d: np.ndarray, p: int) -> float:
    """Score whether angular ordering of 2D-projected points matches residues.

    Computes angles around the projection's fitted-circle center, then
    finds the frequency ``k`` that best explains the angular positions
    as ``theta_r = 2*pi*k*r/p``. Returns ``R^2`` of the best fit across
    all candidate frequencies in ``1..p-1``.

    Args:
        projection_2d: ``(p, 2)`` data projected onto top-2 principal
            components, ordered by class index.
        p: Number of classes (prime). Must equal ``projection_2d.shape[0]``.

    Returns:
        Fourier alignment ``R^2`` in ``[0, 1]``.
    """
    cx, cy, _ = _kasa_circle_fit(projection_2d)
    angles = np.arctan2(projection_2d[:, 1] - cy, projection_2d[:, 0] - cx)

    z_observed = np.exp(1j * angles)
    k_values = np.arange(1, p)
    residue_indices = np.arange(p)
    expected = 2 * np.pi * k_values[:, np.newaxis] * residue_indices[np.newaxis, :] / p
    z_expected = np.exp(1j * expected)
    correlations = np.abs(np.mean(z_observed[np.newaxis, :] * np.conj(z_expected), axis=1)) ** 2
    return float(np.max(correlations))


def _kasa_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D points via the algebraic Kåsa method.

    Solves the least-squares system for ``a, b, c`` in
    ``x^2 + y^2 + a*x + b*y + c = 0``.

    Args:
        points: ``(n, 2)`` array of 2D coordinates.

    Returns:
        Tuple ``(center_x, center_y, radius)``.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coeff, c = result
    cx = -a / 2
    cy = -b_coeff / 2
    radius_sq = cx**2 + cy**2 - c
    radius = np.sqrt(max(radius_sq, 0.0))
    return float(cx), float(cy), float(radius)
