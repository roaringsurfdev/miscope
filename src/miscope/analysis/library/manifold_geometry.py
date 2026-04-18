"""Quadratic surface fitting for intra-group manifold geometry.

Measures whether a neuron group's weight-space distribution lies on a
structured curved surface (saddle or bowl) rather than a flat blob.

The fit uses PC1/PC2 as predictors and PC3 as the response.  Two-stage
regression isolates curvature from tilt:

    Stage 1 (linear):    PC3 = d·PC1 + e·PC2 + f
    Stage 2 (quadratic): PC3 = a·PC1² + b·PC2² + c·PC1·PC2 + d·PC1 + e·PC2 + f

R²_curvature = R²_quadratic − R²_linear captures only the variance explained
by the quadratic terms, net of any planar tilt.  This is robust even when the
projections are not perfectly centered.
"""

from __future__ import annotations

import numpy as np

_FLAT_THRESHOLD = 0.05
_MIN_NEURONS = 7  # minimum to fit 6 quadratic parameters


def fit_quadratic_surface(proj_group: np.ndarray) -> dict[str, float | str]:
    """Fit a quadratic surface to a neuron group's 3D PCA projection.

    Args:
        proj_group: (n_members, 3) float — columns are PC1, PC2, PC3.
            Must have at least 7 rows for a meaningful quadratic fit.

    Returns:
        Dict with keys:
            r2_linear     float  — R² of the linear (planar tilt) fit
            r2_quadratic  float  — R² of the full quadratic fit
            r2_curvature  float  — r2_quadratic − r2_linear
            a             float  — PC1² coefficient
            b             float  — PC2² coefficient
            c             float  — PC1·PC2 coefficient
            shape         str    — "flat/blob", "bowl", or "saddle"
    """
    if proj_group.shape[0] < _MIN_NEURONS:
        return _nan_result()

    pc1, pc2, pc3 = proj_group[:, 0], proj_group[:, 1], proj_group[:, 2]

    r2_linear, _ = _fit_linear(pc1, pc2, pc3)
    r2_quadratic, (a, b, c) = _fit_quadratic(pc1, pc2, pc3)

    r2_curvature = float(np.clip(r2_quadratic - r2_linear, 0.0, 1.0))
    shape = _classify_shape(r2_curvature, a, b, c)

    return {
        "r2_linear": float(r2_linear),
        "r2_quadratic": float(r2_quadratic),
        "r2_curvature": r2_curvature,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "shape": shape,
    }


def _fit_linear(
    pc1: np.ndarray,
    pc2: np.ndarray,
    pc3: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit PC3 = d·PC1 + e·PC2 + f via least squares.

    Returns:
        (r2, coeffs) where coeffs is [d, e, f].
    """
    X = np.column_stack([pc1, pc2, np.ones_like(pc1)])
    coeffs, _, _, _ = np.linalg.lstsq(X, pc3, rcond=None)
    r2 = _compute_r2(pc3, X @ coeffs)
    return r2, coeffs


def _fit_quadratic(
    pc1: np.ndarray,
    pc2: np.ndarray,
    pc3: np.ndarray,
) -> tuple[float, tuple[float, float, float]]:
    """Fit PC3 = a·PC1² + b·PC2² + c·PC1·PC2 + d·PC1 + e·PC2 + f via least squares.

    Returns:
        (r2, (a, b, c)) — R² and the three quadratic coefficients.
    """
    X = np.column_stack([pc1**2, pc2**2, pc1 * pc2, pc1, pc2, np.ones_like(pc1)])
    coeffs, _, _, _ = np.linalg.lstsq(X, pc3, rcond=None)
    r2 = _compute_r2(pc3, X @ coeffs)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return r2, (a, b, c)


def _compute_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination for a regression fit."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _classify_shape(r2_curvature: float, a: float, b: float, c: float) -> str:
    """Classify manifold shape from curvature R² and quadratic coefficients.

    Uses the Hessian determinant (4ab − c²) to distinguish bowl from saddle.
    This is rotation-invariant: a saddle rotated away from the PC axes is
    correctly detected even when a and b share the same sign.
    """
    if r2_curvature < _FLAT_THRESHOLD:
        return "flat/blob"
    if 4 * a * b - c**2 > 0:
        return "bowl"
    return "saddle"


def _nan_result() -> dict[str, float | str]:
    """Return a NaN result for groups too small to fit."""
    return {
        "r2_linear": float("nan"),
        "r2_quadratic": float("nan"),
        "r2_curvature": float("nan"),
        "a": float("nan"),
        "b": float("nan"),
        "c": float("nan"),
        "shape": "flat/blob",
    }
