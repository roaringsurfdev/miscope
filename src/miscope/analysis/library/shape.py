"""Shape characterization primitives.

Per REQ_106 / REQ_109 layering rule: characterizations are *measures*, not
*derivations*. Each function takes an already-projected ndarray (typically
a 2D PCA projection or a curve in some coordinate system) and returns a
typed result describing the shape. Projection / coordinate-frame
materialization is the caller's responsibility — never re-derived inside
a measure.

Centroid-shape characterizations (caller passes a 2D PCA projection):
- characterize_circularity: how well projected points lie on a circle.
- characterize_fourier_alignment: angular ordering matches residue-class ordering.

Curve-shape characterizations (caller passes a 2D/3D curve in coordinates):
- compute_arc_length: cumulative arc-length along a curve.
- detect_self_intersection: locate the lemniscate node — the closest self-approach.
- compute_signed_loop_area: signed area enclosed by a loop segment.
- compute_curvature_profile: signed curvature κ(s) along a 2D curve.
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


# ---------------------------------------------------------------------------
# Curve-shape characterizations (lemniscate analysis)
# ---------------------------------------------------------------------------


def compute_arc_length(curve: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a 2D or 3D curve.

    Args:
        curve: ``(n_points, d)`` array of coordinates.

    Returns:
        ``(n_points,)`` cumulative arc-length, starting at 0.
    """
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def detect_self_intersection(
    curve: np.ndarray,
    min_arc_sep_fraction: float = 0.15,
) -> dict:
    """Find the closest self-approach in a 2D curve — the lemniscate node.

    Scans all pairs of non-adjacent points separated by at least
    ``min_arc_sep_fraction * total_arc_length`` of arc, returning the pair
    with minimum Euclidean distance. This localizes the node where the curve
    crosses itself (or approaches itself most closely).

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        min_arc_sep_fraction: Minimum arc-length gap between candidate pairs
            as a fraction of total arc length. Filters out adjacent segments.
            Default 0.15 (15% of total arc).

    Returns:
        Dict with:
            ``node_position``: ``(2,)`` midpoint of the closest-approach pair.
            ``idx_pair``: ``(i, j)`` index pair of the closest approach.
            ``min_distance``: float — Euclidean distance at closest approach.
            ``arc_length``: ``(n_points,)`` cumulative arc-length array.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]
    min_sep = min_arc_sep_fraction * total_arc

    n = len(curve)
    diff = curve[:, np.newaxis, :] - curve[np.newaxis, :, :]  # (n, n, 2)
    dist_sq = np.sum(diff**2, axis=2)  # (n, n)

    arc_sep = np.abs(arc[:, np.newaxis] - arc[np.newaxis, :])  # (n, n)
    valid = (arc_sep >= min_sep) & np.triu(np.ones((n, n), dtype=bool), k=1)

    if not valid.any():
        mid = n // 2
        return {
            "node_position": curve[mid].copy(),
            "idx_pair": (0, n - 1),
            "min_distance": float("inf"),
            "arc_length": arc,
        }

    dist_sq_masked = np.where(valid, dist_sq, np.inf)
    flat_idx = int(np.argmin(dist_sq_masked))
    i, j = divmod(flat_idx, n)

    return {
        "node_position": (curve[i] + curve[j]) / 2.0,
        "idx_pair": (int(i), int(j)),
        "min_distance": float(np.sqrt(dist_sq[i, j])),
        "arc_length": arc,
    }


def compute_signed_loop_area(
    curve: np.ndarray,
    idx_pair: tuple[int, int],
) -> float:
    """Signed area of the loop enclosed between two self-intersection indices.

    Extracts ``curve[i:j+1]`` and applies the shoelace formula. Sign reflects
    traversal direction: positive = CCW, negative = CW. Overshooters produce
    a larger-magnitude area; never-arrives produce ~0 (loop not completed).

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        idx_pair: ``(i, j)`` indices from :func:`detect_self_intersection`.

    Returns:
        Signed area of the enclosed loop. Near zero if no loop is completed.
    """
    i, j = idx_pair
    loop = curve[i : j + 1]
    if len(loop) < 3:
        return 0.0
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_curvature_profile(
    curve: np.ndarray,
    n_norm_points: int = 100,
) -> dict:
    """Signed curvature κ as a function of normalized arc-length.

    Uses the discrete Frenet-Serret formula::

        κ = (x' · y'' − y' · x'') / (x'² + y'²)^(3/2)

    Derivatives are computed via :func:`numpy.gradient` (central differences)
    using the arc-length parameterization so the result is intrinsic to curve
    shape rather than epoch spacing. The profile is then resampled onto a
    uniform normalized arc-length grid ``[0, 1]`` for cross-variant comparison.

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        n_norm_points: Number of output samples on the normalized grid.

    Returns:
        Dict with:
            ``s_norm``: ``(n_norm_points,)`` normalized arc-length in ``[0, 1]``.
            ``kappa``: ``(n_norm_points,)`` curvature at each ``s_norm`` sample.
            ``s_raw``: ``(n_points,)`` raw cumulative arc-length.
            ``kappa_raw``: ``(n_points,)`` curvature at each original point.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]

    x, y = curve[:, 0], curve[:, 1]

    dx = np.gradient(x, arc)
    dy = np.gradient(y, arc)
    d2x = np.gradient(dx, arc)
    d2y = np.gradient(dy, arc)

    denom = (dx**2 + dy**2) ** 1.5
    kappa_raw = np.where(denom > 1e-12, (dx * d2y - dy * d2x) / denom, 0.0)

    s_norm_raw = arc / total_arc if total_arc > 1e-12 else np.linspace(0.0, 1.0, len(arc))
    s_norm = np.linspace(0.0, 1.0, n_norm_points)
    kappa = np.interp(s_norm, s_norm_raw, kappa_raw)

    return {
        "s_norm": s_norm,
        "kappa": kappa,
        "s_raw": arc,
        "kappa_raw": kappa_raw,
    }
