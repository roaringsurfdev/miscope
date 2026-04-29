"""Representational geometry computation functions.

Shape characterization helpers (circularity, Fourier alignment, circle fit)
and the Fisher matrix helper live here. Clustering metrics (centroids, radii,
dimensionality, center spread, Fisher discriminant) live in
:mod:`miscope.analysis.library.clustering`; PCA primitives in
:mod:`miscope.analysis.library.pca`. Callers should import from the
canonical home; this module's leftover helpers are scoped to shape
characterization (REQ_109 phase 2).

Functions:
- compute_circularity: How well centroids lie on a circle (Kåsa circle fit).
- compute_fourier_alignment: Whether angular ordering matches residue class ordering.
- compute_fisher_matrix: Full pairwise Fisher discriminant matrix from stored data.
- find_circularity_crossovers: Detect epochs where attention circularity rises above / falls below reference sites.
"""

import numpy as np


def compute_circularity(centroids: np.ndarray) -> float:
    """Compute how well centroids lie on a circle in their top-2 PCA subspace.

    Projects centroids into top-2 PCs, fits a circle using the algebraic
    Kåsa method, and returns a score weighted by how much variance the
    top-2 PCs capture:

        raw_score = 1 - (mean_squared_residual / variance_in_2d_plane)
        score = raw_score * variance_explained_ratio

    The weighting ensures that data which is essentially 1D (collinear)
    or high-dimensional (random cloud) scores low even if the 2D
    projection happens to look circular.

    Score of 1.0 means perfect circle, 0.0 means no circular structure.
    Clamped to [0, 1].

    Args:
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Circularity score in [0, 1]
    """
    projected, var_explained = _pca_project_2d(centroids)
    cx, cy, radius = _kasa_circle_fit(projected)
    distances = np.sqrt((projected[:, 0] - cx) ** 2 + (projected[:, 1] - cy) ** 2)
    residuals = distances - radius
    msr = np.mean(residuals**2)
    variance = np.var(projected[:, 0]) + np.var(projected[:, 1])
    if variance < 1e-12:
        return 0.0
    raw_score = 1.0 - msr / variance
    score = raw_score * var_explained
    return float(np.clip(score, 0.0, 1.0))


def compute_fourier_alignment(centroids: np.ndarray, p: int) -> float:
    """Compute whether angular ordering of centroids matches residue class ordering.

    Projects centroids to top-2 PCs, computes angles, then finds the
    frequency k that best explains the angular positions as theta_r = 2*pi*k*r/p.
    Returns R^2 of the best fit.

    Vectorized: tests all frequencies k in one broadcast operation.

    Args:
        centroids: Centroid matrix, shape (p, d) where rows are ordered by class
        p: Prime (number of classes, should equal centroids.shape[0])

    Returns:
        Fourier alignment R^2 in [0, 1]
    """
    projected, _ = _pca_project_2d(centroids)
    cx, cy, _ = _kasa_circle_fit(projected)
    angles = np.arctan2(projected[:, 1] - cy, projected[:, 0] - cx)

    # Test all frequencies k=1..p-1 in one vectorized operation
    z_observed = np.exp(1j * angles)  # (p,)
    k_values = np.arange(1, p)  # (p-1,)
    residue_indices = np.arange(p)  # (p,)
    # Expected angles for each k: (p-1, p)
    expected = 2 * np.pi * k_values[:, np.newaxis] * residue_indices[np.newaxis, :] / p
    z_expected = np.exp(1j * expected)  # (p-1, p)
    # Circular correlation for each k
    correlations = np.abs(np.mean(z_observed[np.newaxis, :] * np.conj(z_expected), axis=1)) ** 2
    return float(np.max(correlations))


def compute_fisher_matrix(
    centroids: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Compute full pairwise Fisher discriminant matrix from stored data.

    J(r, s) = ||mu_r - mu_s||^2 / (radius_r^2 + radius_s^2)

    This function operates on pre-computed centroids and radii (as stored
    in per-epoch artifacts), enabling render-time computation without
    needing raw activations. radii^2 equals within-class variance.

    Args:
        centroids: Class centroid matrix, shape (n_classes, d)
        radii: RMS radius per class, shape (n_classes,)

    Returns:
        Fisher discriminant matrix, shape (n_classes, n_classes).
        Symmetric with zero diagonal.
    """
    variances = radii**2
    diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    pairwise_sq_dists = np.sum(diffs**2, axis=2)
    pairwise_within = variances[:, np.newaxis] + variances[np.newaxis, :]
    fisher_matrix = np.where(
        pairwise_within > 0,
        pairwise_sq_dists / np.maximum(pairwise_within, 1e-12),
        0.0,
    )
    np.fill_diagonal(fisher_matrix, 0.0)
    return fisher_matrix


# --- Private helpers ---


def _pca_project(points: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Project points into their top-N principal components.

    Args:
        points: Matrix of shape (n, d).
        n_components: Number of principal components to return.

    Returns:
        Tuple of (projected points of shape (n, n_components),
                  per-component variance explained of shape (n_components,)).
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; take last n_components, reversed
    n = min(n_components, len(eigenvalues))
    top_vecs = eigenvectors[:, -n:][:, ::-1]
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        var_fracs = np.zeros(n)
    else:
        var_fracs = eigenvalues[-n:][::-1] / total_var
    return centered @ top_vecs, var_fracs


def _pca_project_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Project points into their top-2 principal components.

    Returns a quality score that combines two factors:
    - How much variance the top-2 PCs capture (low for high-D random data)
    - How balanced the two PCs are (low for 1D/collinear data)

    This ensures circular structure is only reported when the data
    genuinely lives in a 2D subspace.

    Args:
        points: Matrix of shape (n, d)

    Returns:
        Tuple of (projected points of shape (n, 2),
                  2D quality score in [0, 1])
    """
    projected, var_fracs = _pca_project(points, n_components=2)
    var_explained = float(var_fracs.sum())
    balance = float(var_fracs[1] / var_fracs[0]) if var_fracs[0] > 1e-12 else 0.0
    quality = var_explained * balance
    return projected, quality


def _kasa_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D points using the algebraic Kåsa method.

    Solves the least-squares system for a, b, c in:
        x^2 + y^2 + ax + by + c = 0

    Args:
        points: 2D points of shape (n, 2)

    Returns:
        Tuple of (center_x, center_y, radius)
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


def find_circularity_crossovers(
    summary_data: dict,
    attn_site: str = "attn_out",
    reference_sites: tuple[str, ...] = ("mlp_out", "resid_post"),
) -> dict:
    """Detect all epochs where attention circularity crosses reference site circularity.

    For each reference site, scans the diff timeseries for every sign change.
    All events are collected and returned sorted by epoch so early-training
    crossovers are visible alongside later ones.

    Args:
        summary_data: Cross-epoch summary from load_summary("repr_geometry").
                      Must contain "epochs" and "{site}_circularity" arrays.
        attn_site: Attention activation site key. Default: "attn_out".
        reference_sites: Sites to compare against. Default: ("mlp_out", "resid_post").

    Returns:
        Dict with keys:
        - "events": list of {"epoch": int, "direction": "rise"|"fall", "site": str}
                    all crossover events across all reference sites, sorted by epoch.
        - "per_site": {site: [{"epoch": int, "direction": "rise"|"fall"}, ...]}
    """
    epochs = np.array(summary_data["epochs"])
    attn_circ = np.array(summary_data[f"{attn_site}_circularity"])

    per_site: dict = {}
    all_events: list[dict] = []
    for site in reference_sites:
        ref_circ = np.array(summary_data[f"{site}_circularity"])
        diff = attn_circ - ref_circ
        signs = np.sign(diff)

        site_events: list[dict] = []
        for i in range(len(signs) - 1):
            if signs[i] <= 0 and signs[i + 1] > 0:
                site_events.append({"epoch": int(epochs[i + 1]), "direction": "rise"})
            elif signs[i] >= 0 and signs[i + 1] < 0:
                site_events.append({"epoch": int(epochs[i + 1]), "direction": "fall"})

        per_site[site] = site_events
        for evt in site_events:
            all_events.append({**evt, "site": site})

    all_events.sort(key=lambda e: e["epoch"])
    return {"events": all_events, "per_site": per_site}
