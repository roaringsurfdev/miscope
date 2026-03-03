"""Representational geometry computation functions.

Pure numpy functions for computing geometric properties of class manifolds
in activation space. No torch dependency — all inputs are numpy arrays.

Functions are vectorized where possible to minimize Python loop overhead
when computing across many classes (p can be 113+).

Functions:
- compute_class_centroids: Mean activation vector per output class
- compute_class_radii: RMS distance from centroid per class
- compute_class_dimensionality: Effective dimensionality (participation ratio) per class
- compute_center_spread: RMS distance of centroids from global centroid
- compute_circularity: How well centroids lie on a circle (Kåsa circle fit)
- compute_fourier_alignment: Whether angular ordering matches residue class ordering
- compute_fisher_discriminant: Pairwise Fisher discriminant ratio statistics
- compute_fisher_matrix: Full pairwise Fisher discriminant matrix from stored data
- compute_global_centroid_pca: Single PCA basis computed across all epochs (REQ_050)
- find_circularity_crossovers: Detect epochs where attention circularity rises above / falls below reference sites
"""

import numpy as np


def compute_class_centroids(
    activations: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute mean activation vector per output class.

    Vectorized: uses np.add.at for scatter-add, no Python loop.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        n_classes: Number of distinct classes (p)

    Returns:
        Centroid matrix, shape (n_classes, d)
    """
    d = activations.shape[1]
    centroids = np.zeros((n_classes, d))
    np.add.at(centroids, labels, activations)
    counts = np.bincount(labels, minlength=n_classes).reshape(-1, 1)
    counts = np.maximum(counts, 1)  # avoid division by zero
    centroids /= counts
    return centroids


def compute_class_radii(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute RMS distance from centroid for each class.

    Vectorized: broadcasts centroids[labels] across all samples,
    then aggregates per class with bincount.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Radii array, shape (n_classes,)
    """
    n_classes = centroids.shape[0]
    # Compute squared distances from each sample to its class centroid
    diffs = activations - centroids[labels]
    sq_dists = np.sum(diffs**2, axis=1)  # (n_samples,)
    # Sum squared distances per class
    sum_sq = np.bincount(labels, weights=sq_dists, minlength=n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    return np.sqrt(sum_sq / counts)


def compute_class_dimensionality(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute effective dimensionality (participation ratio) per class.

    For each class, computes participation ratio from eigenvalues of
    the class covariance matrix:
        D_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)

    This equals 1 if all variance is on one axis, and d if variance
    is uniform across d dimensions.

    Note: This function retains a per-class loop for the eigendecomposition
    since numpy does not support batched eigvalsh. The loop body is kept
    minimal — covariance via matrix multiply, eigvalsh, two reductions.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Effective dimensionality array, shape (n_classes,)
    """
    n_classes = centroids.shape[0]
    dims = np.zeros(n_classes)
    for r in range(n_classes):
        mask = labels == r
        centered = activations[mask] - centroids[r]
        n = centered.shape[0]
        if n < 2:
            dims[r] = 0.0
            continue
        cov = centered.T @ centered / n
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sum_ev = eigenvalues.sum()
        sum_ev_sq = (eigenvalues**2).sum()
        if sum_ev_sq > 0:
            dims[r] = sum_ev**2 / sum_ev_sq
        else:
            dims[r] = 0.0
    return dims


def compute_center_spread(centroids: np.ndarray) -> float:
    """Compute RMS distance of centroids from their global mean.

    Args:
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Center spread (scalar)
    """
    global_centroid = centroids.mean(axis=0)
    diffs = centroids - global_centroid
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


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


def compute_fisher_discriminant(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[float, float]:
    """Compute pairwise Fisher discriminant ratio statistics.

    For each pair of classes (r, s):
        J(r, s) = ||mu_r - mu_s||^2 / (sigma_r^2 + sigma_s^2)

    where sigma_r^2 is the mean within-class variance for class r.

    Vectorized: within-class variances via bincount, pairwise distances
    and Fisher ratios via scipy-style pdist broadcasting.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Tuple of (mean_fisher, min_fisher) across all class pairs
    """
    n_classes = centroids.shape[0]

    # Within-class variance per class (vectorized)
    diffs = activations - centroids[labels]
    sq_dists = np.sum(diffs**2, axis=1)
    sum_sq = np.bincount(labels, weights=sq_dists, minlength=n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    variances = sum_sq / counts  # (n_classes,)

    # Pairwise between-class distances (vectorized)
    # ||mu_r - mu_s||^2 for all pairs using broadcasting
    centroid_diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    pairwise_sq_dists = np.sum(centroid_diffs**2, axis=2)  # (n_classes, n_classes)

    # Pairwise within-class variance sums
    pairwise_within = variances[:, np.newaxis] + variances[np.newaxis, :]

    # Fisher ratios (upper triangle only)
    r_idx, s_idx = np.triu_indices(n_classes, k=1)
    between = pairwise_sq_dists[r_idx, s_idx]
    within = pairwise_within[r_idx, s_idx]

    # Compute ratios, handling zero within-class variance
    valid = within > 0
    if not valid.any():
        return 0.0, 0.0
    fisher_values = np.where(valid, between / np.maximum(within, 1e-12), np.inf)
    finite_mask = np.isfinite(fisher_values)
    if not finite_mask.any():
        return 0.0, 0.0
    mean_fisher = float(np.mean(fisher_values[finite_mask]))
    min_fisher = float(np.min(fisher_values[finite_mask]))
    return mean_fisher, min_fisher


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


def compute_global_centroid_pca(
    centroids_per_epoch: list[np.ndarray],
    variance_threshold: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute a single PCA basis across all epochs for cross-epoch tracking.

    Pools centroid matrices from all epochs into one shared matrix, fits PCA
    once, and projects each epoch's centroids into the shared coordinate frame.
    This produces a consistent basis for tracking centroid trajectories over
    training — unlike per-epoch PCA, the coordinate frame does not rotate.

    The number of retained components is data-driven: the smallest k such that
    the top-k components capture at least `variance_threshold` cumulative
    explained variance from the pooled matrix.

    Args:
        centroids_per_epoch: List of centroid matrices, one per epoch.
            Each has shape (n_classes, d_model). Must be non-empty.
        variance_threshold: Retain components until cumulative explained
            variance reaches this fraction. Default 0.95.

    Returns:
        Dict with:
            "basis": shape (d_model, n_components) — global PCA eigenvectors
            "mean": shape (d_model,) — global mean subtracted before projection
            "projections": shape (n_epochs, n_classes, n_components)
            "explained_variance_ratio": shape (n_components,) — per-component
                explained variance from the pooled matrix
    """
    n_epochs = len(centroids_per_epoch)
    pooled = np.concatenate(centroids_per_epoch, axis=0)  # (n_epochs*n_classes, d_model)
    global_mean = pooled.mean(axis=0)
    centered = pooled - global_mean

    cov = centered.T @ centered / centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; reverse for descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        n_components = 1
    else:
        cumvar = np.cumsum(eigenvalues) / total_var
        passing = np.where(cumvar >= variance_threshold)[0]
        n_components = int(passing[0]) + 1 if len(passing) > 0 else len(eigenvalues)

    basis = eigenvectors[:, :n_components]  # (d_model, n_components)
    var_ratio = (
        eigenvalues[:n_components] / total_var if total_var > 1e-12 else np.zeros(n_components)
    )

    n_classes = centroids_per_epoch[0].shape[0]
    projections = np.empty((n_epochs, n_classes, n_components))
    for i, centroids in enumerate(centroids_per_epoch):
        projections[i] = (centroids - global_mean) @ basis

    return {
        "basis": basis,
        "mean": global_mean,
        "projections": projections,
        "explained_variance_ratio": var_ratio,
    }


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
