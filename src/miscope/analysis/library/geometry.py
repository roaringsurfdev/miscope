"""Representational geometry — Fisher matrix + crossover utility.

Shape characterizations (circularity, Fourier alignment, circle fit)
moved to :mod:`miscope.analysis.library.shape` per REQ_109 phase 2. Clustering
metrics (centroids, radii, dimensionality, center spread, Fisher discriminant)
live in :mod:`miscope.analysis.library.clustering`; PCA primitives in
:mod:`miscope.analysis.library.pca`. What's left here is the small set of
helpers that consume pre-computed centroid/radii artifacts directly without
re-deriving anything.

Functions:
- compute_fisher_matrix: Full pairwise Fisher discriminant matrix from stored data.
- find_circularity_crossovers: Detect epochs where attention circularity rises above / falls below reference sites.
"""

import numpy as np


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
