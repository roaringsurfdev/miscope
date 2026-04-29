"""REQ_050: Global PCA cross-epoch analyzer for centroid trajectories.

Computes a single PCA basis across all training epochs so that centroid
positions can be tracked as coherent trajectories in a consistent coordinate
frame. Unlike per-epoch PCA (which rotates between epochs), the global basis
enables meaningful cross-epoch comparison.
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.pca import pca

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]
_VARIANCE_THRESHOLD = 0.95


def _pca_with_variance_threshold(
    centroids_per_epoch: list[np.ndarray],
    threshold: float = _VARIANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool centroids, fit PCA, truncate to ``threshold`` cumulative variance.

    Returns ``(projections, basis, center, explained_variance_ratio)`` shaped
    for the cross-epoch artifact: ``projections`` is ``(n_epochs, n_classes,
    n_components)``; ``basis`` is ``(d_model, n_components)``.
    """
    n_epochs = len(centroids_per_epoch)
    n_classes = centroids_per_epoch[0].shape[0]
    pooled = np.concatenate(centroids_per_epoch, axis=0)

    full = pca(pooled)
    if full.explained_variance_ratio.sum() < 1e-12:
        n_components = 1
    else:
        cumvar = np.cumsum(full.explained_variance_ratio)
        passing = np.where(cumvar >= threshold)[0]
        n_components = int(passing[0]) + 1 if len(passing) > 0 else len(full.eigenvalues)

    basis = full.basis_vectors[:n_components].T  # (d_model, n_components)
    projections = full.projections[:, :n_components].reshape(n_epochs, n_classes, n_components)
    return projections, basis, full.center, full.explained_variance_ratio[:n_components]


class GlobalCentroidPCA:
    """Cross-epoch analyzer: single PCA basis for centroid trajectories.

    Loads per-epoch centroid data from repr_geometry artifacts, pools
    all epochs into a shared matrix, fits one PCA basis, and projects
    each epoch's centroids into that basis.

    Output (cross_epoch.npz) contains, for each activation site:
        {site}__projections          (n_epochs, n_classes, n_components)
        {site}__basis                (d_model, n_components)
        {site}__mean                 (d_model,)
        {site}__explained_variance_ratio  (n_components,)
    """

    name = "global_centroid_pca"
    requires = ["repr_geometry"]
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Compute global centroid PCA for all activation sites.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context (unused here).

        Returns:
            Dict of arrays for storage in cross_epoch.npz.
        """
        loader = ArtifactLoader(artifacts_dir)
        epoch_artifacts = [loader.load_epoch("repr_geometry", e) for e in epochs]

        result: dict[str, np.ndarray] = {"epochs": np.array(epochs)}

        first = epoch_artifacts[0] if epoch_artifacts else {}
        for site in _SITES:
            # Skip sites absent from the artifacts (e.g. residual stream for MLP)
            if f"{site}_centroids" not in first:
                continue
            centroids_per_epoch = [a[f"{site}_centroids"] for a in epoch_artifacts]
            projections, basis, center, var_ratio = _pca_with_variance_threshold(
                centroids_per_epoch
            )

            result[f"{site}__projections"] = projections
            result[f"{site}__basis"] = basis
            result[f"{site}__mean"] = center
            result[f"{site}__explained_variance_ratio"] = var_ratio

        return result
