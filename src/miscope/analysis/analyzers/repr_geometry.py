"""Representational Geometry Analyzer.

Computes geometric properties of class manifolds in activation space
at multiple sites in the network. Tracks how representational structure
evolves during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle, ActivationContext
from miscope.analysis.library.clustering import (
    compute_center_spread,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
)
from miscope.analysis.library.geometry import compute_fisher_matrix
from miscope.analysis.library.pca import pca
from miscope.analysis.library.shape import (
    characterize_circularity,
    characterize_fourier_alignment,
)

# Activation sites to probe, with extraction config
_SITES = {
    "resid_pre": {"extractor": "residual", "location": "resid_pre"},
    "attn_out": {"extractor": "residual", "location": "attn_out"},
    "mlp_out": {"extractor": "mlp"},
    "resid_post": {"extractor": "residual", "location": "resid_post"},
}

# Summary keys: 11 scalar measures per site
_SCALAR_KEYS = [
    "mean_radius",
    "mean_dim",
    "center_spread",
    "snr",
    "circularity",
    "fourier_alignment",
    "fisher_mean",
    "fisher_min",
    "fisher_argmin_r",
    "fisher_argmin_s",
    "fisher_argmin_diff",
]

# PCA variance fraction keys: top-3 PC variance per site
_PCA_VAR_KEYS = ["pca_var_pc1", "pca_var_pc2", "pca_var_pc3"]


def _get_summary_keys() -> list[str]:
    """Build the full list of summary stat keys across all sites."""
    scalar_keys = [f"{site}_{key}" for site in _SITES for key in _SCALAR_KEYS]
    pca_keys = [f"{site}_{key}" for site in _SITES for key in _PCA_VAR_KEYS]
    return scalar_keys + pca_keys


class RepresentationalGeometryAnalyzer:
    """Computes geometric properties of class manifolds in activation space.

    For each checkpoint, extracts activations at 4 network sites, groups
    them by output class, and computes per-class and global geometric
    measures including centroids, radii, dimensionality, SNR, circularity,
    Fourier alignment, and Fisher discriminant ratios.
    """

    name = "repr_geometry"
    description = "Tracks representational geometry evolution across training"
    architecture_support = ["transformer", "mlp"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Compute geometric measures at all activation sites.

        Args:
            ctx: Analysis context with bundle, probe, and analysis_params.
                 analysis_params may contain 'labels' or 'params' with 'prime'.

        Returns:
            Dict with site-prefixed keys for centroids, radii,
            dimensionality, and global scalar measures.
        """
        p = compute_grid_size_from_dataset(ctx.probe)
        labels = self._compute_labels(ctx.probe, p, ctx.analysis_params)

        result: dict[str, np.ndarray] = {}
        for site_name, site_config in _SITES.items():
            if not ctx.bundle.supports_site(site_config["extractor"]):
                continue
            activations = self._extract_site(ctx.bundle, site_config)
            site_result = self._compute_site_measures(activations, labels, p)
            for key, value in site_result.items():
                result[f"{site_name}_{key}"] = value

        return result

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys (scalars only)."""
        return _get_summary_keys()

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Extract scalar summary stats from epoch result.

        Picks out pre-computed scalar keys and computes PCA variance
        fractions from the stored centroid matrices.
        """
        # Only summarize sites that were actually computed (varies by architecture)
        present_sites = [s for s in _SITES if f"{s}_centroids" in result]
        scalar_keys = [f"{site}_{key}" for site in present_sites for key in _SCALAR_KEYS]
        summary: dict[str, float | np.ndarray] = {key: float(result[key]) for key in scalar_keys}

        # PCA variance fractions are stored per-site by analyze() — no
        # re-derivation needed. Backwards compat: fall back to deriving from
        # centroids if the per-site PCA keys are missing (legacy artifacts
        # written before this analyzer included them).
        for site_name in present_sites:
            for pc_idx in (1, 2, 3):
                key = f"{site_name}_pca_var_pc{pc_idx}"
                if key in result:
                    summary[key] = float(result[key])
                else:
                    centroids = result[f"{site_name}_centroids"]
                    n_components = min(3, centroids.shape[0], centroids.shape[1])
                    var_fracs = pca(centroids, n_components=n_components).explained_variance_ratio
                    padded = np.zeros(3, dtype=np.float64)
                    padded[: var_fracs.shape[0]] = var_fracs
                    for legacy_idx in (1, 2, 3):
                        legacy_key = f"{site_name}_pca_var_pc{legacy_idx}"
                        summary[legacy_key] = float(padded[legacy_idx - 1])
                    break

        return summary

    def _compute_labels(
        self,
        probe: torch.Tensor,
        p: int,
        context: dict[str, Any],
    ) -> np.ndarray:
        """Compute output class labels: (a + b) mod p.

        Uses precomputed labels from context when available (required for
        one-hot encoded probes where probe[:, 0] is not the value of a).
        Falls back to reading a and b directly from the first two probe columns
        for token-indexed probes (transformer family format).
        """
        if "labels" in context:
            return context["labels"]
        probe_np = probe.cpu().numpy()
        a = probe_np[:, 0].astype(int)
        b = probe_np[:, 1].astype(int)
        return (a + b) % p

    def _extract_site(
        self,
        bundle: ActivationBundle,
        site_config: dict,
    ) -> np.ndarray:
        """Extract activations from bundle for a given site."""
        if site_config["extractor"] == "mlp":
            acts = bundle.mlp_post(0, -1)
        else:
            acts = bundle.residual_stream(0, -1, site_config["location"])
        return acts.detach().cpu().numpy()

    def _compute_site_measures(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        p: int,
    ) -> dict[str, Any]:
        """Compute all geometric measures for one activation site."""
        centroids = compute_class_centroids(activations, labels, n_classes=p)
        radii = compute_class_radii(activations, labels, centroids)
        dimensionality = compute_class_dimensionality(activations, labels, n_classes=p)

        mean_radius = np.mean(radii)
        mean_dim = np.mean(dimensionality)
        center_spread = compute_center_spread(centroids)
        snr = (center_spread**2 / mean_radius**2) if mean_radius > 0 else 0.0

        # Single PCA over centroids feeds circularity, fourier_alignment, and
        # the pca_var_pc{1,2,3} summary fractions — collapses what was three
        # redundant SVDs per (epoch, site) down to one.
        n_components = min(3, centroids.shape[0], centroids.shape[1])
        centroid_pca = pca(centroids, n_components=n_components)
        var_ratio = centroid_pca.explained_variance_ratio
        projection_2d = (
            centroid_pca.projections[:, :2] if n_components >= 2 else centroid_pca.projections
        )
        var_explained_2d = (
            float(var_ratio[:2].sum()) if n_components >= 2 else float(var_ratio.sum())
        )
        pca_var_top3 = np.zeros(3, dtype=np.float64)
        pca_var_top3[: var_ratio.shape[0]] = var_ratio

        circularity = (
            characterize_circularity(projection_2d, var_explained_2d) if n_components >= 2 else 0.0
        )
        fourier_align = (
            characterize_fourier_alignment(projection_2d, p) if n_components >= 2 else 0.0
        )
        fisher_mean, fisher_min = compute_fisher_discriminant(
            activations, labels, centroids=centroids
        )

        # Find the argmin pair (weakest separation) from the full Fisher matrix
        fisher_mat = compute_fisher_matrix(centroids, radii)
        r_idx, s_idx = np.triu_indices(p, k=1)
        fisher_upper = fisher_mat[r_idx, s_idx]
        if len(fisher_upper) > 0:
            argmin_idx = int(np.argmin(fisher_upper))
            argmin_r = int(r_idx[argmin_idx])
            argmin_s = int(s_idx[argmin_idx])
            # Circular distance in residue space
            raw_diff = abs(argmin_s - argmin_r)
            argmin_diff = min(raw_diff, p - raw_diff)
        else:
            argmin_r, argmin_s, argmin_diff = 0, 0, 0

        return {
            "centroids": centroids,
            "radii": radii,
            "dimensionality": dimensionality,
            "mean_radius": np.float64(mean_radius),
            "mean_dim": np.float64(mean_dim),
            "center_spread": np.float64(center_spread),
            "snr": np.float64(snr),
            "circularity": np.float64(circularity),
            "fourier_alignment": np.float64(fourier_align),
            "fisher_mean": np.float64(fisher_mean),
            "fisher_min": np.float64(fisher_min),
            "fisher_argmin_r": np.float64(argmin_r),
            "fisher_argmin_s": np.float64(argmin_s),
            "fisher_argmin_diff": np.float64(argmin_diff),
            "pca_var_pc1": np.float64(pca_var_top3[0]),
            "pca_var_pc2": np.float64(pca_var_top3[1]),
            "pca_var_pc3": np.float64(pca_var_top3[2]),
        }  # type: ignore
