"""Frequency Group Weight Geometry Analyzer.

Cross-epoch analyzer that applies the GLUE geometric framework to neuron
frequency groups in weight space. Rather than grouping input probes by
output class (as repr_geometry does), this groups MLP neurons by their
dominant frequency and measures how well-separated those groups are in
W_in and W_out weight space over training.

Key questions:
- Does grokking correspond to the moment frequency groups become distinct
  clusters in weight space?
- Do group centroids form a ring (as class centroids do in activation space)?
- Does each group's internal geometry become more elongated (lower effective
  dimensionality) as neurons specialize?
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.clustering import (
    compute_center_spread,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
)
from miscope.analysis.library.geometry import compute_circularity


class FreqGroupWeightGeometryAnalyzer:
    """Measures geometric separation of frequency groups in weight space.

    Groups neurons by dominant frequency at the final checkpoint (same
    convention as NeuronGroupPCAAnalyzer), then tracks geometric structure
    of those groups in W_in and W_out space over all training epochs.

    Reuses geometry library functions unchanged — neurons are samples,
    frequency group indices are labels.

    Cross-epoch artifact keys:
        group_freqs        int32   (n_groups,)
        group_sizes        int32   (n_groups,)
        Win_centroids      float32 (n_epochs, n_groups, d_model)
        Win_radii          float32 (n_epochs, n_groups)
        Win_dimensionality float32 (n_epochs, n_groups)  -- full participation ratio
        Win_pr3            float32 (n_epochs, n_groups)  -- PR₃ (top-3 eigenvalue shape)
        Win_f_top3         float32 (n_epochs, n_groups)  -- fraction of variance in top 3 PCs
        Win_center_spread  float32 (n_epochs,)
        Win_mean_radius    float32 (n_epochs,)
        Win_snr            float32 (n_epochs,)
        Win_fisher_mean    float32 (n_epochs,)
        Win_fisher_min     float32 (n_epochs,)
        Win_circularity    float32 (n_epochs,)
        Wout_*             same structure for W_out space
        epochs             int32   (n_epochs,)
    """

    name = "freq_group_weight_geometry"
    requires = ["neuron_freq_norm", "parameter_snapshot"]
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Compute frequency group geometry in weight space across all checkpoints."""
        loader = ArtifactLoader(artifacts_dir)
        sorted_epochs = sorted(epochs)

        group_freqs, group_sizes, group_labels = _build_group_labels(loader, sorted_epochs[-1])

        if not group_freqs:
            return _empty_result(sorted_epochs)

        n_groups = len(group_freqs)
        n_epochs = len(sorted_epochs)

        Win_centroids = np.full((n_epochs, n_groups, 0), np.nan, dtype=np.float32)
        Win_radii = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Win_dimensionality = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Win_pr3 = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Win_f_top3 = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Win_center_spread = np.full(n_epochs, np.nan, dtype=np.float32)
        Win_mean_radius = np.full(n_epochs, np.nan, dtype=np.float32)
        Win_snr = np.full(n_epochs, np.nan, dtype=np.float32)
        Win_fisher_mean = np.full(n_epochs, np.nan, dtype=np.float32)
        Win_fisher_min = np.full(n_epochs, np.nan, dtype=np.float32)
        Win_circularity = np.full(n_epochs, np.nan, dtype=np.float32)

        Wout_centroids = np.full((n_epochs, n_groups, 0), np.nan, dtype=np.float32)
        Wout_radii = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Wout_dimensionality = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Wout_pr3 = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Wout_f_top3 = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        Wout_center_spread = np.full(n_epochs, np.nan, dtype=np.float32)
        Wout_mean_radius = np.full(n_epochs, np.nan, dtype=np.float32)
        Wout_snr = np.full(n_epochs, np.nan, dtype=np.float32)
        Wout_fisher_mean = np.full(n_epochs, np.nan, dtype=np.float32)
        Wout_fisher_min = np.full(n_epochs, np.nan, dtype=np.float32)
        Wout_circularity = np.full(n_epochs, np.nan, dtype=np.float32)

        for ep_idx, epoch in enumerate(sorted_epochs):
            snap = loader.load_epoch("parameter_snapshot", epoch)
            is_transformer = "W_E" in snap

            if "W_in" in snap:
                # Rows = neurons: transformer (d_model, d_mlp).T; MLP (d_hidden, 2p) as-is
                W_in = (snap["W_in"].T if is_transformer else snap["W_in"]).astype(np.float64)
                win_geo = _compute_group_geometry(W_in, group_labels, n_groups)
                if ep_idx == 0:
                    d = W_in.shape[1]
                    Win_centroids = np.full((n_epochs, n_groups, d), np.nan, dtype=np.float32)
                Win_centroids[ep_idx] = win_geo["centroids"].astype(np.float32)
                Win_radii[ep_idx] = win_geo["radii"].astype(np.float32)
                Win_dimensionality[ep_idx] = win_geo["dimensionality"].astype(np.float32)
                Win_pr3[ep_idx] = win_geo["pr3"].astype(np.float32)
                Win_f_top3[ep_idx] = win_geo["f_top3"].astype(np.float32)
                Win_center_spread[ep_idx] = win_geo["center_spread"]
                Win_mean_radius[ep_idx] = win_geo["mean_radius"]
                Win_snr[ep_idx] = win_geo["snr"]
                Win_fisher_mean[ep_idx] = win_geo["fisher_mean"]
                Win_fisher_min[ep_idx] = win_geo["fisher_min"]
                Win_circularity[ep_idx] = win_geo["circularity"]

            if "W_out" in snap:
                # Rows = neurons: transformer (d_mlp, d_model) as-is; MLP (p, d_hidden).T
                W_out = (snap["W_out"] if is_transformer else snap["W_out"].T).astype(np.float64)
                wout_geo = _compute_group_geometry(W_out, group_labels, n_groups)
                if ep_idx == 0:
                    d = W_out.shape[1]
                    Wout_centroids = np.full((n_epochs, n_groups, d), np.nan, dtype=np.float32)
                Wout_centroids[ep_idx] = wout_geo["centroids"].astype(np.float32)
                Wout_radii[ep_idx] = wout_geo["radii"].astype(np.float32)
                Wout_dimensionality[ep_idx] = wout_geo["dimensionality"].astype(np.float32)
                Wout_pr3[ep_idx] = wout_geo["pr3"].astype(np.float32)
                Wout_f_top3[ep_idx] = wout_geo["f_top3"].astype(np.float32)
                Wout_center_spread[ep_idx] = wout_geo["center_spread"]
                Wout_mean_radius[ep_idx] = wout_geo["mean_radius"]
                Wout_snr[ep_idx] = wout_geo["snr"]
                Wout_fisher_mean[ep_idx] = wout_geo["fisher_mean"]
                Wout_fisher_min[ep_idx] = wout_geo["fisher_min"]
                Wout_circularity[ep_idx] = wout_geo["circularity"]

        return {
            "group_freqs": np.array(group_freqs, dtype=np.int32),
            "group_sizes": np.array(group_sizes, dtype=np.int32),
            "Win_centroids": Win_centroids,
            "Win_radii": Win_radii,
            "Win_dimensionality": Win_dimensionality,
            "Win_pr3": Win_pr3,
            "Win_f_top3": Win_f_top3,
            "Win_center_spread": Win_center_spread,
            "Win_mean_radius": Win_mean_radius,
            "Win_snr": Win_snr,
            "Win_fisher_mean": Win_fisher_mean,
            "Win_fisher_min": Win_fisher_min,
            "Win_circularity": Win_circularity,
            "Wout_centroids": Wout_centroids,
            "Wout_radii": Wout_radii,
            "Wout_dimensionality": Wout_dimensionality,
            "Wout_pr3": Wout_pr3,
            "Wout_f_top3": Wout_f_top3,
            "Wout_center_spread": Wout_center_spread,
            "Wout_mean_radius": Wout_mean_radius,
            "Wout_snr": Wout_snr,
            "Wout_fisher_mean": Wout_fisher_mean,
            "Wout_fisher_min": Wout_fisher_min,
            "Wout_circularity": Wout_circularity,
            "epochs": np.array(sorted_epochs, dtype=np.int32),
        }


def _build_group_labels(
    loader: ArtifactLoader,
    reference_epoch: int,
) -> tuple[list[int], list[int], np.ndarray]:
    """Assign neurons to frequency groups from the reference epoch.

    Group assignment by argmax of norm_matrix — no threshold, all neurons
    assigned to exactly one group. Groups with fewer than 2 neurons excluded.

    Returns:
        (group_freqs, group_sizes, group_labels) where group_labels is a
        (d_mlp,) int array mapping each neuron to a contiguous group index
        0..n_groups-1. Neurons in excluded groups are mapped to -1.
    """
    norm = loader.load_epoch("neuron_freq_norm", reference_epoch)
    norm_matrix = norm["norm_matrix"]  # (n_freq, d_mlp)
    d_mlp = norm_matrix.shape[1]
    dominant_freq = np.argmax(norm_matrix, axis=0)  # (d_mlp,)

    group_freqs = []
    group_sizes = []
    group_labels = np.full(d_mlp, -1, dtype=np.int32)

    group_idx = 0
    for f in range(norm_matrix.shape[0]):
        members = np.where(dominant_freq == f)[0]
        if len(members) >= 2:
            group_labels[members] = group_idx
            group_freqs.append(f)
            group_sizes.append(len(members))
            group_idx += 1

    return group_freqs, group_sizes, group_labels


def _compute_group_pr3_f_top3(
    W: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PR₃ and f_top3 for each frequency group's weight point cloud.

    PR₃ = (f1+f2+f3)² / (f1²+f2²+f3²) where fi = λi / Σλ (top-3 eigenvalues).
    f_top3 = (λ1+λ2+λ3) / Σλ — fraction of total variance in top 3 PCs.
    The denominator sums ALL eigenvalues, not just the top 3.
    Groups with fewer than 3 neurons receive NaN.
    """
    pr3 = np.full(n_groups, np.nan, dtype=np.float64)
    f_top3 = np.full(n_groups, np.nan, dtype=np.float64)
    for g in range(n_groups):
        members = W[labels == g]
        if len(members) < 3:
            continue
        centered = members - centroids[g]
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = s**2
        total = eigenvalues.sum()
        if total == 0:
            continue
        top3 = eigenvalues[:3]
        f = top3 / total
        denom = (f**2).sum()
        pr3[g] = (f.sum() ** 2) / denom if denom > 0 else np.nan
        f_top3[g] = top3.sum() / total
    return pr3, f_top3


def _compute_group_geometry(
    weights: np.ndarray,
    group_labels: np.ndarray,
    n_groups: int,
) -> dict[str, Any]:
    """Compute geometric measures for all frequency groups in one weight matrix.

    Args:
        weights: (d_mlp, d) — each row is one neuron's weight vector.
            Neurons with group_labels == -1 are excluded before computation.
        group_labels: (d_mlp,) — group index per neuron; -1 means ungrouped.
        n_groups: number of valid groups (labels 0..n_groups-1)

    Returns:
        Dict with per-group arrays (centroids, radii, dimensionality, pr3, f_top3) and
        global scalars (center_spread, mean_radius, snr, fisher_mean,
        fisher_min, circularity).
    """
    grouped_mask = group_labels >= 0
    W = weights[grouped_mask]
    labels = group_labels[grouped_mask]

    centroids = compute_class_centroids(W, labels, n_classes=n_groups)
    radii = compute_class_radii(W, labels, centroids)
    dimensionality = compute_class_dimensionality(W, labels, n_classes=n_groups)
    pr3, f_top3 = _compute_group_pr3_f_top3(W, labels, centroids, n_groups)

    center_spread = float(compute_center_spread(centroids))
    mean_radius = float(np.mean(radii))
    snr = (center_spread**2 / mean_radius**2) if mean_radius > 0 else 0.0
    fisher_mean, fisher_min = compute_fisher_discriminant(W, labels, centroids=centroids)
    circularity = float(compute_circularity(centroids))

    return {
        "centroids": centroids,
        "radii": radii,
        "dimensionality": dimensionality,
        "pr3": pr3,
        "f_top3": f_top3,
        "center_spread": np.float32(center_spread),
        "mean_radius": np.float32(mean_radius),
        "snr": np.float32(snr),
        "fisher_mean": np.float32(fisher_mean),
        "fisher_min": np.float32(fisher_min),
        "circularity": np.float32(circularity),
    }


def _empty_result(epochs: list[int]) -> dict[str, np.ndarray]:
    n = len(epochs)
    empty_scalar = np.full(n, np.nan, dtype=np.float32)
    return {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "Win_centroids": np.empty((n, 0, 0), dtype=np.float32),
        "Win_radii": np.empty((n, 0), dtype=np.float32),
        "Win_dimensionality": np.empty((n, 0), dtype=np.float32),
        "Win_pr3": np.empty((n, 0), dtype=np.float32),
        "Win_f_top3": np.empty((n, 0), dtype=np.float32),
        "Win_center_spread": empty_scalar.copy(),
        "Win_mean_radius": empty_scalar.copy(),
        "Win_snr": empty_scalar.copy(),
        "Win_fisher_mean": empty_scalar.copy(),
        "Win_fisher_min": empty_scalar.copy(),
        "Win_circularity": empty_scalar.copy(),
        "Wout_centroids": np.empty((n, 0, 0), dtype=np.float32),
        "Wout_radii": np.empty((n, 0), dtype=np.float32),
        "Wout_dimensionality": np.empty((n, 0), dtype=np.float32),
        "Wout_pr3": np.empty((n, 0), dtype=np.float32),
        "Wout_f_top3": np.empty((n, 0), dtype=np.float32),
        "Wout_center_spread": empty_scalar.copy(),
        "Wout_mean_radius": empty_scalar.copy(),
        "Wout_snr": empty_scalar.copy(),
        "Wout_fisher_mean": empty_scalar.copy(),
        "Wout_fisher_min": empty_scalar.copy(),
        "Wout_circularity": empty_scalar.copy(),
        "epochs": np.array(epochs, dtype=np.int32),
    }
