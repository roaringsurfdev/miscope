"""Neuron Group PCA Analyzer.

Cross-epoch analyzer that measures within-frequency-group coordination
in weight space. Neurons are grouped by dominant frequency at the final
checkpoint. For each group, tracks PC1 variance explained (alignment)
and mean within-group spread (dispersion) across all training epochs.

High PC1 variance explained → neurons in the group point in similar
directions in weight space (coordinated unit).
Low PC1 variance explained → neurons with the same nominal frequency
are spread across multiple directions (diffuse).
"""

from typing import Any

import numpy as np

from miscope.analysis.inputs import ALL, ArtifactInput, DepsAccessor, ResolvedInputs
from miscope.analysis.library import (
    NEURON_FREQ_NORM_FIELDS,
    extract_neuron_weight_matrix,
    reconstruct_neuron_freq_norm,
)
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# Keys are flat (no prefix): for columnar fields the group/neuron axes flatten to
# rows; the dense trajectory cubes stay single per-variant tensors (group/pc/
# d_model are internal array axes, not separately-addressable blobs).
SPEC = AnalyzerSpec(
    name="neuron_group_pca",
    output_scope="cross_epoch",
    inputs=(
        ArtifactInput("activation_basis_projection"),
        ArtifactInput("parameter_snapshot"),
    ),
    outputs=(
        F.columnar(
            "group_freqs",
            "int32",
            ("variant", "group"),
            "Frequency each neuron group represents.",
        ),
        F.columnar(
            "group_sizes",
            "int32",
            ("variant", "group"),
            "Neuron count in each group.",
        ),
        F.columnar(
            "pc_var",
            "float32",
            ("variant", "epoch", "group", "row_id"),
            "Per-group PCA variance fraction per PC over training (row_id = PC index).",
        ),
        F.columnar(
            "mean_spread",
            "float32",
            ("variant", "epoch", "group"),
            "Mean within-group spread over training.",
        ),
        F.columnar(
            "neuron_group_idx",
            "int32",
            ("variant", "neuron"),
            "Group index assigned to each neuron.",
        ),
        F.columnar(
            "epochs",
            "int32",
            ("variant", "epoch"),
            "Epoch axis labels for the trajectories.",
        ),
        F.columnar(
            "centroid_pca_var",
            "float32",
            ("variant", "row_id"),
            "Variance fraction per PC of the group-centroid PCA (row_id = PC index).",
        ),
        F.tensor(
            "group_bases",
            "float32",
            ("variant",),
            "Per-group PCA basis vectors (group, n_pc, d_model).",
        ),
        F.tensor(
            "group_centers",
            "float32",
            ("variant",),
            "Per-group center in activation space (group, d_model).",
        ),
        F.tensor(
            "projections",
            "float32",
            ("variant",),
            "Per-neuron projections onto group PCs over training (n_epochs, d_mlp, n_pc).",
        ),
        F.tensor(
            "centroid_traj",
            "float32",
            ("variant",),
            "Group-centroid trajectories (n_epochs, group, d_model).",
        ),
        F.tensor(
            "centroid_pca_coords",
            "float32",
            ("variant",),
            "Group centroids projected into a shared PCA space (n_epochs, group, n_pc).",
        ),
        F.tensor(
            "centroid_pca_basis",
            "float32",
            ("variant",),
            "Basis of the shared group-centroid PCA space (n_pc, d_model).",
        ),
    ),
)


@register_analyzer(SPEC)
class NeuronGroupPCAAnalyzer:
    """Measures within-frequency-group coordination in weight space.

    Groups neurons by dominant frequency at the final checkpoint, then
    tracks group alignment (PC1 var explained) and dispersion (mean L2
    spread) over all training epochs. Also pre-computes per-epoch neuron
    projections onto the final-epoch PCA bases for scatter/trajectory views.

    Cross-epoch artifact keys:
        group_freqs       int32   (n_groups,)                frequency index per group
        group_sizes       int32   (n_groups,)                neuron count per group
        pc_var            float32 (n_epochs, n_groups, 3)    per-component variance explained
        mean_spread       float32 (n_epochs, n_groups)       mean L2 distance from centroid
        group_bases       float32 (n_groups, 3, d_model)     reference PCA bases (final epoch)
        group_centers     float32 (n_groups, d_model)        final-epoch group centroids
        projections       float32 (n_epochs, d_mlp, 3)       per-neuron PC coords (NaN if ungrouped)
        neuron_group_idx  int32   (d_mlp,)                   group index per neuron (-1 if ungrouped)
        epochs            int32   (n_epochs,)
    """

    name = "neuron_group_pca"
    requires = ["activation_basis_projection", "parameter_snapshot"]

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute group coordination metrics across all checkpoints."""
        assert inputs.deps is not None
        assert inputs.epochs is not None
        prime = int(context["params"]["prime"])
        epochs = list(inputs.epochs)
        sorted_epochs = sorted(epochs)

        group_freqs, group_members = _assign_groups(inputs.deps, sorted_epochs[-1], prime)

        if not group_freqs:
            return _empty_result(sorted_epochs)

        n_groups = len(group_freqs)
        n_epochs = len(sorted_epochs)
        pc_var = np.full((n_epochs, n_groups, N_COMPONENTS), np.nan, dtype=np.float32)
        mean_spread = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)

        W_ins: list[np.ndarray] = []
        for ep_idx, epoch in enumerate(sorted_epochs):
            snap = inputs.deps.load_epoch("parameter_snapshot", epoch, fields=ALL)
            W_in = extract_neuron_weight_matrix(snap)  # (d_space, M) — architecture-agnostic
            W_ins.append(W_in)
            for g_idx, members in enumerate(group_members):
                pc_var[ep_idx, g_idx], mean_spread[ep_idx, g_idx] = _group_pca_stats(
                    W_in[:, members]
                )

        final_W_in = W_ins[-1]
        group_bases = _compute_group_bases(final_W_in, group_members)
        group_centers = _compute_group_centers(final_W_in, group_members)
        d_mlp = final_W_in.shape[1]

        neuron_group_idx = np.full(d_mlp, -1, dtype=np.int32)
        for g_idx, members in enumerate(group_members):
            neuron_group_idx[members] = g_idx

        projections = _compute_all_projections(
            W_ins, group_bases, group_centers, group_members, d_mlp
        )

        centroid_traj = _compute_centroid_trajectory(W_ins, group_members)
        centroid_pca_coords, centroid_pca_var, centroid_pca_basis = _fit_centroid_pca(centroid_traj)

        return {
            "group_freqs": np.array(group_freqs, dtype=np.int32),
            "group_sizes": np.array([len(m) for m in group_members], dtype=np.int32),
            "pc_var": pc_var,
            "mean_spread": mean_spread,
            "group_bases": group_bases,
            "group_centers": group_centers,
            "projections": projections,
            "neuron_group_idx": neuron_group_idx,
            "epochs": np.array(sorted_epochs, dtype=np.int32),
            "centroid_traj": centroid_traj,
            "centroid_pca_coords": centroid_pca_coords,
            "centroid_pca_var": centroid_pca_var,
            "centroid_pca_basis": centroid_pca_basis,
        }


def _assign_groups(
    deps: DepsAccessor,
    reference_epoch: int,
    prime: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Assign neurons to frequency groups using the reference epoch.

    Returns only groups with at least 2 neurons (PCA requires >= 2).
    Group assignment is by argmax of norm_matrix — no threshold applied,
    so all neurons are assigned to exactly one group.

    Returns:
        (group_freqs, group_members): parallel lists of frequency index
        and member neuron indices for each group.
    """
    projection = deps.load_epoch(
        "activation_basis_projection", reference_epoch, fields=NEURON_FREQ_NORM_FIELDS
    )
    norm_matrix = reconstruct_neuron_freq_norm(projection, prime)  # (n_freq, d_mlp)
    dominant_freq = np.argmax(norm_matrix, axis=0)  # (d_mlp,)

    group_freqs = []
    group_members = []
    for f in range(norm_matrix.shape[0]):
        members = np.where(dominant_freq == f)[0]
        if len(members) >= 2:
            group_freqs.append(f)
            group_members.append(members)

    return group_freqs, group_members


N_COMPONENTS = 3


def _group_pca_stats(group_W: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute per-component variance explained and mean spread for a neuron group.

    Returns variance fractions for the top N_COMPONENTS principal components.
    Groups smaller than N_COMPONENTS get NaN-padded output for missing components.

    Args:
        group_W: (d_model, n_group) weight vectors for neurons in the group

    Returns:
        (pc_var, mean_L2_spread) where pc_var is (N_COMPONENTS,) float32
    """
    centroid = group_W.mean(axis=1, keepdims=True)  # (d_model, 1)
    centered = group_W - centroid  # (d_model, n_group)

    spread = float(np.linalg.norm(centered, axis=0).mean())

    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    total_var = float((s**2).sum())

    pc_var = np.full(N_COMPONENTS, np.nan, dtype=np.float32)
    if total_var > 1e-10:
        n_valid = min(N_COMPONENTS, len(s))
        pc_var[:n_valid] = (s[:n_valid] ** 2 / total_var).astype(np.float32)

    return pc_var, spread


def _compute_centroid_trajectory(
    W_ins: list[np.ndarray],
    group_members: list[np.ndarray],
) -> np.ndarray:
    """Compute the W_in group centroid at every epoch.

    Args:
        W_ins: list of (d_model, d_mlp) snapshots, one per epoch
        group_members: list of member index arrays, one per group

    Returns:
        (n_epochs, n_groups, d_model) float32
    """
    n_epochs = len(W_ins)
    n_groups = len(group_members)
    d_model = W_ins[0].shape[0]
    traj = np.zeros((n_epochs, n_groups, d_model), dtype=np.float32)
    for ep_idx, W_in in enumerate(W_ins):
        for g_idx, members in enumerate(group_members):
            traj[ep_idx, g_idx] = W_in[:, members].mean(axis=1)
    return traj


def _fit_centroid_pca(
    centroid_traj: np.ndarray,
    n_components: int = N_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared PCA on all group centroids across all epochs.

    Stacks every (epoch, group) centroid into one matrix and fits a single
    PCA basis, giving a common coordinate frame for comparing group paths.

    Args:
        centroid_traj: (n_epochs, n_groups, d_model)

    Returns:
        coords:       (n_epochs, n_groups, n_components) — PCA projections
        var_explained (n_components,) float32
        basis:        (n_components, d_model) float32
    """
    n_epochs, n_groups, d_model = centroid_traj.shape
    stacked = centroid_traj.reshape(-1, d_model)  # (n_epochs*n_groups, d_model)
    center = stacked.mean(axis=0)
    X = stacked - center
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, len(S))
    basis = Vt[:k].astype(np.float32)  # (k, d_model)
    var_explained = np.zeros(n_components, dtype=np.float32)
    total_var = float((S**2).sum())
    if total_var > 1e-10:
        var_explained[:k] = (S[:k] ** 2 / total_var).astype(np.float32)
    coords = ((centroid_traj - center) @ basis.T).astype(np.float32)  # (n_epochs, n_groups, k)
    if k < n_components:
        pad = np.zeros((n_epochs, n_groups, n_components - k), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=2)
        basis_pad = np.zeros((n_components - k, d_model), dtype=np.float32)
        basis = np.concatenate([basis, basis_pad], axis=0)
    return coords, var_explained, basis


def _compute_group_bases(W_in: np.ndarray, group_members: list[np.ndarray]) -> np.ndarray:
    """Compute PCA reference bases for all groups from a single W_in snapshot.

    Args:
        W_in: (d_model, d_mlp) weight matrix
        group_members: list of member index arrays, one per group

    Returns:
        (n_groups, N_COMPONENTS, d_model) float32 — top-k left singular vectors per group
    """
    d_model = W_in.shape[0]
    n_groups = len(group_members)
    bases = np.zeros((n_groups, N_COMPONENTS, d_model), dtype=np.float32)
    for g_idx, members in enumerate(group_members):
        centroid = W_in[:, members].mean(axis=1, keepdims=True)
        centered = W_in[:, members] - centroid
        U, _, _ = np.linalg.svd(centered, full_matrices=False)
        k = min(N_COMPONENTS, U.shape[1])
        bases[g_idx, :k] = U[:, :k].T.astype(np.float32)
    return bases


def _compute_group_centers(W_in: np.ndarray, group_members: list[np.ndarray]) -> np.ndarray:
    """Compute group centroids from a single W_in snapshot.

    Args:
        W_in: (d_model, d_mlp) weight matrix
        group_members: list of member index arrays, one per group

    Returns:
        (n_groups, d_model) float32 — centroid per group
    """
    d_model = W_in.shape[0]
    n_groups = len(group_members)
    centers = np.zeros((n_groups, d_model), dtype=np.float32)
    for g_idx, members in enumerate(group_members):
        centers[g_idx] = W_in[:, members].mean(axis=1).astype(np.float32)
    return centers


def _compute_all_projections(
    W_ins: list[np.ndarray],
    group_bases: np.ndarray,
    group_centers: np.ndarray,
    group_members: list[np.ndarray],
    d_mlp: int,
) -> np.ndarray:
    """Project all neurons at each epoch onto their group's PCA basis.

    Projections are centered by the final-epoch group centroid so the ring
    structure is origin-centered at the final epoch and trajectories show
    absolute movement relative to that fixed reference.

    Args:
        W_ins: list of (d_model, d_mlp) snapshots, one per epoch
        group_bases: (n_groups, 3, d_model) final-epoch PCA bases
        group_centers: (n_groups, d_model) final-epoch group centroids
        group_members: list of member index arrays, one per group
        d_mlp: total number of MLP neurons

    Returns:
        (n_epochs, d_mlp, 3) float32 — NaN for ungrouped neurons
    """
    n_epochs = len(W_ins)
    projections = np.full((n_epochs, d_mlp, 3), np.nan, dtype=np.float32)
    for ep_idx, W_in in enumerate(W_ins):
        for g_idx, members in enumerate(group_members):
            centered = W_in[:, members] - group_centers[g_idx, :, np.newaxis]
            projs = group_bases[g_idx] @ centered  # (3, n_members)
            projections[ep_idx, members] = projs.T.astype(np.float32)
    return projections


def _empty_result(epochs: list[int]) -> dict[str, np.ndarray]:
    n = len(epochs)
    return {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "pc_var": np.empty((n, 0, N_COMPONENTS), dtype=np.float32),
        "mean_spread": np.empty((n, 0), dtype=np.float32),
        "group_bases": np.empty((0, N_COMPONENTS, 0), dtype=np.float32),
        "group_centers": np.empty((0, 0), dtype=np.float32),
        "projections": np.empty((n, 0, 3), dtype=np.float32),
        "neuron_group_idx": np.array([], dtype=np.int32),
        "epochs": np.array(epochs, dtype=np.int32),
        "centroid_traj": np.empty((n, 0, 0), dtype=np.float32),
        "centroid_pca_coords": np.empty((n, 0, N_COMPONENTS), dtype=np.float32),
        "centroid_pca_var": np.zeros(N_COMPONENTS, dtype=np.float32),
        "centroid_pca_basis": np.empty((N_COMPONENTS, 0), dtype=np.float32),
    }
