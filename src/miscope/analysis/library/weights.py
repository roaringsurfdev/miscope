"""Weight matrix extraction utilities.

Provides functions for extracting and analyzing model weight matrices
across training checkpoints. Used by parameter trajectory and
effective dimensionality analyzers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle

# Weight matrices to extract, in consistent order
WEIGHT_MATRIX_NAMES = [
    "W_E",
    "W_pos",
    "W_Q",
    "W_K",
    "W_V",
    "W_O",
    "W_in",
    "W_out",
    "W_U",
]

# Architecture-specific weight names not in the transformer set.
# extract_parameter_snapshot tries these in addition to WEIGHT_MATRIX_NAMES.
ARCH_WEIGHT_NAMES: dict[str, list[str]] = {
    "modulo_addition_embed_mlp": ["embed_a", "embed_b"],
}

# Predefined component groups for trajectory/velocity analysis
COMPONENT_GROUPS = {
    "embedding": ["W_E", "W_pos", "W_U"],
    "attention": ["W_Q", "W_K", "W_V", "W_O"],
    "mlp": ["W_in", "W_out"],
    "learned_embeddings": ["embed_a", "embed_b"],  # Learned-embedding MLP only
}


def extract_parameter_snapshot(
    bundle: ActivationBundle,
) -> dict[str, np.ndarray]:
    """Extract all trainable weight matrices from a bundle.

    Returns dict mapping weight matrix names to numpy arrays
    in their original shapes. Tries both the standard transformer names
    (WEIGHT_MATRIX_NAMES) and architecture-specific names (ARCH_WEIGHT_NAMES),
    silently skipping any that the bundle doesn't expose.

    Args:
        bundle: ActivationBundle providing weight access via bundle.weight(name).

    Returns:
        Dict with available weight matrix names as keys, numpy arrays as values.
    """
    all_names = list(WEIGHT_MATRIX_NAMES)
    for extra in ARCH_WEIGHT_NAMES.values():
        all_names.extend(extra)

    result = {}
    for name in all_names:
        try:
            result[name] = _to_numpy(bundle.weight(name))
        except KeyError:
            pass  # Architecture doesn't have this weight
    return result


def extract_neuron_weight_matrix(snapshot: dict) -> np.ndarray:
    """Extract a (d_space, M) neuron weight matrix from a parameter snapshot.

    Returns a column-per-neuron matrix suitable for PCA and geometric analysis,
    dispatching on architecture based on snapshot contents.

      - Transformer: W_in is (d_model, d_mlp) — columns are already neuron vectors.
      - MLP:         W_in is (d_hidden, 2p) — rows are neuron vectors; transpose to
                     (2p, d_hidden) so columns are neuron vectors.

    Args:
        snapshot: parameter_snapshot dict. Presence of 'W_E' signals transformer.

    Returns:
        Array of shape (d_space, M) where each column is one neuron's weight vector.
    """
    W_in = snapshot["W_in"]
    if "W_E" in snapshot:
        return W_in  # (d_model, d_mlp) — columns are neuron vectors
    else:
        return W_in.T  # (2p, d_hidden) — transposed so columns are neuron vectors


ATTENTION_MATRICES = {"W_Q", "W_K", "W_V", "W_O"}


def compute_participation_ratio(
    singular_values: np.ndarray,
) -> float | np.ndarray:
    """Compute participation ratio from singular values.

    PR = (sum(s))^2 / sum(s^2)

    Equals 1.0 when one singular value dominates (rank-1),
    equals n when all n singular values are equal (full rank).

    Args:
        singular_values: 1D array → scalar result, or 2D array
            (e.g., n_heads x d_head) → 1D array of per-row PRs.

    Returns:
        Scalar float for 1D input, 1D numpy array for 2D input.
    """
    if singular_values.ndim == 1:
        s = singular_values
        sum_s = s.sum()
        sum_s2 = (s**2).sum()
        return float(sum_s**2 / sum_s2) if sum_s2 > 0 else 0.0

    # 2D: compute per-row
    sum_s = singular_values.sum(axis=1)
    sum_s2 = (singular_values**2).sum(axis=1)
    mask = sum_s2 > 0
    result = np.zeros(singular_values.shape[0])
    result[mask] = sum_s[mask] ** 2 / sum_s2[mask]
    return result


def compute_weight_singular_values(
    bundle: ActivationBundle,
) -> dict[str, np.ndarray]:
    """Compute singular values of all trainable weight matrices.

    For attention matrices (W_Q, W_K, W_V, W_O), SVD is computed
    per head. Each head's matrix is an independent subspace.

    Args:
        bundle: ActivationBundle providing weight access.

    Returns:
        Dict mapping "sv_{name}" to numpy arrays of singular values.
        Attention matrices: shape (n_heads, d_head).
        Other matrices: shape (min(rows, cols),).
    """
    snapshot = extract_parameter_snapshot(bundle)
    result = {}

    for name in WEIGHT_MATRIX_NAMES:
        if name not in snapshot:
            continue  # Weight not available for this architecture
        matrix = snapshot[name]
        key = f"sv_{name}"

        if name in ATTENTION_MATRICES:
            # Shape: (n_heads, d_model, d_head) or (n_heads, d_head, d_model)
            n_heads = matrix.shape[0]
            head_svs = []
            for h in range(n_heads):
                sv = np.linalg.svd(matrix[h], compute_uv=False)
                head_svs.append(sv)
            result[key] = np.array(head_svs)
        else:
            result[key] = np.linalg.svd(matrix, compute_uv=False)

    return result


def _to_numpy(tensor) -> np.ndarray:
    """Convert a parameter tensor to numpy, detaching from graph."""
    return tensor.detach().cpu().numpy()
