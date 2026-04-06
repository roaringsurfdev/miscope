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

# Predefined component groups for trajectory/velocity analysis
COMPONENT_GROUPS = {
    "embedding": ["W_E", "W_pos", "W_U"],
    "attention": ["W_Q", "W_K", "W_V", "W_O"],
    "mlp": ["W_in", "W_out"],
}


def extract_parameter_snapshot(
    bundle: ActivationBundle,
) -> dict[str, np.ndarray]:
    """Extract all trainable weight matrices from a bundle.

    Returns dict mapping weight matrix names to numpy arrays
    in their original shapes.

    Args:
        bundle: ActivationBundle providing weight access via bundle.weight(name).

    Returns:
        Dict with keys from WEIGHT_MATRIX_NAMES, values are numpy arrays.
    """
    return {name: _to_numpy(bundle.weight(name)) for name in WEIGHT_MATRIX_NAMES}


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
