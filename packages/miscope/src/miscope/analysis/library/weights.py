"""Weight matrix extraction utilities.

Provides functions for extracting and analyzing model weight matrices
across training checkpoints. Used by parameter trajectory and
effective dimensionality analyzers.

REQ_114 migration
-----------------
``extract_parameter_snapshot`` and ``compute_weight_singular_values``
now take a :class:`miscope.architectures.HookedModel` and read weights
via canonical names internally. The returned dict still uses legacy
short names (``W_E``, ``W_in``, ...) so downstream consumers (snapshot
artifact schema, ``extract_neuron_weight_matrix`` dispatch logic) are
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library.pca import compute_svd
from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

if TYPE_CHECKING:
    from miscope.architectures import HookedModel

# Weight matrices to extract, in consistent order. Legacy short names are
# preserved as the artifact / consumer-facing identifiers.
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

ATTENTION_MATRICES = {"W_Q", "W_K", "W_V", "W_O"}


# Map legacy short name → canonical path. Attention matrices are
# represented as their per-head stacked tensor; the canonical path
# resolves to the (n_heads, ...) tensor on TL's HookedTransformer.
# For HookedModel subclasses without these surfaces (MLPs), the lookup
# raises and we silently skip (matching the pre-migration behavior).
_LEGACY_TO_CANONICAL: dict[str, str] = {
    "W_E": canonical_weights.EMBED_WEIGHT,
    "W_pos": canonical_weights.POS_EMBED_WEIGHT,
    "W_U": canonical_weights.UNEMBED_WEIGHT,
    "W_Q": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_Q),
    "W_K": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_K),
    "W_V": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_V),
    "W_O": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_O),
    "W_in": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_IN),
    "W_out": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_OUT),
    "embed_a": canonical_weights.EMBED_A_PATH,
    "embed_b": canonical_weights.EMBED_B_PATH,
}


def extract_parameter_snapshot(model: HookedModel) -> dict[str, np.ndarray]:
    """Extract all trainable weight matrices from a HookedModel.

    Returns dict mapping legacy short names (``W_E``, ``W_Q``, ``W_in``,
    ``embed_a``, ...) to numpy arrays in their original shapes. Tries
    every name in :data:`WEIGHT_MATRIX_NAMES` plus architecture-specific
    names from :data:`ARCH_WEIGHT_NAMES`, silently skipping any the
    model doesn't publish.

    Args:
        model: ``HookedModel`` instance providing canonical-name weight
            access via ``model.get_weight(...)``.

    Returns:
        Dict with available legacy weight names as keys and numpy
        arrays as values.
    """
    all_names = list(WEIGHT_MATRIX_NAMES)
    for extra in ARCH_WEIGHT_NAMES.values():
        all_names.extend(extra)

    result: dict[str, np.ndarray] = {}
    for name in all_names:
        canonical = _LEGACY_TO_CANONICAL.get(name)
        if canonical is None:
            continue
        try:
            tensor = model.get_weight(canonical)
        except KeyError:
            continue  # Architecture doesn't publish this weight
        result[name] = _to_numpy(tensor)
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


def compute_weight_singular_values(model: HookedModel) -> dict[str, np.ndarray]:
    """Compute singular values of all trainable weight matrices.

    For attention matrices (W_Q, W_K, W_V, W_O), SVD is computed
    per head. Each head's matrix is an independent subspace.

    Args:
        model: ``HookedModel`` providing canonical-name weight access.

    Returns:
        Dict mapping "sv_{name}" to numpy arrays of singular values.
        Attention matrices: shape (n_heads, d_head).
        Other matrices: shape (min(rows, cols),).
    """
    spectra = compute_weight_spectra(model)
    return {f"sv_{name}": sv for name, (_, sv, _) in spectra.items()}


def compute_weight_spectra(
    model: HookedModel,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute full SVD (U, S, Vt) of all trainable weight matrices.

    Routes through :func:`miscope.analysis.library.pca.compute_svd` for
    each matrix. Attention matrices are decomposed per head.

    Args:
        model: ``HookedModel`` providing canonical-name weight access.

    Returns:
        Dict mapping legacy weight name to ``(U, S, Vt)``.
        Attention matrices: each component has a leading ``n_heads`` axis.
        Other matrices: thin-SVD shapes.
    """
    snapshot = extract_parameter_snapshot(model)
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for name in WEIGHT_MATRIX_NAMES:
        if name not in snapshot:
            continue  # Weight not available for this architecture
        matrix = snapshot[name]

        if name in ATTENTION_MATRICES:
            n_heads = matrix.shape[0]
            heads = [compute_svd(matrix[h]) for h in range(n_heads)]
            u = np.array([r.left_vectors for r in heads])
            s = np.array([r.singular_values for r in heads])
            vt = np.array([r.right_vectors for r in heads])
            result[name] = (u, s, vt)
        else:
            r = compute_svd(matrix)
            result[name] = (r.left_vectors, r.singular_values, r.right_vectors)

    return result


def _to_numpy(tensor) -> np.ndarray:
    """Convert a parameter tensor to numpy, detaching from graph."""
    return tensor.detach().cpu().numpy()
