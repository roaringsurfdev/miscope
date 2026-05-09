"""REQ_118: Canonical types for the neuron grouping primitive.

`GroupAssignment` and `GroupSummary` are pure data types — no `Variant`,
`Epoch`, or `Site` knowledge. Analyzer wrappers consume the pure forms
and add variant-coupled metadata to the artifact (reference epoch, feature
source, etc.).
"""

from typing import NamedTuple

import numpy as np

# Sentinel for unassigned neurons (e.g., neurons whose dominant feature
# falls below the confidence threshold for argmax_by_basis methods).
UNASSIGNED = -1


class GroupAssignment(NamedTuple):
    """Per-neuron group assignment with provenance metadata.

    Attributes:
        assignments: ``(n_neurons,)`` integer array. Values in
            ``[0, n_groups)`` indicate group membership; the sentinel
            ``UNASSIGNED`` (= -1) marks neurons that did not meet the
            assignment criterion (e.g., below confidence threshold).
        n_groups: Total number of distinct groups represented.
        method: Name of the method used (e.g., ``"kmeans"``,
            ``"argmax_by_basis"``, ``"fourier_argmax"``).
        feature_basis_name: Human-readable identifier for what the
            features were (e.g., ``"weight_signature"``, ``"fourier_w_in"``,
            ``"activation_profile"``). Lets downstream consumers know
            which space the grouping was derived in.
        confidence: ``(n_neurons,)`` float array, optional. For methods
            that produce a confidence score (e.g., ``argmax_by_basis``
            stores the dominant-component fraction), this carries the
            per-neuron value. ``None`` for methods that produce hard
            assignments without a meaningful confidence (e.g., kmeans
            in its default form).
    """

    assignments: np.ndarray
    n_groups: int
    method: str
    feature_basis_name: str
    confidence: np.ndarray | None


class GroupSummary(NamedTuple):
    """Per-group summary statistics derived from a `GroupAssignment`.

    Attributes:
        centroids: ``(n_groups, n_features)`` array of per-group mean
            feature vectors. Excludes unassigned neurons.
        radii: ``(n_groups,)`` per-group RMS distance from centroid.
        n_per_group: ``(n_groups,)`` int — neuron count per group.
        n_unassigned: scalar int — count of neurons with sentinel label.
        fisher_min: scalar float — minimum pairwise Fisher discriminant
            ratio across groups (worst-case separability).
        fisher_mean: scalar float — mean pairwise Fisher discriminant.
        dispersion: scalar float — mean of the per-group radii. A
            single-number summary of within-group spread.
    """

    centroids: np.ndarray
    radii: np.ndarray
    n_per_group: np.ndarray
    n_unassigned: int
    fisher_min: float
    fisher_mean: float
    dispersion: float
