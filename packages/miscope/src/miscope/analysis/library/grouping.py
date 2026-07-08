"""REQ_118: Neuron grouping primitive.

Pure-input contract: functions take ``np.ndarray`` and return canonical
types from :mod:`miscope.core.grouping`. No knowledge of ``Variant``,
``Epoch``, or ``Site`` — analyzer wrappers handle that layer.

Two methods at v1:

- ``"kmeans"`` — data-driven clustering via :func:`scipy.cluster.vq.kmeans2`.
  Used for the universal path when no family-supplied basis exists.
- ``"argmax_by_basis"`` — assignment by dominant component along the
  feature axis. Used when features are already a basis projection (e.g.,
  per-neuron Fourier coefficients on modular addition). Produces a
  per-neuron confidence score equal to the dominant-component fraction
  of the row's L1 mass; neurons whose confidence falls below
  ``confidence_threshold`` are marked as ``UNASSIGNED``.

Group summary metrics reuse :mod:`miscope.analysis.library.clustering`
rather than re-implementing centroid / radius / Fisher computations.
"""

import numpy as np
from scipy.cluster.vq import kmeans2

from miscope.analysis.library.clustering import (
    compute_class_centroids,
    compute_class_radii,
    compute_fisher_discriminant,
)
from miscope.core.grouping import UNASSIGNED, GroupAssignment, GroupSummary

_VALID_METHODS = {"kmeans", "argmax_by_basis"}


def group_neurons(
    features: np.ndarray,
    n_groups: int,
    method: str = "kmeans",
    feature_basis_name: str = "features",
    confidence_threshold: float | None = None,
    random_state: int = 42,
) -> GroupAssignment:
    """Cluster neurons by feature similarity.

    Args:
        features: ``(n_neurons, n_features)`` array. For ``"kmeans"``,
            features can be any basis. For ``"argmax_by_basis"``,
            features are interpreted as a basis projection and
            ``n_features`` should equal the number of basis components.
        n_groups: Number of distinct groups to assign neurons to. For
            ``"argmax_by_basis"``, the assignment range is determined
            by ``np.argmax`` over ``n_features`` columns; ``n_groups``
            in the returned type reflects the actual number of groups
            observed (which may be less than ``n_features`` if some
            basis components are dominant for no neuron).
        method: ``"kmeans"`` or ``"argmax_by_basis"``.
        feature_basis_name: Human-readable identifier for the feature
            space, propagated to the result. Examples:
            ``"weight_signature"``, ``"fourier_w_in"``,
            ``"activation_profile"``.
        confidence_threshold: For ``"argmax_by_basis"`` only. Neurons
            whose dominant-component fraction falls below this value
            are marked ``UNASSIGNED``. ``None`` means accept all
            neurons regardless of confidence.
        random_state: Seed for ``"kmeans"`` initialization.

    Returns:
        ``GroupAssignment`` with ``n_neurons`` entries.

    Raises:
        ValueError: If ``method`` is unknown, ``features`` is not 2D,
            or ``n_groups < 1``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown method '{method}'; valid: {sorted(_VALID_METHODS)}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1 (got {n_groups})")

    if method == "kmeans":
        return _group_kmeans(features, n_groups, feature_basis_name, random_state)
    return _group_argmax_by_basis(features, feature_basis_name, confidence_threshold)


def group_neurons_summary(
    features: np.ndarray,
    assignment: GroupAssignment,
) -> GroupSummary:
    """Per-group summary statistics from a `GroupAssignment`.

    Computes centroids, radii, and Fisher separability across groups,
    excluding unassigned neurons. Composes existing clustering primitives.

    Args:
        features: ``(n_neurons, n_features)`` array — the same feature
            matrix that produced ``assignment``.
        assignment: Output of :func:`group_neurons`.

    Returns:
        ``GroupSummary`` over the assigned neurons. If all neurons are
        unassigned, returns a degenerate summary with empty centroids
        and zeroed metrics.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")

    assigned_mask = assignment.assignments != UNASSIGNED
    n_unassigned = int((~assigned_mask).sum())

    if not assigned_mask.any():
        # Degenerate case: every neuron unassigned.
        return GroupSummary(
            centroids=np.empty((0, features.shape[1]), dtype=np.float64),
            radii=np.empty(0, dtype=np.float64),
            n_per_group=np.empty(0, dtype=np.int64),
            n_unassigned=n_unassigned,
            fisher_min=0.0,
            fisher_mean=0.0,
            dispersion=0.0,
        )

    assigned_features = features[assigned_mask]
    assigned_labels = assignment.assignments[assigned_mask].astype(np.int64)
    n_groups = int(assignment.n_groups)

    centroids = compute_class_centroids(assigned_features, assigned_labels, n_classes=n_groups)
    radii = compute_class_radii(assigned_features, assigned_labels, centroids)
    n_per_group = np.bincount(assigned_labels, minlength=n_groups).astype(np.int64)
    fisher = compute_fisher_discriminant(assigned_features, assigned_labels, centroids)
    dispersion = float(radii.mean()) if radii.size > 0 else 0.0

    return GroupSummary(
        centroids=centroids,
        radii=radii,
        n_per_group=n_per_group,
        n_unassigned=n_unassigned,
        fisher_min=float(fisher.min),
        fisher_mean=float(fisher.mean),
        dispersion=dispersion,
    )


# --- Method implementations ---


def _group_kmeans(
    features: np.ndarray,
    n_groups: int,
    feature_basis_name: str,
    random_state: int,
) -> GroupAssignment:
    """KMeans via scipy.cluster.vq.kmeans2 with deterministic init."""
    rng = np.random.default_rng(random_state)
    # kmeans2 requires float64; uses '++' initialization for k-means++.
    data = np.asarray(features, dtype=np.float64)
    _, assignments = kmeans2(
        data,
        n_groups,
        minit="++",
        seed=rng,  # pyright: ignore[reportCallIssue]
    )
    return GroupAssignment(
        assignments=assignments.astype(np.int64),
        n_groups=n_groups,
        method="kmeans",
        feature_basis_name=feature_basis_name,
        confidence=None,
    )


def _group_argmax_by_basis(
    features: np.ndarray,
    feature_basis_name: str,
    confidence_threshold: float | None,
) -> GroupAssignment:
    """Assign each neuron to its argmax basis component.

    Confidence per neuron is the dominant component's *variance fraction*:
    ``max(features[i]^2) / sum(features[i]^2)`` — i.e., the fraction of
    the row's squared L2 mass concentrated in the dominant component.
    This matches the convention used elsewhere in the project (e.g.,
    `compute_frequency_variance_fractions`) and makes the conventional
    0.7 commitment threshold meaningful here too. Neurons whose confidence
    falls below ``confidence_threshold`` (when supplied) are marked
    ``UNASSIGNED``.
    """
    sq_features = features.astype(np.float64) ** 2
    row_sums = sq_features.sum(axis=1)
    # Avoid divide-by-zero for all-zero rows; their confidence is 0.
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    dominant = sq_features.max(axis=1)
    confidence = (dominant / safe_sums).astype(np.float64)
    confidence = np.where(row_sums > 0, confidence, 0.0)

    # Argmax over magnitude (sign-invariant) gives the dominant component.
    assignments = np.argmax(np.abs(features), axis=1).astype(np.int64)
    if confidence_threshold is not None:
        assignments = np.where(confidence >= confidence_threshold, assignments, UNASSIGNED)

    n_groups = int(features.shape[1])
    return GroupAssignment(
        assignments=assignments,
        n_groups=n_groups,
        method="argmax_by_basis",
        feature_basis_name=feature_basis_name,
        confidence=confidence,
    )
