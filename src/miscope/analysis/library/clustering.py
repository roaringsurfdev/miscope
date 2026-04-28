"""Clustering metrics primitives.

Pure-input contract: functions take ``np.ndarray`` and return arrays or
typed scalars. No knowledge of ``Variant``, ``Epoch``, or ``Site``.

Class-dimensionality routes through :func:`miscope.analysis.library.pca.pca`
rather than re-implementing eigendecomposition — the participation ratio
is a property of the PCA result, not a separate concept.
"""

from typing import NamedTuple

import numpy as np

from miscope.analysis.library.pca import pca


class FisherDiscriminant(NamedTuple):
    """Summary statistics of pairwise Fisher discriminant ratios.

    Attributes:
        mean: Mean Fisher ratio across all class pairs.
        min: Minimum Fisher ratio (worst-case class separability).
    """

    mean: float
    min: float


def compute_class_centroids(
    samples: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """Mean sample vector per class.

    Args:
        samples: ``(n_samples, n_features)`` array.
        labels: ``(n_samples,)`` integer class labels.
        n_classes: Number of distinct classes. ``None`` infers from
            ``labels.max() + 1``.

    Returns:
        ``(n_classes, n_features)`` centroid matrix. Classes with no
        samples have a zero row.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {samples.shape}")
    if labels.shape != (samples.shape[0],):
        raise ValueError(f"labels shape {labels.shape} does not match n_samples {samples.shape[0]}")

    if n_classes is None:
        n_classes = int(labels.max()) + 1 if labels.size > 0 else 0

    n_features = samples.shape[1]
    sums = np.zeros((n_classes, n_features), dtype=samples.dtype)
    np.add.at(sums, labels, samples)
    counts = np.bincount(labels, minlength=n_classes).astype(samples.dtype)
    counts = np.maximum(counts, 1)
    return sums / counts[:, np.newaxis]


def compute_class_radii(
    samples: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """RMS distance from centroid for each class.

    Args:
        samples: ``(n_samples, n_features)`` array.
        labels: ``(n_samples,)`` integer class labels.
        centroids: ``(n_classes, n_features)`` precomputed centroids.

    Returns:
        ``(n_classes,)`` per-class RMS radii. Classes with no samples
        have radius 0.
    """
    n_classes = centroids.shape[0]
    diffs = samples - centroids[labels]
    sq_dists = np.sum(diffs**2, axis=1)
    sum_sq = np.bincount(labels, weights=sq_dists, minlength=n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    return np.sqrt(sum_sq / counts)


def compute_fisher_discriminant(
    samples: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray | None = None,
) -> FisherDiscriminant:
    """Pairwise Fisher discriminant ratios across all class pairs.

    For each pair (r, s):

        J(r, s) = ||μ_r - μ_s||² / (σ_r² + σ_s²)

    where σ_r² is the mean within-class squared deviation for class r.

    Args:
        samples: ``(n_samples, n_features)`` array.
        labels: ``(n_samples,)`` integer class labels.
        centroids: ``(n_classes, n_features)`` precomputed centroids.
            ``None`` computes them internally.

    Returns:
        :class:`FisherDiscriminant` with ``mean`` and ``min`` across pairs.
        Both are 0.0 when no class pair has nonzero within-class variance.
    """
    if centroids is None:
        centroids = compute_class_centroids(samples, labels)

    n_classes = centroids.shape[0]

    diffs = samples - centroids[labels]
    sq_dists = np.sum(diffs**2, axis=1)
    sum_sq = np.bincount(labels, weights=sq_dists, minlength=n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)
    variances = sum_sq / counts

    centroid_diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    pairwise_sq_dists = np.sum(centroid_diffs**2, axis=2)
    pairwise_within = variances[:, np.newaxis] + variances[np.newaxis, :]

    r_idx, s_idx = np.triu_indices(n_classes, k=1)
    between = pairwise_sq_dists[r_idx, s_idx]
    within = pairwise_within[r_idx, s_idx]

    valid = within > 0
    if not valid.any():
        return FisherDiscriminant(mean=0.0, min=0.0)

    fisher_values = np.where(valid, between / np.maximum(within, 1e-12), np.inf)
    finite_mask = np.isfinite(fisher_values)
    if not finite_mask.any():
        return FisherDiscriminant(mean=0.0, min=0.0)

    finite_values = fisher_values[finite_mask]
    return FisherDiscriminant(
        mean=float(finite_values.mean()),
        min=float(finite_values.min()),
    )


def compute_class_dimensionality(
    samples: np.ndarray,
    labels: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """Effective dimensionality (participation ratio) per class.

    Routes through :func:`miscope.analysis.library.pca.pca` for each class
    rather than re-implementing eigendecomposition. Equals 1 if all
    variance is on one axis, ``n_features`` if uniform across all
    dimensions.

    Args:
        samples: ``(n_samples, n_features)`` array.
        labels: ``(n_samples,)`` integer class labels.
        n_classes: Number of distinct classes. ``None`` infers from
            ``labels.max() + 1``.

    Returns:
        ``(n_classes,)`` participation ratio per class. Classes with
        fewer than 2 samples have dimensionality 0.
    """
    if n_classes is None:
        n_classes = int(labels.max()) + 1 if labels.size > 0 else 0

    dims = np.zeros(n_classes)
    for r in range(n_classes):
        class_samples = samples[labels == r]
        if class_samples.shape[0] < 2:
            continue
        dims[r] = pca(class_samples).participation_ratio
    return dims


def compute_center_spread(centroids: np.ndarray) -> float:
    """RMS distance of centroids from their global mean.

    Distinct from :attr:`PCAResult.spread`: the latter uses Bessel-corrected
    variance (``n-1``); this uses ``n``. Use this for class-level spread
    reporting; use ``PCAResult.spread`` when working in the PCA basis.

    Args:
        centroids: ``(n_classes, n_features)`` centroid matrix.

    Returns:
        Scalar RMS spread.
    """
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be 2D, got shape {centroids.shape}")
    global_centroid = centroids.mean(axis=0)
    diffs = centroids - global_centroid
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
