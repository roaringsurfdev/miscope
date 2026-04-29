"""PCA primitives.

Three modes:
    - :func:`pca` — single sample set.
    - :func:`pca_summary` — one basis fit across a stack of sample sets
      (also called *trajectory PCA*).
    - :func:`pca_rolling` — windowed PCA across the sample axis.

All modes use mean-centered SVD via :func:`numpy.linalg.svd`. Sign convention
is whatever ``np.linalg.svd`` returns; consumers handle sign flips downstream.

Pure-input contract: functions take ``np.ndarray`` (or sequences of arrays)
and return :class:`miscope.core.pca.PCAResult`. No knowledge of ``Variant``,
``Epoch``, or ``Site``.
"""

from collections.abc import Sequence

import numpy as np

from miscope.core.pca import PCAResult


def pca(X: np.ndarray, n_components: int | None = None) -> PCAResult:
    """Fit PCA on a single sample set via mean-centered SVD.

    Args:
        X: ``(n_samples, n_features)`` data matrix.
        n_components: Number of components to retain. ``None`` retains
            ``min(n_samples, n_features)``.

    Returns:
        :class:`PCAResult` with basis, projections, and derived metrics.
    """
    if X.ndim != 2:
        raise ValueError(f"pca expects 2D input, got shape {X.shape}")

    n_samples, n_features = X.shape
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    elif n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds max possible {max_components} "
            f"for input shape {X.shape}"
        )

    # Promote to float64 for the centering and SVD: float32 ``X.mean`` and
    # subtraction produce spurious noise on bit-identical samples (e.g.
    # resid_pre at a fixed position) that propagates into singular values
    # and breaks downstream metrics.
    X = np.asarray(X, dtype=np.float64)
    center = X.mean(axis=0)
    Xc = X - center
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Full spectrum: used for ratio normalization and full-spectrum scalars
    # (participation_ratio, rank, spread). Truncation is a presentation choice;
    # data properties don't depend on it.
    denom = max(n_samples - 1, 1)
    all_eigenvalues = S**2 / denom
    total_var = float(all_eigenvalues.sum())

    # Top-k truncation
    singular_values = S[:n_components]
    basis_vectors = Vt[:n_components]
    projections = U[:, :n_components] * singular_values
    eigenvalues = all_eigenvalues[:n_components]

    if total_var > 0:
        explained_variance_ratio = eigenvalues / total_var
    else:
        explained_variance_ratio = np.zeros_like(eigenvalues)

    sq_sum = float((all_eigenvalues**2).sum())
    if sq_sum > 0:
        participation_ratio = total_var**2 / sq_sum
    else:
        participation_ratio = 0.0

    if S.size > 0 and S[0] > 0:
        tol = max(n_samples, n_features) * np.finfo(float).eps * float(S[0])
        rank = int((S > tol).sum())
    else:
        rank = 0

    spread = float(np.sqrt(total_var))

    return PCAResult(
        singular_values=singular_values,
        eigenvalues=eigenvalues,
        basis_vectors=basis_vectors,
        projections=projections,
        explained_variance=eigenvalues,
        explained_variance_ratio=explained_variance_ratio,
        participation_ratio=participation_ratio,
        rank=rank,
        spread=spread,
        center=center,
    )


def pca_summary(
    sample_sets: Sequence[np.ndarray] | np.ndarray,
    n_components: int | None = None,
) -> PCAResult:
    """Fit a single PCA basis across multiple sample sets.

    Pools all sample sets into one matrix, fits PCA once, and projects each
    set into the shared coordinate frame. Consumers reshape ``projections``
    back into per-set form using the input shapes.

    Args:
        sample_sets: Either a list of 2D ``(n_samples, n_features)`` arrays
            (sets may have different sample counts), or a 3D
            ``(n_sets, n_samples_per_set, n_features)`` array for uniform sets.
        n_components: Number of components to retain. ``None`` retains
            the maximum possible.

    Returns:
        :class:`PCAResult`. The basis is shared; projections are stacked
        in input order.
    """
    if isinstance(sample_sets, np.ndarray):
        if sample_sets.ndim == 3:
            n_features = sample_sets.shape[-1]
            stacked = sample_sets.reshape(-1, n_features)
        elif sample_sets.ndim == 2:
            stacked = sample_sets
        else:
            raise ValueError(
                f"pca_summary array input must be 2D or 3D, got shape {sample_sets.shape}"
            )
    else:
        sets = list(sample_sets)
        if not sets:
            raise ValueError("pca_summary requires at least one sample set")
        feature_dims = {s.shape[-1] for s in sets}
        if len(feature_dims) != 1:
            raise ValueError(f"All sample sets must share feature dimension; got {feature_dims}")
        stacked = np.concatenate(sets, axis=0)

    return pca(stacked, n_components=n_components)


def pca_rolling(
    X: np.ndarray,
    window_size: int,
    stride: int = 1,
    n_components: int | None = None,
) -> list[PCAResult]:
    """Fit PCA on each sliding window over the sample axis.

    Window starts step at indices ``0, stride, 2*stride, ...`` while
    ``start + window_size <= n_samples``.

    Args:
        X: ``(n_samples, n_features)`` data matrix; samples assumed ordered.
        window_size: Number of consecutive samples per window.
        stride: Step between window starts. Default 1.
        n_components: Number of components per window. ``None`` retains the
            maximum possible per-window.

    Returns:
        List of :class:`PCAResult`, one per window, in start-index order.
    """
    if X.ndim != 2:
        raise ValueError(f"pca_rolling expects 2D input, got shape {X.shape}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if window_size > X.shape[0]:
        raise ValueError(f"window_size={window_size} exceeds n_samples={X.shape[0]}")

    n_samples = X.shape[0]
    return [
        pca(X[start : start + window_size], n_components=n_components)
        for start in range(0, n_samples - window_size + 1, stride)
    ]
