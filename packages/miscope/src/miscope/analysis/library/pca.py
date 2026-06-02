"""PCA + raw-SVD primitives.

PCA modes (mean-centered SVD):
    - :func:`pca` — single sample set.
    - :func:`pca_summary` — one basis fit across a stack of sample sets
      (also called *trajectory PCA*).
    - :func:`pca_rolling` — windowed PCA across the sample axis.

Raw SVD (no centering):
    - :func:`compute_svd` — SVD of a matrix as a linear map (e.g. weight
      matrices), where the matrix itself is the object of study rather than
      a sample distribution. Returns :class:`miscope.core.svd.SVDResult`.

All routines use :func:`numpy.linalg.svd`. Sign convention is whatever NumPy
returns; consumers handle sign flips downstream.

Pure-input contract: functions take ``np.ndarray`` (or sequences of arrays)
and return typed result objects. No knowledge of ``Variant``, ``Epoch``, or
``Site``.
"""

from collections.abc import Sequence

import numpy as np

from miscope.core.pca import PCAResult
from miscope.core.svd import SVDResult


def _canonicalize_svd_sign(
    U: np.ndarray, Vt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fix the sign gauge of an SVD so the basis is reproducible everywhere.

    WHY THIS EXISTS (surfaced during REQ_133, implemented here; see also the
    REQ_134 byte-regression baseline). ``np.linalg.svd`` is deterministic and
    seedless, but a singular pair ``(u_i, v_i)`` is only defined up to a shared
    sign — ``(-u_i, -v_i)`` is an equally valid decomposition. Which sign LAPACK
    returns is NOT stable across BLAS/LAPACK builds, and it flips under tiny
    floating-point perturbations when singular values are near-degenerate. Left
    uncanonicalized, the stored ``basis``/``projections`` (and every downstream
    consumer, plus the byte-regression checksum) carry an environment-dependent
    sign: the p101 ``global_centroid_pca``/``parameter_trajectory`` artifacts
    drifted purely by sign across environments (basis ``max_rel == 2.0``, i.e.
    ``v`` vs ``-v``) while their eigenvalues matched to ~1e-15. The fix belongs
    at the lens, not at compare-time, so "the PCA/SVD basis of this object" is a
    single canonical answer for the artifact and all of its readers.

    Convention (matches scikit-learn's ``svd_flip``, v-based): for each
    component, flip ``(u_i, v_i)`` together so the largest-magnitude entry of the
    right vector ``v_i`` is positive. Singular values — and therefore
    eigenvalues/explained_variance/rank/center — are sign-invariant and untouched.

    This resolves sign flips, NOT rotation within a truly degenerate
    (equal-singular-value) subspace; none has been observed, but a ``basis`` that
    drifts with ``max_rel != 2.0`` would point at that and need a separate tie-break.
    """
    k = min(U.shape[1], Vt.shape[0])  # number of paired components (thin or full SVD)
    if k == 0:
        return U, Vt
    max_loading_idx = np.argmax(np.abs(Vt[:k]), axis=1)
    signs = np.sign(Vt[np.arange(k), max_loading_idx])
    signs[signs == 0] = 1.0  # a zero vector has no preferred sign — leave it be
    U = U.copy()
    Vt = Vt.copy()
    U[:, :k] *= signs
    Vt[:k] *= signs[:, np.newaxis]
    return U, Vt


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
    # Pin the sign gauge so the basis/projections are reproducible across
    # environments (see _canonicalize_svd_sign for the full rationale).
    U, Vt = _canonicalize_svd_sign(U, Vt)

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


def compute_svd(matrix: np.ndarray, full_matrices: bool = False) -> SVDResult:
    """Raw (non-centered) SVD of a matrix.

    For matrices where the linear map itself is the object of study —
    e.g. weight matrices — rather than a sample distribution. No mean
    centering is applied. Use :func:`pca` instead when the input rows
    are samples drawn from a distribution.

    Args:
        matrix: 2D ``(m, n)`` array.
        full_matrices: Passed through to :func:`numpy.linalg.svd`. Default
            ``False`` returns thin-SVD shapes: ``U: (m, k)``, ``S: (k,)``,
            ``Vt: (k, n)`` with ``k = min(m, n)``.

    Returns:
        :class:`SVDResult` with singular values, left/right vectors, and
        numerical rank.
    """
    if matrix.ndim != 2:
        raise ValueError(f"compute_svd expects 2D input, got shape {matrix.shape}")

    U, S, Vt = np.linalg.svd(matrix, full_matrices=full_matrices)
    # Same sign-gauge canonicalization as pca(): weight_spectra stores these
    # singular vectors, so they must be reproducible across environments
    # (see _canonicalize_svd_sign for the full rationale).
    U, Vt = _canonicalize_svd_sign(U, Vt)

    if S.size > 0 and S[0] > 0:
        m, n = matrix.shape
        tol = max(m, n) * np.finfo(S.dtype).eps * float(S[0])
        rank = int((S > tol).sum())
    else:
        rank = 0

    return SVDResult(
        singular_values=S,
        left_vectors=U,
        right_vectors=Vt,
        rank=rank,
    )
