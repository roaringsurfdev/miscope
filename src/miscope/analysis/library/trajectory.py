"""Parameter trajectory analysis utilities.

Provides PCA projection, velocity, and shape quantification for parameter
space trajectories across training checkpoints.

Shape quantification (lemniscate analysis):
- compute_arc_length: cumulative arc-length along a curve
- detect_self_intersection: locate the node of a self-crossing curve
- compute_signed_loop_area: area enclosed by the loop segment
- compute_curvature_profile: κ(s) as a function of normalized arc-length

Group centroid helpers (promoted from notebooks for reuse):
- fit_centroid_pca: shared PCA basis across all groups and epochs
- normalize_per_group: z-score each group's trajectory independently
"""

import numpy as np
from sklearn.decomposition import PCA

from miscope.analysis.constants import SVD_RANDOM_STATE
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES


def flatten_snapshot(
    snapshot: dict[str, np.ndarray],
    components: list[str] | None = None,
) -> np.ndarray:
    """Flatten selected weight matrices into a single parameter vector.

    Args:
        snapshot: Per-epoch artifact dict from ParameterSnapshotAnalyzer.
        components: Weight matrix names to include. None = all.

    Returns:
        1D array of concatenated, flattened parameters.
    """
    if components is None:
        components = [k for k in WEIGHT_MATRIX_NAMES if k in snapshot]

    parts = [snapshot[k].flatten() for k in components if k in snapshot]
    return np.concatenate(parts)


def compute_pca_trajectory(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    n_components: int = 3,
) -> dict[str, np.ndarray]:
    """Compute PCA projection of parameter trajectory.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.
        n_components: Number of principal components.

    Returns:
        Dict with:
          "projections": (n_epochs, n_components) in PC space
          "explained_variance_ratio": (n_components,) fraction per PC
          "explained_variance": (n_components,) eigenvalues
    """
    vectors = np.array([flatten_snapshot(s, components) for s in snapshots])

    n_components = min(n_components, len(snapshots), vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=SVD_RANDOM_STATE)
    projections = pca.fit_transform(vectors)

    return {
        "projections": projections,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    }


def compute_parameter_velocity(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    epochs: list[int] | None = None,
) -> np.ndarray:
    """Compute parameter velocity between consecutive checkpoints.

    When epochs are provided, velocity is normalized by the epoch gap
    to give displacement per epoch. Without epochs, returns raw L2
    displacement (which is distorted by non-uniform checkpoint spacing).

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.
        epochs: Epoch numbers for each snapshot. When provided,
            velocity is divided by the epoch gap between checkpoints.

    Returns:
        1D array of length (n_epochs - 1).
        With epochs: velocity[i] = ||delta theta|| / (epoch_{i+1} - epoch_i)
        Without epochs: velocity[i] = ||delta theta||
    """
    vectors = [flatten_snapshot(s, components) for s in snapshots]

    velocities = []
    for i in range(len(vectors) - 1):
        delta = vectors[i + 1] - vectors[i]
        displacement = float(np.linalg.norm(delta))
        if epochs is not None:
            gap = epochs[i + 1] - epochs[i]
            displacement = displacement / gap if gap > 0 else 0.0
        velocities.append(displacement)

    return np.array(velocities)


# ---------------------------------------------------------------------------
# Shape quantification — lemniscate analysis
# ---------------------------------------------------------------------------


def compute_arc_length(curve: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a 2D or 3D curve.

    Args:
        curve: (n_points, d) array of coordinates.

    Returns:
        (n_points,) cumulative arc-length, starting at 0.
    """
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def detect_self_intersection(
    curve: np.ndarray,
    min_arc_sep_fraction: float = 0.15,
) -> dict:
    """Find the closest self-approach in a 2D curve — the lemniscate node.

    Scans all pairs of non-adjacent points separated by at least
    `min_arc_sep_fraction * total_arc_length` of arc, returning the pair
    with minimum Euclidean distance. This localizes the node where the curve
    crosses itself (or approaches itself most closely).

    Args:
        curve: (n_points, 2) array of 2D coordinates.
        min_arc_sep_fraction: Minimum arc-length gap between candidate pairs
            as a fraction of total arc length. Filters out adjacent segments.
            Default 0.15 (15% of total arc).

    Returns:
        Dict with:
            "node_position": (2,) midpoint of the closest-approach pair.
            "idx_pair": (i, j) index pair of the closest approach.
            "min_distance": float — Euclidean distance at closest approach.
            "arc_length": (n_points,) cumulative arc-length array.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]
    min_sep = min_arc_sep_fraction * total_arc

    n = len(curve)
    diff = curve[:, np.newaxis, :] - curve[np.newaxis, :, :]  # (n, n, 2)
    dist_sq = np.sum(diff ** 2, axis=2)  # (n, n)

    arc_sep = np.abs(arc[:, np.newaxis] - arc[np.newaxis, :])  # (n, n)
    valid = (arc_sep >= min_sep) & np.triu(np.ones((n, n), dtype=bool), k=1)

    if not valid.any():
        mid = n // 2
        return {
            "node_position": curve[mid].copy(),
            "idx_pair": (0, n - 1),
            "min_distance": float("inf"),
            "arc_length": arc,
        }

    dist_sq_masked = np.where(valid, dist_sq, np.inf)
    flat_idx = int(np.argmin(dist_sq_masked))
    i, j = divmod(flat_idx, n)

    return {
        "node_position": (curve[i] + curve[j]) / 2.0,
        "idx_pair": (int(i), int(j)),
        "min_distance": float(np.sqrt(dist_sq[i, j])),
        "arc_length": arc,
    }


def compute_signed_loop_area(
    curve: np.ndarray,
    idx_pair: tuple[int, int],
) -> float:
    """Signed area of the loop enclosed between two self-intersection indices.

    Extracts `curve[i:j+1]` and applies the shoelace formula.
    Sign reflects traversal direction: positive = CCW, negative = CW.
    Overshooters produce a larger-magnitude area; never-arrives produce ~0
    (loop not completed).

    Args:
        curve: (n_points, 2) array of 2D coordinates.
        idx_pair: (i, j) indices from detect_self_intersection.

    Returns:
        Signed area of the enclosed loop. Near zero if no loop is completed.
    """
    i, j = idx_pair
    loop = curve[i : j + 1]
    if len(loop) < 3:
        return 0.0
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_curvature_profile(
    curve: np.ndarray,
    n_norm_points: int = 100,
) -> dict:
    """Signed curvature κ as a function of normalized arc-length.

    Uses the discrete Frenet-Serret formula:
        κ = (x' · y'' − y' · x'') / (x'² + y'²)^(3/2)

    Derivatives are computed via `np.gradient` (central differences) using
    the arc-length parameterization so the result is intrinsic to curve shape
    rather than epoch spacing.  The profile is then resampled onto a uniform
    normalized arc-length grid [0, 1] for cross-variant comparison.

    Args:
        curve: (n_points, 2) array of 2D coordinates.
        n_norm_points: Number of output samples on the normalized grid.

    Returns:
        Dict with:
            "s_norm": (n_norm_points,) normalized arc-length ∈ [0, 1].
            "kappa": (n_norm_points,) curvature at each s_norm sample.
            "s_raw": (n_points,) raw cumulative arc-length.
            "kappa_raw": (n_points,) curvature at each original point.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]

    x, y = curve[:, 0], curve[:, 1]

    dx = np.gradient(x, arc)
    dy = np.gradient(y, arc)
    d2x = np.gradient(dx, arc)
    d2y = np.gradient(dy, arc)

    denom = (dx ** 2 + dy ** 2) ** 1.5
    kappa_raw = np.where(denom > 1e-12, (dx * d2y - dy * d2x) / denom, 0.0)

    s_norm_raw = arc / total_arc if total_arc > 1e-12 else np.linspace(0.0, 1.0, len(arc))
    s_norm = np.linspace(0.0, 1.0, n_norm_points)
    kappa = np.interp(s_norm, s_norm_raw, kappa_raw)

    return {
        "s_norm": s_norm,
        "kappa": kappa,
        "s_raw": arc,
        "kappa_raw": kappa_raw,
    }


# ---------------------------------------------------------------------------
# Group centroid helpers (promoted for notebook reuse)
# ---------------------------------------------------------------------------


def fit_centroid_pca(
    centroids: np.ndarray,
    n_components: int = 3,
) -> dict:
    """Shared PCA basis fitted across all group centroids and all epochs.

    Pools all (group, epoch) centroid vectors into one matrix, fits PCA once,
    and projects each group's trajectory into the shared coordinate frame.
    This produces stable axes for comparing group trajectories — unlike
    per-group PCA, the coordinate frame is the same for every group.

    Args:
        centroids: (n_groups, n_epochs, d_model) array.
        n_components: Number of principal components.

    Returns:
        Dict with:
            "coords": (n_groups, n_epochs, n_components) projected coordinates.
            "basis": (n_components, d_model) PC directions.
            "center": (d_model,) mean subtracted before projection.
            "explained_variance_ratio": (n_components,) per-component fraction.
    """
    n_groups, n_epochs, d_model = centroids.shape
    stacked = centroids.reshape(-1, d_model)  # (n_groups * n_epochs, d_model)
    center = stacked.mean(axis=0)
    X = stacked - center
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:n_components]
    var_ratio = (S ** 2 / (S ** 2).sum())[:n_components]
    coords = (centroids - center) @ basis.T  # (n_groups, n_epochs, n_components)
    return {
        "coords": coords,
        "basis": basis,
        "center": center,
        "explained_variance_ratio": var_ratio,
    }


def normalize_per_group(coords: np.ndarray) -> np.ndarray:
    """Z-score each group's trajectory independently along the epoch axis.

    Subtracts each group's temporal mean and divides by its temporal std.
    Removes scale differences between groups so trajectory *shapes* are
    directly comparable, regardless of how far each group travels in PC space.

    Args:
        coords: (n_groups, n_epochs, n_components) array.

    Returns:
        Normalized array of the same shape.
    """
    mean = coords.mean(axis=1, keepdims=True)
    std = coords.std(axis=1, keepdims=True).clip(1e-8)
    return (coords - mean) / std
