"""Parameter trajectory analysis utilities.

Snapshot-flattening + parameter velocity for trajectories in weight space.
General-purpose curve-shape characterizations (arc length, self-intersection,
signed loop area, curvature profile) moved to
:mod:`miscope.analysis.library.shape` per REQ_109 phase 2b. PCA primitives
live in :mod:`miscope.analysis.library.pca`; callers that need PCA over
snapshots flatten via :func:`flatten_snapshot` and call ``pca`` directly.

Functions:
- flatten_snapshot: concatenate selected weight matrices into a parameter vector.
- compute_parameter_velocity: per-step displacement, optionally normalized by epoch gap.
- normalize_per_group: z-score each group's trajectory independently.
- normalize_trajectory_pair: center + aspect-preserving scale of a 2D PC-pair trajectory.
- compute_group_trajectory_proximity: sign-corrected pairwise L2 distance between
  normalized group trajectories, keyed by component-group pair.
"""

import numpy as np

from miscope.analysis.library.dynamics import compute_velocity
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES

# Component-group pairs compared by the proximity instrument, in display order.
# Each entry is (pair_key, group_a, group_b).
_PROXIMITY_PAIRS = (
    ("emb_attn", "embedding", "attention"),
    ("emb_mlp", "embedding", "mlp"),
    ("attn_mlp", "attention", "mlp"),
)


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
    vectors = np.array([flatten_snapshot(s, components) for s in snapshots])
    deltas = compute_velocity(vectors)
    displacements = np.linalg.norm(deltas, axis=1)
    if epochs is not None:
        gaps = np.diff(np.asarray(epochs))
        safe_gaps = np.where(gaps > 0, gaps, 1)
        displacements = np.where(gaps > 0, displacements / safe_gaps, 0.0)
    return displacements


def normalize_trajectory_pair(
    pc_x: np.ndarray,
    pc_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale a 2D PC-pair trajectory so its widest axis spans [-0.5, 0.5].

    Preserves aspect ratio (shape and loop structure) within the trajectory.
    Used as the shared normalization for group-overlay and group-proximity
    instruments so distances are comparable across groups of different scale.

    Args:
        pc_x: Per-epoch coordinate along the first PC axis.
        pc_y: Per-epoch coordinate along the second PC axis.

    Returns:
        ``(nx, ny)`` centered, aspect-preserving normalized coordinates.
    """
    cx, cy = pc_x.mean(), pc_y.mean()
    nx, ny = pc_x - cx, pc_y - cy
    scale = max(nx.max() - nx.min(), ny.max() - ny.min())
    if scale > 1e-12:
        nx, ny = nx / scale, ny / scale
    return nx, ny


def compute_group_trajectory_proximity(
    cross_epoch_data: dict[str, np.ndarray],
    col_x: int = 0,
    col_y: int = 1,
) -> dict[str, np.ndarray]:
    """Sign-corrected pairwise L2 distance between normalized group trajectories.

    For each component-group pair, normalizes both groups' PC-pair trajectories
    (aspect-preserving) and computes the per-epoch L2 distance, taking the
    sign-flip-corrected minimum ``min(‖a−b‖, ‖a+b‖)`` since PCA sign gauge is
    arbitrary. Distance near zero means the two groups occupy the same region
    of their respective normalized parameter spaces at that epoch.

    Args:
        cross_epoch_data: From ``ArtifactLoader.load_cross_epoch("parameter_trajectory")``.
            Reads ``{group}__projections`` for embedding/attention/mlp.
        col_x: PC column for the x-axis (0=PC1, 1=PC2).
        col_y: PC column for the y-axis (1=PC2, 2=PC3).

    Returns:
        Dict mapping pair key (``"emb_attn"``, ``"emb_mlp"``, ``"attn_mlp"``) to
        a per-epoch distance array. Pairs whose groups are absent are omitted.
    """
    groups: dict[str, np.ndarray] = {}
    for name in ("embedding", "attention", "mlp"):
        proj_key = f"{name}__projections"
        if proj_key not in cross_epoch_data:
            continue
        proj = cross_epoch_data[proj_key]
        nx, ny = normalize_trajectory_pair(proj[:, col_x], proj[:, col_y])
        groups[name] = np.stack([nx, ny], axis=1)

    proximity: dict[str, np.ndarray] = {}
    for key, a, b in _PROXIMITY_PAIRS:
        if a not in groups or b not in groups:
            continue
        proximity[key] = np.minimum(
            np.linalg.norm(groups[a] - groups[b], axis=1),
            np.linalg.norm(groups[a] + groups[b], axis=1),
        )
    return proximity


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
