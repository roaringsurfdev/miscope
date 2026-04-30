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
"""

import numpy as np

from miscope.analysis.library.dynamics import compute_velocity
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
