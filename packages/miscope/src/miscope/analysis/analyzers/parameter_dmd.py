"""REQ_117 phase 2: Parameter-space DMD analyzer.

Cross-epoch analyzer that mirrors `activation_dmd`'s pipeline but operates
on weight matrix slices partitioned by neuron group. For each
``(group_id, matrix)`` pair where ``matrix in {W_in, W_out}``:

1. Build per-epoch state vector by flattening the group's slice of the
   matrix (``W_in[:, group_neurons]`` or ``W_out[group_neurons, :]``).
2. Fit a global PCA across the per-group trajectory and project to the
   smallest number of components that capture ≥ 95% variance.
3. Run sliding-window DMD on the projected trajectory.
4. Track eigenvalues across windows.
5. Detect regime boundaries via residual peaks.
6. Run per-regime DMD inside each detected segment.

The grouping comes from `neuron_grouping` at a configurable
``reference_epoch`` (defaults to the last available checkpoint). Each
artifact run is keyed to a single grouping snapshot — to study dynamics
relative to a different grouping (e.g., when a transient frequency was
still committed), re-run with a pinned reference epoch.

W_in and W_out are treated as separate "matrices" of analysis the same
way `activation_dmd` treats sites — they may reorganize on different
timelines, and combining them into a single state vector would smear
those timelines together. UNASSIGNED neurons are skipped (their weight
projections are too diffuse to be meaningfully grouped, and they would
inject structureless drift into a nominal group).

Per-variant artifact at ``artifacts/parameter_dmd/cross_epoch.npz`` with
keys namespaced by ``group_{g}__{matrix}__{stage}__{field}``, mirroring
`activation_dmd`'s ``{site}__{stage}__{field}`` convention.
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.dmd import (
    compute_per_regime_dmd,
    compute_windowed_dmd,
    detect_regime_boundaries,
    track_eigenvalues_across_windows,
)
from miscope.analysis.library.pca import pca

_MATRICES = ["W_in", "W_out"]

# Defaults — mirrored on the activation_dmd defaults so the two analyzers
# read with the same temporal resolution.
_WINDOW_SIZE = 10
_WINDOW_STRIDE = 1
_DMD_ENERGY_THRESHOLD = 0.99

# PCA preprocessing target: keep components covering at least this much
# variance of the per-(group, matrix) trajectory.
_PCA_VARIANCE_TARGET = 0.95
# Defensive cap so a fluke high-variance trajectory does not blow up DMD.
_PCA_MAX_COMPONENTS = 50

# Context key for choosing which neuron_grouping snapshot to use.
_CONTEXT_REFERENCE_EPOCH_KEY = "parameter_dmd_reference_epoch"


class ParameterDMD:
    """Cross-epoch analyzer: per-(group, matrix) windowed + per-regime DMD.

    Built parallel to `activation_dmd`. Reads `parameter_snapshot` per-epoch
    artifacts (for the weight matrices) and `neuron_grouping` at a chosen
    reference epoch (for the partition).
    """

    name = "parameter_dmd"
    requires = ["parameter_snapshot", "neuron_grouping"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run per-(group, matrix) windowed + per-regime DMD across epochs.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context. Optional key:
                - ``parameter_dmd_reference_epoch``: int — which
                  ``neuron_grouping`` epoch to use as the partition
                  source. Defaults to the last available checkpoint.

        Returns:
            Dict of arrays for storage in cross_epoch.npz.

        Raises:
            FileNotFoundError: If parameter_snapshot or neuron_grouping
                artifacts are absent.
        """
        loader = ArtifactLoader(artifacts_dir)

        reference_epoch = self._resolve_reference_epoch(loader, epochs, context)
        grouping = loader.load_epoch("neuron_grouping", reference_epoch)
        assignments = np.asarray(grouping["assignments"], dtype=np.int64)
        n_groups = int(grouping["n_groups"])
        n_per_group = np.asarray(grouping["n_per_group"], dtype=np.int64)
        populated_groups = np.where(n_per_group > 0)[0].astype(np.int64)

        # Load weight trajectories selectively (parameter_snapshot is large).
        snapshots = loader.load_epochs(
            "parameter_snapshot", epochs, fields=["W_in", "W_out"]
        )
        # snapshots["W_in"] shape:  (n_epochs, d_model, d_mlp)
        # snapshots["W_out"] shape: (n_epochs, d_mlp, d_model)
        w_in_traj = np.asarray(snapshots["W_in"], dtype=np.float64)
        w_out_traj = np.asarray(snapshots["W_out"], dtype=np.float64)
        n_epochs = w_in_traj.shape[0]

        result: dict[str, np.ndarray] = {
            "epochs": np.array(epochs),
            "reference_epoch": np.array(reference_epoch, dtype=np.int64),
            "n_groups": np.array(n_groups, dtype=np.int64),
            "populated_groups": populated_groups,
            "group_n_neurons": np.array(
                [int(n_per_group[g]) for g in populated_groups], dtype=np.int64
            ),
        }

        for g in populated_groups:
            neurons_in_group = np.where(assignments == int(g))[0]
            for matrix_name in _MATRICES:
                trajectory = self._build_trajectory(
                    w_in_traj, w_out_traj, neurons_in_group, matrix_name
                )
                projected = self._project_pca(trajectory)
                self._run_dmd_pipeline(projected, n_epochs, g, matrix_name, result)

        return result

    def _resolve_reference_epoch(
        self,
        loader: ArtifactLoader,
        epochs: list[int],
        context: dict[str, Any],
    ) -> int:
        """Pick the neuron_grouping epoch to use as the partition source.

        Default: last available `neuron_grouping` epoch. Caller can pin
        via the ``parameter_dmd_reference_epoch`` context key.
        """
        configured = context.get(_CONTEXT_REFERENCE_EPOCH_KEY)
        available = sorted(loader.get_epochs("neuron_grouping"))
        if not available:
            raise FileNotFoundError(
                "parameter_dmd requires neuron_grouping artifacts. "
                "Run neuron_grouping first."
            )
        if configured is None:
            return int(available[-1])
        configured = int(configured)
        # Snap to the nearest available epoch — caller may have requested
        # an epoch the variant didn't checkpoint.
        nearest = min(available, key=lambda e: abs(e - configured))
        return int(nearest)

    def _build_trajectory(
        self,
        w_in_traj: np.ndarray,
        w_out_traj: np.ndarray,
        neurons_in_group: np.ndarray,
        matrix_name: str,
    ) -> np.ndarray:
        """Per-(group, matrix) state trajectory.

        Returns shape ``(n_epochs, d_model * n_group_neurons)``.
        """
        if matrix_name == "W_in":
            # W_in: (n_epochs, d_model, d_mlp) → slice columns for group
            slab = w_in_traj[:, :, neurons_in_group]  # (n_epochs, d_model, n_g)
        elif matrix_name == "W_out":
            # W_out: (n_epochs, d_mlp, d_model) → slice rows for group
            slab = w_out_traj[:, neurons_in_group, :]  # (n_epochs, n_g, d_model)
        else:
            raise ValueError(f"unknown matrix_name '{matrix_name}'")
        n_epochs = slab.shape[0]
        return slab.reshape(n_epochs, -1)

    def _project_pca(self, trajectory: np.ndarray) -> np.ndarray:
        """Fit global PCA on the trajectory and project to top components.

        Retains the smallest ``n_components`` whose cumulative explained
        variance reaches `_PCA_VARIANCE_TARGET`, capped at
        `_PCA_MAX_COMPONENTS`.
        """
        full = pca(trajectory, n_components=None)
        cumvar = np.cumsum(full.explained_variance_ratio)
        # First component index reaching the target.
        passing = np.where(cumvar >= _PCA_VARIANCE_TARGET)[0]
        if len(passing) > 0:
            n_components = int(passing[0]) + 1
        else:
            n_components = len(cumvar)
        n_components = max(1, min(n_components, _PCA_MAX_COMPONENTS))
        return full.projections[:, :n_components]

    def _run_dmd_pipeline(
        self,
        projected: np.ndarray,
        n_epochs: int,
        group_id: int,
        matrix_name: str,
        result: dict[str, np.ndarray],
    ) -> None:
        """Run the four DMD stages and pack outputs into result with the
        ``group_{g}__{matrix_name}__{stage}__{field}`` key convention."""
        window_size = min(_WINDOW_SIZE, n_epochs)
        windowed = compute_windowed_dmd(
            projected,
            window_size=window_size,
            stride=_WINDOW_STRIDE,
            energy_threshold=_DMD_ENERGY_THRESHOLD,
        )
        tracks = track_eigenvalues_across_windows(
            windowed["eigenvalues"], windowed["n_modes_per_window"]
        )
        regimes = detect_regime_boundaries(
            windowed["residual_norm_mean"], threshold=None
        )
        step_starts, step_ends = _regime_segments_to_step_space(
            regimes["segment_starts"],
            regimes["segment_ends"],
            windowed["window_starts"],
            windowed["window_ends"],
            n_epochs,
        )
        per_regime = compute_per_regime_dmd(
            projected,
            step_starts,
            step_ends,
            energy_threshold=_DMD_ENERGY_THRESHOLD,
        )

        prefix = f"group_{int(group_id)}__{matrix_name}"
        result[f"{prefix}__trajectory"] = projected
        result[f"{prefix}__n_components"] = np.array(
            projected.shape[1], dtype=np.int64
        )
        for key, value in windowed.items():
            result[f"{prefix}__windowed__{key}"] = value
        for key, value in tracks.items():
            result[f"{prefix}__tracks__{key}"] = value
        for key, value in regimes.items():
            result[f"{prefix}__regimes__{key}"] = value
        for key, value in per_regime.items():
            result[f"{prefix}__per_regime__{key}"] = value


def _regime_segments_to_step_space(
    regime_starts: np.ndarray,
    regime_ends: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate regime boundaries from window-space to step-space.

    Identical to the helper in `activation_dmd` — windowed-DMD's segment
    indices need to be converted to original-trajectory indices for
    `compute_per_regime_dmd`. Will be promoted to a shared helper if a
    third analyzer needs it.
    """
    if len(regime_starts) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    step_starts = window_starts[regime_starts].astype(np.int64)
    last_window_per_regime = regime_ends - 1
    step_ends = window_ends[last_window_per_regime].astype(np.int64)
    step_ends = np.minimum(step_ends, n_steps)
    return step_starts, step_ends
