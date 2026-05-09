"""REQ_117: Activation-space DMD with windowed + per-regime structure.

Per-site windowed DMD on per-class centroid trajectories in global PCA
space (same input as the legacy `centroid_dmd` analyzer). For each site:

1. Sliding-window DMD across the centroid trajectory.
2. Eigenvalue tracking across windows (greedy nearest-neighbor).
3. Residual-driven regime detection on the per-window mean residual.
4. Per-regime DMD as a recursive second pass — fits a clean linear DMD
   operator inside each detected segment.

Built parallel to `centroid_dmd` per REQ_111. The legacy analyzer is not
modified during this work; both analyzers coexist on disk and in the
registry through the parallel period. Validation outcomes are recorded
in REQ_117's Notes section before REQ_102 retires the legacy analyzer.
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

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

# Defaults agreed in REQ_117 phase 1a sub-phasing.
_WINDOW_SIZE = 10
_WINDOW_STRIDE = 1
_ENERGY_THRESHOLD = 0.99


class ActivationDMD:
    """Cross-epoch analyzer: windowed + per-regime DMD on centroid trajectories.

    Reads `global_centroid_pca/cross_epoch.npz` (the same input the legacy
    `centroid_dmd` analyzer consumes), runs the four-stage REQ_117 pipeline
    per site, and writes a single cross_epoch artifact with namespaced keys.
    """

    name = "activation_dmd"
    requires = ["repr_geometry"]  # transitive: global_centroid_pca validated at runtime

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Run windowed + per-regime DMD per site.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context (unused).

        Returns:
            Dict of arrays for storage in cross_epoch.npz, with keys
            namespaced by site and stage:
              {site}__trajectory                 (n_epochs, state_dim)
              {site}__n_classes                  scalar
              {site}__windowed__*                (windowed DMD outputs)
              {site}__tracks__*                  (eigenvalue tracking outputs)
              {site}__regimes__*                 (regime boundaries in window space)
              {site}__per_regime__*              (per-regime DMD outputs)

        Raises:
            FileNotFoundError: If global_centroid_pca/cross_epoch.npz is absent.
        """
        loader = ArtifactLoader(artifacts_dir)
        if not loader.has_cross_epoch("global_centroid_pca"):
            raise FileNotFoundError(
                "activation_dmd requires global_centroid_pca/cross_epoch.npz. "
                "Run global_centroid_pca first."
            )

        global_pca = loader.load_cross_epoch("global_centroid_pca")
        result: dict[str, np.ndarray] = {"epochs": np.array(epochs)}

        for site in _SITES:
            projections = global_pca[f"{site}__projections"]
            n_epochs, n_classes, n_components = projections.shape
            trajectory = projections.reshape(n_epochs, -1).astype(np.float64)

            # The window size constraint: window must fit inside the trajectory.
            # If a variant has fewer epochs than _WINDOW_SIZE, scale down.
            window_size = min(_WINDOW_SIZE, n_epochs)

            windowed = compute_windowed_dmd(
                trajectory,
                window_size=window_size,
                stride=_WINDOW_STRIDE,
                energy_threshold=_ENERGY_THRESHOLD,
            )

            tracks = track_eigenvalues_across_windows(
                windowed["eigenvalues"], windowed["n_modes_per_window"]
            )

            regimes = detect_regime_boundaries(windowed["residual_norm_mean"], threshold=None)

            step_starts, step_ends = _regime_segments_to_step_space(
                regimes["segment_starts"],
                regimes["segment_ends"],
                windowed["window_starts"],
                windowed["window_ends"],
                n_epochs,
            )
            per_regime = compute_per_regime_dmd(
                trajectory,
                step_starts,
                step_ends,
                energy_threshold=_ENERGY_THRESHOLD,
            )

            result[f"{site}__trajectory"] = trajectory
            result[f"{site}__n_classes"] = np.array(n_classes, dtype=np.int64)
            result[f"{site}__n_components"] = np.array(n_components, dtype=np.int64)

            for key, value in windowed.items():
                result[f"{site}__windowed__{key}"] = value
            for key, value in tracks.items():
                result[f"{site}__tracks__{key}"] = value
            for key, value in regimes.items():
                result[f"{site}__regimes__{key}"] = value
            for key, value in per_regime.items():
                result[f"{site}__per_regime__{key}"] = value

        return result


def _regime_segments_to_step_space(
    regime_starts: np.ndarray,
    regime_ends: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate regime boundaries from window-space to step-space.

    Window-space segments index into the windowed DMD output; step-space
    segments index into the original trajectory. The first window of regime
    k starts at step `window_starts[regime_starts[k]]`. The last window of
    regime k is at window index `regime_ends[k] - 1`; that window covers
    steps up to `window_ends[regime_ends[k] - 1]`.

    Edge case: an empty regime list returns empty arrays.
    """
    if len(regime_starts) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    step_starts = window_starts[regime_starts].astype(np.int64)
    last_window_per_regime = regime_ends - 1
    step_ends = window_ends[last_window_per_regime].astype(np.int64)
    # Clamp to trajectory bounds for safety.
    step_ends = np.minimum(step_ends, n_steps)
    return step_starts, step_ends
