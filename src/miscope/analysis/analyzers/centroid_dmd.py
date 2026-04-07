"""REQ_051: DMD cross-epoch analyzer on centroid class trajectories.

Applies standard exact DMD to the globally-projected centroid trajectories
from REQ_050 (global_centroid_pca). The primary output is the per-step DMD
residual norm — a data-driven, model-agnostic candidate for grokking onset
detection.

Depends on global_centroid_pca/cross_epoch.npz. Validated at runtime;
pipeline requires= points to "repr_geometry" as the transitive dependency
(since the pipeline checks for per-epoch artifacts, not cross_epoch files).
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.dmd import compute_dmd

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

# DMD SVD energy threshold. Conservative (0.99) to retain enough modes for
# accurate residual computation. Flagged for review: a spectral-gap criterion
# could be more principled once real singular value spectra are inspected.
_ENERGY_THRESHOLD = 0.99


class CentroidDMD:
    """Cross-epoch analyzer: standard DMD on global PCA centroid trajectories.

    For each activation site, flattens per-epoch centroid projections into
    a state trajectory and applies exact DMD. Stores eigenvalues, modes,
    amplitudes, and per-step residual norms.

    The residual norm time series is the primary deliverable — a candidate
    metric for grokking onset that does not depend on task performance.
    """

    name = "centroid_dmd"
    requires = ["repr_geometry"]  # transitive: global_centroid_pca validated at runtime
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Apply DMD to globally-projected centroid trajectories.

        Args:
            artifacts_dir: Root artifacts directory for the variant.
            epochs: Sorted list of available epoch numbers.
            context: Family-provided analysis context (unused here).

        Returns:
            Dict of arrays for storage in cross_epoch.npz.

        Raises:
            FileNotFoundError: If global_centroid_pca/cross_epoch.npz is absent.
        """
        loader = ArtifactLoader(artifacts_dir)
        if not loader.has_cross_epoch("global_centroid_pca"):
            raise FileNotFoundError(
                "centroid_dmd requires global_centroid_pca/cross_epoch.npz. "
                "Run global_centroid_pca first."
            )

        global_pca = loader.load_cross_epoch("global_centroid_pca")
        result: dict[str, np.ndarray] = {"epochs": np.array(epochs)}

        for site in _SITES:
            proj_key = f"{site}__projections"
            projections = global_pca[proj_key]  # (n_epochs, n_classes, n_components)
            n_epochs, n_classes, n_components = projections.shape

            # Flatten to state trajectory: (n_epochs, n_classes * n_components)
            trajectory = projections.reshape(n_epochs, -1).astype(np.float64)

            dmd = compute_dmd(trajectory, energy_threshold=_ENERGY_THRESHOLD)

            result[f"{site}__eigenvalues"] = dmd["eigenvalues"]
            result[f"{site}__modes"] = dmd["modes"]
            result[f"{site}__amplitudes"] = dmd["amplitudes"]
            result[f"{site}__residual_norms"] = dmd["residual_norms"]
            result[f"{site}__singular_values"] = dmd["singular_values"]
            result[f"{site}__n_modes"] = dmd["n_modes"]
            result[f"{site}__trajectory"] = trajectory
            result[f"{site}__n_classes"] = np.array(n_classes, dtype=np.int64)

        return result
