"""REQ_038: Parameter trajectory PCA cross-epoch analyzer.

Consumes parameter_snapshot per-epoch artifacts and produces
PCA projections, explained variance, and velocity metrics for
all component groups (all, embedding, attention, mlp).
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.trajectory import (
    compute_parameter_velocity,
    compute_pca_trajectory,
)
from miscope.analysis.library.weights import COMPONENT_GROUPS

# Groups to precompute: "all" + each named component group
_GROUPS = {"all": None, **COMPONENT_GROUPS}


class ParameterTrajectoryPCA:
    """Cross-epoch analyzer for parameter trajectory PCA projection.

    Precomputes PCA projections and velocity for all component groups
    so the dashboard can render trajectories without runtime PCA.
    """

    name = "parameter_trajectory"
    requires = ["parameter_snapshot"]
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute PCA trajectory and velocity for all component groups."""
        loader = ArtifactLoader(artifacts_dir)
        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]

        result: dict[str, np.ndarray] = {"epochs": np.array(epochs)}

        first_snap = snapshots[0] if snapshots else {}
        for group_name, components in _GROUPS.items():
            # Skip groups whose weight matrices are all absent (e.g. "embedding" for MLP)
            if components is not None and not any(k in first_snap for k in components):
                continue
            n_components = min(10, len(snapshots))
            pca = compute_pca_trajectory(snapshots, components, n_components)
            velocity = compute_parameter_velocity(
                snapshots,
                components,
                epochs=epochs,
            )

            result[f"{group_name}__projections"] = pca["projections"]
            result[f"{group_name}__explained_variance_ratio"] = pca["explained_variance_ratio"]
            result[f"{group_name}__explained_variance"] = pca["explained_variance"]
            result[f"{group_name}__velocity"] = velocity

        return result
