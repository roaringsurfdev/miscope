"""REQ_038: Parameter trajectory PCA cross-epoch analyzer.

Consumes parameter_snapshot per-epoch artifacts and produces
PCA projections, explained variance, and velocity metrics for
all component groups (all, embedding, attention, mlp).
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.library.pca import pca
from miscope.analysis.library.trajectory import (
    compute_parameter_velocity,
    flatten_snapshot,
)
from miscope.analysis.library.weights import COMPONENT_GROUPS
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# Groups to precompute: "all" + each named component group
_GROUPS = {"all": None, **COMPONENT_GROUPS}


SPEC = AnalyzerSpec(
    name="parameter_trajectory",
    output_scope="cross_epoch",
    inputs=(ArtifactInput("parameter_snapshot", scope="all_epochs"),),
)


@register_analyzer(SPEC)
class ParameterTrajectoryPCA:
    """Cross-epoch analyzer for parameter trajectory PCA projection."""

    name = "parameter_trajectory"
    requires = ["parameter_snapshot"]

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute PCA trajectory and velocity for all component groups."""
        assert inputs.artifacts_dir is not None
        assert inputs.epochs is not None
        artifacts_dir = inputs.artifacts_dir
        epochs = list(inputs.epochs)
        loader = ArtifactLoader(artifacts_dir)
        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]

        result: dict[str, np.ndarray] = {"epochs": np.array(epochs)}

        first_snap = snapshots[0] if snapshots else {}
        for group_name, components in _GROUPS.items():
            # Skip groups whose weight matrices are all absent (e.g. "embedding" for MLP)
            if components is not None and not any(k in first_snap for k in components):
                continue
            vectors = np.array([flatten_snapshot(s, components) for s in snapshots])
            n_components = min(10, len(snapshots), vectors.shape[1])
            pca_result = pca(vectors, n_components=n_components)
            velocity = compute_parameter_velocity(
                snapshots,
                components,
                epochs=epochs,
            )

            result[f"{group_name}__projections"] = pca_result.projections
            result[f"{group_name}__explained_variance_ratio"] = pca_result.explained_variance_ratio
            result[f"{group_name}__explained_variance"] = pca_result.eigenvalues
            result[f"{group_name}__velocity"] = velocity

        return result
