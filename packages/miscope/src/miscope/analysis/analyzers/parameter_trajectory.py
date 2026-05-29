"""Parameter trajectory cross-epoch analyzer (REQ_038; renamed REQ_111).

Consumes parameter_snapshot per-epoch artifacts and produces trajectory
summaries — currently PCA projections, explained variance, and velocity —
for all component groups (all, embedding, attention, mlp). The name no
longer commits to PCA as the only trajectory summary: future implementations
may layer in additional reductions without breaking the contract.

Transform steps route through REQ_109 primitives: PCA via
:func:`miscope.analysis.library.pca.pca`, finite-difference velocity via
:func:`miscope.analysis.library.dynamics.compute_velocity` (called through
:func:`compute_parameter_velocity`).
"""

from typing import Any

import numpy as np

from miscope.analysis.inputs import ALL, ArtifactInput, ResolvedInputs
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
class ParameterTrajectory:
    """Cross-epoch analyzer for parameter trajectory summaries (PCA + velocity)."""

    name = "parameter_trajectory"
    requires = ["parameter_snapshot"]

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute PCA trajectory and velocity for all component groups."""
        assert inputs.deps is not None
        assert inputs.epochs is not None
        epochs = list(inputs.epochs)
        snapshots = [inputs.deps.load_epoch("parameter_snapshot", e, fields=ALL) for e in epochs]

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
