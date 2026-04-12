"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
    reshape_to_grid,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


class NeuronActivationsAnalyzer:
    """Extracts MLP neuron activations reshaped to input space.

    For each checkpoint, extracts activations from the last token position
    and reshapes them to (d_mlp, p, p) for visualization as heatmaps.
    """

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"
    architecture_support = ["transformer", "mlp"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Extract neuron activations and reshape to (d_mlp, p, p).

        Args:
            ctx: Analysis context with bundle and probe.

        Returns:
            Dict with 'activations' array of shape (d_mlp, p, p)
        """
        # Get grid size from probe
        p = compute_grid_size_from_dataset(ctx.probe)

        # Extract neuron activations at last token position
        neuron_acts = ctx.bundle.mlp_post(0, -1)

        # Reshape to (d_mlp, p, p)
        activations = reshape_to_grid(neuron_acts, p)

        return {"activations": activations.detach().cpu().numpy()}
