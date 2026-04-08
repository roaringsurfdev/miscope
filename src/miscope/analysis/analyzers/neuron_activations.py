"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
    reshape_to_grid,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


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
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """
        Extract neuron activations and reshape to (d_mlp, p, p).

        Args:
            bundle: Activation bundle from the forward pass.
            probe: Full probe tensor (p^2, 3)
            context: Analysis context (not used by this analyzer)

        Returns:
            Dict with 'activations' array of shape (d_mlp, p, p)
        """
        # Get grid size from probe
        p = compute_grid_size_from_dataset(probe)

        # Extract neuron activations at last token position
        neuron_acts = bundle.mlp_post(0, -1)

        # Reshape to (d_mlp, p, p)
        activations = reshape_to_grid(neuron_acts, p)

        return {"activations": activations.detach().cpu().numpy()}
