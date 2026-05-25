"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    reshape_to_grid,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


SPEC = AnalyzerSpec(
    name="neuron_activations",
    category="primary",
    requires_model_weights=False,
    requires_activation_cache=True,  # reads ctx.cache via extract_mlp_activations
    required_hooks=("blocks.0.mlp.hook_out",),
)


@register_analyzer(SPEC)
class NeuronActivationsAnalyzer:
    """Extracts MLP neuron activations reshaped to input space.

    For each checkpoint, extracts activations from the last token position
    and reshapes them to (d_mlp, p, p) for visualization as heatmaps.
    """

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"
    required_hooks: list[str] = ["blocks.0.mlp.hook_out"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Extract neuron activations and reshape to (d_mlp, p, p).

        Args:
            ctx: Analysis context with cache and probe.

        Returns:
            Dict with 'activations' array of shape (d_mlp, p, p)
        """
        assert ctx.cache is not None  # type-narrowing for pyright
        p = compute_grid_size_from_dataset(ctx.probe)
        neuron_acts = extract_mlp_activations(ctx.cache)
        activations = reshape_to_grid(neuron_acts, p)
        return {"activations": activations.detach().cpu().numpy()}
