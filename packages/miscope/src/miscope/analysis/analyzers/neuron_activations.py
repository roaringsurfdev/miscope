"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library import (
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    reshape_to_grid,
)
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="neuron_activations",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
    required_hooks=("blocks.0.mlp.hook_out",),
    outputs=(
        F.tensor(
            "activations",
            "float32",
            ("variant", "epoch"),
            "MLP neuron activations over the full input grid. One blob per epoch; "
            "inner axes are (d_mlp, a, b) — the neuron axis is internal to the array.",
        ),
    ),
)


@register_analyzer(SPEC)
class NeuronActivationsAnalyzer:
    """Extracts MLP neuron activations reshaped to input space.

    For each checkpoint, extracts activations from the last token position
    and reshapes them to (d_mlp, p, p) for visualization as heatmaps.
    """

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract neuron activations and reshape to (d_mlp, p, p)."""
        assert inputs.cache is not None  # type-narrowing for pyright
        p = compute_grid_size_from_dataset(inputs.probe)  # type: ignore
        neuron_acts = extract_mlp_activations(inputs.cache)
        activations = reshape_to_grid(neuron_acts, p)
        return {"activations": activations.detach().cpu().numpy()}
