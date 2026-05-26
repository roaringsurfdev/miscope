"""Coarseness Analyzer.

Computes per-neuron coarseness (low-frequency energy ratio) to quantify
blob vs plaid neuron patterns across training.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    compute_neuron_coarseness,
    extract_mlp_activations,
    reshape_to_grid,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="coarseness",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
    required_hooks=("blocks.0.mlp.hook_out",),
    produces_summary=True,
)


@register_analyzer(SPEC)
class CoarsenessAnalyzer:
    """Computes per-neuron coarseness across training checkpoints."""

    name = "coarseness"
    description = "Computes per-neuron coarseness (low-frequency energy ratio)"

    def __init__(
        self,
        n_low_freqs: int = 3,
        blob_threshold: float = 0.7,
    ):
        self.n_low_freqs = n_low_freqs
        self.blob_threshold = blob_threshold

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-neuron coarseness values."""
        assert inputs.cache is not None  # type-narrowing for pyright
        fourier_basis = context["fourier_basis"]
        p = compute_grid_size_from_dataset(inputs.probe)  # type: ignore

        neuron_acts = extract_mlp_activations(inputs.cache)
        reshaped = reshape_to_grid(neuron_acts, p)
        fourier_neuron_acts = compute_2d_fourier_transform(reshaped, fourier_basis)
        freq_fractions = compute_frequency_variance_fractions(fourier_neuron_acts, p)
        coarseness = compute_neuron_coarseness(freq_fractions, self.n_low_freqs)

        return {"coarseness": coarseness.detach().cpu().numpy()}

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys."""
        return [
            "mean_coarseness",
            "std_coarseness",
            "median_coarseness",
            "p25_coarseness",
            "p75_coarseness",
            "blob_count",
            "coarseness_hist",
        ]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, float | np.ndarray]:
        """Compute summary statistics from this epoch's coarseness result.

        Args:
            result: Dict with 'coarseness' array of shape (d_mlp,)
            context: Analysis context (unused)

        Returns:
            Dict with summary statistics
        """
        coarseness = result["coarseness"]
        return {
            "mean_coarseness": float(np.mean(coarseness)),
            "std_coarseness": float(np.std(coarseness)),
            "median_coarseness": float(np.median(coarseness)),
            "p25_coarseness": float(np.percentile(coarseness, 25)),
            "p75_coarseness": float(np.percentile(coarseness, 75)),
            "blob_count": float(np.sum(coarseness >= self.blob_threshold)),
            "coarseness_hist": np.histogram(coarseness, bins=20, range=(0.0, 1.0))[0].astype(
                np.float64
            ),
        }
