"""Coarseness Analyzer.

Computes per-neuron coarseness (low-frequency energy ratio) to quantify
blob vs plaid neuron patterns across training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    compute_neuron_coarseness,
    reshape_to_grid,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class CoarsenessAnalyzer:
    """Computes per-neuron coarseness across training checkpoints.

    Coarseness is the ratio of low-frequency to total power in a neuron's
    activation pattern. High coarseness (>= 0.7) indicates "blob" neurons
    with large coherent activation regions, while low coarseness (< 0.5)
    indicates "plaid" neurons with fine-grained checkerboard patterns.

    Composes existing library functions:
        extract_mlp_activations -> reshape_to_grid -> compute_2d_fourier_transform
        -> compute_frequency_variance_fractions -> compute_neuron_coarseness
    """

    name = "coarseness"
    description = "Computes per-neuron coarseness (low-frequency energy ratio)"
    architecture_support = ["transformer", "mlp"]

    def __init__(
        self,
        n_low_freqs: int = 3,
        blob_threshold: float = 0.7,
    ):
        self.n_low_freqs = n_low_freqs
        self.blob_threshold = blob_threshold

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute per-neuron coarseness values.

        Args:
            bundle: Activation bundle from the forward pass.
            probe: Full probe tensor (p^2, 3)
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'coarseness' array of shape (d_mlp,)
        """
        fourier_basis = context["fourier_basis"]
        p = compute_grid_size_from_dataset(probe)

        neuron_acts = bundle.mlp_post(0, -1)
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
