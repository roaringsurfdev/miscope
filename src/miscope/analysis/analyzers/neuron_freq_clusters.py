"""Neuron Frequency Clusters Analyzer.

Computes neuron-frequency specialization matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    reshape_to_grid,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class NeuronFreqClustersAnalyzer:
    """Computes neuron-frequency specialization matrix.

    For each checkpoint, computes what fraction of each neuron's variance
    is explained by each frequency, enabling clustering of neurons by
    their frequency specialization.
    """

    name = "neuron_freq_norm"
    description = "Computes neuron-frequency specialization for clustering"
    architecture_support = ["transformer", "mlp"]

    def __init__(self, specialization_threshold: float = 0.9):
        self.specialization_threshold = specialization_threshold

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute fraction of variance explained by each frequency for each neuron.

        Args:
            bundle: Activation bundle from the forward pass.
            probe: Full probe tensor (p^2, 3)
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'norm_matrix' array of shape (n_frequencies, d_mlp)
            where n_frequencies = p // 2
        """
        fourier_basis = context["fourier_basis"]

        # Get grid size from probe
        p = compute_grid_size_from_dataset(probe)

        # Extract neuron activations at last token position
        neuron_acts = bundle.mlp_post(0, -1)

        # Reshape to (d_mlp, p, p)
        reshaped = reshape_to_grid(neuron_acts, p)

        # Compute 2D Fourier transform
        fourier_neuron_acts = compute_2d_fourier_transform(reshaped, fourier_basis)

        # Compute variance fractions by frequency
        neuron_freq_norm = compute_frequency_variance_fractions(fourier_neuron_acts, p)

        return {"norm_matrix": neuron_freq_norm.detach().cpu().numpy()}

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys."""
        return [
            "specialized_count_per_freq",
            "specialized_count_low",
            "specialized_count_mid",
            "specialized_count_high",
            "specialized_count_total",
            "mean_max_frac",
            "median_max_frac",
        ]

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Compute neuron specialization summary statistics.

        Args:
            result: Dict with 'norm_matrix' of shape (n_freq, d_mlp)
            context: Analysis context (unused)

        Returns:
            Dict with per-frequency counts, range bucket counts, and
            specialization strength metrics.
        """
        norm_matrix = result["norm_matrix"]  # (n_freq, d_mlp)
        n_freq = norm_matrix.shape[0]
        threshold = self.specialization_threshold

        # Per-neuron max and dominant frequency
        max_frac_per_neuron = norm_matrix.max(axis=0)  # (d_mlp,)
        dominant_freq_per_neuron = norm_matrix.argmax(axis=0)  # (d_mlp,)

        # Specialization mask: neurons above threshold
        specialized_mask = max_frac_per_neuron >= threshold  # (d_mlp,)

        # Per-frequency counts: neurons whose dominant freq is f AND above threshold
        specialized_count_per_freq = np.zeros(n_freq, dtype=np.float64)
        for f in range(n_freq):
            specialized_count_per_freq[f] = float(
                np.sum((dominant_freq_per_neuron == f) & specialized_mask)
            )

        # Frequency range boundaries (thirds)
        n_low = n_freq // 3
        n_mid = 2 * n_freq // 3

        # Range bucket counts (neurons specialized AND dominant freq in range)
        specialized_count_low = float(np.sum(specialized_mask & (dominant_freq_per_neuron < n_low)))
        specialized_count_mid = float(
            np.sum(
                specialized_mask
                & (dominant_freq_per_neuron >= n_low)
                & (dominant_freq_per_neuron < n_mid)
            )
        )
        specialized_count_high = float(
            np.sum(specialized_mask & (dominant_freq_per_neuron >= n_mid))
        )
        specialized_count_total = float(np.sum(specialized_mask))

        return {
            "specialized_count_per_freq": specialized_count_per_freq,
            "specialized_count_low": specialized_count_low,
            "specialized_count_mid": specialized_count_mid,
            "specialized_count_high": specialized_count_high,
            "specialized_count_total": specialized_count_total,
            "mean_max_frac": float(np.mean(max_frac_per_neuron)),
            "median_max_frac": float(np.median(max_frac_per_neuron)),
        }
