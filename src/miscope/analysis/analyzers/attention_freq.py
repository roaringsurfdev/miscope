"""Attention Head Frequency Specialization Analyzer.

Computes Fourier frequency decomposition of attention patterns per head,
analogous to neuron_freq_clusters for MLP neurons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import einops
import numpy as np

from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


class AttentionFreqAnalyzer:
    """Computes frequency decomposition of attention patterns per head.

    For each checkpoint, extracts the attention pattern for the = → a
    position pair, reshapes to a (n_heads, p, p) grid, applies the modular
    Fourier transform, and computes per-frequency variance fractions.

    Output mirrors neuron_freq_norm: freq_matrix of shape (n_freq, n_heads).
    """

    name = "attention_freq"
    description = "Frequency decomposition of attention patterns per head"
    architecture_support = ["transformer"]

    def __init__(
        self,
        to_position: int = 2,
        from_position: int = 0,
    ):
        self.to_position = to_position
        self.from_position = from_position

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Compute frequency variance fractions for each attention head.

        Args:
            ctx: Analysis context with bundle, probe, and analysis_params.
                 analysis_params must contain 'fourier_basis'.

        Returns:
            Dict with 'freq_matrix' array of shape (n_freq, n_heads)
        """
        fourier_basis = ctx.analysis_params["fourier_basis"]
        p = compute_grid_size_from_dataset(ctx.probe)

        # Extract attention patterns: (p*p, n_heads, n_pos, n_pos)
        attn = ctx.bundle.attention_pattern(0)

        # Select position pair, e.g. = → a: (p*p, n_heads)
        attn_pair = attn[:, :, self.to_position, self.from_position]

        # Reshape batch to (p, p) grid: (n_heads, p, p)
        attn_grid = einops.rearrange(attn_pair, "(a b) h -> h a b", a=p, b=p)

        # 2D Fourier transform: (n_heads, n_comp, n_comp)
        fourier_attn = compute_2d_fourier_transform(attn_grid, fourier_basis)

        # Frequency variance fractions: (n_freq, n_heads)
        freq_matrix = compute_frequency_variance_fractions(fourier_attn, p)

        return {"freq_matrix": freq_matrix.detach().cpu().numpy()}

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys."""
        return [
            "dominant_freq_per_head",
            "max_frac_per_head",
            "mean_specialization",
        ]

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Compute summary statistics from this epoch's frequency result.

        Args:
            result: Dict with 'freq_matrix' array of shape (n_freq, n_heads)
            context: Analysis context (unused)

        Returns:
            Dict with per-head dominant frequency, max fraction, and mean specialization
        """
        freq_matrix = result["freq_matrix"]  # (n_freq, n_heads)
        max_frac_per_head = freq_matrix.max(axis=0)  # (n_heads,)
        dominant_freq_per_head = freq_matrix.argmax(axis=0)  # (n_heads,)

        return {
            "dominant_freq_per_head": dominant_freq_per_head.astype(np.float64),
            "max_frac_per_head": max_frac_per_head.astype(np.float64),
            "mean_specialization": float(max_frac_per_head.mean()),
        }
