"""Attention Patterns Analyzer.

Captures per-head attention patterns across all position pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import einops
import numpy as np
import torch

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class AttentionPatternsAnalyzer:
    """Captures per-head attention patterns across all position pairs.

    For each checkpoint, extracts attention patterns from the cache and
    reshapes them to (n_heads, n_positions, n_positions, p, p) for
    visualization as heatmaps indexed by (a, b) input pairs.
    """

    name = "attention_patterns"
    description = "Captures per-head attention patterns across all position pairs"

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """
        Extract attention patterns and reshape to (n_heads, n_pos, n_pos, p, p).

        Args:
            bundle: Activation bundle from the forward pass.
            probe: Full probe tensor (p^2, 3)
            context: Analysis context (not used by this analyzer)

        Returns:
            Dict with 'patterns' array of shape (n_heads, n_pos, n_pos, p, p)
            where values are attention weights in [0, 1].
        """
        p = compute_grid_size_from_dataset(probe)

        # Shape: (p*p, n_heads, seq_to, seq_from)
        attn = bundle.attention_pattern(0)

        # Reshape batch dim to (p, p) grid for each (head, to_pos, from_pos)
        patterns = einops.rearrange(
            attn,
            "(a b) heads to_pos from_pos -> heads to_pos from_pos a b",
            a=p,
            b=p,
        )

        return {"patterns": patterns.detach().cpu().numpy()}
