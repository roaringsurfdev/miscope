"""Attention Patterns Analyzer.

Captures per-head attention patterns across all position pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import numpy as np

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
)

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


class AttentionPatternsAnalyzer:
    """Captures per-head attention patterns across all position pairs.

    For each checkpoint, extracts attention patterns from the cache and
    reshapes them to (n_heads, n_positions, n_positions, p, p) for
    visualization as heatmaps indexed by (a, b) input pairs.
    """

    name = "attention_patterns"
    description = "Captures per-head attention patterns across all position pairs"
    architecture_support = ["transformer"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Extract attention patterns and reshape to (n_heads, n_pos, n_pos, p, p).

        Args:
            ctx: Analysis context with bundle and probe.

        Returns:
            Dict with 'patterns' array of shape (n_heads, n_pos, n_pos, p, p)
            where values are attention weights in [0, 1].
        """
        p = compute_grid_size_from_dataset(ctx.probe)

        # Shape: (p*p, n_heads, seq_to, seq_from)
        attn = ctx.bundle.attention_pattern(0)

        # Reshape batch dim to (p, p) grid for each (head, to_pos, from_pos)
        patterns = einops.rearrange(
            attn,
            "(a b) heads to_pos from_pos -> heads to_pos from_pos a b",
            a=p,
            b=p,
        )

        return {"patterns": patterns.detach().cpu().numpy()}
