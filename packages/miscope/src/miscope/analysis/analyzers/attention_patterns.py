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
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


SPEC = AnalyzerSpec(
    name="attention_patterns",
    category="primary",
    requires_model_weights=False,
    requires_activation_cache=True,  # reads ctx.cache["blocks.0.attn.hook_pattern"]
    required_hooks=("blocks.0.attn.hook_pattern",),
)


@register_analyzer(SPEC)
class AttentionPatternsAnalyzer:
    """Captures per-head attention patterns across all position pairs.

    For each checkpoint, extracts attention patterns from the cache and
    reshapes them to (n_heads, n_positions, n_positions, p, p) for
    visualization as heatmaps indexed by (a, b) input pairs.
    """

    name = "attention_patterns"
    description = "Captures per-head attention patterns across all position pairs"
    required_hooks: list[str] = ["blocks.0.attn.hook_pattern"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Extract attention patterns and reshape to (n_heads, n_pos, n_pos, p, p).

        Args:
            ctx: Analysis context with cache and probe.

        Returns:
            Dict with 'patterns' array of shape (n_heads, n_pos, n_pos, p, p)
            where values are attention weights in [0, 1].
        """
        assert ctx.cache is not None  # type-narrowing for pyright
        p = compute_grid_size_from_dataset(ctx.probe)

        # Shape: (p*p, n_heads, seq_to, seq_from)
        attn = ctx.cache["blocks.0.attn.hook_pattern"]

        # Reshape batch dim to (p, p) grid for each (head, to_pos, from_pos)
        patterns = einops.rearrange(
            attn,
            "(a b) heads to_pos from_pos -> heads to_pos from_pos a b",
            a=p,
            b=p,
        )

        return {"patterns": patterns.detach().cpu().numpy()}
