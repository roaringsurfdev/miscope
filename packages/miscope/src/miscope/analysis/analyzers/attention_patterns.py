"""Attention Patterns Analyzer.

Captures per-head attention patterns across all position pairs.
"""

from __future__ import annotations

from typing import Any

import einops
import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library import (
    compute_grid_size_from_dataset,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="attention_patterns",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
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

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract attention patterns and reshape to (n_heads, n_pos, n_pos, p, p)."""
        assert inputs.cache is not None  # type-narrowing for pyright
        p = compute_grid_size_from_dataset(inputs.probe)  # type: ignore

        # Shape: (p*p, n_heads, seq_to, seq_from)
        attn = inputs.cache["blocks.0.attn.hook_pattern"]

        # Reshape batch dim to (p, p) grid for each (head, to_pos, from_pos)
        patterns = einops.rearrange(
            attn,
            "(a b) heads to_pos from_pos -> heads to_pos from_pos a b",
            a=p,
            b=p,
        )

        return {"patterns": patterns.detach().cpu().numpy()}
