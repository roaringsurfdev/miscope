"""Parameter Snapshot Analyzer.

Extracts and stores all trainable weight matrices per checkpoint for
parameter trajectory projection, velocity analysis, and downstream
geometric analyses (REQ_029).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import extract_parameter_snapshot
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


SPEC = AnalyzerSpec(
    name="parameter_snapshot",
    category="primary",
    requires_model_weights=True,
    requires_activation_cache=False,  # reads weights only; no cache access
    required_hooks=(),
)


@register_analyzer(SPEC)
class ParameterSnapshotAnalyzer:
    """Stores per-epoch weight matrix snapshots for trajectory analysis.

    Extracts weights via the bundle. Probe and analysis_params are unused.
    """

    name = "parameter_snapshot"
    description = "Stores weight matrix snapshots for trajectory analysis"
    # Reads weights only — runs on any architecture.
    required_hooks: list[str] = []

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Extract all trainable weight matrices from the model.

        Args:
            ctx: Analysis context with model (probe and analysis_params unused).

        Returns:
            Dict mapping weight matrix names to numpy arrays in
            their original shapes (e.g., W_E, W_Q, W_in, etc.)
        """
        assert ctx.model is not None  # type-narrowing for pyright
        return extract_parameter_snapshot(ctx.model)
