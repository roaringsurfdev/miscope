"""Parameter Snapshot Analyzer.

Extracts and stores all trainable weight matrices per checkpoint for
parameter trajectory projection, velocity analysis, and downstream
geometric analyses (REQ_029).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import extract_parameter_snapshot

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


class ParameterSnapshotAnalyzer:
    """Stores per-epoch weight matrix snapshots for trajectory analysis.

    Extracts weights via the bundle. Probe and analysis_params are unused.
    """

    name = "parameter_snapshot"
    description = "Stores weight matrix snapshots for trajectory analysis"
    architecture_support = ["transformer", "mlp"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """Extract all trainable weight matrices from the bundle.

        Args:
            ctx: Analysis context with bundle (probe and analysis_params unused).

        Returns:
            Dict mapping weight matrix names to numpy arrays in
            their original shapes (e.g., W_E, W_Q, W_in, etc.)
        """
        return extract_parameter_snapshot(ctx.bundle)
