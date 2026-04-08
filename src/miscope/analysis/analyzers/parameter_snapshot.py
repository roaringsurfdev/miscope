"""Parameter Snapshot Analyzer.

Extracts and stores all trainable weight matrices per checkpoint for
parameter trajectory projection, velocity analysis, and downstream
geometric analyses (REQ_029).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import extract_parameter_snapshot

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class ParameterSnapshotAnalyzer:
    """Stores per-epoch weight matrix snapshots for trajectory analysis.

    Extracts weights via the bundle. Probe and context are accepted
    for protocol conformance but ignored.
    """

    name = "parameter_snapshot"
    description = "Stores weight matrix snapshots for trajectory analysis"
    architecture_support = ["transformer", "mlp"]

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,  # noqa: ARG002
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Extract all trainable weight matrices from the bundle.

        Args:
            bundle: Activation bundle with checkpoint weights.
            probe: Unused (protocol conformance).
            context: Unused (protocol conformance).

        Returns:
            Dict mapping weight matrix names to numpy arrays in
            their original shapes (e.g., W_E, W_Q, W_in, etc.)
        """
        return extract_parameter_snapshot(bundle)
