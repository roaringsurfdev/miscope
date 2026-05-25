"""Parameter Snapshot Analyzer.

Extracts and stores all trainable weight matrices per checkpoint for
parameter trajectory projection, velocity analysis, and downstream
geometric analyses (REQ_029).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library import extract_parameter_snapshot
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    pass


SPEC = AnalyzerSpec(
    name="parameter_snapshot",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    required_hooks=(),
)


@register_analyzer(SPEC)
class ParameterSnapshotAnalyzer:
    """Stores per-epoch weight matrix snapshots for trajectory analysis.

    Extracts weights from the model; the probe and context are unused.
    """

    name = "parameter_snapshot"
    description = "Stores weight matrix snapshots for trajectory analysis"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract all trainable weight matrices from the model."""
        assert inputs.model is not None  # type-narrowing for pyright
        return extract_parameter_snapshot(inputs.model)
