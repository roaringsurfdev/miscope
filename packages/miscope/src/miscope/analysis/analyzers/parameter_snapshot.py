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
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    pass


# Raw weight matrices — extract-only, no transform. Each stays an .npz blob
# (REQ_110 keeps parameter_snapshot on .npz); the catalog indexes them as
# tensor descriptors keyed by (variant, epoch). Inner axes are the matrix shape.
_WEIGHTS = (
    ("W_E", "Token embedding matrix (d_vocab, d_model)."),
    ("W_pos", "Positional embedding matrix (n_ctx, d_model)."),
    ("W_Q", "Attention query weights (n_heads, d_model, d_head)."),
    ("W_K", "Attention key weights (n_heads, d_model, d_head)."),
    ("W_V", "Attention value weights (n_heads, d_model, d_head)."),
    ("W_O", "Attention output weights (n_heads, d_head, d_model)."),
    ("W_in", "MLP input weights (d_model, d_mlp)."),
    ("W_out", "MLP output weights (d_mlp, d_model)."),
    ("W_U", "Unembedding matrix (d_model, d_vocab)."),
)

SPEC = AnalyzerSpec(
    name="parameter_snapshot",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    required_hooks=(),
    outputs=tuple(
        F.tensor(name, "float32", ("variant", "epoch"), desc) for name, desc in _WEIGHTS
    ),
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
