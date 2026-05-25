"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library import project_onto_fourier_basis
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

SPEC = AnalyzerSpec(
    name="dominant_frequencies",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=True, needs_cache=False),),
    required_hooks=(),
)


@register_analyzer(SPEC)
class DominantFrequenciesAnalyzer:
    """Computes Fourier coefficient norms for embedding weights.

    For each checkpoint, computes (fourier_basis @ W_E).norm(dim=-1),
    identifying which frequencies dominate the learned embedding representation.
    """

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute Fourier coefficient norms for embedding weights.

        Args:
            inputs: ResolvedInputs with model state.
            context: Must contain 'fourier_basis'.

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        assert inputs.model is not None  # type-narrowing for pyright
        fourier_basis = context["fourier_basis"]

        # Get embedding weights, excluding the equals token
        W_E = inputs.model.get_weight("embed.W_E")[:-1]

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
