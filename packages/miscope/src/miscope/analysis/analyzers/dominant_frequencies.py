"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import project_onto_fourier_basis
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


SPEC = AnalyzerSpec(
    name="dominant_frequencies",
    category="primary",
    requires_model_weights=True,
    requires_activation_cache=False,  # reads embed.W_E only; no cache access
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
    # Reads embed.W_E only — runs on transformer-class architectures that
    # publish a shared embedding matrix. Family-level registration (in
    # family.json) is the primary filter; if registered to a family that
    # doesn't publish embed.W_E, ``model.get_weight`` raises KeyError loudly.
    required_hooks: list[str] = []

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Compute Fourier coefficient norms for embedding weights.

        Args:
            ctx: Analysis context with model and analysis_params.
                 analysis_params must contain 'fourier_basis'.

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        assert ctx.model is not None  # type-narrowing for pyright
        fourier_basis = ctx.analysis_params["fourier_basis"]

        # Get embedding weights, excluding the equals token
        W_E = ctx.model.get_weight("embed.W_E")[:-1]

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
