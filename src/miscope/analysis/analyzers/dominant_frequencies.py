"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from miscope.analysis.library import project_onto_fourier_basis

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationContext


class DominantFrequenciesAnalyzer:
    """Computes Fourier coefficient norms for embedding weights.

    For each checkpoint, computes (fourier_basis @ W_E).norm(dim=-1),
    identifying which frequencies dominate the learned embedding representation.
    """

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"
    architecture_support = ["transformer"]

    def analyze(
        self,
        ctx: ActivationContext,
    ) -> dict[str, np.ndarray]:
        """
        Compute Fourier coefficient norms for embedding weights.

        Args:
            ctx: Analysis context with bundle and analysis_params.
                 analysis_params must contain 'fourier_basis'.

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        fourier_basis = ctx.analysis_params["fourier_basis"]

        # Get embedding weights, excluding the equals token
        W_E = ctx.bundle.weight("W_E")[:-1]

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
