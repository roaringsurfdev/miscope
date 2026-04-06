"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from miscope.analysis.library import project_onto_fourier_basis

if TYPE_CHECKING:
    from miscope.analysis.protocols import ActivationBundle


class DominantFrequenciesAnalyzer:
    """Computes Fourier coefficient norms for embedding weights.

    For each checkpoint, computes (fourier_basis @ W_E).norm(dim=-1),
    identifying which frequencies dominate the learned embedding representation.
    """

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"

    def analyze(
        self,
        bundle: ActivationBundle,
        probe: torch.Tensor,  # noqa: ARG002
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute Fourier coefficient norms for embedding weights.

        Args:
            bundle: Activation bundle with checkpoint weights.
            probe: Unused (protocol conformance).
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        fourier_basis = context["fourier_basis"]

        # Get embedding weights, excluding the equals token
        W_E = bundle.weight("W_E")[:-1]

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
